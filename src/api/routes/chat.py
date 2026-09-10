"""
src/api/routes/chat.py — POST /chat, POST /chat/new, DELETE /chat/memory

Session-aware conversational endpoint for the Maya PCOS consultation flow,
with durable cross-session user memory.

Identity: the server mints a signed `user_token`. Clients store it and send it
back to be recognised on a later visit. A raw client-supplied id is never
trusted — that would turn an unguessable ephemeral token into a forgeable
permanent key to someone else's health record.

Concurrency: the handlers are deliberately plain `def`, not `async def`. The
pipeline underneath (LLM calls, CRAG retrieval, DB writes) is fully synchronous,
and an `async def` handler runs it directly on the event loop, serialising every
other request behind it — measured at 4.0s for four concurrent calls versus 1.0s
via the threadpool. Starlette runs `def` handlers in a worker thread, which is
what this workload needs.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.agents.conversation_agent import get_greeting, process_turn, record_report_turn
from config.settings import MAX_REPORT_IMAGES
from src.core.identity import mint_user_token, sign_user_id, verify_user_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Request / Response Models ────────────────────────────────────────────

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class NewChatRequest(BaseModel):
    # Signed token from a previous visit. Omit it for a first-time user.
    user_token: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_token: Optional[str] = None
    history: List[Message] = []              # kept for backward compat (unused)
    report_context: Optional[str] = None     # kept for backward compat (unused)


class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str
    phase: int
    # Always returned so a first-time client can learn and store its identity.
    user_token: str
    is_returning_user: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────

def _resolve_identity(token: Optional[str]) -> tuple[str, str, bool]:
    """
    Turn a client token into (user_id, token_to_return, recognised).

    An invalid or absent token yields a brand-new identity rather than an error:
    a tampered token must degrade to a fresh consultation, never leak a record.
    """
    user_id = verify_user_token(token)
    if user_id:
        return user_id, sign_user_id(user_id), True
    user_id, new_token = mint_user_token()
    return user_id, new_token, False


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/new", response_model=ChatResponse, summary="Start a new consultation")
def new_chat(request: NewChatRequest | None = None) -> ChatResponse:
    """
    Create a new consultation and return Maya's opening line.

    The bot speaks first. For a recognised user the whole long-term memory is
    loaded here and the greeting becomes a welcome-back rather than a request
    for their name.
    """
    token = request.user_token if request else None
    user_id, out_token, _ = _resolve_identity(token)

    result = get_greeting(user_id=user_id)
    return ChatResponse(
        **result,
        user_token=out_token,
        is_returning_user="Welcome back" in result.get("answer", ""),
    )


@router.post("", response_model=ChatResponse, summary="Send a message to Maya")
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Send a user message within an existing consultation.
    If session_id is missing, a new consultation is started and the greeting
    returned — memory-hydrated when the token identifies a known user.
    """
    user_id, out_token, _ = _resolve_identity(request.user_token)

    if not request.session_id:
        result = get_greeting(user_id=user_id)
        return ChatResponse(
            **result,
            user_token=out_token,
            is_returning_user="Welcome back" in result.get("answer", ""),
        )

    result = process_turn(request.session_id, request.query, user_id=user_id)
    return ChatResponse(**result, user_token=out_token)


class ReportResponse(BaseModel):
    answer: str
    sources: list
    session_id: str
    phase: int
    user_token: str
    ok: bool
    parsed_values: dict = {}
    diagnostic_flags: dict = {}
    concordance: dict = {}
    unreadable: list = []
    pages_dropped: int = 0


MAX_UPLOAD_BYTES = 8 * 1024 * 1024          # per file
MAX_TOTAL_UPLOAD_BYTES = 20 * 1024 * 1024   # per request, across all files
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}

# Vision calls are the most expensive thing this service does (~25k input tokens
# and several seconds each), and /chat/report needs no credential — an
# unauthenticated caller could spend money in a loop. This is a small in-process
# limiter: enough to stop casual abuse, not a substitute for a real gateway
# limit in front of the app.
REPORT_RATE_LIMIT = 5           # uploads...
REPORT_RATE_WINDOW = 15 * 60    # ...per user per 15 minutes
_report_calls: dict[str, list[float]] = {}


def _rate_key(request: Request, user_token: Optional[str]) -> str:
    """
    What to rate-limit on.

    NOT the resolved user_id: _resolve_identity mints a brand-new id for every
    request without a valid token, so an anonymous flood would get a fresh
    bucket each time and never be limited — precisely the case this exists for.
    A recognised token limits that user; everyone else is limited by client
    address.
    """
    known = verify_user_token(user_token)
    if known:
        return "u:" + known
    client = getattr(request, "client", None)
    return "ip:" + (getattr(client, "host", None) or "unknown")


def _rate_limit_ok(user_id: str, now: float) -> tuple[bool, int]:
    """Returns (allowed, seconds_until_retry)."""
    hits = [t for t in _report_calls.get(user_id, []) if now - t < REPORT_RATE_WINDOW]
    if len(hits) >= REPORT_RATE_LIMIT:
        return False, int(REPORT_RATE_WINDOW - (now - hits[0])) + 1
    hits.append(now)
    _report_calls[user_id] = hits
    # Opportunistic cleanup so the dict cannot grow without bound.
    if len(_report_calls) > 5000:
        for uid in [u for u, ts in _report_calls.items()
                    if not any(now - t < REPORT_RATE_WINDOW for t in ts)]:
            _report_calls.pop(uid, None)
    return True, 0


@router.post("/report", response_model=ReportResponse,
             summary="Analyse photographs of a lab report inside a consultation")
def analyse_report_endpoint(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more report photos"),
    session_id: Optional[str] = Form(None),
    user_token: Optional[str] = Form(None),
) -> ReportResponse:
    """
    Send report photos mid-consultation.

    The verdict draws on BOTH the bloodwork and what the person already said —
    that combination is the point, because no blood test establishes irregular
    ovulation and the conversation cannot measure testosterone.

    Plain `def`, not `async def`: the pipeline is synchronous and would otherwise
    block the event loop for every other request.
    """
    user_id, out_token, _ = _resolve_identity(user_token)

    allowed, retry_after = _rate_limit_ok(_rate_key(request, user_token), time.monotonic())
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="That's a lot of reports in a short time. Please try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )

    dropped = max(0, len(files) - MAX_REPORT_IMAGES)
    images: list[bytes] = []
    total = 0
    for f in files[:MAX_REPORT_IMAGES]:
        # A MISSING Content-Type used to skip the check entirely, because
        # `if f.content_type and ...` is falsy when the header is absent.
        # It is also attacker-controlled, so the real check is the image
        # validation in transcribe_report; this is only a cheap early reject.
        ctype = (f.content_type or "").lower()
        if ctype not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type {f.content_type or 'unknown'}. "
                       "Send a JPEG, PNG or WebP photo.",
            )
        blob = f.file.read(MAX_UPLOAD_BYTES + 1)
        if len(blob) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Each image must be under 8 MB.")
        total += len(blob)
        if total > MAX_TOTAL_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Those images are too large in total. Please send fewer pages.",
            )
        if blob:
            images.append(blob)

    if not images:
        raise HTTPException(status_code=400, detail="No readable image was uploaded.")

    result = record_report_turn(session_id, images, user_id=user_id)
    # Silent truncation used to leave the UI claiming every page was read.
    if dropped:
        result = dict(result)
        result["answer"] = (
            f"I could only read the first {MAX_REPORT_IMAGES} pages, so {dropped} "
            f"more weren't included.\n\n" + result["answer"]
        )
    return ReportResponse(**result, user_token=out_token, pages_dropped=dropped)


@router.delete("/memory", summary="Erase everything remembered about a user")
def forget_me(request: NewChatRequest) -> dict:
    """
    Delete the user and every fact and consultation recorded for them.

    Owning the store is what makes this a three-line operation rather than a
    support ticket to a vendor.
    """
    user_id = verify_user_token(request.user_token)
    if not user_id:
        raise HTTPException(status_code=400, detail="A valid user_token is required.")

    from src.memory.store import forget_user
    from src.core.session import purge_user_sessions

    # Both halves matter: the database rows AND any live in-memory session, which
    # still holds the transcript and would keep serving it after "deletion".
    existed = forget_user(user_id)
    dropped = purge_user_sessions(user_id)
    logger.info(
        "[chat] erasure for %s — rows=%s live_sessions=%d",
        user_id[:8], existed, dropped,
    )
    return {"deleted": existed, "sessions_dropped": dropped}
