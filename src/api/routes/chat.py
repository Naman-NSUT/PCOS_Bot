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
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


@router.post("/report", response_model=ReportResponse,
             summary="Analyse photographs of a lab report inside a consultation")
def analyse_report_endpoint(
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

    images: list[bytes] = []
    for f in files[:MAX_REPORT_IMAGES]:
        if f.content_type and f.content_type.lower() not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type {f.content_type}. Send a JPEG, PNG or WebP photo.",
            )
        blob = f.file.read(MAX_UPLOAD_BYTES + 1)
        if len(blob) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Each image must be under 8 MB.")
        if blob:
            images.append(blob)

    if not images:
        raise HTTPException(status_code=400, detail="No readable image was uploaded.")

    result = record_report_turn(session_id, images, user_id=user_id)
    return ReportResponse(**result, user_token=out_token)


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
