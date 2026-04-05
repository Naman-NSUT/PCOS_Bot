"""
src/api/routes/chat.py — POST /chat
General-purpose conversational RAG endpoint.
The user asks any PCOS-related question; the system retrieves relevant
clinical guidance via CRAG and generates a Gemini-powered response.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from fastapi import APIRouter
from google import genai
from google.genai import types
from functools import lru_cache
from pydantic import BaseModel

from config.settings import GEMINI_API_KEY, GENERATOR_MODEL
from src.graphs.crag_graph import run_crag

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []
    report_context: Optional[str] = None   # optional parsed lab data injected by UI


class ChatResponse(BaseModel):
    answer: str
    sources: list


_SYSTEM = """\
You are a compassionate, knowledgeable PCOS health assistant backed by a \
clinical knowledge base of guidelines, research papers, and patient education materials.

Your job:
- Answer PCOS-related questions clearly and empathetically.
- Ground every factual claim in the retrieved clinical context when available.
- Use plain language; explain medical terms when you use them.
- NEVER diagnose, prescribe, or replace a doctor's advice.
- If the question is outside PCOS health scope, politely redirect.
- Keep answers concise but complete — use bullet points or numbered steps where helpful.
"""


@lru_cache(maxsize=1)
def _get_client():
    return genai.Client(api_key=GEMINI_API_KEY)


@router.post("", response_model=ChatResponse, summary="Chat with the PCOS RAG assistant")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    # 1. Retrieve relevant context via CRAG
    crag_result = run_crag(request.query)
    context = crag_result.get("context", "")
    sources = crag_result.get("sources", [])

    # 2. Build conversation history string
    history_str = ""
    for msg in request.history[-6:]:   # last 6 messages for context window
        role = "User" if msg.role == "user" else "Assistant"
        history_str += f"{role}: {msg.content}\n"

    # 3. Build user prompt
    user_prompt = ""
    if request.report_context:
        user_prompt += f"[Lab Report Data Provided by User]\n{request.report_context}\n\n"
    if context:
        user_prompt += f"[Retrieved Clinical Context]\n{context}\n\n"
    if history_str:
        user_prompt += f"[Conversation History]\n{history_str}\n"
    user_prompt += f"User's Question: {request.query}"

    # 4. Generate answer
    try:
        resp = _get_client().models.generate_content(
            model=GENERATOR_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=_SYSTEM),
        )
        answer = resp.text.strip()
    except Exception as exc:
        logger.error("[chat] Gemini failed: %s", exc)
        answer = "Sorry, I couldn't generate a response right now. Please try again."

    return ChatResponse(answer=answer, sources=sources)
