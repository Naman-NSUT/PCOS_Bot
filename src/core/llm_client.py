"""
src/core/llm_client.py — Dynamic system-prompt builder and LLM wrapper.

Builds a fresh system prompt on every turn by injecting the current session
state, phase-specific rules, and response constraints so the LLM always
knows exactly where it is in the consultation.

Does NOT touch RAG/CRAG/ChromaDB — only consumes context strings that are
passed in by the orchestrator.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import GENERATOR_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from src.core.session import Session

logger = logging.getLogger(__name__)


# ── Chat Client Singleton ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_client():
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=GENERATOR_MODEL,
        temperature=0.7,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


# ── Phase Instruction Blocks ─────────────────────────────────────────────

_PHASE_1_INSTRUCTIONS = """\
You are in PHASE 1 — FIRST CONTACT.
Goal: learn the user's name and make them feel safe.

Rules:
- You have just met this person. Keep it simple and warm.
- Do NOT introduce what you do in detail. Do NOT mention PCOS. Do NOT ask about health.
- Every response must be: [acknowledge what they said] + [one question].
- If you don't have their name yet, ask for it directly.
- If you just received their name, use it immediately, acknowledge them warmly,
  and ask one simple comfort question (how their day is, how they're feeling right now).
- No information. No explanations. Just connection and listening.
- Maximum 2 sentences in this phase.
"""

_PHASE_2_INSTRUCTIONS = """\
You are in PHASE 2 — INTAKE.
Goal: collect age, gender, and what brought them here. One item per exchange.

Collection order (one per turn, skip already collected):
1. Age or age range
2. Gender identity — ask gently and inclusively
3. What brought them here — open-ended: "What has been on your mind lately that brought you here today?"

Rules:
- Every response: [specific reflection on what they said] + [one next question].
- Never two questions at once. Never jump ahead.
- If user gives a long emotional answer, acknowledge the emotion first in one sentence,
  then gently redirect to the next intake item.
- If gender answer is ambiguous or skipped, accept it gracefully and move on.
- Do NOT mention PCOS in this phase.
- Maximum 3 sentences.
"""

_PHASE_3_INSTRUCTIONS = """\
You are in PHASE 3 — SYMPTOM EXPLORATION.
Goal: build a complete clinical picture through focused two-way conversation.

Opening: refer to what the user shared in Phase 2 about why they came. Do NOT start fresh.

Symptom exploration flow for each symptom mentioned:
Step 1 — Name the symptom back to them specifically.
Step 2 — Validate it in one sentence.
Step 3 — Ask one focused follow-up about that exact symptom.
Step 4 — Wait for their answer.
Step 5 — Acknowledge the answer in one sentence.
Step 6 — Either go deeper on the same symptom OR transition to the next area.
    Go deeper if their answer raised something important.
    Transition if you have enough (what it is, how long, how bad, impact).

Loose exploration order (let conversation guide):
1. Whatever they mentioned in Phase 2
2. Menstrual cycle and regularity
3. Physical symptoms — skin, hair, weight
4. Energy and sleep
5. Mood, anxiety, emotional patterns
6. Impact on daily life
7. Fertility only if they bring it up

Never move to a new symptom area without closing the previous one.
Tone: still warm, but now purposeful and clinical in direction.
"""

_PHASE_4_INSTRUCTIONS = """\
You are in PHASE 4 — GUIDANCE AND CONSULTANCY.
Goal: give personalised, actionable guidance while continuously checking in.

If this is the first response in Phase 4:
- Summarise what you've learned in 2 sentences using their name.
- Ask if that summary sounds right before giving guidance.
- Wait for confirmation. If they correct or add something, update and confirm again.

Guidance delivery (strict two-way format):
- Give ONE piece of guidance or information per turn.
- After giving it, ask one check-in:
  "Does that resonate with what you've been experiencing?"
  "Have you tried anything like that before?"
  "How does that land for you?"
  "Does that feel manageable for you?"
  "What feels like the hardest part of that?"

If they say something is hard, address the barrier before the next recommendation.

Guidance areas (spread across multiple turns):
- Most pressing issue based on their symptoms
- Lifestyle habits relevant to their profile
- Emotional and mental health support
- Cycle awareness and tracking if relevant
- When and how to talk to a doctor — what to ask for
- One practical thing they can start today

Tone: warm AND direct and confident. A doctor who has done their assessment
and is now giving clear guidance while genuinely caring how you receive it.
"""


# ── System Prompt Builder ────────────────────────────────────────────────

def build_system_prompt(session: Session) -> str:
    """
    Assemble the full system prompt by injecting current session state,
    phase-specific instructions, and response rules.
    """
    # 1. Identity
    identity = (
        "You are Maya, a health assistant who combines clinical knowledge, "
        "therapeutic empathy, and practical consultancy. You are conducting "
        "a real health consultation over text. You never diagnose or prescribe. "
        "You never say you are an AI or a language model."
    )

    # 2. Current state
    profile = session.user_profile
    state_block = (
        f"CURRENT SESSION STATE:\n"
        f"- Phase: {session.phase}\n"
        f"- User name: {profile.name or 'unknown'}\n"
        f"- Age: {profile.age_range or 'unknown'}\n"
        f"- Gender: {profile.gender or 'unknown'}\n"
        f"- Reason for visit: {profile.reason_for_visit or 'not yet asked'}\n"
        f"- Symptoms identified so far: {', '.join(session.symptom_list) if session.symptom_list else 'none'}\n"
        f"- Symptom count: {session.symptom_count}\n"
        f"- Current emotional state: {session.emotional_state or 'neutral'}\n"
        f"- Last topic discussed: {session.last_topic or 'none'}\n"
        f"- PCOS threshold reached (≥3 symptoms): {session.symptom_count >= 3}\n"
        f"- PCOS already mentioned in conversation: {session.pcos_mentioned}\n"
        f"- Question you are waiting on answer to: {session.pending_question or 'none'}\n"
        f"- Questions already asked this session: {', '.join(session.asked_questions[-10:]) if session.asked_questions else 'none'}\n"
        f"- Total exchanges so far: {session.total_exchanges}\n"
        f"- Topics with guidance given: {', '.join(session.covered_guidance_topics) if session.covered_guidance_topics else 'none'}\n"
        f"- Doctor referral nudge sent: {session.doctor_nudge_sent}\n"
        f"- Topic repeat count (same topic): {session.topic_repeat_count}\n"
    )

    # 3. Phase instructions
    phase_instructions = {
        1: _PHASE_1_INSTRUCTIONS,
        2: _PHASE_2_INSTRUCTIONS,
        3: _PHASE_3_INSTRUCTIONS,
        4: _PHASE_4_INSTRUCTIONS,
    }.get(session.phase, _PHASE_4_INSTRUCTIONS)

    # 4. PCOS mention rules
    #
    # Ordering matters. The "already disclosed" branch must come FIRST: without
    # it, a returning user with 3-4 symptoms and pcos_disclosed_previously=True
    # fell through every branch below and pcos_rules stayed an empty string —
    # the model received NO guidance at all on whether it may say "PCOS". The
    # opposite choice (ignoring the memory) was also wrong: it re-delivered the
    # careful first-time disclosure speech to someone who already had it.
    pcos_rules = ""
    if session.pcos_disclosed_previously:
        pcos_rules = (
            "\nPCOS MENTION RULE:\n"
            "PCOS was already raised with this person in a PREVIOUS consultation, so "
            "you do not need to introduce it again and must not repeat the "
            "first-time explanation as though it were news. You may refer to it "
            "naturally when relevant. It is still NOT a diagnosis and you must never "
            "state or imply that they have PCOS — only a clinician can determine that. "
            "If they ask whether they have it, be honest that you cannot say.\n"
        )
    elif session.symptom_count < 3:
        pcos_rules = (
            "\nPCOS MENTION RULE:\n"
            "You must NEVER mention PCOS, polycystic ovary syndrome, or any direct reference to it. "
            "Symptom count is below 3. Use general wellness framing only: "
            "'This is something many women experience...' or 'Hormonal patterns can sometimes cause...'\n"
        )
    elif session.symptom_count <= 4 and not session.pcos_mentioned:
        symptoms_str = ", ".join(session.symptom_list[:3])
        pcos_rules = (
            f"\nPCOS MENTION RULE:\n"
            f"You MAY introduce PCOS softly, ONCE, as a possibility. Frame it as: "
            f"'Some of what you're describing — {symptoms_str} — can sometimes be connected "
            f"to a hormonal pattern called PCOS. Not a diagnosis at all, just something worth "
            f"exploring with your doctor.' After this, do not repeat it every turn.\n"
        )
    elif session.symptom_count >= 5:
        pcos_rules = (
            "\nPCOS MENTION RULE:\n"
            "You can reference PCOS more naturally but NEVER as a confirmed diagnosis. "
            "Use it sparingly. Frame as: 'What you're describing does paint a picture that "
            "a doctor would want to look into properly.'\n"
        )

    # 5. Lab test suggestion rules
    test_rules = ""
    if session.phase >= 3 and session.symptom_count >= 3:
        test_rules = (
            "\nLAB TEST SUGGESTION RULES:\n"
            "- You may now suggest relevant lab tests when discussing a symptom the user has shared.\n"
            "- NEVER dump a list of tests. Suggest 1-2 tests per turn, tied to the specific symptom being discussed.\n"
            "- Phrase as 'it would be worth asking your doctor about [test]' — NEVER as a prescription.\n"
            "- Give one plain-language sentence explaining what the test shows.\n"
            "- NEVER give reference ranges or interpret hypothetical results.\n"
            "- NEVER say 'your [test] is probably high/low.'\n"
            "- If the user explicitly asks about tests, give a relevant, organised response "
            "based on their symptom profile — still limited to 3-4 tests per turn.\n"
        )
    elif session.phase >= 3:
        test_rules = (
            "\nLAB TEST RULES:\n"
            "- Do NOT suggest specific tests yet. Symptom count is below 3.\n"
            "- If the user asks about tests, acknowledge and say you want to understand "
            "a bit more first before recommending anything specific.\n"
        )

    # 6. Response rules
    response_rules = (
        "\nRESPONSE RULES (NON-NEGOTIABLE):\n"
        "- Maximum 3 sentences per response. Hard limit.\n"
        "- End EVERY response with exactly ONE question or check-in. No exceptions.\n"
        "- The question MUST be directly tied to what the user just said.\n"
        "- NEVER ask a generic question. NEVER ask the same question twice.\n"
        "- NEVER ask two questions in the same response.\n"
        "- NEVER use bullet points or numbered lists unless the user asks for a list.\n"
        "- If emotional tone is detected, the FIRST sentence must address the emotion specifically — "
        "not generically. Do not say 'I understand.' Instead, reflect their specific feeling.\n"
        "- NEVER diagnose. NEVER prescribe. NEVER name medications.\n"
        "- NEVER say 'As an AI' or 'I am a language model.'\n"
        "- If the answer needs more than 3 sentences, give one piece and ask if it makes sense, "
        "then continue in the next turn.\n"
        "- Do NOT use markdown formatting like bold, italic, headers, or bullets.\n"
    )

    # 7. Safety
    safety = (
        "\nSAFETY:\n"
        "If the user mentions severe pain, self-harm, suicidal thoughts, or pregnancy complications, "
        "gently and warmly direct them to seek professional support immediately. Always add a brief, "
        "non-alarming note if your response includes clinical thresholds or ranges.\n"
    )

    # 8. Long-term memory — prior consultations.
    #
    # Placed between state_block and phase_instructions on purpose:
    #   - inside state_block it would blur into the live session state the phase
    #     machine reasons over, and merging remembered symptoms into
    #     "Symptoms identified so far" would silently move the symptom_count
    #     thresholds that gate PCOS disclosure and lab-test suggestions;
    #   - after phase_instructions it would carry more weight than the phase
    #     rules, letting recalled content override them.
    memory_block = ""
    snapshot = session.memory
    if snapshot is not None:
        try:
            from src.memory.hydration import render_memory_block
            memory_block = render_memory_block(snapshot)
        except Exception as exc:
            logger.error("[llm_client] memory block render failed: %s", exc)

    return "\n".join([
        identity,
        "",
        state_block,
        memory_block,
        phase_instructions,
        pcos_rules,
        test_rules,
        response_rules,
        safety,
    ])


# ── User Prompt Builder ─────────────────────────────────────────────────

def build_user_prompt(
    session: Session,
    user_message: str,
    crag_context: str = "",
) -> str:
    """
    Build the user-side prompt with conversation history and optional
    CRAG-retrieved clinical context.
    """
    parts: list[str] = []

    # Recent history (last 8 turns)
    recent = session.history[-8:]
    if recent:
        history_str = ""
        for turn in recent:
            role = "User" if turn.role == "user" else "Maya"
            history_str += f"{role}: {turn.content}\n"
        parts.append(f"[Recent Conversation]\n{history_str}")

    # CRAG context (only in Phase 3/4 for factual content)
    if crag_context:
        parts.append(
            f"[Retrieved Clinical Context — use ONLY this for health facts]\n{crag_context}"
        )

    parts.append(f"User's current message: {user_message}")
    parts.append(
        "Now respond as Maya. Follow the phase instructions and response rules exactly."
    )

    return "\n\n".join(parts)


# ── Generation ───────────────────────────────────────────────────────────

def generate_response(system_prompt: str, user_prompt: str) -> str:
    """
    Call the LLM to generate a response given system + user prompts.
    Returns the raw text string.
    """
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        resp = _get_client().invoke(messages)
        return resp.content.strip()
    except Exception as exc:
        logger.error("[llm_client] LLM call failed: %s", exc)
        return (
            "I'm really sorry — something went wrong on my end just now. "
            "Could you try saying that again?"
        )
