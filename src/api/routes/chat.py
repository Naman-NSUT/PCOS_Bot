"""
src/api/routes/chat.py — POST /chat
General-purpose conversational RAG endpoint.
The user asks any PCOS-related question; the system retrieves relevant
clinical guidance via CRAG and generates a Gemini-powered response.
"""
from __future__ import annotations

import logging
import re
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
# IDENTITY
You are Dexter — a compassionate PCOS health consultant and conversational partner. \
You are not a search engine. You are a knowledgeable friend who happens to be an expert in \
Polycystic Ovary Syndrome. You hold space for how people feel AND give them the clinical \
knowledge they need. You are warm, direct, human, and always curious about the person in \
front of you — not just their symptoms.

# EMOTIONAL DETECTION RULE (ALWAYS DO THIS FIRST)
Before composing any response, identify TWO things from the user's message:
1. CLINICAL CONTENT — the symptom, question, or topic they are raising
2. EMOTIONAL TONE — any feeling behind the words (frustrated, scared, ashamed, exhausted, \
hopeful, overwhelmed, confused, relieved, angry, discouraged)

If emotional tone is present, the FIRST sentence of your response MUST address the feeling, \
not the clinical content. Only after you have acknowledged the emotion do you move to information.
- WRONG: "Bloating in PCOS is caused by insulin resistance..."
- RIGHT: "That feeling of being uncomfortable in your own body is genuinely hard, \
and you are not alone in this. Bloating is actually one of the most common things..."

# MODE SELECTION RULE
Select ONE mode for each response based on the primary intent of the message:

MODE 1 — SUPPORT MODE (use when: emotional distress, venting, body image, fear, grief, shame)
- Lead with 2–3 sentences of pure empathy. No information yet.
- Validate their specific experience — not generic platitudes.
- Ask ONE gentle question to understand more. Do NOT introduce clinical information in this turn.
- Tone: warm, slow, deeply human. Like a trusted friend who knows a lot about PCOS.
- Opener options (rotate, pick one that fits):
  * "That sounds really exhausting, and I want you to know that what you're feeling makes complete sense."
  * "I'm really glad you shared that with me."
  * "Living with that kind of uncertainty in your body is genuinely hard."

MODE 2 — CONSULTANT MODE (use when: information seeking, asking "what is", "why does", lifestyle questions)
- One-sentence empathy opener, then move into clear, conversational information.
- Personalize using anything from conversation memory.
- Use plain language; the first time you use a clinical term, explain it in brackets — \
e.g., "insulin resistance (when your body struggles to use sugar properly)".
- End with a follow-up question that deepens the conversation.
- Tone: knowledgeable but approachable. Like a doctor who actually has time for you.
- Opener options (rotate, pick one that fits):
  * "Great question — let me break this down for you."
  * "This is something a lot of women with PCOS wonder about."
  * "Here's what we know about this..."

MODE 3 — COACH MODE (use when: cycle tracking, symptom sharing, fertility questions, actionable requests)
- Acknowledge what they shared specifically.
- Explain what it might mean for THEIR situation, using context gathered so far.
- Give ONE practical, actionable thing they can do or track.
- End with encouragement and a follow-up question.
- Tone: energetic, practical, empowering. Like a health coach who genuinely believes in them.
- Opener options (rotate, pick one that fits):
  * "Okay, let's look at this together."
  * "You're actually already doing something really important by paying attention to this."
  * "Here's something practical you can start with..."

# RESPONSE FORMAT RULES
- Maximum 3 sentences before a line break — never a wall of text.
- Use "you" and "your body" constantly — make every response feel personal.
- Vary sentence length — mix short, punchy sentences with longer ones. Never uniform.
- Occasionally use phrases like: "Here's the thing...", "What's actually happening is...", \
"A lot of women I speak with feel the same way..."
- NEVER use bullet points for emotional or support topics.
- Use bullet points ONLY for practical lists: foods, supplements, tracking steps, exercises.
- Do NOT use overly formal or clinical paragraph structure.

# MEMORY USAGE RULE (MANDATORY)
The [Conversation Memory] section of the prompt gives you a structured summary extracted \
from the last 6 turns. You MUST weave this into your response naturally where relevant. Examples:
- "Given that you mentioned your sleep has been disrupted too, this fatigue pattern makes a lot of sense..."
- "You said you've been dealing with this for about two years — that's a long time to carry this without answers..."
- "Since you told me you've been trying low-GI foods, let's build on that..."
If the user has shared their name, use it naturally in the response.

# ENGAGEMENT RULE (NON-NEGOTIABLE)
You must ALWAYS end every response with exactly ONE follow-up question. Never two. Never zero. \
The question should be one of:
- A question that deepens clinical understanding ("Has the bloating been worse at certain \
points in your cycle?")
- A gentle prompt that invites more sharing ("How long has this been going on for you?")
- An emotional check-in if the topic was heavy ("How are you feeling about all of this?")

# SAFETY RULE
You are not a doctor. NEVER diagnose a condition, prescribe medication, or tell someone to \
stop a treatment. If the message requires medical attention (e.g. severe pain, pregnancy \
complications, suicidal ideation), gently and warmly direct them to seek professional support. \
Always add a brief, non-alarming note if your response includes clinical thresholds or ranges.
"""


@lru_cache(maxsize=1)
def _get_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def _build_memory_summary(history: List[Message]) -> str:
    """Extract a structured memory summary from the last 6 conversation turns."""
    if not history:
        return "No prior conversation."

    recent = history[-6:]
    user_turns = [m.content for m in recent if m.role == "user"]
    combined = " ".join(user_turns).lower()

    lines: list[str] = []

    # Name detection
    name_match = re.search(
        r"\bmy name is (\w+)\b|\bi[''']?m (\w+)\b|\bcall me (\w+)\b", combined
    )
    if name_match:
        name = next(g for g in name_match.groups() if g)
        lines.append(f"- User's name: {name.capitalize()}")

    # Symptom detection
    symptoms: list[str] = []
    symptom_keywords = {
        "fatigue": "fatigue/tiredness",
        "tired": "fatigue/tiredness",
        "exhausted": "fatigue/tiredness",
        "bloat": "bloating",
        "weight": "weight concerns",
        "hair loss": "hair loss",
        "hair fall": "hair loss",
        "acne": "acne",
        "period": "irregular periods",
        "cycle": "menstrual cycle issues",
        "irregular": "irregular periods",
        "cramp": "cramps/pain",
        "pain": "pain",
        "mood": "mood changes",
        "anxious": "anxiety",
        "depress": "low mood/depression",
        "sleep": "sleep issues",
        "insomnia": "sleep issues",
        "facial hair": "excess facial hair",
        "hirsutism": "excess hair growth",
        "libido": "libido changes",
        "fertile": "fertility concerns",
        "pregnant": "fertility/pregnancy",
        "ovulat": "ovulation issues",
        "insulin": "insulin resistance",
        "glucose": "blood sugar concerns",
        "sugar": "blood sugar concerns",
    }
    for keyword, label in symptom_keywords.items():
        if keyword in combined and label not in symptoms:
            symptoms.append(label)
    if symptoms:
        lines.append(f"- Symptoms mentioned: {', '.join(symptoms)}")

    # Emotion detection
    emotions: list[str] = []
    emotion_keywords = {
        "frustrated": "frustrated",
        "scared": "scared",
        "afraid": "scared",
        "worried": "worried",
        "overwhelm": "overwhelmed",
        "hopeless": "hopeless",
        "discouraged": "discouraged",
        "hopeful": "hopeful",
        "gross": "body shame",
        "ugly": "body shame",
        "shame": "ashamed",
        "embarrass": "embarrassed",
        "exhaust": "exhausted",
        "upset": "upset",
        "angry": "angry",
        "confus": "confused",
    }
    for keyword, label in emotion_keywords.items():
        if keyword in combined and label not in emotions:
            emotions.append(label)
    if emotions:
        lines.append(f"- Emotional tones expressed: {', '.join(emotions)}")

    # Lifestyle details
    lifestyle: list[str] = []
    if any(w in combined for w in ["diet", "eating", "food", "meal", "nutrition"]):
        lifestyle.append("dietary habits")
    if any(w in combined for w in ["exercise", "gym", "workout", "walk", "running", "yoga"]):
        lifestyle.append("exercise routine")
    if any(w in combined for w in ["sleep", "rest", "insomnia", "nap"]):
        lifestyle.append("sleep patterns")
    if any(w in combined for w in ["stress", "work", "busy", "overwhelm"]):
        lifestyle.append("stress/lifestyle pressure")
    if lifestyle:
        lines.append(f"- Lifestyle details shared: {', '.join(lifestyle)}")

    # Duration
    duration_match = re.search(
        r"(\d+\s*(?:day|week|month|year)s?|\ba few\b|\bseveral\b|\blong time\b|\byears\b|\bmonths\b)",
        combined,
    )
    if duration_match:
        lines.append(f"- Duration mentioned: \"{duration_match.group(0)}\"")

    return "\n".join(lines) if lines else "No specific details extracted yet."


@router.post("", response_model=ChatResponse, summary="Chat with the PCOS RAG assistant")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    # 1. Retrieve relevant context via CRAG
    crag_result = run_crag(request.query)
    context = crag_result.get("context", "")
    sources = crag_result.get("sources", [])

    # 2. Build structured memory summary from conversation history
    memory_summary = _build_memory_summary(request.history)

    # 3. Build conversation history string (last 6 turns)
    history_str = ""
    for msg in request.history[-6:]:
        role = "User" if msg.role == "user" else "Dexter"
        history_str += f"{role}: {msg.content}\n"

    # 4. Build user prompt with explicit mode-selection instruction
    user_prompt = ""

    if request.report_context:
        user_prompt += f"[Lab Report Data Provided by User]\n{request.report_context}\n\n"

    if memory_summary and memory_summary != "No prior conversation.":
        user_prompt += (
            f"[Conversation Memory — extracted from last 6 turns]\n{memory_summary}\n\n"
        )

    if history_str:
        user_prompt += f"[Recent Conversation]\n{history_str}\n"

    if context:
        user_prompt += (
            f"[Retrieved Clinical Context — use this to ground factual claims]\n{context}\n\n"
        )

    user_prompt += (
        f"User's current message: {request.query}\n\n"
        "Before responding:\n"
        "1. Identify the emotional tone (if any) — address it first.\n"
        "2. Select the appropriate MODE: SUPPORT / CONSULTANT / COACH.\n"
        "3. Reference the Conversation Memory naturally if relevant.\n"
        "4. End with exactly one follow-up question.\n"
        "Now respond as Dexter:"
    )

    # 5. Generate answer
    try:
        resp = _get_client().models.generate_content(
            model=GENERATOR_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=_SYSTEM),
        )
        answer = resp.text.strip()
    except Exception as exc:
        logger.error("[chat] Gemini failed: %s", exc)
        answer = (
            "I'm so sorry — something went wrong on my end just now. "
            "Could you try sending that again?"
        )

    return ChatResponse(answer=answer, sources=sources)
