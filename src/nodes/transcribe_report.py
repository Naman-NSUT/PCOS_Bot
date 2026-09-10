"""
src/nodes/transcribe_report.py
Report Node 0 — turn photographs of a lab report into text.

SCOPE DISCIPLINE, and it is the whole point of this module: the vision model is
an OCR layer. It transcribes what is printed and nothing else. It does not
classify values, compare them to reference ranges, or say anything clinical.

Everything downstream — reference ranges, derived ratios, indicator flags —
stays in the existing deterministic engine (parse_report -> flag_indicators),
which is auditable and unit-tested. Letting a model both read AND judge would
quietly move the clinical logic into an unverifiable place, which is exactly
the property this codebase is built to avoid.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Sequence

from openai import OpenAI

from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, VISION_MODEL

logger = logging.getLogger(__name__)

# Images dominate the token bill: a full-resolution phone photo cost ~25k input
# tokens in testing. Long edge 1400px keeps small print legible while cutting
# that substantially, and lab reports are high-contrast text so they survive it.
MAX_EDGE = 1400
MAX_IMAGES = 6          # a multi-page panel; beyond this something is wrong
MAX_BYTES = 8 * 1024 * 1024


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


_SYSTEM = """\
You transcribe laboratory reports from photographs. You are an OCR step, not a \
clinician.

Rules:
- Transcribe ONLY what is printed. Never infer, interpret, diagnose, or comment.
- Never invent an analyte that is not visibly present.
- If a value is blurred, cut off, or ambiguous, OMIT it rather than guessing.
- Copy the units exactly as printed.
- Put non-numeric printed findings (ultrasound comments, technician remarks) in
  "notes" verbatim. Do not summarise them.

Respond with ONLY this JSON object:
{
  "analytes": [{"name": "<as printed>", "value": <number>, "unit": "<as printed>"}],
  "notes": ["<verbatim line>"],
  "unreadable": ["<name of any analyte you could see but could not read>"]
}
"""


def downscale(image_bytes: bytes, max_edge: int = MAX_EDGE) -> bytes:
    """Shrink an oversized photo. Falls back to the original if Pillow is absent."""
    try:
        from PIL import Image
    except ImportError:
        return image_bytes
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if max(img.size) <= max_edge:
            return image_bytes
        ratio = max_edge / max(img.size)
        img = img.convert("RGB").resize(
            (int(img.width * ratio), int(img.height * ratio)),
            Image.LANCZOS,
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("[transcribe] downscale failed, sending original: %s", exc)
        return image_bytes


def _as_data_url(image_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()


def transcribe_images(images: Sequence[bytes]) -> Dict[str, Any]:
    """
    Read one or more report photos into structured analytes.

    Returns {"analytes": [...], "notes": [...], "unreadable": [...], "error": str|None}.
    Never raises: a failed transcription must degrade to "I couldn't read that",
    not break the consultation.
    """
    if not images:
        return {"analytes": [], "notes": [], "unreadable": [], "error": "no images"}

    if len(images) > MAX_IMAGES:
        logger.warning("[transcribe] %d images, using first %d", len(images), MAX_IMAGES)
        images = list(images)[:MAX_IMAGES]

    content: List[Dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Transcribe every analyte from these {len(images)} lab report page(s). "
            "Pages may continue from one another; merge them into one list."
        ),
    }]
    for raw in images:
        if len(raw) > MAX_BYTES:
            logger.warning("[transcribe] skipping oversized image (%d bytes)", len(raw))
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": _as_data_url(downscale(raw))},
        })

    if len(content) == 1:
        return {"analytes": [], "notes": [], "unreadable": [], "error": "no usable images"}

    try:
        resp = _get_client().chat.completions.create(
            model=VISION_MODEL,
            temperature=0,                     # transcription, not composition
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": content},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as exc:
        logger.error("[transcribe] vision call failed: %s", exc)
        return {"analytes": [], "notes": [], "unreadable": [], "error": str(exc)}

    analytes = _clean_analytes(data.get("analytes"))
    notes = [str(n) for n in (data.get("notes") or []) if str(n).strip()][:10]
    unreadable = [str(u) for u in (data.get("unreadable") or []) if str(u).strip()][:10]

    logger.info(
        "[transcribe] %d page(s) -> %d analyte(s), %d note(s), %d unreadable",
        len(images), len(analytes), len(notes), len(unreadable),
    )
    return {"analytes": analytes, "notes": notes, "unreadable": unreadable, "error": None}


def _clean_analytes(raw: Any) -> List[Dict[str, Any]]:
    """
    Keep only well-formed entries. The model controls this payload entirely, so a
    malformed row degrades that one analyte rather than failing the report.
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        out.append({
            "name": name,
            "value": value,
            "unit": str(item.get("unit", "")).strip(),
        })
    return out


def _norm(s: str) -> str:
    """Lowercase, strip everything that is not a letter or digit."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _alias_index() -> List[tuple]:
    """
    (normalised_alias, canonical_name) pairs, longest alias first.

    Built FROM the parser's own table so the two can never drift apart. Adding a
    biomarker in parse_report.py automatically teaches the normaliser about it.
    """
    from src.nodes.parse_report import _BIOMARKERS
    pairs = []
    for canonical, aliases, _units in _BIOMARKERS:
        for alias in aliases:
            # aliases use "." as a regex wildcard (e.g. "DHEA.S"); drop it here
            pairs.append((_norm(alias.replace(".", "")), canonical))
        pairs.append((_norm(canonical), canonical))
    return sorted(set(pairs), key=lambda p: -len(p[0]))


# Printed names that a fuzzy alias match would silently mis-assign. Each one is
# a real analyte that appears on ordinary PCOS-workup panels alongside the one
# it would collide with.
_NEVER_FUZZY = [
    re.compile(r"\bfree\s*t\s*[34]\b|\bft\s*[34]\b", re.IGNORECASE),   # thyroid, not androgen
    re.compile(r"triiodothyronin|thyroxin", re.IGNORECASE),
    re.compile(r"\bnon[\s-]*hdl\b", re.IGNORECASE),                       # not HDL
    re.compile(r"\bmacroprolactin\b", re.IGNORECASE),                     # not prolactin
    re.compile(r"\b(post[\s-]*prandial|pp|random|2\s*h(ou)?r)\b", re.IGNORECASE),  # not fasting
    re.compile(r"\bratio\b", re.IGNORECASE),                              # lipid ratios
    re.compile(r"\banti[\s-]*tpo|thyroid\s*peroxidase|\btg\s*ab\b", re.IGNORECASE),
]


def normalise_analyte_name(name: str) -> str:
    """
    Map a name as PRINTED on a report onto the canonical name the parser knows.

    The vision step is told to transcribe verbatim, which is right — but labs
    print "Testosterone, Total", "DHEA-Sulphate", "Luteinizing Hormone (LH)".
    Feeding those straight to the regex parser silently dropped both androgens,
    which then made the hyperandrogenism criterion look unsupported by the labs
    when it was in fact the strongest signal on the page.

    Normalising here rather than asking the model for canonical names keeps
    interpretation out of the model and in testable code.
    """
    raw = name.strip()

    # Labs commonly print "Long Name (ABBREV)" — the abbreviation is the most
    # reliable key, so try it first.
    candidates = []
    if "(" in raw and ")" in raw:
        inner = raw[raw.rfind("(") + 1: raw.rfind(")")]
        if inner.strip():
            candidates.append(inner)
        candidates.append(raw[: raw.rfind("(")])
    candidates.append(raw)

    # Anything matching one of these is NOT the analyte a fuzzy match would
    # claim, and must never be normalised. The fuzzy pass previously mapped
    # thyroid "Free T3"/"Free T4" onto "Free Testosterone" — the alias "Free T"
    # normalises to "freet" and "freet3".startswith("freet") — so a completely
    # normal thyroid panel produced Free Testosterone 3.1 against a (0.3, 1.9)
    # range, i.e. critical_high, and asserted biochemical hyperandrogenism.
    # That is the single most consequential lab claim this system makes.
    for rx in _NEVER_FUZZY:
        if rx.search(raw):
            return raw

    index = _alias_index()
    for cand in candidates:
        n = _norm(cand)
        if not n:
            continue
        for alias, canonical in index:            # exact only
            if n == alias:
                return canonical

    # Fuzzy passes are restricted to aliases long enough to be unambiguous, and
    # a candidate may only EXTEND an alias with non-digits: "free t" -> "free t3"
    # is a different analyte, whereas "testosterone" -> "testosterone, total" is
    # the same one written out.
    for cand in candidates:
        n = _norm(cand)
        if not n:
            continue
        for alias, canonical in index:
            if len(alias) < 5:
                continue
            if n.startswith(alias) and not n[len(alias):len(alias) + 1].isdigit():
                return canonical
            if n.endswith(alias):
                return canonical
        for alias, canonical in index:
            if len(alias) >= 6 and alias in n:
                return canonical
    return raw


def to_report_text(analytes: Sequence[Dict[str, Any]]) -> str:
    """
    Render transcribed analytes into the canonical text the existing regex parser
    already understands.

    Deliberately routes back through parse_report_node rather than trusting the
    model's structure directly: that keeps ONE implementation of reference ranges,
    unit handling and derived values (LH/FSH, HOMA-IR), and it is the tested one.
    """
    return ", ".join(
        f"{normalise_analyte_name(a['name'])}: {a['value']}"
        f"{(' ' + a['unit']) if a['unit'] else ''}"
        for a in analytes
    )
