"""
src/core/identity.py — Durable, server-issued user identity.

Cross-session memory needs a stable key, but a raw client-supplied id would turn
an unguessable ephemeral token into a forgeable permanent key to someone's health
record: type a stranger's id, read their symptom history.

So the server mints the id and returns it wrapped in an HMAC signature. The client
stores the signed token and sends it back; anything that fails verification is
treated as a brand-new user rather than trusted. stdlib only — no new dependency.

Token format:  <user_id>.<hex hmac-sha256 of user_id>
"""
from __future__ import annotations

import hmac
import logging
import secrets
from hashlib import sha256
from typing import Optional

from config.settings import APP_SECRET_KEY

logger = logging.getLogger(__name__)

_SEPARATOR = "."
_USER_ID_BYTES = 16          # 128 bits of entropy — not enumerable


def _signature(user_id: str) -> str:
    return hmac.new(
        APP_SECRET_KEY.encode("utf-8"),
        user_id.encode("utf-8"),
        sha256,
    ).hexdigest()


def mint_user_token() -> tuple[str, str]:
    """
    Create a brand-new user identity.
    Returns (user_id, signed_token). Store the id; hand the token to the client.
    """
    user_id = secrets.token_hex(_USER_ID_BYTES)
    return user_id, f"{user_id}{_SEPARATOR}{_signature(user_id)}"


def sign_user_id(user_id: str) -> str:
    """Re-wrap a known user_id as a signed token (for responses)."""
    return f"{user_id}{_SEPARATOR}{_signature(user_id)}"


def verify_user_token(token: Optional[str]) -> Optional[str]:
    """
    Validate a client-supplied token and return the user_id it carries.

    Returns None for anything untrusted — missing, malformed, or wrong signature.
    Callers must treat None as "new user", never as an error to surface, so a
    tampered token degrades to a fresh consultation instead of leaking a record.
    """
    if not token or _SEPARATOR not in token:
        return None

    user_id, _, provided = token.rpartition(_SEPARATOR)
    if not user_id or not provided:
        return None

    # compare_digest: constant-time, so a wrong signature cannot be brute-forced
    # byte-by-byte from response timing. It raises TypeError on a str containing
    # non-ASCII, so compare bytes — a token with a stray unicode character must
    # degrade to "new user", not 500 the request.
    try:
        expected = _signature(user_id).encode("ascii")
        if not hmac.compare_digest(provided.encode("utf-8"), expected):
            logger.warning("[identity] rejected token with bad signature")
            return None
    except (UnicodeError, TypeError, ValueError):
        logger.warning("[identity] rejected malformed token")
        return None

    return user_id
