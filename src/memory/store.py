"""
src/memory/store.py — Persistence API for long-term user memory.

Everything the app needs to remember is ALREADY structured before it gets here:
symptom tags come from regex intent detection, guidance topics and suggested
tests come from lookup tables. So this layer performs ZERO LLM calls — the fact
extraction that hosted memory vendors bill for is work this app already did
deterministically, for free.

All writes are idempotent per (user, predicate, object): re-observing a fact
bumps `occurrences` and `last_seen_at` rather than duplicating a row.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session as SASession, sessionmaker

from config.settings import MEMORY_DB_URL
from src.memory.models import (
    Asserter, Base, Consultation, Fact, Predicate, User, Verification, utcnow,
)

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def init_store(db_url: Optional[str] = None) -> None:
    """Create the engine and tables. Safe to call repeatedly."""
    global _engine, _SessionFactory
    url = db_url or MEMORY_DB_URL
    kwargs: dict = {"future": True}
    if url in ("sqlite://", "sqlite:///:memory:"):
        # An in-memory SQLite database lives inside a single CONNECTION, so a
        # second thread gets a fresh, empty one — "no such table". FastAPI runs
        # sync handlers in a threadpool, so API tests silently exercised no
        # memory at all (the errors are swallowed because memory is advisory).
        # StaticPool shares one connection across threads.
        from sqlalchemy.pool import StaticPool
        kwargs.update(poolclass=StaticPool,
                      connect_args={"check_same_thread": False})
    _engine = create_engine(url, **kwargs)
    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    logger.info("[memory] store ready — %s", url)


def reset_store_for_tests(db_url: str = "sqlite://") -> None:
    """Point the store at a throwaway in-memory database (tests only)."""
    init_store(db_url)


@contextmanager
def _session():
    if _SessionFactory is None:
        init_store()
    s: SASession = _SessionFactory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ── Users ─────────────────────────────────────────────────────────────────

def get_user(user_id: str) -> Optional[dict]:
    """
    Read a user without creating one.

    Reads must not write: get_or_create_user on the hydration path inserted a row
    for every first-time visitor, filling the table with people who never return.
    Rows are created by the write path instead, when there is something to
    remember.
    """
    with _session() as s:
        user = s.get(User, user_id)
        if user is None:
            return None
        return {
            "user_id": user.user_id,
            "preferred_name": user.preferred_name,
            "age_range": user.age_range,
            "gender": user.gender,
            "consultation_count": user.consultation_count,
            "created_at": user.created_at,
            "last_seen_at": user.last_seen_at,
            "is_new": False,
        }


def get_or_create_user(user_id: str) -> dict:
    """Fetch a user, creating the row if this id is new. Returns a plain dict."""
    with _session() as s:
        user = s.get(User, user_id)
        created = False
        if user is None:
            user = User(user_id=user_id)
            s.add(user)
            created = True
        else:
            user.last_seen_at = utcnow()
        s.flush()
        return {
            "user_id": user.user_id,
            "preferred_name": user.preferred_name,
            "age_range": user.age_range,
            "gender": user.gender,
            "consultation_count": user.consultation_count,
            "created_at": user.created_at,
            "last_seen_at": user.last_seen_at,
            "is_new": created,
        }


def update_profile(user_id: str, *, name: str | None = None,
                   age_range: str | None = None, gender: str | None = None) -> None:
    """
    Persist stable profile fields. Only non-None values are written, so a session
    that never learns a gender does not erase a previously known one.
    """
    if not (name or age_range or gender):
        # Nothing to store. Creating an empty row here would mint a user record
        # for anyone who merely sent a message, defeating the read-only rule.
        return

    with _session() as s:
        user = s.get(User, user_id)
        if user is None:
            user = User(user_id=user_id)
            s.add(user)
        if name:      user.preferred_name = name
        if age_range: user.age_range = age_range
        if gender:    user.gender = gender
        user.last_seen_at = utcnow()


# ── Facts (the bi-temporal edge list) ─────────────────────────────────────

def record_fact(user_id: str, predicate: str, obj: str,
                consultation_id: Optional[str] = None,
                asserter: str = Asserter.USER,
                verification: str = Verification.REPORTED) -> None:
    """
    Assert an edge.

    Idempotent within an episode: re-observing a LIVE edge bumps its counters.
    Re-observing a RESOLVED edge opens a NEW episode row, leaving the previous
    episode's interval intact.
    """
    if predicate not in Predicate.ALL:
        raise ValueError(f"unknown predicate: {predicate}")
    if not obj:
        return

    with _session() as s:
        live = s.execute(
            select(Fact).where(
                Fact.user_id == user_id,
                Fact.predicate == predicate,
                Fact.object == obj,
                Fact.invalid_at.is_(None),
            )
        ).scalar_one_or_none()

        if live is not None:
            live.occurrences += 1
            live.last_seen_at = utcnow()
            return

        # A recurrence opens a NEW episode. Reopening the resolved row in place
        # (clearing invalid_at and moving valid_at forward) would erase the
        # earlier [valid_at, invalid_at) interval — so "had irregular periods
        # Jan-Jun, then again from September" would collapse into one row
        # claiming it began in September. That destroys exactly what a
        # longitudinal symptom record exists to hold: periodicity, episode
        # duration, and time to recurrence.
        prior = s.execute(
            select(func.max(Fact.episode)).where(
                Fact.user_id == user_id,
                Fact.predicate == predicate,
                Fact.object == obj,
            )
        ).scalar()

        if s.get(User, user_id) is None:
            s.add(User(user_id=user_id))       # first thing worth remembering
        s.add(Fact(
            user_id=user_id, predicate=predicate, object=obj,
            episode=(prior or 0) + 1,
            asserter=asserter, verification=verification,
            consultation_id=consultation_id,
        ))


def record_facts(user_id: str, predicate: str, objects: Iterable[str],
                 consultation_id: Optional[str] = None,
                 asserter: str = Asserter.USER,
                 verification: str = Verification.REPORTED) -> None:
    for obj in objects:
        record_fact(user_id, predicate, obj, consultation_id, asserter, verification)


def resolve_fact(user_id: str, predicate: str, obj: str,
                 when: Optional[datetime] = None) -> bool:
    """
    Mark an edge as no longer true — the capability an append-only memory lacks.

    Sets invalid_at rather than deleting, so "had irregular periods Jan-Jun" stays
    answerable while ceasing to be a CURRENT symptom. Returns True if something
    was resolved.
    """
    with _session() as s:
        live = s.execute(
            select(Fact).where(
                Fact.user_id == user_id,
                Fact.predicate == predicate,
                Fact.object == obj,
                Fact.invalid_at.is_(None),
            )
        ).scalar_one_or_none()
        if live is None:
            return False
        live.invalid_at = when or utcnow()
        return True


def get_facts(user_id: str, predicate: Optional[str] = None,
              include_resolved: bool = False) -> list[dict]:
    """Read edges for a user, most-reaffirmed first."""
    with _session() as s:
        stmt = select(Fact).where(Fact.user_id == user_id)
        if predicate:
            stmt = stmt.where(Fact.predicate == predicate)
        if not include_resolved:
            stmt = stmt.where(Fact.invalid_at.is_(None))
        stmt = stmt.order_by(Fact.occurrences.desc(), Fact.last_seen_at.desc())
        return [
            {
                "predicate": f.predicate, "object": f.object,
                "valid_at": f.valid_at, "invalid_at": f.invalid_at,
                "occurrences": f.occurrences, "last_seen_at": f.last_seen_at,
                "episode": f.episode, "asserter": f.asserter,
                "verification": f.verification,
                "is_resolved": f.invalid_at is not None,
            }
            for f in s.execute(stmt).scalars()
        ]


def episode_history(user_id: str, predicate: str, obj: str) -> list[dict]:
    """
    Every episode of one fact, oldest first, each with its own interval.

    The query the bi-temporal model exists to answer: when has this symptom been
    present, and for how long each time?
    """
    with _session() as s:
        rows = s.execute(
            select(Fact).where(
                Fact.user_id == user_id,
                Fact.predicate == predicate,
                Fact.object == obj,
            ).order_by(Fact.episode)
        ).scalars().all()
        return [
            {
                "episode": f.episode,
                "valid_at": f.valid_at,
                "invalid_at": f.invalid_at,
                "occurrences": f.occurrences,
                "is_resolved": f.invalid_at is not None,
                "duration_days": (
                    (f.invalid_at - f.valid_at).days if f.invalid_at else None
                ),
            }
            for f in rows
        ]


def co_occurring_symptoms(user_id: str, tag: str) -> list[str]:
    """
    Graph traversal: which other symptoms are live alongside this one.

    A one-hop neighbourhood query over the edge list — the relationship reasoning
    a graph DB would provide, expressed as a self-join.
    """
    with _session() as s:
        anchor = s.execute(
            select(Fact.id).where(
                Fact.user_id == user_id,
                Fact.predicate == Predicate.REPORTED_SYMPTOM,
                Fact.object == tag,
                Fact.invalid_at.is_(None),
            )
        ).scalar_one_or_none()
        if anchor is None:
            return []
        rows = s.execute(
            select(Fact.object).where(
                Fact.user_id == user_id,
                Fact.predicate == Predicate.REPORTED_SYMPTOM,
                Fact.object != tag,
                Fact.invalid_at.is_(None),
            ).order_by(Fact.occurrences.desc())
        ).scalars().all()
        return list(rows)


# ── Consultations ─────────────────────────────────────────────────────────

def start_consultation(user_id: str, consultation_id: str) -> None:
    with _session() as s:
        if s.get(Consultation, consultation_id) is not None:
            return
        user = s.get(User, user_id)
        if user is None:
            # Called on the first real write of a turn, which may land before any
            # fact or profile field exists. Create the row here rather than
            # dropping the consultation record.
            user = User(user_id=user_id)
            s.add(user)
            s.flush()
        s.add(Consultation(consultation_id=consultation_id, user_id=user_id))
        user.consultation_count += 1
        user.last_seen_at = utcnow()


def finish_consultation(consultation_id: str, *, phase_reached: int,
                        exchange_count: int, closure_mode: Optional[str],
                        summary: Optional[str] = None) -> None:
    with _session() as s:
        c = s.get(Consultation, consultation_id)
        if c is None:
            return
        c.ended_at = utcnow()
        c.phase_reached = phase_reached
        c.exchange_count = exchange_count
        c.closure_mode = closure_mode
        c.summary = summary


def last_consultation_at(user_id: str) -> Optional[datetime]:
    with _session() as s:
        return s.execute(
            select(func.max(Consultation.started_at)).where(Consultation.user_id == user_id)
        ).scalar()


def forget_user(user_id: str) -> bool:
    """Erasure request: delete the user and every fact/consultation. Cascades."""
    with _session() as s:
        user = s.get(User, user_id)
        if user is None:
            return False
        s.delete(user)
        return True
