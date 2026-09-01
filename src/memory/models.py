"""
src/memory/models.py — SQLAlchemy schema for long-term user memory.

Three tables. The interesting one is `facts`, which is a bi-temporal edge list
rooted at the user — the "graph layer", without a graph database.

Every row is an edge:  (user) --[predicate]--> (object)

Two independent timelines are tracked per edge, which is what makes a clinical
memory trustworthy:

  valid_at / invalid_at  — when the fact was true IN THE WORLD
  recorded_at            — when the system learned it

That separation is what lets a symptom STOP being true. An append-only memory
would keep "irregular periods" alive forever alongside "periods are regular
again", and let retrieval ranking pick a winner. Here, resolving a symptom sets
invalid_at, the edge stops being current, and the history stays queryable.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ── Predicates (the edge types of the graph) ──────────────────────────────

class Asserter:
    USER = "user"
    SYSTEM = "system"


class Verification:
    REPORTED = "reported"     # the user said it (or the detector inferred it)
    CONFIRMED = "confirmed"   # explicitly reaffirmed
    REFUTED = "refuted"       # the user denied it


class Predicate:
    REPORTED_SYMPTOM   = "REPORTED_SYMPTOM"     # object = symptom tag
    SUGGESTED_TEST     = "SUGGESTED_TEST"       # object = lab test name
    RECEIVED_GUIDANCE  = "RECEIVED_GUIDANCE"    # object = topic tag
    EXPRESSED_EMOTION  = "EXPRESSED_EMOTION"    # object = emotion tag
    DISCLOSED_PCOS     = "DISCLOSED_PCOS"       # object = "pcos"
    MENTIONED_DOCTOR   = "MENTIONED_DOCTOR"     # object = "visit"

    ALL = {
        REPORTED_SYMPTOM, SUGGESTED_TEST, RECEIVED_GUIDANCE,
        EXPRESSED_EMOTION, DISCLOSED_PCOS, MENTIONED_DOCTOR,
    }


class User(Base):
    __tablename__ = "users"

    user_id:      Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at:   Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Stable profile. reason_for_visit is deliberately ABSENT: it is per-visit,
    # and carrying it forward makes the bot answer this month's consultation
    # with last month's reason.
    preferred_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    age_range:      Mapped[str | None] = mapped_column(String(40),  nullable=True)
    gender:         Mapped[str | None] = mapped_column(String(40),  nullable=True)

    consultation_count: Mapped[int] = mapped_column(Integer, default=0)

    facts:         Mapped[list["Fact"]]         = relationship(back_populates="user", cascade="all, delete-orphan")
    consultations: Mapped[list["Consultation"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Fact(Base):
    """One bi-temporal edge from the user to a value."""
    __tablename__ = "facts"

    id:      Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)

    predicate: Mapped[str] = mapped_column(String(40))
    object:    Mapped[str] = mapped_column(String(200))

    # World time
    valid_at:   Mapped[datetime]      = mapped_column(DateTime, default=utcnow)
    invalid_at: Mapped[datetime|None] = mapped_column(DateTime, nullable=True, default=None)
    # System time
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # How many times this EPISODE was reaffirmed. Drives "dominant concerns"
    # without any LLM having to decide what the user keeps coming back to.
    occurrences: Mapped[int] = mapped_column(Integer, default=1)

    # Which episode of this fact this row is: 1 for the first time the user
    # reported it, 2 after it resolved and came back, and so on. A recurrence
    # opens a NEW row rather than reopening the old one, so each episode keeps
    # its own [valid_at, invalid_at) interval. Collapsing them would destroy
    # exactly what a longitudinal symptom record is for — periodicity, episode
    # duration, and time to recurrence.
    episode: Mapped[int] = mapped_column(Integer, default=1)

    # Who asserted this. The predicate name encodes it implicitly today
    # (REPORTED_SYMPTOM is the user, SUGGESTED_TEST is the system), but that
    # breaks as soon as a predicate can come from either side.
    asserter: Mapped[str] = mapped_column(String(16), default="user")   # user | system

    # How firmly it is held. A regex false positive and a clearly stated symptom
    # are otherwise indistinguishable, leaving no way to express "unconfirmed".
    verification: Mapped[str] = mapped_column(String(16), default="reported")
    # reported | confirmed | refuted

    consultation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user: Mapped["User"] = relationship(back_populates="facts")

    __table_args__ = (
        # At most ONE live edge per (user, predicate, object). Resolved edges have
        # invalid_at set and are excluded, so history accumulates without ever
        # producing two contradictory "current" rows.
        Index("ix_facts_lookup", "user_id", "predicate", "invalid_at"),
    )


class Consultation(Base):
    __tablename__ = "consultations"

    consultation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id:         Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)

    started_at: Mapped[datetime]      = mapped_column(DateTime, default=utcnow)
    ended_at:   Mapped[datetime|None] = mapped_column(DateTime, nullable=True)

    phase_reached:  Mapped[int]        = mapped_column(Integer, default=1)
    closure_mode:   Mapped[str | None] = mapped_column(String(40), nullable=True)
    exchange_count: Mapped[int]        = mapped_column(Integer, default=0)
    summary:        Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="consultations")
