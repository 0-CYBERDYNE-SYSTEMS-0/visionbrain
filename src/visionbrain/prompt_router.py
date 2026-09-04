"""Prompt router — splits a user query into open-vocabulary SAM targets and a semantic question.

VisionBrain uses two types of AI models with different capabilities:
- SAM 3.1: open-vocabulary text prompts. Any noun phrase can be segmented and
  tracked — "kayak", "buoy", "crane", "yellow school bus" — with no built-in
  vocabulary. Plurals and multi-word phrases are handled by SAM natively.
- Falcon Perception + Gemma 4: reason about semantics — what the detections mean,
  whether something is damaged, what action to take.

This module routes a user query to the appropriate model(s) by splitting it into:
  - segment_targets: trackable noun phrases for SAM's multi-prompt detector
  - semantic_query: the original query, passed through unchanged (Gemma reasons
    over the full ask)

Routing is pure partitioning: the token stream is split into phrases at generic
stopwords (articles, conjunctions, prepositions, auxiliaries, filler verbs), so
this module contains NO domain vocabulary — every non-stopword phrase passes
through to SAM. Phrases are deduped case-insensitively (first-seen order) and
capped at 8 prompts (SAM multiplex sanity limit: more prompts than that degrade
detection quality and latency).

Example:
    route("boats and people near the pier")
    -> PromptResult(
        segment_targets=["boats", "people", "pier"],
        semantic_query="boats and people near the pier",
    )
    route("trucks blocking the north access road")
    -> PromptResult(
        segment_targets=["trucks blocking", "north access road"],
        semantic_query="trucks blocking the north access road",
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────────
# Generic closed-class words only — articles, conjunctions, prepositions,
# auxiliaries, and imperative/filler words. Deliberately contains NO domain
# nouns: SAM 3.1 is open-vocabulary, so every other phrase is a valid target.
# ──────────────────────────────────────────────────────────────────────────────

STOPWORDS: frozenset[str] = frozenset({
    # Articles
    "a", "an", "the",
    # Conjunctions
    "and", "or", "but",
    # Prepositions
    "in", "on", "at", "by", "near", "of", "to", "from", "with", "over",
    "under", "along", "across", "behind", "beside", "between", "around",
    "into", "onto", "off", "out", "up", "down", "through", "during",
    "before", "after", "against", "beyond", "within",
    # Auxiliaries / filler
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did",
    "please", "find", "show", "look", "detect", "track", "count", "watch",
    "me", "my", "we", "our", "all", "any", "some", "every",
    "that", "this", "these", "those", "it", "its", "there", "here",
})

# Maximum SAM prompts per query — more than this degrades the multiplex detector.
MAX_TARGETS: int = 8

_PUNCT_RE = re.compile(r'[.,;:!?"\'()]')
_NUMBER_RE = re.compile(r"[0-9.]+")


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    """Result of routing a user query.

    Attributes:
        segment_targets: open-vocabulary noun phrases for SAM 3.1's
                         multi-prompt detector, as typed and in first-seen
                         order, capped at MAX_TARGETS. Empty list means the
                         query was empty or stopword-only (caller should
                         handle fallback via route_fallback()).
        semantic_query: the original query, unchanged — Falcon Perception and
                        Gemma 4 reason over the full ask.
        original_query: the stripped query (for logging/debugging).
        routed_from: description of how the routing happened (for debugging).
    """
    segment_targets: list[str]
    semantic_query: str
    original_query: str
    routed_from: str


# ──────────────────────────────────────────────────────────────────────────────
# Routing logic
# ──────────────────────────────────────────────────────────────────────────────

def _is_pure_number(phrase: str) -> bool:
    """True when a phrase contains only digits and decimal points."""
    return bool(_NUMBER_RE.fullmatch(phrase.replace(" ", "")))


def _extract_phrases(query: str) -> list[str]:
    """Partition a query into noun phrases at stopwords.

    Phrases keep their original casing and token spelling (no stemming — SAM
    handles plurals and multi-word noun phrases natively). Empty phrases and
    pure numbers are dropped; the rest are deduped case-insensitively in
    first-seen order and capped at MAX_TARGETS.
    """
    # Lowercasing never introduces or removes whitespace, so this single
    # token list preserves the original casing while remaining comparable
    # against the lowercase stopword set.
    tokens = _PUNCT_RE.sub(" ", query).split()

    phrases: list[str] = []
    seen: set[str] = set()
    current: list[str] = []

    def _flush() -> None:
        if not current:
            return
        phrase = " ".join(current)
        key = phrase.lower()
        if key not in seen and not _is_pure_number(phrase):
            seen.add(key)
            phrases.append(phrase)
        current.clear()

    for tok in tokens:
        if tok.lower() in STOPWORDS:
            _flush()
        else:
            current.append(tok)
    _flush()

    return phrases[:MAX_TARGETS]


def route(query: str) -> PromptResult:
    """Split a user query into open-vocabulary SAM targets and a semantic question.

    The token stream is partitioned into noun phrases at generic stopwords and
    every phrase is passed to SAM 3.1 as typed (open-vocabulary pass-through,
    capped at MAX_TARGETS = 8 prompts). The semantic query is the full
    original text, unchanged.

    Args:
        query: natural-language query, e.g.
               "boats and people near the pier"

    Returns:
        PromptResult with SAM phrases and the full query for Falcon/Gemma.
        If no trackable phrases are found, segment_targets will be empty
        (caller should handle fallback via route_fallback()).
    """
    original = query.strip()
    segment_targets = _extract_phrases(original)

    if segment_targets:
        routed_from = f"SAM phrases: {segment_targets} | Gemma: full query"
    else:
        routed_from = "No trackable phrases — pass to Falcon/Gemma as-is"

    return PromptResult(
        segment_targets=segment_targets,
        semantic_query=original,
        original_query=original,
        routed_from=routed_from,
    )


def route_fallback(query: str) -> list[str]:
    """Return a default list of SAM prompts if route() produces no segment_targets.

    Used when the user query is empty or stopword-only and no trackable
    phrases can be extracted. Falls back to common general-purpose
    objects that cover the most ground.
    """
    return ["person", "vehicle", "building", "animal"]
