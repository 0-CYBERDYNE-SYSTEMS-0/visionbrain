"""Prompt router — splits a user query into concrete SAM targets and semantic reasoning questions.

VisionBrain uses two types of AI models with different capabilities:
- SAM 3.1: handles concrete, segmentable objects (person, car, roof, building).
  It cannot reason about abstract concepts like "damage", "opportunity", or "condition".
- Falcon Perception + Gemma 4: reason about semantics — what the detections mean,
  whether something is damaged, what action to take.

This module routes a user query to the appropriate model(s) by splitting it into:
  - segment_targets: concrete nouns SAM can find and track
  - semantic_query: the abstract question Falcon/Gemma should answer

Example:
    route("trucks blocking the north access road")
    -> PromptResult(
        segment_targets=["truck"],
        semantic_query="blocking the north access road"
    )
    route("person acting suspiciously near the gate")
    -> PromptResult(
        segment_targets=["person"],
        semantic_query="acting suspiciously near the gate"
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
# Concrete nouns that SAM 3.1 can segment — these are things SAM has
# been shown examples of and can detect with bounding boxes + masks.
# ──────────────────────────────────────────────────────────────────────────────

CONCRETE_NOUNS: set[str] = {
    # People
    "person", "people", "pedestrian", "worker", "crowd", "child", "adult",
    # Vehicles & transport
    "car", "truck", "pickup", "van", "bus", "suv", "motorcycle", "bicycle",
    "bike", "trailer", "tractor", "vehicle", "boat", "ship", "train",
    "aircraft", "airplane", "helicopter", "drone",
    # Animals (livestock and wildlife)
    "cow", "cattle", "bull", "steer", "heifer", "calf", "calves",
    "sheep", "lamb", "ewe", "ram",
    "goat", "kid", "nanny", "billy",
    "horse", "mare", "stallion", "gelding", "foal", "pony",
    "pig", "sow", "boar", "piglet", "hog",
    "chicken", "hen", "rooster", "poultry", "duck", "goose", "turkey",
    "llama", "alpaca", "donkey", "mule",
    "dog", "cat", "bird", "deer", "elk", "bear", "animal",
    # Structures & infrastructure
    "fence", "post", "rail", "gate", "barn", "shed", "structure",
    "building", "house", "silo", "roof", "shingles", "wall", "road",
    "bridge", "tower", "pole", "panel", "sign", "barrier", "container",
    "tank", "pond", "cistern", "corral", "pen", "enclosure",
    "hay", "bale", "stack",
    # Vegetation & terrain
    "crop", "row", "plant", "tree", "bush", "hedge",
    "grass", "weed", "brush", "scrub", "rock", "boulder",
    "stream", "creek", "river", "drainage", "ditch", "lake",
    "mud", "puddle", "flood", "erosion",
    # Equipment & objects
    "feeder", "bunk", "equipment", "machine", "dumpster", "pile",
    "camera", "sensor", "solar panel",
}

# Abstract/behavioral terms that SAM cannot segment — these are for Falcon/Gemma
ABSTRACT_TERMS: set[str] = {
    # Damage / health
    "damage", "damaged", "broken", "cracked", "leak", "leaking",
    "injury", "injured", "wound", "lame", "limping",
    "sick", "ill", "diseased", "infection", "infected",
    "dead", "dying", "carcass",
    # Condition / state
    "condition", "state", "status", "quality",
    "healthy", "unhealthy", "distressed", "stressed",
    "missing", "absent", "gone", "disappeared",
    "overgrown", "undergrown", "thin", "fat",
    # Anomaly / concern
    "anomaly", "unusual", "abnormal", "out of place",
    "opportunity", "concern", "issue", "problem",
    "risk", "threat", "danger", "dangerous",
    "predator", "intruder",
    # Behavior
    "behavior", "behaviour", "movement", "moving", "still",
    "grouped", "isolated", "alone", "separated",
    "grazing", "resting", "running", "fighting",
    "eating", "drinking",
    # Activity & behavior
    "parked", "moving", "stopped", "queued",
    "gathered", "scattered", "loitering", "wandering",
    # Site / environmental concerns
    "overgrown", "eroded", "flooded", "waterlogged",
    "bare patch", "bare ground", "mud hole",
    "litter", "debris", "spill",
}

# Multi-word compounds — match as a unit for SAM targeting.
# (compound_label, set of surface forms)
COMPOUND_MAP: list[tuple[str, set[str]]] = [
    ("trough", {"water trough", "feeding trough", "water tank", "feeding tank"}),
    ("fence", {"fence down", "fence damage", "fence broken", "fence breach", "fence gap"}),
    ("pasture", {"bare pasture", "overgrazed pasture"}),
    ("bare ground", {"bare patch", "bare ground", "bare spot", "bare area"}),
    ("water", {"standing water", "flooded area", "water puddle"}),
    ("vehicle", {"parked car", "parked truck", "moving vehicle", "stalled vehicle"}),
]


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    """Result of routing a user query.

    Attributes:
        segment_targets: concrete nouns SAM 3.1 can detect and track.
                         These labels are passed to SAM's multi-prompt detector.
                         Empty list means no concrete objects in the query.
        semantic_query: the abstract reasoning question for Falcon Perception
                        and Gemma 4. This is the "why" behind the search —
                        what the user wants to understand about the footage.
        original_query: the unmodified query (for logging/debugging).
        routed_from: description of how the routing happened (for debugging).
    """
    segment_targets: list[str]
    semantic_query: str
    original_query: str
    routed_from: str


# ──────────────────────────────────────────────────────────────────────────────
# Routing logic
# ──────────────────────────────────────────────────────────────────────────────

def _extract_compound_labels(query_lower: str) -> tuple[set[str], set[int]]:
    """Find all matched compound patterns.

    Returns:
        (set of canonical labels, set of word positions that are part of compounds)
    """
    labels: set[str] = set()
    skip_words: set[int] = set()

    for label, forms in COMPOUND_MAP:
        for form in forms:
            words = form.split()
            pattern = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
            if re.search(pattern, query_lower):
                labels.add(label)
                # Mark all words in the compound as "skip" for further extraction
                # (we'll figure out positions from the full query split)
                for w in words:
                    skip_words.add(w)  # type: ignore[arg-type]

    return labels, skip_words


def route(query: str) -> PromptResult:
    """Split a user query into SAM targets and semantic questions.

    Args:
        query: natural-language query, e.g.
               "person acting suspiciously near the gate"

    Returns:
        PromptResult with concrete SAM targets and the semantic reasoning question.
        If no concrete objects are found, segment_targets will be empty and
        semantic_query will be the original query (caller should handle fallback).
    """
    original = query.strip()
    if not original:
        return PromptResult(
            segment_targets=[],
            semantic_query="",
            original_query=original,
            routed_from="empty query",
        )

    q_lower = original.lower()

    # ── Step 1: Extract compound patterns ─────────────────────────────────────
    compound_labels: set[str] = set()
    # Collect all words that are part of compounds so we can skip them later
    compound_word_set: set[str] = set()
    for label, forms in COMPOUND_MAP:
        for form in forms:
            words = form.split()
            pattern = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
            if re.search(pattern, q_lower):
                compound_labels.add(label)
                compound_word_set.update(words)

    # ── Step 2: Find which compound forms actually appear in the query ─────────
    # and record their word positions for removal
    matched_compound_words: list[str] = []
    for label, forms in COMPOUND_MAP:
        for form in forms:
            pattern = r'\b' + r'\s+'.join(re.escape(w) for w in form.split()) + r'\b'
            if re.search(pattern, q_lower):
                matched_compound_words.extend(form.split())
    matched_compound_word_set = set(matched_compound_words)

    # ── Step 3: Extract concrete nouns ─────────────────────────────────────────
    found_concrete: set[str] = set()
    words_original = original.split()
    words_lower = [w.lower().rstrip(".,!?") for w in words_original]

    for i, (wl, w) in enumerate(zip(words_lower, words_original)):
        # Skip words that are part of a matched compound
        if w.lower().rstrip(".,!?") in matched_compound_word_set:
            continue

        # Match against CONCRETE_NOUNS (allow 's'/'es' plural suffix)
        for noun in CONCRETE_NOUNS:
            if wl == noun or wl == noun + "s" or wl == noun + "es":
                # Don't add "field" as a SAM target in most contexts
                # (it's usually a location, not a segmentable object in queries)
                if noun not in ("field",):
                    found_concrete.add(noun)
                break

    # Combine compound labels and found concrete nouns
    all_targets = compound_labels | found_concrete
    segment_targets = sorted(all_targets)

    # ── Step 4: Build semantic query ───────────────────────────────────────────
    # Remove concrete noun words and compound words from original query.
    # Keep abstract terms and key content words.
    skip_lower = matched_compound_word_set | {
        noun
        for noun in CONCRETE_NOUNS
        if noun not in ("field",)
    }
    skip_plurals = {noun + "s" for noun in skip_lower} | {noun + "es" for noun in skip_lower}

    semantic_parts = []
    for wl, w in zip(words_lower, words_original):
        # Skip if word is a concrete noun (or its plural)
        if wl in skip_lower or wl in skip_plurals:
            continue
        # Keep the original word casing
        semantic_parts.append(w)

    semantic_query = " ".join(semantic_parts).strip()

    # If semantic query is mostly stopwords, use the abstract terms we found instead
    if len(semantic_query) < 3 or not re.search(r'[a-z]{3,}', semantic_query):
        abstract_parts = []
        for term in ABSTRACT_TERMS:
            pattern = rf"\b{re.escape(term)}(s?ed)?(?![a-z])"
            if re.search(pattern, q_lower):
                # Extract the matching text
                m = re.search(pattern, q_lower)
                if m:
                    abstract_parts.append(m.group(0))
        if abstract_parts:
            semantic_query = " ".join(abstract_parts)
        else:
            semantic_query = original

    # ── Step 5: Determine routing explanation ──────────────────────────────────
    if segment_targets and semantic_query:
        routed_from = f"SAM: {segment_targets} | Falcon/Gemma: {semantic_query}"
    elif segment_targets:
        routed_from = f"SAM only: {segment_targets}"
    else:
        routed_from = "No concrete targets — pass to Falcon/Gemma as-is"

    return PromptResult(
        segment_targets=segment_targets,
        semantic_query=semantic_query,
        original_query=original,
        routed_from=routed_from,
    )


def route_fallback(query: str) -> list[str]:
    """Return a default list of SAM prompts if route() produces no segment_targets.

    Used when the user query is purely abstract (e.g. "anomalies on the site")
    and no concrete objects can be extracted. Falls back to common general-purpose
    objects that cover the most ground.
    """
    return ["person", "vehicle", "building", "animal"]