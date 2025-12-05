# src/affectgpt/memory_system.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime

from .emotion import PAD, NeuroChemistry


@dataclass
class MemoryItem:
    """
    A simple memory record.

    kind: "stm" (short-term) or "ltm" (long-term)
    text: human-readable summary
    label: emotion label from the appraiser (e.g. "sadness", "joy", "safety")
    importance: [0,1] heuristic importance score
    pad: PAD state at time of memory
    brain: NeuroChemistry state at time of memory
    """
    kind: str
    text: str
    label: str
    importance: float
    pad: PAD
    brain: NeuroChemistry
    timestamp: datetime


# ── Heuristics ────────────────────────────────────────────────────────────────

def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def score_importance(user_text: str, label: str) -> float:
    """
    Rough importance heuristic:
    - higher for self-referential + emotional content
    - very high for safety / crisis style content
    """
    text = user_text.lower()
    score = 0.2  # base

    # self-related
    if _contains_any(text, ["i ", "i'm", "im ", "my ", "me ", "mine "]):
        score += 0.2

    # preferences / identity
    if _contains_any(text, ["i like", "i love", "i enjoy", "i hate", "i prefer"]):
        score += 0.2

    # time / plans
    if _contains_any(text, ["tomorrow", "next week", "in a year", "plan", "goal"]):
        score += 0.1

    # strong emotion labels from the appraiser
    if label in {"sadness", "anxiety", "anger", "shame", "fear"}:
        score += 0.2
    if label in {"joy", "gratitude", "pride"}:
        score += 0.15

    # safety / crisis style
    if label == "safety":
        score = 0.95

    # clip to [0,1]
    return max(0.0, min(1.0, score))


def summarise_interaction(user_text: str, label: str) -> str:
    """
    Create a compact LTM summary phrase.
    """
    cleaned = " ".join(user_text.strip().split())
    if len(cleaned) > 140:
        cleaned = cleaned[:140] + "…"

    if label == "safety":
        return f"The user expressed a crisis/safety concern: \"{cleaned}\""
    elif label in {"sadness", "anxiety", "fear", "anger"}:
        return f"The user felt {label} about: \"{cleaned}\""
    elif label in {"joy", "gratitude", "pride"}:
        return f"The user shared a positive moment ({label}): \"{cleaned}\""
    else:
        return f"User said: \"{cleaned}\""


# ── Memory update + retrieval ────────────────────────────────────────────────

def update_memories(
    stm: List[MemoryItem],
    ltm: List[MemoryItem],
    *,
    user_text: str,
    bot_text: str,
    label: str,
    pad: PAD,
    brain: NeuroChemistry,
    max_stm: int = 20,
    max_ltm: int = 12,
    ltm_threshold: float = 0.6,
) -> Tuple[List[MemoryItem], List[MemoryItem]]:
    """
    Update short-term and long-term memories given a completed turn.
    Returns (new_stm, new_ltm).
    """
    importance = score_importance(user_text, label)
    now = datetime.utcnow()

    # Short-term memory: store the full exchange.
    stm_item = MemoryItem(
        kind="stm",
        text=f"User: {user_text}\nBot: {bot_text}",
        label=label,
        importance=importance,
        pad=pad,
        brain=brain,
        timestamp=now,
    )
    new_stm = stm + [stm_item]
    if len(new_stm) > max_stm:
        new_stm = new_stm[-max_stm:]

    # Long-term memory: only store important stuff.
    new_ltm = list(ltm)
    if importance >= ltm_threshold:
        summary = summarise_interaction(user_text, label)
        ltm_item = MemoryItem(
            kind="ltm",
            text=summary,
            label=label,
            importance=importance,
            pad=pad,
            brain=brain,
            timestamp=now,
        )
        new_ltm.append(ltm_item)
        # keep only the most important N
        new_ltm = sorted(new_ltm, key=lambda m: m.importance, reverse=True)[:max_ltm]

    return new_stm, new_ltm


def build_memory_context(
    stm: List[MemoryItem],
    ltm: List[MemoryItem],
    max_stm: int = 4,
    max_ltm: int = 6,
) -> str:
    """
    Build a text block describing key memories, to prepend to the LLM prompt.
    """
    parts: List[str] = []
    if ltm:
        parts.append("Important things the user has shared before:")
        for m in sorted(ltm, key=lambda m: m.timestamp, reverse=True)[:max_ltm]:
            parts.append(f"- {m.text}")

    if stm:
        parts.append("")
        parts.append("Recent conversation snippets:")
        for m in stm[-max_stm:]:
            parts.append(f"- {m.text.replace(chr(10), ' / ')}")

    return "\n".join(parts).strip()
