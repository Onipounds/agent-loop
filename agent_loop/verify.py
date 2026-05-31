"""Verification: the closed loop that stops the agent declaring success blindly.

The core check is *grounding*: every number the agent states in its final answer
must actually have appeared in a tool result during this run (within a small
tolerance). If the agent invents a figure, verification fails and the runtime
feeds the failure back so the agent can retry.

This is deliberately simple and transparent. The README explains the trade-off:
cheap, deterministic, catches the most common failure (hallucinated numbers),
at the cost of not understanding meaning. Stronger checks (re-computation,
semantic checks) can layer on top.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .memory import Memory

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Tokens like pump-7, S7-1500, abc123: a run containing letters and digits/
# hyphens. We blank these out before reading numbers so identifiers aren't
# mistaken for numeric claims.
_IDENT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-\d+)+\b|\b[A-Za-z]+\d[A-Za-z0-9]*\b")


@dataclass
class Check:
    ok: bool
    feedback: str = ""


def extract_numbers(text: str) -> list[float]:
    cleaned = _IDENT_RE.sub(" ", text)
    return [float(m) for m in _NUM_RE.findall(cleaned)]


def verify(task: str, answer: str, memory: Memory, tol: float = 0.01) -> Check:
    """Check that numbers in the answer are grounded in tool results."""
    answer_nums = extract_numbers(answer)
    if not answer_nums:
        # No numeric claims to ground; accept (a text answer).
        return Check(ok=True)

    seen = memory.numbers_seen()
    ungrounded = []
    for n in answer_nums:
        grounded = any(_close(n, s, tol) for s in seen)
        if not grounded:
            ungrounded.append(n)

    if ungrounded:
        return Check(
            ok=False,
            feedback=(
                "Verification failed: the values "
                f"{ungrounded} in your answer do not match any tool result. "
                "Only state numbers you obtained from a tool. Re-check using the "
                "tools and answer again."
            ),
        )
    return Check(ok=True)


def _close(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tol * scale
