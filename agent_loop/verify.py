"""Verification: the closed loop that stops the agent declaring success blindly.

Two layers, cheapest first:

1. **Grounding** (``verify_grounding``): every number in the answer must match a
   value that appeared in some tool result this run. Catches *invented* numbers.
2. **Re-computation** (``verify_recompute``): when the answer states an aggregate
   (mean / max / min) over a metric, independently recompute it from the raw
   readings in the tool results and confirm the agent's figure matches. Catches
   numbers that are grounded-looking but *wrong* (e.g. the agent fetched the data
   but computed the average incorrectly).

``verify`` runs grounding then re-computation; the first failure is returned and
fed back to the agent for a retry. Both are deterministic and need no model call.
The trade-off (see README): these reason about numbers, not meaning.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from .memory import Memory

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Tokens like pump-7, S7-1500, abc123: a run containing letters and digits/
# hyphens. We blank these out before reading numbers so identifiers aren't
# mistaken for numeric claims.
_IDENT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-\d+)+\b|\b[A-Za-z]+\d[A-Za-z0-9]*\b")

_AGG_WORDS = {
    "mean": "mean", "average": "mean", "avg": "mean",
    "maximum": "max", "max": "max", "highest": "max", "peak": "max",
    "minimum": "min", "min": "min", "lowest": "min",
}


@dataclass
class Check:
    ok: bool
    feedback: str = ""


def extract_numbers(text: str) -> list[float]:
    cleaned = _IDENT_RE.sub(" ", text)
    return [float(m) for m in _NUM_RE.findall(cleaned)]


def _close(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tol * scale


# --- layer 1: grounding -----------------------------------------------------

def verify_grounding(task: str, answer: str, memory: Memory, tol: float = 0.01) -> Check:
    """Check that numbers in the answer appeared in some tool result."""
    answer_nums = extract_numbers(answer)
    if not answer_nums:
        return Check(ok=True)  # no numeric claims to ground

    seen = memory.numbers_seen()
    ungrounded = [n for n in answer_nums if not any(_close(n, s, tol) for s in seen)]
    if ungrounded:
        return Check(
            ok=False,
            feedback=(
                f"Verification failed (grounding): the values {ungrounded} in your "
                "answer do not match any tool result. Only state numbers you obtained "
                "from a tool. Re-check using the tools and answer again."
            ),
        )
    return Check(ok=True)


# --- layer 2: re-computation ------------------------------------------------

def _raw_value_lists(memory: Memory) -> list[list[float]]:
    """Pull every numeric ``values`` list out of the tool results seen this run."""
    lists: list[list[float]] = []
    for result in memory.tool_results_seen:
        if isinstance(result, dict) and isinstance(result.get("values"), list):
            nums = [float(v) for v in result["values"]
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if nums:
                lists.append(nums)
    return lists


def _aggregate(kind: str, values: list[float]) -> float:
    if kind == "mean":
        return statistics.fmean(values)
    if kind == "max":
        return max(values)
    if kind == "min":
        return min(values)
    raise ValueError(kind)


def verify_recompute(task: str, answer: str, memory: Memory, tol: float = 0.01) -> Check:
    """If the task asks for an aggregate, recompute it from raw data and compare.

    Only fires when (a) the task names a single aggregate, (b) we have a raw
    values list to compute from, and (c) the answer states some number. Otherwise
    it abstains (returns ok) -- it never blocks answers it can't independently check.

    Note on tolerance: this uses a tight *absolute* tolerance, not the relative
    one used for grounding. Re-computation is checking whether a reported figure
    is *correct*, and a relative (percentage) tolerance scales with the magnitude
    of the values -- for readings around 70, a 1% band is +/-0.7, wide enough to
    accept a wrong mean. An aggregate should match to within rounding, so we
    compare absolutely.
    """
    kinds = {_AGG_WORDS[w] for w in _AGG_WORDS if re.search(rf"\b{w}\b", task.lower())}
    if len(kinds) != 1:
        return Check(ok=True)  # no single clear aggregate to re-check
    kind = next(iter(kinds))

    value_lists = _raw_value_lists(memory)
    if not value_lists:
        return Check(ok=True)  # nothing to recompute from

    answer_nums = extract_numbers(answer)
    if not answer_nums:
        return Check(ok=True)

    # The expected value is the aggregate over the most recent raw list.
    expected = _aggregate(kind, value_lists[-1])
    abs_tol = 0.05  # the answer should match the aggregate to within rounding
    if any(abs(n - expected) <= abs_tol for n in answer_nums):
        return Check(ok=True)

    return Check(
        ok=False,
        feedback=(
            f"Verification failed (re-computation): the {kind} of the data is "
            f"{round(expected, 4)}, which does not appear in your answer. "
            f"Recompute the {kind} from the readings and answer with the correct value."
        ),
    )


# --- combined ---------------------------------------------------------------

def verify(task: str, answer: str, memory: Memory, tol: float = 0.01) -> Check:
    """Run grounding, then re-computation. Return the first failure."""
    for layer in (verify_grounding, verify_recompute):
        check = layer(task, answer, memory, tol)
        if not check.ok:
            return check
    return Check(ok=True)

