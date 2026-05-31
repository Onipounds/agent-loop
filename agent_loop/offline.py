"""A tiny deterministic 'planner' for offline mode.

This is NOT a model. It's a rule-based stand-in that reads the task text,
decides which tools to call, and scripts a FakeModel to call them and then
answer. It exists so the agent runs end-to-end with no API key -- for demos,
CI, and so reviewers can try it instantly.

The real intelligence in real mode comes from AnthropicModel; this just makes
the offline experience honest and reproducible. It is intentionally simple and
only understands the demo task shapes (averages, max, thresholds on telemetry).
"""
from __future__ import annotations

import re
import statistics

from .models import FakeModel, Scripted
from .tools import build_default_registry, ToolRegistry, query_telemetry


_DEVICE_RE = re.compile(r"\b([a-z]+-\d+)\b", re.I)
_LIMIT_RE = re.compile(r"limit of (\d+(?:\.\d+)?)", re.I)


def _pick_metric(task: str) -> str:
    t = task.lower()
    for metric in ("temperature", "pressure", "vibration"):
        if metric in t:
            return metric
    return "temperature"


def _pick_agg(task: str) -> str:
    t = task.lower()
    if "max" in t or "maximum" in t or "highest" in t:
        return "max"
    if "min" in t or "minimum" in t or "lowest" in t:
        return "min"
    return "mean"


def make_offline_model(task: str, tools: ToolRegistry | None = None) -> FakeModel:
    """Build a FakeModel scripted to solve a telemetry task deterministically."""
    device_m = _DEVICE_RE.search(task)
    device = device_m.group(1) if device_m else "pump-3"
    metric = _pick_metric(task)
    agg = _pick_agg(task)
    limit_m = _LIMIT_RE.search(task)

    # Compute the true values from the data so the script is correct & grounded.
    data = query_telemetry(device=device, metric=metric, window=60)
    values = data.get("values", [])
    if not values:
        return FakeModel([Scripted(text=f"I could not find data for {device} {metric}.")])

    if agg == "max":
        agg_value = max(values)
        expr = f"max({','.join(map(str, values))})"
    elif agg == "min":
        agg_value = min(values)
        expr = f"min({','.join(map(str, values))})"
    else:
        agg_value = round(statistics.fmean(values), 2)
        expr = f"round(mean({','.join(map(str, values))}),2)"

    script = [
        Scripted(tool_calls=[("query_telemetry", {"device": device, "metric": metric})]),
        Scripted(tool_calls=[("calculate", {"expression": expr})]),
    ]

    if limit_m:
        limit = float(limit_m.group(1))
        within = agg_value <= limit
        script.append(
            Scripted(tool_calls=[("check_threshold",
                                  {"value": agg_value, "limit": limit, "kind": "max"})])
        )
        verdict = "within" if within else "above"
        script.append(Scripted(
            text=f"The {agg} {metric} on {device} is {agg_value}, "
                 f"which is {verdict} the safe limit of {limit}."
        ))
    else:
        script.append(Scripted(
            text=f"The {agg} {metric} on {device} is {agg_value}."
        ))

    return FakeModel(script)
