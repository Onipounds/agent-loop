"""Test suite. All deterministic, no network, no API key."""
from __future__ import annotations

import pytest

from agent_loop.tools import build_default_registry, calculate, check_threshold, query_telemetry
from agent_loop.memory import Memory
from agent_loop.models import FakeModel, Scripted, ToolCall
from agent_loop.verify import verify, extract_numbers
from agent_loop.runtime import run
from agent_loop.offline import make_offline_model
from agent_loop.evals import load_tasks, run_evals, check_task, Task
from agent_loop.runtime import Result


# --- tools ------------------------------------------------------------------

def test_calculate_mean():
    assert calculate("mean(2,4,6)") == {"result": 4.0}


def test_calculate_blocks_builtins():
    out = calculate("__import__('os').system('echo hi')")
    assert "error" in out  # no builtins available -> error, not execution


def test_query_telemetry_returns_values():
    out = query_telemetry("pump-3", "temperature", window=5)
    assert out["count"] == 5
    assert all(isinstance(v, (int, float)) for v in out["values"])


def test_query_telemetry_unknown_key():
    out = query_telemetry("nope", "temperature")
    assert "error" in out and "available" in out


def test_check_threshold_max():
    assert check_threshold(70, 75, "max")["within_limit"] is True
    assert check_threshold(80, 75, "max")["within_limit"] is False


def test_registry_executes_and_handles_bad_args():
    reg = build_default_registry()
    assert reg.execute("calculate", {"expression": "sum(1,2,3)"}) == {"result": 6}
    assert "error" in reg.execute("calculate", {"wrong": "arg"})
    assert "error" in reg.execute("does_not_exist", {})


def test_registry_schemas_shape():
    reg = build_default_registry()
    schemas = reg.schemas()
    assert {s["name"] for s in schemas} == {"calculate", "query_telemetry", "check_threshold"}
    for s in schemas:
        assert "input_schema" in s and s["input_schema"]["type"] == "object"


# --- memory -----------------------------------------------------------------

def test_memory_records_numbers_seen():
    mem = Memory()
    call = ToolCall(id="1", name="query_telemetry", input={})
    mem.add_assistant_tool_calls([call])
    mem.add_tool_results([(call, {"values": [70.0, 71.5], "count": 2})])
    seen = mem.numbers_seen()
    assert 70.0 in seen and 71.5 in seen and 2.0 in seen


def test_memory_numbers_ignore_booleans():
    mem = Memory()
    call = ToolCall(id="1", name="check_threshold", input={})
    mem.add_tool_results([(call, {"within_limit": True, "limit": 75})])
    assert 75.0 in mem.numbers_seen()
    assert 1.0 not in mem.numbers_seen()  # True must not become 1.0


# --- verification -----------------------------------------------------------

def test_extract_numbers():
    assert extract_numbers("avg is 70.8 and max 88") == [70.8, 88.0]


def test_verify_passes_text_answer():
    assert verify("q", "All systems nominal.", Memory()).ok


def test_verify_rejects_ungrounded_number():
    mem = Memory()
    call = ToolCall(id="1", name="t", input={})
    mem.add_tool_results([(call, {"values": [70.0, 71.0]})])
    check = verify("q", "The average is 99.9", mem)
    assert not check.ok
    assert "99.9" in check.feedback


def test_verify_accepts_grounded_number():
    mem = Memory()
    call = ToolCall(id="1", name="t", input={})
    mem.add_tool_results([(call, {"result": 70.8})])
    assert verify("q", "The average is 70.8", mem).ok


# --- runtime loop -----------------------------------------------------------

def test_run_completes_multistep_task():
    tools = build_default_registry()
    model = FakeModel([
        Scripted(tool_calls=[("query_telemetry", {"device": "pump-3", "metric": "temperature"})]),
        Scripted(tool_calls=[("calculate", {"expression": "round(mean(70.0,71.0),2)"})]),
        Scripted(text="The average temperature on pump-3 is 70.5."),
    ])
    result = run("avg temp pump-3?", model, tools)
    assert result.verified
    assert "70.5" in result.answer
    assert result.n_model_calls == 3


def test_run_rejects_then_retries():
    tools = build_default_registry()
    # First answer invents a number -> rejected; second is grounded -> accepted.
    model = FakeModel([
        Scripted(tool_calls=[("calculate", {"expression": "round(mean(70.0,71.0),2)"})]),
        Scripted(text="The average is 123.4"),     # ungrounded -> rejected
        Scripted(text="The average is 70.5"),      # grounded   -> accepted
    ])
    result = run("avg?", model, tools)
    assert result.verified
    assert any(s.kind == "rejected" for s in result.steps)
    assert "70.5" in result.answer


def test_run_handles_tool_error_without_crashing():
    tools = build_default_registry()
    model = FakeModel([
        Scripted(tool_calls=[("query_telemetry", {"device": "ghost", "metric": "temperature"})]),
        Scripted(text="No data was available."),
    ])
    result = run("data for ghost?", model, tools)
    assert result.verified  # text answer, no numbers to ground


def test_run_respects_max_steps():
    tools = build_default_registry()
    # Always asks for a tool, never answers -> must stop at max_steps.
    model = FakeModel([Scripted(tool_calls=[("calculate", {"expression": "1+1"})])] * 20)
    result = run("loop forever", model, tools, max_steps=3)
    assert not result.verified
    assert result.n_model_calls == 3


# --- offline planner + evals ------------------------------------------------

def test_offline_model_solves_threshold_task():
    tools = build_default_registry()
    task = "What was the average temperature on pump-3, within the safe limit of 75 degrees?"
    result = run(task, make_offline_model(task, tools), tools)
    assert result.verified
    assert "within" in result.answer.lower()


def test_offline_model_max_query():
    tools = build_default_registry()
    task = "What is the maximum temperature on pump-7?"
    result = run(task, make_offline_model(task, tools), tools)
    assert result.verified
    assert "88" in result.answer


def test_evals_all_pass_offline():
    tools = build_default_registry()
    tasks = load_tasks()
    report = run_evals(tasks, lambda t: make_offline_model(t.prompt, tools), tools)
    assert report.passed == report.total
    assert report.total >= 3


def test_check_task_fails_on_unverified():
    task = Task(name="x", prompt="p", expect_substring=None, expect_verified=True)
    passed, reason = check_task(task, Result(answer="x", verified=False))
    assert not passed and "verified" in reason
