# agent-loop

A small, transparent **AI agent harness** in Python: a tool-calling loop, working memory, a **self-verification step** that stops the agent declaring success on invented results, and an **eval harness** to score it — all with a **deterministic offline mode** so the whole thing runs in CI with no API key and no network.

![CI](https://github.com/Onipounds/agent-loop/actions/workflows/ci.yml/badge.svg)
&nbsp;Python 3.10–3.12&nbsp;·&nbsp;MIT

## What it is and why I built it

This is a demonstration-quality agent **runtime** built from first principles — the loop, tool orchestration, memory, and verification are implemented directly rather than pulled from a framework, so the mechanics are visible and testable. I built it to understand how agents actually work: how a sentence becomes a sequence of tool calls against real functions, and how you stop a model from confidently reporting a number it never computed.

It is intentionally separate from my company's work — public techniques and a small bundled dataset only.

## Quickstart

```bash
pip install -e .
# Runs with zero setup — no API key needed:
agent-loop run --offline "What was the average temperature on pump-3, and is it within the safe limit of 75 degrees?"
agent-loop eval --offline
```

To use a real model, install the extra and set a key:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-...
agent-loop run "What is the maximum temperature on pump-7?"
```

## How the loop works

```
add the task to memory
repeat up to max_steps:
    ask the model what to do, given the transcript + available tool schemas
    if it requests tools:
        run each tool (errors are captured and fed back, never crash the loop)
        append the results to memory; loop again
    else (it produced a final answer):
        verify the answer (close the loop)
        if verified -> return it
        else -> append the failure reason to memory and let it retry
return best effort if the step budget runs out
```

In real mode this uses the Anthropic Messages API tool-use flow: the model returns `tool_use` blocks, the runtime executes them and returns `tool_result` blocks, and this repeats until the model produces a final answer. In offline mode a deterministic scripted stand-in drives the same loop.

## Tools

Three small, real tools the agent can compose: `query_telemetry` (reads a bundled dataset), `calculate` (safe arithmetic/stats in a locked-down namespace), and `check_threshold`. Adding a tool is one `register(...)` call with a function and a JSON schema; the registry handles execution and error capture.

## Closing the loop: verification

The headline feature, in two layers (cheapest first), run after every candidate answer:

1. **Grounding** — every number in the answer must match a value that actually appeared in a tool result this run (within tolerance). Catches *invented* numbers.
2. **Re-computation** — when the answer states an aggregate (mean / max / min), the verifier independently recomputes it from the raw readings and confirms the agent's figure matches. Catches numbers that look grounded but are *wrong* — e.g. the agent fetched the data but computed the average incorrectly.

If either fails, the reason is fed back and the agent retries. This is what turns "the model said so" into "the model showed its work, and the work checks out."

Both layers are deliberately simple, transparent, and deterministic — no extra model call. They reason about numbers, not meaning; semantic checks could layer on top.

A note on tolerance, because it bit me: grounding uses a *relative* tolerance, but re-computation deliberately uses a tight *absolute* one. An early version reused the relative (1%) tolerance for re-computation — and for readings around 70, a 1% band is ±0.7, wide enough to wave through a wrong mean (it accepted 70.6 when the true mean was 70.88). Re-computation is checking *correctness*, so it compares absolutely. My own end-to-end test caught it.

## Offline / deterministic mode

The model sits behind a small `Model` interface with two implementations: `AnthropicModel` (real) and `FakeModel` (a scripted, deterministic stand-in). Nothing else in the codebase imports the Anthropic SDK. That single seam is why the entire harness — loop, tools, verification, and evals — runs in CI with no key and no network, and why the tests are reproducible. It's the same dependency-isolation pattern I use across my other projects, applied to an LLM.

## Evals

`agent-loop eval --offline` runs the agent over a task set (`tasks/tasks.json`) and reports pass rate and average steps:

```
avg_temp_within_limit        PASS  (4 steps)
max_temp_pump7               PASS  (3 steps)
pressure_mean                PASS  (3 steps)
--------------------------------------------------
3/3 tasks passed, avg 3.3 model calls
```

Each task pairs a prompt with a checker (expected substring and/or "must be verified"). This is how you answer "how do you know it works?" with a number instead of a shrug.

## MCP server

`python -m agent_loop.mcp_server` (after `pip install -e ".[mcp]"`) exposes the tools as an MCP server, so any MCP-capable client can call them. It exposes the tools, not the whole agent — small and composable on purpose.

## Design decisions & trade-offs

- **Hand-built loop, not a framework.** I wanted the mechanics visible and the dependencies near-zero. For a production system with many agent types I'd reach for a framework, but you should understand the loop before you abstract it.
- **`max_steps` guard.** Agents need a hard stop, or a bad plan loops forever and burns cost. The budget is explicit and the run reports how many model calls it used.
- **Tool errors are fed back, not raised.** A tool that fails returns an `{"error": ...}` result the model can see and react to, which is far more robust than crashing the run.
- **Layered verification over trust.** Grounding (cheap, catches invented numbers) plus re-computation (independently recomputes aggregates, catches wrong-but-grounded numbers). Both deterministic. Two bugs worth keeping in mind, both caught by my own tests: an early version read the `7` in `pump-7` as a numeric claim and rejected correct answers (fixed by stripping identifier tokens before extracting numbers); and re-computation first reused a relative tolerance wide enough to accept a wrong mean (fixed with a tight absolute tolerance). Verification logic needs its own tests.
- **Model behind an interface.** Determinism and zero-cost CI, at the price of the offline path not exercising the real model's behaviour — so both paths exist.

## Limitations / what I'd do at scale

This is a single-process, serial harness with in-memory state. To grow it: run tool calls concurrently where they're independent; move memory to a store so runs survive restarts; add rate-limit and retry handling around the model call; track cost-per-task as a first-class metric; and expand the eval set, since that's what tells you whether a change actually helped.

## Testing & CI

~26 tests covering the tools, memory, verification (grounding, re-computation, and the identifier edge case), the loop (multi-step completion, rejection-and-retry, tool-error handling, the step budget), the offline planner, and the eval runner. CI runs the suite and the offline eval on Python 3.10, 3.11, and 3.12.

## License

MIT.
