# Milestone 6.1 — LangSmith AI Latency Observability

## 1. Overview and Purpose

**Milestone 6.1 is an observability milestone, NOT an optimization milestone.**

Startary AI's reasoning engine executes a multi-node LangGraph loop coordinating fast-tier triage, baseline sufficiency assessment, strategic diagnosis, weak link selection, and experiment generation. While functionally robust, end-to-end response times have shown elevated latency during local and testing workflows.

Before optimizing any prompts, model selections, or graph topology, we require **precise, non-intrusive runtime measurements**.

This integration embeds LangSmith observability across the AI-only core (`v0.4.1.4-alpha-core`) on branch `ai-latency-langsmith`, enabling developers to inspect:
- Total active AI execution duration per turn (excluding human think time).
- Individual LangGraph node durations.
- Individual LLM call durations and token metrics.
- Model resolution and tier verification (fast vs. reasoning tiers).
- Token-budget retries, expansion parameters, and failure recovery.
- Experiment design and critique latency.
- Interrupt and resume lifecycle boundaries.

---

## 2. Core Architectural Principles

### 2.1 Opt-in and Zero Overhead by Default
- Tracing is **disabled by default** (`LANGSMITH_TRACING=false`).
- When disabled:
  - No network connections or API calls to LangSmith are made.
  - No background worker threads are spawned.
  - LangGraph Pregel runs natively without tracer callbacks.
  - Zero latency or token overhead is introduced into normal execution.
- Tracing is activated only when explicitly requested via `LANGSMITH_TRACING=true` in `.env` or via CLI flags (`--trace` / `--langsmith`) with a valid `LANGSMITH_API_KEY`.

### 2.2 Privacy and IP Masking by Default
- To protect founder ideas, intellectual property, and proprietary startup context:
  - `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true` are enforced by default at the client and environment level.
  - Metadata sanitization (`sanitize_metadata()`) filters all span tags and metadata dictionaries:
    - Drops keys containing: `prompt`, `message`, `raw_context`, `context`, `founder_name`, `founder_messages`, `startup_summary`, `user_input`, `input`, `secret`, `key`, `password`, `credential`, `auth`.
    - Automatically strips multi-line strings and raw text blocks longer than 120 characters.
  - Only operational and synthetic tags are emitted (e.g. `scenario_id:scen-e7f09a12`, `founder_turn_number:0`, `workflow_phase:UNDERSTAND`, `tier:fast`, `model:gpt-5.6-luna`).

### 2.3 Strict Separation of AI Latency vs Human Waiting Time
- LangGraph interrupts pause execution at `ask_and_wait_for_founder`.
- The CLI synchronously prompts the founder via `read_founder_input("> ")`.
- Founder think time occurs entirely **outside** `graph.invoke()`.
- Each turn generates a dedicated root trace span (`startary_turn_0`, `startary_turn_1`, etc.), measuring exclusively active computation latency.

---

## 3. Implementation Architecture

### 3.1 Settings & Environment (`startary/config.py`)
Configured in `Settings`:
- `langsmith_tracing: bool = False` (opt-in)
- `langsmith_api_key: Optional[str] = None`
- `langsmith_project: str = "startary-ai-latency"`
- `langsmith_hide_inputs: bool = True`
- `langsmith_hide_outputs: bool = True`
- `langsmith_workspace_id: Optional[str] = None`
- `langsmith_endpoint: Optional[str] = None`

### 3.2 Observability Module (`startary/observability/langsmith.py`)
- `is_tracing_enabled(settings)`: Determines if tracing is active.
- `setup_langsmith_environment(settings)`: Exports standardized `LANGSMITH_*` and `LANGCHAIN_*` environment variables, while ensuring tracing is explicitly disabled when unconfigured.
- `sanitize_metadata(metadata)`: Guarantees zero sensitive founder data leakage into trace spans.
- `get_langsmith_tracer(settings)`: Constructs a singleton `LangChainTracer` configured with input/output privacy masks.
- `build_trace_config(...)`: Builds a standard LangGraph `RunnableConfig` containing `thread_id`, `run_name`, sanitized metadata, tags, and the tracer callback (when active).
- `flush_langsmith_traces(timeout)`: Flushes background trace queue before process exit, registered via `atexit` and invoked in CLI `finally` blocks.

### 3.3 LLM Provider Instrumentation (`startary/llm/provider.py`)
`get_llm_client()` attaches standard metadata and tags to every ChatOpenAI client instance:
- `metadata`:
  - `model_tier`: `"fast"` or `"reasoning"`
  - `configured_model`: Configured model name from settings
  - `resolved_model`: Actually resolved model (accounting for fallback)
  - `fallback_used`: Boolean indicating fallback activation
  - `max_completion_tokens`: Completion token budget
- `tags`:
  - `f"tier:{tier}"`
  - `f"model:{resolved_model}"`

### 3.4 Token-Exhaustion Retry Tracking (`startary/graph/nodes.py`)
In `_invoke_structured_with_token_retry`:
- **Attempt 1**:
  - `metadata`: `{"attempt": 1, "is_retry": False, "base_max_tokens": base_max_tokens, "model_tier": tier}`
  - `tags`: `["attempt:1", f"tier:{tier}", f"node:{node_name}"]`
- **Attempt 2 (Retry on LengthFinishReasonError)**:
  - `metadata`: `{"attempt": 2, "is_retry": True, "retry_reason": "token_budget_exhausted", "base_max_tokens": base_max_tokens, "expanded_max_tokens": expanded_tokens, "reasoning_effort": "low"}`
  - `tags`: `["attempt:2", "retry:token_expansion", f"tier:{tier}", f"node:{node_name}"]`

### 3.5 CLI Integration (`startary/cli.py`)
- Inspects CLI flags `--trace` and `--langsmith`.
- Invokes `setup_langsmith_environment(settings)` and prints initialization banner.
- Uses `build_trace_config()` for:
  - Initial context evaluation (`turn_number = 0`, `run_type = "initial_context"`)
  - Founder resume turns (`turn_number += 1`, `run_type = "founder_reply"`)
  - Timeout retries (`run_type = "timeout_retry"`)
- Enforces `finally: flush_langsmith_traces()` to guarantee all spans are delivered prior to process termination.

---

## 4. How to Inspect Runs in LangSmith

When running with tracing enabled, open [https://smith.langchain.com](https://smith.langchain.com) and navigate to the project `startary-ai-latency`.

### 4.1 Trace Hierarchy
```
startary_turn_0 (Chain / Pregel)
├── extract_startup_state:attempt_1 (ChatOpenAI - FAILED LengthFinishReasonError)
├── extract_startup_state:retry_token_expansion (ChatOpenAI - SUCCEEDED, tokens: 2500)
├── assess_baseline_state_sufficiency (ChatOpenAI - fast tier)
├── assess_strategic_context_sufficiency (ChatOpenAI - fast tier)
└── ask_and_wait_for_founder (Interrupt paused)

startary_turn_1 (Chain / Pregel - after founder resume)
├── merge_founder_clarification (Deterministic update)
├── diagnose_startup (ChatOpenAI - reasoning tier)
├── assess_startup_areas (Deterministic / Fast scoring)
├── select_weak_link (Deterministic rule engine)
├── reality_check (ChatOpenAI - reasoning tier)
├── generate_critical_assumption (ChatOpenAI - reasoning tier)
└── generate_experiment (ChatOpenAI - reasoning tier)
```

### 4.2 Key Metrics to Check
1. **Total Duration (Root Spans)**: Shows exact wall-clock milliseconds spent in AI computation for each turn.
2. **Slowest Nodes**: Sort children by Duration to pinpoint whether latency is concentrated in `diagnose_startup`, `reality_check`, or `generate_experiment`.
3. **Model Tier & Name**: Verify that fast nodes ran on `gpt-5.6-luna` (or `gpt-4o-mini`) and strategic nodes ran on `gpt-5.6-sol` (or `gpt-4o`).
4. **Token Counts**: When reported by provider, inspect prompt tokens, completion tokens, and reasoning tokens.
5. **Retries**: Search for tag `attempt:2` to audit any nodes that experienced token exhaustion.

---

## 5. Local Setup and PowerShell Run Instructions

### 5.1 Verification Checklist
Before running, ensure:
1. Working directory: `Startary_AI_Latency`
2. Git branch: `ai-latency-langsmith`
3. Base revision: `v0.4.1.4-alpha-core`

### 5.2 Enabling Tracing in PowerShell
Set environment variables in your current PowerShell session:

```powershell
# Set your LangSmith API Key
$env:LANGSMITH_API_KEY = "lsv2_pt_your_actual_key_here"

# Set Project Name
$env:LANGSMITH_PROJECT = "startary-ai-latency"

# Enable Tracing (opt-in)
$env:LANGSMITH_TRACING = "true"

# Enforce privacy masking (inputs/outputs hidden)
$env:LANGSMITH_HIDE_INPUTS = "true"
$env:LANGSMITH_HIDE_OUTPUTS = "true"
```

Or configure them in `.env`:
```ini
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_your_actual_key_here
LANGSMITH_PROJECT=startary-ai-latency
LANGSMITH_HIDE_INPUTS=true
LANGSMITH_HIDE_OUTPUTS=true
```

### 5.3 Running the Startary CLI
Execute with the `--trace` flag:

```powershell
python -m startary.cli --trace
```

You will see:
```text
[LangSmith] Latency observability ACTIVE (project: 'startary-ai-latency', hide_inputs=True, hide_outputs=True)
```

### 5.4 PitchVision AI Test Input
When prompted for startup context, use the standardized synthetic benchmark:

```text
Startup Name: PitchVision AI
Phase: Idea validation
Context:
We are building PitchVision AI, an automated video analytics and tactical feedback tool for amateur football clubs.
Team: Two founders.
Availability: 15 hours per founder per week.
Budget: $1,500 total capital.
Customer Access: Direct access to 2 local amateur football teams and 3 licensed youth coaches.
Status: Idea stage. We have not built the software yet. Zero paying customers and zero revenue.
Current Problem: Coaches spend 4+ hours every Sunday manually tagging match recordings and struggle to deliver individual player feedback before Tuesday training.
```

