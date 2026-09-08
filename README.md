# Startary AI System

**Milestone 0 — Local Development Foundation**

> **Core rule**: The LLM is not memory. LangGraph is orchestration. Durable persistence will belong to the database (Milestone 2+).

---

## What Milestone 0 Establishes

| Capability | Status |
|---|---|
| Canonical startup taxonomy (10 areas) | ✅ |
| Core domain schemas (Pydantic v2) | ✅ |
| Assumption as first-class object | ✅ |
| Experiment explicitly references an assumption | ✅ |
| Evidence as first-class object | ✅ |
| Central LangGraph state contract | ✅ |
| Minimal deterministic LangGraph graph | ✅ |
| Configuration (no required env vars) | ✅ |
| Automated test suite | ✅ |
| Terminal development entry point | ✅ |

---

## Architecture Overview

```
startary/
├── config.py               # Pydantic-settings; no required keys in M0
├── domain/
│   ├── taxonomy.py         # StartupArea — canonical single definition
│   ├── enums.py            # WorkflowPhase, AssumptionStatus, ExperimentStatus, …
│   ├── project.py          # StartupProject
│   ├── state.py            # StartupState (area-keyed dict)
│   ├── assessment.py       # StartupAreaAssessment
│   ├── diagnosis.py        # Diagnosis
│   ├── assumption.py       # Assumption (first-class)
│   ├── experiment.py       # Experiment (always references assumption)
│   ├── evidence.py         # Evidence (founder vs. externally retrieved)
│   └── trace.py            # AITrace (for future debuggability)
├── graph/
│   ├── graph_state.py      # StartaryGraphState (TypedDict for LangGraph)
│   ├── nodes.py            # initialize_state, verify_state (deterministic)
│   └── builder.py          # build_graph() → compiled LangGraph
└── cli.py                  # Terminal development entry point
tests/
├── test_taxonomy.py
├── test_domain.py
├── test_graph_state.py
├── test_graph.py
└── test_config.py
```

### Two distinct state layers

| Layer | Type | Purpose |
|---|---|---|
| **Domain objects** | Pydantic `BaseModel` | Validated, immutable domain records |
| **Graph state** | `TypedDict` | Compact orchestration context for LangGraph nodes |

These layers are kept separate intentionally. Domain objects are the source of truth; graph state carries only what is needed for the current reasoning step.

---

## Prerequisites

- Python 3.11+
- pip

No API keys. No database. No Docker.

---

## Installation

```bash
# From the project root:
pip install -e ".[dev]"
```

This installs the `startary` package in editable mode along with `langgraph`, `pydantic`, `pydantic-settings`, `python-dotenv`, `pytest`, and `pytest-asyncio`.

**Optional**: copy the environment template (no values are required yet):

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

---

## Running the Terminal Prototype

```bash
python -m startary.cli
# or, after pip install:
startary-dev
```

Expected output (abbreviated):

```
╔══════════════════════════════════════════════════════╗
║        STARTARY AI SYSTEM — DEVELOPMENT MODE         ║
║                   Milestone 0                        ║
╚══════════════════════════════════════════════════════╝

Enter your startup name (or press Enter for 'My Startup'):
> Startary

Project created.
  Project ID : 3fa85f64-…
  Phase      : UNDERSTAND
  Iteration  : 1

Enter a brief description of your startup
…
Running LangGraph skeleton…

──────────────────────────────────────────────────────
Graph executed successfully.

  Project ID : 3fa85f64-…
  Title      : Startary
  Phase      : UNDERSTAND
  Iteration  : 1
  Areas set  : 1 / 10

  Startup context captured:
    [Problem] AI learning system for first-time founders…
──────────────────────────────────────────────────────

Milestone 0 foundation is operational.
No AI API key was used or required.
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite verifies architecture-critical contracts, not trivial implementation details:

- Taxonomy: 10 canonical areas, no duplicates, StrEnum behaviour
- Domain: all schemas reject invalid data (bad status, bad area, missing assumption reference, scale violations)
- Graph state: valid state passes `verify_state`; missing fields produce descriptive errors
- Graph: compiles and runs without an API key; output conforms to state contract
- Config: loads with zero environment variables

---

## Current Architecture Boundaries

| Component | Milestone 0 |
|---|---|
| `StartupArea` taxonomy | ✅ Defined centrally |
| Domain schemas | ✅ All 8 schemas defined and validated |
| `StartaryGraphState` | ✅ TypedDict contract established |
| LangGraph skeleton | ✅ `initialize_state → verify_state → END` |
| Persistence | In-memory only |
| AI / LLM calls | None |
| Diagnosis logic | None |
| Experiment generation | None |
| Evidence analysis | None |

---

## What Is NOT Implemented (by design)

- Startup diagnosis or scoring
- Weak-link selection
- Critical-assumption generation
- Experiment design or critic
- Evidence interpretation or re-diagnosis
- Pivot logic or support routing
- RAG, embeddings, or web search
- LLM provider calls of any kind
- Frontend, API, or authentication
- PostgreSQL or any database
- Docker or deployment configuration

These belong to future milestones.

---

## Workflow Phases

| Phase | Meaning |
|---|---|
| `UNDERSTAND` | Founder is describing their startup; system builds initial state |
| `DIAGNOSE` | System evaluates startup areas (Milestone 1+) |
| `TEST` | An assumption is being tested via an experiment (Milestone 2+) |
| `WAITING_FOR_EVIDENCE` | Experiment is live; system waits for real-world data (Milestone 3+) |
| `LEARN` | Evidence returned; system re-evaluates and updates state (Milestone 3+) |

---

## Notes for Milestone 1

Before starting Milestone 1:

1. LangGraph node for **diagnosis** should be added in `startary/graph/nodes.py` and wired in `builder.py` after `verify_state`.
2. The `OPENAI_API_KEY` variable in `.env` will become required.
3. The `Diagnosis`, `StartupAreaAssessment`, and `weak_area` fields in `StartaryGraphState` are already prepared.
4. Domain schemas are frozen — avoid changing field names in `Assumption`, `Experiment`, or `Evidence` as they are referenced by tests.

