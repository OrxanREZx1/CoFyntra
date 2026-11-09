# Cofytra AI ↔ Django Contract v1

**Status:** draft for freeze. See [Open contract decisions](#open-contract-decisions); every item there needs agreement before v1 is frozen.
**Source of truth:** [`cofytra_ai/models.py`](../cofytra_ai/models.py). Every JSON example below is validated by [`tests/cofytra_ai/test_models.py`](../tests/cofytra_ai/test_models.py), so if this document and the code ever disagree, the tests fail.

## 1. Architecture

```
React / Next.js  →  Django REST Framework  →  cofytra_ai (Python package, same process)  →  startary.llm.provider (OpenAI)
```

- `cofytra_ai` is a plain Python package, imported by Django and called in-process. There is no AI HTTP service, no FastAPI, and no separate port.
- The frontend never calls `cofytra_ai`. Only Django views or services do.
- `cofytra_ai` never imports Django, never touches the database or ORM models, and keeps no state between calls.
- The existing Startary LangGraph workflow (`startary/graph/`, `startary/cli.py`) is neither used nor modified.

| Django owns | `cofytra_ai` owns |
|---|---|
| Authentication, project membership, permissions | Prompting, LLM calls, retries |
| Storing private messages and message history | Validating model output against this contract |
| Knowledge-sharing rules and persistence (assigning IDs) | Deterministic reference-integrity checks |
| Proposals, approval and committed startup state | Nothing persistent |
| Choosing and trimming the context sent in each request | Rejecting requests that break the contract |
| Converting response models into DB writes | |

## 2. Python interface

```python
from cofytra_ai import service, CofytraAIError
from cofytra_ai.models import ChatRequest, ChatResponse  # …and the other request/response models

def chat(request: ChatRequest, *, settings=None, llm_factory=None) -> ChatResponse
def extract_knowledge(request: ExtractRequest, *, settings=None, llm_factory=None) -> ExtractResponse
def catch_up(request: CatchupRequest, *, settings=None, llm_factory=None) -> CatchupResponse
def detect_conflicts(request: ConflictRequest, *, settings=None, llm_factory=None) -> ConflictResponse
def draft_proposal(request: ProposalRequest, *, settings=None, llm_factory=None) -> ProposalResponse
```

- Calls are **synchronous and blocking**: one LLM call per invocation, typically 1–10 s, with a 30 s timeout on the fast tier. See decision D2.
- Django builds the request with `XxxRequest.model_validate(data)`. A `pydantic.ValidationError` at that point is a Django bug, because the context is assembled from trusted data. Convert it with `CofytraAIError.from_validation_error(exc)` if a uniform error is wanted.
- Django serializes the response with `response.model_dump(mode="json")`.
- `settings` and `llm_factory` are for tests and configuration only. Django calls `service.chat(request)` and nothing more.
- Before it returns, every function checks its response with `models.contract_violations(request, response)`. Django may call the same check again as a defence.
- **Current state:** the five functions are contract stubs. They raise `CofytraAIError` with code `not_implemented`.

## 3. Data layers and identity boundaries

Four layers are kept strictly separate. Each layer has its own model, and no model can carry a different layer's data.

| Layer | Model | Who sees it | Has an ID? | Allowed in |
|---|---|---|---|---|
| **Private conversation message** | `PrivateMessage` | Only its own founder | Yes, Django's | `chat`, `extract_knowledge`, and only the **requesting founder's own** messages |
| **Shared knowledge candidate** | `KnowledgeCandidate` | Only the founder who extracted it, until they share it | **No.** Django assigns an ID when the candidate is accepted | `extract_knowledge` output only |
| **Shared persisted knowledge** | `SharedKnowledgeItem` | All project members | Yes, Django's | Input to every capability |
| **Team-approved committed state** | `CommittedState` / `CommittedDecision` | All project members | Yes, Django's | Optional input to `chat`, `catch_up`, `draft_proposal` |

Identity rules, enforced by the models:

- `PrivateMessage` has a `role`. `founder` messages must have `author_id`; `assistant` messages must have `author_id: null`.
- In `chat` and `extract_knowledge`, every founder message must have `author_id == founder.id`. A message from another founder is rejected.
- **Catch-up has no message field at all.** `CatchupRequest` forbids unknown fields, so a `messages`, `recent_messages` or `private_messages` key fails validation. Every knowledge item in it must be authored by `other_founder`, and `viewer` must differ from `other_founder`.
- `SharedKnowledgeItem` has no `source_message_ids`, so shared knowledge never exposes private message IDs. It can carry `source_quote`, which the author approved when they shared the item.
- Proposals are always `status: "draft"`. Nothing in the AI output can approve, commit or change committed state.

## 4. Vocabularies

### Knowledge categories

`problem`, `target_customer`, `market`, `assumption`, `evidence`

### Epistemic status

Cofytra **never verifies claims**, so no status means "verified".

| Status | Meaning |
|---|---|
| `founder_hypothesis` | A belief, guess or assertion by a founder, with no observation reported |
| `founder_observation` | Something a founder reports observing first-hand: interviews, usage data, sales |
| `ai_suggestion` | An idea that came from the AI and that a founder chose to keep. It was never stated by a founder |
| `external_source` | A founder cites an external source (report, article, dataset). Cofytra has not checked it. `external_reference` is required |

Rules (enforced on both `SharedKnowledgeItem` and `KnowledgeCandidate`):

- `assumption` → `founder_hypothesis` or `ai_suggestion` only.
- `evidence` → `founder_observation` or `external_source` only.
- `problem`, `target_customer`, `market` → any status.
- `external_reference` is required with `external_source` and forbidden with every other status.
- An extracted `KnowledgeCandidate` can never be `ai_suggestion`, because extraction reads founder-authored text only.

### Confidence

`low`, `medium`, `high`. This is the AI's confidence that the extraction is faithful. It is **not** a measure of whether the claim is true.

## 5. Common conventions

- **IDs** are opaque strings of 1–128 characters matching `^[A-Za-z0-9][A-Za-z0-9_.:-]*$`. Django supplies every ID. UUIDs and `kn-123`-style IDs both fit.
- **Timestamps** are ISO-8601 with a timezone, for example `2026-11-09T09:00:00Z`. Naive datetimes are rejected.
- **Unknown fields** are rejected everywhere.
- **Text** is stripped of surrounding whitespace, and an empty string is rejected. Use `null` or omit the field to mean "absent", never `""`.
- **Optional fields:** `null` and omission mean the same thing, "not provided / unknown". `committed_state: null` means no state has been committed yet.
- **Lists** default to `[]`. An empty list means "nothing of this kind exists in the context Django chose to send".
- **Ordering:** private messages are ordered oldest to newest (`created_at` non-decreasing).
- **`meta`** is on every response:
  - `request_id` echoes the request.
  - `contract_version` is `"v1"`.
  - `prompt_version` is set on every response.
  - `model`, `tier`, `latency_ms`, `attempts` describe the LLM call.
  - `tier: "none"` with `model: null` and `attempts: 0` means the answer was produced deterministically, with no LLM call (for example, an empty context).

### Context limits

Django trims context, most recent first. **A request over any limit fails validation; nothing is truncated silently.**

| Limit | Value |
|---|---|
| Private messages per request (`recent_messages`, `messages`) | 20 |
| Characters per message / chat `message` | 4,000 |
| Knowledge items per request | 60 |
| Characters per knowledge statement | 500 |
| Characters per `source_quote` | 1,000 |
| Committed decisions in `committed_state` | 30 |
| Pending proposals | 20 |
| Proposal `topic` | 300 characters |
| ID length | 128 |

Output limits: reply 6,000 characters; summary 2,000; 20 candidates; 10 points per catch-up bucket; 10 conflicts; 5 trade-offs; 10 warnings.

### Empty context behaviour

| Capability | Empty input | Behaviour |
|---|---|---|
| `chat` | no history, no knowledge, no committed state | Valid. Answers from the message alone |
| `extract_knowledge` | `messages: []` | **Invalid** (at least 1 message, at least 1 founder message). An empty `candidates` list is a valid result |
| `catch_up` | `knowledge: []` and `pending_proposals: []` | Valid. Answered deterministically (`tier: "none"`), see example 7 |
| `detect_conflicts` | fewer than 2 distinct authors | Valid. Returns `conflicts: []` without an LLM call |
| `draft_proposal` | `knowledge: []` | Valid. The draft carries a `no_supporting_knowledge` warning |

## 6. Capabilities

### 6.1 `chat`: context-aware founder assistant

The input is the founder's own recent private messages, the shared knowledge, and the committed state. Django stores both the founder's message and the returned `reply` as private messages.

<!-- example: chat.request -->
```json
{
  "request_id": "req-chat-001",
  "project_id": "proj-001",
  "founder": {"id": "u-ana", "display_name": "Ana"},
  "message": "Should we focus on independent clinics or diagnostic labs first?",
  "recent_messages": [
    {"id": "msg-101", "role": "founder", "author_id": "u-ana", "text": "I talked to 8 clinic managers this week. 5 said scheduling eats about 3 hours a day.", "created_at": "2026-11-09T09:00:00Z"},
    {"id": "msg-102", "role": "assistant", "author_id": null, "text": "That is a strong signal. Did any of them mention what they use today?", "created_at": "2026-11-09T09:00:05Z"},
    {"id": "msg-103", "role": "founder", "author_id": "u-ana", "text": "Most use paper calendars. I think they would pay 50 EUR a month for something better.", "created_at": "2026-11-09T09:02:00Z"}
  ],
  "shared_knowledge": [
    {"id": "kn-201", "author_id": "u-ben", "category": "target_customer", "statement": "Diagnostic labs are the best first customer because they handle the highest booking volume.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-08T15:00:00Z"},
    {"id": "kn-202", "author_id": "u-ben", "category": "market", "statement": "There are roughly 12,000 diagnostic labs in the EU.", "epistemic_status": "external_source", "source_quote": "An industry report says there are about 12,000 diagnostic labs in the EU.", "external_reference": "EU Diagnostics Industry Report 2025", "shared_at": "2026-11-08T15:05:00Z"}
  ],
  "committed_state": {
    "version": 2,
    "problem": "Small healthcare providers lose hours each day to manual appointment scheduling.",
    "target_customer": null,
    "market": null,
    "decisions": [
      {"id": "dec-301", "title": "Start B2B, not B2C", "summary": "We sell to healthcare providers, not to patients.", "committed_at": "2026-11-07T18:00:00Z", "referenced_knowledge_ids": []}
    ]
  }
}
```

<!-- example: chat.response -->
```json
{
  "reply": "You have first-hand signal from clinics (5 of 8 managers report about 3 hours a day lost), while Ben's case for labs is still a hypothesis backed by a market-size figure. Since the team already committed to B2B, a fair next step is to give labs the same test: five lab manager interviews this week, then compare.",
  "referenced_knowledge_ids": ["kn-201", "kn-202"],
  "referenced_decision_ids": ["dec-301"],
  "meta": {"request_id": "req-chat-001", "contract_version": "v1", "prompt_version": "cofytra.chat.v1", "model": "gpt-5.6-luna", "tier": "fast", "latency_ms": 1840, "attempts": 1}
}
```

Validation:
- Founder messages belong to `founder.id`.
- Message IDs and knowledge IDs are unique.
- Every referenced ID exists in `shared_knowledge` or `committed_state.decisions`.

Django:
- Persist the message and the reply as private messages.
- Optionally call `extract_knowledge` afterwards.

### 6.2 `extract_knowledge`: shared knowledge candidates

This extracts candidates from the founder's own messages. **The call shares nothing.** Django shows the candidates to the founder, and only the ones they accept become `SharedKnowledgeItem`s, with Django-assigned IDs.

<!-- example: extract_knowledge.request -->
```json
{
  "request_id": "req-extract-001",
  "project_id": "proj-001",
  "founder": {"id": "u-ana", "display_name": "Ana"},
  "messages": [
    {"id": "msg-101", "role": "founder", "author_id": "u-ana", "text": "I talked to 8 clinic managers this week. 5 said scheduling eats about 3 hours a day.", "created_at": "2026-11-09T09:00:00Z"},
    {"id": "msg-102", "role": "assistant", "author_id": null, "text": "That is a strong signal. Did any of them mention what they use today?", "created_at": "2026-11-09T09:00:05Z"},
    {"id": "msg-103", "role": "founder", "author_id": "u-ana", "text": "Most use paper calendars. I think they would pay 50 EUR a month for something better.", "created_at": "2026-11-09T09:02:00Z"}
  ],
  "existing_knowledge": [
    {"id": "kn-201", "author_id": "u-ben", "category": "target_customer", "statement": "Diagnostic labs are the best first customer because they handle the highest booking volume.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-08T15:00:00Z"}
  ]
}
```

<!-- example: extract_knowledge.response -->
```json
{
  "candidates": [
    {"category": "evidence", "statement": "5 of 8 interviewed clinic managers report scheduling takes about 3 hours a day.", "epistemic_status": "founder_observation", "confidence": "high", "source_message_ids": ["msg-101"], "source_quote": "5 said scheduling eats about 3 hours a day", "external_reference": null, "duplicate_of_knowledge_id": null},
    {"category": "problem", "statement": "Most interviewed clinics manage appointments with paper calendars.", "epistemic_status": "founder_observation", "confidence": "medium", "source_message_ids": ["msg-103"], "source_quote": "Most use paper calendars.", "external_reference": null, "duplicate_of_knowledge_id": null},
    {"category": "assumption", "statement": "Clinics would pay 50 EUR per month for a better scheduling tool.", "epistemic_status": "founder_hypothesis", "confidence": "high", "source_message_ids": ["msg-103"], "source_quote": "I think they would pay 50 EUR a month for something better", "external_reference": null, "duplicate_of_knowledge_id": null}
  ],
  "dropped": {"unknown_source": 0, "quote_not_found": 1, "invalid_status": 0},
  "meta": {"request_id": "req-extract-001", "contract_version": "v1", "prompt_version": "cofytra.extract.v1", "model": "gpt-5.6-luna", "tier": "fast", "latency_ms": 2310, "attempts": 1}
}
```

Validation:
- `messages` holds 1–20 entries with at least one founder message, and all founder messages belong to `founder.id`.
- Candidate rules:
  - `source_message_ids` must reference **founder-role** messages in this request. Assistant text is never source material.
  - `source_quote` must appear **verbatim**, ignoring whitespace and case, in at least one cited message.
  - `external_reference` must also appear in a cited message.
  - `duplicate_of_knowledge_id` must be an ID from `existing_knowledge`.
  - The status and category rules from §4 apply.
- The service discards any model-proposed candidate that breaks these rules and counts it in `dropped`; it never repairs a candidate by inventing anything. In the example, one candidate was dropped because its quote was not in the source.

Django:
- Present the candidates for confirmation and persist the accepted ones.
- A `duplicate_of_knowledge_id` is a hint: merge or skip.
- Decide whether to keep `source_message_ids` internally (see D4).

### 6.3 `catch_up`: co-founder catch-up

This summarizes the **other founder's shared knowledge** and the pending proposals. It never receives private messages.

Buckets are assigned by category:
- `discoveries` ← `problem`, `target_customer`, `market`
- `assumptions` ← `assumption`
- `evidence` ← `evidence`

A point may only cite items whose category belongs to its bucket. Point text must keep the item's epistemic status explicit, for example "Ben believes…" for a hypothesis.

<!-- example: catch_up.request -->
```json
{
  "request_id": "req-catchup-001",
  "project_id": "proj-001",
  "viewer": {"id": "u-ana", "display_name": "Ana"},
  "other_founder": {"id": "u-ben", "display_name": "Ben"},
  "since": "2026-11-08T00:00:00Z",
  "knowledge": [
    {"id": "kn-201", "author_id": "u-ben", "category": "target_customer", "statement": "Diagnostic labs are the best first customer because they handle the highest booking volume.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-08T15:00:00Z"},
    {"id": "kn-202", "author_id": "u-ben", "category": "market", "statement": "There are roughly 12,000 diagnostic labs in the EU.", "epistemic_status": "external_source", "source_quote": "An industry report says there are about 12,000 diagnostic labs in the EU.", "external_reference": "EU Diagnostics Industry Report 2025", "shared_at": "2026-11-08T15:05:00Z"},
    {"id": "kn-203", "author_id": "u-ben", "category": "assumption", "statement": "Labs will connect our tool to their existing lab information system within a week.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-08T15:10:00Z"},
    {"id": "kn-204", "author_id": "u-ben", "category": "evidence", "statement": "Two lab managers replied to a cold email asking for a demo.", "epistemic_status": "founder_observation", "source_quote": "Two lab managers replied to my cold email asking for a demo.", "external_reference": null, "shared_at": "2026-11-08T15:20:00Z"}
  ],
  "pending_proposals": [
    {"id": "prop-401", "title": "Target diagnostic labs as the first customer", "author_id": "u-ben", "created_at": "2026-11-08T16:00:00Z", "referenced_knowledge_ids": ["kn-201", "kn-204"]}
  ],
  "committed_state": null
}
```

<!-- example: catch_up.response -->
```json
{
  "summary": "Ben has been exploring diagnostic labs as the first customer. He has early interest from two labs and a market-size figure from an industry report; his case that labs are the best first customer, and that integration is quick, are still untested beliefs. He has proposed targeting labs first.",
  "discoveries": [
    {"text": "Ben believes diagnostic labs are the best first customer because of their booking volume (hypothesis).", "knowledge_ids": ["kn-201"]},
    {"text": "An industry report Ben cites puts the EU lab market at about 12,000 labs (external source, not checked).", "knowledge_ids": ["kn-202"]}
  ],
  "assumptions": [
    {"text": "Ben assumes labs can integrate the tool with their existing system within a week.", "knowledge_ids": ["kn-203"]}
  ],
  "evidence": [
    {"text": "Ben reports two lab managers asked for a demo after a cold email.", "knowledge_ids": ["kn-204"]}
  ],
  "pending_decisions": [
    {"proposal_id": "prop-401", "text": "Ben proposes targeting diagnostic labs first; it awaits team approval."}
  ],
  "meta": {"request_id": "req-catchup-001", "contract_version": "v1", "prompt_version": "cofytra.catchup.v1", "model": "gpt-5.6-luna", "tier": "fast", "latency_ms": 2050, "attempts": 1}
}
```

**Empty context:** with no new knowledge and no pending proposals, the service answers without an LLM call.

<!-- example: catch_up_empty.request -->
```json
{
  "request_id": "req-catchup-002",
  "project_id": "proj-001",
  "viewer": {"id": "u-ana"},
  "other_founder": {"id": "u-ben", "display_name": "Ben"},
  "since": "2026-11-09T00:00:00Z"
}
```

<!-- example: catch_up_empty.response -->
```json
{
  "summary": "Ben has not shared new knowledge or proposals since your last catch-up.",
  "meta": {"request_id": "req-catchup-002", "contract_version": "v1", "prompt_version": "cofytra.catchup.v1", "model": null, "tier": "none", "latency_ms": 0, "attempts": 0}
}
```

Validation:
- There is **no field for private messages**, and unknown keys are rejected.
- `viewer` and `other_founder` must be different founders.
- Every knowledge item must be authored by `other_founder`, and if `since` is set, every item must have `shared_at >= since`.
- Every cited ID must exist in the request, and each must be cited in its category's bucket.
- `pending_decisions[].proposal_id` must be one of the `pending_proposals`.

Django:
- Select the other founder's shared items since the viewer's last catch-up.
- Track "last caught up at" per viewer.

### 6.4 `detect_conflicts`: conflict detection

This finds pairs of claims **from different founders** that contradict (`contradiction`: both cannot be true) or pull the strategy in different directions (`tension`). Both claims are returned with their exact ID, author and statement.

<!-- example: detect_conflicts.request -->
```json
{
  "request_id": "req-conflicts-001",
  "project_id": "proj-001",
  "knowledge": [
    {"id": "kn-201", "author_id": "u-ben", "category": "target_customer", "statement": "Diagnostic labs are the best first customer because they handle the highest booking volume.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-08T15:00:00Z"},
    {"id": "kn-203", "author_id": "u-ben", "category": "assumption", "statement": "Labs will connect our tool to their existing lab information system within a week.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-08T15:10:00Z"},
    {"id": "kn-205", "author_id": "u-ana", "category": "target_customer", "statement": "Independent clinics are the best first customer because their managers report a daily scheduling burden.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-09T09:30:00Z"},
    {"id": "kn-206", "author_id": "u-ana", "category": "assumption", "statement": "Healthcare providers will not connect a scheduling tool to their existing systems in the first year.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-09T09:35:00Z"}
  ]
}
```

<!-- example: detect_conflicts.response -->
```json
{
  "conflicts": [
    {
      "claim_a": {"knowledge_id": "kn-201", "author_id": "u-ben", "statement": "Diagnostic labs are the best first customer because they handle the highest booking volume."},
      "claim_b": {"knowledge_id": "kn-205", "author_id": "u-ana", "statement": "Independent clinics are the best first customer because their managers report a daily scheduling burden."},
      "kind": "contradiction",
      "explanation": "Both name a different segment as the single best first customer; the team can only start with one.",
      "question_to_resolve": "Which segment shows stronger, comparable evidence of pain and willingness to pay?"
    },
    {
      "claim_a": {"knowledge_id": "kn-203", "author_id": "u-ben", "statement": "Labs will connect our tool to their existing lab information system within a week."},
      "claim_b": {"knowledge_id": "kn-206", "author_id": "u-ana", "statement": "Healthcare providers will not connect a scheduling tool to their existing systems in the first year."},
      "kind": "tension",
      "explanation": "The founders expect very different integration effort, which changes what the MVP must include.",
      "question_to_resolve": "Does the MVP need an integration, or must it work standalone?"
    }
  ],
  "meta": {"request_id": "req-conflicts-001", "contract_version": "v1", "prompt_version": "cofytra.conflicts.v1", "model": "gpt-5.6-luna", "tier": "fast", "latency_ms": 2780, "attempts": 1}
}
```

Validation:
- The two claims in a conflict must be different items **by different authors**.
- No pair of claims may appear twice, in either order.
- Every `knowledge_id` must exist in the request, with `author_id` and `statement` **exactly** equal to the request item.

Django:
- Send items from at least two founders.
- Decide whether conflicts are persisted or recomputed (see D6).

### 6.5 `draft_proposal`: decision proposal drafting

This drafts a proposal, optionally to resolve a conflict. **The result is always `status: "draft"`.** Django creates the proposal record, runs approval, and updates the committed state.

<!-- example: draft_proposal.request -->
```json
{
  "request_id": "req-proposal-001",
  "project_id": "proj-001",
  "requested_by": {"id": "u-ana", "display_name": "Ana"},
  "topic": "Choose our first target customer segment",
  "knowledge": [
    {"id": "kn-201", "author_id": "u-ben", "category": "target_customer", "statement": "Diagnostic labs are the best first customer because they handle the highest booking volume.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-08T15:00:00Z"},
    {"id": "kn-204", "author_id": "u-ben", "category": "evidence", "statement": "Two lab managers replied to a cold email asking for a demo.", "epistemic_status": "founder_observation", "source_quote": "Two lab managers replied to my cold email asking for a demo.", "external_reference": null, "shared_at": "2026-11-08T15:20:00Z"},
    {"id": "kn-205", "author_id": "u-ana", "category": "target_customer", "statement": "Independent clinics are the best first customer because their managers report a daily scheduling burden.", "epistemic_status": "founder_hypothesis", "source_quote": null, "external_reference": null, "shared_at": "2026-11-09T09:30:00Z"},
    {"id": "kn-207", "author_id": "u-ana", "category": "evidence", "statement": "5 of 8 interviewed clinic managers report scheduling takes about 3 hours a day.", "epistemic_status": "founder_observation", "source_quote": "5 said scheduling eats about 3 hours a day", "external_reference": null, "shared_at": "2026-11-09T09:40:00Z"}
  ],
  "conflict": {"claim_a_id": "kn-201", "claim_b_id": "kn-205"},
  "committed_state": {
    "version": 2,
    "problem": "Small healthcare providers lose hours each day to manual appointment scheduling.",
    "target_customer": null,
    "market": null,
    "decisions": [
      {"id": "dec-301", "title": "Start B2B, not B2C", "summary": "We sell to healthcare providers, not to patients.", "committed_at": "2026-11-07T18:00:00Z", "referenced_knowledge_ids": []}
    ]
  }
}
```

<!-- example: draft_proposal.response -->
```json
{
  "proposal": {
    "status": "draft",
    "title": "Run a one-week clinic vs. lab comparison before choosing the first segment",
    "rationale": "Both segment choices are still hypotheses. Clinics have stronger first-hand pain evidence (5 of 8 managers), while labs show early demo interest (2 replies). A short, equal test gives comparable evidence before committing.",
    "trade_offs": [
      "Delays the segment decision by about one week.",
      "Clinics show stronger pain evidence, labs show stronger buying interest; the test may still be inconclusive."
    ],
    "next_action": "Each founder runs five interviews with the other founder's segment using the same question script.",
    "referenced_knowledge_ids": ["kn-201", "kn-204", "kn-205", "kn-207"],
    "referenced_decision_ids": ["dec-301"]
  },
  "warnings": [
    {"code": "relies_on_hypothesis", "message": "Both segment claims are untested founder hypotheses.", "knowledge_ids": ["kn-201", "kn-205"]},
    {"code": "unresolved_conflict", "message": "This proposal addresses an open conflict between founders.", "knowledge_ids": ["kn-201", "kn-205"]}
  ],
  "meta": {"request_id": "req-proposal-001", "contract_version": "v1", "prompt_version": "cofytra.proposal.v1", "model": "gpt-5.6-luna", "tier": "fast", "latency_ms": 3120, "attempts": 1}
}
```

Validation:
- `conflict` IDs must be two different items, both in `knowledge`.
- `status` must be `"draft"`.
- `trade_offs` holds 1–5 entries.
- Every referenced and warning ID must exist in the request.

Django:
- Create a pending proposal from the draft (the founder may edit it first).
- Run approval, then update `CommittedState` and its `version`.

## 7. Errors

Every failure raises `cofytra_ai.CofytraAIError`. `exc.info` is an `ErrorInfo` model and is safe to log or return, because it never contains founder text or model output. Internally, the service retries once for malformed model output (with a corrective message), once for transient provider errors, and once with an expanded token budget. It **never** returns partial or repaired content.

| `code` | Meaning | `retryable` | Suggested DRF status |
|---|---|---|---|
| `invalid_request` | Request broke the contract. This is a Django-side bug | false | 500 |
| `provider_config` | API key missing/invalid, model not accessible | false | 503 |
| `provider_unavailable` | Timeout, rate limit, 5xx after the internal retry | true | 503 |
| `invalid_model_output` | Output still malformed after the corrective retry | true | 502 |
| `token_budget` | Output budget exhausted after the one expansion | false | 502 |
| `not_implemented` | Capability is still a contract stub | false | 501 |
| `internal` | Unexpected failure | false | 500 |

<!-- example: error.provider_unavailable -->
```json
{"code": "provider_unavailable", "message": "The language model provider did not respond in time.", "retryable": true, "request_id": "req-chat-001", "details": []}
```

<!-- example: error.invalid_request -->
```json
{"code": "invalid_request", "message": "Request failed Cofytra AI contract validation.", "retryable": false, "request_id": "req-catchup-003", "details": [{"field": "messages", "issue": "extra_forbidden"}]}
```

`retryable` is fixed by `code`, and a mismatch fails validation. For `retryable: true`, Django decides whether to retry: retry once at most for user-facing calls, otherwise show "AI unavailable, try again".

## 8. Reference-integrity rules (`contract_violations`)

`models.contract_violations(request, response)` returns a list of violations, and an empty list means the response is valid. The service runs it before returning, so a response that fails it is never returned.

1. `meta.request_id` equals the request's `request_id`.
2. Every ID in a response comes from the same request. No ID is ever generated by the model.
3. Extraction quotes and external references appear verbatim (ignoring whitespace and case) in the cited founder messages.
4. Catch-up points cite items of the right category for their bucket.
5. Conflict claims match their request items exactly (ID, author, statement).

## 9. Backend (Django) responsibilities

- Authenticate the caller and check project membership **before** building any request.
- Put only the caller's own private messages into `chat` and `extract_knowledge`. Never put private messages into `catch_up`, `detect_conflicts` or `draft_proposal`.
- Trim context to the limits in §5, most recent first.
- Treat extraction output as candidates: share only what the founder confirms, and assign IDs.
- Own proposal records, approval, `CommittedState` and its `version`. Never commit anything because the AI said so.
- Map `CofytraAIError` to HTTP responses, and log `exc.info`, never the request body.
- Run `cofytra_ai` in the same Python environment that has the `startary` dependencies (`langchain-openai`, `pydantic>=2`, `pydantic-settings`) and an `OPENAI_API_KEY` (see D7).

## Open contract decisions

These need agreement with the Django developer before v1 is frozen. The current schema shows the proposed default for each.

| # | Decision | Proposed default |
|---|---|---|
| D1 | **ID format.** Django IDs must match `^[A-Za-z0-9][A-Za-z0-9_.:-]*$`, ≤128 chars. Do integer PKs get stringified? | Stringify every PK or UUID (`"42"` is valid) |
| D2 | **Sync vs. async.** Calls block for 1–10 s (30 s timeout on the fast tier). Run them inline in the DRF view, or in a background task (Celery/RQ/thread) with polling? | Inline for the hackathon demo, with a request timeout above 35 s |
| D3 | **Knowledge-sharing flow.** Are extracted candidates auto-shared or confirmed by the founder? Can a founder edit a candidate before sharing, and if so, does its status stay the same? | Founder confirms. Edits keep the category but Django re-checks the status rules via `SharedKnowledgeItem` |
| D4 | **Provenance link.** Does Django keep `source_message_ids` for shared items (internal only)? The contract never sends them to the other founder | Keep them internally, never expose them |
| D5 | **Saving AI suggestions.** Can a founder save an idea from a chat reply as shared knowledge (`ai_suggestion`)? `chat` doesn't currently return candidates | Allowed later; `chat` stays reply-only in v1 |
| D6 | **Conflict lifecycle.** Persist detected conflicts (with a dismiss/resolve state) or recompute on demand? When does Django call `detect_conflicts`: after each share, or on button press? | Button press, recomputed, not persisted |
| D7 | **Runtime and config.** `cofytra_ai` needs the `startary` package importable and reads `OPENAI_API_KEY` / model names from env or a `.env` **relative to the process working directory**. Django must provide these through its settings or env | Django passes a `Settings` object or sets env vars; no reliance on the working directory |
| D8 | **Projects with more than two founders.** Catch-up is per other founder, and conflicts are across any authors. Are teams of 3+ in scope? | Two founders for the hackathon; the contract already handles N |
| D9 | **Committed-state shape.** v1 carries `problem`, `target_customer`, `market` and decisions. Does Django need more fields (stage, business model)? | Keep the three fields for v1, and add optional fields in v1.x |
| D10 | **Verification.** No status means "verified". Does the team want a Django-owned verification flag (for example, an attached interview file) in the future? | Out of scope for v1 |
| D11 | **Catch-up window.** Django tracks "last caught up at" and sends `since`. Is that per viewer, per other founder? | Per (viewer, other_founder) pair |
| D12 | **Invalid-request status code.** `invalid_request` maps to 500, treated as a Django bug, because the context is trusted. Use 400 instead? | 500, logged |
