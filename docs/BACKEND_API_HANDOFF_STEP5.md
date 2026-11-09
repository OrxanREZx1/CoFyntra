# Co-founder catch-up and conflict detection API

These are authenticated Django REST endpoints backed by the frozen `cofytra_ai` v1 request and
response models. `project_id` is a project UUID. The caller is always taken from authentication;
request bodies must be empty JSON objects (`{}`), and unknown client fields are rejected.

## Catch-up

`POST /api/projects/{project_id}/catch-up/`

The caller must be an active project founder. Django resolves the other active founder and passes only
that founder's published `SharedKnowledge`, pending proposals, and committed state to
`CatchupRequest`. Private messages and internal `KnowledgeSourceMessage` provenance are never read.

Response shape:

```json
{
  "catch_up": {
    "summary": "[mock] Ben shared 1 knowledge items; 0 proposals await team approval.",
    "discoveries": [
      {
        "text": "[mock] Ben shared (founder_hypothesis): Diagnostic labs are the best first customer.",
        "knowledge_ids": ["7ef96e00-0000-4000-8000-000000000001"]
      }
    ],
    "assumptions": [],
    "evidence": [],
    "pending_decisions": [],
    "meta": {
      "request_id": "catchup-...",
      "contract_version": "v1",
      "prompt_version": "cofytra.mock.catchup.v1",
      "model": null,
      "tier": "none",
      "latency_ms": 0,
      "attempts": 0
    }
  },
  "pagination": {
    "has_more": false,
    "batch_size": 1,
    "remaining_at_request": 0
  }
}
```

The response's `catch_up` object is the validated `CatchupResponse`; bucket references are contract IDs
for knowledge supplied to that request. `has_more` indicates that the caller should POST again to process
the next bounded batch. AI failures return the standard safe AI error response and do not advance the
cursor. A caller without active membership receives an access denial; unknown body fields receive
`400`.

### Cursor semantics and backlog

The cursor is scoped to `(project, viewer, other_founder)`. Each successful batch advances a
Django-owned composite position `(last_caught_up_at, last_knowledge_id)`. Items are ordered by
`(shared_at, id)` and the next request selects items strictly after that position. The AI contract
continues to receive `since = last_caught_up_at`; the UUID tie-breaker is internal. Each request carries
at most 60 knowledge items, oldest first. Equal timestamps can span batches without being skipped.
The cursor advances only after the AI adapter validates the response. A failed call leaves it unchanged.

A migrated timestamp-only cursor has a null tie-breaker; for compatibility, equal-timestamp items are
revisited from the start rather than risking omissions. New cursors record both values. Empty catch-ups
do not fabricate updates or advance a cursor. When no cursor exists, the initial catch-up includes all
currently published knowledge from the other founder. Pending proposals and committed-state context
follow the frozen request contract; the cursor tracks shared knowledge, not proposal state.

## Conflict detection

`POST /api/projects/{project_id}/conflicts/detect/`

The caller must be an active project founder. Django supplies up to 60 published project knowledge
items using the existing balanced-by-author context builder. Conflict results are recomputed on every
request and are not persisted.

Response shape:

```json
{
  "conflicts": [
    {
      "claim_a": {
        "knowledge_id": "7ef96e00-0000-4000-8000-000000000001",
        "author_id": "2",
        "statement": "Diagnostic labs are the best first customer.",
        "category": "target_customer",
        "epistemic_status": "founder_hypothesis"
      },
      "claim_b": {
        "knowledge_id": "8af96e00-0000-4000-8000-000000000002",
        "author_id": "1",
        "statement": "Independent clinics show the strongest need.",
        "category": "target_customer",
        "epistemic_status": "founder_observation"
      },
      "kind": "tension",
      "explanation": "[mock] Both founders made different claims about target_customer.",
      "question_to_resolve": "[mock] Which claim has stronger evidence?"
    }
  ],
  "meta": {
    "request_id": "conflicts-...",
    "contract_version": "v1",
    "prompt_version": "cofytra.mock.conflicts.v1",
    "model": null,
    "tier": "none",
    "latency_ms": 0,
    "attempts": 0
  },
  "context": {"knowledge_count": 2, "omitted_knowledge": 0}
}
```

`claim_a` and `claim_b` retain the contract's IDs, founder attribution, and statements; `category` and
`epistemic_status` are resolved from the authorized request context so hypotheses remain visibly distinct
from observations. `question_to_resolve` is the contract's suggested discussion or verification question.
References that do not match request knowledge fail as an AI output error; they are never silently
discarded.

## Mock mode and limitations

With the explicit development configuration `COFYTRA_AI_MODE=mock` (and debug enabled), both routes use
the deterministic contract-valid `MockBackend`. Its metadata has `tier: "none"`, `model: null`, and
`attempts: 0`; no LLM is called. Service mode remains separate and can return `not_implemented` until
the AI team delivers the capability implementation.

Knowledge is sourced only from persisted, published `SharedKnowledge`. Extraction jobs awaiting safety
validation are not part of either request. The sensitive-content guard is still unavailable, so
extraction-to-publication is not operational; these endpoints can be validated with safely seeded,
authorized shared knowledge. No catch-up or conflict result changes committed startup state.
