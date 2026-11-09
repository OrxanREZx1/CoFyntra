# CoFyntra backend API handoff — Step 4

This is the proposed frontend contract for private chat and shared knowledge. The AI package contract in
[`COFYTRA_AI_BACKEND_CONTRACT_V1.md`](./COFYTRA_AI_BACKEND_CONTRACT_V1.md) remains separate. These routes do
not change the AI contract or implement catch-up, conflict detection, or proposal approval.

All routes require an authenticated user with active membership in the project. Project IDs are UUIDs.
List routes use stable ascending `(created_at, id)` ordering and accept `page` (default `1`) and
`page_size` (default `20`, maximum `100`). List responses have the shape:

```json
{
  "results": [],
  "page": 1,
  "page_size": 20,
  "has_next": false
}
```

## Send and list private messages

`POST /api/projects/{project_id}/messages/`

```json
{
  "content": "We interviewed eight clinic managers.",
  "client_message_id": "web-01J9Y4"
}
```

`client_message_id` is required, unique per founder and project, and must match
`^[A-Za-z0-9][A-Za-z0-9_.:-]*$` (maximum 128 characters). Repeating the same key and content returns
the existing saved reply, if any. Reusing the key with different content returns `409
idempotency_key_reused`. Successful first requests return `201`; retries return `200`.

Successful response:

```json
{
  "founder_message_id": "60b2a84a-...",
  "assistant_message_id": "59248bb0-...",
  "message": {
    "id": "60b2a84a-...",
    "role": "founder",
    "content": "We interviewed eight clinic managers.",
    "created_at": "2026-11-09T04:10:00Z",
    "ai_response": null,
    "extraction_status": "pending"
  },
  "reply": {
    "id": "59248bb0-...",
    "role": "assistant",
    "content": "Let's compare the interview findings.",
    "created_at": "2026-11-09T04:10:01Z",
    "ai_response": {
      "reply": "Let's compare the interview findings.",
      "referenced_knowledge_ids": [],
      "referenced_decision_ids": [],
      "meta": {"request_id": "chat-...", "prompt_version": "v1"}
    },
    "extraction_status": null
  },
  "request_id": "chat-...",
  "response_meta": {"request_id": "chat-...", "prompt_version": "v1"},
  "chat_status": "succeeded",
  "extraction": {"job_id": 1, "status": "pending"}
}
```

The assistant response is stored in the founder's private thread. Chat failures keep the founder message
and return `chat_status: "failed"`, a safe error code, and retryability; retry with the same idempotency
key. `GET /api/projects/{project_id}/messages/` returns only the authenticated founder's messages and
includes the extraction status for founder messages.

## List shared knowledge

`GET /api/projects/{project_id}/knowledge/`

```json
{
  "results": [
    {
      "id": "7ef96e...",
      "author": {"id": "12", "display_name": "Ana Founder"},
      "category": "problem",
      "statement": "Clinics lose time to manual scheduling.",
      "epistemic_status": "founder_observation",
      "source_quote": "Scheduling takes hours.",
      "external_reference": null,
      "shared_at": "2026-11-09T04:10:00Z"
    }
  ],
  "page": 1,
  "page_size": 20,
  "has_next": false
}
```

The endpoint intentionally omits private message IDs and raw chat history. Conflicting knowledge items
are returned independently.

## Extraction worker and sharing gate

Founder messages create one durable `ExtractionJob` each. Run the PostgreSQL worker with:

```powershell
docker compose -f backend/compose.yaml run --rm backend python manage.py migrate --noinput
docker compose -f backend/compose.yaml run --rm backend python manage.py process_extraction_jobs --once
```

For continuous polling, omit `--once`. The worker uses PostgreSQL `SELECT ... FOR UPDATE SKIP LOCKED`,
records attempt counts and leases, and recovers stale jobs. It saves validated extraction output before
publication so a crashed worker can resume without repeating inference.

**Automatic publication is not operational yet.** The AI team's sensitive-content guard has not been
delivered. Until a real guard is integrated and verified, jobs with candidates finish extraction in
`awaiting_safety_validation`; no candidate or source quote is published. Empty extractions complete
without publication. The existing fail-closed service is not bypassed by configuration or a mock guard.
