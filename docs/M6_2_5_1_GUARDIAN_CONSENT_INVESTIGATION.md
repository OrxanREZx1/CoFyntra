# Startary M6.2.5.1: Guardian-Consent Rejection

**Scope:** an offline investigation plus a minimal code correction. No paid calls, no commit or push, and `deploy-alpha` was not touched. **972 offline tests pass.**

## Verdict

- **Confirmed defect, so a false positive is possible.** The check could raise `GUARDIAN_CONSENT_MISSING` for an experiment that involved no minors.
- **Can't confirm which case your run was.** The second draft isn't stored anywhere locally: the CLI checkpointer is in memory, and LangSmith hides outputs. I did not reconstruct it.
- **What your telemetry pins down.** The only violation was `GUARDIAN_CONSENT_MISSING`, with no `CONSENT_STEP_MISSING`. So the draft (1) recorded people, (2) already contained a consent or permission step, and (3) contained no parent or guardian wording. Before this fix, the "involves minors" condition could be met by the experiment or **by any youth word in the founder's conversation**. All three draft shapes below produced exactly your telemetry under the old code:

| Draft shape (illustrative) | Old result | New result |
|---|---|---|
| A. Records interviews with the adult coaches; youth academies appear only in the founder's messages | `GUARDIAN_CONSENT_MISSING` (**false positive**) | Advisory `POSSIBLE_MINORS` (basis: `YOUTH_MENTION`), passed to the critic |
| B. The experiment names "coaches at youth football academies" (a market and adult roles) and records the coach interviews | `GUARDIAN_CONSENT_MISSING` (**false positive**) | Advisory only |
| C. Films U14 academy matches; club permission but no parent/guardian step | `GUARDIAN_CONSENT_MISSING` (genuine) | `GUARDIAN_CONSENT_MISSING` (basis: `EXPERIMENT_TEXT`); unchanged |

## Exact cause in the code (`startary/domain/experiment_contract.py`, before this fix)

1. **`minors_known = _MINORS.search(text) or _MINORS.search(founder_context)`.** `founder_context` is the raw context plus every founder message, so a single mention of "youth" anywhere in the conversation made every recording experiment "involve minors".
2. **`_MINORS` matched bare words:** youth, junior, kids, academy players, and similar. That included markets ("youth football academies") and adult roles ("coaches at youth academies", "coaches of youth teams").
3. **What was recorded didn't matter.** Recording an interview with an adult coach was treated like filming players.
4. **Telemetry recorded only the code**, not what established it, so the live failure couldn't be classified afterwards.

## Correction (minimal; the safeguard is kept)

A guardian/club authorization step is now **required (a violation)** only when the experiment itself involves minors:

| Basis (recorded in telemetry) | Condition |
|---|---|
| `EXPERIMENT_PARTICIPANTS` | The experiment's declared participants (`target_participants_or_scope`, `required_access`) are minors: interviewing, surveying or testing with them, with or without recording |
| `EXPERIMENT_TEXT` | The experiment films athletes, matches or training, **or** collects identifiable data (names, ages, dates of birth, contacts, photos), about named minor players or teams (U14 squad, academy players, players aged 13–15, children, …) |
| `PARTICIPANT_ACCESS` | The experiment films athletes or matches **and** the founder's `customer_access` fact (who they can actually test with) names minor teams or players, unless the experiment states the subjects are adults |

**What is unchanged or only advisory:**
- **Unchanged:**
  - **Recording consent** (`CONSENT_STEP_MISSING`) is still required for anyone recorded, adults included.
  - **The guardian requirement** still applies whenever the experiment records, collects data about, or directly involves minors.
- **Advisory only (`POSSIBLE_MINORS`, sent to the LLM critic):**
  - **Ages unknown:** footage of athletes with no stated ages (basis `UNSTATED_AGES`).
  - **Youth mentioned elsewhere:** youth football appears only in the founder's conversation or as a market (basis `YOUTH_MENTION`). The note tells the critic to confirm everyone recorded is 18 or older, or to require guardian authorization.

**Other pieces of the fix:**
- **Adult roles:** minor phrases that only qualify an adult role ("coaches of youth teams") are not treated as participants.
- **New telemetry field:** `contract_finding_bases`, for example `GUARDIAN_CONSENT_MISSING:EXPERIMENT_TEXT`. It appears in both the local critic decision record and the LangSmith `experiment_critic_decision` span. It contains no free text and passes `sanitize_metadata` unchanged.

**Files:**
- `startary/domain/experiment_contract.py`
- `tests/test_m625_experiment_loop_correctness.py`: two updates. The metadata key list gains the new field. The minors test now establishes minors through the participant-access fact instead of founder chat.
- `tests/test_m6251_guardian_consent.py` (new, 19 tests)
- `tests/test_langsmith_observability.py`: an isolation fix. `test_tracing_disabled_by_default` read your `.env`, where tracing is now enabled for live runs; it now checks the defaults with `_env_file=None`.

## Tests

The new file is `tests/test_m6251_guardian_consent.py` (19 tests).

**Adult-only interviews (6 tests):**
- recorded coach interviews with youth academies as the market, in context or in the experiment text;
- "coaches of youth teams" as participants;
- explicitly adult coaches;
- unrecorded interviews;
- recording consent still required for adults.

**Recording minors (6 tests):**
- filming named U14 players;
- a guardian step satisfies the requirement;
- no consent at all raises both violations;
- the participant-access fact names youth squads;
- an explicit adult subject overrides youth access;
- minors as direct survey participants without recording.

**Identifiable data about minors (2 tests):**
- U14 roster names and dates of birth require guardian authorization;
- adult coach email addresses are advisory only.

**Ambiguous ages (3 tests):**
- footage with unstated ages gives an advisory (`UNSTATED_AGES`);
- footage plus youth-market context gives an advisory (`YOUTH_MENTION`);
- the advisory reaches the critic as a note.

**Graph replays (real graph and MemorySaver) of your measured sequence:** round 1 is an LLM REVISE failing evidence_validity and falsifiability.
- **Round-2 draft records adult coach interviews:** it now reaches the LLM critic and is accepted, and the critic's message carries the age-confirmation note.
- **Round-2 draft films U14 players:** it still fails closed (deterministic REVISE, then failure stop), with basis `GUARDIAN_CONSENT_MISSING:EXPERIMENT_TEXT` in telemetry.

**Mutation check:** with the old minors logic swapped back in, the two false-positive tests fail, and the graph replay ends at `handle_experiment_failure_stop`, as in your live run.

**Full offline suite:** 972 passed, 5 deselected (live), 0 network attempts. That is the 953 tests after M6.2.5 plus the 19 new ones.

## Can a deterministic violation found during regeneration be corrected without a restart? (Investigation only; not implemented)

**Today:** round 2's deterministic REVISE sets `critic_revision_count = 2`. That leads to `handle_experiment_failure_stop`, which pauses for founder learning input. The founder's reply then re-enters `merge_founder_answer → interpret_founder_turn → extract → …`, the full understand pipeline, before a new experiment is generated. That is the expensive restart.

| Option | What it does | Cost | Quality risk |
|---|---|---|---|
| **A. Deterministic remediation for safeguard-class violations** (recommended first) | For `CONSENT_STEP_MISSING` / `GUARDIAN_CONSENT_MISSING` only, append a code-authored founder action: "before recording, obtain permission from everyone recorded; for players under 18, parent/guardian and club authorization; store privately and delete after analysis". Re-run the contract, then send the patched draft to the LLM critic (which has not yet judged draft 2) | 0 extra generations; the critique that a valid draft would have received anyway | The generic step text is code-authored and must not claim legal compliance. It must **not** be used for substantive violations (budget, prototype, required fields, assumption mismatch), because those change the experiment itself |
| B. Targeted LLM repair | One bounded call: draft plus violation messages, "change only what is required" | About one generation call (likely shorter output) | This is the M6.2.6 draft-editing architecture; it needs its own tests and an A/B run |
| C. Separate deterministic-revision budget | Allow one extra full regeneration when only contract violations remain | About one generation plus one critique (from the trace: about 25–40 s plus about 20–25 s; estimate) | Most expensive; the next draft can fail differently |

Option A keeps, and actually enforces, the guardian safeguard instead of removing it. It needs your approval because it changes loop behaviour, and it would land in M6.2.6 alongside option B.

## Next live run

`contract_finding_bases` will show directly whether a guardian violation came from the experiment (`EXPERIMENT_PARTICIPANTS`, `EXPERIMENT_TEXT`, `PARTICIPANT_ACCESS`) or whether youth context only produced an advisory (`POSSIBLE_MINORS:YOUTH_MENTION`). No latency change is claimed.
