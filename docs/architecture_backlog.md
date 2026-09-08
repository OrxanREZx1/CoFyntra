# Startary Architecture Backlog

This document tracks identified architectural limitations, structural trade-offs, and research candidates for future system milestones.

---

## WL-001 — Weak-Link Risk Model Saturation

- **Status**: OPEN / DEFERRED
- **Component**: `startary.graph.nodes.select_weak_link` / Area Risk Assessment
- **Date Logged**: 2026-09-01 (Milestone 2.5.1)

### 1. Current Production Behavior
Decision Risk for each of the 10 canonical startup areas is calculated deterministically via the multiplicative formula:

$$\text{Decision Risk} = \text{Importance} \times \text{Uncertainty} \times \text{Evidence Gap} \times \text{Consequence of Being Wrong}$$

Where each factor is scored qualitatively as $\text{LOW}=1, \text{MEDIUM}=2, \text{HIGH}=3$, yielding a score range of $1 \dots 81$.
When multiple areas tie, a deterministic tie-breaking hierarchy is applied:
1. Higher Consequence of Being Wrong
2. Higher Uncertainty
3. Higher Evidence Gap
4. Higher Importance
5. Canonical taxonomy index (Problem $\to$ Target User $\to \dots \to$ Founder Constraints)

### 2. Observed Issue: Score Saturation
In early-stage startup scenarios where the founder has provided initial problem/user context, multiple strategically different areas frequently receive `HIGH` across all 4 dimensions.

**Live Example (Mynera Live Test)**:
Six distinct startup areas simultaneously scored 81/81:
- Problem: 81
- Target User: 81
- Pain / Urgency: 81
- Existing Alternatives: 81
- Value Proposition: 81
- Demand Signal: 81

Because all 6 areas tied across score and all 4 factor levels, the final selection fell back entirely to the canonical taxonomy tie-breaker (Problem). While Problem was a reasonable choice, selection occurred through arbitrary tie-breaking rather than epistemic differentiation.

---

### 3. Future Research Dimensions

The following dimensions are research candidates for future milestones. **They are NOT yet implemented.**

#### A. Dependency & Prerequisite Structure
Startup areas exhibit natural epistemological dependencies:
```
Problem
   ↓
Target User / Pain
   ↓
Value Proposition
   ↓
Demand Signal
   ↓
Acquisition / Revenue
```
If **Problem** is fundamentally uncertain, downstream uncertainties (e.g. Value Proposition, Demand) derive directly from that upstream uncertainty. Resolving an upstream uncertainty may render downstream assumptions obsolete or entirely redefined. Future iterations should explore DAG-based dependency weighting.

#### B. Decision Leverage
*Core Question*: *"If we reduce uncertainty in area $X$, how many important downstream startup decisions become clearer or invalid?"*
An area with high Decision Leverage unlocks multiple downstream choices, whereas an isolated area (e.g. Legal/Compliance) may have high uncertainty but low immediate leverage on product direction.

#### C. Sequencing & Reversibility
- **Two-Way Door Decisions**: Easily reversible pivots (e.g. adjusting a landing page headline).
- **One-Way Door Decisions**: Irreversible commitments (e.g. building extensive custom infrastructure or signing multi-year enterprise contracts).
Areas governing irreversible commitments should carry higher priority.

---

### 4. Telemetry & Evidence Collection Strategy
Before altering the production scoring algorithm, Startary instruments lightweight development telemetry in `select_weak_link`:
- `max_score`: Highest calculated decision risk in the run ($1 \dots 81$).
- `max_score_tie_count`: Number of areas tied for `max_score`.
- `saturated_area_count`: Number of areas receiving maximum score ($81$).
- `tie_breaker_used`: Boolean indicating whether tie-breaking was necessary.

Across future live runs, we will evaluate:
1. Frequency of score saturation across different startup stages (raw idea, problem exploration, MVP planning).
2. Whether expert evaluation agrees with the tie-broken winner versus an upstream dependency candidate.
3. Whether experiments generated from the selected weak link successfully unblock downstream decisions.

*Redesign of the risk model will proceed only after sufficient live observational evidence is gathered.*

---

## SS-001 — Rich Startup State Completeness vs Conversation Friction

- **Status**: OPEN / ACTIVE IN PROGRESS
- **Component**: `startary.domain.execution_context`, `startary.domain.startup_progress`, `startary.graph.nodes.extract_startup_state`, `startary.graph.nodes.plan_information_gap`
- **Date Logged**: 2026-09-03 (Milestone 2.6)

### 1. Problem Statement
Before Milestone 3 generates experiments, Startary requires practical execution and progress context (team size, availability, budget, funding, stage, MVP status, users, revenue, customer access, existing assets) to assess experiment feasibility. However, asking the founder for all missing fields upfront degrades the user experience into a tedious onboarding questionnaire, driving high founder friction and abandonment.

### 2. Current Production Approach
Continuous opportunistic extraction + adaptive information-gap planning:
1. **Startup State Extractor (`extract_startup_state`)**: Passively and continuously extracts operational facts and progress markers from all founder-authored messages with explicit provenance (`EXPLICIT`, `INFERRED`, `UNKNOWN`, `CONFLICTING`) and confidence (`LOW`, `MEDIUM`, `HIGH`).
2. **Deterministic Merge (`merge_execution_state`)**: Merges deltas using strict provenance hierarchy (explicit outranks inferred; newer explicit supersedes older; explicit conflicts flagged without silent overwrites).
### 3. Milestone 2.6.1 Evolution: Structured Baseline Check-In
During live testing in Milestone 2.6, passive extraction alone proved insufficient for early-stage founders, who naturally focus 100% of their initial remarks on their conceptual idea. As a result, Startary exited the `UNDERSTAND` phase with almost all execution and progress fields remaining `UNKNOWN`.

**Decision in Milestone 2.6.1**:
Baseline profile collection is intentionally structured and brief (3–5 grouped questions max) rather than purely opportunistic.
*Reason*: Passive extraction alone leaves important startup-state fields empty when founders naturally focus only on their idea/problem. Grounding the operational baseline first ensures Startary understands the entity it is advising before assessing strategic risk.

### 4. Metrics & Evidence Collection Strategy
Across upcoming live tests and milestones, the following telemetry and empirical data will be tracked:
- **Clarification question count**: Average number of clarification turns required before reaching decision sufficiency.
- **Passive extraction rate**: Percentage of profile and progress fields extracted without direct inquiry.
- **Founder friction & drop-off**: Qualitative founder feedback on whether questioning felt conversational vs questionnaire-like.
- **Experiment pre-requisite accuracy**: Which execution fields are genuinely required before experiment generation in Milestone 3, versus which remain `USEFUL_LATER`.
- **Question necessity**: Number of asked questions that yielded marginal or unused decision value.
- **Stage & status inference accuracy**: Consistency and validity of inferred `DevelopmentStage` and `ProductStatus`.
- **Founder correction frequency**: Rate at which founders explicitly correct previously inferred or stored operational facts.

---

## WL-002 — Weak-Link Selection Confidence & Clarification Effectiveness

- **Status**: ACTIVE / INSTRUMENTED
- **Component**: `startary.graph.nodes.assess_weak_link_discrimination`, `select_weak_link_question`
- **Date Logged**: 2026-09-03 (Milestone 2.6.1)

### 1. Problem Statement
The deterministic scoring formula ($Imp \times Unc \times Gap \times Con$) frequently produces score saturation (multiple 81s) or ties where the winning weak link is picked solely via canonical taxonomy fallback. In Milestone 2.6.1, we introduced the **Weak-Link Discrimination Loop** to evaluate `selection_confidence` (`LOW`, `MEDIUM`, `HIGH`) and ask targeted differentiating questions before Reality Check.

### 3. Milestone 2.6.2 Resolution & Routing Integrity
In Milestone 2.6.2, a routing bug was identified where `route_on_weak_link` ignored `should_clarify=True` due to state key mismatches and routed prematurely to `reality_check`.
- **Resolution**: `should_clarify` was established as authoritative in `StartaryGraphState` and `route_on_weak_link`.
- **Loop Topology**: After the founder answers a discriminating question, the full reasoning loop re-executes:
  `extract_startup_state -> merge_execution_state -> diagnose_startup -> merge_diagnosis -> assess_startup_areas -> select_weak_link -> assess_weak_link_discrimination`.
  This allows the founder's answer to alter area risk scores and recalculate the weak-link winner deterministically.

---

## CA-001 — Critical Assumption Grounding Integrity

- **Status**: ACTIVE / INSTRUMENTED
- **Component**: `startary.graph.nodes.generate_critical_assumption`, `validate_critical_assumption_grounding`, `startary.llm.prompts` (`critical_assumption_v2`)
- **Date Logged**: 2026-09-03 (Milestone 2.6.2)

### 1. Problem Statement
During initial tests, Critical Assumption generation frequently hallucinated specific ungrounded details to make the hypothesis appear more testable:
- Arbitrary numeric thresholds (e.g. *"at least 20 orders per week"*)
- Invented frequencies (e.g. *"at least weekly"*)
- Speculative failure modes (e.g. *"missed, delayed, or incorrectly recorded deliveries"*)
- Arbitrary ranking/comparative claims (e.g. *"among their top three operational frustrations"*)

This violated evidence discipline because Reality Check had explicitly concluded that these specific mechanisms, frequencies, and thresholds were unknown.

### 2. Core Epistemic Policy
- **WHAT MUST BE TRUE vs HOW TO TEST IT**: The Critical Assumption represents the single core belief that must be true for the startup to proceed. Experiment parameters (sample sizes, interview targets, conversion rates, durations) belong in Milestone 3, NOT in the assumption, unless explicitly stated by the founder.
- **Reality Check as Hard Constraint**: Anything listed in Reality Check's `what_evidence_does_not_support` must NEVER reappear as an asserted fact inside the assumption.
- **Deterministic Grounding Validation**: Every assumption must be validated against founder context and Reality Check constraints before being saved to state. If ungrounded numbers, frequencies, or speculative mechanisms are detected, generation is retried once with explicit corrective feedback; if still ungrounded, generation fails safely.

### 3. Instrumented Telemetry
Startary tracks the following metrics:
1. **Validation retry rate**: Percentage of generated assumptions rejected on the first attempt due to ungrounded claims.
2. **Prohibited parameter leakage**: Frequency of attempts introducing ungrounded percentages, order volumes, or frequency claims.
3. **Reality Check constraint violations**: Frequency of attempts asserting unevidenced failure mechanisms.
4. **Founder correction rate**: Frequency of founders disputing assumptions as misrepresenting their actual belief.
5. **Downstream experiment utility**: Tracking whether grounded qualitative assumptions transition cleanly into Milestone 3 experiment designs without losing falsifiability.

---

## EX-001 — Experiment Decision Rule Calibration

- **Status**: ACTIVE / INSTRUMENTED
- **Component**: `startary.graph.nodes.generate_experiment`, `critique_experiment`, `assess_can_execute`, `startary.domain.experiment`
- **Date Logged**: 2026-09-03 (Milestone 3)

### 1. Problem Statement
In Milestone 3, Startary introduces the Experiment Generator and Experiment Critic to design actionable learning tests for active Critical Assumptions. A key design balance is distinguishing heuristic decision rules from factual market claims:
- The generator specifies provisional parameters (e.g. *"interview 6–8 sellers"*, *"at least 5 of 8 describe the problem"*).
- These parameters must have explicit provenance (`AI_PROPOSED`) so they are never conflated with verified customer statistics.
- The Critic evaluates whether these parameters create falsifiable boundaries without introducing fake precision (e.g. *"7.3 out of 10 users"*).

### 2. Instrumented Telemetry & Calibration Tracking
Across upcoming live runs and milestones, Startary instruments the following metrics to calibrate decision rules and critic efficacy:
1. **Experiment type distribution**: Frequency of each `ExperimentType` (e.g. `PROBLEM_INTERVIEW`, `PROTOTYPE_TEST`, `PRICING_TEST`) generated across different weak-link startup areas.
2. **Assumption area alignment**: Tracking whether assumption type correctly drives experiment type (e.g., Problem $\to$ interviews/observation, Demand $\to$ behavioral trials, Pricing $\to$ payment path).
3. **AI-proposed parameter calibration**: Distribution of sample sizes and support thresholds proposed across early-stage vs post-MVP contexts.
4. **Critic revision rate**: Percentage of generated experiments that trigger `REVISE` on the first critique pass.
5. **Critic dimension failure breakdown**: Which specific dimensions fail most frequently (`assumption_alignment`, `evidence_validity`, `feasibility`, `cost_efficiency`, `falsifiability`, `scope`).
6. **Revision improvement rate**: Percentage of revised experiments that successfully pass the second critic evaluation.
7. **Readiness distribution**: Proportion of experiments classified as `CAN_EXECUTE`, `NEEDS_HELP`, and `BLOCKED` in `assess_can_execute`.
8. **Downstream outcome utility**: In future milestones, tracking whether the executed experiments produce clear support/contradiction signals that change the startup's next recommended action versus yielding inconclusive outcomes.


