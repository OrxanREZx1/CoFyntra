"""
Milestone M4.1.3.4 — Domain Isolation, Hardcoded Startup Removal & Recovery Grounding.
Comprehensive verification suite covering:
  - Ecozib live failure reproduction (zero commercial kitchen / refrigerator / food spoilage leakage)
  - 5 synthetic startup domain-mutation tests (Medical, Agritech, FinTech, EdTech, Logistics)
  - Multi-project sequential and concurrent cross-isolation
  - Repository hardcode audit scanner (AST & regex over startary/ runtime codebase)
  - Dynamic RecoveryContext extraction and validate_recovery_grounding behavior
  - Section 6 generic deterministic fallback compliance
  - Full system regression stability across LastBite, ShelfSense, Mynera, and Ecozib
"""

import ast
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from startary.domain.build_readiness import (
    BuildReadinessAction,
    validate_domain_grounding,
    sanitize_domain_leakage,
)
from startary.domain.enums import (
    CriticVerdict,
    ExecutionReadiness,
    ExperimentDesignStatus,
    ExperimentStatus,
    WorkflowPhase,
)
from startary.domain.taxonomy import StartupArea
from startary.domain.experiment import (
    FallbackLearningAction,
    RecoveryContext,
    build_generic_deterministic_fallback,
    build_recovery_context,
    compose_fallback_learning_action,
    validate_recovery_grounding,
    render_fallback_learning_action_text,
)
from startary.graph.nodes import (
    assess_build_readiness,
    handle_experiment_failure_stop,
)


# ===========================================================================
# 1. LIVE PRODUCTION FAILURE REPRODUCTION: ECOZIB
# ===========================================================================

def test_01_ecozib_live_failure_reproduction_has_zero_foreign_leakage():
    """
    1. Ecozib live failure reproduction:
    Active startup: Ecozib (smart waste collection, litter-bin fill-level monitoring, route planning).
    Critic output rejects with decision-maker / operational workflow issues.
    MUST NEVER output commercial kitchens, refrigerator temperatures, food spoilage, or bakeries.
    MUST produce waste collection / route planning grounded action.
    """
    ecozib_problem = (
        "Waste collection organizations experience recurring collection-planning friction "
        "because they lack reliable visibility into which litter bins require collection."
    )
    ecozib_state = {
        "project_id": "proj-ecozib-live",
        "startup_name": "Ecozib",
        "startup_state": {
            "name": "Ecozib",
            "target_user": "Municipal waste collection organizations and route planners",
            "problem": ecozib_problem,
            "value_prop": "Smart bin fill-level monitoring for optimized municipal collection routes",
        },
        "weak_area": "Problem",
        "active_assumption": {
            "id": "asmp-ecozib-01",
            "statement": ecozib_problem,
            "area": "Problem",
        },
        "founder_messages": [
            {"role": "founder", "content": "We are building IoT sensors for public litter bins in Baku to optimize collection routes."}
        ],
        "founder_evidence": [],
    }
    critic_output = {
        "verdict": "REVISE",
        "issues": [
            "The proposed experiment fails to verify whether waste collection decision-makers experience collection-planning friction.",
            "Unverified operational workflow access and decision authority.",
        ],
    }
    rejected_exp = {
        "id": "exp-ecozib-rejected",
        "assumption_id": "asmp-ecozib-01",
        "target_participants_or_scope": "Municipal waste collection dispatchers and fleet managers",
        "hypothesis": ecozib_problem,
        "objective": "Survey municipal dispatchers on collection routes",
    }

    action = compose_fallback_learning_action(
        assumption=ecozib_state["active_assumption"],
        weak_link="Problem",
        critic_output=critic_output,
        rejected_experiment=rejected_exp,
        state=ecozib_state,
    )

    rendered = action.response_text.lower()
    full_text = " ".join([
        action.primary_action,
        action.who_to_engage,
        action.smallest_thing_to_resolve_next,
        " ".join(action.what_to_ask_or_observe),
        " ".join(action.what_to_bring_back),
        " ".join(action.what_not_to_infer),
        rendered,
    ]).lower()

    # Absolute foreign domain invariants
    forbidden_terms = [
        "refrigerator", "fridge", "commercial kitchen", "kitchen operator",
        "kitchen manager", "food spoilage", "temperature monitoring",
        "bakery", "bakeries", "pastry", "surplus bag", "surplus food",
    ]
    for forbidden in forbidden_terms:
        assert forbidden not in full_text, f"Production-blocking failure: foreign term '{forbidden}' leaked into Ecozib recovery!"

    # Grounding in Ecozib's domain
    assert any(w in full_text for w in ["waste", "collection", "bin", "litter", "route", "planning"])
    assert len(action.source_basis) > 0


# ===========================================================================
# 2-6. SYNTHETIC DOMAIN-MUTATION TESTS (5 DIVERSE DOMAINS)
# ===========================================================================

def test_02_synthetic_domain_medical_diagnostics_isolation():
    """2. Medical diagnostics startup produces clinic/radiology grounded recovery with zero bakery/kitchen/waste leakage."""
    problem = "Outpatient imaging clinics experience acute reporting delays because radiologist scan volume exceeds daytime capacity."
    state = {
        "project_id": "proj-radscan",
        "startup_name": "RadScan AI",
        "startup_state": {
            "target_user": "Private imaging clinic medical directors",
            "problem": problem,
        },
        "weak_area": "Problem",
        "active_assumption": {"id": "asmp-rad-1", "statement": problem, "area": "Problem"},
    }
    critic_output = {
        "verdict": "REVISE",
        "issues": ["Clinic medical director decision-maker access not established beyond founder colleague."],
    }
    action = compose_fallback_learning_action(
        assumption=state["active_assumption"],
        weak_link="Problem",
        critic_output=critic_output,
        state=state,
    )
    text = f"{action.primary_action} {action.who_to_engage} {action.response_text}".lower()
    for forbidden in ["bakery", "kitchen", "refrigerator", "waste collection", "litter bin"]:
        assert forbidden not in text
    assert any(k in text for k in ["clinic", "medical", "imaging", "decision-maker"])


def test_03_synthetic_domain_agritech_soil_monitoring_isolation():
    """3. Agritech soil monitoring produces orchard/irrigation grounded recovery with zero foreign domain leakage."""
    problem = "Commercial hazelnut growers face crop yield loss because manual soil moisture probing is too infrequent during dry spells."
    state = {
        "project_id": "proj-agritech",
        "startup_name": "TerraCrop",
        "startup_state": {
            "target_user": "Commercial hazelnut orchard farm managers",
            "problem": problem,
        },
        "weak_area": "Target User",
        "active_assumption": {"id": "asmp-agri-1", "statement": problem, "area": "Target User"},
    }
    critic_output = {
        "verdict": "REVISE",
        "issues": ["Unverified farm manager decision-maker access; testing only family farm."],
    }
    action = compose_fallback_learning_action(
        assumption=state["active_assumption"],
        weak_link="Target User",
        critic_output=critic_output,
        state=state,
    )
    text = f"{action.primary_action} {action.who_to_engage} {action.response_text}".lower()
    for forbidden in ["bakery", "kitchen", "refrigerator", "waste", "cybersecurity"]:
        assert forbidden not in text
    assert any(k in text for k in ["farm", "grower", "orchard", "hazelnut", "decision-maker", "operational manager"])


def test_04_synthetic_domain_fintech_freelance_invoicing_isolation():
    """4. FinTech invoicing startup produces invoice reconciliation grounded recovery with zero foreign domain leakage."""
    problem = "Freelance digital marketers experience cash flow shortfalls because client invoice tracking lacks automated payment reminders."
    state = {
        "project_id": "proj-fintech",
        "startup_name": "InvoiceFlow",
        "startup_state": {
            "target_user": "Freelance digital marketers",
            "problem": problem,
        },
        "weak_area": "Problem",
        "active_assumption": {"id": "asmp-fin-1", "statement": problem, "area": "Problem"},
    }
    critic_output = {
        "verdict": "REVISE",
        "issues": ["Missing denominator: rate of payment delay measured without total issued invoices baseline."],
    }
    action = compose_fallback_learning_action(
        assumption=state["active_assumption"],
        weak_link="Problem",
        critic_output=critic_output,
        state=state,
    )
    text = f"{action.primary_action} {action.who_to_engage} {action.response_text}".lower()
    for forbidden in ["bakery", "kitchen", "refrigerator", "waste collection", "litter bin"]:
        assert forbidden not in text
    assert "denominator" in text or "baseline" in text


def test_05_synthetic_domain_edtech_language_tutoring_isolation():
    """5. EdTech tutoring startup produces lesson confounding grounded recovery with zero foreign leakage."""
    problem = "Expat software engineers struggle with technical presentation fluency in English."
    state = {
        "project_id": "proj-edtech",
        "startup_name": "TechTutor",
        "startup_state": {
            "target_user": "Expat software engineers in Berlin",
            "problem": problem,
        },
        "weak_area": "Solution / MVP",
        "active_assumption": {"id": "asmp-ed-1", "statement": problem, "area": "Solution / MVP"},
    }
    critic_output = {
        "verdict": "REVISE",
        "issues": ["Confounding variables: free coffee voucher bundled with interview obscured primary value proposition."],
    }
    action = compose_fallback_learning_action(
        assumption=state["active_assumption"],
        weak_link="Solution / MVP",
        critic_output=critic_output,
        state=state,
    )
    text = f"{action.primary_action} {action.who_to_engage} {action.response_text}".lower()
    for forbidden in ["bakery", "kitchen", "refrigerator", "waste collection", "litter bin"]:
        assert forbidden not in text
    assert "constant" in text or "isolated" in text or "confound" in text


def test_06_synthetic_domain_logistics_freight_matching_isolation():
    """6. Logistics dispatch startup produces dispatch window grounded recovery with zero foreign leakage."""
    problem = "Independent long-haul freight drivers incur empty return miles due to uncoordinated regional backhaul booking."
    state = {
        "project_id": "proj-logistics",
        "startup_name": "BackhaulDirect",
        "startup_state": {
            "target_user": "Independent long-haul freight drivers",
            "problem": problem,
        },
        "weak_area": "Target User",
        "active_assumption": {
            "id": "asmp-log-1",
            "statement": "There is a definable segment of independent freight drivers near regional logistics terminals available for short-notice backhaul loads.",
            "area": "Target User",
        },
    }
    critic_output = {
        "verdict": "REVISE",
        "issues": [
            "Actual proximity during specific dispatch pickup window was not verified",
            "Participant unavailability during dispatch window confounds booking rate",
        ],
    }
    action = compose_fallback_learning_action(
        assumption=state["active_assumption"],
        weak_link="Target User",
        critic_output=critic_output,
        state=state,
    )
    text = f"{action.primary_action} {action.who_to_engage} {action.response_text}".lower()
    for forbidden in ["bakery", "kitchen", "refrigerator", "waste collection"]:
        assert forbidden not in text
    assert "window" in text or "proximity" in text


# ===========================================================================
# 7-11. MULTI-PROJECT SEQUENTIAL & CONCURRENT CROSS-ISOLATION TESTS
# ===========================================================================

def test_07_sequential_cross_isolation_lastbite_then_ecozib():
    """7. Executing LastBite followed immediately by Ecozib leaves zero LastBite contamination in Ecozib."""
    lastbite_state = {
        "project_id": "proj-lastbite",
        "startup_name": "LastBite",
        "startup_state": {
            "target_user": "Local bakery customers",
            "problem": "Bakery end-of-day surplus bread is discarded.",
        },
        "weak_area": "Target User",
        "active_assumption": {
            "id": "asmp-lb",
            "statement": "There is a definable segment of people near independent Baku bakeries and small cafes with recurring closing-time surplus who collect surplus food.",
            "area": "Target User",
        },
    }
    lb_critic = {"verdict": "REVISE", "issues": ["Actual proximity during each specific offer window was not verified"]}
    lb_action = compose_fallback_learning_action(
        assumption=lastbite_state["active_assumption"],
        weak_link="Target User",
        critic_output=lb_critic,
        state=lastbite_state,
    )
    assert "bakery" in lb_action.response_text.lower() or "cafe" in lb_action.response_text.lower()

    # Now execute Ecozib in the exact same Python process
    ecozib_state = {
        "project_id": "proj-ecozib",
        "startup_name": "Ecozib",
        "startup_state": {
            "target_user": "Municipal waste collection planners",
            "problem": "Waste collection organizations experience recurring collection-planning friction.",
        },
        "weak_area": "Problem",
        "active_assumption": {
            "id": "asmp-eco",
            "statement": "Waste collection organizations experience recurring collection-planning friction.",
            "area": "Problem",
        },
    }
    eco_critic = {"verdict": "REVISE", "issues": ["Decision-maker access unverified for waste fleet operations"]}
    eco_action = compose_fallback_learning_action(
        assumption=ecozib_state["active_assumption"],
        weak_link="Problem",
        critic_output=eco_critic,
        state=ecozib_state,
    )

    eco_text = eco_action.response_text.lower()
    assert "bakery" not in eco_text
    assert "surplus" not in eco_text
    assert "bread" not in eco_text
    assert "cafe" not in eco_text


def test_08_sequential_cross_isolation_ecozib_then_medscan():
    """8. Executing Ecozib followed by MedScan leaves zero waste terms in MedScan."""
    ecozib_state = {
        "project_id": "proj-ecozib",
        "startup_state": {"target_user": "Waste operators", "problem": "Litter bin overflows."},
        "active_assumption": {"id": "a-1", "statement": "Litter bin fill levels lack visibility.", "area": "Problem"},
    }
    compose_fallback_learning_action(assumption=ecozib_state["active_assumption"], state=ecozib_state)

    med_state = {
        "project_id": "proj-med",
        "startup_state": {"target_user": "Hospital radiologists", "problem": "Scan triage delays."},
        "active_assumption": {"id": "a-2", "statement": "Hospital radiologists experience scan backlog.", "area": "Problem"},
    }
    med_action = compose_fallback_learning_action(assumption=med_state["active_assumption"], state=med_state)
    med_text = med_action.response_text.lower()

    assert "bin" not in med_text
    assert "litter" not in med_text
    assert "waste collection" not in med_text
    assert re.search(r"\bwaste\b", med_text) is None


def test_09_sequential_cross_isolation_medscan_then_lastbite():
    """9. Executing MedScan followed by LastBite leaves zero medical terms in LastBite."""
    med_state = {
        "project_id": "proj-med",
        "startup_state": {"target_user": "Hospital radiologists", "problem": "Scan triage delays."},
        "active_assumption": {"id": "a-med", "statement": "Hospital radiologists experience scan backlog.", "area": "Problem"},
    }
    compose_fallback_learning_action(assumption=med_state["active_assumption"], state=med_state)

    lb_state = {
        "project_id": "proj-lb",
        "startup_state": {"target_user": "Bakery customers", "problem": "Surplus bakery food."},
        "active_assumption": {"id": "a-lb", "statement": "Baku consumers near bakeries will collect closing-time surplus.", "area": "Target User"},
    }
    lb_action = compose_fallback_learning_action(assumption=lb_state["active_assumption"], state=lb_state)
    lb_text = lb_action.response_text.lower()

    assert "radiologist" not in lb_text
    assert "scan" not in lb_text
    assert "hospital" not in lb_text


def test_10_concurrent_multi_project_cross_isolation():
    """10. Running 5 distinct startup domains concurrently across threads maintains strict domain isolation."""
    domains = [
        ("p-bakery", "Bakery customers", "Unsold bread in neighborhood bakeries.", "Bakery customers near bakeries buy surplus bread.", ["bakery", "bread"]),
        ("p-waste", "Municipal waste operators", "Public litter bins overflow.", "Waste collection organizations monitor litter bins.", ["waste", "bin"]),
        ("p-med", "Radiologists", "Clinical imaging scan backlogs.", "Clinic radiologists triage urgent scans.", ["radiologist", "scan"]),
        ("p-agri", "Hazelnut orchard managers", "Hazelnut orchards suffer drought.", "Hazelnut growers inspect soil moisture.", ["hazelnut", "orchard"]),
        ("p-freight", "Freight drivers", "Empty return freight loads.", "Long-haul freight drivers book backhaul loads.", ["freight", "truck"]),
    ]

    def _run_domain(d):
        pid, u, prob, stmt, expected = d
        state = {
            "project_id": pid,
            "startup_state": {"target_user": u, "problem": prob},
            "active_assumption": {"id": f"asmp-{pid}", "statement": stmt, "area": "Problem"},
        }
        action = compose_fallback_learning_action(
            assumption=state["active_assumption"],
            critic_output={"issues": [f"Unverified {u} decision-maker workflow access"]},
            state=state,
        )
        return pid, action.response_text.lower(), expected

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(_run_domain, domains))

    for pid, text, expected_terms in results:
        # Must contain at least one expected term
        assert any(e in text for e in expected_terms), f"{pid} failed to contain its expected terms"
        # Must NOT contain foreign terms from the other domains
        if pid != "p-bakery":
            assert "bakery" not in text and "bread" not in text
        if pid != "p-waste":
            assert "litter bin" not in text and "waste collection" not in text
        if pid != "p-med":
            assert "radiologist" not in text and "scan triage" not in text


def test_11_state_immutability_and_cache_leak_prevention():
    """11. Mutating state post-execution does not leak into subsequent calls."""
    state = {
        "project_id": "proj-mut-1",
        "startup_name": "Original",
        "startup_state": {"target_user": "Warehouse clerks", "problem": "Forklift maintenance schedule."},
        "active_assumption": {"id": "a-1", "statement": "Warehouse clerks log forklift hours.", "area": "Problem"},
    }
    action1 = compose_fallback_learning_action(assumption=state["active_assumption"], state=state)
    assert "forklift" in action1.response_text.lower() or "warehouse" in action1.response_text.lower()

    # External mutation
    state["startup_name"] = "Corrupted"
    state["startup_state"]["target_user"] = "CORRUPTED USER"

    # Fresh state for different project
    clean_state = {
        "project_id": "proj-mut-2",
        "startup_name": "SolarClean",
        "startup_state": {"target_user": "Solar panel technicians", "problem": "Dust accumulation on solar panels."},
        "active_assumption": {"id": "a-2", "statement": "Technicians clean solar panels.", "area": "Problem"},
    }
    action2 = compose_fallback_learning_action(assumption=clean_state["active_assumption"], state=clean_state)
    text2 = action2.response_text.lower()
    assert "forklift" not in text2
    assert "corrupted" not in text2


# ===========================================================================
# 12-16. REPOSITORY HARDCODE AUDIT SCANNERS
# ===========================================================================

def test_12_repository_audit_experiment_py_has_no_hardcoded_startup_branches():
    """12. AST inspection of startary/domain/experiment.py verifies NO hardcoded FridgeGuard/LastBite branches."""
    target_path = Path("startary/domain/experiment.py")
    assert target_path.exists()
    content = target_path.read_text(encoding="utf-8")

    tree = ast.parse(content)
    # Scan all string constants inside functions in experiment.py
    banned = ["fridgeguard", "lastbite", "shelfsense"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val_lower = node.value.lower()
            for b in banned:
                assert b not in val_lower, f"Banned hardcoded startup literal '{b}' found in experiment.py AST: {node.value[:60]}"


def test_13_repository_audit_nodes_py_has_no_hardcoded_startups_or_food_domain():
    """13. Scan startary/graph/nodes.py: verifies is_food_domain is removed and no hardcoded startup assumption branches."""
    target_path = Path("startary/graph/nodes.py")
    content = target_path.read_text(encoding="utf-8")

    # is_food_domain must be completely removed
    assert "is_food_domain" not in content, "is_food_domain variable still found in startary/graph/nodes.py!"

    # No hardcoded FridgeGuard or LastBite assumption branches
    tree = ast.parse(content)
    banned = ["fridgeguard", "lastbite"]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_generate_deterministic_fallback_critical_assumption":
            func_source = ast.unparse(node).lower()
            for b in banned:
                assert b not in func_source, f"Banned startup '{b}' found in _generate_deterministic_fallback_critical_assumption"


def test_14_repository_audit_prompts_py_has_no_student_cybersecurity_bakery_hardcodes():
    """14. Scan startary/llm/prompts.py: verifies few-shot examples do not contain IT courses, students, or bakeries."""
    target_path = Path("startary/llm/prompts.py")
    content = target_path.read_text(encoding="utf-8")

    # Key strings that were previously hardcoded in prompts.py
    assert "cybersecurity" not in content.lower()
    assert "it courses" not in content.lower()
    assert "interviewed 8 bakeries" not in content.lower()


def test_15_repository_audit_build_readiness_py_clean():
    """15. Scan startary/domain/build_readiness.py: verifies clean invariants."""
    target_path = Path("startary/domain/build_readiness.py")
    content = target_path.read_text(encoding="utf-8")
    banned = ["lastbite", "fridgeguard", "mynera", "shelfsense"]
    for b in banned:
        assert b not in content.lower()


def test_16_repository_comprehensive_codebase_scanner():
    """16. Comprehensive scanner across all startary/ python files verifying 0 hardcoded startup names in code literals."""
    startary_dir = Path("startary")
    py_files = list(startary_dir.rglob("*.py"))
    assert len(py_files) >= 10

    banned = ["lastbite", "fridgeguard", "shelfsense"]
    violations = []

    for f in py_files:
        content = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content, filename=str(f))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Ignore docstrings at module level
                val_lower = node.value.lower()
                for b in banned:
                    if b in val_lower and not any(allowed in val_lower for allowed in ["test", "fixture", "mock"]):
                        violations.append((str(f), b, node.value[:60]))

    assert not violations, f"Hardcoded startup names found in startary/ runtime code: {violations}"


# ===========================================================================
# 17-21. DYNAMIC RECOVERY CONTEXT & GROUNDING VALIDATION TESTS
# ===========================================================================

def test_17_build_recovery_context_full_extraction():
    """17. build_recovery_context extracts all fields and builds vocabulary set from multi-source state."""
    state = {
        "project_id": "proj-extract-1",
        "startup_name": "PipeInspect",
        "startup_state": {
            "target_user": "Municipal water pipe inspectors",
            "problem": "Pipeline acoustic leak detection is manual and slow.",
        },
        "weak_area": "Problem",
        "founder_messages": [
            {"role": "founder", "content": "We use acoustic listening rods on underground valves."}
        ],
        "founder_evidence": [
            {"raw_content": "Tested 14 fire hydrants in Sabail district with 2 inspectors."}
        ],
    }
    assumption = {
        "id": "asmp-pipe",
        "statement": "Municipal water pipe inspectors experience frequent undetected acoustic leaks.",
        "area": "Problem",
    }
    critic = {"issues": ["Hydrant sample was restricted to Sabail district"]}

    ctx = build_recovery_context(state=state, assumption=assumption, critic_output=critic)

    assert ctx.project_id == "proj-extract-1"
    assert ctx.startup_name == "PipeInspect"
    assert "inspectors" in ctx.target_user.lower()
    assert "pipeline" in ctx.active_problem.lower() or "acoustic" in ctx.active_problem.lower()
    assert ctx.current_weak_link == "Problem"
    assert "asmp-pipe" in assumption["id"]
    assert "sabail" in ctx.domain_vocabulary
    assert "hydrants" in ctx.domain_vocabulary
    assert "acoustic" in ctx.domain_vocabulary


def test_18_validate_recovery_grounding_passes_for_valid_action():
    """18. validate_recovery_grounding returns True for a properly grounded action."""
    ctx = RecoveryContext(
        project_id="proj-valid",
        startup_name="ValidStartup",
        active_problem="Logistics load dispatch friction",
        target_user="Freight drivers",
        current_weak_link="Problem",
        critical_assumption="Freight drivers experience dispatch delays.",
        domain_vocabulary={"freight", "drivers", "dispatch", "delays", "logistics", "loads"},
    )
    action = FallbackLearningAction(
        assumption_id="asmp-1",
        target_area="Problem",
        learning_goal="Whether freight drivers experience dispatch delays.",
        why_fallback_is_needed="Test did not meet quality bar.",
        primary_action="Talk with 3–5 freight drivers about dispatch delays.",
        who_to_engage="3–5 freight drivers",
        what_to_ask_or_observe=["Ask for recent dispatch delay."],
        what_to_bring_back=["One incident per driver."],
        what_not_to_infer=["Willingness to pay."],
        smallest_thing_to_resolve_next="Verify dispatch friction firsthand.",
        response_text="STARTARY — EXPERIMENT DESIGN NEEDS REVISION\nTalk with freight drivers.",
    )
    is_valid, reason = validate_recovery_grounding(action, ctx)
    assert is_valid is True
    assert "grounded" in reason.lower()


def test_19_validate_recovery_grounding_rejects_kitchen_in_waste_domain():
    """19. validate_recovery_grounding rejects commercial kitchen injection in waste collection domain."""
    ctx = RecoveryContext(
        project_id="proj-ecozib",
        startup_name="Ecozib",
        active_problem="Litter bin overflow",
        target_user="Waste collection managers",
        domain_vocabulary={"waste", "collection", "litter", "bin", "overflow"},
    )
    action = FallbackLearningAction(
        assumption_id="asmp-1",
        target_area="Problem",
        learning_goal="Goal",
        why_fallback_is_needed="Why",
        primary_action="Engage commercial kitchen operators to monitor refrigerator temperature.",
        who_to_engage="Commercial kitchen operators",
        what_to_ask_or_observe=["Ask about food spoilage."],
        what_to_bring_back=["Incident."],
        what_not_to_infer=["Inference."],
        smallest_thing_to_resolve_next="Verify refrigerator monitoring.",
    )
    is_valid, reason = validate_recovery_grounding(action, ctx)
    assert is_valid is False
    assert any(term in reason.lower() for term in ["refrigerator", "commercial kitchen", "kitchen operator"])


def test_20_validate_recovery_grounding_rejects_bakery_in_medical_domain():
    """20. validate_recovery_grounding rejects bakery injection in medical imaging domain."""
    ctx = RecoveryContext(
        project_id="proj-med",
        startup_name="MedScan",
        active_problem="Scan triage",
        target_user="Radiologists",
        domain_vocabulary={"radiologists", "scan", "triage", "hospital"},
    )
    action = FallbackLearningAction(
        assumption_id="asmp-1",
        target_area="Problem",
        learning_goal="Goal",
        why_fallback_is_needed="Why",
        primary_action="Run closing-time surplus offer at participating bakery.",
        who_to_engage="Bakery customers",
        what_to_ask_or_observe=["Surplus food pickup."],
        what_to_bring_back=["Pickup."],
        what_not_to_infer=["Inference."],
        smallest_thing_to_resolve_next="Verify pickup window.",
    )
    is_valid, reason = validate_recovery_grounding(action, ctx)
    assert is_valid is False
    assert "bakery" in reason.lower()


def test_21_validate_recovery_grounding_rejects_empty_action_fields():
    """21. validate_recovery_grounding rejects actions missing essential fields."""
    ctx = RecoveryContext(project_id="p-1", domain_vocabulary={"test"})
    action = FallbackLearningAction(
        assumption_id="asmp-1",
        target_area="Problem",
        learning_goal="Goal",
        why_fallback_is_needed="Why",
        primary_action="",
        who_to_engage="",
        smallest_thing_to_resolve_next="",
    )
    is_valid, reason = validate_recovery_grounding(action, ctx)
    assert is_valid is False
    assert "missing" in reason.lower()


# ===========================================================================
# 22-25. SECTION 6 GENERIC DETERMINISTIC FALLBACK TESTS
# ===========================================================================

def test_22_build_generic_deterministic_fallback_section_6_structure():
    """22. build_generic_deterministic_fallback conforms exactly to Section 6 specifications."""
    ctx = RecoveryContext(
        project_id="proj-fallback-test",
        target_user="municipal route planners",
        active_problem="litter bin monitoring",
        current_weak_link="Problem",
        critical_assumption="Route planners lack bin visibility.",
        critic_rejection_reasons=["Sample too small", "Fails to establish urgency"],
    )
    fallback = build_generic_deterministic_fallback(ctx)

    assert fallback.smallest_thing_to_resolve_next == (
        "Collect the minimum missing information required to redesign the test around the current assumption."
    )
    assert "Find 3–5 reachable people (municipal route planners)" in fallback.primary_action
    assert "3–5 reachable municipal route planners" in fallback.who_to_engage
    assert any("Concrete examples and recent incidents directly relevant to" in item for item in fallback.what_to_bring_back)
    assert any("Specific factual details addressing the unresolved questions" in item for item in fallback.what_to_bring_back)


def test_23_generic_fallback_source_basis():
    """23. Generic fallback records generic_deterministic_recovery in source_basis."""
    ctx = RecoveryContext(
        project_id="p-src",
        current_weak_link="Acquisition",
        critical_assumption="Test assumption",
    )
    fallback = build_generic_deterministic_fallback(ctx)
    assert "generic_deterministic_recovery" in fallback.source_basis
    assert "Acquisition" in fallback.source_basis


def test_24_generic_fallback_rendering_compliance():
    """24. Generic fallback renders required markdown headers and boundary notices."""
    ctx = RecoveryContext(
        project_id="p-render",
        current_weak_link="Problem",
        critical_assumption="Test assumption",
    )
    fallback = build_generic_deterministic_fallback(ctx)
    rendered = fallback.response_text

    assert "STARTARY — EXPERIMENT DESIGN NEEDS REVISION" in rendered
    assert "WHAT DO WE STILL NEED TO LEARN?" in rendered
    assert "WHY THE TEST COULD NOT BE TRUSTED:" in rendered
    assert "THE SMALLEST THING TO RESOLVE NEXT:" in rendered
    assert "WHAT SHOULD I DO NEXT?" in rendered
    assert "WHAT SHOULD I BRING BACK?" in rendered
    assert "WHAT SHOULD I NOT CONCLUDE YET?" in rendered
    assert "This action is for learning, not validation." in rendered


def test_25_compose_fallback_learning_action_automatic_substitution_on_ungrounded():
    """25. When grounding validation fails, compose_fallback_learning_action seamlessly substitutes generic fallback."""
    # Active startup is drone courier; critic introduces foreign term that corrupts generation
    state = {
        "project_id": "proj-drone",
        "startup_name": "DroneExpress",
        "startup_state": {"target_user": "Rural clinic nurses", "problem": "Blood vial transport."},
        "active_assumption": {"id": "a-drone", "statement": "Nurses require drone blood transport.", "area": "Problem"},
        "raw_context": "DroneExpress delivers blood samples by autonomous quadcopter.",
    }
    # Critic mentions "refrigerator" or "commercial kitchen" which would have triggered old Category D
    critic_output = {
        "verdict": "REVISE",
        "issues": ["Commercial kitchen decision-maker access was not verified"],
    }
    action = compose_fallback_learning_action(
        assumption=state["active_assumption"],
        weak_link="Problem",
        critic_output=critic_output,
        state=state,
    )

    # Must NOT output commercial kitchen or refrigerator
    assert "commercial kitchen" not in action.response_text.lower()
    assert "refrigerator" not in action.response_text.lower()
    # Must be grounded or safely substituted
    assert action.who_to_engage != ""
    assert action.primary_action != ""


# ===========================================================================
# 26-30. FULL SYSTEM REGRESSION & GRAPH NODE TESTS
# ===========================================================================

def test_26_lastbite_regression_produces_grounded_action():
    """26. LastBite scenario continues producing closing-time offer grounded fallback."""
    state = {
        "project_id": "proj-lb-reg",
        "startup_name": "LastBite",
        "startup_state": {
            "target_user": "Nearby Baku consumers",
            "problem": "Unsold bakery surplus.",
        },
        "weak_area": "Target User",
        "active_assumption": {
            "id": "asmp-lb-reg",
            "statement": "There is a definable segment of people near independent Baku bakeries and small cafes with recurring closing-time surplus who collect surplus food.",
            "area": "Target User",
        },
    }
    critic = {
        "verdict": "REVISE",
        "issues": [
            "Actual proximity during specific offer window was not verified",
            "Non-reservations could reflect participant unavailability",
        ],
    }
    action = compose_fallback_learning_action(
        assumption=state["active_assumption"],
        weak_link="Target User",
        critic_output=critic,
        state=state,
    )
    text = action.response_text.lower()
    assert "closing-time" in text or "pre-closing" in text or "pickup window" in text
    assert "bakery" in text or "cafe" in text


def test_27_shelfsense_regression_produces_grounded_action():
    """27. ShelfSense scenario continues producing grocery store grounded fallback."""
    state = {
        "project_id": "proj-shelf-reg",
        "startup_name": "ShelfSense",
        "startup_state": {
            "target_user": "Independent grocery store owners",
            "problem": "Perishable dairy and produce stockouts.",
        },
        "weak_area": "Problem",
        "active_assumption": {
            "id": "asmp-shelf",
            "statement": "Independent grocery store owners experience stockouts and spoilage as an urgent problem.",
            "area": "Problem",
        },
    }
    critic = {"verdict": "REVISE", "issues": ["Free tally sheet is a weak proxy for adoption."]}
    action = compose_fallback_learning_action(
        assumption=state["active_assumption"],
        weak_link="Problem",
        critic_output=critic,
        state=state,
    )
    text = f"{action.who_to_engage} {action.primary_action} {action.response_text}".lower()
    assert "grocery" in text or "store" in text or "stockouts" in text
    assert "refrigerator" not in text
    assert "bakery" not in text


def test_28_ecozib_handle_experiment_failure_stop_graph_node():
    """28. handle_experiment_failure_stop graph node for Ecozib produces grounded fallback in state."""
    state = {
        "project_id": "proj-ecozib-node",
        "startup_name": "Ecozib",
        "startup_state": {
            "name": "Ecozib",
            "target_user": "Waste collection operators",
            "problem": "Litter bin route planning friction",
        },
        "weak_area": "Problem",
        "active_assumption": {
            "id": "asmp-eco-node",
            "statement": "Waste collection organizations experience recurring collection-planning friction.",
            "area": "Problem",
        },
        "proposed_experiment_output": {
            "id": "exp-failed-node",
            "objective": "Test waste collection routes",
            "hypothesis": "Waste collection organizations experience recurring collection-planning friction.",
        },
        "experiment_critic_output": {
            "verdict": "REVISE",
            "issues": ["Decision-maker access unverified for waste fleet operations"],
        },
        "iteration": 1,
    }

    result = handle_experiment_failure_stop(state)

    assert result["phase"] == WorkflowPhase.WAITING_FOR_LEARNING_INPUT
    assert result["experiment_design_status"] == ExperimentDesignStatus.QUALITY_BAR_NOT_MET.value
    fb = result["fallback_learning_action"]
    assert isinstance(fb, dict)

    fb_text = str(fb).lower()
    assert "refrigerator" not in fb_text
    assert "commercial kitchen" not in fb_text
    assert "food spoilage" not in fb_text
    assert "bakery" not in fb_text
    assert "waste" in fb_text or "collection" in fb_text or "bin" in fb_text


def test_29_assess_build_readiness_ecozib_has_no_surplus_bag_scope():
    """29. assess_build_readiness for Ecozib produces domain-neutral/Ecozib scope, never surplus bags or bakeries."""
    state = {
        "project_id": "proj-ecozib-br",
        "startup_name": "Ecozib",
        "raw_context": "Ecozib provides smart litter bin monitoring for municipal waste collection.",
        "founder_messages": [
            {"role": "founder", "content": "I want to start coding the mobile route app."}
        ],
        "active_assumption": {
            "statement": "Waste collection organizations will adopt automated route dispatch.",
            "area": "Problem",
        },
        "founder_evidence": [],
    }
    result = assess_build_readiness(state)
    br = result["build_readiness"]
    assert br["decision"] == BuildReadinessAction.CONTINUE_MANUAL.value
    reason = br["reason"].lower()
    assert "bakery" not in reason
    assert "surplus" not in reason
    assert "merchant" not in reason


def test_30_assess_build_readiness_lastbite_escalation_preserved():
    """30. assess_build_readiness for LastBite preserves BUILD_MINIMAL_PROTOTYPE escalation when prerequisites met."""
    state = {
        "project_id": "proj-lb-escalation",
        "active_assumption": {
            "statement": "Merchants and buyers can reliably publish and reserve time-sensitive surplus without founder coordination.",
            "area": "Solution / MVP",
        },
        "founder_evidence": [
            {
                "raw_content": "10 unconnected bakeries contacted, 3 agreed, Bakery A and Bakery B ran pilots with genuine surplus.",
                "evidence_type": "EXPERIMENT_RESULT",
            },
            {
                "raw_content": "Across two separate evenings, 32 fresh nearby adults reached, paid deposits, 5 completed pickups, behavior repeated on both evenings.",
                "evidence_type": "EXPERIMENT_RESULT",
            },
        ],
        "founder_messages": [
            {"role": "founder", "content": "Manual WhatsApp coordination is now the main bottleneck. I cannot keep texting every customer and store."}
        ],
    }
    result = assess_build_readiness(state)
    br = result["build_readiness"]

    assert br["decision"] == BuildReadinessAction.BUILD_MINIMAL_PROTOTYPE.value
    assert br["manual_learning_sufficient"] is True
    assert br["next_uncertainty_requires_product"] is True
    min_scope = br["minimum_build_scope"]
    assert len(min_scope) >= 3
    assert any("merchant" in s.lower() and "surplus" in s.lower() for s in min_scope)
    assert any("buyer" in s.lower() and "reserve" in s.lower() for s in min_scope)
    assert any("system" in s.lower() or "inventory" in s.lower() for s in min_scope)
