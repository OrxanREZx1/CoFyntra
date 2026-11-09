import pytest

from core.models import (
    CommittedDecision,
    CommittedState,
    DecisionProposal,
    PrivateMessage,
    Project,
    ProposalVersion,
    SharedKnowledge,
)


@pytest.fixture(autouse=True)
def _no_paid_llm_calls(monkeypatch):
    """Backend tests are offline: any accidental OpenAI client would get a dummy key and an unroutable URL."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test-dummy")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:9")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def ana(django_user_model):
    return django_user_model.objects.create_user(username="ana", password="unused-test-password")


@pytest.fixture
def ben(django_user_model):
    return django_user_model.objects.create_user(username="ben", password="unused-test-password")


@pytest.fixture
def cara(django_user_model):
    return django_user_model.objects.create_user(username="cara", password="unused-test-password")


@pytest.fixture
def project(ana, ben):
    """The hackathon shape: exactly two founders in one project."""
    p = Project.objects.create(title="Clinic scheduling")
    p.add_founder(ana)
    p.add_founder(ben)
    return p


@pytest.fixture
def other_project(cara):
    p = Project.objects.create(title="Unrelated startup")
    p.add_founder(cara)
    return p


@pytest.fixture
def make_message():
    def _make(project, owner, content="I talked to 8 clinic managers this week.", role="founder", **kwargs):
        return PrivateMessage.objects.create(project=project, owner=owner, role=role, content=content, **kwargs)
    return _make


@pytest.fixture
def make_knowledge():
    def _make(project, author, **kwargs):
        fields = {
            "category": "problem",
            "statement": "Clinics lose hours to manual scheduling.",
            "epistemic_status": "founder_observation",
            "source_quote": "scheduling eats about 3 hours a day",
            "origin": "extracted",
        }
        fields.update(kwargs)
        return SharedKnowledge.objects.create(project=project, author=author, **fields)
    return _make


@pytest.fixture
def commit_revision():
    """Create committed revision N with its (committed) proposal, version and decision."""
    def _commit(project, author, revision, title="Start B2B, not B2C", **state_fields):
        proposal = DecisionProposal.objects.create(project=project, created_by=author, base_state_revision=revision - 1)
        version = ProposalVersion.create_next(proposal=proposal, created_by=author, title=title, rationale="Agreed.",
                                              trade_offs=["None noted."], next_action="Proceed.")
        proposal.status = DecisionProposal.Status.COMMITTED
        proposal.save()
        state = CommittedState.objects.create(project=project, revision=revision, **state_fields)
        return CommittedDecision.objects.create(project=project, source_proposal=proposal,
                                                source_proposal_version=version, committed_state=state,
                                                title=title, summary="We sell to providers.")
    return _commit
