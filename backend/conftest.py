import pytest

from core.models import PrivateMessage, Project, SharedKnowledge


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
