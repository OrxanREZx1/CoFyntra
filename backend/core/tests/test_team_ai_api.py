from datetime import timedelta

import pytest
from django.utils import timezone
from rest_framework.test import APIClient

from cofytra_ai.models import (
    CatchupPoint,
    CatchupResponse,
    ClaimRef,
    Conflict,
    ConflictResponse,
    ConflictKind,
    ErrorCode,
    ResponseMeta,
)
from core import models as core_models
from core.models import CatchupCursor, ExtractionJob, PrivateMessage
from core.api.team_ai import _advance_catchup_cursor
from core.services import ai_adapter
from core.services.ai_adapter import AIAdapterError
from core.services.ai_mock import MockBackend

pytestmark = pytest.mark.django_db


def _client(user):
    client = APIClient()
    client.force_authenticate(user=user)
    return client


def _catchup_path(project):
    return f"/api/projects/{project.pk}/catch-up/"


def _conflicts_path(project):
    return f"/api/projects/{project.pk}/conflicts/detect/"


def _enable_mock(monkeypatch, *, catchup_requests=None, conflict_requests=None):
    backend = MockBackend()
    original_catchup = ai_adapter.catch_up
    original_conflicts = ai_adapter.detect_conflicts

    def catch_up(request, *, backend=backend):
        if catchup_requests is not None:
            catchup_requests.append(request)
        return original_catchup(request, backend=backend)

    def detect_conflicts(request, *, backend=backend):
        if conflict_requests is not None:
            conflict_requests.append(request)
        return original_conflicts(request, backend=backend)

    monkeypatch.setattr(ai_adapter, "catch_up", catch_up)
    monkeypatch.setattr(ai_adapter, "detect_conflicts", detect_conflicts)
    return backend


def _meta(request_id):
    return ResponseMeta(
        request_id=request_id,
        prompt_version="test.v1",
        model=None,
        tier="none",
        latency_ms=0,
        attempts=0,
    )


def test_each_founder_catches_up_on_the_other_founders_shared_knowledge(project, ana, ben, make_knowledge, monkeypatch):
    ana_item = make_knowledge(project, ana, statement="Ana published a scheduling finding.")
    ben_item = make_knowledge(project, ben, statement="Ben published a customer finding.")
    requests = []
    _enable_mock(monkeypatch, catchup_requests=requests)

    ana_response = _client(ana).post(_catchup_path(project), {}, format="json")
    ben_response = _client(ben).post(_catchup_path(project), {}, format="json")

    assert ana_response.status_code == ben_response.status_code == 200
    assert [item.id for item in requests[0].knowledge] == [str(ben_item.pk)]
    assert [item.id for item in requests[1].knowledge] == [str(ana_item.pk)]
    assert requests[0].other_founder.id == str(ben.pk)
    assert requests[1].other_founder.id == str(ana.pk)


def test_catchup_never_reads_or_returns_private_messages_or_provenance(
    project, ana, ben, make_message, make_knowledge, monkeypatch
):
    private = make_message(project, ben, "PRIVATE-TRANSCRIPT-SECRET")
    knowledge = make_knowledge(project, ben, statement="Published claim from a founder.")
    requests = []
    _enable_mock(monkeypatch, catchup_requests=requests)

    response = _client(ana).post(_catchup_path(project), {}, format="json")
    body = response.content.decode()

    assert response.status_code == 200
    assert "PRIVATE-TRANSCRIPT-SECRET" not in body
    assert str(private.pk) not in body
    assert str(private.pk) not in requests[0].model_dump_json()
    assert str(knowledge.pk) in body
    assert not PrivateMessage.objects.filter(owner=ben).exclude(pk=private.pk).exists()


def test_catchup_cursor_advances_only_after_validated_success(project, ana, ben, make_knowledge, monkeypatch):
    item = make_knowledge(project, ben)
    requests = []
    _enable_mock(monkeypatch, catchup_requests=requests)

    response = _client(ana).post(_catchup_path(project), {}, format="json")

    assert response.status_code == 200
    cursor = CatchupCursor.objects.get(project=project, viewer=ana, other_founder=ben)
    assert cursor.last_caught_up_at == item.shared_at
    assert str(cursor.last_knowledge_id) == str(item.pk)
    assert response.json()["pagination"] == {
        "has_more": False,
        "batch_size": 1,
        "remaining_at_request": 0,
    }


def test_catchup_ai_failure_does_not_create_or_advance_cursor(project, ana, ben, make_knowledge, monkeypatch):
    make_knowledge(project, ben)

    def fail(_request):
        raise AIAdapterError("catch_up", ErrorCode.PROVIDER_UNAVAILABLE, "catchup-test")

    monkeypatch.setattr(ai_adapter, "catch_up", fail)
    response = _client(ana).post(_catchup_path(project), {}, format="json")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == ErrorCode.PROVIDER_UNAVAILABLE
    assert not CatchupCursor.objects.filter(project=project, viewer=ana, other_founder=ben).exists()


def test_repeated_catchup_does_not_replay_processed_knowledge(project, ana, ben, make_knowledge, monkeypatch):
    item = make_knowledge(project, ben)
    requests = []
    _enable_mock(monkeypatch, catchup_requests=requests)
    client = _client(ana)

    first = client.post(_catchup_path(project), {}, format="json")
    second = client.post(_catchup_path(project), {}, format="json")

    assert first.status_code == second.status_code == 200
    assert [len(request.knowledge) for request in requests] == [1, 0]
    assert requests[0].knowledge[0].id == str(item.pk)
    assert requests[1].knowledge == []
    assert second.json()["catch_up"]["meta"]["tier"] == "none"


def test_catchup_cursor_never_moves_backwards_for_a_stale_response(project, ana, ben, make_knowledge):
    timestamp = timezone.now() - timedelta(hours=1)
    for index in range(2):
        make_knowledge(project, ben, statement=f"Same-time claim {index}.", shared_at=timestamp)
    ordered = list(project.knowledge.filter(author=ben).order_by("shared_at", "id"))
    latest = ordered[-1]
    stale = ordered[0]
    CatchupCursor.objects.create(
        project=project,
        viewer=ana,
        other_founder=ben,
        last_caught_up_at=latest.shared_at,
        last_knowledge_id=latest.pk,
    )

    _advance_catchup_cursor(
        project=project,
        viewer=ana,
        other_founder=str(ben.pk),
        covered_at=stale.shared_at,
        knowledge_id=str(stale.pk),
    )

    cursor = CatchupCursor.objects.get(project=project, viewer=ana, other_founder=ben)
    assert cursor.last_knowledge_id == latest.pk


def test_large_backlog_with_identical_timestamps_is_processed_without_skips(
    project, ana, ben, make_knowledge, monkeypatch
):
    timestamp = timezone.now() - timedelta(hours=1)
    items = [
        make_knowledge(project, ben, statement=f"Shared knowledge {index}.", shared_at=timestamp)
        for index in range(125)
    ]
    requests = []
    _enable_mock(monkeypatch, catchup_requests=requests)
    client = _client(ana)
    responses = []

    while True:
        response = client.post(_catchup_path(project), {}, format="json")
        responses.append(response)
        assert response.status_code == 200
        if not response.json()["pagination"]["has_more"]:
            break
        assert len(responses) < 5

    processed = [item.id for request in requests for item in request.knowledge]
    expected = [
        str(pk)
        for pk in project.knowledge.filter(author=ben).order_by("shared_at", "id").values_list("pk", flat=True)
    ]
    assert [len(request.knowledge) for request in requests] == [60, 60, 5]
    assert processed == expected
    assert len(set(processed)) == len(items)
    assert [response.json()["pagination"]["has_more"] for response in responses] == [True, True, False]


def test_catchup_rejects_client_supplied_founder_ids(project, ana, ben, monkeypatch):
    _enable_mock(monkeypatch)
    response = _client(ana).post(
        _catchup_path(project),
        {"other_founder_id": str(ana.pk), "viewer_id": str(ben.pk)},
        format="json",
    )
    assert response.status_code == 400
    assert not CatchupCursor.objects.exists()


def test_catchup_denies_nonmember(project, cara, monkeypatch):
    _enable_mock(monkeypatch)
    response = _client(cara).post(_catchup_path(project), {}, format="json")
    assert response.status_code == 403


def test_catchup_excludes_cross_project_and_awaiting_safety_knowledge(
    project, other_project, ana, ben, cara, make_knowledge, make_message, monkeypatch
):
    allowed = make_knowledge(project, ben, statement="Published and authorized.")
    make_knowledge(other_project, cara, statement="Cross-project claim.")
    private = make_message(project, ben, "Candidate awaiting the safety guard.")
    ExtractionJob.objects.create(
        project=project,
        founder=ben,
        source_message=private,
        status="awaiting_safety_validation",
        extraction_result=[{"statement": "Not yet published."}],
    )
    requests = []
    _enable_mock(monkeypatch, catchup_requests=requests)

    response = _client(ana).post(_catchup_path(project), {}, format="json")

    assert response.status_code == 200
    assert [item.id for item in requests[0].knowledge] == [str(allowed.pk)]
    text = response.content.decode()
    assert "Cross-project claim." not in text
    assert "Not yet published." not in text


def test_catchup_empty_context_returns_valid_no_llm_response(project, ana, monkeypatch):
    backend = _enable_mock(monkeypatch)

    response = _client(ana).post(_catchup_path(project), {}, format="json")

    assert response.status_code == 200
    assert response.json()["catch_up"]["discoveries"] == []
    assert response.json()["catch_up"]["assumptions"] == []
    assert response.json()["catch_up"]["evidence"] == []
    assert response.json()["catch_up"]["meta"]["tier"] == "none"
    assert backend.calls == ["catch_up"]
    assert not CatchupCursor.objects.exists()


def test_invalid_catchup_ai_reference_is_rejected_without_cursor_advance(project, ana, ben, make_knowledge, monkeypatch):
    make_knowledge(project, ben)

    class InvalidBackend:
        name = "invalid-test"

        def call(self, capability, request):
            return CatchupResponse(
                summary="Invalid reference.",
                discoveries=[CatchupPoint(text="Outside the request.", knowledge_ids=["kn-outside-project"])],
                meta=_meta(request.request_id),
            )

    monkeypatch.setattr(ai_adapter, "get_backend", lambda: InvalidBackend())
    response = _client(ana).post(_catchup_path(project), {}, format="json")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == ErrorCode.INVALID_MODEL_OUTPUT
    assert not CatchupCursor.objects.exists()


def test_conflict_detection_preserves_both_founders_and_epistemic_labels(
    project, ana, ben, make_knowledge, monkeypatch
):
    ben_claim = make_knowledge(
        project, ben, category="target_customer", epistemic_status="founder_hypothesis",
        statement="Diagnostic labs are the best customer.",
    )
    ana_claim = make_knowledge(
        project, ana, category="target_customer", epistemic_status="founder_observation",
        statement="Independent clinics show the strongest need.",
    )
    requests = []
    _enable_mock(monkeypatch, conflict_requests=requests)

    response = _client(ana).post(_conflicts_path(project), {}, format="json")
    data = response.json()

    assert response.status_code == 200
    assert len(data["conflicts"]) == 1
    conflict = data["conflicts"][0]
    claims = {conflict["claim_a"]["knowledge_id"]: conflict["claim_a"],
              conflict["claim_b"]["knowledge_id"]: conflict["claim_b"]}
    assert claims[str(ben_claim.pk)]["author_id"] == str(ben.pk)
    assert claims[str(ben_claim.pk)]["epistemic_status"] == "founder_hypothesis"
    assert claims[str(ana_claim.pk)]["author_id"] == str(ana.pk)
    assert claims[str(ana_claim.pk)]["epistemic_status"] == "founder_observation"
    assert conflict["question_to_resolve"]
    assert len(requests[0].knowledge) == 2
    assert data["meta"]["tier"] == "none"


def test_conflicts_include_only_authorized_published_knowledge(
    project, other_project, ana, ben, cara, make_knowledge, make_message, monkeypatch
):
    own = make_knowledge(project, ana, category="target_customer", epistemic_status="founder_hypothesis")
    other = make_knowledge(project, ben, category="target_customer", epistemic_status="founder_hypothesis")
    external = make_knowledge(other_project, cara, statement="Cross-project claim.")
    private = make_message(project, ben, "PRIVATE-CONFLICT-CONTEXT")
    ExtractionJob.objects.create(
        project=project,
        founder=ben,
        source_message=private,
        status="awaiting_safety_validation",
        extraction_result=[{"statement": "Unpublished candidate."}],
    )
    requests = []
    _enable_mock(monkeypatch, conflict_requests=requests)

    response = _client(ana).post(_conflicts_path(project), {}, format="json")

    assert response.status_code == 200
    assert {item.id for item in requests[0].knowledge} == {str(own.pk), str(other.pk)}
    assert str(external.pk) not in response.content.decode()
    assert str(private.pk) not in response.content.decode()
    assert "PRIVATE-CONFLICT-CONTEXT" not in response.content.decode()


def test_conflict_detection_rejects_nonmember(project, cara, monkeypatch):
    _enable_mock(monkeypatch)
    response = _client(cara).post(_conflicts_path(project), {}, format="json")
    assert response.status_code == 403


def test_conflict_detection_is_recomputed_and_not_persisted(project, ana, ben, make_knowledge, monkeypatch):
    make_knowledge(project, ana, category="target_customer", epistemic_status="founder_hypothesis")
    make_knowledge(project, ben, category="target_customer", epistemic_status="founder_hypothesis")
    requests = []
    _enable_mock(monkeypatch, conflict_requests=requests)
    client = _client(ana)

    first = client.post(_conflicts_path(project), {}, format="json")
    second = client.post(_conflicts_path(project), {}, format="json")

    assert first.status_code == second.status_code == 200
    assert first.json()["conflicts"] == second.json()["conflicts"]
    assert len(requests) == 2
    assert not hasattr(core_models, "Conflict")


def test_invalid_conflict_reference_is_rejected(project, ana, ben, make_knowledge, monkeypatch):
    make_knowledge(project, ben, category="target_customer", epistemic_status="founder_hypothesis")
    make_knowledge(project, ana, category="target_customer", epistemic_status="founder_hypothesis")

    class InvalidBackend:
        name = "invalid-test"

        def call(self, capability, request):
            valid_other_claim = next(item for item in request.knowledge if item.author_id != str(ben.pk))
            return ConflictResponse(
                conflicts=[Conflict(
                    claim_a=ClaimRef(knowledge_id="kn-not-authorized", author_id=str(ben.pk), statement="Foreign."),
                    claim_b=ClaimRef(knowledge_id=valid_other_claim.id,
                                     author_id=valid_other_claim.author_id,
                                     statement=valid_other_claim.statement),
                    kind=ConflictKind.TENSION,
                    explanation="These claims conflict.",
                    question_to_resolve="Which is correct?",
                )],
                meta=_meta(request.request_id),
            )

    monkeypatch.setattr(ai_adapter, "get_backend", lambda: InvalidBackend())
    response = _client(ana).post(_conflicts_path(project), {}, format="json")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == ErrorCode.INVALID_MODEL_OUTPUT


def test_conflict_detection_empty_context_returns_empty_result(project, ana, monkeypatch):
    backend = _enable_mock(monkeypatch)

    response = _client(ana).post(_conflicts_path(project), {}, format="json")

    assert response.status_code == 200
    assert response.json()["conflicts"] == []
    assert response.json()["meta"]["tier"] == "none"
    assert backend.calls == ["detect_conflicts"]


def test_conflict_ai_failure_does_not_persist_results(project, ana, monkeypatch):
    monkeypatch.setattr(
        ai_adapter,
        "detect_conflicts",
        lambda _request: (_ for _ in ()).throw(
            AIAdapterError("detect_conflicts", ErrorCode.PROVIDER_UNAVAILABLE, "conflict-test")
        ),
    )

    response = _client(ana).post(_conflicts_path(project), {}, format="json")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == ErrorCode.PROVIDER_UNAVAILABLE
