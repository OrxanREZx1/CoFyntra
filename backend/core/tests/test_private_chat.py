from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier

import pytest
from django.db import close_old_connections, connection, connections
from django.utils import timezone
from rest_framework.test import APIClient

from cofytra_ai.models import (
    ChatResponse,
    EpistemicStatus,
    ExtractResponse,
    KnowledgeCategory,
    ResponseMeta,
)
from core.services import ai_context
from core.models import (
    ExtractionJob,
    ExtractionJobStatus,
    KnowledgeSourceMessage,
    PrivateMessage,
    SharedKnowledge,
)
from core.services import ai_adapter
from core.services.ai_adapter import AIAdapterError
from core.services.extraction_jobs import claim_next_job, process_claimed_job, run_one_job

pytestmark = pytest.mark.django_db


def _client(user):
    client = APIClient()
    client.force_authenticate(user=user)
    return client


def _chat_response(request, reply="Full assistant answer with details."):
    return ChatResponse(
        reply=reply,
        referenced_knowledge_ids=[],
        referenced_decision_ids=[],
        meta=ResponseMeta(
            request_id=request.request_id,
            prompt_version="test.chat.v1",
            model=None,
            tier="none",
            latency_ms=0,
            attempts=0,
        ),
    )


def _extract_response(request, source_message, *, quote=None):
    message = next(item for item in request.messages if item.id == str(source_message.pk))
    cited_quote = quote if quote is not None else message.text[:40]
    return ExtractResponse(
        candidates=[{
            "category": KnowledgeCategory.PROBLEM,
            "statement": "Clinics lose time to manual scheduling.",
            "epistemic_status": EpistemicStatus.FOUNDER_OBSERVATION,
            "confidence": "high",
            "source_message_ids": [str(source_message.pk)],
            "source_quote": cited_quote,
        }],
        meta=ResponseMeta(
            request_id=request.request_id,
            prompt_version="test.extract.v1",
            model=None,
            tier="none",
            latency_ms=0,
            attempts=0,
        ),
    )


def _post(client, project, message_id="client-1", content="We interviewed eight clinic managers."):
    return client.post(
        f"/api/projects/{project.pk}/messages/",
        {"content": content, "client_message_id": message_id},
        format="json",
    )


def test_founder_posts_and_retrieves_private_messages_without_running_extraction(
    project, ana, ben, monkeypatch
):
    client = _client(ana)
    calls = {"chat": 0, "extract": 0}

    def chat(request):
        calls["chat"] += 1
        return _chat_response(request)

    def extract(_request):
        calls["extract"] += 1
        raise AssertionError("extraction must run asynchronously")

    monkeypatch.setattr(ai_adapter, "chat", chat)
    monkeypatch.setattr(ai_adapter, "extract_knowledge", extract)
    response = _post(client, project)

    assert response.status_code == 201
    data = response.json()
    assert data["chat_status"] == "succeeded"
    assert data["extraction"]["status"] == ExtractionJobStatus.PENDING
    assert data["request_id"] == data["response_meta"]["request_id"]
    assert data["reply"]["content"] == "Full assistant answer with details."
    assert data["reply"]["ai_response"]["reply"] == "Full assistant answer with details."
    assert calls == {"chat": 1, "extract": 0}
    assert ExtractionJob.objects.filter(source_message_id=data["founder_message_id"]).count() == 1

    own = client.get(f"/api/projects/{project.pk}/messages/").json()["results"]
    assert [item["content"] for item in own] == [
        "We interviewed eight clinic managers.",
        "Full assistant answer with details.",
    ]
    assert own[0]["extraction_status"] == ExtractionJobStatus.PENDING

    other_founder_history = _client(ben).get(f"/api/projects/{project.pk}/messages/").json()["results"]
    assert other_founder_history == []


def test_retry_with_same_idempotency_key_returns_saved_reply_and_rejects_reuse(
    project, ana, monkeypatch
):
    calls = []
    monkeypatch.setattr(ai_adapter, "chat", lambda request: calls.append(request) or _chat_response(request))
    client = _client(ana)

    first = _post(client, project)
    retry = _post(client, project)
    conflict = _post(client, project, content="Different content with the same key.")

    assert first.status_code == 201
    assert retry.status_code == 200
    assert retry.json()["founder_message_id"] == first.json()["founder_message_id"]
    assert retry.json()["assistant_message_id"] == first.json()["assistant_message_id"]
    assert conflict.status_code == 409
    assert PrivateMessage.objects.filter(project=project, owner=ana, role="founder").count() == 1
    assert PrivateMessage.objects.filter(project=project, owner=ana, role="assistant").count() == 1
    assert ExtractionJob.objects.count() == 1
    assert len(calls) == 1


def test_chat_provider_failure_keeps_message_and_same_key_can_retry(project, ana, monkeypatch):
    calls = {"count": 0}

    def chat(request):
        calls["count"] += 1
        if calls["count"] == 1:
            from cofytra_ai.models import ErrorCode
            raise AIAdapterError("chat", ErrorCode.PROVIDER_UNAVAILABLE, request.request_id)
        return _chat_response(request, "Recovered assistant response.")

    monkeypatch.setattr(ai_adapter, "chat", chat)
    client = _client(ana)
    first = _post(client, project)

    assert first.status_code == 503
    assert first.json()["chat_status"] == "failed"
    assert first.json()["retryable"] is True
    assert PrivateMessage.objects.filter(project=project, owner=ana, role="founder").count() == 1
    assert not PrivateMessage.objects.filter(project=project, owner=ana, role="assistant").exists()
    assert first.json()["extraction"]["status"] == ExtractionJobStatus.PENDING

    retried = _post(client, project)
    assert retried.status_code == 200
    assert retried.json()["reply"]["content"] == "Recovered assistant response."
    assert PrivateMessage.objects.filter(project=project, owner=ana, role="founder").count() == 1
    assert PrivateMessage.objects.filter(project=project, owner=ana, role="assistant").count() == 1
    assert ExtractionJob.objects.count() == 1


def test_private_chat_requires_valid_idempotency_key_and_membership(project, ana, cara, monkeypatch):
    monkeypatch.setattr(ai_adapter, "chat", lambda request: _chat_response(request))
    response = _post(_client(ana), project, message_id="not valid")
    assert response.status_code == 400
    assert _post(_client(cara), project).status_code == 403
    assert not PrivateMessage.objects.exists()


def test_both_members_can_read_shared_knowledge_without_private_source_ids(
    project, ana, ben, cara, make_message, make_knowledge
):
    source = make_message(project, ana, "A private source that must never appear in shared history.")
    item = make_knowledge(
        project,
        ana,
        category="problem",
        epistemic_status="founder_observation",
        statement="Scheduling is a problem for clinics.",
        source_quote="Scheduling takes hours.",
    )
    KnowledgeSourceMessage.objects.create(knowledge=item, message=source)

    for user in (ana, ben):
        response = _client(user).get(f"/api/projects/{project.pk}/knowledge/")
        assert response.status_code == 200
        data = response.json()["results"][0]
        assert data["id"] == str(item.pk)
        assert data["author"]["id"] == str(ana.pk)
        assert data["epistemic_status"] == "founder_observation"
        assert data["source_quote"] == "Scheduling takes hours."
        assert str(source.pk) not in response.content.decode()
        assert "private source" not in response.content.decode()
    assert _client(cara).get(f"/api/projects/{project.pk}/knowledge/").status_code == 403
    assert SharedKnowledge.objects.filter(project=project).count() == 1


def test_private_message_pages_have_stable_order_and_bounded_size(project, ana, make_message):
    baseline = timezone.now()
    messages = [
        make_message(project, ana, f"Private message {index}.", created_at=baseline + timedelta(seconds=index))
        for index in range(3)
    ]
    client = _client(ana)

    first = client.get(f"/api/projects/{project.pk}/messages/?page=1&page_size=1")
    second = client.get(f"/api/projects/{project.pk}/messages/?page=2&page_size=1")
    too_large = client.get(f"/api/projects/{project.pk}/messages/?page_size=101")

    assert [first.json()["results"][0]["id"], second.json()["results"][0]["id"]] == [
        str(messages[0].pk), str(messages[1].pk)
    ]
    assert first.json()["has_next"] is True
    assert too_large.status_code == 400


def test_extraction_saves_candidates_but_never_publishes_without_sensitive_guard(
    project, ana, monkeypatch, make_message
):
    source = make_message(project, ana, "We observed that clinic scheduling takes two hours.")
    job = ExtractionJob.objects.create(project=project, founder=ana, source_message=source)
    calls = []

    def extract(request):
        calls.append(request)
        return _extract_response(request, source)

    monkeypatch.setattr(ai_adapter, "extract_knowledge", extract)
    result = run_one_job()
    job.refresh_from_db()

    assert result.status == ExtractionJobStatus.AWAITING_SAFETY_VALIDATION
    assert job.status == ExtractionJobStatus.AWAITING_SAFETY_VALIDATION
    assert job.last_error_code == "sensitive_content_guard_not_integrated"
    assert job.extraction_result[0]["category"] == "problem"
    assert job.extraction_result[0]["epistemic_status"] == "founder_observation"
    assert job.extraction_result[0]["confidence"] == "high"
    assert job.extraction_result[0]["source_message_ids"] == [str(source.pk)]
    assert job.extraction_result[0]["source_quote"] in source.content
    assert job.founder_id == source.owner_id
    assert len(calls) == 1
    assert not SharedKnowledge.objects.filter(project=project).exists()

    assert run_one_job() is None
    assert not SharedKnowledge.objects.filter(project=project).exists()


def test_invalid_extraction_candidate_is_rejected_before_publication(project, ana, monkeypatch, make_message):
    source = make_message(project, ana, "The founder reported a real observation.")
    job = ExtractionJob.objects.create(project=project, founder=ana, source_message=source)
    monkeypatch.setattr(
        ai_adapter,
        "extract_knowledge",
        lambda request: _extract_response(request, source, quote="not present in the source"),
    )

    result = run_one_job()
    job.refresh_from_db()

    assert result.status == ExtractionJobStatus.FAILED
    assert job.last_error_code == "invalid_candidate_or_provenance"
    assert job.extraction_result is not None
    assert claim_next_job() is None
    assert not SharedKnowledge.objects.filter(project=project).exists()


def test_extraction_failure_retries_cached_results_without_duplicate_publication(
    project, ana, monkeypatch, make_message
):
    source = make_message(project, ana, "We heard the same scheduling concern repeatedly.")
    job = ExtractionJob.objects.create(project=project, founder=ana, source_message=source)
    monkeypatch.setattr(ai_adapter, "extract_knowledge",
                        lambda request: _extract_response(request, source))

    first = run_one_job()
    assert first.status == ExtractionJobStatus.AWAITING_SAFETY_VALIDATION
    assert run_one_job() is None
    assert ExtractionJob.objects.filter(pk=job.pk).count() == 1
    assert SharedKnowledge.objects.filter(project=project).count() == 0


def test_stale_worker_claim_recovers_saved_extraction_result(project, ana, make_message, monkeypatch):
    source = make_message(project, ana, "We observed a repeated scheduling issue.")
    job = ExtractionJob.objects.create(project=project, founder=ana, source_message=source)
    request = ai_context.build_extract_request(
        project=project, founder=ana, source_message=source,
    ).request
    cached = [_extract_response(request, source).candidates[0].model_dump(mode="json")]
    stale = timezone.now() - timedelta(minutes=20)
    ExtractionJob.objects.filter(pk=job.pk).update(
        status=ExtractionJobStatus.PROCESSING,
        attempt_count=1,
        claimed_at=stale,
        claim_token="00000000-0000-0000-0000-000000000001",
        extraction_result=cached,
    )
    monkeypatch.setattr(ai_adapter, "extract_knowledge",
                        lambda _request: (_ for _ in ()).throw(AssertionError("cached output should be reused")))

    claimed = claim_next_job(now=timezone.now(), lease_timeout=timedelta(minutes=10))
    assert claimed.pk == job.pk
    result = process_claimed_job(claimed)
    job.refresh_from_db()
    assert result.status == ExtractionJobStatus.AWAITING_SAFETY_VALIDATION
    assert job.attempt_count == 2
    assert job.extraction_result == cached


@pytest.mark.django_db(transaction=True)
def test_postgresql_workers_cannot_claim_the_same_job(project, ana, make_message):
    if connection.vendor != "postgresql":
        pytest.skip("SKIP LOCKED concurrency guarantee is PostgreSQL-specific.")
    source = make_message(project, ana, "A message waiting for extraction.")
    job = ExtractionJob.objects.create(project=project, founder=ana, source_message=source)
    barrier = Barrier(2)

    def claim():
        close_old_connections()
        try:
            barrier.wait(timeout=5)
            claimed = claim_next_job()
            return claimed.pk if claimed else None
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _index: claim(), range(2)))

    assert claims.count(job.pk) == 1
    assert claims.count(None) == 1
