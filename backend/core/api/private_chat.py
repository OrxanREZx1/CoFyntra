import logging

from django.db import IntegrityError, transaction
from django.shortcuts import get_object_or_404
from rest_framework import serializers
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from cofytra_ai.models import MAX_ID_CHARS, MAX_MESSAGE_CHARS
from core.models import (
    ExtractionJob,
    PrivateMessage,
    PrivateMessageRole,
    Project,
    ProjectMembership,
    SharedKnowledge,
)
from core.services import ai_adapter, ai_context
from core.services.ai_adapter import AIAdapterError

logger = logging.getLogger("cofytra.api.private_chat")
_CLIENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
_PAGE_SIZE_MAX = 100


class PrivateMessageInputSerializer(serializers.Serializer):
    content = serializers.CharField(min_length=1, max_length=MAX_MESSAGE_CHARS, trim_whitespace=True)
    client_message_id = serializers.RegexField(
        _CLIENT_ID_PATTERN,
        max_length=MAX_ID_CHARS,
        trim_whitespace=True,
    )

    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError("Expected a JSON object.")
        extra = set(data) - {"content", "client_message_id"}
        if extra:
            raise serializers.ValidationError({key: ["Unknown field."] for key in sorted(extra)})
        return super().to_internal_value(data)


class PageQuerySerializer(serializers.Serializer):
    page = serializers.IntegerField(min_value=1, default=1)
    page_size = serializers.IntegerField(min_value=1, max_value=_PAGE_SIZE_MAX, default=20)

    def to_internal_value(self, data):
        extra = set(data) - {"page", "page_size"}
        if extra:
            raise serializers.ValidationError({key: ["Unknown parameter."] for key in sorted(extra)})
        return super().to_internal_value(data)


def _require_project_member(project, user):
    if not ProjectMembership.objects.filter(project=project, user=user, is_active=True).exists():
        raise PermissionDenied("Not an active founder of this project.")


def _message_payload(message: PrivateMessage) -> dict:
    job = getattr(message, "extraction_job", None) if message.role == PrivateMessageRole.FOUNDER else None
    return {
        "id": str(message.pk),
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
        "ai_response": message.ai_response,
        "extraction_status": job.status if job else None,
    }


def _get_or_create_founder_message(*, project, founder, content, client_message_id):
    existing = PrivateMessage.objects.filter(
        project=project,
        owner=founder,
        role=PrivateMessageRole.FOUNDER,
        client_message_id=client_message_id,
    ).first()
    if existing is not None:
        if existing.content != content:
            return existing, None, False
        job, _ = ExtractionJob.objects.get_or_create(
            source_message=existing,
            defaults={"project": project, "founder": founder},
        )
        return existing, job, False

    try:
        with transaction.atomic():
            message = PrivateMessage.objects.create(
                project=project,
                owner=founder,
                role=PrivateMessageRole.FOUNDER,
                content=content,
                client_message_id=client_message_id,
            )
            job = ExtractionJob.objects.create(project=project, founder=founder, source_message=message)
        return message, job, True
    except IntegrityError:
        existing = PrivateMessage.objects.filter(
            project=project,
            owner=founder,
            role=PrivateMessageRole.FOUNDER,
            client_message_id=client_message_id,
        ).first()
        if existing is None:
            raise
        if existing.content != content:
            return existing, None, False
        job, _ = ExtractionJob.objects.get_or_create(
            source_message=existing,
            defaults={"project": project, "founder": founder},
        )
        return existing, job, False


def _existing_reply(message):
    return PrivateMessage.objects.filter(
        reply_to=message,
        role=PrivateMessageRole.ASSISTANT,
    ).first()


def _save_reply(*, project, founder, founder_message, response):
    response_data = response.model_dump(mode="json")
    try:
        with transaction.atomic():
            return PrivateMessage.objects.create(
                project=project,
                owner=founder,
                role=PrivateMessageRole.ASSISTANT,
                content=response.reply,
                reply_to=founder_message,
                ai_response=response_data,
            )
    except IntegrityError:
        reply = _existing_reply(founder_message)
        if reply is None:
            raise
        return reply


def _chat_response(founder_message, reply, job, *, status_code):
    response_data = reply.ai_response or {}
    meta = response_data.get("meta", {})
    return Response(
        {
            "founder_message_id": str(founder_message.pk),
            "assistant_message_id": str(reply.pk),
            "message": _message_payload(founder_message),
            "reply": _message_payload(reply),
            "request_id": meta.get("request_id"),
            "response_meta": meta,
            "chat_status": "succeeded",
            "extraction": {"job_id": job.pk, "status": job.status},
        },
        status=status_code,
    )


class PrivateMessagesView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id):
        project = get_object_or_404(Project, pk=project_id)
        _require_project_member(project, request.user)
        query = PageQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)
        page = query.validated_data["page"]
        page_size = query.validated_data["page_size"]
        rows = (PrivateMessage.objects.filter(project=project, owner=request.user)
                .select_related("extraction_job").order_by("created_at", "pk"))
        start = (page - 1) * page_size
        results = list(rows[start:start + page_size])
        return Response({
            "results": [_message_payload(message) for message in results],
            "page": page,
            "page_size": page_size,
            "has_next": rows.count() > start + len(results),
        })

    def post(self, request, project_id):
        project = get_object_or_404(Project, pk=project_id)
        _require_project_member(project, request.user)
        serializer = PrivateMessageInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        founder_message, job, created = _get_or_create_founder_message(
            project=project,
            founder=request.user,
            content=serializer.validated_data["content"],
            client_message_id=serializer.validated_data["client_message_id"],
        )
        if job is None:
            return Response(
                {"error": {"code": "idempotency_key_reused",
                           "message": "This client_message_id was already used for a different message.",
                           "retryable": False}},
                status=409,
            )

        reply = _existing_reply(founder_message)
        if reply is not None:
            return _chat_response(founder_message, reply, job, status_code=200)

        try:
            built = ai_context.build_chat_request(
                project=project,
                founder=request.user,
                message=founder_message,
            )
            ai_response = ai_adapter.chat(built.request)
        except AIAdapterError as exc:
            logger.warning("private_chat_failed project_id=%s founder_message_id=%s error_code=%s",
                           project.pk, founder_message.pk, exc.code.value)
            return Response(
                {
                    "founder_message_id": str(founder_message.pk),
                    "assistant_message_id": None,
                    "message": _message_payload(founder_message),
                    "reply": None,
                    "request_id": exc.request_id,
                    "response_meta": None,
                    "chat_status": "failed",
                    "retryable": exc.retryable,
                    "error": {"code": exc.code.value, "message": exc.message},
                    "extraction": {"job_id": job.pk, "status": job.status},
                },
                status=exc.http_status,
            )

        reply = _save_reply(
            project=project,
            founder=request.user,
            founder_message=founder_message,
            response=ai_response,
        )
        return _chat_response(founder_message, reply, job, status_code=201 if created else 200)


class SharedKnowledgeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id):
        project = get_object_or_404(Project, pk=project_id)
        _require_project_member(project, request.user)
        query = PageQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)
        page = query.validated_data["page"]
        page_size = query.validated_data["page_size"]
        rows = (SharedKnowledge.objects.filter(project=project)
                .select_related("author").order_by("shared_at", "pk"))
        start = (page - 1) * page_size
        items = list(rows[start:start + page_size])
        return Response({
            "results": [
                {
                    "id": str(item.pk),
                    "author": {
                        "id": str(item.author_id),
                        "display_name": (item.author.get_full_name() or item.author.get_username()).strip(),
                    },
                    "category": item.category,
                    "statement": item.statement,
                    "epistemic_status": item.epistemic_status,
                    "source_quote": item.source_quote,
                    "external_reference": item.external_reference,
                    "shared_at": item.shared_at,
                }
                for item in items
            ],
            "page": page,
            "page_size": page_size,
            "has_next": rows.count() > start + len(items),
        })
