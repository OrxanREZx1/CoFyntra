import uuid

from django.db import IntegrityError, transaction
from django.shortcuts import get_object_or_404
from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from cofytra_ai.models import ErrorCode, contract_violations
from core.models import CatchupCursor, Project
from core.services import ai_adapter, ai_context
from core.services.ai_adapter import AIAdapterError


class EmptyInputSerializer(serializers.Serializer):
    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError("Expected a JSON object.")
        if data:
            raise serializers.ValidationError({key: ["Unknown field."] for key in sorted(data)})
        return {}


def _advance_catchup_cursor(*, project, viewer, other_founder, covered_at, knowledge_id):
    if covered_at is None or knowledge_id is None:
        return
    knowledge_uuid = uuid.UUID(knowledge_id)
    lookup = {"project": project, "viewer": viewer, "other_founder_id": other_founder}

    with transaction.atomic():
        cursor = CatchupCursor.objects.select_for_update().filter(**lookup).first()
        if cursor is None:
            try:
                with transaction.atomic():
                    CatchupCursor.objects.create(
                        **lookup,
                        last_caught_up_at=covered_at,
                        last_knowledge_id=knowledge_uuid,
                    )
                return
            except IntegrityError:
                cursor = CatchupCursor.objects.select_for_update().get(**lookup)

        if covered_at < cursor.last_caught_up_at:
            return
        if (covered_at == cursor.last_caught_up_at and cursor.last_knowledge_id is not None
                and knowledge_uuid.int <= cursor.last_knowledge_id.int):
            return
        cursor.last_caught_up_at = covered_at
        cursor.last_knowledge_id = knowledge_uuid
        cursor.save(update_fields=["last_caught_up_at", "last_knowledge_id", "updated_at"])


def _check_contract_response(request, response, capability):
    if contract_violations(request, response):
        raise AIAdapterError(capability, ErrorCode.INVALID_MODEL_OUTPUT, request.request_id)


class CatchUpView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, project_id):
        project = get_object_or_404(Project, pk=project_id)
        serializer = EmptyInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        built = ai_context.build_catchup_request(project=project, viewer=request.user)
        result = ai_adapter.catch_up(built.request)
        _check_contract_response(built.request, result, "catch_up")

        other_founder = built.request.other_founder
        if built.report.catch_up_covered_knowledge_id is not None:
            _advance_catchup_cursor(
                project=project,
                viewer=request.user,
                other_founder=other_founder.id,
                covered_at=built.report.catch_up_covered_until,
                knowledge_id=built.report.catch_up_covered_knowledge_id,
            )

        return Response({
            "catch_up": result.model_dump(mode="json"),
            "pagination": {
                "has_more": not built.report.catch_up_complete,
                "batch_size": len(built.request.knowledge),
                "remaining_at_request": built.report.omitted.get("knowledge", 0),
            },
        })


class DetectConflictsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, project_id):
        project = get_object_or_404(Project, pk=project_id)
        serializer = EmptyInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        built = ai_context.build_conflict_request(project=project, requested_by=request.user)
        result = ai_adapter.detect_conflicts(built.request)
        _check_contract_response(built.request, result, "detect_conflicts")

        items = {item.id: item for item in built.request.knowledge}
        conflicts = []
        for conflict in result.conflicts:
            data = conflict.model_dump(mode="json")
            for side in ("claim_a", "claim_b"):
                item = items[data[side]["knowledge_id"]]
                data[side]["category"] = item.category.value
                data[side]["epistemic_status"] = item.epistemic_status.value
            conflicts.append(data)

        return Response({
            "conflicts": conflicts,
            "meta": result.meta.model_dump(mode="json"),
            "context": {
                "knowledge_count": len(built.request.knowledge),
                "omitted_knowledge": built.report.omitted.get("knowledge", 0),
            },
        })
