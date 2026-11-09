"""DRF exception handler: consistent, content-free error bodies for AI and validation failures."""

from __future__ import annotations

from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

from core.services.ai_adapter import AIAdapterError


def exception_handler(exc, context):
    if isinstance(exc, AIAdapterError):
        return Response(exc.public_payload(), status=exc.http_status)
    if isinstance(exc, DjangoValidationError):
        fields = exc.message_dict if hasattr(exc, "error_dict") else {"non_field_errors": exc.messages}
        return Response({"error": {"code": "validation_error", "message": "The request is invalid.",
                                   "retryable": False, "fields": fields}}, status=400)
    return drf_exception_handler(exc, context)
