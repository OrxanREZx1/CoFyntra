"""The single exception type the Cofytra AI service raises to Django."""

from __future__ import annotations

from typing import Optional

from pydantic import ValidationError

from cofytra_ai.models import ERROR_RETRYABLE, SUGGESTED_HTTP_STATUS, ErrorCode, ErrorInfo, FieldIssue


class CofytraAIError(Exception):
    """
    Raised by every ``cofytra_ai.service`` function on failure. Never carries founder text or model
    output. Django catches it and maps ``info.code`` to an HTTP response (``suggested_http_status``).
    """

    def __init__(self, code: ErrorCode, message: str, request_id: Optional[str] = None,
                 details: Optional[list[FieldIssue]] = None):
        self.info = ErrorInfo(code=code, message=message, retryable=ERROR_RETRYABLE[code],
                              request_id=request_id, details=details or [])
        super().__init__(f"{code.value}: {message}")

    @property
    def code(self) -> ErrorCode:
        return self.info.code

    @property
    def retryable(self) -> bool:
        return self.info.retryable

    @property
    def suggested_http_status(self) -> int:
        return SUGGESTED_HTTP_STATUS[self.info.code]

    @classmethod
    def from_validation_error(cls, exc: ValidationError, request_id: Optional[str] = None) -> "CofytraAIError":
        """Field locations and error types only; input values are never copied into the error."""
        details = [
            FieldIssue(field=(".".join(str(p) for p in e["loc"]) or "<root>")[:300], issue=e["type"][:300])
            for e in exc.errors(include_input=False, include_url=False)[:50]
        ]
        return cls(ErrorCode.INVALID_REQUEST, "Request failed Cofytra AI contract validation.", request_id, details)
