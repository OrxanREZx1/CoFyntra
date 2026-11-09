"""
Tests for token budget recovery, error classification, and non-identical retry behavior.
Milestone M4.1.4.1.
"""

from unittest.mock import MagicMock, patch
import pytest

from startary.config import Settings
from startary.cli import ErrorCategory, classify_cli_error
from startary.graph.nodes import _invoke_structured_with_token_retry
from startary.llm.provider import TokenBudgetExhaustedError, is_token_budget_exhausted_error


class MockLengthError(Exception):
    """Simulates an OpenAI / LangChain LengthFinishReasonError."""
    pass


def test_is_token_budget_exhausted_error():
    assert is_token_budget_exhausted_error(TokenBudgetExhaustedError("out of tokens"))
    assert is_token_budget_exhausted_error(MockLengthError("length limit was reached"))
    assert is_token_budget_exhausted_error(RuntimeError("finish_reason='length'"))
    assert is_token_budget_exhausted_error(Exception("hit max_completion_tokens"))
    assert not is_token_budget_exhausted_error(ValueError("invalid format"))
    assert not is_token_budget_exhausted_error(None)


def test_invoke_success_no_retry():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = {"status": "ok"}

    settings = Settings()
    res, latency = _invoke_structured_with_token_retry(
        structured_llm=mock_llm,
        messages=[{"role": "user", "content": "hi"}],
        settings=settings,
        tier="reasoning",
        base_max_tokens=2500,
        node_name="critique_experiment",
        schema_cls=dict,
    )

    assert res == {"status": "ok"}
    assert mock_llm.invoke.call_count == 1


@patch("startary.graph.nodes._safe_get_llm_client")
def test_invoke_length_error_recovers_with_expanded_budget(mock_get_client):
    # Attempt 1 fails with length error
    mock_llm1 = MagicMock()
    mock_llm1.invoke.side_effect = MockLengthError("length limit was reached")

    # Attempt 2 (expanded) succeeds
    mock_llm2 = MagicMock()
    mock_structured2 = MagicMock()
    mock_structured2.invoke.return_value = {"status": "recovered"}
    mock_llm2.with_structured_output.return_value = mock_structured2
    mock_get_client.return_value = mock_llm2

    settings = Settings()
    res, latency = _invoke_structured_with_token_retry(
        structured_llm=mock_llm1,
        messages=[{"role": "user", "content": "hi"}],
        settings=settings,
        tier="reasoning",
        base_max_tokens=1000,
        node_name="critique_experiment",
        schema_cls=dict,
    )

    assert res == {"status": "recovered"}
    # Verify retry called with expanded budget: max(1000 * 1.6 = 1600, 1000 + 1500 = 2500) = 2500
    mock_get_client.assert_called_once_with(
        settings,
        tier="reasoning",
        max_completion_tokens=2500,
        reasoning_effort="low",
    )
    assert mock_structured2.invoke.call_count == 1


@patch("startary.graph.nodes._safe_get_llm_client")
def test_invoke_length_error_recovers_with_expanded_budget_from_2500(mock_get_client):
    # Attempt 1 fails with length error on 2500 token budget
    mock_llm1 = MagicMock()
    mock_llm1.invoke.side_effect = MockLengthError("length limit was reached")

    # Attempt 2 (expanded to 4000) succeeds
    mock_llm2 = MagicMock()
    mock_structured2 = MagicMock()
    mock_structured2.invoke.return_value = {"status": "recovered"}
    mock_llm2.with_structured_output.return_value = mock_structured2
    mock_get_client.return_value = mock_llm2

    settings = Settings(fast_model_name="gpt-5.6-luna")
    res, latency = _invoke_structured_with_token_retry(
        structured_llm=mock_llm1,
        messages=[{"role": "user", "content": "hi"}],
        settings=settings,
        tier="fast",
        base_max_tokens=2500,
        node_name="extract_startup_state",
        schema_cls=dict,
    )

    assert res == {"status": "recovered"}
    # Verify retry called with expanded budget: max(2500 * 1.6 = 4000, 2500 + 1500 = 4000) = 4000
    mock_get_client.assert_called_once_with(
        settings,
        tier="fast",
        max_completion_tokens=4000,
        reasoning_effort="low",
    )
    assert mock_structured2.invoke.call_count == 1


@patch("startary.graph.nodes._safe_get_llm_client")
def test_invoke_length_error_retry_exhausted_aborts_without_loop(mock_get_client):
    # Attempt 1 fails with length error
    mock_llm1 = MagicMock()
    mock_llm1.invoke.side_effect = MockLengthError("length limit was reached")

    # Attempt 2 also fails with length error
    mock_llm2 = MagicMock()
    mock_structured2 = MagicMock()
    mock_structured2.invoke.side_effect = MockLengthError("length limit was reached again")
    mock_llm2.with_structured_output.return_value = mock_structured2
    mock_get_client.return_value = mock_llm2

    settings = Settings()
    with pytest.raises(TokenBudgetExhaustedError) as exc_info:
        _invoke_structured_with_token_retry(
            structured_llm=mock_llm1,
            messages=[{"role": "user", "content": "hi"}],
            settings=settings,
            tier="reasoning",
            base_max_tokens=1000,
            node_name="critique_experiment",
            schema_cls=dict,
        )

    assert "token budget exhausted even after expanding cap" in str(exc_info.value)
    # Exactly one retry occurred; no infinite retry loops
    assert mock_structured2.invoke.call_count == 1


def test_cli_error_classification_token_budget():
    token_err = TokenBudgetExhaustedError("completion tokens exhausted")
    cat, title, guidance = classify_cli_error(token_err)
    assert cat == ErrorCategory.TOKEN_BUDGET_EXHAUSTED
    assert "Token Budget Exhausted" in title

    length_err = MockLengthError("length limit was reached")
    cat, title, guidance = classify_cli_error(length_err)
    assert cat == ErrorCategory.TOKEN_BUDGET_EXHAUSTED
    assert "Token Budget Exhausted" in title

