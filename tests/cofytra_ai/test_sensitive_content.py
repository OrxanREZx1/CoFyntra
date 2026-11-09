"""
Offline regression tests for the F3 sensitive-content guard. All "sensitive" values are fake test values
or published examples (e.g. the IBAN registry example DE89..., test card 4111...).
"""

from __future__ import annotations

import json
import logging

import pytest

from cofytra_ai import extraction, service
from cofytra_ai.errors import CofytraAIError
from cofytra_ai.extraction_guards import describe, guard_candidates, sensitive_kinds
from tests.cofytra_ai.test_extraction import NO_DROPS, _cand, _drops, _llm, _request, _run, _settings

POSITIVES = [
    # credentials
    ("credential", "Our OpenAI key is sk-proj-AbCdEf1234567890XyZ"),
    ("credential", "Stripe uses sk_live_51HxYzAbCdEf12 in prod"),
    ("credential", "AWS key AKIAIOSFODNN7EXAMPLE is in the repo"),
    ("credential", "token ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
    ("credential", "Authorization: Bearer abcDEF1234567890xyz"),
    ("credential", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"),
    ("credential", "the admin password is Hunter2!"),
    ("credential", "pwd: s3cr3t!!"),
    ("credential", "api_key=9f8e7d6c5b4a3210"),
    ("credential", "db at postgres://admin:pa55word@db.example.com/prod"),
    # contact details
    ("email", "Contact dr.weber@clinic-berlin.de for the pilot"),
    ("phone", "Dr. Weber's mobile is +49 151 2345678"),
    ("phone", "call (555) 123-4567 tomorrow"),
    ("phone", "her number is 555-123-4567"),
    ("phone", "reach the clinic at 0151 2345678"),
    # payment / bank
    ("payment_card", "charge 4111 1111 1111 1111 for the pilot"),
    ("payment_card", "card 5500-0000-0000-0004 expires soon"),
    ("bank_account", "wire it to DE89 3704 0044 0532 0130 00"),
    ("bank_account", "IBAN GB82WEST12345698765432"),
    ("bank_account", "account number: 12345678"),
    ("bank_account", "sort code 12-34-56"),
    # confidentiality markers
    ("confidential", "This is confidential: Ben may leave the company"),
    ("confidential", "Please don't share this with Ben, but the pilot failed"),
    ("confidential", "Off the record, LabCorp is unhappy"),
    ("confidential", "Keep this between us: we might pivot"),
    ("confidential", "We are under NDA with the hospital group"),
    ("confidential", "Confidential: revenue dropped"),
]

SAFE = [
    "There are about 12,000 diagnostic labs in the EU.",
    "5 of 8 clinic managers say scheduling takes 3 hours a day.",
    "Clinics would pay 50 EUR a month for something better.",
    "We target $1,200,000 ARR by 2027 with 1,000,000 potential users.",
    "Pricing: 49 EUR/month or 490 EUR/year, 20% discount for annual plans.",
    "Growth of +20% month over month, now 300 users and +15 new clinics.",
    "Patients care about confidential handling of their health data.",
    "We need a password reset flow and SSO for clinics.",
    "Our API key management is a selling point for developers.",
    "Pilot ran from 2026-11-09 to 2026-12-01 with 3 clinics.",
    "Call volume rose from 1200 to 4500 calls per week.",
    "Token-based pricing at 0.002 USD per 1000 tokens.",
    "We are ISO 27001 and SOC 2 Type II certified.",
    "Labs process 10000 samples per day; ticket #12345 tracks the bug.",
    "Clinics account for 30% of revenue; version 2.3.1 shipped.",
    "A 15-minute demo at 3pm convinced 2 of 5 lab managers.",
    "Contact form conversion is 4.5% on the landing page.",
    "The team is three founders with 120 hours per week.",
]


@pytest.mark.parametrize("kind,text", POSITIVES)
def test_detects_sensitive_content(kind, text):
    assert kind in sensitive_kinds(text)


@pytest.mark.parametrize("text", SAFE)
def test_ordinary_startup_text_is_not_flagged(text):
    assert sensitive_kinds(text) == set()


def test_luhn_and_iban_checksums_prevent_false_positives():
    assert "payment_card" not in sensitive_kinds("order 4111 1111 1111 1112 shipped")  # fails Luhn
    assert "bank_account" not in sensitive_kinds("code DE89 3704 0044 0532 0130 01")    # fails mod-97


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------

EMAIL_MSG = "Dr. Weber agreed to a two-week pilot. You can reach her at dr.weber@clinic-berlin.de any time."
SAFE_CAND = _cand("evidence", "Dr. Weber agreed to a two-week pilot.", "founder_observation", ["msg-1"],
                  "Dr. Weber agreed to a two-week pilot.")
LEAKY_CAND = _cand("evidence", "Dr. Weber agreed to a pilot and can be reached by email.", "founder_observation", ["msg-1"],
                   "You can reach her at dr.weber@clinic-berlin.de any time.")


def test_sensitive_candidate_dropped_under_invalid_status_safe_one_kept():
    resp, _ = _run(_request(EMAIL_MSG), {"candidates": [SAFE_CAND, LEAKY_CAND]})
    assert [c.source_quote for c in resp.candidates] == ["Dr. Weber agreed to a two-week pilot."]
    assert _drops(resp) == {**NO_DROPS, "invalid_status": 1}
    assert "clinic-berlin" not in resp.model_dump_json()


def test_quotes_are_never_edited_to_remove_sensitive_values():
    # A "redacted" quote is not verbatim, so the normal quote rule rejects it; nothing is silently rewritten.
    redacted = dict(LEAKY_CAND, source_quote="You can reach her at [email] any time.")
    resp, _ = _run(_request(EMAIL_MSG), {"candidates": [redacted]})
    assert resp.candidates == [] and _drops(resp) == {**NO_DROPS, "quote_not_found": 1}


@pytest.mark.parametrize("field,value", [
    ("statement", "Pilot contact is dr.weber@clinic-berlin.de."),
    ("source_quote", "You can reach her at dr.weber@clinic-berlin.de any time."),
])
def test_sensitive_value_in_statement_or_quote_is_rejected(field, value):
    resp, _ = _run(_request(EMAIL_MSG), {"candidates": [dict(SAFE_CAND, **{field: value})]})
    assert resp.candidates == [] and _drops(resp)["invalid_status"] == 1


def test_sensitive_external_reference_is_rejected():
    msg = "Per the report sent by jane.doe@labmarket.io there are about 12,000 diagnostic labs in the EU."
    cand = _cand("market", "There are about 12,000 diagnostic labs in the EU.", "external_source", ["msg-1"],
                 "there are about 12,000 diagnostic labs", ref="jane.doe@labmarket.io")
    resp, _ = _run(_request(msg), {"candidates": [cand]})
    assert resp.candidates == [] and _drops(resp)["invalid_status"] == 1


def test_sensitive_check_runs_before_source_checks():
    leaky_unknown_source = dict(LEAKY_CAND, source_message_ids=["msg-999"])
    resp, _ = _run(_request(EMAIL_MSG), {"candidates": [leaky_unknown_source]})
    assert _drops(resp) == {**NO_DROPS, "invalid_status": 1}


def test_safe_startup_candidates_pass_untouched():
    msg = "I talked to 8 clinic managers. 5 said they would pay 49 EUR a month; there are about 12,000 labs in the EU."
    cand = _cand("evidence", "5 of 8 interviewed clinic managers said they would pay 49 EUR a month.",
                 "founder_observation", ["msg-1"], "5 said they would pay 49 EUR a month")
    resp, _ = _run(_request(msg), {"candidates": [cand]})
    assert len(resp.candidates) == 1 and _drops(resp) == NO_DROPS


@pytest.mark.parametrize("kind,text", POSITIVES[:1] + POSITIVES[6:7] + POSITIVES[10:12] + POSITIVES[15:16]
                         + POSITIVES[17:18] + POSITIVES[21:22])
def test_every_kind_is_rejected_end_to_end(kind, text):
    cand = _cand("problem", "Clinics struggle with scheduling.", "founder_observation", ["msg-1"], text)
    outcome = guard_candidates(_request(text), [cand])
    assert outcome.candidates == [] and outcome.dropped.invalid_status == 1
    assert outcome.corrections["sensitive_content"] == 1 and outcome.corrections[f"sensitive:{kind}"] == 1


# ---------------------------------------------------------------------------
# No leakage in logs, errors or diagnostics
# ---------------------------------------------------------------------------

SECRETS = ("dr.weber@clinic-berlin.de", "sk-proj-AbCdEf1234567890XyZ", "Hunter2!", "+49 151 2345678")
LEAKY_TEXT = ("Dr. Weber agreed to a pilot, mail dr.weber@clinic-berlin.de or call +49 151 2345678. "
              "Our key is sk-proj-AbCdEf1234567890XyZ and the admin password is Hunter2!")


def _leaky_candidates() -> list[dict]:
    return [_cand("evidence", "Pilot agreed.", "founder_observation", ["msg-1"], q) for q in (
        "mail dr.weber@clinic-berlin.de or call +49 151 2345678",
        "Our key is sk-proj-AbCdEf1234567890XyZ",
        "the admin password is Hunter2!",
    )]


def test_rejection_never_leaks_values_into_logs_or_diagnostics(caplog):
    caplog.set_level(logging.DEBUG)
    resp, _ = _run(_request(LEAKY_TEXT), {"candidates": _leaky_candidates()})
    assert resp.candidates == [] and _drops(resp)["invalid_status"] == 3
    diagnostics = json.dumps(describe(guard_candidates(_request(LEAKY_TEXT), _leaky_candidates())))
    for secret in SECRETS:
        assert secret not in caplog.text and secret not in resp.model_dump_json() and secret not in diagnostics


def test_provider_and_validation_errors_never_leak_values(caplog):
    caplog.set_level(logging.DEBUG)
    bad = {"candidates": [{"category": "nonsense", "statement": LEAKY_TEXT[:400]}]}  # malformed twice
    scripted = _llm(bad, bad)
    with pytest.raises(CofytraAIError) as exc_info:
        service.extract_knowledge(_request(LEAKY_TEXT), settings=_settings(), llm_factory=scripted.get_llm_client)
    err = exc_info.value
    for secret in SECRETS:
        for text in (caplog.text, str(err), err.info.model_dump_json()):
            assert secret not in text
    assert err.__cause__ is None


def test_prompt_rule_and_version():
    assert "SENSITIVE CONTENT" in extraction.SYSTEM_PROMPT
    for phrase in ("passwords, API keys, tokens", "email addresses, phone numbers", "payment card or bank",
                   "marks as confidential", "do not extract it"):
        assert phrase in extraction.SYSTEM_PROMPT
    assert extraction.PROMPT_VERSION == "cofytra.extract.v2"
