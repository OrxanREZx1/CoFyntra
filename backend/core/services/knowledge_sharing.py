"""
D3 automatic-sharing gate (fail closed).

D3 is final: validated extraction candidates are shared automatically, without founder confirmation.
It must not go live until the AI team's sensitive-content guard is integrated and tested. That guard is
not in this repository yet, and its interface is unknown, so it is NOT emulated here and there is no
setting that can bypass it. Until STEP 4 wires the delivered guard in, automatic sharing is refused.
"""

from __future__ import annotations


class AutoShareBlocked(Exception):
    """Raised when automatic sharing is attempted before every safety prerequisite is met."""


# Set by code (not configuration) only when the AI team's guard is integrated and covered by tests.
SENSITIVE_CONTENT_GUARD_INTEGRATED = False


def auto_share_status() -> tuple[bool, str]:
    if not SENSITIVE_CONTENT_GUARD_INTEGRATED:
        return False, "sensitive_content_guard_not_integrated"
    return True, "ok"


def ensure_auto_share_allowed() -> None:
    allowed, reason = auto_share_status()
    if not allowed:
        raise AutoShareBlocked(reason)
