"""Per-key model-policy boundary enforcement.

Contract (services/policy.py): a key's model_policy gates access on BOTH the
requested tag and the meta-resolved concrete model. Deny wins over allow; an
empty/``["*"]`` allow admits everything. A router alias (e.g. ``mlpal``) must
not be a way to reach a denied concrete model, because deny is evaluated
against the resolved tag as well as the requested one.

These assert at the service layer on purpose: the live gateway returns 503 when
no provider is configured, which short-circuits *before* policy runs and masks
the check end to end. Testing the policy engine directly is the only way to pin
the boundary without a provider credential. Covers ``is_routing_allowed``, the
static listing-side mirror of the request-time ``check_model_access`` gate.
"""

from __future__ import annotations

from mlpal_assistants_service.services.policy import PolicyService

# Restricted key: exactly one Anthropic model allowed, nothing else.
RESTRICTED = {"allow": ["claude-haiku-4-5-20251001"], "deny": []}
# Broad allow, but one model explicitly denied — exercises deny-wins.
DENY_WINS = {"allow": ["claude-*"], "deny": ["claude-opus-5"]}


def test_allowed_model_passes():
    """The one model on the allow-list is admitted when requested directly."""
    assert PolicyService.is_routing_allowed(
        RESTRICTED, "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"
    )


def test_model_absent_from_allowlist_is_blocked():
    """A model not matching a non-wildcard allow-list is refused. (Blocked by
    allow-miss, not by deny — this is the allow-list doing its job.)"""
    assert not PolicyService.is_routing_allowed(RESTRICTED, "claude-opus-5", "claude-opus-5")


def test_deny_on_resolved_blocks_an_allowed_alias():
    """THE core boundary: the alias is allow-matched, so only a deny evaluated
    against the RESOLVED model can block it. claude-sonnet-5 matches allow
    'claude-*', but resolves to claude-opus-5 which is denied -> blocked. If
    this returned True, an allowed alias could smuggle you to a denied model."""
    assert not PolicyService.is_routing_allowed(DENY_WINS, "claude-sonnet-5", "claude-opus-5")


def test_deny_wins_over_allow_same_tag():
    """Deny takes precedence even when the allow-glob matches the same tag."""
    assert not PolicyService.is_routing_allowed(DENY_WINS, "claude-opus-5", "claude-opus-5")


def test_wildcard_allow_admits_anything():
    """``['*']`` allow is the unrestricted default."""
    assert PolicyService.is_routing_allowed({"allow": ["*"], "deny": []}, "mlpal", "gpt-6-sol")


def test_no_policy_is_unrestricted():
    """A null/empty policy admits everything (default for a key with no
    model_policy set)."""
    assert PolicyService.is_routing_allowed(None, "mlpal", "claude-opus-5")
