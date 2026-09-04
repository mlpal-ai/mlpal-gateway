"""SQLAlchemy models for the assistants schema.

Note: Users are managed in a separate schema (configured via MLPAL_USER_SCHEMA).
The api_keys and usage_logs tables reference user_id as an integer FK to the
external users table, but without database-level FK constraints for schema flexibility.
"""

from mlpal_assistants_service.db.models.api_key import APIKey
from mlpal_assistants_service.db.models.base import Base
from mlpal_assistants_service.db.models.connections import TenantConnection, TenantModel
from mlpal_assistants_service.db.models.feed import FeedInstall, GatewayMeta
from mlpal_assistants_service.db.models.image_job import ImageJob
from mlpal_assistants_service.db.models.meta_routing import MetaModelRouting
from mlpal_assistants_service.db.models.model_feedback import (
    FEEDBACK_OUTCOMES,
    ModelFeedback,
)
from mlpal_assistants_service.db.models.model_pricing import ModelPricing
from mlpal_assistants_service.db.models.model_registry import ModelRegistry
from mlpal_assistants_service.db.models.request_payload import RequestPayload
from mlpal_assistants_service.db.models.usage_log import UsageLog
from mlpal_assistants_service.db.models.user_billing_status import (
    BillingStatus,
    UserBillingStatus,
)
from mlpal_assistants_service.db.models.user_credits import (
    FREE_CREDITS_BY_ROLE,
    CreditType,
    UserCredits,
)

__all__ = [
    "TenantConnection",
    "TenantModel",
    "FeedInstall",
    "GatewayMeta",
    "ImageJob",
    "Base",
    "APIKey",
    "MetaModelRouting",
    "ModelFeedback",
    "RequestPayload",
    "FEEDBACK_OUTCOMES",
    "ModelRegistry",
    "ModelPricing",
    "UsageLog",
    "UserBillingStatus",
    "BillingStatus",
    "UserCredits",
    "CreditType",
    "FREE_CREDITS_BY_ROLE",
]
