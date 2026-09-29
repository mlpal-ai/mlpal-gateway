"""API v1 router."""

from fastapi import APIRouter

from mlpal_assistants_service.api.v1.admin import router as admin_router
from mlpal_assistants_service.api.v1.audio import router as audio_router
from mlpal_assistants_service.api.v1.chat import router as chat_router
from mlpal_assistants_service.api.v1.embeddings import router as embeddings_router
from mlpal_assistants_service.api.v1.images import router as images_router
from mlpal_assistants_service.api.v1.internal import router as internal_router
from mlpal_assistants_service.api.v1.keys import router as keys_router
from mlpal_assistants_service.api.v1.models import router as models_router
from mlpal_assistants_service.api.v1.usage import router as usage_router

# NOTE: the messages router (Bedrock-mantle Anthropic passthrough) is NOT bundled
# here — it collides on /v1/messages with the universal core, so both are mounted
# at the composition root (api/mounting.py) by the messages seam.

router = APIRouter()

router.include_router(admin_router, prefix="/admin", tags=["Admin"])
router.include_router(chat_router, prefix="/chat", tags=["Chat"])
router.include_router(embeddings_router, prefix="/embeddings", tags=["Embeddings"])
router.include_router(images_router, prefix="/images", tags=["Images"])
router.include_router(audio_router, prefix="/audio", tags=["Audio"])
router.include_router(keys_router, prefix="/keys", tags=["API Keys"])
router.include_router(internal_router, prefix="/internal", tags=["Internal"])
router.include_router(models_router, prefix="/models", tags=["Models"])
router.include_router(usage_router, prefix="/usage", tags=["Usage"])
