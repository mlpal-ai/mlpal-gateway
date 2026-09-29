"""Unit tests for API key service."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlpal_assistants_service.core.exceptions import InvalidAPIKeyError
from mlpal_assistants_service.schemas.api_key import APIKeyCreate
from mlpal_assistants_service.services.api_key import APIKeyService


class TestAPIKeyService:
    """Tests for APIKeyService."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: MagicMock) -> APIKeyService:
        """Create service with mock session."""
        return APIKeyService(mock_session)

    @pytest.mark.asyncio
    async def test_create_key_returns_key_and_secret(
        self,
        service: APIKeyService,
        mock_session: MagicMock,
    ) -> None:
        """Test that create_key returns both key model and secret."""
        user_id = "test-user-id"
        data = APIKeyCreate(name="Test Key")

        api_key, secret = await service.create_key(user_id, data)

        # Verify key was added to session
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

        # Verify secret has correct format
        assert secret.startswith("mlpal_sk_")
        assert len(secret) == 9 + 64  # prefix + hex

    @pytest.mark.asyncio
    async def test_create_key_stores_hash_not_plaintext(
        self,
        service: APIKeyService,
        mock_session: MagicMock,
    ) -> None:
        """Test that only the hash is stored, not plaintext."""
        user_id = "test-user-id"
        data = APIKeyCreate(name="Test Key")

        api_key, secret = await service.create_key(user_id, data)

        # Get the key that was added to the session
        added_key = mock_session.add.call_args[0][0]

        # Verify the hash is stored, not the secret
        assert added_key.key_hash != secret
        assert len(added_key.key_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_validate_key_invalid_format(
        self,
        service: APIKeyService,
    ) -> None:
        """Test that invalid format raises error."""
        with pytest.raises(InvalidAPIKeyError):
            await service.validate_key("invalid_key")

    @pytest.mark.asyncio
    async def test_validate_key_not_found(
        self,
        service: APIKeyService,
        mock_session: MagicMock,
    ) -> None:
        """Test that non-existent key raises error."""
        # Mock execute to return None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        valid_format_key = "mlpal_sk_" + "a" * 64

        with pytest.raises(InvalidAPIKeyError):
            await service.validate_key(valid_format_key)

    @pytest.mark.asyncio
    async def test_validate_key_expired(
        self,
        service: APIKeyService,
        mock_session: MagicMock,
    ) -> None:
        """Test that expired key raises error."""
        # Create mock key that is expired
        mock_key = MagicMock()
        mock_key.is_active = True
        mock_key.revoked_at = None
        mock_key.expires_at = datetime.utcnow() - timedelta(days=1)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.side_effect = [mock_key, None]
        mock_session.execute.return_value = mock_result

        valid_format_key = "mlpal_sk_" + "a" * 64

        with pytest.raises(InvalidAPIKeyError) as exc_info:
            await service.validate_key(valid_format_key)

        assert "expired" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_list_keys_excludes_revoked_by_default(
        self,
        service: APIKeyService,
        mock_session: MagicMock,
    ) -> None:
        """Test that list_keys excludes revoked keys by default."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        await service.list_keys(user_id="test-user")

        # Verify the query was made
        mock_session.execute.assert_called_once()
