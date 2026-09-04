"""Model Pricing - stores pricing information for compute unit calculation."""

from datetime import date
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Computed,
    Date,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from mlpal_assistants_service.db.models.base import Base, TimestampMixin


class ModelPricing(Base, TimestampMixin):
    """
    Model pricing table.

    Stores pricing information used to calculate compute units.
    Supports different rates for input vs output tokens and
    effective dates for price changes.
    """

    __tablename__ = "model_pricing"
    __table_args__ = (
        UniqueConstraint("model_tag", "operation", "effective_date", name="uq_pricing"),
        Index("idx_pricing_model_op", "model_tag", "operation"),
        Index("idx_pricing_effective", "effective_date"),
        {"schema": "assistants"},
    )

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Model identification
    model_tag: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    # Operation type (chat, embedding, image_gen, transcribe, tts)
    operation: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    # Pricing tier (budget, standard, premium)
    tier: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Provider cost (what we pay)
    # Rate per unit (e.g., per 1K tokens, per image, per minute)
    input_rate: Mapped[Decimal] = mapped_column(
        Numeric(12, 8),
        nullable=False,
    )

    output_rate: Mapped[Decimal] = mapped_column(
        Numeric(12, 8),
        nullable=False,
    )

    # Per-model cache-read list price (same unit as input_rate). NULL = the
    # provider prices cache reads at the standard multiple of input
    # (settings.cache_read_multiplier, 0.10x); set only when a model breaks
    # the rule — e.g. claude-fable-5-1 at $0.25/MTok (0.025x of input).
    cache_read_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 8),
        nullable=True,
    )

    # Rate unit description
    rate_unit: Mapped[str] = mapped_column(
        String(50),
        default="per_1k_tokens",
        nullable=False,
    )

    # Markup multiplier (default 3x)
    markup_multiplier: Mapped[Decimal] = mapped_column(
        Numeric(4, 2),
        default=Decimal("3.0"),
        nullable=False,
    )

    # Effective date (for price changes)
    effective_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # CU conversion rate (1 CU = $10 by default)
    cu_to_dollar: Mapped[Decimal] = mapped_column(
        Numeric(6, 2),
        default=Decimal("10.0"),
        nullable=False,
        comment="Dollar value of 1 CU (default: 10.0)",
    )

    # Postgres GENERATED ALWAYS AS ... STORED columns. Declaring Computed makes
    # SQLAlchemy exclude them from INSERT/UPDATE (writing them raises
    # GeneratedAlwaysError) while still reading them back after flush.
    input_cu_rate: Mapped[Decimal] = mapped_column(
        Numeric(12, 8),
        Computed("(input_rate * markup_multiplier) / cu_to_dollar", persisted=True),
        comment="CU per unit for input (generated)",
    )

    output_cu_rate: Mapped[Decimal] = mapped_column(
        Numeric(12, 8),
        Computed("(output_rate * markup_multiplier) / cu_to_dollar", persisted=True),
        comment="CU per unit for output (generated)",
    )

    def __repr__(self) -> str:
        return f"<ModelPricing(model={self.model_tag}, op={self.operation}, effective={self.effective_date})>"

    def calculate_cu(
        self,
        input_units: int | Decimal = 0,
        output_units: int | Decimal = 0,
    ) -> Decimal:
        """
        Calculate compute units for a request.

        Uses pre-computed CU rates (input_cu_rate, output_cu_rate) from the database.
        These are generated columns: (dollar_rate * markup) / cu_to_dollar

        Args:
            input_units: Number of input units (tokens, characters, images, minutes)
            output_units: Number of output units (tokens, characters, images, minutes)

        Returns:
            Compute units (1 CU = $10 by default)

        Rate units and their divisors:
            - per_1m_tokens: divide by 1,000,000
            - per_1k_tokens: divide by 1,000
            - per_1m_characters: divide by 1,000,000
            - per_image: no division (rate is per image)
            - per_minute: no division (rate is per minute)
        """
        input_units = Decimal(str(input_units))
        output_units = Decimal(str(output_units))

        # Determine divisor based on rate_unit
        if self.rate_unit == "per_1m_tokens":
            divisor = Decimal("1000000")
        elif self.rate_unit == "per_1k_tokens":
            divisor = Decimal("1000")
        elif self.rate_unit == "per_1m_characters":
            divisor = Decimal("1000000")
        elif self.rate_unit in ("per_image", "per_minute"):
            divisor = Decimal("1")  # Rate is already per unit
        else:
            # Default to per_1m_tokens for safety
            divisor = Decimal("1000000")

        # Use pre-computed CU rates from generated columns
        input_cu = input_units * self.input_cu_rate / divisor
        output_cu = output_units * self.output_cu_rate / divisor

        return input_cu + output_cu

    def calculate_provider_cost(
        self,
        input_units: int | Decimal = 0,
        output_units: int | Decimal = 0,
    ) -> Decimal:
        """Calculate raw provider cost without markup."""
        return self.calculate_cu(input_units, output_units) / self.markup_multiplier
