"""Alembic environment configuration."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.db.models import Base

# Alembic Config object
config = context.config

# Setup logging
if config.config_file_name is not None:
    # disable_existing_loggers=False: the app runs migrations in-process at startup;
    # the default (True) silently disables every logger created before this
    # point — all of the service's own stdlib loggers.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Model metadata for autogenerate
target_metadata = Base.metadata

# Get database URL from settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url_sync)


def include_object(object, name, type_, reflected, compare_to):
    """Only include objects in our schema."""
    if type_ == "table":
        # Only include tables in assistants schema
        return object.schema == settings.db_schema
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=settings.db_schema,
        include_schemas=True,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Create schema if it doesn't exist (needed for alembic_version table)
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.db_schema}"))
        connection.commit()

        # Set search_path to assistants schema
        connection.execute(text(f"SET search_path TO {settings.db_schema}, public"))

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=settings.db_schema,
            include_schemas=True,
            include_object=include_object,
            transaction_per_migration=True,
        )

        with context.begin_transaction():
            context.run_migrations()

        # Ensure changes are committed
        connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
