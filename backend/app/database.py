from collections.abc import AsyncGenerator
from pathlib import Path
from sqlalchemy import inspect, text, event

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings


settings = get_settings()
database_url = settings.async_database_url
if database_url.startswith("sqlite"):
    db_path = database_url.split("///", 1)[-1]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine_options = {"pool_pre_ping": True, "echo": False}
if not database_url.startswith("sqlite"):
    engine_options.update({"pool_size": 10, "max_overflow": 20})
else:
    engine_options['connect_args'] = {'timeout': 30}
engine = create_async_engine(database_url, **engine_options)
if database_url.startswith('sqlite'):
    @event.listens_for(engine.sync_engine, 'connect')
    def _sqlite_connection(connection, _):
        cursor = connection.cursor()
        cursor.execute('PRAGMA busy_timeout=30000')
        cursor.close()
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_database() -> None:
    if database_url.startswith('sqlite'):
        # Long evidence/model reads must not block unrelated projects' progress writes.
        async with engine.connect() as connection:
            await connection.exec_driver_sql('PRAGMA journal_mode=WAL')
            await connection.commit()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"] for column in inspect(sync_connection).get_columns("reproduction_jobs")
            }
        )
        if "model_id" not in columns:
            await connection.execute(text(
                "ALTER TABLE reproduction_jobs ADD COLUMN model_id VARCHAR(128) "
                "DEFAULT 'deepseek-v4-flash-0731' NOT NULL"
            ))
        message_columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"] for column in inspect(sync_connection).get_columns("project_copilot_messages")
            }
        )
        if "conversation_id" not in message_columns:
            column_type = "UUID" if connection.dialect.name == "postgresql" else "CHAR(32)"
            await connection.execute(text(
                f"ALTER TABLE project_copilot_messages ADD COLUMN conversation_id {column_type}"
            ))


async def close_database() -> None:
    await engine.dispose()
