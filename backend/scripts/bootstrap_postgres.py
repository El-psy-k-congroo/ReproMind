"""创建 ReproMind 独立数据库并初始化业务表。

示例：
python scripts/bootstrap_postgres.py \
  --admin-url postgresql://USER:PASSWORD@127.0.0.1:15432/eduagent
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--admin-url",
        default=os.getenv("ADMIN_DATABASE_URL", ""),
        help="可创建数据库的 PostgreSQL 连接串",
    )
    parser.add_argument("--database", default="repromind")
    return parser.parse_args()


def ensure_database(admin_url: str, database: str) -> None:
    if not admin_url:
        raise SystemExit("缺少 --admin-url 或 ADMIN_DATABASE_URL")
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,62}", database):
        raise SystemExit("数据库名不合法")

    with psycopg.connect(admin_url, autocommit=True, connect_timeout=8) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (database,)
        ).fetchone()
        if not exists:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
            print(f"created database: {database}", flush=True)
        else:
            print(f"database already exists: {database}", flush=True)


async def create_schema(database_url: str) -> None:
    os.environ["DATABASE_URL"] = database_url
    from app import models  # noqa: F401
    from app.database import close_database, init_database

    await init_database()
    await close_database()


def main() -> None:
    args = parse_args()
    ensure_database(args.admin_url, args.database)
    parsed = make_url(args.admin_url)
    target = parsed.set(
        drivername="postgresql+asyncpg",
        database=args.database,
        query={},
    )
    asyncio.run(create_schema(target.render_as_string(hide_password=False)))
    print(f"schema ready: {args.database}", flush=True)


if __name__ == "__main__":
    main()
