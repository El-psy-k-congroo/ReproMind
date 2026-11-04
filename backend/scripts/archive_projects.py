"""Hide selected projects from lists while preserving their records and files.

Run from backend with the same configuration as the application:
  python scripts/archive_projects.py hide --ids-file ids.json --snapshot saved.json
  python scripts/archive_projects.py restore --snapshot saved.json

The IDs file is a JSON array of project UUIDs. Keep snapshots outside Git.
"""

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, update

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import AsyncSessionLocal, close_database, engine
from app.models import Project


def read_ids(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("项目编号清单必须是非空 JSON 数组")
    return list(dict.fromkeys(str(UUID(value)) for value in values))


async def apply_visibility(args: argparse.Namespace) -> None:
    fingerprint = hashlib.sha256(str(engine.url).encode()).hexdigest()
    if args.action == "hide":
        project_ids = read_ids(args.ids_file)
        previous = dict.fromkeys(project_ids, "active")
        expected = {"active"}
    else:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        if snapshot.get("version") != 1 or snapshot.get("database") != fingerprint:
            raise ValueError("恢复清单版本或数据库不匹配")
        records = snapshot.get("projects")
        if not isinstance(records, list) or not records:
            raise ValueError("恢复清单中没有项目")
        previous = {str(UUID(item["id"])): item["status"] for item in records}
        if any(value != "active" for value in previous.values()):
            raise ValueError("恢复清单包含不支持的原始状态")
        project_ids = list(previous)
        expected = {"archived", "active"}

    async with AsyncSessionLocal() as session:
        async with session.begin():
            rows = (await session.execute(
                select(Project.id, Project.status).where(Project.id.in_(project_ids))
            )).all()
            if len(rows) != len(project_ids):
                raise ValueError("部分项目不存在，未修改任何项目")
            if any(row.status not in expected for row in rows):
                raise ValueError("项目状态已变化，未修改任何项目")
            if args.action == "hide":
                snapshot = {
                    "version": 1,
                    "database": fingerprint,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "projects": [{"id": row.id, "status": row.status} for row in rows],
                }
                # Exclusive creation preserves earlier recovery records.
                with args.snapshot.open("x", encoding="utf-8") as output:
                    json.dump(snapshot, output, ensure_ascii=False, indent=2)
                result = await session.execute(
                    update(Project).where(Project.id.in_(project_ids), Project.status == "active")
                    .values(status="archived", updated_at=Project.updated_at)
                )
            else:
                result = await session.execute(
                    update(Project).where(Project.id.in_(project_ids), Project.status == "archived")
                    .values(status="active", updated_at=Project.updated_at)
                )
        print(f"{args.action}: {result.rowcount}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    hide = commands.add_parser("hide", help="隐藏指定项目，保留可恢复记录")
    hide.add_argument("--ids-file", type=Path, required=True)
    hide.add_argument("--snapshot", type=Path, required=True)
    restore = commands.add_parser("restore", help="恢复清单中的隐藏项目")
    restore.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    try:
        await apply_visibility(args)
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
