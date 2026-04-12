#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
EXPORTS_DIR = ROOT / "skill-exports"


def slug_to_name(slug: str) -> str:
    return slug


def build_export(slug: str, content: str) -> list[dict[str, object]]:
    now = int(time.time())
    return [
        {
            "id": slug,
            "user_id": "",
            "name": slug_to_name(slug),
            "description": "",
            "meta": {"tags": ["chemistry", "openwebui"]},
            "is_active": True,
            "access_grants": [],
            "updated_at": now,
            "created_at": now,
            "user": None,
            "write_access": True,
            "content": content,
        }
    ]


def export_skill(skill_path: Path, output_path: Path) -> None:
    slug = skill_path.stem
    content = skill_path.read_text(encoding="utf-8")
    payload = build_export(slug, content)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Open WebUI skill-export JSON files from markdown skill drafts.")
    parser.add_argument(
        "--skill",
        action="append",
        help="Specific markdown skill file to export. Defaults to all files in openwebui/skills.",
    )
    args = parser.parse_args()

    if args.skill:
        skill_paths = [Path(path).resolve() for path in args.skill]
    else:
        skill_paths = sorted(SKILLS_DIR.glob("*.md"))

    for skill_path in skill_paths:
        output_path = EXPORTS_DIR / f"{skill_path.stem}.json"
        export_skill(skill_path, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
