"""Archive existing outputs for version 1 compatibility launches."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def archive_paths(paths: list[Path]) -> list[str]:
    """Rename non-empty files with one timestamp, without overwriting."""
    archived: list[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            destination = path.with_name(f"{path.name}.{timestamp}")
            if destination.exists():
                counter = 2
                while destination.exists():
                    destination = path.with_name(
                        f"{path.name}.{timestamp}.{counter}"
                    )
                    counter += 1
            path.rename(destination)
            archived.append(str(destination))
    return archived


def archive_previous_outputs(job_dir: str, job_name: str) -> list[str]:
    """Archive the legacy output, error, and scheduler script paths."""
    return archive_paths([
        Path(job_dir) / f"{job_name}{extension}"
        for extension in (".out", ".err", ".job")
    ])


__all__ = ["archive_paths", "archive_previous_outputs"]
