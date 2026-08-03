"""Run a small NWChem job through the typed local or Slurm boundary.

The script copies a committed H2 input into an isolated work directory and
writes machine-readable evidence without recording environment values.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
from typing import Any
from uuid import uuid4

from chemtools.application.execution import ExecutionService
from chemtools.application.nwchem_execution import (
    launch_nwchem_with_service,
    register_nwchem_launch_with_service,
)
from chemtools.application.nwchem_monitoring import (
    watch_nwchem_status_with_service,
)
from chemtools.execution.legacy_profiles import (
    load_runner_profiles,
    resolve_runner_profile,
)
from chemtools.programs.nwchem.strategy.diagnose import parse_scf


ROOT = Path(__file__).resolve().parents[1]
SMOKE_INPUT = ROOT / "examples" / "smoke" / "nwchem_h2.nw"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _run_directory(work_root: Path) -> Path:
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_directory = work_root / f"{timestamp}-{uuid4().hex[:8]}"
    run_directory.mkdir(parents=True)
    return run_directory


def _profile_executor(
    profiles_path: Path,
    profile_name: str,
) -> str:
    profiles = load_runner_profiles(str(profiles_path))
    profile = resolve_runner_profile(profiles, profile_name)
    kind = (profile.get("launcher") or {}).get("kind", "direct")
    if kind == "direct":
        return "local"
    if kind == "scheduler":
        scheduler = profile.get("scheduler") or {}
        launcher = profile.get("launcher") or {}
        scheduler_type = (
            scheduler.get("system")
            or launcher.get("scheduler_type")
            or "slurm"
        )
        if str(scheduler_type).lower() != "slurm":
            raise ValueError(
                "the typed smoke runner supports local and Slurm profiles"
            )
        return "slurm"
    raise ValueError(f"unsupported launcher kind: {kind!r}")


def run_smoke(arguments: argparse.Namespace) -> tuple[Path, bool]:
    started_at = _utc_now()
    profiles_path = arguments.profiles_path.expanduser().resolve()
    work_root = arguments.work_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    run_directory = _run_directory(work_root)
    evidence_path = run_directory / "evidence.json"
    input_path = run_directory / SMOKE_INPUT.name
    shutil.copy2(SMOKE_INPUT, input_path)

    executor = _profile_executor(profiles_path, arguments.profile)
    if executor != arguments.expect_executor:
        raise ValueError(
            f"profile {arguments.profile!r} resolves to {executor!r}, "
            f"expected {arguments.expect_executor!r}"
        )

    service = ExecutionService(
        enable_execution=True,
        registry_db_path=run_directory / "registry.sqlite",
    )
    launch_id = None
    cancellation = None
    evidence: dict[str, Any] = {
        "schema_version": "chemtools.execution-smoke/1",
        "started_at": started_at.isoformat(),
        "completed_at": None,
        "host": platform.node(),
        "profile": {
            "name": arguments.profile,
            "path": str(profiles_path),
            "sha256": _sha256(profiles_path),
            "executor": executor,
        },
        "input": {
            "source": str(SMOKE_INPUT),
            "path": str(input_path),
            "sha256": _sha256(input_path),
        },
        "request": {
            "nodes": 1,
            "mpi_ranks": arguments.mpi_ranks,
            "omp_threads": 1,
            "walltime": arguments.walltime,
            "timeout_seconds": arguments.timeout_seconds,
        },
        "success": False,
    }

    try:
        resource_overrides = {
            "nodes": 1,
            "mpi_ranks": arguments.mpi_ranks,
            "omp_threads": 1,
        }
        if executor == "slurm":
            resource_overrides["walltime"] = arguments.walltime
        launched = launch_nwchem_with_service(
            service,
            input_path=str(input_path),
            profile=arguments.profile,
            profiles_path=str(profiles_path),
            resource_overrides=resource_overrides,
        )
        launch_id = launched["launch_id"]
        registration = register_nwchem_launch_with_service(
            service,
            launch_id=launch_id,
            job_name=launched["job_name"],
            input_file=str(input_path),
            profile=arguments.profile,
        )
        watched = watch_nwchem_status_with_service(
            service,
            process_id=launched.get("process_id"),
            job_id=launched.get("job_id"),
            profile=arguments.profile if executor == "slurm" else None,
            output_path=launched["output_file"],
            input_path=str(input_path),
            error_path=launched["error_file"],
            profiles_path=str(profiles_path),
            poll_interval_seconds=arguments.poll_interval_seconds,
            adaptive_polling=False,
            timeout_seconds=arguments.timeout_seconds,
            stall_timeout_seconds=None,
        )
        record = service.get_launch_record(launch_id)
        output_path = Path(launched["output_file"])
        error_path = Path(launched["error_file"])
        scf = (
            parse_scf(str(output_path))
            if output_path.is_file()
            else {
                "status": "missing",
                "total_energy_hartree": None,
                "iteration_count": 0,
            }
        )
        success = (
            watched["terminal"] is True
            and record.status == "completed"
            and record.return_code == 0
            and scf["status"] == "converged"
            and scf["total_energy_hartree"] is not None
        )
        evidence.update({
            "launch": {
                "launch_id": launch_id,
                "target": record.target,
                "effective_argv": list(record.argv),
                "process_id": record.process_id,
                "job_id": record.job_id,
            },
            "registration": {
                "run_uid": registration["run_uid"],
                "run_id": registration["run_id"],
            },
            "watch": {
                "terminal": watched["terminal"],
                "stop_reason": watched["stop_reason"],
                "poll_count": watched["poll_count"],
                "overall_status": watched["overall_status"],
            },
            "execution_record": {
                "status": record.status,
                "return_code": record.return_code,
                "elapsed_seconds": record.elapsed_seconds,
            },
            "scientific_check": {
                "scf_status": scf["status"],
                "total_energy_hartree": scf[
                    "total_energy_hartree"
                ],
                "iteration_count": scf["iteration_count"],
            },
            "artifacts": {
                "stdout": _artifact_evidence(output_path),
                "stderr": _artifact_evidence(error_path),
            },
            "success": success,
        })
    except Exception as exc:
        evidence["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        success = False
    finally:
        if launch_id is not None:
            record = service.get_launch_record(launch_id)
            if record.status in {"started", "submitted", "cancel_failed"}:
                try:
                    cancelled = service.cancel_recorded(launch_id)
                    cancellation = {
                        "status": cancelled.record.status,
                        "error": cancelled.record.error,
                    }
                except Exception as exc:
                    cancellation = {
                        "status": "cancel_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        if cancellation is not None:
            evidence["cleanup"] = cancellation
        evidence["completed_at"] = _utc_now().isoformat()
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return evidence_path, success


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the committed H2 case through a configured local or "
            "Slurm NWChem target."
        )
    )
    parser.add_argument(
        "--profiles-path",
        type=Path,
        required=True,
        help="version 1 runner-profile YAML or JSON file",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="profile name to execute",
    )
    parser.add_argument(
        "--expect-executor",
        choices=("local", "slurm"),
        required=True,
        help="refuse to run if the profile resolves to another executor",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="parent directory for the isolated timestamped run",
    )
    parser.add_argument("--mpi-ranks", type=int, default=2)
    parser.add_argument("--walltime", default="00:05:00")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
    )
    return parser


def main() -> int:
    evidence_path, success = run_smoke(_parser().parse_args())
    print(evidence_path)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
