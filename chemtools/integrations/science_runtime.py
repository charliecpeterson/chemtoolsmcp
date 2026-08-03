"""Read-only probe for the optional companion scientific Python runtime."""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCIENCE_RUNTIME_PYTHON_ENV = "CHEMTOOLS_SCIENCE_PYTHON"
SCIENCE_RUNTIME_PROBE_SCHEMA = "chemtools.science-runtime-probe/1"
MAX_SCIENCE_RUNTIME_PROBE_BYTES = 64 * 1024
MAX_SCIENCE_RUNTIME_RESULT_BYTES = 256 * 1024
_PROBE_SENTINEL = "CHEMTOOLS_SCIENCE_PROBE="
_RUNNER_RESULT_SENTINEL = "CHEMTOOLS_SCIENCE_RESULT="
_PROBE_SCRIPT = f"""
import contextlib
import importlib
import importlib.metadata
import io
import json
import sys

_PACKAGES = {{
    "pyscf": ("pyscf", "pyscf"),
    "rdkit": ("rdkit", "rdkit"),
    "openbabel": ("openbabel", "openbabel"),
    "h5py": ("h5py", "h5py"),
    "basis_set_exchange": ("basis_set_exchange", "basis_set_exchange"),
    "orbitron": ("orbitron", "orbitron"),
}}

def _package_probe(module_name, distribution_name):
    suppressed_stdout = io.StringIO()
    suppressed_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(suppressed_stdout), contextlib.redirect_stderr(suppressed_stderr):
            module = importlib.import_module(module_name)
    except Exception as error:
        return {{
            "status": "unavailable",
            "error": f"{{type(error).__name__}}: {{str(error)[:512]}}",
        }}
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
    return {{
        "status": "available",
        "version": str(version) if version is not None else None,
    }}

payload = {{
    "schema_version": "{SCIENCE_RUNTIME_PROBE_SCHEMA}",
    "python": {{
        "executable": sys.executable,
        "implementation": sys.implementation.name,
        "version": sys.version.split()[0],
    }},
    "packages": {{
        name: _package_probe(module_name, distribution_name)
        for name, (module_name, distribution_name) in _PACKAGES.items()
    }},
}}
print("{_PROBE_SENTINEL}" + json.dumps(payload, sort_keys=True))
"""


class ScienceRuntimeError(RuntimeError):
    """Base error for the optional companion scientific runtime."""


class ScienceRuntimeUnavailableError(ScienceRuntimeError):
    """The configured companion interpreter cannot be used."""


class ScienceRuntimeProtocolError(ScienceRuntimeError):
    """The companion runtime returned an invalid probe response."""


class ScienceRuntimeCommandError(ScienceRuntimeError):
    """The fixed companion-runtime probe did not complete successfully."""

    def __init__(
        self,
        message: str,
        *,
        argv: tuple[str, ...],
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class ScienceRuntimeProbe:
    python: dict[str, str]
    packages: dict[str, dict[str, str | None]]


def resolve_science_runtime_python(
    executable: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the explicitly configured companion interpreter."""
    configured = executable
    origin = "explicit companion Python interpreter"
    if configured is None:
        configured = os.environ.get(SCIENCE_RUNTIME_PYTHON_ENV)
        origin = SCIENCE_RUNTIME_PYTHON_ENV
    if not configured:
        raise ScienceRuntimeUnavailableError(
            "companion scientific runtime is unavailable; set "
            f"{SCIENCE_RUNTIME_PYTHON_ENV} to its Python interpreter"
        )

    raw = os.path.expanduser(os.fspath(configured))
    if os.sep not in raw:
        raise ScienceRuntimeUnavailableError(
            f"{origin} must be an explicit interpreter path: {raw}"
        )
    path = Path(raw).resolve()
    if not path.is_file():
        raise ScienceRuntimeUnavailableError(f"{origin} is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise ScienceRuntimeUnavailableError(f"{origin} is not executable: {path}")
    return path


class ScienceRuntimeClient:
    """Run fixed, read-only operations in the companion runtime."""

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        *,
        timeout_seconds: float = 15.0,
        max_probe_bytes: int = MAX_SCIENCE_RUNTIME_PROBE_BYTES,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if isinstance(max_probe_bytes, bool) or not isinstance(max_probe_bytes, int):
            raise ValueError("max_probe_bytes must be a positive integer")
        if max_probe_bytes <= 0:
            raise ValueError("max_probe_bytes must be a positive integer")
        self.executable = resolve_science_runtime_python(executable)
        self.timeout_seconds = timeout_seconds
        self.max_probe_bytes = max_probe_bytes

    def probe(self) -> ScienceRuntimeProbe:
        argv = (str(self.executable), "-c", _PROBE_SCRIPT)
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except OSError as error:
            raise ScienceRuntimeCommandError(
                f"could not run companion scientific runtime probe: {error}",
                argv=argv,
            ) from error
        except subprocess.TimeoutExpired as error:
            raise ScienceRuntimeCommandError(
                "companion scientific runtime probe timed out",
                argv=argv,
                stderr=_bounded_text(error.stderr),
            ) from error

        if completed.returncode != 0:
            raise ScienceRuntimeCommandError(
                "companion scientific runtime probe failed",
                argv=argv,
                returncode=completed.returncode,
                stderr=_bounded_text(completed.stderr),
            )
        if len(completed.stdout.encode("utf-8")) > self.max_probe_bytes:
            raise ScienceRuntimeProtocolError(
                "companion scientific runtime probe exceeded the response limit"
            )

        sentinel_lines = [
            line[len(_PROBE_SENTINEL):]
            for line in completed.stdout.splitlines()
            if line.startswith(_PROBE_SENTINEL)
        ]
        if len(sentinel_lines) != 1:
            raise ScienceRuntimeProtocolError(
                "companion scientific runtime probe returned no unique JSON response"
            )
        try:
            payload = json.loads(sentinel_lines[0])
        except json.JSONDecodeError as error:
            raise ScienceRuntimeProtocolError(
                "companion scientific runtime probe returned invalid JSON"
            ) from error
        return _parse_probe_payload(payload)

    def rdkit_preflight(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._runner_operation("rdkit-preflight", request)

    def openbabel_convert(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._runner_operation("openbabel-convert", request)

    def orbitron_periodic_electronic_structure(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._runner_operation("orbitron-periodic-electronic-structure", request)

    def orbitron_structure_identity(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._runner_operation("orbitron-structure-identity", request)

    def orbitron_nbo(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._runner_operation("orbitron-nbo", request)

    def qmcpack_hdf5_inspect(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._runner_operation("qmcpack-hdf5-inspect", request)

    def bse_render(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._runner_operation("bse-render", request)

    def _runner_operation(
        self,
        operation: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        if operation not in {
            "rdkit-preflight",
            "openbabel-convert",
            "orbitron-periodic-electronic-structure",
            "orbitron-structure-identity",
            "orbitron-nbo",
            "qmcpack-hdf5-inspect",
            "bse-render",
        }:
            raise ValueError(f"unsupported read-only companion operation: {operation}")
        if not isinstance(request, dict):
            raise TypeError("companion request must be an object")
        argv = (
            str(self.executable),
            str(_science_runner_path()),
            operation,
        )
        try:
            completed = subprocess.run(
                list(argv),
                input=json.dumps(request, sort_keys=True),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except OSError as error:
            raise ScienceRuntimeCommandError(
                f"could not run companion {operation}: {error}",
                argv=argv,
            ) from error
        except subprocess.TimeoutExpired as error:
            raise ScienceRuntimeCommandError(
                f"companion {operation} timed out",
                argv=argv,
                stderr=_bounded_text(error.stderr),
            ) from error
        if completed.returncode != 0:
            raise ScienceRuntimeCommandError(
                f"companion {operation} failed",
                argv=argv,
                returncode=completed.returncode,
                stderr=_bounded_text(completed.stderr),
            )
        return parse_science_runner_output(completed.stdout)


def science_runner_path() -> Path:
    """Return the fixed runner bundled with the installed Chemtools package."""
    return _science_runner_path()


def parse_science_runner_output(stdout: str) -> dict[str, Any]:
    if len(stdout.encode("utf-8")) > MAX_SCIENCE_RUNTIME_RESULT_BYTES:
        raise ScienceRuntimeProtocolError(
            "companion scientific runtime result exceeded the response limit"
        )
    sentinel_lines = [
        line[len(_RUNNER_RESULT_SENTINEL):]
        for line in stdout.splitlines()
        if line.startswith(_RUNNER_RESULT_SENTINEL)
    ]
    if len(sentinel_lines) != 1:
        raise ScienceRuntimeProtocolError(
            "companion scientific runtime returned no unique JSON result"
        )
    try:
        payload = json.loads(sentinel_lines[0])
    except json.JSONDecodeError as error:
        raise ScienceRuntimeProtocolError(
            "companion scientific runtime returned invalid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ScienceRuntimeProtocolError(
            "companion scientific runtime result must be an object"
        )
    schema = payload.get("schema_version")
    if not isinstance(schema, str) or not schema.startswith("chemtools."):
        raise ScienceRuntimeProtocolError(
            "companion scientific runtime result has an invalid schema version"
        )
    return payload


def _science_runner_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "science_runner.py"
    if not path.is_file():
        raise ScienceRuntimeUnavailableError(
            f"Chemtools science runner is unavailable: {path}"
        )
    return path


def _parse_probe_payload(payload: Any) -> ScienceRuntimeProbe:
    if not isinstance(payload, dict):
        raise ScienceRuntimeProtocolError("companion probe payload must be an object")
    if payload.get("schema_version") != SCIENCE_RUNTIME_PROBE_SCHEMA:
        raise ScienceRuntimeProtocolError(
            "companion probe returned an unsupported schema version"
        )
    python = payload.get("python")
    if not isinstance(python, dict) or set(python) != {
        "executable",
        "implementation",
        "version",
    }:
        raise ScienceRuntimeProtocolError("companion probe has invalid Python metadata")
    if not all(isinstance(value, str) and value for value in python.values()):
        raise ScienceRuntimeProtocolError("companion probe has invalid Python values")

    packages = payload.get("packages")
    expected_packages = {
        "pyscf",
        "rdkit",
        "openbabel",
        "h5py",
        "basis_set_exchange",
        "orbitron",
    }
    if not isinstance(packages, dict) or set(packages) != expected_packages:
        raise ScienceRuntimeProtocolError("companion probe has invalid package metadata")
    normalized_packages: dict[str, dict[str, str | None]] = {}
    for name, package in packages.items():
        if not isinstance(package, dict):
            raise ScienceRuntimeProtocolError(
                f"companion probe package {name!r} must be an object"
            )
        status = package.get("status")
        if status == "available":
            if set(package) != {"status", "version"}:
                raise ScienceRuntimeProtocolError(
                    f"companion probe package {name!r} has invalid available fields"
                )
            version = package["version"]
            if version is not None and not isinstance(version, str):
                raise ScienceRuntimeProtocolError(
                    f"companion probe package {name!r} has invalid version"
                )
            normalized_packages[name] = {"status": status, "version": version}
        elif status == "unavailable":
            if set(package) != {"status", "error"}:
                raise ScienceRuntimeProtocolError(
                    f"companion probe package {name!r} has invalid unavailable fields"
                )
            error = package["error"]
            if not isinstance(error, str) or not error:
                raise ScienceRuntimeProtocolError(
                    f"companion probe package {name!r} has invalid error"
                )
            normalized_packages[name] = {"status": status, "error": error}
        else:
            raise ScienceRuntimeProtocolError(
                f"companion probe package {name!r} has invalid status"
            )
    return ScienceRuntimeProbe(python=dict(python), packages=normalized_packages)


def _bounded_text(value: str | bytes | None, limit: int = 4_096) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[:limit]
