"""Read-only subprocess boundary for Orbitron's versioned JSON commands."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chemtools.integrations.orbitron_analysis_validation import (
    OrbitronPayloadError,
    validate_geometry_provenance,
    validate_orbital_analysis,
    validate_population_analysis,
    validate_vibration_analysis,
)

ORBITRON_CLI_ENV = "CHEMTOOLS_ORBITRON_CLI"
MAX_ORBITRON_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
MAX_ORBITRON_JSON_BYTES = 2 * 1024 * 1024
MAX_ORBITRON_RENDER_BYTES = 8 * 1024 * 1024
_RENDER_WIDTH = 1024
_RENDER_HEIGHT = 768
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SUPPORTED_SCHEMAS = {
    "info": "orbitron.info/2",
    "inspect": "orbitron.inspect/2",
    "analyze_geometry": "orbitron.analyze.geometry/3",
    "analyze_orbitals": "orbitron.analyze.orbitals/2",
    "analyze_populations": "orbitron.analyze.populations/2",
    "analyze_vibrations": "orbitron.analyze.vibrations/4",
}
_OPERATION_ARGUMENTS = {
    "info": ("info",),
    "inspect": ("inspect",),
    "analyze_geometry": ("analyze", "geometry"),
    "analyze_orbitals": ("analyze", "orbitals"),
    "analyze_populations": ("analyze", "populations"),
    "analyze_vibrations": ("analyze", "vibrations"),
}
_OPERATION_TRAILING_ARGUMENTS = {
    "info": (),
    "inspect": (),
    "analyze_geometry": (),
    "analyze_orbitals": ("--frontier", "3"),
    "analyze_populations": ("--top", "8"),
    "analyze_vibrations": ("--top", "10"),
}
_VERSION_RE = re.compile(
    r"^orbitron-cli\s+(?P<version>\S+)\s+"
    r"\((?P<commit>[0-9a-fA-F]+(?:-dirty)?)\)$"
)


class OrbitronError(RuntimeError):
    """Base error for the optional Orbitron integration."""


class OrbitronUnavailableError(OrbitronError):
    """The configured Orbitron executable cannot be used."""


class OrbitronCommandError(OrbitronError):
    """Orbitron refused or failed a fixed Chemtools operation."""

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


class OrbitronProtocolError(OrbitronError):
    """Orbitron returned output outside the pinned machine contract."""


@dataclass(frozen=True)
class OrbitronVersion:
    version: str
    commit: str
    raw: str


@dataclass(frozen=True)
class OrbitronResponse:
    operation: str
    source: str
    schema: str
    producer: dict[str, str]
    warnings: tuple[dict[str, Any], ...]
    payload: dict[str, Any]
    stderr: str
    version: OrbitronVersion


@dataclass(frozen=True)
class OrbitronRender:
    source: str
    image: bytes
    width: int
    height: int
    stderr: str
    version: OrbitronVersion


def resolve_orbitron_cli(executable: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicit override, the environment setting, or PATH."""
    configured = executable
    origin = "explicit Orbitron executable"
    if configured is None:
        configured = os.environ.get(ORBITRON_CLI_ENV)
        origin = ORBITRON_CLI_ENV
    if configured is None:
        configured = shutil.which("orbitron")
        origin = "PATH"
    if not configured:
        raise OrbitronUnavailableError(
            f"Orbitron is unavailable; set {ORBITRON_CLI_ENV} or add orbitron to PATH"
        )

    raw = os.path.expanduser(os.fspath(configured))
    if os.sep not in raw:
        resolved_command = shutil.which(raw)
        if resolved_command is None:
            raise OrbitronUnavailableError(
                f"{origin} does not resolve to an executable: {raw}"
            )
        raw = resolved_command

    path = Path(raw).resolve()
    if not path.is_file():
        raise OrbitronUnavailableError(f"{origin} is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise OrbitronUnavailableError(f"{origin} is not executable: {path}")
    return path


class OrbitronClient:
    """Invoke pinned Orbitron operations with no persistent output."""

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_source_bytes: int = MAX_ORBITRON_SOURCE_BYTES,
        max_json_bytes: int = MAX_ORBITRON_JSON_BYTES,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        _validate_positive_limit("max_source_bytes", max_source_bytes)
        _validate_positive_limit("max_json_bytes", max_json_bytes)
        self.executable = resolve_orbitron_cli(executable)
        self.timeout_seconds = timeout_seconds
        self.max_source_bytes = max_source_bytes
        self.max_json_bytes = max_json_bytes
        self._version: OrbitronVersion | None = None

    def probe(self) -> OrbitronVersion:
        if self._version is not None:
            return self._version

        argv = (str(self.executable), "--version")
        completed = self._run(argv)
        first_line = (
            completed.stdout.splitlines()[0].strip()
            if completed.stdout
            else ""
        )
        match = _VERSION_RE.fullmatch(first_line)
        if match is None:
            raise OrbitronProtocolError(
                f"unrecognized Orbitron version response: {first_line!r}"
            )
        self._version = OrbitronVersion(
            version=match.group("version"),
            commit=match.group("commit").lower(),
            raw=completed.stdout.strip(),
        )
        return self._version

    def info(self, source: str | os.PathLike[str]) -> OrbitronResponse:
        return self._json_operation("info", source)

    def inspect(self, source: str | os.PathLike[str]) -> OrbitronResponse:
        return self._json_operation("inspect", source)

    def analyze_geometry(
        self,
        source: str | os.PathLike[str],
    ) -> OrbitronResponse:
        return self._json_operation("analyze_geometry", source)

    def analyze_orbitals(
        self,
        source: str | os.PathLike[str],
    ) -> OrbitronResponse:
        return self._json_operation("analyze_orbitals", source)

    def analyze_populations(
        self,
        source: str | os.PathLike[str],
    ) -> OrbitronResponse:
        return self._json_operation("analyze_populations", source)

    def analyze_vibrations(
        self,
        source: str | os.PathLike[str],
    ) -> OrbitronResponse:
        return self._json_operation("analyze_vibrations", source)

    def render(self, source: str | os.PathLike[str]) -> OrbitronRender:
        version = self.probe()
        source_path = self._source_path(source)
        try:
            with tempfile.TemporaryDirectory(
                prefix=".chemtools-orbitron-",
                dir=source_path.parent,
            ) as temporary_directory:
                image_path = Path(temporary_directory) / "render.png"
                completed = self._run((
                    str(self.executable),
                    "--quiet",
                    "--max-file-size",
                    str(self.max_source_bytes),
                    "render",
                    str(source_path),
                    "--output",
                    str(image_path),
                    "--width",
                    str(_RENDER_WIDTH),
                    "--height",
                    str(_RENDER_HEIGHT),
                ))
                image, width, height = _read_rendered_png(image_path)
        except OSError as error:
            raise OrbitronCommandError(
                f"could not create temporary Orbitron render output: {error}",
                argv=(),
            ) from error
        return OrbitronRender(
            source=str(source_path),
            image=image,
            width=width,
            height=height,
            stderr=completed.stderr,
            version=version,
        )

    def _json_operation(
        self,
        operation: str,
        source: str | os.PathLike[str],
    ) -> OrbitronResponse:
        version = self.probe()
        source_path = self._source_path(source)

        argv = (
            str(self.executable),
            "--quiet",
            "--max-file-size",
            str(self.max_source_bytes),
            *_OPERATION_ARGUMENTS[operation],
            str(source_path),
            "--json",
            *_OPERATION_TRAILING_ARGUMENTS[operation],
        )
        completed = self._run(argv)
        if len(completed.stdout.encode("utf-8")) > self.max_json_bytes:
            raise OrbitronProtocolError(
                f"Orbitron {operation} JSON exceeds the configured "
                f"Chemtools limit of {self.max_json_bytes} bytes"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise OrbitronProtocolError(
                f"Orbitron {operation} returned invalid JSON: {error.msg}"
            ) from error
        if not isinstance(payload, dict):
            raise OrbitronProtocolError(
                f"Orbitron {operation} returned {type(payload).__name__}, "
                "expected an object"
            )

        schema = payload.get("schema")
        expected_schema = _SUPPORTED_SCHEMAS[operation]
        if schema != expected_schema:
            raise OrbitronProtocolError(
                f"unsupported Orbitron {operation} schema {schema!r}; "
                f"expected {expected_schema!r}"
            )

        producer = payload.get("producer")
        if not isinstance(producer, dict):
            raise OrbitronProtocolError("Orbitron response has no producer object")
        if producer.get("name") != "orbitron":
            raise OrbitronProtocolError(
                f"unexpected Orbitron producer name: {producer.get('name')!r}"
            )
        if producer.get("version") != version.version:
            raise OrbitronProtocolError(
                "Orbitron producer version does not match the probed executable"
            )
        if str(producer.get("commit", "")).lower() != version.commit:
            raise OrbitronProtocolError(
                "Orbitron producer commit does not match the probed executable"
            )

        warnings = _validate_warnings(payload.get("warnings"))
        try:
            if operation == "analyze_geometry":
                _validate_geometry_analysis(payload, source_path)
            elif operation == "analyze_orbitals":
                validate_orbital_analysis(payload, source_path)
            elif operation == "analyze_populations":
                validate_population_analysis(payload, source_path)
            elif operation == "analyze_vibrations":
                validate_vibration_analysis(payload, source_path)
        except OrbitronPayloadError as error:
            raise OrbitronProtocolError(str(error)) from error
        return OrbitronResponse(
            operation=operation,
            source=str(source_path),
            schema=schema,
            producer={str(key): str(value) for key, value in producer.items()},
            warnings=warnings,
            payload=payload,
            stderr=completed.stderr,
            version=version,
        )

    def _source_path(self, source: str | os.PathLike[str]) -> Path:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise OrbitronCommandError(
                f"Orbitron source is not a file: {source_path}",
                argv=(),
            )
        source_size = source_path.stat().st_size
        if source_size > self.max_source_bytes:
            raise OrbitronCommandError(
                "Orbitron source exceeds the configured Chemtools limit of "
                f"{self.max_source_bytes} bytes: {source_path}",
                argv=(),
            )
        return source_path

    def _run(self, argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise OrbitronCommandError(
                f"Orbitron command timed out after {self.timeout_seconds:g} seconds",
                argv=argv,
                stderr=_timeout_text(error.stderr),
            ) from error
        except OSError as error:
            raise OrbitronUnavailableError(
                f"could not execute Orbitron at {self.executable}: {error}"
            ) from error

        if completed.returncode != 0:
            raise OrbitronCommandError(
                f"Orbitron command exited with status {completed.returncode}",
                argv=argv,
                returncode=completed.returncode,
                stderr=completed.stderr,
            )
        return completed


def _read_rendered_png(path: Path) -> tuple[bytes, int, int]:
    if not path.is_file():
        raise OrbitronProtocolError("Orbitron render did not create a PNG file")
    size_bytes = path.stat().st_size
    if size_bytes > MAX_ORBITRON_RENDER_BYTES:
        raise OrbitronProtocolError(
            "Orbitron render exceeds the configured Chemtools limit of "
            f"{MAX_ORBITRON_RENDER_BYTES} bytes"
        )
    image = path.read_bytes()
    if len(image) < 24:
        raise OrbitronProtocolError("Orbitron render PNG is truncated")
    if image[:8] != _PNG_SIGNATURE or image[12:16] != b"IHDR":
        raise OrbitronProtocolError("Orbitron render did not produce a PNG image")
    width = int.from_bytes(image[16:20], "big")
    height = int.from_bytes(image[20:24], "big")
    if (width, height) != (_RENDER_WIDTH, _RENDER_HEIGHT):
        raise OrbitronProtocolError(
            "Orbitron render dimensions do not match Chemtools' fixed "
            f"{_RENDER_WIDTH}x{_RENDER_HEIGHT} contract"
        )
    return image, width, height


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _validate_positive_limit(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_warnings(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise OrbitronProtocolError("Orbitron response warnings must be a list")
    warnings: list[dict[str, Any]] = []
    for index, warning in enumerate(value):
        if not isinstance(warning, dict):
            raise OrbitronProtocolError(
                f"Orbitron warning {index} must be an object"
            )
        if warning.get("source") not in {"loader", "cli", "analysis"}:
            raise OrbitronProtocolError(
                f"Orbitron warning {index} has an invalid source"
            )
        if (
            not isinstance(warning.get("code"), str)
            or not warning["code"].strip()
        ):
            raise OrbitronProtocolError(
                f"Orbitron warning {index} has no non-empty code"
            )
        if "message" in warning and not isinstance(warning["message"], str):
            raise OrbitronProtocolError(
                f"Orbitron warning {index} message must be a string"
            )
        warnings.append(dict(warning))
    return tuple(warnings)


def _validate_geometry_analysis(
    payload: dict[str, Any],
    source_path: Path,
) -> None:
    if payload.get("path") != str(source_path):
        raise OrbitronProtocolError(
            "Orbitron geometry analysis path does not match the source"
        )
    output_format = payload.get("format")
    if output_format is not None and not isinstance(output_format, str):
        raise OrbitronProtocolError(
            "Orbitron geometry analysis format must be a string or null"
        )
    if payload.get("distance_unit") != "angstrom":
        raise OrbitronProtocolError(
            "Orbitron geometry analysis distance_unit must be angstrom"
        )
    try:
        validate_geometry_provenance(payload, "Orbitron geometry analysis")
    except OrbitronPayloadError as error:
        raise OrbitronProtocolError(str(error)) from error

    atoms = _nonnegative_int(payload, "atoms", "Orbitron geometry analysis")
    bonds = _nonnegative_int(payload, "bonds", "Orbitron geometry analysis")
    dangling_atoms = _nonnegative_int(
        payload,
        "dangling_atoms",
        "Orbitron geometry analysis",
    )
    if dangling_atoms > atoms:
        raise OrbitronProtocolError(
            "Orbitron geometry analysis has more dangling atoms than atoms"
        )

    elements = _count_map(payload, "elements")
    if sum(elements.values()) != atoms:
        raise OrbitronProtocolError(
            "Orbitron geometry analysis element counts do not equal atoms"
        )
    coordination = _count_map(payload, "coordination")
    if sum(coordination.values()) != atoms:
        raise OrbitronProtocolError(
            "Orbitron geometry analysis coordination counts do not equal atoms"
        )

    bounding_box = payload.get("bounding_box")
    if not isinstance(bounding_box, dict):
        raise OrbitronProtocolError(
            "Orbitron geometry analysis bounding_box must be an object"
        )
    minimum = _finite_vector(bounding_box, "min")
    maximum = _finite_vector(bounding_box, "max")
    if any(low > high for low, high in zip(minimum, maximum)):
        raise OrbitronProtocolError(
            "Orbitron geometry analysis bounding_box has min above max"
        )
    _finite_vector(payload, "center")
    span = _finite_vector(payload, "span")
    if any(value < 0 for value in span):
        raise OrbitronProtocolError(
            "Orbitron geometry analysis span must be non-negative"
        )

    _validate_bond_lengths(payload.get("bond_lengths"), bonds)
    _validate_unit_cell(payload.get("unit_cell"))


def _nonnegative_int(
    mapping: dict[str, Any],
    field_name: str,
    contract: str,
) -> int:
    value = mapping.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OrbitronProtocolError(
            f"{contract} {field_name} must be a non-negative integer"
        )
    return value


def _count_map(
    payload: dict[str, Any],
    field_name: str,
) -> dict[str, int]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise OrbitronProtocolError(
            f"Orbitron geometry analysis {field_name} must be an object"
        )
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise OrbitronProtocolError(
                f"Orbitron geometry analysis {field_name} has an invalid count"
            )
    return value


def _finite_vector(
    payload: dict[str, Any],
    field_name: str,
) -> tuple[float, float, float]:
    value = payload.get(field_name)
    if not isinstance(value, list) or len(value) != 3:
        raise OrbitronProtocolError(
            f"Orbitron geometry analysis {field_name} must have three values"
        )
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value
    ):
        raise OrbitronProtocolError(
            f"Orbitron geometry analysis {field_name} must be finite"
        )
    return tuple(float(item) for item in value)


def _validate_bond_lengths(value: object, bonds: int) -> None:
    if value is None:
        if bonds != 0:
            raise OrbitronProtocolError(
                "Orbitron geometry analysis omitted nonzero bond lengths"
            )
        return
    if not isinstance(value, dict):
        raise OrbitronProtocolError(
            "Orbitron geometry analysis bond_lengths must be an object or null"
        )
    if _nonnegative_int(
        value,
        "count",
        "Orbitron geometry analysis bond_lengths",
    ) != bonds:
        raise OrbitronProtocolError(
            "Orbitron geometry analysis bond-length count does not equal bonds"
        )
    statistics = []
    for field_name in ("min", "max", "mean", "std_dev"):
        statistic = value.get(field_name)
        if (
            isinstance(statistic, bool)
            or not isinstance(statistic, (int, float))
            or not math.isfinite(statistic)
            or statistic < 0
        ):
            raise OrbitronProtocolError(
                "Orbitron geometry analysis bond-length statistics must be "
                "finite and non-negative"
            )
        statistics.append(float(statistic))
    minimum, maximum, mean, _ = statistics
    if minimum > mean or mean > maximum:
        raise OrbitronProtocolError(
            "Orbitron geometry analysis bond-length statistics are inconsistent"
        )


def _validate_unit_cell(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise OrbitronProtocolError(
            "Orbitron geometry analysis unit_cell must be an object or null"
        )
    for field_name in ("a", "b", "c"):
        _finite_vector(value, field_name)
    periodic = value.get("periodic")
    if (
        not isinstance(periodic, list)
        or len(periodic) != 3
        or any(not isinstance(item, bool) for item in periodic)
    ):
        raise OrbitronProtocolError(
            "Orbitron geometry analysis unit_cell.periodic must have three booleans"
        )
