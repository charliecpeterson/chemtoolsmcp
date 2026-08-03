"""Optional adapters for tools maintained outside Chemtools."""

from .orbitron import (
    ORBITRON_CLI_ENV,
    OrbitronClient,
    OrbitronCommandError,
    OrbitronError,
    OrbitronProtocolError,
    OrbitronResponse,
    OrbitronUnavailableError,
    OrbitronVersion,
    resolve_orbitron_cli,
)
from .science_runtime import (
    SCIENCE_RUNTIME_PYTHON_ENV,
    ScienceRuntimeClient,
    ScienceRuntimeCommandError,
    ScienceRuntimeError,
    ScienceRuntimeProtocolError,
    ScienceRuntimeUnavailableError,
    resolve_science_runtime_python,
)

__all__ = [
    "ORBITRON_CLI_ENV",
    "OrbitronClient",
    "OrbitronCommandError",
    "OrbitronError",
    "OrbitronProtocolError",
    "OrbitronResponse",
    "OrbitronUnavailableError",
    "OrbitronVersion",
    "resolve_orbitron_cli",
    "SCIENCE_RUNTIME_PYTHON_ENV",
    "ScienceRuntimeClient",
    "ScienceRuntimeCommandError",
    "ScienceRuntimeError",
    "ScienceRuntimeProtocolError",
    "ScienceRuntimeUnavailableError",
    "resolve_science_runtime_python",
]
