"""Built-in program and MCP tool providers assembled by the CLI layer.

This catalog registers backend objects with core separately from loading MCP
tool metadata. Dynamic discovery remains outside the built-in contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import re
from typing import Any, Callable, Iterable

from chemtools.core import registry
from chemtools.core.program import ProgramBackend


@dataclass(frozen=True)
class BuiltinBackendSpec:
    name: str
    program_module: str
    backend_attribute: str
    tools_module: str
    definitions_attribute: str


@dataclass(frozen=True)
class ToolDefinitionProvider:
    module: str
    attribute: str


BUILTIN_BACKENDS: tuple[BuiltinBackendSpec, ...] = (
    BuiltinBackendSpec(
        name="nwchem",
        program_module="chemtools.programs.nwchem",
        backend_attribute="NWCHEM",
        tools_module="chemtools.mcp.tools.nwchem",
        definitions_attribute="_nwchem_tool_definitions",
    ),
    BuiltinBackendSpec(
        name="molcas",
        program_module="chemtools.programs.molcas",
        backend_attribute="MOLCAS",
        tools_module="chemtools.mcp.tools.molcas",
        definitions_attribute="molcas_tool_definitions",
    ),
    BuiltinBackendSpec(
        name="dirac",
        program_module="chemtools.programs.dirac",
        backend_attribute="DIRAC",
        tools_module="chemtools.mcp.tools.dirac",
        definitions_attribute="dirac_tool_definitions",
    ),
    BuiltinBackendSpec(
        name="grasp",
        program_module="chemtools.programs.grasp",
        backend_attribute="GRASP",
        tools_module="chemtools.mcp.tools.grasp",
        definitions_attribute="grasp_tool_definitions",
    ),
    BuiltinBackendSpec(
        name="qe",
        program_module="chemtools.programs.qe",
        backend_attribute="QE",
        tools_module="chemtools.mcp.tools.qe",
        definitions_attribute="qe_tool_definitions",
    ),
    BuiltinBackendSpec(
        name="qmcpack",
        program_module="chemtools.programs.qmcpack",
        backend_attribute="QMCPACK",
        tools_module="chemtools.mcp.tools.qmcpack",
        definitions_attribute="qmcpack_tool_definitions",
    ),
)

GENERIC_TOOL_DEFINITIONS = ToolDefinitionProvider(
    module="chemtools.mcp.tools.generic",
    attribute="generic_tool_definitions",
)
GUIDED_TOOL_DEFINITIONS = ToolDefinitionProvider(
    module="chemtools.mcp.tools.guided",
    attribute="guided_tool_definitions",
)
ORBITRON_TOOL_DEFINITIONS = ToolDefinitionProvider(
    module="chemtools.mcp.tools.orbitron",
    attribute="orbitron_tool_definitions",
)
SCIENCE_RUNTIME_TOOL_DEFINITIONS = ToolDefinitionProvider(
    module="chemtools.mcp.tools.science_runtime",
    attribute="science_runtime_tool_definitions",
)
KNOWLEDGE_TOOL_DEFINITIONS = ToolDefinitionProvider(
    module="chemtools.mcp.tools.knowledge",
    attribute="knowledge_tool_definitions",
)
REFERENCE_TOOL_DEFINITIONS = ToolDefinitionProvider(
    module="chemtools.mcp.tools.reference",
    attribute="reference_tool_definitions",
)


def validate_catalog(
    specs: Iterable[BuiltinBackendSpec] = BUILTIN_BACKENDS,
) -> tuple[BuiltinBackendSpec, ...]:
    catalog = tuple(specs)
    names = [spec.name for spec in catalog]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate built-in backend names: {names}")
    for spec in catalog:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.name):
            raise ValueError(f"invalid built-in backend name: {spec.name!r}")
        for field_name in (
            "program_module",
            "backend_attribute",
            "tools_module",
            "definitions_attribute",
        ):
            if not getattr(spec, field_name):
                raise ValueError(
                    f"built-in backend {spec.name!r} has empty {field_name}"
                )
    return catalog


def builtin_program_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in validate_catalog())


def load_backend(spec: BuiltinBackendSpec) -> ProgramBackend:
    backend = getattr(import_module(spec.program_module), spec.backend_attribute)
    if not isinstance(backend, ProgramBackend):
        raise TypeError(
            f"{spec.program_module}.{spec.backend_attribute} "
            "must be a ProgramBackend"
        )
    if getattr(backend, "name", None) != spec.name:
        raise ValueError(
            f"catalog name {spec.name!r} does not match backend "
            f"{getattr(backend, 'name', None)!r}"
        )
    return backend


def register_builtin_backends() -> tuple[ProgramBackend, ...]:
    backends = tuple(load_backend(spec) for spec in validate_catalog())
    for backend in backends:
        registry.register(backend)
    return backends


def load_tool_modules() -> None:
    import_module(GENERIC_TOOL_DEFINITIONS.module)
    import_module(GUIDED_TOOL_DEFINITIONS.module)
    import_module(ORBITRON_TOOL_DEFINITIONS.module)
    import_module(SCIENCE_RUNTIME_TOOL_DEFINITIONS.module)
    import_module(KNOWLEDGE_TOOL_DEFINITIONS.module)
    import_module(REFERENCE_TOOL_DEFINITIONS.module)
    for spec in validate_catalog():
        import_module(spec.tools_module)


def _load_definition_provider(
    provider: ToolDefinitionProvider | BuiltinBackendSpec,
) -> Callable[[], list[dict[str, Any]]]:
    module_name = (
        provider.module
        if isinstance(provider, ToolDefinitionProvider)
        else provider.tools_module
    )
    attribute = (
        provider.attribute
        if isinstance(provider, ToolDefinitionProvider)
        else provider.definitions_attribute
    )
    definition_provider = getattr(import_module(module_name), attribute)
    if not callable(definition_provider):
        raise TypeError(f"{module_name}.{attribute} is not callable")
    return definition_provider


def load_tool_definitions(
    provider: ToolDefinitionProvider | BuiltinBackendSpec,
) -> list[dict[str, Any]]:
    definitions = _load_definition_provider(provider)()
    if not isinstance(definitions, list):
        raise TypeError("tool definition provider must return a list")
    return definitions


def catalog_tool_definitions() -> list[dict[str, Any]]:
    definitions = load_tool_definitions(GENERIC_TOOL_DEFINITIONS)
    definitions.extend(load_tool_definitions(GUIDED_TOOL_DEFINITIONS))
    definitions.extend(load_tool_definitions(ORBITRON_TOOL_DEFINITIONS))
    definitions.extend(load_tool_definitions(SCIENCE_RUNTIME_TOOL_DEFINITIONS))
    definitions.extend(load_tool_definitions(KNOWLEDGE_TOOL_DEFINITIONS))
    definitions.extend(load_tool_definitions(REFERENCE_TOOL_DEFINITIONS))
    for spec in validate_catalog():
        definitions.extend(load_tool_definitions(spec))
    return definitions


__all__ = [
    "BuiltinBackendSpec",
    "ToolDefinitionProvider",
    "BUILTIN_BACKENDS",
    "GENERIC_TOOL_DEFINITIONS",
    "GUIDED_TOOL_DEFINITIONS",
    "ORBITRON_TOOL_DEFINITIONS",
    "KNOWLEDGE_TOOL_DEFINITIONS",
    "REFERENCE_TOOL_DEFINITIONS",
    "validate_catalog",
    "builtin_program_names",
    "load_backend",
    "register_builtin_backends",
    "load_tool_modules",
    "load_tool_definitions",
    "catalog_tool_definitions",
]
