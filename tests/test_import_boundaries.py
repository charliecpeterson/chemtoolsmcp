"""Migrated internal modules must bypass public compatibility APIs."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
INTERNAL_LAYER_DIRECTORIES = (
    "application",
    "core",
    "execution",
    "integrations",
    "knowledge",
    "persistence",
    "programs",
    "reference",
)
TARGETS = (
    *sorted(
        path
        for directory in INTERNAL_LAYER_DIRECTORIES
        for path in (ROOT / "chemtools" / directory).rglob("*.py")
    ),
    ROOT / "chemtools" / "mcp" / "tools" / "generic.py",
)
MIGRATED_NWCHEM_MCP_MODULES = (
    ROOT / "chemtools" / "mcp" / "tools" / "nwchem_analysis.py",
    ROOT / "chemtools" / "mcp" / "tools" / "nwchem_docs.py",
    ROOT / "chemtools" / "mcp" / "tools" / "nwchem_input.py",
    ROOT / "chemtools" / "mcp" / "tools" / "nwchem_jobs.py",
    ROOT / "chemtools" / "mcp" / "tools" / "nwchem_parse.py",
)
APPLICATION_ARCHIVE_ADAPTERS = {
    ROOT / "chemtools" / "application" / "nwchem_execution.py": {
        "archive_previous_outputs",
    },
}
RESOURCE_INSPECTION_CALLERS = {
    ROOT / "chemtools" / "mcp" / "tools" / "generic.py": {
        "get_local_resource_budget",
        "query_partition_specs",
    },
    (
        ROOT
        / "chemtools"
        / "programs"
        / "nwchem"
        / "strategy"
        / "workflow_state.py"
    ): {"query_partition_specs"},
}
REMOVED_REDUNDANT_EXECUTION_MODULES = (
    "chemtools.application.dirac_execution",
    "chemtools.application.dirac_monitoring",
    "chemtools.programs.dirac.scheduler",
    "chemtools.application.grasp_monitoring",
    "chemtools.programs.grasp.scheduler",
    "chemtools.application.molcas_execution",
    "chemtools.application.molcas_monitoring",
    "chemtools.programs.molcas.scheduler",
    "chemtools.application.qe_execution",
    "chemtools.application.qmcpack_execution",
    "chemtools.mcp.tools.qe_execution",
    "chemtools.core.runner",
    "chemtools.execution.legacy_runner",
)
CORE_COMPATIBILITY_IMPORTS = {
    "chemtools/core/artifact_registry.py": {"chemtools.persistence.artifacts"},
    "chemtools/core/eval.py": {"chemtools.application.evaluation"},
    "chemtools/core/legacy_artifacts.py": {
        "chemtools.application.legacy_artifacts",
    },
    "chemtools/core/registry_db.py": {"chemtools.persistence.sqlite"},
    "chemtools/core/run_records.py": {"chemtools.persistence.runs"},
    "chemtools/core/run_registry.py": {
        "chemtools.application.run_registry",
    },
}


def _chemtools_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "chemtools" or module.startswith("chemtools."):
                found.append((node.lineno, module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "chemtools" or alias.name.startswith("chemtools."):
                    found.append((node.lineno, alias.name))
    return found


def _facade_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (
                module == "chemtools"
                or module.startswith("chemtools.api")
                or module == "chemtools.mcp.tools.nwchem"
            ):
                found.append((node.lineno, module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if (
                    alias.name == "chemtools"
                    or alias.name.startswith("chemtools.api")
                    or alias.name == "chemtools.mcp.tools.nwchem"
                ):
                    found.append((node.lineno, alias.name))
    return found


def test_migrated_internal_modules_do_not_import_public_facades():
    violations = {}
    for path in TARGETS:
        imports = _facade_imports(path)
        if imports:
            violations[str(path.relative_to(ROOT))] = imports

    assert violations == {}


def test_internal_layers_do_not_import_mcp():
    violations = {}
    for path in TARGETS:
        if "/mcp/" in path.as_posix():
            continue
        imports = [
            item
            for item in _chemtools_imports(path)
            if item[1] == "chemtools.mcp"
            or item[1].startswith("chemtools.mcp.")
        ]
        if imports:
            violations[str(path.relative_to(ROOT))] = imports

    assert violations == {}


def test_core_implementation_depends_only_on_core():
    violations = {}
    observed_compatibility = {}
    for path in sorted((ROOT / "chemtools" / "core").glob("*.py")):
        relative = str(path.relative_to(ROOT))
        allowed = CORE_COMPATIBILITY_IMPORTS.get(relative, set())
        outward = {
            module
            for _, module in _chemtools_imports(path)
            if module != "chemtools.core"
            and not module.startswith("chemtools.core.")
        }
        unexpected = sorted(outward - allowed)
        if unexpected:
            violations[relative] = unexpected
        if allowed:
            observed_compatibility[relative] = outward

    assert violations == {}
    assert observed_compatibility == CORE_COMPATIBILITY_IMPORTS


def test_execution_adapters_do_not_import_program_packages():
    violations = {}
    for path in sorted((ROOT / "chemtools" / "execution").glob("*.py")):
        relative = str(path.relative_to(ROOT))
        program_imports = {
            module
            for _, module in _chemtools_imports(path)
            if module == "chemtools.programs"
            or module.startswith("chemtools.programs.")
        }
        if program_imports:
            violations[relative] = sorted(program_imports)

    assert violations == {}


def test_removed_legacy_profile_facade_is_absent():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                "assert importlib.util.find_spec("
                "'chemtools.execution.legacy_profiles') is None"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_removed_legacy_status_modules_are_absent():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                "assert importlib.util.find_spec("
                "'chemtools.execution.legacy_status') is None; "
                "assert importlib.util.find_spec("
                "'chemtools.programs.nwchem.legacy_status') is None"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_removed_redundant_execution_modules_are_absent():
    modules = repr(REMOVED_REDUNDANT_EXECUTION_MODULES)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                f"modules = {modules}; "
                "assert all(importlib.util.find_spec(module) is None "
                "for module in modules)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_guided_nwchem_planner_bypasses_legacy_runner():
    planner = (
        ROOT
        / "chemtools"
        / "programs"
        / "nwchem"
        / "_plugin_launcher.py"
    )

    assert "chemtools.execution.legacy_runner" not in {
        module for _, module in _chemtools_imports(planner)
    }


def test_nwchem_mcp_execution_adapter_bypasses_legacy_runner():
    adapter = ROOT / "chemtools" / "application" / "nwchem_execution.py"

    imports = {module for _, module in _chemtools_imports(adapter)}
    assert "chemtools.execution.legacy_runner" not in imports
    assert "chemtools.programs.nwchem.runner" not in imports


def test_application_adapters_import_archive_policy_from_focused_owner():
    observed = {}
    violations = {}
    for path, expected_names in APPLICATION_ARCHIVE_ADAPTERS.items():
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        focused_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "chemtools.execution.legacy_archive"
            for alias in node.names
        }
        legacy_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "chemtools.execution.legacy_runner"
            for alias in node.names
            if alias.name.startswith("archive")
        }
        relative = str(path.relative_to(ROOT))
        observed[relative] = focused_names
        if legacy_names:
            violations[relative] = legacy_names

    assert violations == {}
    assert observed == {
        str(path.relative_to(ROOT)): names
        for path, names in APPLICATION_ARCHIVE_ADAPTERS.items()
    }


def test_resource_callers_import_focused_inspection_owner():
    observed = {}
    violations = {}
    for path, expected_names in RESOURCE_INSPECTION_CALLERS.items():
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        focused_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "chemtools.execution.resource_inspection"
            for alias in node.names
        }
        legacy_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "chemtools.execution.legacy_runner"
            for alias in node.names
            if alias.name in expected_names
        }
        relative = str(path.relative_to(ROOT))
        observed[relative] = focused_names & expected_names
        if legacy_names:
            violations[relative] = legacy_names

    assert violations == {}
    assert observed == {
        str(path.relative_to(ROOT)): names
        for path, names in RESOURCE_INSPECTION_CALLERS.items()
    }


def test_program_packages_do_not_import_application_or_reference_layers():
    forbidden_prefixes = (
        "chemtools.application",
        "chemtools.integrations",
        "chemtools.knowledge",
        "chemtools.mcp",
        "chemtools.persistence",
        "chemtools.reference",
    )
    violations = {}
    for path in sorted((ROOT / "chemtools" / "programs").rglob("*.py")):
        imports = [
            item
            for item in _chemtools_imports(path)
            if item[1].startswith(forbidden_prefixes)
        ]
        if imports:
            violations[str(path.relative_to(ROOT))] = imports

    assert violations == {}


def test_persistence_adapters_depend_only_on_core_and_persistence():
    violations = {}
    for path in sorted((ROOT / "chemtools" / "persistence").glob("*.py")):
        unexpected = [
            item
            for item in _chemtools_imports(path)
            if not (
                item[1] == "chemtools.core"
                or item[1].startswith("chemtools.core.")
                or item[1] == "chemtools.persistence"
                or item[1].startswith("chemtools.persistence.")
            )
        ]
        if unexpected:
            violations[str(path.relative_to(ROOT))] = unexpected

    assert violations == {}


def test_migrated_nwchem_mcp_modules_do_not_import_base_wildcard():
    violations = {}
    for path in MIGRATED_NWCHEM_MCP_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        wildcard_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "chemtools.mcp.tools._nwchem_base"
            and any(alias.name == "*" for alias in node.names)
        ]
        if wildcard_lines:
            violations[str(path.relative_to(ROOT))] = wildcard_lines

    assert violations == {}


def test_migrated_nwchem_mcp_modules_use_program_owned_action_strategy():
    legacy_module = "chemtools.mcp.tools._nwchem_next_actions"
    violations = {}
    for path in MIGRATED_NWCHEM_MCP_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == legacy_module
        ]
        if lines:
            violations[str(path.relative_to(ROOT))] = lines

    assert violations == {}


def test_old_nwchem_action_import_is_an_exact_compatibility_alias():
    from chemtools.mcp.tools._nwchem_next_actions import _build_next_actions
    from chemtools.programs.nwchem.strategy.legacy_next_actions import (
        build_legacy_next_actions,
    )

    assert _build_next_actions is build_legacy_next_actions


def test_migrated_nwchem_mcp_families_do_not_load_broad_base():
    for path in MIGRATED_NWCHEM_MCP_MODULES:
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib, sys; "
                    f"importlib.import_module({module!r}); "
                    "assert 'chemtools.mcp.tools._nwchem_base' not in sys.modules"
                ),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr


def test_nwchem_catalog_provider_does_not_load_legacy_aggregator():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, chemtools.mcp.tools._nwchem_provider, sys; "
                "assert importlib.util.find_spec("
                "'chemtools.mcp.tools._nwchem_base') is None; "
                "assert importlib.util.find_spec("
                "'chemtools.mcp.tools.nwchem') is None; "
                "assert 'chemtools.mcp.tools._nwchem_base' not in sys.modules; "
                "assert 'chemtools.mcp.tools.nwchem' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
