"""Track the NWChem runtime-database state active at each task directive.

The parser follows sequential input semantics without trying to model the
complete NWChem grammar.
"""

from __future__ import annotations

import shlex
from typing import Any

from chemtools.core.common import read_text
from chemtools.programs.nwchem.input.basis_library import (
    normalize_element_symbol,
)
from chemtools.programs.nwchem.xc_consistency import canonical_xc_alias


SPIN_MULTIPLICITIES = {
    "singlet": 1,
    "doublet": 2,
    "triplet": 3,
    "quartet": 4,
    "quintet": 5,
    "sextet": 6,
    "septet": 7,
    "octet": 8,
}
_STATEFUL_MODULES = {"dft", "mcscf", "scf"}
_TCE_METHODS = {
    "ccsd": "CCSD",
    "ccsd(t)": "CCSD(T)",
    "mbpt2": "MP2",
    "mp2": "MP2",
}
_REFERENCE_KEYWORDS = {
    "dft": {
        "odft": "open_shell",
        "rodft": "open_shell",
    },
    "scf": {
        "rhf": "closed_shell",
        "rohf": "open_shell",
        "uhf": "open_shell",
    },
}
_GEOMETRY_FLAGS = {
    "adjust",
    "autoz",
    "bqbq",
    "center",
    "nocenter",
    "noautoz",
    "noautosym",
    "noprint",
}
_GEOMETRY_VALUE_KEYWORDS = {
    "ang2au",
    "angstrom_to_au",
    "nuc",
    "nucl",
    "nucleus",
    "units",
}
_GEOMETRY_CHANGING_OPERATIONS = {
    "dynamics",
    "opt",
    "optimization",
    "optimize",
    "saddle",
}


def parse_task_states(path: str) -> list[dict[str, Any]]:
    """Return the NWChem database state active at each task directive."""
    contents = read_text(path)
    charge: int | None = 0
    charge_source = "default"
    active_geometry = "geometry"
    geometries: dict[str, dict[str, Any]] = {}
    module_multiplicities = _default_module_multiplicities()
    module_references: dict[str, dict[str, str]] = {}
    restart_state = False
    task_states: list[dict[str, Any]] = []
    current_block: str | None = None
    current_geometry_name: str | None = None
    geometry_block_index = -1
    ecp_core_electrons: dict[str, int] = {}
    ecp_library_elements: set[str] = set()
    ecp_library_assignments: dict[str, str] = {}
    ecp_default_library = False
    ecp_default_library_name: str | None = None
    ecp_uses_external_library_file = False
    ecp_sets: dict[str, dict[str, Any]] = {}
    active_ecp_name = "ecp basis"
    current_ecp_name: str | None = None
    basis_sets: dict[str, dict[str, str]] = {}
    active_basis_name = "ao basis"
    current_basis_name: str | None = None
    current_basis_mode: str | None = None
    current_basis_mode_source: str | None = None
    dft_xc: dict[str, Any] | None = None
    tce_method: str | None = None

    for tokens in _logical_input_tokens(contents):
        if not tokens:
            continue
        keyword = tokens[0].lower()

        if current_block == "geometry":
            if keyword == "end":
                if current_geometry_name is not None:
                    geometries[current_geometry_name] = {
                        "name": current_geometry_name,
                        "block_index": geometry_block_index,
                        "source": "input",
                    }
                current_block = None
                current_geometry_name = None
            continue

        if current_block == "basis":
            if keyword == "end":
                if current_basis_name is not None:
                    basis_sets[current_basis_name] = {
                        "name": current_basis_name,
                        "mode": current_basis_mode or "cartesian",
                        "mode_source": (
                            current_basis_mode_source or "default"
                        ),
                        "source": "input",
                    }
                    if current_basis_name == "ao basis":
                        active_basis_name = "ao basis"
                current_block = None
                current_basis_name = None
                current_basis_mode = None
                current_basis_mode_source = None
            continue

        if current_block in _STATEFUL_MODULES:
            if keyword == "end":
                current_block = None
                continue
            if multiplicity := _multiplicity_setting(tokens):
                module_multiplicities[current_block] = multiplicity
            if reference_class := _REFERENCE_KEYWORDS.get(
                current_block,
                {},
            ).get(keyword):
                module_references[current_block] = {
                    "kind": keyword,
                    "class": reference_class,
                    "source": "explicit",
                }
            if current_block == "dft" and keyword == "xc":
                dft_xc = _xc_state(tokens[1:])
            continue

        if current_block == "tce":
            if keyword == "end":
                current_block = None
                continue
            if method := _TCE_METHODS.get(keyword):
                tce_method = method
            continue

        if current_block == "ecp":
            if keyword == "end":
                if current_ecp_name is not None:
                    ecp_sets[current_ecp_name] = _ecp_state(
                        source=(
                            "ambiguous" if restart_state else "explicit"
                        ),
                        core_electrons=ecp_core_electrons,
                        library_elements=ecp_library_elements,
                        library_assignments=ecp_library_assignments,
                        default_library=ecp_default_library,
                        default_library_name=ecp_default_library_name,
                        uses_external_library_file=(
                            ecp_uses_external_library_file
                        ),
                        name=current_ecp_name,
                    )
                    if current_ecp_name == "ecp basis":
                        active_ecp_name = "ecp basis"
                current_block = None
                current_ecp_name = None
                continue
            element = _element_from_tag(tokens[0])
            if element is None:
                continue
            if len(tokens) >= 2 and tokens[1].lower() == "library":
                family, uses_external_file = _library_reference(tokens[2:])
                ecp_uses_external_library_file |= uses_external_file
                if tokens[0] == "*":
                    ecp_default_library = True
                    ecp_default_library_name = family
                else:
                    ecp_library_elements.add(element)
                    if family is not None:
                        ecp_library_assignments[element] = family
                continue
            electron_token = None
            if len(tokens) >= 3 and tokens[1].lower() == "nelec":
                electron_token = tokens[2]
            elif len(tokens) >= 2:
                electron_token = tokens[1]
            if electron_token is not None:
                try:
                    ecp_core_electrons[element] = int(electron_token)
                except ValueError:
                    pass
            continue

        if keyword in {"start", "restart"}:
            charge = 0 if keyword == "start" else None
            charge_source = "default" if keyword == "start" else "restart"
            active_geometry = "geometry"
            geometries = {}
            module_multiplicities = (
                _default_module_multiplicities()
                if keyword == "start"
                else {}
            )
            module_references = {}
            restart_state = keyword == "restart"
            ecp_core_electrons = {}
            ecp_library_elements = set()
            ecp_library_assignments = {}
            ecp_default_library = False
            ecp_default_library_name = None
            ecp_uses_external_library_file = False
            ecp_sets = {}
            active_ecp_name = "ecp basis"
            current_ecp_name = None
            basis_sets = {}
            active_basis_name = "ao basis"
            dft_xc = None
            tce_method = None
            continue

        if keyword == "geometry":
            geometry_block_index += 1
            current_block = "geometry"
            current_geometry_name = _geometry_name(tokens[1:])
            continue

        if keyword in _STATEFUL_MODULES and len(tokens) == 1:
            current_block = keyword
            continue

        if keyword == "tce" and len(tokens) == 1:
            current_block = "tce"
            continue

        if keyword == "basis":
            (
                current_basis_name,
                current_basis_mode,
                current_basis_mode_source,
            ) = _basis_header_state(tokens[1:])
            current_block = "basis"
            continue

        if keyword == "ecp":
            current_ecp_name = _ecp_header_name(tokens[1:])
            ecp_core_electrons = {}
            ecp_library_elements = set()
            ecp_library_assignments = {}
            ecp_default_library = False
            ecp_default_library_name = None
            ecp_uses_external_library_file = False
            current_block = "ecp"
            continue

        if keyword == "set" and len(tokens) >= 3:
            if tokens[1].lower() == "geometry":
                active_geometry = tokens[2]
            elif tokens[1].lower() == "ao basis":
                active_basis_name = tokens[2]
            elif tokens[1].lower() == "ecp basis":
                active_ecp_name = tokens[2]
            continue

        if keyword == "charge" and len(tokens) >= 2:
            try:
                charge = int(tokens[1])
            except ValueError:
                charge = None
            charge_source = "explicit"
            continue

        if keyword != "task" or len(tokens) < 2:
            continue

        module = tokens[1].lower()
        operation = (
            tokens[2].lower()
            if len(tokens) >= 3
            else ("property" if module == "dplot" else "energy")
        )
        geometry = geometries.get(active_geometry)
        multiplicity = module_multiplicities.get(module)
        if multiplicity is None and module not in _STATEFUL_MODULES:
            multiplicity = module_multiplicities.get("scf")
        reference = module_references.get(module)
        if reference is None and module not in _STATEFUL_MODULES:
            reference = module_references.get("scf")
        if module in {"dft", "scf"}:
            reference_module = module
        elif module not in _STATEFUL_MODULES:
            reference_module = "scf"
        else:
            reference_module = None
        if (
            reference is None
            and not restart_state
            and multiplicity is not None
            and reference_module is not None
        ):
            reference_class = (
                "closed_shell"
                if multiplicity["value"] == 1
                else "open_shell"
            )
            reference = {
                "kind": reference_class,
                "class": reference_class,
                "source": "default",
            }
            if module == "dft" and reference_class == "open_shell":
                module_references["dft"] = dict(reference)
        state = {
            "task_index": len(task_states),
            "module": module,
            "operation": operation,
            "charge": charge,
            "charge_source": charge_source,
            "multiplicity": (
                multiplicity["value"]
                if multiplicity is not None
                else None
            ),
            "multiplicity_source": (
                multiplicity["source"]
                if multiplicity is not None
                else None
            ),
            "reference": (
                dict(reference)
                if reference is not None
                else {
                    "kind": None,
                    "class": None,
                    "source": "restart" if restart_state else "unresolved",
                }
            ),
            "basis": _active_basis_state(
                basis_sets,
                active_basis_name,
                restart_state,
            ),
            "ecp": _active_ecp_state(
                ecp_sets,
                active_ecp_name,
                restart_state,
            ),
            "geometry": (
                dict(geometry)
                if geometry is not None
                else {
                    "name": active_geometry,
                    "block_index": None,
                    "source": "unresolved",
                }
            ),
        }
        if module in {"dft", "tddft"} and dft_xc is not None:
            state["xc"] = dict(dft_xc)
        if module == "tce" and tce_method is not None:
            state["method"] = tce_method
            state["method_source"] = "explicit_tce_keyword"
        task_states.append(state)
        if operation in _GEOMETRY_CHANGING_OPERATIONS and geometry is not None:
            geometries[active_geometry] = {
                **geometry,
                "source": "task_result",
                "source_task_index": state["task_index"],
            }

    return task_states


def _default_module_multiplicities() -> dict[str, dict[str, Any]]:
    return {
        module: {"value": 1, "source": "default"}
        for module in ("dft", "scf")
    }


def _ecp_state(
    *,
    source: str,
    core_electrons: dict[str, int],
    library_elements: set[str],
    library_assignments: dict[str, str],
    default_library: bool,
    default_library_name: str | None,
    uses_external_library_file: bool,
    name: str = "ecp basis",
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "source": source,
        "core_electrons": dict(core_electrons),
        "library_elements": sorted(library_elements),
        "default_library": default_library,
    }
    if library_assignments:
        state["library_assignments"] = dict(library_assignments)
    if default_library_name is not None:
        state["default_library_name"] = default_library_name
    if uses_external_library_file:
        state["uses_external_library_file"] = True
    if name != "ecp basis":
        state["name"] = name
    return state


def _active_ecp_state(
    ecp_sets: dict[str, dict[str, Any]],
    active_name: str,
    restart_state: bool,
) -> dict[str, Any]:
    if ecp := ecp_sets.get(active_name):
        return dict(ecp)
    source = "restart" if restart_state else "none"
    state = _ecp_state(
        source=source if active_name == "ecp basis" else "unresolved",
        core_electrons={},
        library_elements=set(),
        library_assignments={},
        default_library=False,
        default_library_name=None,
        uses_external_library_file=False,
        name=active_name,
    )
    return state


def _ecp_header_name(tokens: list[str]) -> str:
    names = [
        token
        for token in tokens
        if token.lower() not in {"noprint", "print"}
    ]
    return names[-1] if names else "ecp basis"


def _basis_header_state(
    tokens: list[str],
) -> tuple[str, str, str]:
    name = "ao basis"
    mode = "cartesian"
    mode_source = "default"
    options = {"bse", "cartesian", "noprint", "print", "rel", "spherical"}
    for token in tokens:
        lowered = token.lower()
        if lowered in {"cartesian", "spherical"}:
            mode = lowered
            mode_source = "explicit"
        elif lowered not in options:
            name = token
    return name, mode, mode_source


def _library_reference(tokens: list[str]) -> tuple[str | None, bool]:
    lowered = [token.lower() for token in tokens]
    stop = next(
        (
            index
            for index, token in enumerate(lowered)
            if token in {"except", "file", "rel"}
        ),
        len(tokens),
    )
    names = tokens[:stop]
    return (names[-1] if names else None), "file" in lowered


def _active_basis_state(
    basis_sets: dict[str, dict[str, str]],
    active_name: str,
    restart_state: bool,
) -> dict[str, str | None]:
    if basis := basis_sets.get(active_name):
        return dict(basis)
    return {
        "name": active_name,
        "mode": None,
        "mode_source": None,
        "source": "restart" if restart_state else "unresolved",
    }


def _element_from_tag(tag: str) -> str | None:
    if tag == "*":
        return "*"
    try:
        return normalize_element_symbol(tag)
    except ValueError:
        return None


def _logical_input_tokens(contents: str) -> list[list[str]]:
    joined_lines: list[str] = []
    for raw_line in contents.splitlines():
        if joined_lines and joined_lines[-1].rstrip().endswith("\\"):
            joined_lines[-1] = (
                joined_lines[-1].rstrip()[:-1].rstrip()
                + " "
                + raw_line.strip()
            )
        else:
            joined_lines.append(raw_line)

    logical_lines: list[list[str]] = []
    for line in joined_lines:
        lexer = shlex.shlex(
            line,
            posix=True,
            punctuation_chars=";",
        )
        lexer.commenters = "#"
        lexer.whitespace_split = True
        tokens: list[str] = []
        try:
            for token in lexer:
                if token == ";":
                    if tokens:
                        logical_lines.append(tokens)
                        tokens = []
                else:
                    tokens.append(token)
        except ValueError:
            continue
        if tokens:
            logical_lines.append(tokens)
    return logical_lines


def _geometry_name(tokens: list[str]) -> str | None:
    candidates: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lowered = token.lower()
        if lowered in _GEOMETRY_FLAGS:
            index += 1
            continue
        if lowered in _GEOMETRY_VALUE_KEYWORDS:
            index += 2
            continue
        if lowered == "print":
            index += 2 if (
                index + 1 < len(tokens)
                and tokens[index + 1].lower() == "xyz"
            ) else 1
            continue
        if lowered == "autosym":
            index += 1
            if index < len(tokens):
                try:
                    float(tokens[index].replace("d", "e").replace("D", "E"))
                except ValueError:
                    pass
                else:
                    index += 1
            continue
        candidates.append(token)
        index += 1
    if not candidates:
        return "geometry"
    return candidates[0] if len(candidates) == 1 else None


def _multiplicity_setting(tokens: list[str]) -> dict[str, Any] | None:
    keyword = tokens[0].lower()
    if keyword in SPIN_MULTIPLICITIES:
        return {
            "value": SPIN_MULTIPLICITIES[keyword],
            "source": keyword,
        }
    if keyword not in {"mult", "multiplicity", "nopen"} or len(tokens) < 2:
        return None
    try:
        value = int(tokens[1])
    except ValueError:
        return None
    if keyword == "mult":
        value = abs(value)
    return {
        "value": value + 1 if keyword == "nopen" else value,
        "source": keyword,
    }


def _xc_state(tokens: list[str]) -> dict[str, Any]:
    if len(tokens) == 1:
        alias = canonical_xc_alias(tokens[0])
        if alias is not None:
            return {
                "name": alias,
                "tokens": list(tokens),
                "source": "explicit_alias",
            }
    return {
        "name": None,
        "tokens": list(tokens),
        "source": "explicit_expression",
    }


__all__ = ["SPIN_MULTIPLICITIES", "parse_task_states"]
