"""Parse and review the supported subset of Quantum ESPRESSO pw.x inputs.

The parser covers scalar namelist assignments and the structural cards needed
for SCF, relax, and vc-relax review. It intentionally does not evaluate
general Fortran expressions or validate pseudopotential contents.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from chemtools.core.types import LintIssue


_CARD_NAMES = (
    "ADDITIONAL_K_POINTS",
    "ATOMIC_FORCES",
    "ATOMIC_POSITIONS",
    "ATOMIC_SPECIES",
    "ATOMIC_VELOCITIES",
    "CELL_PARAMETERS",
    "CONSTRAINTS",
    "HUBBARD",
    "K_POINTS",
    "OCCUPATIONS",
    "SOLVENTS",
)
_CARD_RE = re.compile(
    rf"^\s*({'|'.join(_CARD_NAMES)})\b(.*)$",
    re.IGNORECASE,
)
_NAMELIST_START_RE = re.compile(r"^\s*&([A-Za-z][A-Za-z0-9_]*)\b(.*)$")
_ASSIGNMENT_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*=\s*(.*?)\s*$"
)
_SUPPORTED_CALCULATIONS = frozenset({"scf", "relax", "vc-relax"})
_UNSUPPORTED_PROGRAM_NAMELISTS = {
    "bands": "bands.x",
    "dos": "dos.x",
    "inputpp": "pp.x",
    "projwfc": "projwfc.x",
}
_K_POINT_OPTIONS = frozenset({
    "automatic",
    "crystal",
    "crystal_b",
    "crystal_c",
    "gamma",
    "tpiba",
    "tpiba_b",
    "tpiba_c",
})
_K_POINT_PATH_OPTIONS = frozenset({"crystal_b", "tpiba_b"})
_POSITION_OPTIONS = frozenset({
    "alat",
    "angstrom",
    "bohr",
    "crystal",
    "crystal_sg",
})
_CELL_UNITS = frozenset({"alat", "angstrom", "bohr"})
_MAX_COORDINATE_EXPRESSION_LENGTH = 256
_MAX_COORDINATE_EXPRESSION_NODES = 64
_FORTRAN_EXPONENT_RE = re.compile(
    r"(?<![A-Za-z_])"
    r"(?P<mantissa>(?:\d+(?:\.\d*)?|\.\d+))"
    r"[dD](?P<exponent>[+-]?\d+)"
)


@dataclass(frozen=True)
class _Namelist:
    name: str
    values: dict[str, Any]
    lines: dict[str, int]
    start_line: int
    closed: bool


@dataclass(frozen=True)
class _Card:
    name: str
    option: str | None
    rows: tuple[tuple[int, str], ...]
    line: int


@dataclass(frozen=True)
class _Document:
    namelists: dict[str, _Namelist]
    cards: dict[str, _Card]


def parse_pw_input(path: str | Path) -> dict[str, Any]:
    """Return a compact, JSON-serializable summary of one pw.x input."""
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    return parse_pw_text(text)


def parse_pw_text(text: str) -> dict[str, Any]:
    """Return the same compact summary without requiring a file path."""
    return _public_summary(_parse_document(text))


def lint_pw_input(text: str) -> list[LintIssue]:
    """Check the pw.x structure and cross-references currently supported."""
    document = _parse_document(text)
    unsupported_program = _unsupported_qe_program(document)
    if unsupported_program is not None:
        return [_issue(
            "error",
            (
                f"This is a Quantum ESPRESSO {unsupported_program} input; the "
                "current Chemtools QE reviewer supports pw.x inputs only."
            ),
        )]
    issues: list[LintIssue] = []

    for name in ("control", "system", "electrons"):
        namelist = document.namelists.get(name)
        if namelist is None:
            issues.append(_issue(
                "error",
                f"Required &{name.upper()} namelist is missing.",
                suggested_fix=f"&{name.upper()}\n/",
            ))
        elif not namelist.closed:
            issues.append(_issue(
                "error",
                f"&{name.upper()} namelist at line {namelist.start_line} is not closed.",
                line=namelist.start_line,
                suggested_fix="/",
            ))

    control = _values(document, "control")
    system = _values(document, "system")
    calculation = str(control.get("calculation", "scf")).lower()
    if calculation not in _SUPPORTED_CALCULATIONS:
        issues.append(_issue(
            "warning",
            (
                f"calculation={calculation!r} parses, but the current Chemtools "
                "review covers only scf, relax, and vc-relax semantics."
            ),
            line=_value_line(document, "control", "calculation"),
        ))

    nat = _positive_integer(system.get("nat"))
    ntyp = _positive_integer(system.get("ntyp"))
    if nat is None:
        issues.append(_required_positive_issue(document, "nat"))
    if ntyp is None:
        issues.append(_required_positive_issue(document, "ntyp"))

    ecutwfc = _real(system.get("ecutwfc"))
    if ecutwfc is None or ecutwfc <= 0:
        issues.append(_issue(
            "error",
            "&SYSTEM requires a positive ecutwfc value in Ry.",
            line=_value_line(document, "system", "ecutwfc"),
            suggested_fix="ecutwfc = <positive cutoff in Ry>",
        ))

    _check_species(document, ntyp, issues)
    _check_positions(document, nat, issues)
    _check_species_references(document, issues)
    _check_cell(document, system.get("ibrav"), issues)
    _check_k_points(document, issues)
    _check_occupations(document, issues)
    return issues


def unsupported_qe_program(text: str) -> str | None:
    """Identify an unsupported QE post-processing input by its top-level namelist."""
    return _unsupported_qe_program(_parse_document(text))


def _parse_document(text: str) -> _Document:
    lines = text.splitlines()
    return _Document(
        namelists=_parse_namelists(lines),
        cards=_parse_cards(lines),
    )


def _unsupported_qe_program(document: _Document) -> str | None:
    for name, program in _UNSUPPORTED_PROGRAM_NAMELISTS.items():
        if name in document.namelists:
            return program
    return None


def _parse_namelists(lines: list[str]) -> dict[str, _Namelist]:
    namelists: dict[str, _Namelist] = {}
    index = 0
    while index < len(lines):
        opening = _NAMELIST_START_RE.match(_strip_comment(lines[index]))
        if opening is None:
            index += 1
            continue

        name = opening.group(1).lower()
        start_line = index + 1
        values: dict[str, Any] = {}
        value_lines: dict[str, int] = {}
        pending = opening.group(2)
        closed = False

        while True:
            clean = _strip_comment(pending).strip()
            if clean == "/":
                closed = True
                break
            for fragment in _split_unquoted(clean, ","):
                match = _ASSIGNMENT_RE.match(fragment.strip())
                if match is None:
                    continue
                key = re.sub(r"\s+", "", match.group(1)).lower()
                values[key] = _parse_scalar(match.group(2))
                value_lines[key] = index + 1

            index += 1
            if index >= len(lines):
                break
            pending = lines[index]

        namelists[name] = _Namelist(
            name=name,
            values=values,
            lines=value_lines,
            start_line=start_line,
            closed=closed,
        )
        index += 1
    return namelists


def _parse_cards(lines: list[str]) -> dict[str, _Card]:
    headers: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        match = _CARD_RE.match(_strip_comment(line))
        if match is not None:
            headers.append((index, match))

    cards: dict[str, _Card] = {}
    for position, (index, match) in enumerate(headers):
        end = headers[position + 1][0] if position + 1 < len(headers) else len(lines)
        rows: list[tuple[int, str]] = []
        for row_index in range(index + 1, end):
            clean = _strip_comment(lines[row_index]).strip()
            if not clean or clean.startswith(("&", "/")):
                continue
            rows.append((row_index + 1, clean))

        name = match.group(1).lower()
        cards[name] = _Card(
            name=name,
            option=_card_option(match.group(2)),
            rows=tuple(rows),
            line=index + 1,
        )
    return cards


def _public_summary(document: _Document) -> dict[str, Any]:
    control = _values(document, "control")
    system = _values(document, "system")
    species = _species_rows(document.cards.get("atomic_species"))
    positions_card = document.cards.get("atomic_positions")
    cell_card = document.cards.get("cell_parameters")
    k_card = document.cards.get("k_points")

    indexed_magnetization = {
        key.removeprefix("starting_magnetization(").removesuffix(")"): value
        for key, value in system.items()
        if key.startswith("starting_magnetization(")
    }
    return {
        "format": "qe-pw-input/1",
        "calculation": str(control.get("calculation", "scf")).lower(),
        "namelists": {
            name: namelist.values
            for name, namelist in document.namelists.items()
        },
        "assignment_lines": {
            name: namelist.lines
            for name, namelist in document.namelists.items()
        },
        "card_lines": {
            name: card.line
            for name, card in document.cards.items()
        },
        "system": {
            "ibrav": system.get("ibrav"),
            "nat": system.get("nat"),
            "ntyp": system.get("ntyp"),
            "ecutwfc_ry": system.get("ecutwfc"),
            "ecutrho_ry": system.get("ecutrho"),
            "occupations": system.get("occupations"),
            "smearing": system.get("smearing"),
            "degauss_ry": system.get("degauss"),
            "nspin": system.get("nspin", 1),
            "starting_magnetization": indexed_magnetization,
        },
        "atomic_species": species,
        "atomic_positions": {
            "units": positions_card.option if positions_card else None,
            "atoms": _position_rows(positions_card),
        } if positions_card else None,
        "cell_parameters": {
            "units": cell_card.option if cell_card else None,
            "vectors": _numeric_rows(cell_card, width=3),
        } if cell_card else None,
        "k_points": _k_points_summary(k_card),
    }


def _check_species(
    document: _Document,
    ntyp: int | None,
    issues: list[LintIssue],
) -> None:
    card = document.cards.get("atomic_species")
    if card is None:
        issues.append(_issue(
            "error",
            "ATOMIC_SPECIES card is missing.",
            suggested_fix="ATOMIC_SPECIES\n<label> <mass> <pseudopotential>",
        ))
        return
    species = _species_rows(card)
    if len(species) != len(card.rows):
        issues.append(_issue(
            "error",
            "Every ATOMIC_SPECIES row must contain a label, mass, and pseudopotential filename.",
            line=card.line,
        ))
    labels = [row["label"] for row in species]
    if len(labels) != len(set(labels)):
        issues.append(_issue(
            "error",
            "ATOMIC_SPECIES labels must be unique.",
            line=card.line,
        ))
    if ntyp is not None and len(species) != ntyp:
        issues.append(_issue(
            "error",
            f"ntyp={ntyp} but ATOMIC_SPECIES contains {len(species)} valid row(s).",
            line=card.line,
            suggested_fix="Make ntyp match the number of ATOMIC_SPECIES rows.",
        ))


def _check_positions(
    document: _Document,
    nat: int | None,
    issues: list[LintIssue],
) -> None:
    card = document.cards.get("atomic_positions")
    if card is None:
        issues.append(_issue(
            "error",
            "ATOMIC_POSITIONS card is missing.",
            suggested_fix="ATOMIC_POSITIONS <units>\n<label> <x> <y> <z>",
        ))
        return
    if card.option is not None and card.option not in _POSITION_OPTIONS:
        issues.append(_issue(
            "error",
            (
                f"ATOMIC_POSITIONS option {card.option!r} is not recognized; "
                "use alat, bohr, angstrom, crystal, or crystal_sg."
            ),
            line=card.line,
        ))
        return
    if card.option == "crystal_sg":
        space_group = _positive_integer(
            _values(document, "system").get("space_group")
        )
        if space_group is None:
            issues.append(_issue(
                "error",
                "ATOMIC_POSITIONS crystal_sg requires a positive space_group number.",
                line=card.line,
                suggested_fix="space_group = <International Tables number>",
            ))
            return
        issues.append(_issue(
            "info",
            (
                "ATOMIC_POSITIONS crystal_sg uses symmetry expansion; "
                "coordinate and atom-count checks are not available."
            ),
            line=card.line,
        ))
        return
    atoms = _position_rows(card)
    if len(atoms) != len(card.rows):
        issues.append(_issue(
            "error",
            "Every ATOMIC_POSITIONS row must contain a label and three numeric coordinates.",
            line=card.line,
        ))
    for line, row in card.rows:
        constraints = row.split()[4:]
        if constraints and (
            len(constraints) != 3
            or any(value not in {"0", "1"} for value in constraints)
        ):
            issues.append(_issue(
                "error",
                (
                    "ATOMIC_POSITIONS constraints must be three values, "
                    "each equal to 0 or 1."
                ),
                line=line,
            ))
    if nat is not None and len(atoms) != nat:
        issues.append(_issue(
            "error",
            f"nat={nat} but ATOMIC_POSITIONS contains {len(atoms)} valid row(s).",
            line=card.line,
            suggested_fix="Make nat match the number of ATOMIC_POSITIONS rows.",
        ))


def _check_species_references(
    document: _Document,
    issues: list[LintIssue],
) -> None:
    species = _species_rows(document.cards.get("atomic_species"))
    atoms = _position_rows(document.cards.get("atomic_positions"))
    known = {row["label"] for row in species}
    unknown = sorted({atom["label"] for atom in atoms} - known)
    if unknown:
        positions = document.cards["atomic_positions"]
        issues.append(_issue(
            "error",
            f"ATOMIC_POSITIONS uses labels absent from ATOMIC_SPECIES: {unknown}.",
            line=positions.line,
            suggested_fix="Add the missing species rows or correct the atom labels.",
        ))


def _check_cell(
    document: _Document,
    raw_ibrav: Any,
    issues: list[LintIssue],
) -> None:
    ibrav = _integer(raw_ibrav)
    card = document.cards.get("cell_parameters")
    if ibrav == 0 and card is None:
        issues.append(_issue(
            "error",
            "ibrav=0 requires a CELL_PARAMETERS card.",
            line=_value_line(document, "system", "ibrav"),
            suggested_fix="CELL_PARAMETERS <alat|bohr|angstrom>\n<v1>\n<v2>\n<v3>",
        ))
        return
    if ibrav not in (None, 0) and card is not None:
        issues.append(_issue(
            "error",
            f"CELL_PARAMETERS is present with ibrav={ibrav}; this card is for ibrav=0.",
            line=card.line,
            suggested_fix="Remove CELL_PARAMETERS or set ibrav=0.",
        ))
    if card is not None and card.option is not None and card.option not in _CELL_UNITS:
        issues.append(_issue(
            "error",
            (
                f"CELL_PARAMETERS option {card.option!r} is not recognized; "
                "use alat, bohr, or angstrom."
            ),
            line=card.line,
        ))
    if card is not None and len(_numeric_rows(card, width=3)) != 3:
        issues.append(_issue(
            "error",
            "CELL_PARAMETERS must contain exactly three numeric vectors.",
            line=card.line,
        ))


def _check_k_points(document: _Document, issues: list[LintIssue]) -> None:
    card = document.cards.get("k_points")
    if card is None:
        issues.append(_issue(
            "error",
            "K_POINTS card is missing.",
            suggested_fix="K_POINTS gamma",
        ))
        return
    option = card.option or "tpiba"
    if option not in _K_POINT_OPTIONS:
        issues.append(_issue(
            "error",
            f"K_POINTS option {option!r} is not recognized by this pw.x parser.",
            line=card.line,
            suggested_fix=(
                "Use tpiba, automatic, crystal, gamma, tpiba_b, crystal_b, "
                "tpiba_c, or crystal_c."
            ),
        ))
        return
    if option == "gamma":
        if card.rows:
            issues.append(_issue(
                "error",
                "K_POINTS gamma must not contain a data row.",
                line=card.line,
                suggested_fix="K_POINTS gamma",
            ))
        return
    if option != "automatic":
        _check_explicit_k_points(document, card, option, issues)
        return
    if len(card.rows) != 1:
        issues.append(_issue(
            "error",
            "K_POINTS automatic requires exactly one row with six integers.",
            line=card.line,
        ))
        return
    tokens = card.rows[0][1].split()
    try:
        values = [int(token) for token in tokens]
    except ValueError:
        values = []
    if len(values) != 6:
        issues.append(_issue(
            "error",
            "K_POINTS automatic requires six integers: nk1 nk2 nk3 sk1 sk2 sk3.",
            line=card.rows[0][0],
        ))
        return
    if any(value <= 0 for value in values[:3]):
        issues.append(_issue(
            "error",
            "The three automatic k-point grid dimensions must be positive.",
            line=card.rows[0][0],
        ))
    if any(value not in (0, 1) for value in values[3:]):
        issues.append(_issue(
            "error",
            "The three automatic k-point shifts must each be 0 or 1.",
            line=card.rows[0][0],
        ))


def _check_explicit_k_points(
    document: _Document,
    card: _Card,
    option: str,
    issues: list[LintIssue],
) -> None:
    if not card.rows:
        issues.append(_issue(
            "error",
            f"K_POINTS {option} requires a point count and point rows.",
            line=card.line,
        ))
        return
    nks = _positive_integer(card.rows[0][1])
    if nks is None:
        issues.append(_issue(
            "error",
            f"K_POINTS {option} requires a positive integer point count.",
            line=card.rows[0][0],
        ))
        return
    point_rows = card.rows[1:]
    if len(point_rows) != nks:
        issues.append(_issue(
            "error",
            (
                f"K_POINTS {option} declares {nks} point(s) but contains "
                f"{len(point_rows)} row(s)."
            ),
            line=card.rows[0][0],
        ))
    for line, row in point_rows:
        fields = row.split()
        if len(fields) != 4 or any(_real(value) is None for value in fields):
            issues.append(_issue(
                "error",
                f"Each K_POINTS {option} row requires four numeric values.",
                line=line,
            ))

    calculation = str(_values(document, "control").get("calculation", "scf")).lower()
    if option in _K_POINT_PATH_OPTIONS and calculation != "bands":
        issues.append(_issue(
            "error",
            (
                f"K_POINTS {option} defines a band path but "
                f"calculation={calculation!r}."
            ),
            line=card.line,
            suggested_fix=(
                "Use calculation='bands' for a band path, or use integration "
                "k-points for this calculation."
            ),
        ))


def _check_occupations(
    document: _Document,
    issues: list[LintIssue],
) -> None:
    system = _values(document, "system")
    if str(system.get("occupations", "")).lower() != "smearing":
        return
    if not system.get("smearing"):
        issues.append(_issue(
            "error",
            "occupations='smearing' requires the smearing method.",
            line=_value_line(document, "system", "occupations"),
            suggested_fix="smearing = '<method>'",
        ))
    degauss = _real(system.get("degauss"))
    if degauss is None or degauss <= 0:
        issues.append(_issue(
            "error",
            "occupations='smearing' requires a positive degauss value in Ry.",
            line=_value_line(document, "system", "occupations"),
            suggested_fix="degauss = <positive width in Ry>",
        ))


def _species_rows(card: _Card | None) -> list[dict[str, Any]]:
    if card is None:
        return []
    species: list[dict[str, Any]] = []
    for line, row in card.rows:
        fields = row.split()
        if len(fields) < 3:
            continue
        mass = _real(fields[1])
        if mass is None:
            continue
        species.append({
            "label": fields[0],
            "mass_amu": mass,
            "pseudopotential": fields[2],
            "line": line,
        })
    return species


def _position_rows(card: _Card | None) -> list[dict[str, Any]]:
    if card is None:
        return []
    atoms: list[dict[str, Any]] = []
    for line, row in card.rows:
        fields = row.split()
        if len(fields) < 4:
            continue
        coordinates = [_coordinate_expression(value) for value in fields[1:4]]
        if any(value is None for value in coordinates):
            continue
        atoms.append({
            "label": fields[0],
            "coordinates": coordinates,
            "constraints": [_integer(value) for value in fields[4:7]],
            "line": line,
        })
    return atoms


def _numeric_rows(card: _Card | None, *, width: int) -> list[list[float]]:
    if card is None:
        return []
    parsed: list[list[float]] = []
    for _, row in card.rows:
        fields = row.split()
        if len(fields) != width:
            continue
        values = [_real(value) for value in fields]
        if any(value is None for value in values):
            continue
        parsed.append([float(value) for value in values if value is not None])
    return parsed


def _k_points_summary(card: _Card | None) -> dict[str, Any] | None:
    if card is None:
        return None
    option = card.option or "tpiba"
    summary: dict[str, Any] = {"option": option}
    if option == "automatic" and len(card.rows) == 1:
        try:
            values = [int(token) for token in card.rows[0][1].split()]
        except ValueError:
            values = []
        if len(values) == 6:
            summary["grid"] = values[:3]
            summary["shift"] = values[3:]
    elif option != "gamma":
        nks = _positive_integer(card.rows[0][1]) if card.rows else None
        summary["declared_count"] = nks
        summary["points"] = [
            {
                "coordinates": [_real(value) for value in row.split()[:3]],
                "weight": _real(row.split()[3]),
                "line": line,
            }
            for line, row in card.rows[1:]
            if len(row.split()) == 4
            and all(_real(value) is not None for value in row.split())
        ]
    return summary


def _values(document: _Document, name: str) -> dict[str, Any]:
    namelist = document.namelists.get(name)
    return namelist.values if namelist else {}


def _value_line(document: _Document, namelist: str, key: str) -> int | None:
    block = document.namelists.get(namelist)
    return block.lines.get(key) if block else None


def _required_positive_issue(document: _Document, key: str) -> LintIssue:
    return _issue(
        "error",
        f"&SYSTEM requires a positive integer {key} value.",
        line=_value_line(document, "system", key),
        suggested_fix=f"{key} = <positive integer>",
    )


def _issue(
    level: str,
    message: str,
    *,
    line: int | None = None,
    suggested_fix: str | None = None,
) -> LintIssue:
    return {
        "level": level,  # type: ignore[typeddict-item]
        "message": message,
        "line": line,
        "suggested_fix": suggested_fix,
    }


def _strip_comment(line: str) -> str:
    quote: str | None = None
    for index, character in enumerate(line):
        if character in {"'", '"'}:
            if quote == character:
                quote = None
            elif quote is None:
                quote = character
        elif character == "!" and quote is None:
            return line[:index]
    return line


def _split_unquoted(value: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    for index, character in enumerate(value):
        if character in {"'", '"'}:
            if quote == character:
                quote = None
            elif quote is None:
                quote = character
        elif character == separator and quote is None:
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return parts


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().rstrip(",")
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    lowered = cleaned.lower()
    if lowered in {".true.", "true"}:
        return True
    if lowered in {".false.", "false"}:
        return False
    integer = _integer(cleaned)
    if integer is not None:
        return integer
    real = _real(cleaned)
    return real if real is not None else cleaned


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return None


def _positive_integer(value: Any) -> int | None:
    integer = _integer(value)
    return integer if integer is not None and integer > 0 else None


def _real(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if not isinstance(value, str):
        return None
    try:
        parsed = float(value.strip().replace("D", "e").replace("d", "e"))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _coordinate_expression(value: str) -> float | None:
    if (
        not value
        or len(value) > _MAX_COORDINATE_EXPRESSION_LENGTH
        or any(character.isspace() for character in value)
        or value.startswith("+")
        or "**" in value
    ):
        return None
    normalized = _FORTRAN_EXPONENT_RE.sub(
        r"\g<mantissa>e\g<exponent>",
        value.replace("^", "**"),
    )
    try:
        expression = ast.parse(normalized, mode="eval")
        if sum(1 for _ in ast.walk(expression)) > _MAX_COORDINATE_EXPRESSION_NODES:
            return None
        result = _evaluate_coordinate_expression(expression.body)
    except (ArithmeticError, SyntaxError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _evaluate_coordinate_expression(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_coordinate_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
    ):
        left = _evaluate_coordinate_expression(node.left)
        right = _evaluate_coordinate_expression(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if abs(right) > 1024:
            raise ValueError("coordinate exponent exceeds the supported range")
        result = left**right
        if isinstance(result, complex):
            raise ValueError("coordinate expression must be real")
        return float(result)
    raise ValueError("unsupported coordinate expression")


def _card_option(raw: str) -> str | None:
    option = raw.strip()
    if not option:
        return None
    if option[0] in "({" and option[-1] in ")}":
        option = option[1:-1]
    return option.strip().lower() or None


__all__ = [
    "lint_pw_input",
    "parse_pw_input",
    "parse_pw_text",
    "unsupported_qe_program",
]
