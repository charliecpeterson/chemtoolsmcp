"""DIRAC text-output parser.

The DIRAC output format is section-driven (the input echo, banner, then
sections per task: SCF, MOLTRA, MP2, COSCI, etc.). This parser extracts:

- Banner version
- Echoed input (.inp + .mol) — for re-parsing what the user requested
- SCF iteration trace (energy, deltaE, gradient, DIIS step, label per iter)
- Final SCF / DFT / total energy
- Per-symmetry HOMO/LUMO blocks from the RESOLVE step
- Symmetry orbital counts (per-irrep dimension, large/small)
- Mulliken population (MULPOP) output — per-atom totals + per-spinor detail
- Spinor eigenvalue spectrum (all Electronic eigenvalue no. lines)
- COSCI state energies (when the COSCI CI module ran)
- Open-shell setup detection (.CLOSED SHELL + .OPEN SHELL directives)

Higher-level interpretation (verdicts, recovery suggestions, MO swap
recommendations) lives in chemtools/programs/dirac/strategy/.
"""

from __future__ import annotations

import re
from typing import Any


# DIRAC banner pattern — the @@ ASCII art lines are stable across versions.
_BANNER_RE = re.compile(r"^\s*\*+\s*$\s*\*.*DIRAC", re.MULTILINE | re.DOTALL)
_VERSION_RE = re.compile(r"Release\s+(?:DIRAC[\s_-]*)?([\d.]+)", re.IGNORECASE)
_SCF_ITER_RE = re.compile(
    r"^\s*It\.\s+(\d+)\s+(-?\d+\.\d+(?:[ED][+-]?\d+)?)\s+"
    r"([\d.+\-DE]+)\s+([\d.+\-DE]+)\s+([\d.+\-DE]+)"
    r"(?:\s+(DIIS)\s+(\d+))?",
    re.MULTILINE,
)
_TOTAL_ENERGY_RE = re.compile(
    r"^\s*Total energy\s+:\s+(-?\d+\.\d+)", re.MULTILINE
)
_HOMO_LUMO_RE = re.compile(
    r"E_HOMO\.\.\.E_LUMO,\s+symmetry\s+(\d+):\s+(.+)$", re.MULTILINE
)
_SYM_ORB_COUNTS_RE = re.compile(
    r"Number of orbitals in each symmetry:\s+((?:\s*-?\d+)+)", re.MULTILINE
)
_LARGE_ORB_COUNTS_RE = re.compile(
    r"Number of large orbitals in each symmetry:\s+((?:\s*-?\d+)+)", re.MULTILINE
)
_SMALL_ORB_COUNTS_RE = re.compile(
    r"Number of small orbitals in each symmetry:\s+((?:\s*-?\d+)+)", re.MULTILINE
)
_SYMMETRY_DETECTED_RE = re.compile(
    r"The following symmetry elements were found:\s+(.+)$", re.MULTILINE
)
# Spinor eigenvalue lines (MULPOP VECPOP output):
#   * Electronic eigenvalue no.   1: -4751.5641396016   (Occupation : f = 1.0000)  s 1/2;  1/2
# j_label format: "s 1/2", "p 3/2", "d 5/2", "f 7/2" — includes slash, space, digits
_SPINOR_EIGEN_RE = re.compile(
    r"^\*\s+Electronic eigenvalue no\.\s+(\d+):\s+(-?[\d.]+)\s+"
    r"\(Occupation\s*:\s*f\s*=\s*([\d.]+)\)\s+([\w /]+?);\s*([-\d/ ]+)",
    re.MULTILINE,
)
# Per-spinor MULPOP block start (same pattern, anchored to end-of-line)
_MULPOP_SPINOR_RE = re.compile(
    r"^\*\s+Electronic eigenvalue no\.\s+(\d+):\s+(-?[\d.]+)\s+"
    r"\(Occupation\s*:\s*f\s*=\s*([\d.]+)\)\s+([\w /]+?);\s*([-\d/ ]+)\s*$",
    re.MULTILINE,
)
_MULPOP_GROSS_ALPHA_RE = re.compile(
    r"^\s+alpha\s+([\d.]+)\s+\|\s+(.*)", re.MULTILINE
)
_MULPOP_GROSS_BETA_RE = re.compile(
    r"^\s+beta\s+([\d.]+)\s+\|\s+(.*)", re.MULTILINE
)


def looks_like_dirac(head: str) -> bool:
    """Detect a DIRAC output from the first ~8 KB of text.

    DIRAC outputs all carry the ``@@@@@`` ASCII art banner and at least one
    "DIRAC master/nodes" allocation line. We accept either as the trigger,
    plus the ``Release DIRAC`` and ``pam-dirac`` strings for back-compat
    with stdout-captured runs that strip the banner.
    """
    return (
        "DIRAC" in head
        and (
            "@@@@@" in head
            or "DIRAC master" in head
            or "Release DIRAC" in head
            or "pam-dirac" in head
        )
    )


def parse_version(text: str) -> str | None:
    m = _VERSION_RE.search(text[:5000])
    return m.group(1) if m else None


def parse_scf_iterations(text: str) -> list[dict[str, Any]]:
    """Return the SCF iteration trace: one dict per `It. N` line."""
    out: list[dict[str, Any]] = []
    for m in _SCF_ITER_RE.finditer(text):
        out.append({
            "iter": int(m.group(1)),
            "energy_hartree": _to_float(m.group(2)),
            "delta_e": _to_float(m.group(3)),
            "gradient_max": _to_float(m.group(4)),
            "step_size": _to_float(m.group(5)),
            "diis_n": int(m.group(7)) if m.group(7) else None,
        })
    return out


def parse_total_energy(text: str) -> float | None:
    matches = list(_TOTAL_ENERGY_RE.finditer(text))
    if not matches:
        return None
    return _to_float(matches[-1].group(1))


def parse_symmetry(text: str) -> dict[str, Any]:
    """Detected molecular symmetry + per-irrep orbital counts."""
    out: dict[str, Any] = {"point_group_elements": None}
    m = _SYMMETRY_DETECTED_RE.search(text)
    if m:
        elems = m.group(1).split()
        out["point_group_elements"] = elems
    for key, rx in (
        ("orbitals_per_symmetry", _SYM_ORB_COUNTS_RE),
        ("large_per_symmetry", _LARGE_ORB_COUNTS_RE),
        ("small_per_symmetry", _SMALL_ORB_COUNTS_RE),
    ):
        m = rx.search(text)
        if m:
            out[key] = [int(x) for x in m.group(1).split()]
    return out


def parse_homo_lumo_blocks(text: str) -> list[dict[str, Any]]:
    """Per-symmetry HOMO/LUMO lines from a RESOLVE section.

    Each line looks like:
      ``E_HOMO...E_LUMO, symmetry 1:    151  -0.13431  152  -0.06816``

    Returns one dict per match with symmetry index + parsed orbital list.
    DIRAC prints multiple of these (one per SCF iter that hit RESOLVE),
    so the last set is usually the converged state.
    """
    out: list[dict[str, Any]] = []
    for m in _HOMO_LUMO_RE.finditer(text):
        tokens = m.group(2).split()
        # Tokens alternate (index, energy, index, energy, ...)
        orbitals: list[dict[str, float]] = []
        for i in range(0, len(tokens) - 1, 2):
            try:
                orbitals.append({
                    "index": int(tokens[i]),
                    "energy_hartree": _to_float(tokens[i + 1]),
                })
            except ValueError:
                continue
        out.append({
            "symmetry": int(m.group(1)),
            "orbitals": orbitals,
        })
    return out


def parse_open_shell_setup(text: str) -> dict[str, Any] | None:
    """Detect .CLOSED SHELL / .OPEN SHELL configuration from the echoed input.

    Returns ``{closed_shell: [...], open_shell: [...], aoc: bool}`` or None
    if the run is a plain (no shell directive) closed-shell SCF.
    """
    closed = re.search(
        r"\.CLOSED SHELL\s*\n((?:\s+\d.*\n)+)", text
    )
    opened = re.search(
        r"\.OPEN SHELL\s*\n\s*(\d+)\s*\n((?:\s+[\d/,]+.*\n)+)", text
    )
    if not closed and not opened:
        return None
    result: dict[str, Any] = {"aoc": bool(opened)}
    if closed:
        result["closed_shell"] = [
            int(x) for x in closed.group(1).split()
        ]
    if opened:
        n_open_blocks = int(opened.group(1))
        # Each line is "n_electrons/orb_a,orb_b,..."
        open_lines = opened.group(2).strip().splitlines()
        blocks: list[dict[str, Any]] = []
        for ln in open_lines[:n_open_blocks]:
            ln = ln.strip()
            if "/" in ln:
                n_e, orb_spec = ln.split("/", 1)
                blocks.append({
                    "n_electrons": int(n_e.strip()),
                    "orbital_spec": orb_spec.strip(),
                })
        result["open_shell_blocks"] = blocks
    return result


def detect_task_kinds(text: str) -> list[str]:
    """Identify which DIRAC tasks ran (SCF, DFT, MP2, COSCI, MOLTRA, ...).

    Driven by the .WAVE FUNCTION / .ANALYZE / etc. echoed directives plus
    section headers in the output.
    """
    kinds: list[str] = []
    if re.search(r"\.SCF\s*\n", text):
        # SCF can mean HF or DFT depending on HAMILTONIAN block
        if re.search(r"\*\*HAMILTONIAN\s*\n(?:[^*]*\n)*?\.DFT", text):
            kinds.append("dft")
        else:
            kinds.append("scf")
    if "MP2" in text and re.search(r"\*\*?\s*MOLTRA|\.MP2\s*\n", text):
        kinds.append("mp2")
    if re.search(r"\.COSCI\s*\n|COSCI module", text):
        kinds.append("cosci")
    if re.search(r"\*\*\s*KRCI|\.KRCICALC", text):
        kinds.append("krci")
    if re.search(r"\*\*\s*RELCCSD|\.CCSD", text):
        kinds.append("ccsd")
    if re.search(r"\.TDA|\.RESPON", text):
        kinds.append("response")
    return kinds


def parse_mulliken(text: str) -> dict[str, Any] | None:
    """Extract MULPOP / Mulliken-style population output if present.

    DIRAC's MULPOP prints electronic+spin population per atom and angular-
    momentum decomposition (vec-pop). We return a flat dict with the per-
    atom totals; full breakdown stays in raw_text.
    """
    block = re.search(
        r"Mulliken population analysis\b.*?(?=^\s*\*|\Z)",
        text, re.DOTALL | re.MULTILINE,
    )
    if not block:
        return None
    body = block.group(0)
    atom_lines = re.findall(
        r"^\s+(\w+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)?", body, re.MULTILINE
    )
    return {
        "atoms": [
            {
                "label": a[0],
                "electronic_population": _to_float(a[1]),
                "spin_population": _to_float(a[2]) if a[2] else None,
            }
            for a in atom_lines[:50]
        ],
        "raw_text_snippet": body[:500],
    }


def parse_spinor_spectrum(text: str) -> list[dict]:
    """Parse the full spinor eigenvalue spectrum from MULPOP VECPOP output.

    Each "Electronic eigenvalue no. N: E (Occupation : f = OCC)  j_label; mj"
    line is parsed into a dict.  Positronic spinors (very large positive
    energies, > 37500 Ha) are omitted — they are unphysical artefacts of the
    4-component basis.

    Returns a list sorted by eigenvalue number, with:
      index, energy_hartree, occupation, j_label, mj, angular_momentum
    """
    out: list[dict] = []
    for m in _SPINOR_EIGEN_RE.finditer(text):
        energy = _to_float(m.group(2))
        if energy is not None and energy > 37500:
            # Positronic (small-component) spinors — skip
            continue
        j_label = m.group(4).strip()  # e.g. "s 1/2", "f 7/2"
        parts = j_label.split()
        ang_mom = parts[0] if parts else None  # s/p/d/f/g
        out.append({
            "index": int(m.group(1)),
            "energy_hartree": energy,
            "occupation": _to_float(m.group(3)),
            "j_label": j_label,
            "mj": m.group(5).strip(),
            "angular_momentum": ang_mom,
        })
    return out


def parse_mulliken_detail(text: str) -> list[dict]:
    """Parse per-spinor MULPOP gross-population blocks.

    For each occupied spinor where DIRAC printed a MULPOP block, returns:
      index, energy_hartree, occupation, j_label, mj,
      alpha_total, beta_total,
      alpha_by_label: {label: population},
      beta_by_label:  {label: population}

    Only spinors that appear in the MULPOP section are returned (DIRAC
    skips spinors with no gross population > the print threshold).
    """
    # Find the Mulliken population analysis block
    block_m = re.search(
        r"Mulliken population analysis\b.*?\*{40,}", text,
        re.DOTALL | re.MULTILINE,
    )
    if not block_m:
        return []
    body = text[block_m.start():]

    results: list[dict] = []
    # Split on spinor header lines
    spinor_headers = list(_MULPOP_SPINOR_RE.finditer(body))
    for i, hdr in enumerate(spinor_headers):
        end = spinor_headers[i + 1].start() if i + 1 < len(spinor_headers) else len(body)
        chunk = body[hdr.start():end]

        energy = _to_float(hdr.group(2))
        if energy is not None and energy > 37500:
            continue  # positronic

        j_label = hdr.group(4).strip()
        parts = j_label.split()
        ang_mom = parts[0] if parts else None

        # Column labels line — between separator (====) and alpha/beta rows
        col_header_m = re.search(
            r"Gross\s+Total\s+\|(.*)", chunk
        )
        col_labels: list[str] = []
        if col_header_m:
            raw = col_header_m.group(1)
            # Labels are fixed-width columns of ~15 chars; split by 2+ spaces
            col_labels = [c.strip() for c in re.split(r"\s{2,}", raw.strip()) if c.strip()]

        # Alpha row
        alpha_m = _MULPOP_GROSS_ALPHA_RE.search(chunk)
        alpha_total = None
        alpha_by_label: dict[str, float] = {}
        if alpha_m:
            alpha_total = _to_float(alpha_m.group(1))
            vals = [_to_float(v) for v in alpha_m.group(2).split()]
            for label, val in zip(col_labels, vals):
                if val is not None and val > 0.001:
                    alpha_by_label[label] = val

        # Beta row
        beta_m = _MULPOP_GROSS_BETA_RE.search(chunk)
        beta_total = None
        beta_by_label: dict[str, float] = {}
        if beta_m:
            beta_total = _to_float(beta_m.group(1))
            vals = [_to_float(v) for v in beta_m.group(2).split()]
            for label, val in zip(col_labels, vals):
                if val is not None and val > 0.001:
                    beta_by_label[label] = val

        results.append({
            "index": int(hdr.group(1)),
            "energy_hartree": energy,
            "occupation": _to_float(hdr.group(3)),
            "j_label": j_label,
            "mj": hdr.group(5).strip(),
            "angular_momentum": ang_mom,
            "alpha_total": alpha_total,
            "beta_total": beta_total,
            "alpha_by_label": alpha_by_label,
            "beta_by_label": beta_by_label,
        })
    return results


def parse_cosci_energies(text: str) -> dict | None:
    """Parse COSCI state-energy table if present.

    DIRAC prints::

        Obtained COSCI states are as follows:

        1        0.000000000          0.000000   1   1   1   1   0 ...
        2        2.972174508      23972.206548   1   1   1   1   0 ...

    Columns: state_index, relative_energy_eV, relative_energy_cm1,
             spinor_occupations...

    Returns None if no COSCI output is found.  Otherwise returns::

        {
          "n_states": int,
          "states": [
            {
              "state": int,          # 1-based
              "energy_ev": float,    # relative to ground state
              "energy_cm1": float,   # relative to ground state
              "spinor_occupations": [int, ...]
            },
            ...
          ],
          "ground_energy_hartree": float | None,  # absolute SCF energy before COSCI
        }
    """
    m = re.search(
        r"Obtained COSCI states are as follows:\s*\n((?:\s+\d+\s+[\d.]+\s+[\d.]+.*\n)+)",
        text, re.MULTILINE,
    )
    if not m:
        return None
    table_text = m.group(1)
    states: list[dict] = []
    for line in table_text.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            e_ev = _to_float(parts[1])
            e_cm1 = _to_float(parts[2])
            occ = [int(x) for x in parts[3:] if x.isdigit()]
        except (ValueError, IndexError):
            continue
        states.append({
            "state": idx,
            "energy_ev": e_ev,
            "energy_cm1": e_cm1,
            "spinor_occupations": occ,
        })
    if not states:
        return None
    # Try to grab the absolute SCF energy that preceded the COSCI run
    total_e = parse_total_energy(text)
    return {
        "n_states": len(states),
        "states": states,
        "ground_energy_hartree": total_e,
    }


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # DIRAC uses D for Fortran double, e.g. "3.87D+01"
    s = s.replace("D", "E").replace("d", "e")
    try:
        return float(s)
    except ValueError:
        return None


def parse_output(path: str, contents: str | None = None) -> dict[str, Any]:
    """Single-pass parse of a DIRAC text output. Returns a flat dict.

    This is the "cheap default" — fits in agent context. Drill-down parsers
    (orbital coefficients, full MULPOP detail, per-state CI vectors) load
    on demand via the plugin's get_orbitals / get_state_analysis methods.
    """
    if contents is None:
        with open(path, encoding="utf-8", errors="replace") as f:
            contents = f.read()

    scf_iters = parse_scf_iterations(contents)
    total_energy = parse_total_energy(contents)
    symmetry = parse_symmetry(contents)
    open_shell = parse_open_shell_setup(contents)
    homo_lumo = parse_homo_lumo_blocks(contents)
    tasks = detect_task_kinds(contents)
    mulliken = parse_mulliken(contents)
    version = parse_version(contents)
    spinor_spectrum = parse_spinor_spectrum(contents)
    mulliken_detail = parse_mulliken_detail(contents)
    cosci = parse_cosci_energies(contents) if "cosci" in tasks else None

    converged = bool(scf_iters) and abs(
        (scf_iters[-1].get("delta_e") or 1.0)
    ) < 1e-7 if scf_iters else False

    # Per-atom Mulliken charge summary (electronic pop → charge = Z - pop)
    mulliken_charges: list[dict] | None = None
    if mulliken and mulliken.get("atoms"):
        mulliken_charges = mulliken["atoms"]

    return {
        "program": "dirac",
        "program_version": version,
        "file": path,
        "total_energy_hartree": total_energy,
        "scf_iterations": scf_iters,
        "scf_n_iterations": len(scf_iters),
        "scf_converged": converged,
        "symmetry": symmetry,
        "open_shell_setup": open_shell,
        "homo_lumo_per_symmetry": homo_lumo,
        "tasks_detected": tasks,
        "mulliken": mulliken,
        "mulliken_charges": mulliken_charges,
        # Spinor spectrum: index, energy, occupation, j_label, mj
        # Only electronic spinors (positronic > 37500 Ha stripped).
        # Empty list if .VECPOP was not requested in the input.
        "spinor_spectrum": spinor_spectrum,
        # Count spinors with any occupation (includes fractional AOC occupations).
        # Use this to verify total electron count = sum of occupations × 2 (Kramers).
        "n_occupied_spinors": sum(
            1 for s in spinor_spectrum if (s.get("occupation") or 0) > 0
        ),
        # Per-spinor MULPOP detail with alpha/beta populations by AO label.
        # Empty list if .MULPOP was not in the input.
        "mulliken_detail": mulliken_detail,
        # COSCI state energies (eV + cm-1, relative to ground state).
        # None if COSCI was not run.
        "cosci": cosci,
    }
