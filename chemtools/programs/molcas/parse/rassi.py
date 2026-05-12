"""Parser for the Molcas RASSI (RAS State Interaction) module output block.

Tested against a ZnO singlets+triplets+SOC reference run (5 MPI procs, 6
input states from 2 JOBIPH files, MEES + MESO + dipole + AMFI + Properties).

Captures the high-value pieces an agent needs to reason about a
state-interaction calculation:

  * input_states: which JobIph and root each input state came from, plus
    spin multiplicity / active electrons (from the per-JOBIPH "General data
    section")
  * common_data: per-irrep INACTIVE / ACTIVE / RAS1/2/3 / SECONDARY / BASIS
  * computation_flags: spin-orbit on/off, energy source (EJob vs Heff),
    operators slated for matrix-element printing
  * input_state_energies: `:: RASSI State N Total energy: ...`
  * spin_free_states: per-state EMIN-relative au + eV + cm-1, plus the
    NATURAL ORBITAL occupation vectors for each eigenstate (when NRNATO>0)
  * dipole_oscillator_strengths_spin_free: From, To, oscillator strength,
    Einstein Ax/Ay/Az + total
  * so_states: SO state energies (absolute + EMIN-relative + eV + cm-1) and
    composition (top-N spin-free states + spin + weights)
  * soc_matrix_elements: rows of the "Complex SO-Hamiltonian matrix elements
    over spin components" table (state pairs above SOCOupling threshold)
"""

from __future__ import annotations

import re
from typing import Any


_FLOAT_RE = r"-?\d+\.\d+(?:[Ee][+-]?\d+)?"

# --- General data per JOBIPH ---------------------------------------------------
_JOBIPH_HEADER_RE = re.compile(
    r"Specific data for JOBIPH file\s+(\S+)\s*\n\s*-+\s*\n",
    re.M,
)
_STATE_IRREP_RE = re.compile(r"STATE IRREP:\s+(\d+)")
_SPIN_MULT_RE = re.compile(r"SPIN MULTIPLICITY:\s+(\d+)")
_ACTIVE_E_RE = re.compile(r"ACTIVE ELECTRONS:\s+(\d+)")
_MAX_RAS1_RE = re.compile(r"MAX RAS1 HOLES:\s+(\d+)")
_MAX_RAS3_RE = re.compile(r"MAX RAS3 ELECTRONS:\s+(\d+)")
_NCONFIG_RE = re.compile(r"NR OF CONFIG:\s+(\d+)")
_CASSCF_TITLE_RE = re.compile(r"CASSCF title \(first line only\):\s*\n\s+(\S.*?)\s*$", re.M)

# --- Common data ---------------------------------------------------------------
_NR_IRREPS_RE = re.compile(r"Nr of irreps:\s+(\d+)")
_PER_IRREP_RE = re.compile(
    r"^\s*(INACTIVE|ACTIVE|SECONDARY|BASIS|RAS1|RAS2|RAS3)\s+(\d+)\s+((?:\d+\s+)*\d+)?\s*$",
    re.M,
)

# --- Computation flags ---------------------------------------------------------
_SPIN_ORBIT_FLAG_RE = re.compile(r"EIGENSTATES OF SPIN-ORBIT HAMILTONIAN WILL BE COMPUTED")
_SPIN_FREE_FLAG_RE = re.compile(r"EIGENSTATES OF A SPIN-FREE HAMILTONIAN WILL BE COMPUTED")
_EJOB_RE = re.compile(r"WARNING: EJOB used")
_NRNATO_RE = re.compile(r"NRNATO=\s+(\d+)")
_NSTATES_RE = re.compile(r"Nr of states:\s+(\d+)")

# --- Input-state mapping table -------------------------------------------------
# State:  1  2  3  ...
# JobIph: 1  1  1  ...
# Root nr: 1 2  3  ...
_STATE_LINE_RE = re.compile(r"^\s*State:\s+((?:\d+\s+)+\d+|\d+)\s*$", re.M)
_JOBIPH_LINE_RE = re.compile(r"^\s*JobIph:\s+((?:\d+\s+)+\d+|\d+)\s*$", re.M)
_ROOT_LINE_RE = re.compile(r"^\s*Root nr:\s+((?:\d+\s+)+\d+|\d+)\s*$", re.M)

# --- Input state energies + spin-free / SO state energies ----------------------
_RASSI_STATE_E_RE = re.compile(
    r"::\s*RASSI State\s+(\d+)\s+Total energy:\s+(" + _FLOAT_RE + r")"
)
_SO_RASSI_STATE_E_RE = re.compile(
    r"::\s*SO-RASSI State\s+(\d+)\s+Total energy:\s+(" + _FLOAT_RE + r")"
)

# Spin-free energies table:
#  SF State    Relative EMIN(au)   Rel lowest level(eV)    D:o, cm**(-1)
#      4         0.0000000000        0.0000000000              0.0000
_SF_TABLE_HEADER_RE = re.compile(
    r"SPIN-FREE ENERGIES:\s*\n\s*\(Shifted by EMIN \(a\.u\.\) =\s+(" + _FLOAT_RE + r")\)\s*\n"
    r"\s*\n\s*SF State\s+Relative EMIN.*?\n",
    re.DOTALL,
)
_SF_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*$",
    re.M,
)

# SO eigenvalue table — same layout, different header
_SO_TABLE_HEADER_RE = re.compile(
    r"Eigenvalues of complex Hamiltonian:\s*\n\s*-+\s*\n"
    r"\s*\(Shifted by EMIN \(a\.u\.\) =\s+(" + _FLOAT_RE + r")\)\s*\n"
    r"\s*\n\s*SO State\s+Relative EMIN.*?\n",
    re.DOTALL,
)
_SO_ROW_RE = _SF_ROW_RE  # same shape

# SO composition table:
#  SO State  Total energy (au)           Spin-free states, spin, and weights
#   1        -0.000004       4 1.0  0.9998    5 1.0  0.0002    1 0.0  0.0000 ...
_SO_COMPOSITION_HEADER_RE = re.compile(
    r"Weights of the five most important spin-orbit-free states for each spin-orbit s\s*tate\.",
)
_SO_COMP_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(" + _FLOAT_RE + r")\s+([\d\.\s]+)\s*$",
    re.M,
)

# Dipole transition strengths (spin-free):
#  From   To   Osc. strength   Einstein Ax, Ay, Az          Total A
#     4    6   4.10E-05       2.72E+03  3.91E+03  5.92E-03   6.62E+03
_DIPOLE_SF_HEADER_RE = re.compile(
    r"Dipole transition strengths \(spin-free states\):\s*\n\s*-+\s*\n"
    r"\s*for osc\. strength at least\s+(" + _FLOAT_RE + r")\s*\n"
    r"\s*\n\s*From\s+To\s+Osc\. strength\s+Einstein.*?\n\s*-+\s*\n",
    re.DOTALL,
)
_DIPOLE_SF_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE
    + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*$",
    re.M,
)

# Dipole transition strengths (SO):
_DIPOLE_SO_HEADER_RE = re.compile(
    r"Dipole transition strengths \(SO states\):\s*\n\s*-+\s*\n"
    r"\s*for osc\. strength at least\s+(" + _FLOAT_RE + r")\s*\n"
    r"\s*\n\s*From\s+To\s+Osc\. strength\s+Einstein.*?\n\s*-+\s*\n",
    re.DOTALL,
)

# SOC matrix elements table:
#  I1  S1  MS1    I2  S2  MS2    Real part    Imag part      Absolute
#     5  1.0  0.0    2  0.0  0.0        -0.000       -43.586        43.586
_SOC_HEADER_RE = re.compile(
    r"Complex SO-Hamiltonian matrix elements over\s*\n"
    r"\s*spin components of spin-free eigenstates \(SFS\):\s*\n"
    r"\s*\(In cm-1\. Print threshold:\s+(" + _FLOAT_RE + r")\s+cm-1\)\s*\n"
    r"\s*-+\s*\n"
    r"\s*\n\s*I1\s+S1\s+MS1\s+I2\s+S2\s+MS2.*?\n",
    re.DOTALL,
)
_SOC_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+"
    r"(\d+)\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+"
    r"(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*$",
    re.M,
)

# Natural orbital occupation:
#  NATURAL ORBITALS FOR EIGENSTATE NR N
#  ORBITALS ARE WRITTEN ONTO FILE ID = SIORB.N
#  OCCUPATION NUMBERS:
#  SYMMETRY SPECIES: 1
#  2.00000 2.00000 ... (multiple lines)
_NO_HEADER_RE = re.compile(r"NATURAL ORBITALS FOR EIGENSTATE NR\s+(\d+)")
# "0.99999 1.99842" or "-0.00000 0.00000"; floats may also be smashed together
# without spaces (e.g. "0.99999-0.00000"), so tokenize defensively.
_OCCUPATION_TOKEN_RE = re.compile(r"-?\d+\.\d+(?:[Ee][+-]?\d+)?")


def parse_rassi(text: str) -> dict[str, Any]:
    """Parse the body of a single &RASSI module block.

    Pass in the slice between `--- Start Module: rassi` and `--- Stop Module: rassi`.
    """
    info: dict[str, Any] = {
        "module": "rassi",
        "input_jobiphs": _parse_input_jobiphs(text),
        "common_data": _parse_common_data(text),
        "computation_flags": _parse_computation_flags(text),
        "input_state_mapping": _parse_input_state_mapping(text),
        "input_state_energies": _parse_input_state_energies(text),
        "spin_free_states": _parse_spin_free_states(text),
        "so_states": _parse_so_states(text),
        "so_composition": _parse_so_composition(text),
        "soc_matrix_elements": _parse_soc_matrix_elements(text),
        "dipole_oscillator_strengths_spin_free": _parse_dipole_strengths(text, _DIPOLE_SF_HEADER_RE),
        "dipole_oscillator_strengths_so": _parse_dipole_strengths(text, _DIPOLE_SO_HEADER_RE),
        "natural_orbital_occupations": _parse_natural_orbital_occupations(text),
    }
    info["energy_summary"] = _summarize_energies(info)
    info["soc_summary"] = _summarize_soc(info)
    return info


# --- Sub-parsers ---------------------------------------------------------------

def _parse_input_jobiphs(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    headers = list(_JOBIPH_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else _next_section(text, m.end())
        body = text[m.end():end]
        rec: dict[str, Any] = {"jobiph_label": m.group(1)}
        if (mt := _CASSCF_TITLE_RE.search(body)):
            rec["casscf_title"] = mt.group(1).strip()
        for key, pattern in (
            ("state_irrep", _STATE_IRREP_RE),
            ("spin_multiplicity", _SPIN_MULT_RE),
            ("active_electrons", _ACTIVE_E_RE),
            ("max_ras1_holes", _MAX_RAS1_RE),
            ("max_ras3_electrons", _MAX_RAS3_RE),
            ("n_configs", _NCONFIG_RE),
        ):
            mm = pattern.search(body)
            if mm:
                rec[key] = int(mm.group(1))
        out.append(rec)
    return out


def _parse_common_data(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if (m := _NR_IRREPS_RE.search(text)):
        out["n_irreps"] = int(m.group(1))
    per_irrep: dict[str, list[int]] = {}
    for m in _PER_IRREP_RE.finditer(text):
        label = m.group(1).lower()
        total = int(m.group(2))
        per_irrep_str = m.group(3) or ""
        try:
            per_irrep_vec = [int(x) for x in per_irrep_str.split()] if per_irrep_str.strip() else [total]
        except ValueError:
            per_irrep_vec = [total]
        per_irrep[label] = per_irrep_vec
        out.setdefault("totals", {})[label] = total
    if per_irrep:
        out["per_irrep"] = per_irrep
    return out


def _parse_computation_flags(text: str) -> dict[str, Any]:
    return {
        "compute_spin_free": bool(_SPIN_FREE_FLAG_RE.search(text)),
        "compute_spin_orbit": bool(_SPIN_ORBIT_FLAG_RE.search(text)),
        "energies_from_jobiph_ejob": bool(_EJOB_RE.search(text)),
        "natural_orbitals_requested": (int(m.group(1)) if (m := _NRNATO_RE.search(text)) else 0),
        "n_states": (int(m.group(1)) if (m := _NSTATES_RE.search(text)) else None),
    }


def _parse_input_state_mapping(text: str) -> list[dict[str, Any]]:
    """Walk the State / JobIph / Root nr triple and return one entry per input state."""
    state_match = _STATE_LINE_RE.search(text)
    job_match = _JOBIPH_LINE_RE.search(text)
    root_match = _ROOT_LINE_RE.search(text)
    if not (state_match and job_match and root_match):
        return []
    states = [int(x) for x in state_match.group(1).split()]
    jobiphs = [int(x) for x in job_match.group(1).split()]
    roots = [int(x) for x in root_match.group(1).split()]
    n = min(len(states), len(jobiphs), len(roots))
    return [
        {"state_index": states[i], "jobiph_index": jobiphs[i], "root_in_jobiph": roots[i]}
        for i in range(n)
    ]


def _parse_input_state_energies(text: str) -> list[dict[str, Any]]:
    return [
        {"state_index": int(m.group(1)), "energy_hartree": float(m.group(2))}
        for m in _RASSI_STATE_E_RE.finditer(text)
    ]


def _parse_spin_free_states(text: str) -> dict[str, Any] | None:
    h = _SF_TABLE_HEADER_RE.search(text)
    if not h:
        return None
    e_min = float(h.group(1))
    body = text[h.end():_next_section(text, h.end())]
    rows = []
    for m in _SF_ROW_RE.finditer(body):
        rows.append(
            {
                "sf_state_index": int(m.group(1)),
                "energy_relative_au": float(m.group(2)),
                "energy_relative_ev": float(m.group(3)),
                "energy_relative_cm1": float(m.group(4)),
                "absolute_energy_au": e_min + float(m.group(2)),
            }
        )
        # Stop at first non-row line by limiting to small block — but
        # finditer with re.M handles it via line-anchored regex
    return {"e_min_au": e_min, "rows": rows}


def _parse_so_states(text: str) -> dict[str, Any] | None:
    h = _SO_TABLE_HEADER_RE.search(text)
    if not h:
        return None
    e_min = float(h.group(1))
    # Stop at the next "Weights of" or empty-line block
    body_end = text.find("Weights of the five most", h.end())
    if body_end == -1:
        body_end = h.end() + 4000
    body = text[h.end():body_end]
    rows = []
    for m in _SO_ROW_RE.finditer(body):
        rows.append(
            {
                "so_state_index": int(m.group(1)),
                "energy_relative_au": float(m.group(2)),
                "energy_relative_ev": float(m.group(3)),
                "energy_relative_cm1": float(m.group(4)),
                "absolute_energy_au": e_min + float(m.group(2)),
            }
        )
    # Also fold in the absolute "::    SO-RASSI State N Total energy:" entries
    abs_energies = {
        int(m.group(1)): float(m.group(2))
        for m in _SO_RASSI_STATE_E_RE.finditer(text)
    }
    for r in rows:
        if (abs_e := abs_energies.get(r["so_state_index"])) is not None:
            r["absolute_energy_au"] = abs_e  # prefer the printed absolute
    return {"e_min_au": e_min, "rows": rows}


def _parse_so_composition(text: str) -> list[dict[str, Any]] | None:
    h = _SO_COMPOSITION_HEADER_RE.search(text)
    if not h:
        return None
    body = text[h.end():_next_section(text, h.end())]
    rows = []
    for line in body.splitlines():
        line = line.rstrip()
        if not line.strip() or line.lstrip().startswith("-"):
            continue
        # "    1        -0.000004       4 1.0  0.9998    5 1.0  0.0002    1 0.0  0.0000 ..."
        m = re.match(r"^\s*(\d+)\s+(" + _FLOAT_RE + r")\s+(.+)$", line)
        if not m:
            continue
        contributions: list[dict[str, Any]] = []
        rest = m.group(3).split()
        i = 0
        while i + 2 < len(rest):
            try:
                sf_state = int(rest[i])
                spin = float(rest[i + 1])
                weight = float(rest[i + 2])
            except ValueError:
                break
            contributions.append({"sf_state": sf_state, "spin": spin, "weight": weight})
            i += 3
        rows.append(
            {
                "so_state_index": int(m.group(1)),
                "energy_relative_au": float(m.group(2)),
                "top_contributions": contributions,
            }
        )
    return rows


def _parse_soc_matrix_elements(text: str) -> dict[str, Any] | None:
    h = _SOC_HEADER_RE.search(text)
    if not h:
        return None
    threshold = float(h.group(1))
    body_end = text.find("Total energies including SO-coupling", h.end())
    if body_end == -1:
        body_end = h.end() + 50000
    body = text[h.end():body_end]
    rows = []
    for m in _SOC_ROW_RE.finditer(body):
        rows.append(
            {
                "i1": int(m.group(1)),
                "s1": float(m.group(2)),
                "ms1": float(m.group(3)),
                "i2": int(m.group(4)),
                "s2": float(m.group(5)),
                "ms2": float(m.group(6)),
                "real_cm1": float(m.group(7)),
                "imag_cm1": float(m.group(8)),
                "absolute_cm1": float(m.group(9)),
            }
        )
    return {
        "threshold_cm1": threshold,
        "rows": rows,
        "n_above_threshold": len(rows),
    }


def _parse_dipole_strengths(text: str, header_re: re.Pattern) -> dict[str, Any] | None:
    h = header_re.search(text)
    if not h:
        return None
    threshold = float(h.group(1))
    body = text[h.end():_next_section(text, h.end())]
    rows = []
    for m in _DIPOLE_SF_ROW_RE.finditer(body):
        rows.append(
            {
                "from_state": int(m.group(1)),
                "to_state": int(m.group(2)),
                "oscillator_strength": float(m.group(3)),
                "einstein_ax": float(m.group(4)),
                "einstein_ay": float(m.group(5)),
                "einstein_az": float(m.group(6)),
                "einstein_total": float(m.group(7)),
            }
        )
    return {
        "threshold": threshold,
        "rows": rows,
        "n_transitions": len(rows),
    }


def _parse_natural_orbital_occupations(text: str) -> list[dict[str, Any]]:
    """Per-eigenstate natural orbital occupation vectors."""
    out: list[dict[str, Any]] = []
    headers = list(_NO_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        next_marker_pos = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        # Stop at a clear non-NO marker
        for stop_marker in ("NATURAL SPIN ORBITALS", "Spin-orbit section", "*****"):
            p = text.find(stop_marker, m.end(), next_marker_pos)
            if p != -1:
                next_marker_pos = min(next_marker_pos, p)
        body = text[m.end():next_marker_pos]
        # Walk lines until OCCUPATION NUMBERS, then collect floats per symmetry block
        per_sym: dict[int, list[float]] = {}
        current_sym: int | None = None
        in_occ = False
        for line in body.splitlines():
            if "OCCUPATION NUMBERS" in line:
                in_occ = True
                continue
            if not in_occ:
                continue
            if (sm := re.match(r"\s*SYMMETRY SPECIES:\s+(\d+)", line)):
                current_sym = int(sm.group(1))
                per_sym[current_sym] = []
                continue
            tokens = _OCCUPATION_TOKEN_RE.findall(line)
            if tokens and current_sym is not None:
                per_sym[current_sym].extend(float(t) for t in tokens)
            elif not line.strip():
                continue
            elif line.lstrip().startswith(("***", "NATURAL", "ORBITAL")):
                break
        if per_sym:
            all_occs = [o for occs in per_sym.values() for o in occs]
            out.append(
                {
                    "eigenstate": int(m.group(1)),
                    "occupations_by_symmetry": per_sym,
                    "n_orbitals_total": len(all_occs),
                    "active_window_occupations": [o for o in all_occs if 0.02 <= o <= 1.98],
                }
            )
    return out


def _next_section(text: str, start: int) -> int:
    """Find the next clear section divider after `start`."""
    candidates = [
        text.find("\n\n\n", start),
        text.find("****************", start),
        text.find("\n--- ", start),
        text.find("++ ", start + 1),
    ]
    pos = [c for c in candidates if c != -1]
    return min(pos) if pos else len(text)


def _summarize_energies(info: dict[str, Any]) -> dict[str, Any]:
    sf = info.get("spin_free_states")
    so = info.get("so_states")
    summary: dict[str, Any] = {}
    if sf and sf["rows"]:
        summary["spin_free_ground_au"] = min(r["absolute_energy_au"] for r in sf["rows"])
        summary["n_spin_free_states"] = len(sf["rows"])
    if so and so["rows"]:
        summary["spin_orbit_ground_au"] = min(r["absolute_energy_au"] for r in so["rows"])
        summary["n_spin_orbit_states"] = len(so["rows"])
        # Lowest singlet → triplet etc. structure isn't computable without
        # composition info; emit total SOC stabilization instead.
        if sf and sf["rows"]:
            summary["soc_stabilization_au"] = (
                summary["spin_orbit_ground_au"] - summary["spin_free_ground_au"]
            )
            summary["soc_stabilization_cm1"] = summary["soc_stabilization_au"] * 219474.63
    return summary


def _summarize_soc(info: dict[str, Any]) -> dict[str, Any] | None:
    soc = info.get("soc_matrix_elements")
    if not soc:
        return None
    rows = soc.get("rows") or []
    if not rows:
        return {"threshold_cm1": soc.get("threshold_cm1"), "n_above_threshold": 0}
    sorted_rows = sorted(rows, key=lambda r: -r["absolute_cm1"])
    return {
        "threshold_cm1": soc.get("threshold_cm1"),
        "n_above_threshold": len(rows),
        "max_abs_cm1": sorted_rows[0]["absolute_cm1"],
        "top_5_largest": sorted_rows[:5],
    }
