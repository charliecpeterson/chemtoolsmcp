"""Bundled OpenMolcas documentation accessor.

Mirrors chemtools/programs/nwchem/docs.py but adapts for the Molcas docs
layout, which is a tree of subdirectories (programs/, users_guide/,
tutorials/, advanced_examples/, installation/, overview/) of Markdown files
sourced from the OpenMolcas Sphinx site.

Public API: list_docs, search_docs, lookup_module_syntax, find_examples,
read_doc_excerpt, get_topic_guide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chemtools as _chemtools_pkg


_DATA_DIR = Path(_chemtools_pkg.__file__).resolve().parent / "data" / "molcas"
DOCS_ROOT = _DATA_DIR / "docs"


# Subdirectories that contain example-style content. Hits in these dirs get
# a small ranking bonus when callers ask for examples.
_EXAMPLE_SUBDIRS = ("advanced_examples", "tutorials")


@dataclass(frozen=True)
class _DocMatch:
    file_path: Path
    line_number: int
    line_text: str
    score: float


def _iter_doc_files() -> list[Path]:
    if not DOCS_ROOT.is_dir():
        return []
    return sorted(p for p in DOCS_ROOT.rglob("*.md") if p.is_file())


def _rel(path: Path) -> str:
    return str(path.relative_to(DOCS_ROOT))


def list_docs() -> list[dict[str, Any]]:
    return [
        {
            "name": _rel(p),
            "size_bytes": p.stat().st_size,
        }
        for p in _iter_doc_files()
    ]


def search_docs(
    query: str,
    *,
    max_results: int = 8,
    context_lines: int = 2,
    subdir: str | None = None,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("query must be non-empty")
    tokens = _tokenize(query)
    phrase = query.casefold()
    matches: list[_DocMatch] = []

    files = _iter_doc_files()
    if subdir:
        norm = subdir.strip("/").casefold()
        files = [p for p in files if _rel(p).casefold().startswith(norm + "/")]

    for file_path in files:
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel_lc = _rel(file_path).casefold()
        file_name_bonus = 1.5 if any(token in rel_lc for token in tokens) else 0.0
        for idx, line in enumerate(lines, start=1):
            line_lc = line.casefold()
            score = file_name_bonus
            if phrase in line_lc:
                score += 12.0
            token_hits = sum(1 for token in tokens if token in line_lc)
            if token_hits == 0:
                continue
            score += float(token_hits * 2)
            if _looks_like_heading(line):
                score += 1.0
            matches.append(
                _DocMatch(
                    file_path=file_path,
                    line_number=idx,
                    line_text=line.strip(),
                    score=score,
                )
            )

    ranked = sorted(matches, key=lambda m: (-m.score, _rel(m.file_path), m.line_number))
    deduped: list[_DocMatch] = []
    seen: set[tuple[str, int]] = set()
    for match in ranked:
        key = (str(match.file_path), match.line_number)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
        if len(deduped) >= max_results:
            break

    results = [_format_search_match(m, context_lines=context_lines) for m in deduped]
    return {"query": query, "result_count": len(results), "results": results}


def lookup_module_syntax(module_name: str, *, max_results: int = 1) -> dict[str, Any]:
    """Find the dedicated programs/<module>.md page for a Molcas module."""
    name = module_name.strip().lower()
    if not name:
        raise ValueError("module_name must be non-empty")
    candidate = DOCS_ROOT / "programs" / f"{name}.md"
    if not candidate.exists():
        # Try to find by partial match
        prog_dir = DOCS_ROOT / "programs"
        if prog_dir.is_dir():
            for p in prog_dir.iterdir():
                if p.stem.lower() == name or p.stem.lower().startswith(name):
                    candidate = p
                    break
    if not candidate.exists():
        return {"module": name, "found": False, "result_count": 0, "results": []}
    excerpt = read_doc_excerpt(_rel(candidate), start_line=1, end_line=120)
    return {
        "module": name,
        "found": True,
        "doc_path": _rel(candidate),
        "result_count": 1,
        "results": [excerpt],
    }


def find_examples(topic: str, *, max_results: int = 6) -> dict[str, Any]:
    topic = topic.strip()
    if not topic:
        raise ValueError("topic must be non-empty")
    tokens = _tokenize(topic)
    candidates: list[_DocMatch] = []
    for file_path in _iter_doc_files():
        rel_lc = _rel(file_path).casefold()
        if not any(rel_lc.startswith(d + "/") for d in _EXAMPLE_SUBDIRS):
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, start=1):
            line_lc = line.casefold()
            token_hits = sum(1 for token in tokens if token in line_lc)
            if token_hits == 0:
                continue
            score = float(token_hits * 3)
            if _looks_like_heading(line):
                score += 1.5
            candidates.append(
                _DocMatch(
                    file_path=file_path,
                    line_number=idx,
                    line_text=line.strip(),
                    score=score,
                )
            )
    ranked = sorted(candidates, key=lambda m: (-m.score, _rel(m.file_path), m.line_number))
    results = [_format_search_match(m, context_lines=4) for m in ranked[:max_results]]
    return {"topic": topic, "result_count": len(results), "results": results}


def read_doc_excerpt(
    doc_name: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    query: str | None = None,
    context_lines: int = 8,
) -> dict[str, Any]:
    path = Path(doc_name)
    if not path.is_absolute():
        path = (DOCS_ROOT / doc_name).resolve()
        # Guard against escaping DOCS_ROOT via "../..".
        if DOCS_ROOT not in path.parents and path != DOCS_ROOT:
            raise ValueError(f"path escapes docs root: {doc_name}")
    if not path.exists():
        raise ValueError(f"doc not found: {doc_name}")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if query:
        query_lc = query.casefold()
        for idx, line in enumerate(lines, start=1):
            if query_lc in line.casefold():
                start = max(1, idx - context_lines)
                end = min(len(lines), idx + context_lines)
                return _excerpt_payload(path, lines, start, end, matched_line=idx)
        raise ValueError(f"query not found in {_rel(path)}")
    start = max(1, start_line or 1)
    end = min(len(lines), end_line or min(len(lines), start + context_lines * 2))
    return _excerpt_payload(path, lines, start, end, matched_line=None)


# Topic catalog — high-value Molcas concepts the agent needs to look up
# before drafting inputs (especially around active-space design and
# multi-reference workflows).
_TOPICS: dict[str, dict[str, Any]] = {
    "rasscf_active_space": {
        "summary": (
            "RASSCF active spaces are partitioned into Inactive / RAS1 / RAS2 / RAS3 / "
            "Secondary spaces. RAS2 is the CAS-like full-CI region. RAS1 holds orbitals "
            "from which a limited number of holes are allowed; RAS3 holds orbitals into "
            "which a limited number of electrons can be promoted. Selection requires "
            "physical insight (chemically active orbitals) plus inspection of starting "
            "MOs (GuessOrb / SCF) for orbital character."
        ),
        "search": "RASSCF Inactive RAS1 RAS2 RAS3 active space NACTEL",
        "modules": ("rasscf",),
    },
    "caspt2_setup": {
        "summary": (
            "CASPT2 builds on a RASSCF reference. Critical inputs: state symmetry and "
            "root selection (matching the RASSCF roots), IPEA shift (default 0.25 from "
            "Molcas 6.4 unless MOLCAS_NEW_DEFAULTS=YES), level shifts (SHIFT or "
            "IMAGINARY SHIFT for intruder states), and choice of MS / XMS / RMS / XDW "
            "for state-mixing problems. Always check reference weight ≥ 0.7."
        ),
        "search": "CASPT2 IPEA SHIFT IMAGINARY MULTISTATE XMS reference weight intruder",
        "modules": ("caspt2",),
    },
    "xms_caspt2": {
        "summary": (
            "Use XMS-CASPT2 (or RMS / XDW variants) when CASSCF states of the same "
            "symmetry mix strongly under the dynamic correlation perturbation — common "
            "for closely-spaced excited states or valence/Rydberg mixing."
        ),
        "search": "XMS XMULTISTATE RMS XDW CASPT2 mixed states",
        "modules": ("caspt2",),
    },
    "ipea_shift": {
        "summary": (
            "The IPEA shift modifies the zeroth-order Hamiltonian to reduce systematic "
            "errors of CASPT2. Default 0.25 since Molcas 6.4. Setting IPEAShift=0.0 "
            "reproduces the original CASPT2."
        ),
        "search": "IPEA shift Hamiltonian zeroth order",
        "modules": ("caspt2",),
    },
    "alaska_gradients": {
        "summary": (
            "ALASKA computes nuclear gradients. For analytical CASPT2 / RASPT2 "
            "gradients, the CASPT2 input must include the GRDT keyword to precompute "
            "quantities used in MCLR and ALASKA."
        ),
        "search": "ALASKA gradient GRDT MCLR analytic",
        "modules": ("alaska",),
    },
    "mclr_freq": {
        "summary": (
            "MCLR (multiconfigurational linear response) provides analytic Hessians for "
            "MCSCF / RASSCF references. Pair it with MCKINLEY for second derivative "
            "integrals."
        ),
        "search": "MCLR MCKINLEY Hessian second derivative analytic",
        "modules": ("mclr", "mckinley"),
    },
    "rassi_state_interaction": {
        "summary": (
            "RASSI (RAS State-Interaction) computes overlaps and matrix elements "
            "between non-orthogonal CASSCF / RASSCF wave functions: spin-orbit coupling, "
            "transition moments, NACMEs."
        ),
        "search": "RASSI state interaction spin orbit transition moment",
        "modules": ("rassi",),
    },
    "inporb_format": {
        "summary": (
            "INPORB (alias RasOrb / ScfOrb / GssOrb) is the human-readable orbital "
            "file. Sections: #INPORB <version>, #INFO (per-symmetry orbital counts), "
            "#ORB (MO coefficients), #OCC (occupation numbers), #INDEX (typeindex: "
            "I/A/2/3/S = inactive/RAS1/RAS2/RAS3/secondary). Use INPORB editing to "
            "reorder orbitals before RASSCF."
        ),
        "search": "INPORB INFO ORB OCC INDEX type index inactive RAS",
        "modules": ("rasscf",),
    },
    "scf_setup": {
        "summary": (
            "Molcas SCF supports closed-shell, ROHF (charge / spin), and UHF (KSDFT). "
            "Use GUESSORB for a first guess; use HFMP2 for MP2."
        ),
        "search": "SCF charge spin UHF ROHF GuessOrb",
        "modules": ("scf",),
    },
    "atomization_workflow": {
        "summary": (
            "Atomization / binding-energy workflow at CASSCF / CASPT2: (1) Call "
            "prepare_molcas_atomization with molecule + multiplicity + basis. The "
            "orchestrator generates 1 molecule input + 1 atomic-reference input per "
            "unique element at consistent CAS theory (auto-summed dimensions so "
            "check_molcas_active_space_consistency passes). DKH is applied uniformly "
            "when any element is Z>=19, and &SCF is dropped on high-spin TM atoms "
            "(Cr ⁷S, Mn ⁶S, Fe ⁵D) where ROHF won't converge from GuessOrb. "
            "(2) Run all N inputs (small atoms first). (3) Call "
            "check_molcas_active_space_consistency to verify CAS sums + character. "
            "(4) Call compute_molcas_reaction_energy(products=atoms, reactants=[mol], "
            "energy_kind='rasscf' or 'caspt2') for the binding energy. Validated "
            "end-to-end on CrO: D_e(CASSCF) = 47 kcal/mol, D_e(CASPT2) = 95 kcal/mol "
            "(experimental 108). H₂O atomization: 193 / 180 D_e/D_0 at CASSCF(8,6)."
        ),
        "search": "atomization binding energy dissociation",
        "modules": ("rasscf", "caspt2"),
    },
    "caspt2_intruder_protection": {
        "summary": (
            "CASPT2 on TM / DKH systems frequently hits intruder states — small "
            "denominators on excitations into the diffuse virtual space blow up "
            "the amplitudes. Symptom: _RC_NOT_CONVERGED_ in CASPT2 + reference "
            "weight < 0.7. Fix: add `Imaginary 0.1` (or 0.05 for mild cases) to the "
            "&CASPT2 block. Validated on CrO CAS(10,9): no shift → ref_weight 0.0002, "
            "abort; with imag=0.1 → ref_weight 0.87, converged. The atomization + "
            "caspt2_chain orchestrators auto-set imaginary_shift=0.1 when DKH is on. "
            "Reference weight ≥ 0.70 is the trust threshold; < 0.85 is caution band."
        ),
        "search": "CASPT2 intruder imaginary shift reference weight denominator",
        "modules": ("caspt2",),
    },
    "ts_search_procedure": {
        "summary": (
            "Transition-state search via prepare_molcas_opt_freq_workflow with "
            "transition_state=True. Workflow: (1) Pick a starting geometry close to "
            "the TS (bond breaking ~midway, key angle distorted). (2) The "
            "orchestrator emits MCKINLEY+MCLR BEFORE the SLAPAF loop by default "
            "(compute_initial_hessian=True for TS) so SLAPAF identifies the imaginary "
            "mode to follow. (3) SLAPAF TS uses eigenvector-following. (4) After "
            "convergence, MCKINLEY+MCLR re-runs to verify exactly one imaginary "
            "frequency at the saddle. (5) Confirm the imaginary mode IS the "
            "reaction coordinate via the 'Angle/Bond X1 Y1 Z1' label in the freq "
            "table. Validated on HCN→HNC: i1202 cm⁻¹ = N-C-H bend. Watch for "
            "spurious small-magnitude imaginary modes (i<70 cm⁻¹) in the final-freq "
            "table — those are unprojected translation/rotation, harmless."
        ),
        "search": "SLAPAF TS transition state FindTS PRFC saddle imaginary",
        "modules": ("slapaf", "mckinley", "mclr"),
    },
    "irc_followup": {
        "summary": (
            "IRC verifies a TS connects the expected reactant + product. "
            "Workflow: (1) Run prepare_molcas_opt_freq_workflow(transition_state=True) "
            "to get the TS + its imaginary mode. (2) Call prepare_molcas_irc_workflow "
            "with ts_output_file pointing at the prior log — orchestrator parses "
            "'The Cartesian Reaction vector' section and emits REACtion vector in "
            "SLAPAF IRC. (3) Run the IRC input; it walks both directions until "
            "energy rises. (4) Inspect $Project.mep.molden + Opt.xyz for endpoint "
            "geometries. Critical gotcha: 'Nuclear coordinates for the next "
            "iteration' in the prior log is in BOHR — pass geometry_units='bohr' "
            "when forwarding those coords. Validated on HCN→HNC: TS @ -92.829 "
            "→ descended monotonically over 4 hypersphere points, H atom migrated "
            "from C side to N side."
        ),
        "search": "IRC intrinsic reaction coordinate MEP SLAPAF REACtion vector",
        "modules": ("slapaf",),
    },
    "character_aware_active_space": {
        "summary": (
            "RASSCF active spaces from GuessOrb on TM atoms often grab the wrong "
            "shells (4s + 3 of 4p instead of 3d×5 + 4s). Detection: "
            "suggest_molcas_orbital_swaps with target_atom_pattern='Cr' + "
            "target_ao_pattern='3d' compares each active orbital's dominant AO to "
            "the target; returns swap candidates ranked by character match score. "
            "Recovery: refine_molcas_active_space applies the swaps to RasOrb + "
            "writes a refined input with FILEORB injected. Validated on CrO atomic "
            "reference: wrong CAS at -1039.96 au → swapped → correct CAS at "
            "-1040.72 au (480 kcal/mol lower). The atomization orchestrator now "
            "auto-handles this for TM atoms by skipping SCF (lets GuessOrb feed "
            "directly into RASSCF)."
        ),
        "search": "active orbital swap RasOrb INPORB FILEORB 3d character",
        "modules": ("rasscf",),
    },
    "recovery_patterns": {
        "summary": (
            "When a Molcas run fails, suggest_molcas_recovery walks a "
            "priority-ordered rule engine (11 failure modes) and returns failure_class "
            "+ root_cause + fix_recipe + next_actions. apply_molcas_recovery then "
            "regex-edits the input to fix mechanical failures: drop &SCF for "
            "scf_no_convergence + scf_single_electron; bump RASSCF Iteration to "
            "(100,50) for rasscf_no_convergence; add Imaginary 0.1 for "
            "caspt2_intruder / caspt2_low_ref_weight; 2× MOLCAS_MEM for "
            "memory_exceeded; &SEWARD → &GATEWAY for missing_basis_in_loop; bump "
            "SLAPAF Iterations for slapaf_no_convergence. Manual-only fixes "
            "(routed via verdict=manual_intervention_required): jobiph_missing → "
            "prepare_molcas_excited_states, ga_segfault → rebuild GA with "
            "--with-mpi-ts OR force -np 1, nactel_parity → "
            "compute_molcas_active_space_partition, seward_angstrom_symmetry → "
            "regenerate via draft_molcas_input (auto Angstrom→bohr internally)."
        ),
        "search": "recovery failure SCF convergence RASSCF CASPT2 intruder",
        "modules": ("scf", "rasscf", "caspt2"),
    },
    "pes_scan": {
        "summary": (
            "Constrained-geometry PES scans via prepare_molcas_scan_workflow. "
            "Generates one optimization input per scan point, with the chosen "
            "bond/angle/dihedral locked via &GATEWAY Constraint + SLAPAF. "
            "chain_orbitals=True (default) uses FILEORB from the previous point's "
            "RasOrb/ScfOrb for warm-start. Watch out for bent↔linear transitions "
            "(e.g. r(C-H) on HCN past 1.4 Å) which trip SLAPAF's "
            "'BMtrx_internal: nq < nQQ' error because 3N-6 internals collapse to "
            "3N-5 at linearity. Use angle coordinates or perturb off-axis for "
            "those systems. Validated on H₂O O-H bond scan: 4 points from 0.85 "
            "to 1.15 Å, clean potential well with minimum at 0.95 Å (experimental "
            "0.957)."
        ),
        "search": "Constraint Value SLAPAF scan PES potential energy surface bond angle dihedral",
        "modules": ("slapaf", "gateway"),
    },
    "tm_atom_setup": {
        "summary": (
            "Transition-metal atomic references (Cr ⁷S, Mn ⁶S, Fe ⁵D, V ⁴F, "
            "Co ⁴F) need three special considerations: (1) ROHF doesn't converge "
            "from GuessOrb — drop the &SCF block and let RASSCF use GuessOrb "
            "directly. (2) ANO-RCC basis sets are contracted for relativistic "
            "eigenfunctions; absolute energies need DKH (`Relativistic R02O02` in "
            "SEWARD). Mixing DKH and non-DKH across reaction species is a bug. "
            "(3) GuessOrb's energy-based ordering puts 3p and 3d at similar "
            "depths; RASSCF can pick wrong shells in active. Always run "
            "suggest_molcas_orbital_swaps after RASSCF converges on TM atoms. "
            "The atomization orchestrator bakes all three rules in via skip_scf "
            "+ relativistic='auto' + the bundled ATOMIC_GROUND_STATES table."
        ),
        "search": "transition metal Cr 3d 4s ROHF Charge GuessOrb DKH Relativistic",
        "modules": ("rasscf", "seward"),
    },
}


def get_topic_guide(topic: str) -> dict[str, Any]:
    normalized = _normalize_topic(topic)
    info = _TOPICS.get(normalized)
    if not info:
        raise ValueError(
            "unsupported topic; use one of: " + ", ".join(sorted(_TOPICS))
        )
    search_results = search_docs(info["search"], max_results=6, context_lines=3)["results"]
    module_excerpts: list[dict[str, Any]] = []
    for module in info.get("modules", ()):
        try:
            module_doc = lookup_module_syntax(module)
            if module_doc.get("found"):
                module_excerpts.extend(module_doc["results"])
        except Exception:
            pass
    return {
        "topic": normalized,
        "summary": info["summary"],
        "results": search_results,
        "module_pages": module_excerpts,
    }


def _excerpt_payload(path: Path, lines: list[str], start: int, end: int, matched_line: int | None) -> dict[str, Any]:
    excerpt = [
        {
            "line_number": idx,
            "text": lines[idx - 1],
            "matched": idx == matched_line,
        }
        for idx in range(start, end + 1)
    ]
    return {
        "file_name": _rel(path),
        "start_line": start,
        "end_line": end,
        "matched_line": matched_line,
        "excerpt": excerpt,
    }


def _normalize_topic(topic: str) -> str:
    value = topic.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "rasscf": "rasscf_active_space",
        "active_space": "rasscf_active_space",
        "caspt2": "caspt2_setup",
        "ipea": "ipea_shift",
        "xms": "xms_caspt2",
        "ms_caspt2": "xms_caspt2",
        "alaska": "alaska_gradients",
        "frequency": "mclr_freq",
        "freq": "mclr_freq",
        "rassi": "rassi_state_interaction",
        "soc": "rassi_state_interaction",
        "spin_orbit": "rassi_state_interaction",
        "inporb": "inporb_format",
        "rasorb": "inporb_format",
        "scforb": "inporb_format",
        "scf": "scf_setup",
    }
    return aliases.get(value, value)


def _format_search_match(match: _DocMatch, *, context_lines: int) -> dict[str, Any]:
    excerpt = read_doc_excerpt(
        str(match.file_path),
        start_line=max(1, match.line_number - context_lines),
        end_line=match.line_number + context_lines,
    )
    return {
        "file_name": _rel(match.file_path),
        "line_number": match.line_number,
        "line_text": match.line_text,
        "score": match.score,
        "excerpt": excerpt["excerpt"],
    }


def _tokenize(text: str) -> list[str]:
    tokens = [token for token in re.split(r"[^a-zA-Z0-9_+-]+", text.casefold()) if token]
    return [token for token in tokens if len(token) >= 2]


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    # Markdown headings
    if stripped.startswith("#"):
        return True
    if len(stripped) < 120 and stripped == stripped.title():
        return True
    return stripped.endswith(":") or stripped.startswith(
        ("Example", "Examples", "SCF", "RASSCF", "CASPT2", "RASSI", "ALASKA", "MCLR")
    )
