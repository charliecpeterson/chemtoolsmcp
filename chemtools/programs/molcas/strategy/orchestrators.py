"""Thick orchestrators that chain multiple Molcas strategy + drafter + binary
calls into a single decision-making tool, returning a Diagnosis envelope ready
for an agent to act on.

Mirror NWChem's prepare_*_setup pattern: push deterministic reasoning into
Python so small LLMs can chain calls without re-deriving the logic.

Currently provided:

  refine_active_space(...) — closes the active-space-tuning loop in one call:
      parse existing RASSCF .out → analyze occupation-based verdict →
      suggest character-aware swaps → apply swaps to RasOrb → write a refined
      input file with FILEORB → prepare safe launch plan → return next_actions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text

from chemtools.programs.molcas.parse.output import parse_output_full
from chemtools.programs.molcas.parse.mos import parse_last_mo_block
from chemtools.programs.molcas.binary.orbitals import swap_orbitals_in_inporb
from chemtools.programs.molcas.runtime import prepare_launch
from chemtools.programs.molcas.strategy.active_space import (
    analyze_active_space,
    suggest_orbital_swaps_by_character,
)


def refine_active_space(
    output_file: str,
    *,
    target_atom_pattern: str,
    target_ao_pattern: str,
    rasorb_file: str | None = None,
    input_file: str | None = None,
    output_dir: str | None = None,
    refined_job_name: str | None = None,
    apply_swaps: bool = True,
    symmetry: int = 1,
    max_swaps: int = 5,
    apptainer_sif: str | None = None,
    profile: dict | None = None,
    requested_np: int = 1,
) -> dict[str, Any]:
    """Close the active-space tuning loop in one tool call.

    Reads an existing RASSCF run, analyzes its active space (occupation-based
    verdict), proposes character-aware swaps to bring target-character orbitals
    into the active window, optionally writes a swapped RasOrb + a refined
    input file that uses ``FILEORB``, and prepares a safe launch plan for the
    refined run.

    Parameters
    ----------
    output_file
        Path to the existing Molcas .out file (must have a RASSCF task).
    target_atom_pattern, target_ao_pattern
        Patterns passed to suggest_orbital_swaps_by_character, e.g. ("Cr", "3d").
    rasorb_file
        Path to the source RasOrb file. Default: derived from output_file stem
        (e.g. ``cro_v1.RasOrb`` next to ``cro_v1.out``).
    input_file
        Path to the source .input file. Default: derived from output_file stem.
    output_dir
        Directory for the refined RasOrb + .input files. Default: same as
        output_file directory.
    refined_job_name
        Base name for refined files. Default: ``<source_stem>_refined``.
    apply_swaps
        If True, write the swapped RasOrb + refined input. If False, return the
        suggestions only (dry-run).
    symmetry
        1-indexed irrep. Default 1 (correct for C1).
    max_swaps
        Cap on the number of swaps to apply. Default 5 (mostly defensive — the
        suggester rarely returns more).
    apptainer_sif, profile, requested_np
        Forwarded to prepare_launch for the launch plan.

    Returns a structured dict; see implementation for full shape. Key fields:
      verdict                "active_space_ok" / "needs_refinement" / "no_target_matches"
      current_state          parsed energy + active-space summary
      swap_suggestions       list of {active_orbital, swap_with, rationale}
      refined_input          path to the refined .input (if apply_swaps)
      refined_orbital_file   path to the swapped RasOrb (if apply_swaps)
      launch_plan            command + env (if apply_swaps and inputs found)
      next_actions           agent-actionable list
    """
    output_path = Path(output_file)
    if not output_path.is_file():
        raise FileNotFoundError(f"output file not found: {output_file}")

    contents = read_text(output_file)
    full = parse_output_full(output_file, contents)
    rasscf_task = next(
        (p for p in full["task_payloads"] if p["module"] == "RASSCF"),
        None,
    )
    if rasscf_task is None:
        return {
            "verdict": "no_rasscf_task",
            "error": "no_rasscf_task",
            "message": f"No RASSCF task in {output_file}",
            "next_actions": [
                {
                    "tool": "parse_molcas_output",
                    "args": {"output_file": output_file},
                    "rationale": "Verify the output file actually contains a RASSCF run.",
                }
            ],
        }

    rasscf_payload = rasscf_task["details"]
    current_state = {
        "active_space_signature": rasscf_payload.get("active_space_signature"),
        "rasscf_root_energies": rasscf_payload.get("root_energies"),
        "wave_function": rasscf_payload.get("wave_function"),
        "orbital_specs": rasscf_payload.get("orbital_specs"),
        "scf_total_hartree": full["energy_summary"].get("scf_total_hartree"),
        "primary_energy_hartree": full["energy_summary"].get("primary_energy_hartree"),
        "primary_label": full["energy_summary"].get("primary_label"),
    }

    # Step 1: occupation-based analysis (existing analyze_active_space)
    occupation_analysis = analyze_active_space(rasscf_payload)

    # Step 2: character-aware suggester — needs the LAST MO block from the
    # RASSCF task body (has dominant_aos per orbital)
    line_start, line_end = rasscf_task["line_range"]
    lines = contents.splitlines()
    block_text = "\n".join(lines[line_start - 1: line_end])
    mo_block = parse_last_mo_block(block_text, parse_coefficients=True)
    if mo_block is None:
        return {
            "verdict": "no_mo_block",
            "error": "no_mo_block",
            "message": "RASSCF task has no '++ Molecular orbitals:' block — cannot do character-aware analysis.",
            "current_state": current_state,
            "occupation_analysis": occupation_analysis,
            "next_actions": [
                {
                    "tool": "analyze_molcas_active_space",
                    "args": {"output_file": output_file},
                    "rationale": "Fall back to occupation-only verdict.",
                }
            ],
        }

    suggester = suggest_orbital_swaps_by_character(
        mo_block=mo_block,
        rasscf_orbital_specs=rasscf_payload.get("orbital_specs", {}),
        target_atom_pattern=target_atom_pattern,
        target_ao_pattern=target_ao_pattern,
        symmetry=symmetry,
    )
    suggested_swaps = suggester["suggested_swaps"][:max_swaps]

    # Choose verdict from analyses
    verdict = _choose_verdict(occupation_analysis, suggester, suggested_swaps)

    result: dict[str, Any] = {
        "verdict": verdict,
        "current_state": current_state,
        "occupation_analysis": {
            "verdict": occupation_analysis.get("verdict"),
            "per_root_quality": occupation_analysis.get("per_root_quality"),
        },
        "character_analysis": {
            "target_pattern": suggester["target_pattern"],
            "active_misses": suggester["active_misses"],
            "outside_matches": suggester["outside_matches"],
            "rationale": suggester["rationale"],
        },
        "swap_suggestions": suggested_swaps,
        "next_actions": [],
    }

    if not suggested_swaps:
        result["next_actions"] = [
            {
                "tool": "validate_molcas_caspt2_setup",
                "rationale": (
                    "No character-aware swaps suggested — active space already contains "
                    f"all the {target_atom_pattern} {target_ao_pattern} orbitals it can. "
                    "Proceed to CASPT2 validation."
                ),
            }
        ]
        return result

    if not apply_swaps:
        result["next_actions"] = [
            {
                "tool": "refine_molcas_active_space",
                "args": {"output_file": output_file, "apply_swaps": True},
                "rationale": (
                    f"Dry-run found {len(suggested_swaps)} candidate swap(s). "
                    "Re-call with apply_swaps=True to write the refined orbital file + input."
                ),
            }
        ]
        return result

    # --- Apply swaps and generate refined files ---
    src_rasorb = _resolve_rasorb(output_path, rasorb_file)
    src_input = _resolve_input(output_path, input_file)
    out_dir = Path(output_dir) if output_dir else output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    job_name = refined_job_name or f"{output_path.stem}_refined"

    if src_rasorb is None:
        result["warnings"] = [
            f"Could not find source RasOrb file (looked next to {output_file}); pass rasorb_file= explicitly."
        ]
        return result

    refined_rasorb = out_dir / f"{job_name}.RasOrb"
    swap_pairs = [(s["active_orbital"], s["swap_with"]) for s in suggested_swaps]
    swap_summary = swap_orbitals_in_inporb(
        input_path=str(src_rasorb),
        output_path=str(refined_rasorb),
        swaps=swap_pairs,
        symmetry=symmetry,
    )
    result["refined_orbital_file"] = str(refined_rasorb)
    result["swap_summary"] = swap_summary

    if src_input is None:
        result["warnings"] = result.get("warnings", []) + [
            f"Source .input file not found next to {output_file}; cannot generate refined input. "
            "Wrote refined RasOrb only."
        ]
        result["next_actions"] = [
            {
                "tool": "draft_molcas_input",
                "rationale": (
                    "Hand-craft an input that reads the swapped RasOrb via FILEORB, OR pass "
                    "input_file= to this orchestrator with the source .input path."
                ),
            }
        ]
        return result

    refined_input_path = out_dir / f"{job_name}.input"
    refined_text = _add_fileorb_to_input(
        src_input.read_text(encoding="utf-8"),
        str(refined_rasorb),
    )
    refined_input_path.write_text(refined_text, encoding="utf-8")
    result["refined_input"] = str(refined_input_path)

    # Build launch plan
    plan = prepare_launch(
        str(refined_input_path),
        profile=profile,
        requested_np=requested_np,
        job_name=job_name,
        apptainer_sif=apptainer_sif,
    )
    result["launch_plan"] = plan
    result["next_actions"] = [
        {
            "tool": "shell_execute",  # informational only — agent or user runs it
            "args": {"command": plan["command_str"], "env": plan["env"]},
            "rationale": (
                f"Run the refined CASSCF with the swapped orbital file. "
                f"Compare RASSCF energy against the baseline {current_state.get('primary_energy_hartree')}. "
                "Lower energy → swap helped; higher → swap moved the calculation to a different "
                "(possibly worse) local minimum."
            ),
        },
        {
            "tool": "parse_molcas_output",
            "args": {"output_file": str(refined_input_path).replace(".input", ".out")},
            "rationale": "After the refined run completes, parse to compare energies and active-space character.",
        },
    ]
    return result


# --- helpers ------------------------------------------------------------------


def _choose_verdict(
    occupation_analysis: dict, suggester: dict, suggested_swaps: list[dict]
) -> str:
    """Combine occupation + character verdicts into a single recommendation."""
    occ_verdict = occupation_analysis.get("verdict", "unknown")
    has_targets = bool(suggester["outside_matches"])
    has_misses = bool(suggester["active_misses"])
    if not has_targets and not has_misses:
        return "no_target_matches"
    if not has_misses:
        return "active_space_ok"
    if has_misses and not suggested_swaps:
        return "needs_chemist_input"  # misses exist but no suitable swap candidates
    if occ_verdict == "poor":
        return "needs_refinement"
    return "needs_refinement" if suggested_swaps else "active_space_ok"


def _resolve_rasorb(output_path: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    # Try common patterns
    stem = output_path.stem
    parent = output_path.parent
    for cand in (parent / f"{stem}.RasOrb", parent.parent / f"{stem}.RasOrb"):
        if cand.is_file():
            return cand
    return None


def _resolve_input(output_path: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    stem = output_path.stem
    parent = output_path.parent
    for cand in (parent / f"{stem}.input", parent.parent / f"{stem}.input"):
        if cand.is_file():
            return cand
    return None


_LUMORB_RE = re.compile(r"^([ \t]*)LumOrb[ \t]*$", re.M | re.I)
_FILEORB_BLOCK_RE = re.compile(
    r"^[ \t]*FILEORB[ \t]*\n[ \t]*\S+[ \t]*\n", re.M | re.I
)


def _add_fileorb_to_input(input_text: str, rasorb_path: str) -> str:
    """Insert a FILEORB block pointing at rasorb_path into the RASSCF section.

    If FILEORB already exists, replaces its path. Otherwise inserts immediately
    after the first ``LumOrb`` keyword found (typically in the RASSCF block).
    """
    fileorb_block = f"FILEORB\n {rasorb_path}\n"
    if _FILEORB_BLOCK_RE.search(input_text):
        return _FILEORB_BLOCK_RE.sub(fileorb_block, input_text, count=1)
    if _LUMORB_RE.search(input_text):
        return _LUMORB_RE.sub(lambda m: f"{m.group(0)}\n{fileorb_block.rstrip()}", input_text, count=1)
    # No LumOrb? Insert before End of input of the RASSCF block (fallback)
    rasscf_start = input_text.find("&RASSCF")
    if rasscf_start == -1:
        raise ValueError("input has no &RASSCF block — cannot insert FILEORB")
    end_of_input = input_text.find("End of input", rasscf_start)
    if end_of_input == -1:
        raise ValueError("&RASSCF block missing 'End of input' — input format is malformed")
    return input_text[:end_of_input] + fileorb_block + input_text[end_of_input:]


# --- Greenfield CASSCF / CASPT2 setup orchestrator ----------------------------


# Valence d-electron counts per transition-metal element (oxidation state 0).
# For ions, the agent should pass cas_active_electrons explicitly.
_TM_D_ELECTRONS_NEUTRAL: dict[str, int] = {
    # First row (3d series)
    "Sc": 1,  "Ti": 2,  "V": 3,   "Cr": 5,  "Mn": 5,
    "Fe": 6,  "Co": 7,  "Ni": 8,  "Cu": 10, "Zn": 10,
    # Second row (4d series) — neutral configurations
    "Y": 1,   "Zr": 2,  "Nb": 4,  "Mo": 5,  "Tc": 5,
    "Ru": 7,  "Rh": 8,  "Pd": 10, "Ag": 10, "Cd": 10,
    # Third row (5d series)
    "La": 1,  "Hf": 2,  "Ta": 3,  "W": 4,   "Re": 5,
    "Os": 6,  "Ir": 7,  "Pt": 9,  "Au": 10, "Hg": 10,
}


def suggest_cas_from_hint(
    *,
    chemistry_hint: str,
    atoms: list[dict],
    charge: int,
    multiplicity: int = 1,
    total_electrons: int | None = None,
) -> dict[str, Any]:
    """Derive (cas_active_electrons, cas_active_orbitals) from a chemistry hint.

    Hints supported:
      "valence_d"     all transition-metal 3d/4d/5d orbitals (5 per metal atom);
                      electrons from neutral configuration minus ionic charge
                      attributed to the metals (simplest possible accounting).
      "frontier_pair" 2 electrons / 2 orbitals — HOMO/LUMO of a closed-shell.
                      Only valid for closed-shell molecules.

    Returns {cas_active_electrons, cas_active_orbitals, rationale, warnings}.
    """
    hint = chemistry_hint.lower().strip()
    if hint == "valence_d":
        tm_atoms = [a for a in atoms if a.get("symbol", "").capitalize() in _TM_D_ELECTRONS_NEUTRAL]
        if not tm_atoms:
            return {
                "error": "no_transition_metal",
                "message": "chemistry_hint='valence_d' but no transition-metal atom in the molecule.",
            }
        n_metals = len(tm_atoms)
        # 5 d orbitals per metal (valence d shell)
        cas_active_orbitals = 5 * n_metals
        # Electron count: sum neutral-state d electrons, distribute the charge
        # equally across metals (very approximate). The agent should override
        # this for complex bonding situations.
        total_d_electrons = sum(
            _TM_D_ELECTRONS_NEUTRAL[a["symbol"].capitalize()] for a in tm_atoms
        )
        cas_active_electrons = max(0, total_d_electrons - charge)
        warnings = []
        n_unpaired = multiplicity - 1
        # Parity check: cas_active_electrons MUST have the same parity as n_unpaired
        # (so that cas_active_electrons - n_unpaired is even = doubly-occupied pairs).
        # If the d-only count has wrong parity, the metal almost certainly
        # contributes an s-shell electron to bonding/SOMO — bump by 1.
        if (cas_active_electrons - n_unpaired) % 2 != 0:
            cas_active_electrons += 1
            warnings.append(
                f"valence_d adjusted to {cas_active_electrons}e (was {cas_active_electrons-1}) "
                f"to match parity required by multiplicity={multiplicity}. This typically "
                "reflects the metal's s-shell electron participating in bonding or being a SOMO."
            )
        if total_electrons is not None and (total_electrons - cas_active_electrons) % 2 != 0:
            warnings.append(
                f"Even after parity correction, inactive electron count "
                f"{total_electrons - cas_active_electrons} is odd. Override "
                "cas_active_electrons explicitly."
            )
        if cas_active_electrons > 2 * cas_active_orbitals:
            warnings.append(
                f"valence_d hint gave {cas_active_electrons}e in {cas_active_orbitals} "
                "orbitals — would overfill. Cap or expand the active space."
            )
        if n_metals > 1:
            warnings.append(
                "Multi-metal valence_d hint applies a simple charge-on-each-metal "
                "rule — verify the oxidation-state assignment matches your chemistry."
            )
        return {
            "cas_active_electrons": cas_active_electrons,
            "cas_active_orbitals": cas_active_orbitals,
            "rationale": (
                f"valence_d: {n_metals} TM atom(s); 5 d-orbitals each → "
                f"{cas_active_orbitals} orbitals; neutral-state d electron count "
                f"({total_d_electrons}) minus molecular charge ({charge}) → "
                f"{cas_active_electrons} active electrons."
            ),
            "warnings": warnings,
        }
    if hint == "frontier_pair":
        return {
            "cas_active_electrons": 2,
            "cas_active_orbitals": 2,
            "rationale": "frontier_pair: HOMO/LUMO 2e/2o active space (closed-shell).",
            "warnings": [],
        }
    return {
        "error": "unknown_hint",
        "message": f"Unknown chemistry_hint {chemistry_hint!r}; supported: 'valence_d', 'frontier_pair'.",
    }


def prepare_casscf_setup(
    *,
    atoms: list[dict],
    charge: int,
    multiplicity: int,
    basis: str | dict[str, str],
    title: str | None = None,
    method: str = "CASSCF",
    geometry_units: str = "angstrom",
    chemistry_hint: str | None = None,
    cas_active_electrons: int | None = None,
    cas_active_orbitals: int | None = None,
    program_options: dict[str, Any] | None = None,
    job_name: str | None = None,
    write_input_to: str | None = None,
    apptainer_sif: str | None = None,
    profile: dict | None = None,
    requested_np: int = 1,
) -> dict[str, Any]:
    """Greenfield orchestrator: draft + lint + plan + diagnose a fresh CASSCF
    or CASPT2 calculation.

    Either pass (cas_active_electrons, cas_active_orbitals) explicitly OR
    pass chemistry_hint to derive them.

    Returns a dict with:
        verdict           "ready_to_launch" / "lint_blocked" / "missing_cas"
        active_space      derived (M, N) + rationale
        input_text        drafted input
        input_path        where the input was written (if write_input_to given
                          or job_name set with implicit path)
        lint_issues       list of LintIssue records
        launch_plan       command + env (if input was written)
        next_actions      agent-actionable list
    """
    # Imported here to avoid circular imports — draft.py uses analyze_active_space
    from chemtools.programs.molcas.input.draft import draft_molcas_input
    from chemtools.programs.molcas.input.lint import lint_molcas_input

    program_options = dict(program_options or {})
    warnings: list[str] = []

    # Step 1: resolve the active space
    cas_source = "explicit"
    cas_rationale: str | None = None
    if cas_active_electrons is None or cas_active_orbitals is None:
        if chemistry_hint is None:
            return {
                "verdict": "missing_cas",
                "error": "missing_cas_spec",
                "message": (
                    "Provide either (cas_active_electrons, cas_active_orbitals) "
                    "explicitly OR chemistry_hint='valence_d'/'frontier_pair'."
                ),
                "next_actions": [
                    {
                        "tool": "prepare_molcas_casscf_setup",
                        "rationale": "Re-call with an explicit CAS or a chemistry_hint.",
                    }
                ],
            }
        # Compute total electrons for the parity check
        from chemtools.programs.molcas.input._utils import total_electrons as _total_e
        try:
            n_elec_total = _total_e(atoms, charge)
        except Exception:
            n_elec_total = None
        suggestion = suggest_cas_from_hint(
            chemistry_hint=chemistry_hint,
            atoms=atoms,
            charge=charge,
            multiplicity=multiplicity,
            total_electrons=n_elec_total,
        )
        if "error" in suggestion:
            return {
                "verdict": "missing_cas",
                "error": suggestion["error"],
                "message": suggestion["message"],
            }
        cas_active_electrons = suggestion["cas_active_electrons"]
        cas_active_orbitals = suggestion["cas_active_orbitals"]
        cas_rationale = suggestion["rationale"]
        warnings.extend(suggestion.get("warnings", []))
        cas_source = f"hint:{chemistry_hint}"

    # Step 2: build the InputSpec and draft
    program_options.setdefault("cas_active_electrons", cas_active_electrons)
    program_options.setdefault("cas_active_orbitals", cas_active_orbitals)
    program_options.setdefault("memory_mb", 2000)
    program_options.setdefault("inline_basis", True)

    spec = {
        "atoms": atoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "method": method,
        "basis": basis,
        "task": "energy",
        "title": title or f"{method} CAS({cas_active_electrons},{cas_active_orbitals})",
        "geometry_units": geometry_units,
        "program_options": program_options,
    }
    input_text = draft_molcas_input(spec)
    lint_issues = lint_molcas_input(input_text)
    n_errors = sum(1 for i in lint_issues if i.get("level") == "error")

    result: dict[str, Any] = {
        "verdict": "ready_to_launch" if n_errors == 0 else "lint_blocked",
        "active_space": {
            "cas_active_electrons": cas_active_electrons,
            "cas_active_orbitals": cas_active_orbitals,
            "source": cas_source,
            "rationale": cas_rationale,
        },
        "method": method,
        "input_text": input_text,
        "lint_issues": lint_issues,
        "n_lint_errors": n_errors,
        "n_lint_warnings": sum(1 for i in lint_issues if i.get("level") == "warning"),
        "warnings": warnings,
    }

    # Step 3: optionally write the input + build launch plan
    target_path: Path | None = None
    if write_input_to:
        target_path = Path(write_input_to)
    elif job_name:
        target_path = Path.cwd() / f"{job_name}.input"
    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(input_text, encoding="utf-8")
        result["input_path"] = str(target_path)
        if n_errors == 0:
            plan = prepare_launch(
                str(target_path),
                profile=profile,
                requested_np=requested_np,
                job_name=job_name,
                apptainer_sif=apptainer_sif,
            )
            result["launch_plan"] = plan
            result["next_actions"] = [
                {
                    "tool": "shell_execute",
                    "args": {"command": plan["command_str"], "env": plan["env"]},
                    "rationale": (
                        f"Run the {method} calculation. Active space: CAS("
                        f"{cas_active_electrons},{cas_active_orbitals})."
                    ),
                },
                {
                    "tool": "parse_molcas_output",
                    "args": {"output_file": str(target_path).replace(".input", ".out")},
                    "rationale": "After it finishes, parse to confirm convergence and active-space quality.",
                },
                {
                    "tool": "analyze_molcas_active_space",
                    "args": {"output_file": str(target_path).replace(".input", ".out")},
                    "rationale": "Check NO occupations are in the healthy active window.",
                },
            ]
        else:
            result["next_actions"] = [
                {
                    "tool": "lint_molcas_input",
                    "args": {"input_text": input_text},
                    "rationale": f"Fix the {n_errors} lint error(s) before launching.",
                }
            ]
    else:
        # No write requested — just return the drafted input for inspection
        result["next_actions"] = [
            {
                "tool": "prepare_molcas_casscf_setup",
                "args": {"write_input_to": "<path>"},
                "rationale": "Re-call with write_input_to=<path> to materialize the input file + launch plan.",
            }
        ]

    return result


# --- CASPT2 chain orchestrator ------------------------------------------------


def prepare_caspt2_chain(
    rasscf_output_file: str,
    *,
    rasorb_file: str | None = None,
    input_file: str | None = None,
    output_dir: str | None = None,
    job_name: str | None = None,
    variant: str | None = None,
    ipea_shift: float | None = None,
    real_shift: float | None = None,
    imaginary_shift: float | None = None,
    sigma_p_regularization: float | None = None,
    target_root: int | None = None,
    properties: bool = False,
    grdt: bool = False,
    apptainer_sif: str | None = None,
    profile: dict | None = None,
    requested_np: int = 1,
) -> dict[str, Any]:
    """Continuation orchestrator: take a converged RASSCF output and chain
    CASPT2 on top with intelligent default settings.

    Decisions made automatically (overridable via kwargs):
      * SS-CASPT2 vs MS-CASPT2 based on RASSCF n_roots
      * IPEA shift defaults to 0.25 (Molcas default since v6.4)
      * If RASSCF active-space verdict is "poor" or "marginal", emits an
        imaginary shift of 0.1 as insurance against weak intruders
      * Frozen vector mirrors the RASSCF frozen (defaults to none in Molcas
        CASPT2 — auto-freezing of deep cores)

    Short-circuits to ``verdict="needs_active_space_refinement"`` if the RASSCF
    active space is "poor", pointing the agent at ``refine_molcas_active_space``
    first.
    """
    from chemtools.programs.molcas.input.draft import draft_molcas_input
    from chemtools.programs.molcas.input.lint import lint_molcas_input

    output_path = Path(rasscf_output_file)
    if not output_path.is_file():
        raise FileNotFoundError(rasscf_output_file)

    contents = read_text(rasscf_output_file)
    full = parse_output_full(rasscf_output_file, contents)
    rasscf_task = next(
        (p for p in full["task_payloads"] if p["module"] == "RASSCF"),
        None,
    )
    if rasscf_task is None:
        return {
            "verdict": "no_rasscf_task",
            "error": "no_rasscf_task",
            "message": f"No RASSCF task in {rasscf_output_file}",
        }
    rasscf_payload = rasscf_task["details"]

    # Step 1: check whether RASSCF converged
    if not rasscf_payload.get("converged", True):
        return {
            "verdict": "rasscf_unconverged",
            "error": "rasscf_unconverged",
            "message": "The RASSCF task did not reach convergence. Fix RASSCF before running CASPT2.",
            "next_actions": [
                {
                    "tool": "parse_molcas_output",
                    "args": {"output_file": rasscf_output_file},
                    "rationale": "Re-inspect RASSCF iterations and convergence diagnostics.",
                }
            ],
        }

    # Step 2: analyze active space quality
    analysis = analyze_active_space(rasscf_payload)
    if analysis["verdict"] == "poor":
        return {
            "verdict": "needs_active_space_refinement",
            "active_space_analysis": analysis,
            "message": (
                "RASSCF active space verdict is 'poor' — running CASPT2 on top "
                "of an unhealthy reference wastes cycles. Refine the active "
                "space first."
            ),
            "next_actions": [
                {
                    "tool": "refine_molcas_active_space",
                    "args": {"output_file": rasscf_output_file},
                    "rationale": "Suggest character-aware swaps before continuing to CASPT2.",
                }
            ],
        }

    # Step 3: derive CASPT2 settings from RASSCF state
    n_roots = (rasscf_payload.get("ci_expansion") or {}).get("n_roots") or 1
    derived_variant = variant or ("MS" if n_roots > 1 else "SS")
    derived_ipea = ipea_shift if ipea_shift is not None else 0.25
    # Heuristic for shift: "marginal" → suggest imaginary 0.1 unless user overrode
    if imaginary_shift is None and real_shift is None and sigma_p_regularization is None:
        if analysis["verdict"] == "marginal":
            derived_imag = 0.1
            shift_rationale = (
                f"Active space verdict is 'marginal' — emitting imaginary shift 0.1 "
                "as insurance against weak intruders."
            )
        else:
            derived_imag = 0.0
            shift_rationale = "Active space verdict 'healthy' — no shift applied."
    else:
        derived_imag = imaginary_shift if imaginary_shift is not None else 0.0
        shift_rationale = "Shift overridden by caller."

    # Step 4: build a continuation input — re-runs SEWARD + SCF + RASSCF (with
    # FILEORB → previous RASSCF orbitals) then CASPT2 with the derived settings.
    src_rasorb = _resolve_rasorb(output_path, rasorb_file)
    src_input = _resolve_input(output_path, input_file)
    if src_input is None:
        return {
            "verdict": "missing_source_input",
            "error": "missing_source_input",
            "message": (
                f"Could not find source .input file next to {rasscf_output_file}. "
                "Pass input_file= explicitly."
            ),
        }
    if src_rasorb is None:
        return {
            "verdict": "missing_source_rasorb",
            "error": "missing_source_rasorb",
            "message": (
                f"Could not find source .RasOrb file next to {rasscf_output_file}. "
                "Pass rasorb_file= explicitly."
            ),
        }
    out_dir = Path(output_dir) if output_dir else output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    new_job = job_name or f"{output_path.stem}_caspt2"

    # Modify the source input to:
    #  - Promote method to CASPT2 (add a CASPT2 block if missing)
    #  - Inject FILEORB → source RasOrb in the RASSCF block
    refined_input_path = out_dir / f"{new_job}.input"
    src_text = src_input.read_text(encoding="utf-8")
    refined_text = _add_fileorb_to_input(src_text, str(src_rasorb))
    refined_text = _ensure_caspt2_block(
        refined_text,
        variant=derived_variant,
        n_roots=n_roots,
        ipea_shift=derived_ipea,
        imaginary_shift=derived_imag,
        real_shift=real_shift or 0.0,
        sigma_p_regularization=sigma_p_regularization,
        target_root=target_root,
        properties=properties,
        grdt=grdt,
        frozen_per_symmetry=(rasscf_payload.get("orbital_specs", {}) or {}).get("frozen"),
        title=f"{Path(rasscf_output_file).stem} → {derived_variant}-CASPT2",
    )
    refined_input_path.write_text(refined_text, encoding="utf-8")

    # Step 5: lint
    lint_issues = lint_molcas_input(refined_text)
    n_errors = sum(1 for i in lint_issues if i.get("level") == "error")

    # Step 6: build launch plan
    plan = prepare_launch(
        str(refined_input_path),
        profile=profile,
        requested_np=requested_np,
        job_name=new_job,
        apptainer_sif=apptainer_sif,
    )

    return {
        "verdict": "ready_to_launch" if n_errors == 0 else "lint_blocked",
        "source_rasscf_energy": (rasscf_payload.get("root_energies") or [{}])[0].get("energy_hartree"),
        "active_space_analysis": {
            "verdict": analysis["verdict"],
            "signature": analysis.get("signature"),
        },
        "caspt2_settings": {
            "variant": derived_variant,
            "n_roots": n_roots,
            "ipea_shift": derived_ipea,
            "imaginary_shift": derived_imag,
            "real_shift": real_shift or 0.0,
            "sigma_p_regularization": sigma_p_regularization,
            "target_root": target_root,
            "shift_rationale": shift_rationale,
        },
        "refined_input": str(refined_input_path),
        "lint_issues": lint_issues,
        "n_lint_errors": n_errors,
        "launch_plan": plan,
        "next_actions": [
            {
                "tool": "shell_execute",
                "args": {"command": plan["command_str"], "env": plan["env"]},
                "rationale": (
                    f"Run the {derived_variant}-CASPT2 chain. The new RASSCF reads converged "
                    "orbitals from the previous run via FILEORB → fast warm-start."
                ),
            },
            {
                "tool": "validate_molcas_caspt2_setup",
                "args": {"output_file": str(refined_input_path).replace(".input", ".out")},
                "rationale": "After completion, check reference weight + intruder report.",
            },
        ],
    }


# Insertion helpers ----------------------------------------------------------

_CASPT2_BLOCK_RE = re.compile(r"&CASPT2\b.*?End of input", re.S | re.I)


def _ensure_caspt2_block(
    input_text: str,
    *,
    variant: str,
    n_roots: int,
    ipea_shift: float | None,
    imaginary_shift: float,
    real_shift: float,
    sigma_p_regularization: float | None,
    target_root: int | None,
    properties: bool,
    grdt: bool,
    frozen_per_symmetry: list[int] | None,
    title: str,
) -> str:
    """Replace (or insert) a CASPT2 block at the END of the input."""
    from chemtools.programs.molcas.input.caspt2 import render_caspt2_block

    caspt2_text = render_caspt2_block(
        title=title,
        variant=variant,  # type: ignore[arg-type]
        n_roots=n_roots,
        target_root=target_root,
        frozen_per_symmetry=frozen_per_symmetry if frozen_per_symmetry and any(frozen_per_symmetry) else None,
        ipea_shift=ipea_shift,
        real_shift=real_shift,
        imaginary_shift=imaginary_shift,
        sigma_p_regularization=sigma_p_regularization,
        properties=properties,
        grdt=grdt,
    )
    if _CASPT2_BLOCK_RE.search(input_text):
        return _CASPT2_BLOCK_RE.sub(caspt2_text.rstrip(), input_text, count=1)
    # Append at the end
    return input_text.rstrip() + "\n\n" + caspt2_text


# --- Excited-states workflow orchestrator -------------------------------------


def prepare_excited_states_workflow(
    *,
    atoms: list[dict],
    charge: int,
    basis: str | dict[str, str],
    cas_active_electrons: int,
    cas_active_orbitals: int,
    n_singlets: int = 0,
    n_triplets: int = 0,
    method: str = "MS-CASPT2",
    compute_soc: bool = False,
    properties: list[str] | None = None,
    title: str | None = None,
    geometry_units: str = "angstrom",
    symmetry: str | None = None,
    n_symmetries: int = 1,
    occupied_per_symmetry: list[int] | None = None,
    n_basis_per_symmetry: list[int] | None = None,
    rasscf_inactive_per_symmetry: list[int] | None = None,
    rasscf_active_per_symmetry: list[int] | None = None,
    ipea_shift: float = 0.25,
    imaginary_shift: float = 0.1,
    inline_basis: bool = True,
    memory_mb: int = 4000,
    job_name: str | None = None,
    write_input_to: str | None = None,
    apptainer_sif: str | None = None,
    profile: dict | None = None,
    requested_np: int = 1,
) -> dict[str, Any]:
    """Multi-state excited-states orchestrator.

    Generates a full input that chains:
      1. SEWARD (+ optional symmetry generators)
      2. SCF (closed-shell singlet — required as common starting orbitals)
      3. RASSCF over n_singlets states (Spin=1) if n_singlets > 0
         + EMIL `>>COPY $Project.JobIph JOB001` so RASSI can find it
      4. RASSCF over n_triplets states (Spin=3) if n_triplets > 0
         + EMIL `>>COPY $Project.JobIph JOB002`
      5. CASPT2 for the singlet group (method='CASPT2' or 'MS-CASPT2' etc.)
      6. CASPT2 for the triplet group (same)
      7. RASSI combining both groups (with SPINorbit if compute_soc=True)

    The simplest valid configuration is (n_singlets=N, n_triplets=0,
    method='MS-CASPT2', compute_soc=False) — that's the canonical 'N excited
    singlets' workflow.

    Requires at least one of n_singlets / n_triplets to be > 0.
    """
    from chemtools.programs.molcas.input._utils import (
        auto_label, normalize_atoms, total_electrons,
    )
    from chemtools.programs.molcas.input.seward import render_seward_block
    from chemtools.programs.molcas.input.scf import render_scf_block
    from chemtools.programs.molcas.input.rasscf import (
        compute_active_space_partition, render_rasscf_block,
    )
    from chemtools.programs.molcas.input.caspt2 import render_caspt2_block
    from chemtools.programs.molcas.input.rassi import (
        render_rassi_block, render_jobiph_copy,
    )
    from chemtools.programs.molcas.input.lint import lint_molcas_input

    if n_singlets <= 0 and n_triplets <= 0:
        return {
            "verdict": "missing_states",
            "error": "missing_states",
            "message": "At least one of n_singlets / n_triplets must be > 0.",
        }

    atoms_norm = auto_label(normalize_atoms(atoms))
    n_elec = total_electrons(atoms_norm, charge)
    if n_elec % 2 != 0:
        return {
            "verdict": "open_shell_unsupported",
            "error": "open_shell_unsupported",
            "message": (
                f"Total electron count {n_elec} is odd — this orchestrator currently "
                "assumes a closed-shell SCF starting point. Use prepare_molcas_casscf_setup "
                "for open-shell cases."
            ),
        }

    # Resolve the CAS partition (used identically for both spin groups)
    partition = compute_active_space_partition(
        n_electrons=n_elec,
        cas_active_electrons=cas_active_electrons,
        cas_active_orbitals=cas_active_orbitals,
        n_symmetries=n_symmetries,
        n_basis_per_symmetry=n_basis_per_symmetry,
        n_inactive_per_symmetry=rasscf_inactive_per_symmetry,
        active_per_symmetry=rasscf_active_per_symmetry,
    )

    blocks: list[str] = [f">>> Export MOLCAS_MEM={memory_mb}\n"]
    # 1. SEWARD
    blocks.append(
        render_seward_block(
            atoms=atoms_norm,
            basis=basis,
            title=title or f"Excited states — {n_singlets}S + {n_triplets}T",
            symmetry=symmetry,
            geometry_units=geometry_units,
            inline_basis=inline_basis,
        )
    )

    # 2. SCF — closed-shell singlet starting point
    blocks.append(
        render_scf_block(
            n_electrons=n_elec,
            multiplicity=1,
            n_symmetries=n_symmetries,
            occupied_per_symmetry=occupied_per_symmetry,
            title=title,
        )
    )

    # 3-4. RASSCF per multiplicity, with JobIph copy in between
    jobiph_groups: list[dict] = []
    if n_singlets > 0:
        blocks.append(
            render_rasscf_block(
                multiplicity=1,
                state_symmetry=1,
                nactel=partition["nactel"],
                frozen=partition["frozen"],
                inactive=partition["inactive"],
                ras2=partition["ras2"],
                ras1=partition["ras1"],
                ras3=partition["ras3"],
                title=f"{title or 'Excited states'} — {n_singlets} singlets",
                n_roots=n_singlets,
                root_for_optimization=1,
            )
        )
        blocks.append(render_jobiph_copy("JOB001"))
        jobiph_groups.append({"name": "JOB001", "n_states": n_singlets, "multiplicity": 1})

    if n_triplets > 0:
        blocks.append(
            render_rasscf_block(
                multiplicity=3,
                state_symmetry=1,
                nactel=partition["nactel"],
                frozen=partition["frozen"],
                inactive=partition["inactive"],
                ras2=partition["ras2"],
                ras1=partition["ras1"],
                ras3=partition["ras3"],
                title=f"{title or 'Excited states'} — {n_triplets} triplets",
                n_roots=n_triplets,
                root_for_optimization=1,
            )
        )
        idx = len(jobiph_groups) + 1
        blocks.append(render_jobiph_copy(f"JOB00{idx}"))
        jobiph_groups.append({"name": f"JOB00{idx}", "n_states": n_triplets, "multiplicity": 3})

    # 5-6. CASPT2 per group
    method_upper = method.upper()
    caspt2_variant = "MS" if method_upper in {"MS-CASPT2", "CASPT2"} and (n_singlets > 1 or n_triplets > 1) else (
        method_upper.split("-")[0] if method_upper.startswith(("MS", "XMS", "RMS", "XDW")) else "SS"
    )
    do_caspt2 = method_upper in {"CASPT2", "MS-CASPT2", "XMS-CASPT2", "RMS-CASPT2", "XDW-CASPT2"}
    if do_caspt2:
        # Use the right variant keyword based on n_roots; SS if N=1, MS if N>1
        # but allow user-specified XMS/RMS/XDW to win
        if method_upper.startswith(("XMS", "RMS", "XDW")):
            base_variant = method_upper.split("-")[0]
        elif method_upper in {"MS-CASPT2", "CASPT2"}:
            base_variant = "MS"
        else:
            base_variant = "SS"

        for g_idx, group in enumerate(jobiph_groups):
            # CRITICAL: restore THIS group's JobIph as $Project.JobIph before
            # CASPT2 reads. Without this, CASPT2 reads whatever the last
            # RASSCF wrote (the triplet one if both ran) — silently wrong.
            blocks.append(f">>COPY {group['name']} $Project.JobIph\n")
            n_states_in_group = group["n_states"]
            this_variant = base_variant if n_states_in_group > 1 else "SS"
            mult = group["multiplicity"]
            blocks.append(
                render_caspt2_block(
                    title=f"CASPT2 — {n_states_in_group} {'singlets' if mult == 1 else 'triplets'}",
                    variant=this_variant,  # type: ignore[arg-type]
                    n_roots=n_states_in_group,
                    frozen_per_symmetry=partition["frozen"] if any(partition["frozen"]) else None,
                    ipea_shift=ipea_shift,
                    imaginary_shift=imaginary_shift,
                )
            )

    # 7. RASSI for SOC or transition properties
    needs_rassi = compute_soc or (properties and len(jobiph_groups) > 1)
    if needs_rassi:
        rassi_props: list[tuple[str, int]] | None = None
        if properties:
            rassi_props = []
            for prop in properties:
                # Default to 3 components (x/y/z) if not specified
                for comp in (1, 2, 3):
                    rassi_props.append((prop, comp))
        blocks.append(
            render_rassi_block(
                title="SOC + transition properties" if compute_soc else "Transition properties",
                jobiph_groups=jobiph_groups,
                e_job=True,
                spin_orbit=compute_soc,
                natural_orbitals=min(n_singlets, 3) if n_singlets > 0 else None,
                properties=rassi_props,
                so_properties=rassi_props if compute_soc else None,
            )
        )

    input_text = "\n".join(blocks)
    lint_issues = lint_molcas_input(input_text)
    n_errors = sum(1 for i in lint_issues if i.get("level") == "error")

    result: dict[str, Any] = {
        "verdict": "ready_to_launch" if n_errors == 0 else "lint_blocked",
        "workflow_steps": [
            "SEWARD",
            "SCF",
            *([f"RASSCF (singlets, {n_singlets} roots)"] if n_singlets > 0 else []),
            *([f"RASSCF (triplets, {n_triplets} roots)"] if n_triplets > 0 else []),
            *([f"CASPT2 × {len(jobiph_groups)} group(s)"] if do_caspt2 else []),
            *(["RASSI (SOC + properties)"] if needs_rassi else []),
        ],
        "active_space": {
            "cas_active_electrons": cas_active_electrons,
            "cas_active_orbitals": cas_active_orbitals,
            "partition": partition,
        },
        "spin_groups": jobiph_groups,
        "method": method_upper,
        "compute_soc": compute_soc,
        "input_text": input_text,
        "lint_issues": lint_issues,
        "n_lint_errors": n_errors,
        "n_lint_warnings": sum(1 for i in lint_issues if i.get("level") == "warning"),
    }

    # Optional: write input + build launch plan
    target_path: Path | None = None
    if write_input_to:
        target_path = Path(write_input_to)
    elif job_name:
        target_path = Path.cwd() / f"{job_name}.input"
    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(input_text, encoding="utf-8")
        result["input_path"] = str(target_path)
        if n_errors == 0:
            plan = prepare_launch(
                str(target_path),
                profile=profile,
                requested_np=requested_np,
                job_name=job_name,
                apptainer_sif=apptainer_sif,
            )
            result["launch_plan"] = plan
            result["next_actions"] = [
                {
                    "tool": "shell_execute",
                    "args": {"command": plan["command_str"], "env": plan["env"]},
                    "rationale": (
                        f"Run the excited-states workflow: SCF + "
                        f"{n_singlets}S/{n_triplets}T RASSCF(s)"
                        + (f" + {len(jobiph_groups)}-group {caspt2_variant}-CASPT2" if do_caspt2 else "")
                        + (" + RASSI (SOC)" if compute_soc else "")
                        + "."
                    ),
                },
                *([{
                    "tool": "parse_molcas_rassi",
                    "args": {"output_file": str(target_path).replace(".input", ".out")},
                    "rationale": "After completion, extract SOC matrix elements and SF/SO eigenstate energies.",
                }] if needs_rassi else []),
                {
                    "tool": "parse_molcas_output",
                    "args": {"output_file": str(target_path).replace(".input", ".out")},
                    "rationale": "Inspect SA-CASSCF root energies and any CASPT2 results.",
                },
            ]
        else:
            result["next_actions"] = [
                {
                    "tool": "lint_molcas_input",
                    "args": {"input_text": input_text},
                    "rationale": f"Fix the {n_errors} lint error(s) before launching.",
                }
            ]
    return result


# --- Opt + Freq workflow orchestrator -----------------------------------------


def prepare_opt_freq_workflow(
    *,
    atoms: list[dict],
    charge: int,
    multiplicity: int,
    basis: str | dict[str, str],
    method: str = "CASSCF",
    cas_active_electrons: int | None = None,
    cas_active_orbitals: int | None = None,
    do_frequency: bool = True,
    do_optimization: bool = True,
    transition_state: bool = False,
    title: str | None = None,
    geometry_units: str = "angstrom",
    symmetry: str | None = None,
    n_symmetries: int = 1,
    occupied_per_symmetry: list[int] | None = None,
    n_basis_per_symmetry: list[int] | None = None,
    rasscf_inactive_per_symmetry: list[int] | None = None,
    rasscf_active_per_symmetry: list[int] | None = None,
    inline_basis: bool = True,
    memory_mb: int = 2000,
    max_opt_iterations: int | None = None,
    numerical_gradients: bool = False,
    iroot_freq: int | None = None,
    job_name: str | None = None,
    write_input_to: str | None = None,
    apptainer_sif: str | None = None,
    profile: dict | None = None,
    requested_np: int = 1,
) -> dict[str, Any]:
    """Geometry-optimization + analytical-frequency workflow orchestrator.

    Generates a Molcas input that wraps SEWARD + (SCF on first iter only) +
    RASSCF + ALASKA + SLAPAF in an EMIL ``>>> Do while <<< / >>> ENDDO <<<``
    loop. After convergence the optimization exits the loop and (if
    do_frequency=True) MCKINLEY + MCLR run on the converged geometry to
    produce analytical second derivatives + harmonic frequencies.

    Supports:
      * method='SCF' or 'CASSCF' (CASPT2 opt requires GRDT — defer to future)
      * Single-point freq (do_optimization=False) — just MCKINLEY + MCLR
      * Transition-state search (transition_state=True) — passes TS to SLAPAF
      * Constrained / numerical opts via the SLAPAF / ALASKA toggles
    """
    from chemtools.programs.molcas.input._utils import (
        auto_label, normalize_atoms, total_electrons,
    )
    from chemtools.programs.molcas.input.seward import render_seward_block
    from chemtools.programs.molcas.input.scf import render_scf_block
    from chemtools.programs.molcas.input.rasscf import (
        compute_active_space_partition, render_rasscf_block,
    )
    from chemtools.programs.molcas.input.opt_freq import (
        render_alaska_block, render_slapaf_block,
        render_mckinley_block, render_mclr_block,
        do_while_open, do_while_close,
        if_iter_one_open, if_iter_one_close,
    )
    from chemtools.programs.molcas.input.lint import lint_molcas_input

    if not (do_optimization or do_frequency):
        return {
            "verdict": "nothing_to_do",
            "error": "nothing_to_do",
            "message": "Set at least one of do_optimization / do_frequency to True.",
        }

    method_upper = method.upper()
    if method_upper not in {"SCF", "HF", "CASSCF", "RASSCF"}:
        return {
            "verdict": "unsupported_method",
            "error": "unsupported_method",
            "message": (
                f"opt+freq orchestrator currently supports SCF/HF/CASSCF/RASSCF; "
                f"got {method!r}. CASPT2 opt needs GRDT plumbing — future work."
            ),
        }
    use_cas = method_upper in {"CASSCF", "RASSCF"}

    atoms_norm = auto_label(normalize_atoms(atoms))
    n_elec = total_electrons(atoms_norm, charge)

    partition = None
    if use_cas:
        if cas_active_electrons is None or cas_active_orbitals is None:
            return {
                "verdict": "missing_cas_spec",
                "error": "missing_cas_spec",
                "message": "CASSCF method needs cas_active_electrons + cas_active_orbitals.",
            }
        partition = compute_active_space_partition(
            n_electrons=n_elec,
            cas_active_electrons=cas_active_electrons,
            cas_active_orbitals=cas_active_orbitals,
            n_symmetries=n_symmetries,
            n_basis_per_symmetry=n_basis_per_symmetry,
            n_inactive_per_symmetry=rasscf_inactive_per_symmetry,
            active_per_symmetry=rasscf_active_per_symmetry,
        )

    blocks: list[str] = [f">>> Export MOLCAS_MEM={memory_mb}\n"]

    # When optimizing, basis + geom MUST live in &GATEWAY so they persist to
    # the RunFile and the loop's bare `&SEWARD` calls can find them.
    # Single-point freq can use a plain &SEWARD block (single integration).
    blocks.append(
        render_seward_block(
            atoms=atoms_norm,
            basis=basis,
            title=title or f"{method_upper} opt+freq",
            symmetry=symmetry,
            geometry_units=geometry_units,
            inline_basis=inline_basis,
            use_gateway=do_optimization,
        )
    )

    if do_optimization:
        blocks.append(do_while_open())
        # SEWARD re-runs each iteration to re-integrate at the updated geometry
        blocks.append("&SEWARD\nEnd of input\n")
        # SCF: for CASSCF opt, gate to iter 1 only (just gives initial orbitals
        # for RASSCF; subsequent RASSCF iterations warm-start from JobIph). For
        # SCF-only opt, SCF runs every iteration (recomputes density at the new
        # geometry, warm-starting from ScfOrb).
        if use_cas:
            blocks.append(if_iter_one_open())
        blocks.append(
            render_scf_block(
                n_electrons=n_elec,
                multiplicity=multiplicity,
                n_symmetries=n_symmetries,
                occupied_per_symmetry=occupied_per_symmetry,
                title=title,
            )
        )
        if use_cas:
            blocks.append(if_iter_one_close())
        # RASSCF every iteration (if we're doing CAS opt)
        if use_cas:
            blocks.append(
                render_rasscf_block(
                    multiplicity=multiplicity,
                    state_symmetry=1,
                    nactel=partition["nactel"],
                    frozen=partition["frozen"],
                    inactive=partition["inactive"],
                    ras2=partition["ras2"],
                    ras1=partition["ras1"],
                    ras3=partition["ras3"],
                    title=title,
                    n_roots=1,
                )
            )
        # Gradients
        blocks.append(render_alaska_block(numerical=numerical_gradients))
        # Step + new geometry
        blocks.append(
            render_slapaf_block(
                iterations=max_opt_iterations,
                transition_state=transition_state,
            )
        )
        blocks.append(do_while_close())

    if do_frequency:
        # After opt convergence (or for single-point freq), compute the analytic
        # Hessian. If we ran a full opt loop the RASSCF wave function is
        # already current at the converged geometry; if not (single-point
        # freq), we need to run SCF + RASSCF first.
        if not do_optimization:
            blocks.append("&SEWARD\nEnd of input\n")
            blocks.append(
                render_scf_block(
                    n_electrons=n_elec,
                    multiplicity=multiplicity,
                    n_symmetries=n_symmetries,
                    occupied_per_symmetry=occupied_per_symmetry,
                    title=title,
                )
            )
            if use_cas:
                blocks.append(
                    render_rasscf_block(
                        multiplicity=multiplicity,
                        state_symmetry=1,
                        nactel=partition["nactel"],
                        frozen=partition["frozen"],
                        inactive=partition["inactive"],
                        ras2=partition["ras2"],
                        ras1=partition["ras1"],
                        ras3=partition["ras3"],
                        title=title,
                        n_roots=1,
                    )
                )
        blocks.append(render_mckinley_block(title=f"{title or method_upper} second derivatives"))
        if use_cas:
            blocks.append(render_mclr_block(iroot=iroot_freq, title="MCLR analytic Hessian"))

    input_text = "".join(blocks)
    lint_issues = lint_molcas_input(input_text)
    n_errors = sum(1 for i in lint_issues if i.get("level") == "error")

    result: dict[str, Any] = {
        "verdict": "ready_to_launch" if n_errors == 0 else "lint_blocked",
        "workflow_steps": (
            (["GATEWAY+SEWARD"] +
             (["Opt loop (SEWARD+SCF+(RASSCF)+ALASKA+SLAPAF)"] if do_optimization else []) +
             (["MCKINLEY + MCLR (analytic Hessian + frequencies)"] if do_frequency else []))
        ),
        "method": method_upper,
        "active_space": (
            {"cas_active_electrons": cas_active_electrons,
             "cas_active_orbitals": cas_active_orbitals,
             "partition": partition}
            if use_cas else None
        ),
        "do_optimization": do_optimization,
        "do_frequency": do_frequency,
        "transition_state": transition_state,
        "input_text": input_text,
        "lint_issues": lint_issues,
        "n_lint_errors": n_errors,
        "n_lint_warnings": sum(1 for i in lint_issues if i.get("level") == "warning"),
    }

    target_path: Path | None = None
    if write_input_to:
        target_path = Path(write_input_to)
    elif job_name:
        target_path = Path.cwd() / f"{job_name}.input"
    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(input_text, encoding="utf-8")
        result["input_path"] = str(target_path)
        if n_errors == 0:
            plan = prepare_launch(
                str(target_path),
                profile=profile,
                requested_np=requested_np,
                job_name=job_name,
                apptainer_sif=apptainer_sif,
            )
            result["launch_plan"] = plan
            next_actions = [
                {
                    "tool": "shell_execute",
                    "args": {"command": plan["command_str"], "env": plan["env"]},
                    "rationale": (
                        f"Run the {method_upper} "
                        + ("TS opt" if transition_state else "geometry opt" if do_optimization else "")
                        + (" + " if do_optimization and do_frequency else "")
                        + ("analytic-Hessian freq" if do_frequency else "")
                        + "."
                    ),
                },
            ]
            if do_optimization:
                next_actions.append(
                    {
                        "tool": "parse_molcas_trajectory",
                        "args": {"output_file": str(target_path).replace(".input", ".out")},
                        "rationale": "After completion, inspect the SLAPAF iteration history + converged geometry.",
                    }
                )
            if do_frequency:
                next_actions.extend([
                    {
                        "tool": "parse_molcas_frequencies",
                        "args": {"output_file": str(target_path).replace(".input", ".out")},
                        "rationale": "Extract harmonic frequencies + IR intensities + normal modes.",
                    },
                    {
                        "tool": "parse_molcas_thermochem",
                        "args": {"output_file": str(target_path).replace(".input", ".out")},
                        "rationale": "Pull ZPVE + S + U + H + G per temperature.",
                    },
                ])
            result["next_actions"] = next_actions
        else:
            result["next_actions"] = [
                {
                    "tool": "lint_molcas_input",
                    "args": {"input_text": input_text},
                    "rationale": f"Fix the {n_errors} lint error(s) before launching.",
                }
            ]
    return result
