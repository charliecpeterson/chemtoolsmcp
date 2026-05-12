"""Single-dispatch case analysis for Molcas outputs.

Two complementary tools:

  summarize_molcas_output(output_file) -> flat structured summary
    Method, primary energy, active space, geometry, key warnings, ref
    weight, frequencies/thermochem when present. Answers "what happened
    in this run?" in one call so the agent doesn't have to chain
    parse_molcas_output + parse_molcas_thermochem + parse_molcas_frequencies
    + parse_final_geometry + parse_last_mo_block by hand.

  analyze_molcas_case(output_file) -> summary + verdict + issues
    Wraps summarize_molcas_output with a quality assessment: runs
    validate_molcas_caspt2_setup when CASPT2 is present, analyze_active_space
    when RASSCF is present, cross-checks charge/spin parity and active-space
    coherence, and emits a verdict (healthy / caution / problematic) +
    issues list + recommended next_actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.core.common import read_text
from chemtools.programs.molcas.parse.output import parse_output_full
from chemtools.programs.molcas.parse.thermochem import parse_thermochem_block
from chemtools.programs.molcas.parse.freq import parse_last_freq_block
from chemtools.programs.molcas.parse.geometry import parse_final_geometry
from chemtools.programs.molcas.parse.mos import parse_last_mo_block
from chemtools.programs.molcas.strategy.active_space import (
    analyze_active_space,
    validate_caspt2_setup,
)


# ---------------------------------------------------------------------------
# summarize_molcas_output
# ---------------------------------------------------------------------------


def _pick_real_freqs(modes: list[dict]) -> list[float]:
    """Drop translation/rotation modes (|f| < 5 cm-1 and zero-occupation rows)."""
    out: list[float] = []
    for m in modes:
        f = m.get("frequency_cm1")
        if f is None:
            continue
        if abs(f) < 5.0:
            continue
        out.append(float(f))
    return out


def summarize_molcas_output(output_file: str) -> dict[str, Any]:
    """Return a flat structured summary of a Molcas .out / .log run.

    Output (all fields optional — missing means the corresponding info
    wasn't in the run):
        method                  e.g. "CASPT2", "RASSCF root 1", "SCF"
        modules_run             list of module names in order
        primary_energy_au       primary energy via the parser's hierarchy
        scf_energy_au           SCF total
        rasscf_root1_au         RASSCF root 1 (if present)
        caspt2_root1_au         CASPT2 root 1 (if present)
        caspt2_reference_weight 0..1 (CASPT2 only)
        active_space            {n_active_electrons, n_active_orbitals,
                                 inactive, basis_functions, spin}
        geometry                {atoms: [{symbol, x, y, z}], n_atoms, units}
        bond_lengths            list of {atoms, length_angstrom} (auto-computed
                                from final geometry; up to 6 shortest pairs)
        frequencies_cm1         list of real-valued frequencies (cm⁻¹) — drops
                                trans/rot near-zero modes
        imaginary_frequencies   negative-valued (real imaginary) frequencies
        zpve_kcal_per_mol       zero-point vibrational energy
        thermochem_298_15       enthalpy/Gibbs/entropy at 298.15 K (if freq run)
        warnings                module-level warnings from parse_molcas_output
        n_warnings              count
        log_path                input path (for traceback)
    """
    log_path = Path(output_file)
    if not log_path.is_file():
        raise FileNotFoundError(f"Output file not found: {output_file}")
    text = read_text(output_file)
    full = parse_output_full(output_file, text)

    summary: dict[str, Any] = {
        "log_path": output_file,
        "modules_run": [t.get("module") for t in full.get("task_payloads", [])],
        "warnings": full.get("warnings") or [],
    }
    summary["n_warnings"] = len(summary["warnings"])

    es = full.get("energy_summary") or {}
    summary["method"] = es.get("primary_label")
    summary["primary_energy_au"] = es.get("primary_energy_hartree")
    summary["scf_energy_au"] = es.get("scf_total_hartree")
    rasscf_roots = es.get("rasscf_root_energies") or []
    caspt2_roots = es.get("caspt2_root_energies") or []
    if rasscf_roots:
        summary["rasscf_root1_au"] = rasscf_roots[0].get("energy_hartree")
    if caspt2_roots:
        summary["caspt2_root1_au"] = caspt2_roots[0].get("energy_hartree")

    # Find CASPT2 reference weight in the CASPT2 task payload (if present)
    for tp in full.get("task_payloads", []):
        if tp.get("module") == "CASPT2":
            details = tp.get("details", {}) or {}
            roots = details.get("roots") or []
            if roots:
                rw = roots[0].get("reference_weight")
                if rw is not None:
                    summary["caspt2_reference_weight"] = rw
            break

    # Active space from RASSCF task
    rasscf_task = next(
        (t for t in full.get("task_payloads", []) if t.get("module") == "RASSCF"),
        None,
    )
    if rasscf_task:
        details = rasscf_task.get("details", {}) or {}
        specs = details.get("orbital_specs", {}) or {}
        wave = details.get("wave_function", {}) or {}
        summary["active_space"] = {
            "n_active_electrons": wave.get("active_electrons"),
            "n_active_orbitals": sum(specs.get("active", []) or [0]),
            "n_inactive_orbitals": sum(specs.get("inactive", []) or [0]),
            "n_basis_functions": sum(specs.get("basis_functions", []) or [0]),
            "spin": wave.get("spin"),
            "n_symmetries": specs.get("n_symmetries"),
        }

    # Geometry — final converged structure
    geom = parse_final_geometry(text)
    if geom and geom.get("atoms"):
        atoms = geom["atoms"]
        summary["geometry"] = {
            "atoms": [
                {"symbol": a.get("symbol"), "x": a.get("x"), "y": a.get("y"), "z": a.get("z")}
                for a in atoms
            ],
            "n_atoms": len(atoms),
            "units": geom.get("units", "angstrom"),
        }
        if len(atoms) <= 12:  # bond table only for small systems
            summary["bond_lengths"] = _compute_short_bond_table(atoms)

    # Frequencies — from MCLR analytic Hessian path
    freq_block = parse_last_freq_block(text)
    if freq_block and freq_block.get("modes"):
        modes = freq_block["modes"]
        real_freqs = _pick_real_freqs(modes)
        summary["frequencies_cm1"] = [f for f in real_freqs if f > 0]
        summary["imaginary_frequencies_cm1"] = [f for f in real_freqs if f < 0]
        summary["zpve_kcal_per_mol"] = freq_block.get("zpve_kcal_per_mol")

    # Thermochem (also from freq run)
    thermo = parse_thermochem_block(text)
    if thermo and thermo.get("standard_298_15"):
        std = thermo["standard_298_15"]
        summary["thermochem_298_15"] = {
            "temperature_k": std.get("temperature_k"),
            "zpve_kcal_per_mol": thermo.get("zpve_kcal_per_mol"),
            "thermal_enthalpy_au": std.get("thermal_enthalpy_au"),
            "thermal_gibbs_au": std.get("thermal_gibbs_au"),
            "enthalpy_total_au": std.get("enthalpy_total_au"),
            "gibbs_total_au": std.get("gibbs_total_au"),
            "entropy_total_kcal_per_mol_K": (std.get("entropy_kcal_per_mol_K") or {}).get("total"),
        }

    return summary


def _compute_short_bond_table(atoms: list[dict]) -> list[dict[str, Any]]:
    """Return up to 6 shortest interatomic distances (Angstrom). Skips
    duplicates and limits to chemically-relevant range (<= 2.5 Å)."""
    pairs: list[tuple[float, int, int]] = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            dx = atoms[i]["x"] - atoms[j]["x"]
            dy = atoms[i]["y"] - atoms[j]["y"]
            dz = atoms[i]["z"] - atoms[j]["z"]
            r = (dx * dx + dy * dy + dz * dz) ** 0.5
            if r <= 2.5:
                pairs.append((r, i, j))
    pairs.sort()
    out: list[dict[str, Any]] = []
    for r, i, j in pairs[:6]:
        out.append({
            "atoms": [f"{atoms[i].get('symbol')}{i+1}", f"{atoms[j].get('symbol')}{j+1}"],
            "length_angstrom": round(r, 4),
        })
    return out


# ---------------------------------------------------------------------------
# analyze_molcas_case
# ---------------------------------------------------------------------------


def analyze_molcas_case(output_file: str) -> dict[str, Any]:
    """Run summarize_molcas_output + quality cross-checks.

    Adds to the summary:
      verdict             "healthy" | "caution" | "problematic"
      issues              list of {severity, message, hint} dicts
      next_actions        actionable list for the agent
      caspt2_validation   verdict from validate_molcas_caspt2_setup (if CASPT2)
      active_space_quality verdict from analyze_molcas_active_space (if RASSCF)
    """
    from chemtools.core.issues import IssueCollector
    from chemtools.core.case_analysis import (
        classify_imaginary_modes,
        check_charge_spin_parity,
    )

    summary = summarize_molcas_output(output_file)
    text = read_text(output_file)
    full = parse_output_full(output_file, text)

    coll = IssueCollector()
    next_actions: list[dict[str, Any]] = []

    # ----- RASSCF active-space quality -----
    rasscf_task = next(
        (t for t in full.get("task_payloads", []) if t.get("module") == "RASSCF"),
        None,
    )
    if rasscf_task:
        try:
            aa = analyze_active_space(rasscf_task.get("details", {}) or {})
            summary["active_space_quality"] = {
                "verdict": aa.get("verdict"),
                "per_root_quality": aa.get("per_root_quality"),
            }
            av = aa.get("verdict")
            if av == "poor":
                coll.add(
                    "problematic",
                    "Active space verdict is 'poor' (no truly active orbitals — all near 0 or 2 occupation).",
                    hint="Either shrink the CAS to match what's chemically active, or use a different reference (HF/DFT may be more appropriate).",
                )
            elif av == "marginal":
                coll.add(
                    "caution",
                    "Active space verdict is 'marginal' (fewer than half the orbitals carry truly active occupations).",
                    hint="Consider character-aware orbital swaps (refine_molcas_active_space) or trimming the CAS.",
                )
                next_actions.append({
                    "tool": "refine_molcas_active_space",
                    "args": {"output_file": output_file},
                    "rationale": "Marginal active space — refine_active_space can suggest character-aware swaps.",
                })
        except Exception as exc:  # noqa: BLE001
            coll.add("info", f"analyze_active_space failed: {exc}", hint="RASSCF data may be malformed.")

    # ----- CASPT2 reference weight + intruder check -----
    caspt2_task = next(
        (t for t in full.get("task_payloads", []) if t.get("module") == "CASPT2"),
        None,
    )
    if caspt2_task:
        try:
            cv = validate_caspt2_setup(caspt2_task.get("details", {}) or {})
            summary["caspt2_validation"] = {
                "verdict": cv.get("verdict"),
                "warnings": cv.get("warnings"),
            }
            v = cv.get("verdict")
            if v == "unreliable":
                coll.add(
                    "problematic",
                    "CASPT2 reference weight below trust threshold (<0.70).",
                    hint="Increase CAS size, or add `Imaginary 0.1` to suppress intruders, or check for state mixing (try MS/XMS CASPT2).",
                )
                next_actions.append({
                    "tool": "validate_molcas_caspt2_setup",
                    "args": {"output_file": output_file},
                    "rationale": "Reference weight is below trust band — review intruder + reference-quality diagnostics.",
                })
            elif v == "caution":
                coll.add(
                    "caution",
                    "CASPT2 reference weight in caution band (0.70-0.85).",
                    hint="Result is likely meaningful but consider tighter active space or MS-CASPT2 if multiple states mix.",
                )
        except Exception as exc:  # noqa: BLE001
            coll.add("info", f"validate_caspt2_setup failed: {exc}", hint="CASPT2 data may be malformed.")

    # ----- Imaginary frequencies (generic helper from core) -----
    imag = summary.get("imaginary_frequencies_cm1") or []
    classified = classify_imaginary_modes(imag)
    physical_imag = classified["physical"]
    artifact_imag = classified["artifacts"]
    if physical_imag:
        if len(physical_imag) == 1:
            coll.add(
                "info",
                f"One imaginary frequency: {physical_imag[0]:.2f} cm⁻¹. "
                "Correct for a TS, suspicious for a minimum.",
                hint="If you expected a minimum, follow the imaginary mode (displace_molcas_geometry_along_mode) and re-optimize.",
            )
        else:
            coll.add(
                "caution",
                f"{len(physical_imag)} physical imaginary frequencies: "
                f"{[round(f, 1) for f in physical_imag]}",
                hint="Multi-imaginary means the geometry isn't a stationary point — opt likely incomplete or wrong starting point.",
            )
    if artifact_imag:
        coll.add(
            "info",
            f"{len(artifact_imag)} small-magnitude imaginary mode(s) "
            f"(|f| < 50 cm⁻¹): {[round(f, 1) for f in artifact_imag]}. "
            "Likely translation/rotation projection artifacts, not chemical.",
            hint="Safe to ignore unless the geometry is unusual (linear molecule, weak bond).",
        )

    # ----- Cross-check: charge × spin parity (generic) -----
    if rasscf_task:
        details = rasscf_task.get("details", {}) or {}
        wave = details.get("wave_function", {}) or {}
        n_act_e = wave.get("active_electrons")
        spin = wave.get("spin")  # half-integer S → multiplicity = 2S+1
        if n_act_e is not None and spin is not None:
            mult = int(round(2 * float(spin))) + 1
            parity_issue = check_charge_spin_parity(n_act_e, mult)
            if parity_issue:
                # Replace the core's generic hint with the Molcas-specific
                # one (compute_molcas_active_space_partition).
                parity_issue["hint"] = (
                    "Recompute the active-space partition with "
                    "compute_molcas_active_space_partition."
                )
                coll.add(**parity_issue)

    # ----- Lots of warnings is itself a caution -----
    n_warn = summary.get("n_warnings", 0)
    if n_warn >= 10:
        coll.add(
            "caution",
            f"Run emitted {n_warn} warnings.",
            hint="Inspect warnings list — many warnings can mask real failures.",
        )

    summary["verdict"] = coll.verdict
    summary["issues"] = coll.issues
    summary["next_actions"] = next_actions
    return summary
