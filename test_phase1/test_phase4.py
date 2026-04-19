#!/usr/bin/env python3
"""Smoke-test all Phase 4 features locally."""

import json
import os
import sys
from pathlib import Path

# Add repo root to path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FREQ_OUTPUT = REPO / "nwchemaitest" / "am2pba3h2_s-031326.out"
OPT_OUTPUT = REPO / "nwchemaitest" / "am2pba3h2_s_opt2.out"

PASS = 0
FAIL = 0


def assert_(condition, message=""):
    if not condition:
        raise AssertionError(message)


def report(name, result, check_fn=None):
    global PASS, FAIL
    try:
        if check_fn:
            check_fn(result)
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        FAIL += 1


# ============================================================
print("\n=== P4-1. parse_nwchem_thermochem ===")
# ============================================================
from chemtools import parse_nwchem_thermochem

if FREQ_OUTPUT.exists():
    result = parse_nwchem_thermochem(str(FREQ_OUTPUT))
    report("returns E_scf", result,
           lambda r: assert_(r["E_scf_hartree"] is not None, "no E_scf"))
    report("E_scf is reasonable", result,
           lambda r: assert_(r["E_scf_hartree"] < -2000, f"E_scf={r['E_scf_hartree']}"))
    report("returns ZPE", result,
           lambda r: assert_(r["ZPE_hartree"] is not None and r["ZPE_hartree"] > 0,
                              f"ZPE={r['ZPE_hartree']}"))
    report("returns ZPE_kcal", result,
           lambda r: assert_(r["ZPE_kcal_mol"] is not None and r["ZPE_kcal_mol"] > 100,
                              f"ZPE_kcal={r['ZPE_kcal_mol']}"))
    report("returns H(T)", result,
           lambda r: assert_(r["H_T_hartree"] is not None, "no H_T"))
    report("H(T) > E_scf (positive thermal correction)", result,
           lambda r: assert_(r["H_T_hartree"] > r["E_scf_hartree"],
                              "H should be above E_scf"))
    report("returns G(T)", result,
           lambda r: assert_(r["G_T_hartree"] is not None, "no G_T"))
    report("G(T) < H(T) (entropy lowers G)", result,
           lambda r: assert_(r["G_T_hartree"] < r["H_T_hartree"],
                              "G should be below H"))
    report("returns entropy", result,
           lambda r: assert_(r["S_total_cal_mol_K"] is not None and r["S_total_cal_mol_K"] > 0,
                              f"S={r['S_total_cal_mol_K']}"))
    report("returns Cv", result,
           lambda r: assert_(r["Cv_total_cal_mol_K"] is not None and r["Cv_total_cal_mol_K"] > 0))
    report("detects imaginary modes", result,
           lambda r: assert_(r["imaginary_modes_count"] > 0, "should have imaginary modes"))
    report("warns about imaginary modes", result,
           lambda r: assert_(any("imaginary" in w for w in r["warnings"]),
                              "no imaginary mode warning"))
    report("returns temperature", result,
           lambda r: assert_(abs(r["T_K"] - 298.15) < 0.1, f"T={r['T_K']}"))
    report("returns method", result,
           lambda r: assert_(r["method"] == "DFT", f"method={r['method']}"))
    report("E+ZPE computed", result,
           lambda r: assert_(r["E_plus_ZPE_hartree"] is not None))
    report("TS computed", result,
           lambda r: assert_(r["TS_hartree"] is not None and r["TS_hartree"] > 0))
    print(f"         E_scf={result['E_scf_hartree']:.6f} Ha")
    print(f"         ZPE={result['ZPE_kcal_mol']:.1f} kcal/mol")
    print(f"         H(T)={result['H_T_hartree']:.6f} Ha")
    print(f"         G(T)={result['G_T_hartree']:.6f} Ha")
    print(f"         S={result['S_total_cal_mol_K']:.2f} cal/mol/K")
else:
    print(f"  SKIP  freq output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== P4-1b. parse_nwchem_thermochem MCP dispatch ===")
# ============================================================
from chemtools.mcp.nwchem import dispatch_tool

if FREQ_OUTPUT.exists():
    mcp_result = dispatch_tool("parse_nwchem_thermochem", {
        "output_file": str(FREQ_OUTPUT),
    })
    report("MCP dispatch works", mcp_result,
           lambda r: assert_("E_scf_hartree" in r, "no E_scf_hartree in MCP result"))
    report("MCP result matches direct", mcp_result,
           lambda r: assert_(abs(r["E_scf_hartree"] - result["E_scf_hartree"]) < 1e-10))


# ============================================================
print("\n=== P4-2. summarize_electronic_structure ===")
# ============================================================
from chemtools import summarize_electronic_structure

if FREQ_OUTPUT.exists():
    es = summarize_electronic_structure(str(FREQ_OUTPUT))
    report("returns total_energy", es,
           lambda r: assert_(r["total_energy_hartree"] is not None))
    report("returns HOMO-LUMO gap", es,
           lambda r: assert_(r["homo_lumo_gap_ev"] is not None and r["homo_lumo_gap_ev"] > 0,
                              f"gap={r['homo_lumo_gap_ev']}"))
    report("gap is reasonable (< 15 eV)", es,
           lambda r: assert_(r["homo_lumo_gap_ev"] < 15, f"gap={r['homo_lumo_gap_ev']}"))
    report("has HOMO info", es,
           lambda r: assert_(r["homo"] is not None and "vector_number" in r["homo"]))
    report("has LUMO info", es,
           lambda r: assert_(r["lumo"] is not None and "energy_ev" in r["lumo"]))
    report("HOMO has dominant_character", es,
           lambda r: assert_(r["homo"]["dominant_character"], "no dominant_character"))
    report("returns somo_count", es,
           lambda r: assert_(r["somo_count"] == 5, f"somo_count={r['somo_count']}"))
    report("detects Am metal center", es,
           lambda r: assert_(any(m["element"] == "Am" for m in r["metal_centers"]),
                              "no Am metal center"))
    report("Am has charge", es,
           lambda r: assert_(
               any(m["mulliken_charge"] is not None and m["mulliken_charge"] > 0
                   for m in r["metal_centers"]),
               "Am should have positive charge"))
    report("Am has spin density", es,
           lambda r: assert_(
               any(m["mulliken_spin_density"] is not None and m["mulliken_spin_density"] > 4
                   for m in r["metal_centers"]),
               "Am should have ~5 spin density"))
    report("has top_charge_sites", es,
           lambda r: assert_(len(r["top_charge_sites"]) > 0))
    report("has top_spin_density_sites", es,
           lambda r: assert_(len(r["top_spin_density_sites"]) > 0))
    report("total spin density is ~5", es,
           lambda r: assert_(abs(r["total_spin_density"] - 5.0) < 0.5,
                              f"total_spin={r['total_spin_density']}"))
    report("detects charge from output", es,
           lambda r: assert_(r["charge"] == 1, f"charge={r['charge']}"))
    report("detects multiplicity from output", es,
           lambda r: assert_(r["multiplicity"] == 6, f"mult={r['multiplicity']}"))
    report("spin state is consistent", es,
           lambda r: assert_(r["spin_state_consistent"] is True,
                              f"consistent={r['spin_state_consistent']}"))
    report("expected_unpaired = 5", es,
           lambda r: assert_(r["expected_unpaired_electrons"] == 5))
    report("no warnings for good Am complex", es,
           lambda r: assert_(len(r["warnings"]) == 0, f"warnings={r['warnings']}"))
    mc = [m for m in es["metal_centers"] if m["element"] == "Am"][0]
    print(f"         gap={es['homo_lumo_gap_ev']} eV, somo={es['somo_count']}")
    print(f"         Am charge={mc['mulliken_charge']:.2f}, spin={mc['mulliken_spin_density']:.2f}")
else:
    print(f"  SKIP  freq output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== P4-2b. summarize_electronic_structure MCP dispatch ===")
# ============================================================
if FREQ_OUTPUT.exists():
    mcp_es = dispatch_tool("summarize_nwchem_electronic_structure", {
        "output_file": str(FREQ_OUTPUT),
    })
    report("MCP dispatch works", mcp_es,
           lambda r: assert_("homo_lumo_gap_ev" in r))


# ============================================================
print("\n=== P4-3. compute_reaction_energy with thermochem ===")
# ============================================================
from chemtools import compute_reaction_energy

if FREQ_OUTPUT.exists():
    # Same species → all deltas should be zero
    r = compute_reaction_energy(
        species={"A": str(FREQ_OUTPUT), "B": str(FREQ_OUTPUT)},
        reactants={"A": 1},
        products={"B": 1},
        include_thermochem=True,
    )
    report("delta_e is zero (same species)", r,
           lambda r: assert_(abs(r["delta_e_kcal_mol"]) < 1e-8,
                              f"delta_e={r['delta_e_kcal_mol']}"))
    report("delta_e_plus_zpe present", r,
           lambda r: assert_("delta_e_plus_zpe_kcal_mol" in r))
    report("delta_e_plus_zpe is zero", r,
           lambda r: assert_(abs(r["delta_e_plus_zpe_kcal_mol"]) < 1e-8))
    report("delta_h present", r,
           lambda r: assert_("delta_h_kcal_mol" in r))
    report("delta_h is zero", r,
           lambda r: assert_(abs(r["delta_h_kcal_mol"]) < 1e-8))
    report("delta_g present", r,
           lambda r: assert_("delta_g_kcal_mol" in r))
    report("delta_g is zero", r,
           lambda r: assert_(abs(r["delta_g_kcal_mol"]) < 1e-8))
    report("breakdown has H_T", r,
           lambda r: assert_(r["species_breakdown"][0].get("H_T_hartree") is not None))
    report("breakdown has G_T", r,
           lambda r: assert_(r["species_breakdown"][0].get("G_T_hartree") is not None))
    report("warns about imaginary modes", r,
           lambda r: assert_(any("imaginary" in w for w in r["warnings"])))

    # Without thermochem — should NOT have delta_h
    r2 = compute_reaction_energy(
        species={"A": str(FREQ_OUTPUT)},
        reactants={"A": 1},
        products={"A": 1},
    )
    report("no thermochem by default", r2,
           lambda r: assert_("delta_h_kcal_mol" not in r, "should not have delta_h"))
else:
    print(f"  SKIP  freq output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== P4-4. track_spin_state_across_optimization ===")
# ============================================================
from chemtools import track_spin_state_across_optimization

if OPT_OUTPUT.exists():
    ts = track_spin_state_across_optimization(str(OPT_OUTPUT))
    report("returns n_steps", ts,
           lambda r: assert_(r["n_steps"] > 0, f"n_steps={r['n_steps']}"))
    report("has steps list", ts,
           lambda r: assert_(len(r["steps"]) == r["n_steps"]))
    report("each step has energy", ts,
           lambda r: assert_(all(s["energy_hartree"] is not None for s in r["steps"])))
    report("most steps have S2", ts,
           lambda r: assert_(sum(1 for s in r["steps"] if s["s2_computed"] is not None) >= r["n_steps"] - 1,
                              "too many missing S2 values"))
    report("s2_exact is set", ts,
           lambda r: assert_(r["s2_exact"] is not None and r["s2_exact"] > 0))
    report("s2_exact is 8.75 (sextet Am)", ts,
           lambda r: assert_(abs(r["s2_exact"] - 8.75) < 0.01, f"s2_exact={r['s2_exact']}"))
    report("no spin flip detected", ts,
           lambda r: assert_(r["flip_detected"] is False))
    report("flip_steps is empty", ts,
           lambda r: assert_(len(r["flip_steps"]) == 0))
    report("no energy jumps", ts,
           lambda r: assert_(len(r["energy_jumps"]) == 0))
    report("no recommendation for clean optimization", ts,
           lambda r: assert_(r["recommendation"] is None))

    # Check S2 values are monotonically decreasing (roughly)
    s2_vals = [s["s2_computed"] for s in ts["steps"] if s["s2_computed"] is not None]
    report("S2 values are stable (no large jumps)", ts,
           lambda r: assert_(max(s2_vals) - min(s2_vals) < 0.1,
                              f"S2 range: {min(s2_vals):.4f} - {max(s2_vals):.4f}"))
    print(f"         steps={ts['n_steps']}, s2_exact={ts['s2_exact']}")
    print(f"         S2 range: {min(s2_vals):.4f} - {max(s2_vals):.4f}")
else:
    print(f"  SKIP  opt output not found: {OPT_OUTPUT}")


# ============================================================
print("\n=== P4-4b. track_spin_state MCP dispatch ===")
# ============================================================
if OPT_OUTPUT.exists():
    mcp_ts = dispatch_tool("track_nwchem_spin_state", {
        "output_file": str(OPT_OUTPUT),
    })
    report("MCP dispatch works", mcp_ts,
           lambda r: assert_("n_steps" in r))
    report("MCP matches direct call", mcp_ts,
           lambda r: assert_(r["n_steps"] == ts["n_steps"]))


# ============================================================
print("\n=== P4-5. MCP tool count ===")
# ============================================================
from chemtools.mcp import nwchem as mcp_nwchem

tool_count = len(mcp_nwchem.tool_definitions())
report(f"tool count is 75 (was 72)", tool_count,
       lambda c: assert_(c == 75, f"tool_count={c}"))

tool_names = {t["name"] for t in mcp_nwchem.tool_definitions()}
new_tools = [
    "parse_nwchem_thermochem",
    "summarize_nwchem_electronic_structure",
    "track_nwchem_spin_state",
]
for tool in new_tools:
    report(f"tool '{tool}' registered", tool_names,
           lambda names, t=tool: assert_(t in names, f"missing: {t}"))

# compute_reaction_energy should still exist (enhanced, not new)
report("compute_reaction_energy still registered", tool_names,
       lambda names: assert_("compute_reaction_energy" in names))


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"Phase 4 Results: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
