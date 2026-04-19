#!/usr/bin/env python3
"""Smoke-test Phase 5 gap-fill tools."""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TEST_DIR = Path(__file__).resolve().parent
MOCK_INPUT = TEST_DIR / "test_mol.nw"
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
print("\n=== P5-1. basis_library_summary ===")
# ============================================================
from chemtools.mcp.nwchem import dispatch_tool

r = dispatch_tool("basis_library_summary", {})
report("returns basis_sets", r,
       lambda r: assert_("basis_sets" in r, f"keys={list(r.keys())}"))
report("has count", r,
       lambda r: assert_(r["count"] > 100, f"count={r.get('count')}"))
report("basis_sets is a list", r,
       lambda r: assert_(isinstance(r["basis_sets"], list) and len(r["basis_sets"]) > 0))
print(f"         {r['count']} basis sets in library")


# ============================================================
print("\n=== P5-2. check_nwchem_spin_charge_state ===")
# ============================================================
if OPT_OUTPUT.exists():
    r = dispatch_tool("check_nwchem_spin_charge_state", {
        "output_file": str(OPT_OUTPUT),
    })
    report("returns result dict", r,
           lambda r: assert_(isinstance(r, dict)))
    report("has state info", r,
           lambda r: assert_(any(k in r for k in ("s2_value", "spin_state", "charge", "multiplicity", "recommended_next_action")),
                              f"keys={list(r.keys())}"))
    print(f"         keys: {list(r.keys())[:8]}")
else:
    print(f"  SKIP  output not found: {OPT_OUTPUT}")


# ============================================================
print("\n=== P5-3. inspect_nwchem_geometry ===")
# ============================================================
if MOCK_INPUT.exists():
    r = dispatch_tool("inspect_nwchem_geometry", {
        "input_file": str(MOCK_INPUT),
    })
    report("returns result dict", r,
           lambda r: assert_(isinstance(r, dict)))
    print(f"         keys: {list(r.keys())[:8]}")
else:
    print(f"  SKIP  input not found: {MOCK_INPUT}")


# ============================================================
print("\n=== P5-4. parse_nwchem_tasks ===")
# ============================================================
if FREQ_OUTPUT.exists():
    r = dispatch_tool("parse_nwchem_tasks", {
        "output_file": str(FREQ_OUTPUT),
    })
    report("returns dict", r,
           lambda r: assert_(isinstance(r, dict)))
    report("has task data", r,
           lambda r: assert_("generic_tasks" in r or "tasks" in r, f"keys={list(r.keys())}"))
    n_tasks = len(r.get("tasks", r.get("generic_tasks", [])))
    print(f"         {n_tasks} tasks found, keys={list(r.keys())[:5]}")
else:
    print(f"  SKIP  output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== P5-5. parse_nwchem_trajectory ===")
# ============================================================
if OPT_OUTPUT.exists():
    r = dispatch_tool("parse_nwchem_trajectory", {
        "output_file": str(OPT_OUTPUT),
    })
    report("returns dict", r,
           lambda r: assert_(isinstance(r, dict)))
    report("has steps", r,
           lambda r: assert_("steps" in r, f"keys={list(r.keys())}"))
    if r.get("steps"):
        step0 = r["steps"][0]
        step_keys = list(step0.keys()) if isinstance(step0, dict) else []
        report("steps have data", r,
               lambda r: assert_(len(r["steps"]) > 0, "empty steps"))
        print(f"         step keys: {step_keys[:6]}")
    print(f"         {len(r.get('steps', []))} opt steps")
else:
    print(f"  SKIP  output not found: {OPT_OUTPUT}")


# ============================================================
print("\n=== P5-6. review_nwchem_input_request ===")
# ============================================================
r = dispatch_tool("review_nwchem_input_request", {
    "formula": "C6H6",
    "module": "dft",
    "functional": "b3lyp",
    "charge": 0,
    "multiplicity": 1,
    "default_basis": "cc-pVTZ",
})
report("returns dict", r,
       lambda r: assert_(isinstance(r, dict)))
report("has some review content", r,
       lambda r: assert_(len(r) > 0, "empty result"))
print(f"         keys: {list(r.keys())[:8]}")


# ============================================================
print("\n=== P5-7. review_nwchem_progress ===")
# ============================================================
if FREQ_OUTPUT.exists():
    r = dispatch_tool("review_nwchem_progress", {
        "output_file": str(FREQ_OUTPUT),
    })
    report("returns dict", r,
           lambda r: assert_(isinstance(r, dict)))
    print(f"         keys: {list(r.keys())[:8]}")
else:
    print(f"  SKIP  output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== P5-8. summarize_nwchem_output ===")
# ============================================================
if FREQ_OUTPUT.exists():
    r = dispatch_tool("summarize_nwchem_output", {
        "output_file": str(FREQ_OUTPUT),
    })
    report("returns dict", r,
           lambda r: assert_(isinstance(r, dict)))
    report("has outcome", r,
           lambda r: assert_("outcome" in r, f"keys={list(r.keys())}"))
    print(f"         outcome={r.get('outcome')}")
else:
    print(f"  SKIP  output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== P5-9. Tool count ===")
# ============================================================
from chemtools.mcp import nwchem as mcp_nwchem

tool_count = len(mcp_nwchem.tool_definitions())
report(f"tool count is 93", tool_count,
       lambda c: assert_(c == 96, f"tool_count={c}"))

tool_names = {t["name"] for t in mcp_nwchem.tool_definitions()}
phase5_tools = [
    "basis_library_summary",
    "check_nwchem_spin_charge_state",
    "inspect_nwchem_geometry",
    "parse_nwchem_tasks",
    "parse_nwchem_trajectory",
    "review_nwchem_input_request",
    "review_nwchem_progress",
    "summarize_nwchem_output",
]
for tool in phase5_tools:
    report(f"tool '{tool}' registered", tool_names,
           lambda names, t=tool: assert_(t in names, f"missing: {t}"))


print(f"\n{'='*60}")
print(f"Phase 5 Results: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
