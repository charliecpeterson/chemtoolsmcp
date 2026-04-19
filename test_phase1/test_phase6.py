#!/usr/bin/env python3
"""Smoke-test Phase 6: eval tools + create_nwchem_dft_input_from_request."""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TEST_DIR = Path(__file__).resolve().parent
MOCK_INPUT = TEST_DIR / "test_mol.nw"
TRAIN_DIR = REPO / "nwchem-test" / "train"

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
print("\n=== P6-1. evaluate_nwchem_case ===")
# ============================================================
from chemtools.mcp.nwchem import dispatch_tool

case_file = TRAIN_DIR / "h2o2_imaginary_freq" / "case.json"
if case_file.exists():
    r = dispatch_tool("evaluate_nwchem_case", {"case_path": str(case_file)})
    report("returns case_id", r,
           lambda r: assert_("case_id" in r))
    report("returns passed boolean", r,
           lambda r: assert_(isinstance(r["passed"], bool)))
    report("has checks", r,
           lambda r: assert_(r["check_count"] > 0, f"check_count={r['check_count']}"))
    report("has diagnosis", r,
           lambda r: assert_("diagnosis" in r))
    report("has workflow", r,
           lambda r: assert_("workflow" in r))
    report("case passes", r,
           lambda r: assert_(r["passed"], f"fail_count={r['fail_count']}"))
    print(f"         case={r['case_id']}, checks={r['check_count']}, passed={r['pass_count']}")
else:
    print(f"  SKIP  case file not found: {case_file}")


# ============================================================
print("\n=== P6-2. evaluate_nwchem_cases (batch) ===")
# ============================================================
if TRAIN_DIR.exists():
    r = dispatch_tool("evaluate_nwchem_cases", {"path": str(TRAIN_DIR)})
    report("returns case_count", r,
           lambda r: assert_(r["case_count"] > 0))
    report("returns passed_case_count", r,
           lambda r: assert_("passed_case_count" in r))
    report("returns failed_case_count", r,
           lambda r: assert_("failed_case_count" in r))
    report("returns results list", r,
           lambda r: assert_(len(r["results"]) == r["case_count"]))
    report("all cases pass", r,
           lambda r: assert_(r["failed_case_count"] == 0,
                              f"failed={r['failed_case_count']}"))
    print(f"         {r['case_count']} cases, {r['passed_case_count']} passed")
else:
    print(f"  SKIP  train directory not found: {TRAIN_DIR}")


# ============================================================
print("\n=== P6-3. create_nwchem_dft_input_from_request — missing requirements ===")
# ============================================================
r = dispatch_tool("create_nwchem_dft_input_from_request", {
    "formula": "H2O",
    "xc_functional": "b3lyp",
    "charge": 0,
    "multiplicity": 1,
})
report("returns ready_to_create", r,
       lambda r: assert_("ready_to_create" in r))
report("not ready without geometry", r,
       lambda r: assert_(r["ready_to_create"] is False))
report("has review", r,
       lambda r: assert_("review" in r))
report("has next_action", r,
       lambda r: assert_("next_action" in r))
print(f"         ready={r['ready_to_create']}, next_action={r.get('next_action')}")


# ============================================================
print("\n=== P6-4. create_nwchem_dft_input_from_request — successful creation ===")
# ============================================================
tmpdir = tempfile.mkdtemp(prefix="chemtools_p6_")

if MOCK_INPUT.exists():
    r = dispatch_tool("create_nwchem_dft_input_from_request", {
        "geometry_file": str(MOCK_INPUT),
        "xc_functional": "b3lyp",
        "default_basis": "cc-pVDZ",
        "charge": 0,
        "multiplicity": 1,
        "task_operations": ["energy"],
        "memory": "total 2000 mb",
        "output_dir": tmpdir,
        "write_file": True,
    })
    report("ready_to_create is True", r,
           lambda r: assert_(r["ready_to_create"] is True, f"ready={r['ready_to_create']}"))
    report("created is True", r,
           lambda r: assert_(r["created"] is True))
    report("has written_file", r,
           lambda r: assert_(r.get("written_file"), "no written_file"))
    if r.get("written_file"):
        report("written file exists", r,
               lambda r: assert_(Path(r["written_file"]).exists()))
        text = Path(r["written_file"]).read_text()
        report("input has dft block", text,
               lambda t: assert_("dft" in t.lower()))
        report("input has basis", text,
               lambda t: assert_("basis" in t.lower()))
        report("input has task", text,
               lambda t: assert_("task" in t.lower()))
    report("has next_action in result", r,
           lambda r: assert_(r.get("next_action") == "input_created"))
    print(f"         created={r.get('created')}, file={r.get('written_file')}")
else:
    print(f"  SKIP  mock input not found: {MOCK_INPUT}")


# ============================================================
print("\n=== P6-5. create_nwchem_dft_input_from_request — with DFT settings ===")
# ============================================================
if MOCK_INPUT.exists():
    r = dispatch_tool("create_nwchem_dft_input_from_request", {
        "geometry_file": str(MOCK_INPUT),
        "xc_functional": "m06",
        "default_basis": "6-31G*",
        "charge": 1,
        "multiplicity": 2,
        "task_operations": ["optimize"],
        "dft_settings": ["grid fine", "convergence energy 1e-8"],
        "output_dir": tmpdir,
        "write_file": True,
    })
    report("created with m06", r,
           lambda r: assert_(r.get("created") is True))
    if r.get("written_file"):
        text = Path(r["written_file"]).read_text()
        report("has m06 functional", text,
               lambda t: assert_("m06" in t.lower()))
        report("has grid fine", text,
               lambda t: assert_("grid fine" in t.lower()))
        report("has charge 1", text,
               lambda t: assert_("charge 1" in t))
else:
    print(f"  SKIP  mock input not found")


# ============================================================
print("\n=== P6-6. Tool count ===")
# ============================================================
from chemtools.mcp import nwchem as mcp_nwchem

tool_count = len(mcp_nwchem.tool_definitions())
report(f"tool count is 96", tool_count,
       lambda c: assert_(c == 96, f"tool_count={c}"))

tool_names = {t["name"] for t in mcp_nwchem.tool_definitions()}
phase6_tools = [
    "evaluate_nwchem_case",
    "evaluate_nwchem_cases",
    "create_nwchem_dft_input_from_request",
]
for tool in phase6_tools:
    report(f"tool '{tool}' registered", tool_names,
           lambda names, t=tool: assert_(t in names, f"missing: {t}"))


# ============================================================
# Cleanup
# ============================================================
shutil.rmtree(tmpdir, ignore_errors=True)

print(f"\n{'='*60}")
print(f"Phase 6 Results: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
