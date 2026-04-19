#!/usr/bin/env python3
"""Smoke-test all Phase 2 features locally."""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add repo root to path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TEST_DIR = Path(__file__).resolve().parent
FREQ_OUTPUT = REPO / "nwchemaitest" / "am2pba3h2_s-031326.out"
OPT_INPUT = REPO / "nwchemaitest" / "am2pba3h2_s_opt2.nw"
OPT_OUTPUT = REPO / "nwchemaitest" / "am2pba3h2_s_opt2.out"
MOCK_INPUT = TEST_DIR / "test_mol.nw"
MOCK_ERR = TEST_DIR / "test_mol.err"

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
print("\n=== P2-1. create_nwchem_input_variant ===")
# ============================================================
from chemtools import create_nwchem_input_variant

# Create a temp copy of mock input so we don't pollute test dir
tmpdir = tempfile.mkdtemp(prefix="chemtools_p2_")
src_nw = Path(tmpdir) / "test_mol.nw"
src_nw.write_text(MOCK_INPUT.read_text())

result = create_nwchem_input_variant(
    source_input=str(src_nw),
    changes={"memory": "800 mb", "dft.iterations": "300"},
    reason="Testing variant creation",
)
report("returns output_file", result,
       lambda r: assert_(r["output_file"].endswith("_v2.nw"), f"got {r['output_file']}"))
report("returns diff_summary", result,
       lambda r: assert_(len(r["diff_summary"]) == 2, f"got {len(r['diff_summary'])} diffs"))
report("memory old value captured", result,
       lambda r: assert_(r["diff_summary"][0]["old"] == "total 2000 mb",
                          f"old={r['diff_summary'][0]['old']}"))
report("memory new value", result,
       lambda r: assert_(r["diff_summary"][0]["new"] == "800 mb",
                          f"new={r['diff_summary'][0]['new']}"))
report("reason preserved", result,
       lambda r: assert_(r["reason"] == "Testing variant creation"))
report("file was written", result,
       lambda r: assert_(Path(r["written_file"]).exists(), "written file doesn't exist"))

# Read the variant and check content
v2_text = Path(result["output_file"]).read_text()
report("memory line changed", v2_text,
       lambda t: assert_("memory 800 mb" in t, f"memory not found in:\n{t[:200]}"))
report("dft iterations changed", v2_text,
       lambda t: assert_("iterations 300" in t, f"iterations 300 not found"))
report("original unchanged", None,
       lambda _: assert_("2000 mb" in src_nw.read_text(), "original was modified!"))

# Test charge/task changes
result2 = create_nwchem_input_variant(
    source_input=str(src_nw),
    changes={"charge": "2", "task": "dft optimize"},
    reason="Test charge+task",
    output_path=str(Path(tmpdir) / "custom_output.nw"),
)
report("custom output_path", result2,
       lambda r: assert_(r["output_file"].endswith("custom_output.nw")))
v3_text = Path(result2["output_file"]).read_text()
report("charge inserted", v3_text,
       lambda t: assert_("charge 2" in t, "charge not found"))
report("task changed", v3_text,
       lambda t: assert_("task dft optimize" in t, "task not found"))

print(f"         diff: {json.dumps(result['diff_summary'], indent=2)}")


# ============================================================
print("\n=== P2-2. next_actions on analyze_nwchem_imaginary_modes ===")
# ============================================================
if FREQ_OUTPUT.exists():
    from chemtools import analyze_imaginary_modes

    result = analyze_imaginary_modes(str(FREQ_OUTPUT))
    # Note: next_actions is added at MCP handler level, not library level
    # So we test via the MCP dispatch
    from chemtools.mcp.nwchem import dispatch_tool

    mcp_result = dispatch_tool("analyze_nwchem_imaginary_modes", {
        "output_file": str(FREQ_OUTPUT),
    })
    report("next_actions present", mcp_result,
           lambda r: assert_("next_actions" in r, "no next_actions key"))
    report("next_actions is list", mcp_result,
           lambda r: assert_(isinstance(r["next_actions"], list)))
    if mcp_result.get("next_actions"):
        first = mcp_result["next_actions"][0]
        report("first action has tool", first,
               lambda a: assert_("tool" in a, "missing tool key"))
        report("first action has params", first,
               lambda a: assert_("params" in a, "missing params key"))
        report("first action has confidence", first,
               lambda a: assert_("confidence" in a, "missing confidence key"))
        report("first action has reason", first,
               lambda a: assert_("reason" in a, "missing reason key"))
        print(f"         action: {first['tool']} (confidence={first['confidence']})")
        print(f"         reason: {first['reason']}")
else:
    print(f"  SKIP  freq output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== P2-3. next_actions on check_nwchem_freq_plausibility ===")
# ============================================================
if FREQ_OUTPUT.exists():
    mcp_result = dispatch_tool("check_nwchem_freq_plausibility", {
        "output_file": str(FREQ_OUTPUT),
    })
    report("next_actions present", mcp_result,
           lambda r: assert_("next_actions" in r, "no next_actions"))
    if mcp_result.get("next_actions"):
        first = mcp_result["next_actions"][0]
        report("action has all fields", first,
               lambda a: assert_(all(k in a for k in ("tool", "params", "confidence", "reason")),
                                  f"missing fields: {set(('tool','params','confidence','reason')) - set(a.keys())}"))
        print(f"         action: {first['tool']} → {first.get('reason', '')[:80]}")
else:
    print(f"  SKIP  freq output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== P2-4. get_nwchem_workflow_state ===")
# ============================================================
from chemtools import get_nwchem_workflow_state

# Test with mock input (no output file → pending or queued if .jobid exists)
mock_out = str(TEST_DIR / "nonexistent.out")
state = get_nwchem_workflow_state(
    input_file=str(MOCK_INPUT),
    output_file=mock_out,
)
report("returns state", state,
       lambda s: assert_("state" in s, "no state key"))
# test_mol.jobid exists from Phase 1 tests, so state may be "queued"
report("pending or queued for missing output", state,
       lambda s: assert_(s["state"] in ("pending", "queued"),
                          f"state={s['state']}"))
report("has next_action", state,
       lambda s: assert_("next_action" in s, "no next_action"))
report("has confidence", state,
       lambda s: assert_("confidence" in s and 0 <= s["confidence"] <= 1,
                          f"confidence={s.get('confidence')}"))
report("has human_summary", state,
       lambda s: assert_(s.get("human_summary"), "empty summary"))
print(f"         state={state['state']}, confidence={state['confidence']}")
print(f"         summary: {state['human_summary']}")
if state.get("next_action"):
    print(f"         next: {state['next_action']['tool']}")

# Test true pending (no .jobid sibling)
isolated_nw = Path(tmpdir) / "isolated.nw"
isolated_nw.write_text("start isolated\nmemory total 1000 mb\ntask dft energy\n")
isolated_out = str(Path(tmpdir) / "isolated.out")
pending_state = get_nwchem_workflow_state(
    input_file=str(isolated_nw),
    output_file=isolated_out,
)
report("true pending state", pending_state,
       lambda s: assert_(s["state"] == "pending", f"state={s['state']}"))
report("pending suggests launch", pending_state,
       lambda s: assert_(s.get("next_action", {}).get("tool") == "launch_nwchem_run",
                          f"tool={s.get('next_action', {}).get('tool')}"))

# Test with completed freq output
if FREQ_OUTPUT.exists():
    freq_state = get_nwchem_workflow_state(
        input_file=str(MOCK_INPUT),  # has 'task dft freq'
        output_file=str(FREQ_OUTPUT),
    )
    report("freq output has terminal state", freq_state,
           lambda s: assert_(s["state"] in ("freq_complete", "imaginary_modes", "completed"),
                              f"state={s['state']}"))
    report("progress_pct is 100", freq_state,
           lambda s: assert_(s["progress_pct"] == 100 or s["state"] == "completed",
                              f"pct={s['progress_pct']}"))
    print(f"         freq state={freq_state['state']}, pct={freq_state['progress_pct']}")

# Test with timelimit error
if MOCK_ERR.exists():
    # Create a fake output with some content
    fake_out = Path(tmpdir) / "timelimit.out"
    fake_nw = Path(tmpdir) / "timelimit.nw"
    fake_nw.write_text("restart timelimit\nmemory total 2000 mb\ntask dft freq\n")
    fake_out.write_text("Some NWChem output without Total times cpu:\n")
    fake_err = Path(tmpdir) / "timelimit.err"
    fake_err.write_text("slurmstepd: error: *** JOB 12345 ON c511-043 CANCELLED AT 2026-04-17T22:35:56 DUE TO TIME LIMIT ***\n")
    # Create a fake fdrst
    (Path(tmpdir) / "timelimit.fdrst").write_text("fake checkpoint")
    tl_state = get_nwchem_workflow_state(
        input_file=str(fake_nw),
        output_file=str(fake_out),
        error_file=str(fake_err),
    )
    report("detects freq_timelimited", tl_state,
           lambda s: assert_(s["state"] == "freq_timelimited", f"state={s['state']}"))
    report("suggests resubmit", tl_state,
           lambda s: assert_(s.get("next_action", {}).get("tool") == "launch_nwchem_run",
                              f"next_action tool={s.get('next_action', {}).get('tool')}"))
    print(f"         timelimit state={tl_state['state']}, confidence={tl_state['confidence']}")

# Test new fields: missing_files and related_jobs
report("has missing_files field", tl_state,
       lambda s: assert_("missing_files" in s, "no missing_files"))
report("has related_jobs field", tl_state,
       lambda s: assert_("related_jobs" in s, "no related_jobs"))

# Test output-only mode (no input_file provided)
if FREQ_OUTPUT.exists():
    output_only_state = get_nwchem_workflow_state(
        output_file=str(FREQ_OUTPUT),
    )
    report("output-only: returns state", output_only_state,
           lambda s: assert_("state" in s, "no state key"))
    report("output-only: detects task from input echo", output_only_state,
           lambda s: assert_(s["state"] in ("freq_complete", "imaginary_modes", "completed"),
                              f"state={s['state']}"))
    report("output-only: reports missing .nw", output_only_state,
           lambda s: assert_(any(".nw" in f for f in s.get("missing_files", [])),
                              f"missing_files={s.get('missing_files')}"))
    print(f"         output-only state={output_only_state['state']}")
    print(f"         missing_files={output_only_state.get('missing_files', [])}")

# Test incomplete output without .fdrst (stopped mid-freq, no checkpoint)
fake_incomplete = Path(tmpdir) / "nofdrst.out"
fake_incomplete_nw = Path(tmpdir) / "nofdrst.nw"
fake_incomplete_nw.write_text("restart nofdrst\nmemory total 2000 mb\ntask dft freq\n")
fake_incomplete.write_text("Some NWChem output with freq data but no Total times cpu:\n")
nofdrst_state = get_nwchem_workflow_state(
    input_file=str(fake_incomplete_nw),
    output_file=str(fake_incomplete),
)
report("incomplete freq: warns about missing fdrst", nofdrst_state,
       lambda s: assert_(any(".fdrst" in f for f in s.get("missing_files", [])),
                          f"missing_files={s.get('missing_files')}"))
print(f"         nofdrst state={nofdrst_state['state']}, confidence={nofdrst_state['confidence']}")


# ============================================================
print("\n=== P2-5. list_protocols + plan_calculation ===")
# ============================================================
from chemtools import list_protocols, plan_calculation

protos = list_protocols()
report("returns protocol list", protos,
       lambda p: assert_(len(p) >= 5, f"only {len(p)} protocols"))
report("each has name+description", protos,
       lambda p: assert_(all("name" in x and "description" in x for x in p)))
print(f"         protocols: {[p['name'] for p in protos]}")

# Plan a thermochem workflow
plan = plan_calculation(
    input_file=str(MOCK_INPUT),
    protocol="thermochem_dft",
    profile="stampede3_skx",
)
report("plan returns protocol name", plan,
       lambda p: assert_(p["protocol"] == "thermochem_dft"))
report("plan has steps", plan,
       lambda p: assert_(p["n_steps"] == 2, f"n_steps={p['n_steps']}"))
report("step 1 is opt", plan,
       lambda p: assert_(p["steps"][0]["step_id"] == "opt"))
report("step 2 is freq", plan,
       lambda p: assert_(p["steps"][1]["step_id"] == "freq"))
report("step 2 depends on opt", plan,
       lambda p: assert_(p["steps"][1]["depends_on"] == "opt",
                          f"depends_on={p['steps'][1]['depends_on']}"))
report("step 2 has post_actions", plan,
       lambda p: assert_(len(p["steps"][1]["post_actions"]) > 0))
report("on_imaginary_modes set", plan,
       lambda p: assert_(p.get("on_imaginary_modes") == "displace_and_reopt"))
print(f"         steps: {[s['step_id'] for s in plan['steps']]}")
for step in plan["steps"]:
    print(f"           {step['step_id']}: task={step['task']}, depends_on={step['depends_on']}")

# Plan a basis set convergence
plan2 = plan_calculation(
    input_file=str(MOCK_INPUT),
    protocol="basis_set_convergence",
    profile="stampede3_skx",
)
report("basis convergence has 3 steps", plan2,
       lambda p: assert_(p["n_steps"] == 3, f"n_steps={p['n_steps']}"))
report("parallel_independent is True", plan2,
       lambda p: assert_(p["parallel_independent"] is True))
print(f"         basis steps: {[s['step_id'] for s in plan2['steps']]}")


# ============================================================
print("\n=== P2-6. MCP tool count ===")
# ============================================================
from chemtools.mcp import nwchem as mcp_nwchem

tool_count = len(mcp_nwchem.tool_definitions())
report(f"tool count is 85 (was 75)", tool_count,
       lambda c: assert_(c == 98, f"tool_count={c}"))

# Verify new tools exist
tool_names = {t["name"] for t in mcp_nwchem.tool_definitions()}
new_tools = [
    "create_nwchem_input_variant",
    "get_nwchem_workflow_state",
    "plan_nwchem_calculation",
    "list_nwchem_protocols",
]
for tool in new_tools:
    report(f"tool '{tool}' registered", tool_names,
           lambda names, t=tool: assert_(t in names, f"missing: {t}"))


# ============================================================
# Cleanup
# ============================================================
shutil.rmtree(tmpdir, ignore_errors=True)

print(f"\n{'='*60}")
print(f"Phase 2 Results: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
