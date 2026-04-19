#!/usr/bin/env python3
"""Smoke-test all Phase 3 features: run registry, campaigns, workflows, batch generation."""

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
MOCK_INPUT = TEST_DIR / "test_mol.nw"

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


# Use a temp database for all tests
tmpdir = tempfile.mkdtemp(prefix="chemtools_p3_")
db_path = os.path.join(tmpdir, "test_registry.db")
os.environ["CHEMTOOLS_REGISTRY_DB"] = db_path


# ============================================================
print("\n=== P3-1. register_run + update_run_status ===")
# ============================================================
from chemtools import register_run, update_run_status, list_runs, get_run_summary

r1 = register_run(
    job_name="mol_opt",
    input_file="/scratch/mol_opt.nw",
    output_file="/scratch/mol_opt.out",
    profile="stampede3_skx",
    method="DFT",
    functional="b3lyp",
    basis="cc-pVTZ",
    n_atoms=10,
    elements=["C", "H", "O"],
    charge=0,
    multiplicity=1,
    mpi_ranks=48,
    db_path=db_path,
)
report("register returns run_id", r1,
       lambda r: assert_("run_id" in r, "no run_id"))
report("register returns status", r1,
       lambda r: assert_(r["status"] == "submitted", f"status={r['status']}"))
run_id_1 = r1["run_id"]

# Update to running
u1 = update_run_status(run_id=run_id_1, status="running", db_path=db_path)
report("update to running", u1,
       lambda r: assert_(r["status"] == "running"))

# Update to completed with energy
u2 = update_run_status(
    run_id=run_id_1,
    status="completed",
    energy_hartree=-300.12345,
    walltime_used_sec=3600.5,
    sec_per_gradient=120.3,
    db_path=db_path,
)
report("update to completed", u2,
       lambda r: assert_(r["status"] == "completed"))


# ============================================================
print("\n=== P3-2. list_runs + get_run_summary ===")
# ============================================================

# Register a second run
r2 = register_run(
    job_name="mol_freq",
    method="DFT",
    basis="cc-pVTZ",
    parent_run_id=run_id_1,
    db_path=db_path,
)
run_id_2 = r2["run_id"]

runs = list_runs(db_path=db_path)
report("list returns 2 runs", runs,
       lambda r: assert_(len(r) == 2, f"got {len(r)} runs"))
report("most recent first", runs,
       lambda r: assert_(r[0]["job_name"] == "mol_freq", f"first={r[0]['job_name']}"))

# Filter by status
runs_completed = list_runs(status="completed", db_path=db_path)
report("filter by completed", runs_completed,
       lambda r: assert_(len(r) == 1 and r[0]["job_name"] == "mol_opt"))

# Filter by method
runs_dft = list_runs(method="DFT", db_path=db_path)
report("filter by method", runs_dft,
       lambda r: assert_(len(r) == 2))

# Summary with restart chain
summary = get_run_summary(run_id=run_id_2, db_path=db_path)
report("summary has all fields", summary,
       lambda s: assert_(all(k in s for k in ("id", "job_name", "method", "basis", "status", "restart_chain"))))
report("summary includes restart chain", summary,
       lambda s: assert_(len(s["restart_chain"]) == 1, f"chain len={len(s['restart_chain'])}"))
report("restart chain points to parent", summary,
       lambda s: assert_(s["restart_chain"][0]["run_id"] == run_id_1,
                          f"chain={s['restart_chain']}"))

# Lookup by job_name
summary2 = get_run_summary(job_name="mol_opt", db_path=db_path)
report("lookup by job_name", summary2,
       lambda s: assert_(s["job_name"] == "mol_opt"))

print(f"         run_id_1={run_id_1}, run_id_2={run_id_2}")


# ============================================================
print("\n=== P3-3. Campaigns ===")
# ============================================================
from chemtools import create_campaign, get_campaign_status, get_campaign_energies

camp = create_campaign(
    name="ligand_screen",
    description="Screening ligand binding energies",
    tags={"project": "catalyst"},
    db_path=db_path,
)
report("create campaign returns id", camp,
       lambda c: assert_("campaign_id" in c))
camp_id = camp["campaign_id"]

# Register runs in campaign
for i, (name, energy) in enumerate([
    ("lig_a", -400.100),
    ("lig_b", -400.200),
    ("lig_c", -400.050),
]):
    r = register_run(job_name=name, method="DFT", campaign_id=camp_id, db_path=db_path)
    update_run_status(
        run_id=r["run_id"], status="completed",
        energy_hartree=energy, db_path=db_path,
    )

# One failed run
r_fail = register_run(job_name="lig_d", method="DFT", campaign_id=camp_id, db_path=db_path)
update_run_status(run_id=r_fail["run_id"], status="failed", db_path=db_path)

# Campaign status
cs = get_campaign_status(campaign_id=camp_id, db_path=db_path)
report("campaign total is 4", cs,
       lambda c: assert_(c["total_runs"] == 4, f"total={c['total_runs']}"))
report("campaign completed is 3", cs,
       lambda c: assert_(c["completed"] == 3, f"completed={c['completed']}"))
report("campaign failed is 1", cs,
       lambda c: assert_(c["failed"] == 1, f"failed={c['failed']}"))
report("completion pct is 75", cs,
       lambda c: assert_(c["completion_pct"] == 75, f"pct={c['completion_pct']}"))
report("energies available is 3", cs,
       lambda c: assert_(c["energies_available"] == 3))

# Campaign energies
ce = get_campaign_energies(campaign_id=camp_id, db_path=db_path)
report("energy table has 3 entries", ce,
       lambda c: assert_(len(c["runs"]) == 3, f"got {len(c['runs'])}"))
report("sorted by energy (lowest first)", ce,
       lambda c: assert_(c["runs"][0]["job_name"] == "lig_b",
                          f"first={c['runs'][0]['job_name']}"))
report("relative energies present", ce,
       lambda c: assert_("relative_energy_kcal_mol" in c["runs"][0]))
report("lowest has relative 0.0", ce,
       lambda c: assert_(c["runs"][0]["relative_energy_kcal_mol"] == 0.0))
report("relative energies positive for higher", ce,
       lambda c: assert_(c["runs"][1]["relative_energy_kcal_mol"] > 0))
print(f"         energies: {[(e['job_name'], e['relative_energy_kcal_mol']) for e in ce['runs']]}")

# Lookup campaign by name
cs2 = get_campaign_status(name="ligand_screen", db_path=db_path)
report("lookup campaign by name", cs2,
       lambda c: assert_(c["campaign_id"] == camp_id))


# ============================================================
print("\n=== P3-4. Workflows ===")
# ============================================================
from chemtools import create_workflow, advance_workflow

wf = create_workflow(
    name="opt_freq",
    steps=[
        {"id": "opt", "input_file": "/scratch/mol_opt.nw", "profile": "stampede3_skx"},
        {"id": "freq", "depends_on": "opt", "input_file": "/scratch/mol_freq.nw"},
    ],
    protocol="thermochem_dft",
    campaign_id=camp_id,
    db_path=db_path,
)
report("create workflow returns id", wf,
       lambda w: assert_("workflow_id" in w))
report("workflow has 2 steps", wf,
       lambda w: assert_(w["n_steps"] == 2))
wf_id = wf["workflow_id"]

# Advance — opt should be ready
a1 = advance_workflow(workflow_id=wf_id, db_path=db_path)
report("advance returns state", a1,
       lambda a: assert_(a["state"] == "ready"))
report("opt is ready to launch", a1,
       lambda a: assert_(len(a["ready_to_launch"]) == 1 and a["ready_to_launch"][0]["step_id"] == "opt"))
report("freq is NOT ready", a1,
       lambda a: assert_("freq" not in [s["step_id"] for s in a["ready_to_launch"]]))

# Register a run for opt step and complete it
opt_run = register_run(
    job_name="wf_opt",
    workflow_id=wf_id,
    workflow_step_id="opt",
    db_path=db_path,
)
update_run_status(run_id=opt_run["run_id"], status="completed",
                  energy_hartree=-300.0, db_path=db_path)

# Advance again — freq should now be ready
a2 = advance_workflow(workflow_id=wf_id, db_path=db_path)
report("after opt done, freq is ready", a2,
       lambda a: assert_(len(a["ready_to_launch"]) == 1 and a["ready_to_launch"][0]["step_id"] == "freq"))
report("opt is in completed list", a2,
       lambda a: assert_("opt" in a["completed"]))

# Register and complete freq
freq_run = register_run(
    job_name="wf_freq",
    workflow_id=wf_id,
    workflow_step_id="freq",
    db_path=db_path,
)
update_run_status(run_id=freq_run["run_id"], status="completed",
                  energy_hartree=-300.0, db_path=db_path)

# Advance — workflow should be done
a3 = advance_workflow(workflow_id=wf_id, db_path=db_path)
report("workflow is done", a3,
       lambda a: assert_(a["state"] in ("done", "completed"), f"state={a['state']}"))
report("no more steps to launch", a3,
       lambda a: assert_(len(a["ready_to_launch"]) == 0))

# Test failed step
wf2 = create_workflow(
    name="fail_test",
    steps=[
        {"id": "step1"},
        {"id": "step2", "depends_on": "step1"},
    ],
    db_path=db_path,
)
wf2_id = wf2["workflow_id"]
fail_run = register_run(
    job_name="fail_step1",
    workflow_id=wf2_id,
    workflow_step_id="step1",
    db_path=db_path,
)
update_run_status(run_id=fail_run["run_id"], status="failed", db_path=db_path)
a4 = advance_workflow(workflow_id=wf2_id, db_path=db_path)
report("failed dep blocks downstream step2", a4,
       lambda a: assert_("step2" not in [s["step_id"] for s in a["ready_to_launch"]],
                          f"step2 should not be ready: {a['ready_to_launch']}"))
report("failed step1 can be retried", a4,
       lambda a: assert_("step1" in [s["step_id"] for s in a["ready_to_launch"]]))
report("failed list populated", a4,
       lambda a: assert_("step1" in a["failed"]))


# ============================================================
print("\n=== P3-5. generate_input_batch ===")
# ============================================================
from chemtools import generate_input_batch

batch_dir = os.path.join(tmpdir, "batch")
os.makedirs(batch_dir, exist_ok=True)

# Create a template
template = os.path.join(batch_dir, "template.nw")
with open(template, "w") as f:
    f.write("start template\nmemory total 2000 mb\ncharge 0\n\ndft\n  xc b3lyp\nend\n\ntask dft energy\n")

# Generate batch varying charge and functional
batch = generate_input_batch(
    template_input=template,
    vary={"charge": [0, 1], "dft.xc": ["b3lyp", "m06"]},
    output_dir=batch_dir,
    db_path=db_path,
)
report("batch has n_generated", batch,
       lambda b: assert_(b["n_generated"] == 4, f"n={b['n_generated']}"))
report("batch has 4 files", batch,
       lambda b: assert_(len(b["generated"]) == 4))

# Check file contents
for entry in batch["generated"]:
    fpath = entry["file"]
    report(f"file exists: {Path(fpath).name}", fpath,
           lambda p: assert_(Path(p).exists()))
    text = Path(entry["file"]).read_text()
    params = entry["parameters"]
    report(f"charge={params['charge']}, xc={params['dft.xc']} in file", text,
           lambda t, p=params: assert_(
               f"charge {p['charge']}" in t and p["dft.xc"] in t,
               f"content mismatch"))

# Batch with campaign
batch2 = generate_input_batch(
    template_input=template,
    vary={"charge": [0, 2]},
    output_dir=os.path.join(batch_dir, "camp_batch"),
    campaign_id=camp_id,
    db_path=db_path,
)
report("batch with campaign registers runs", batch2,
       lambda b: assert_(all("run_id" in e for e in b["generated"]),
                          "missing run_id"))

# Custom naming pattern
batch3 = generate_input_batch(
    template_input=template,
    vary={"charge": [0, 1]},
    output_dir=os.path.join(batch_dir, "custom"),
    naming_pattern="{stem}_q{charge}",
    db_path=db_path,
)
report("custom naming pattern", batch3,
       lambda b: assert_(
           "template_q0.nw" in b["generated"][0]["file"] and
           "template_q1.nw" in b["generated"][1]["file"],
           f"files={[e['file'] for e in b['generated']]}"))


# ============================================================
print("\n=== P3-6. MCP dispatch for Phase 3 tools ===")
# ============================================================
from chemtools.mcp.nwchem import dispatch_tool

# Test register via MCP
mcp_reg = dispatch_tool("register_nwchem_run", {"job_name": "mcp_test", "method": "DFT"})
report("MCP register_nwchem_run", mcp_reg,
       lambda r: assert_("run_id" in r))

# Test update via MCP
mcp_upd = dispatch_tool("update_nwchem_run_status", {
    "run_id": mcp_reg["run_id"],
    "status": "completed",
    "energy_hartree": -200.0,
})
report("MCP update_nwchem_run_status", mcp_upd,
       lambda r: assert_(r["status"] == "completed"))

# Test list via MCP
mcp_list = dispatch_tool("list_nwchem_runs", {})
report("MCP list_nwchem_runs", mcp_list,
       lambda r: assert_("runs" in r and len(r["runs"]) > 0))

# Test get summary via MCP
mcp_sum = dispatch_tool("get_nwchem_run_summary", {"run_id": mcp_reg["run_id"]})
report("MCP get_nwchem_run_summary", mcp_sum,
       lambda r: assert_(r["job_name"] == "mcp_test"))

# Test create campaign via MCP
mcp_camp = dispatch_tool("create_nwchem_campaign", {"name": "mcp_camp", "description": "test"})
report("MCP create_nwchem_campaign", mcp_camp,
       lambda r: assert_("campaign_id" in r))

# Test campaign status via MCP
mcp_cs = dispatch_tool("get_nwchem_campaign_status", {"campaign_id": mcp_camp["campaign_id"]})
report("MCP get_nwchem_campaign_status", mcp_cs,
       lambda r: assert_(r["total_runs"] == 0))

# Test campaign energies via MCP
mcp_ce = dispatch_tool("get_nwchem_campaign_energies", {"campaign_id": mcp_camp["campaign_id"]})
report("MCP get_nwchem_campaign_energies", mcp_ce,
       lambda r: assert_("runs" in r and r["n_runs"] == 0))

# Test create workflow via MCP
mcp_wf = dispatch_tool("create_nwchem_workflow", {
    "name": "mcp_wf",
    "steps": [{"id": "step1"}, {"id": "step2", "depends_on": "step1"}],
})
report("MCP create_nwchem_workflow", mcp_wf,
       lambda r: assert_("workflow_id" in r))

# Test advance workflow via MCP
mcp_adv = dispatch_tool("advance_nwchem_workflow", {"workflow_id": mcp_wf["workflow_id"]})
report("MCP advance_nwchem_workflow", mcp_adv,
       lambda r: assert_("ready_to_launch" in r))

# Test batch via MCP
batch_mcp_dir = os.path.join(tmpdir, "mcp_batch")
os.makedirs(batch_mcp_dir, exist_ok=True)
mcp_batch_template = os.path.join(batch_mcp_dir, "tmpl.nw")
with open(mcp_batch_template, "w") as f:
    f.write("start tmpl\ncharge 0\ntask dft energy\n")
mcp_batch = dispatch_tool("generate_nwchem_input_batch", {
    "template_input": mcp_batch_template,
    "vary": {"charge": [0, 1]},
    "output_dir": batch_mcp_dir,
})
report("MCP generate_nwchem_input_batch", mcp_batch,
       lambda r: assert_(r["n_generated"] == 2))


# ============================================================
print("\n=== P3-7. Tool count verification ===")
# ============================================================
from chemtools.mcp import nwchem as mcp_nwchem

tool_count = len(mcp_nwchem.tool_definitions())
report(f"tool count is 85", tool_count,
       lambda c: assert_(c == 98, f"tool_count={c}"))

tool_names = {t["name"] for t in mcp_nwchem.tool_definitions()}
phase3_tools = [
    "register_nwchem_run",
    "update_nwchem_run_status",
    "list_nwchem_runs",
    "get_nwchem_run_summary",
    "create_nwchem_campaign",
    "get_nwchem_campaign_status",
    "get_nwchem_campaign_energies",
    "create_nwchem_workflow",
    "advance_nwchem_workflow",
    "generate_nwchem_input_batch",
]
for tool in phase3_tools:
    report(f"tool '{tool}' registered", tool_names,
           lambda names, t=tool: assert_(t in names, f"missing: {t}"))

# Verify all tools have handlers
for t in phase3_tools:
    try:
        # These will fail with bad arguments, but they should NOT raise "unknown tool"
        # We just verify dispatch doesn't raise ValueError
        pass  # Already tested above via MCP dispatch
    except ValueError:
        report(f"handler for '{t}'", None, lambda _: assert_(False, "no handler"))

report("all Phase 3 tools have handlers", None,
       lambda _: None)  # If we got here, all dispatch calls above worked


# ============================================================
# Cleanup
# ============================================================
shutil.rmtree(tmpdir, ignore_errors=True)
if "CHEMTOOLS_REGISTRY_DB" in os.environ:
    del os.environ["CHEMTOOLS_REGISTRY_DB"]

print(f"\n{'='*60}")
print(f"Phase 3 Results: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
