#!/usr/bin/env python3
"""Smoke-test all Phase 1 features locally."""

import json
import os
import sys
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
MOCK_JOBID = TEST_DIR / "test_mol.jobid"

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
print("\n=== 1. analyze_imaginary_modes compact default ===")
# ============================================================
if FREQ_OUTPUT.exists():
    from chemtools import analyze_imaginary_modes

    # Compact (default)
    compact = analyze_imaginary_modes(str(FREQ_OUTPUT))
    report("compact returns result", compact,
           lambda r: assert_(r["significant_imaginary_mode_count"] > 0,
                             f"expected imaginary modes, got {r['significant_imaginary_mode_count']}"))
    report("compact has detail=compact", compact,
           lambda r: assert_(r.get("detail") == "compact", f"detail={r.get('detail')}"))
    report("compact strips displacements", compact,
           lambda r: assert_(
               all("displacements_cartesian" not in m for m in r["modes"]),
               "found displacements_cartesian in compact mode"))

    compact_size = len(json.dumps(compact))

    # Full
    full = analyze_imaginary_modes(str(FREQ_OUTPUT), detail="full")
    report("full has detail=full", full,
           lambda r: assert_(r.get("detail") == "full", f"detail={r.get('detail')}"))
    report("full includes displacements", full,
           lambda r: assert_(
               any(m.get("displacements_cartesian") for m in r["modes"]),
               "no displacements_cartesian in full mode"))

    full_size = len(json.dumps(full))
    report(f"compact ({compact_size:,} chars) << full ({full_size:,} chars)",
           (compact_size, full_size),
           lambda sizes: assert_(sizes[0] < sizes[1] * 0.5,
                                  f"compact not much smaller: {sizes[0]} vs {sizes[1]}"))
else:
    print(f"  SKIP  freq output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== 2. parse_nwchem_freq_progress ===")
# ============================================================
if FREQ_OUTPUT.exists():
    from chemtools import parse_freq_progress

    progress = parse_freq_progress(str(FREQ_OUTPUT))
    print(f"         atoms={progress['n_atoms']}, total={progress['n_total_displacements']}, "
          f"done={progress['n_done_cumulative']}, pct={progress['pct_complete']}%")
    report("returns n_atoms", progress,
           lambda r: assert_(r["n_atoms"] and r["n_atoms"] > 0, f"n_atoms={r['n_atoms']}"))
    report("returns total displacements", progress,
           lambda r: assert_(r["n_total_displacements"] and r["n_total_displacements"] > 0,
                              f"total={r['n_total_displacements']}"))
    report("pct_complete is reasonable", progress,
           lambda r: assert_(r["pct_complete"] is not None and 0 <= r["pct_complete"] <= 100,
                              f"pct={r['pct_complete']}"))
else:
    print(f"  SKIP  freq output not found: {FREQ_OUTPUT}")


# ============================================================
print("\n=== 3. prepare_nwchem_freq_restart ===")
# ============================================================
from chemtools import prepare_freq_restart

# With mock files (has restart keyword, fdrst, db)
restart = prepare_freq_restart(
    input_file=str(MOCK_INPUT),
    output_file=str(TEST_DIR / "test_mol.out"),
    profile="stampede3_spr",
)
report("returns result", restart, lambda r: assert_(isinstance(r, dict)))
report("ready_to_restart is bool", restart,
       lambda r: assert_(isinstance(r["ready_to_restart"], bool)))
report("has restart keyword", restart,
       lambda r: assert_(r["has_restart_keyword"] is True, "should detect 'restart' keyword"))
report("fdrst exists", restart,
       lambda r: assert_(r["fdrst"]["exists"] is True, "fdrst should exist"))
report("db exists", restart,
       lambda r: assert_(r["db_exists"] is True, "db should exist"))
print(f"         ready={restart['ready_to_restart']}, issues={restart['issues']}")


# ============================================================
print("\n=== 4. archive_previous_outputs ===")
# ============================================================
from chemtools.runner import archive_previous_outputs

# Create some dummy files to archive
archive_dir = TEST_DIR / "archive_test"
archive_dir.mkdir(exist_ok=True)
(archive_dir / "myjob.out").write_text("old output")
(archive_dir / "myjob.err").write_text("old errors")
(archive_dir / "myjob.job").write_text("old job script")

archived = archive_previous_outputs(str(archive_dir), "myjob")
report("archives 3 files", archived,
       lambda r: assert_(len(r) == 3, f"archived {len(r)} files, expected 3"))
report("originals removed", None,
       lambda _: assert_(
           not (archive_dir / "myjob.out").exists(),
           "myjob.out should have been renamed"))
report("archives exist", archived,
       lambda r: assert_(all(Path(p).exists() for p in r), "some archives missing"))
print(f"         archived: {[Path(p).name for p in archived]}")

# Test idempotency (no files to archive now)
archived2 = archive_previous_outputs(str(archive_dir), "myjob")
report("no-op when nothing to archive", archived2,
       lambda r: assert_(len(r) == 0, f"should be empty, got {len(r)}"))

# Cleanup
import shutil
shutil.rmtree(archive_dir)


# ============================================================
print("\n=== 5. preflight_check ===")
# ============================================================
from chemtools import preflight_check

# Set profiles path to the example
os.environ["CHEMTOOLS_RUNNER_PROFILES"] = str(REPO / "examples" / "tacc_stampede3" / "runner_profiles.yaml")
try:
    pf = preflight_check(
        input_file=str(MOCK_INPUT),
        profile="stampede3_skx",
    )
    report("returns result", pf, lambda r: assert_(isinstance(r, dict)))
    report("has ready_to_submit", pf,
           lambda r: assert_("ready_to_submit" in r, "missing ready_to_submit"))
    report("has checks list", pf,
           lambda r: assert_(len(r["checks"]) > 0, "no checks"))
    report("lint check present", pf,
           lambda r: assert_(any(c["check"] == "lint" for c in r["checks"]),
                              "lint check missing"))
    print(f"         {pf['summary']}")
    for c in pf["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"           [{status}] {c['check']}")
except Exception as e:
    print(f"  FAIL  preflight_check: {e}")
    import traceback; traceback.print_exc()
    FAIL += 1


# ============================================================
print("\n=== 6. query_partition_specs + get_local_resource_budget ===")
# ============================================================
from chemtools.runner import query_partition_specs, get_local_resource_budget

# Local (should always work)
local = get_local_resource_budget()
report("local budget returns cores", local,
       lambda r: assert_(r["physical_cores"] > 0, f"cores={r['physical_cores']}"))
report("local budget returns memory", local,
       lambda r: assert_(r["total_mem_mb"] is not None and r["total_mem_mb"] > 0,
                          f"mem={r['total_mem_mb']}"))
report("local budget returns arch", local,
       lambda r: assert_(r["cpu_arch"] in ("avx512", "avx2", "arm", "generic"),
                          f"arch={r['cpu_arch']}"))
print(f"         cores={local['physical_cores']}, avail={local['available_cores']}, "
      f"mem={local['total_mem_mb']}MB, arch={local['cpu_arch']}")

# Partition query (will gracefully fail if no sinfo)
part = query_partition_specs("skx", "slurm")
has_sinfo = part["node_memory_mb"] is not None
if has_sinfo:
    report("sinfo returned memory", part,
           lambda r: assert_(r["node_memory_mb"] > 0))
    report("sinfo returned cores", part,
           lambda r: assert_(r["cpus_per_node"] > 0))
    print(f"         skx: mem={part['node_memory_mb']}MB, cores={part['cpus_per_node']}, "
          f"arch={part['cpu_arch']}")
else:
    print(f"         sinfo not available (expected on local machine), fallback OK")
    report("graceful fallback", part,
           lambda r: assert_(r["cpu_arch"] == "generic", f"arch={r['cpu_arch']}"))


# ============================================================
print("\n=== 7. suggest_resources ===")
# ============================================================
from chemtools import suggest_resources

# Use local hw specs
sr = suggest_resources(
    input_file=str(MOCK_INPUT),
    hw_specs=local,
)
report("returns mpi_ranks", sr,
       lambda r: assert_(r["mpi_ranks"] > 0, f"ranks={r['mpi_ranks']}"))
report("returns memory_per_rank_mb", sr,
       lambda r: assert_(r["memory_per_rank_mb"] > 0, f"mem={r['memory_per_rank_mb']}"))
report("returns estimated_basis_functions", sr,
       lambda r: assert_(r["estimated_basis_functions"] > 0, f"bf={r['estimated_basis_functions']}"))
report("ranks <= available cores", sr,
       lambda r: assert_(r["mpi_ranks"] <= local["available_cores"],
                          f"ranks {r['mpi_ranks']} > cores {local['available_cores']}"))
print(f"         ranks={sr['mpi_ranks']}, mem/rank={sr['memory_per_rank_mb']}MB, "
      f"BF={sr['estimated_basis_functions']}, BF/rank={sr['bf_per_rank_actual']}")
print(f"         rationale: {sr['rationale']}")

# Also test with the real Am complex if available
if OPT_INPUT and OPT_INPUT.exists():
    sr2 = suggest_resources(
        input_file=str(OPT_INPUT),
        hw_specs={"cpus_per_node": 48, "node_memory_mb": 192000, "cpu_arch": "skx"},
    )
    print(f"         Am complex on SKX: ranks={sr2['mpi_ranks']}, BF={sr2['estimated_basis_functions']}, "
          f"BF/rank={sr2['bf_per_rank_actual']}")
    report("Am complex reasonable rank count", sr2,
           lambda r: assert_(1 <= r["mpi_ranks"] <= 48, f"ranks={r['mpi_ranks']}"))


# ============================================================
print("\n=== 8. stale .jobid detection ===")
# ============================================================
from chemtools.runner import _extract_job_id_from_err, inspect_nwchem_run_status

# Test the helper directly
err_id = _extract_job_id_from_err(str(MOCK_ERR))
report("extracts job ID from .err", err_id,
       lambda r: assert_(r == "9999999", f"got {r}, expected 9999999"))

# Test cross-check in inspect_nwchem_run_status
# .jobid says 1234567, .err says 9999999
status = inspect_nwchem_run_status(
    output_path=str(TEST_DIR / "test_mol.out"),
    error_path=str(MOCK_ERR),
)
report("detects stale jobid", status,
       lambda r: assert_(r.get("jobid_stale_warning") is not None,
                          "no stale warning"))
if status.get("jobid_stale_warning"):
    print(f"         warning: {status['jobid_stale_warning']}")


# ============================================================
print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
