# chemtools-nwchem MCP: Improvement Roadmap

Derived from two real NWChem sessions on TACC Stampede3 (Am-complex DFT/freq
calculations), code review of the repo at
https://github.com/charliecpeterson/chemtoolsmcp, and planning discussions
around scale, campaign management, and smaller-model friendliness.
Intended as a specification for an AI coding assistant to implement.

**Document sections:**
1. Dynamic Resource Detection
2. Resource Recommendation Engine
3. Run History Database
4. Freq Calculation Tools
5. Output Parsing Improvements
6. Workflow Connectivity Fixes
7. Preflight Check Tool
8. Output File Versioning
9. Auto Memory Suggestion
10. Smaller-Model Friendly Tool Design
11. Multi-Run / Campaign Management
12. Better NWChem Analysis Tools
13. Self-Improving / Adaptive Tools
14. Summary Priority Tables
15. Notes on Empirical Calibration

---

## 1. Dynamic Resource Detection (Highest Priority)

### 1.1 HPC: Query Partition Specs at Submit Time

**Problem**: Runner profiles currently hardcode `mpi_ranks` and have no
`node_memory_mb` field. When switching partitions (e.g., SKX → SPR on
Stampede3), the hardcoded memory per rank exceeded the node's physical RAM,
causing immediate NWChem MA_init crashes. The user had no way to know the
correct value without manual `sinfo` queries.

**Solution**: Add a `query_partition_specs()` function that runs the
scheduler's native info command and parses real node specs. Call it
automatically inside `launch_nwchem_run` before resource decisions are made.

```python
# chemtools/runner.py  (new function)

def query_partition_specs(
    partition: str,
    scheduler_type: str,           # "slurm" | "pbs" | "lsf"
    cache: dict | None = None,     # pass a dict to cache per session
) -> dict:
    """
    Query the scheduler for real node specs on a partition.
    Returns node_memory_mb, cpus_per_node, cpu_arch, and raw features.
    Falls back to None values if the query fails (e.g., on login-less systems).
    """
    if cache is not None and partition in cache:
        return cache[partition]

    result = {"node_memory_mb": None, "cpus_per_node": None,
              "cpu_arch": "generic", "features": []}

    if scheduler_type == "slurm":
        import subprocess, shutil
        if not shutil.which("sinfo"):
            return result
        try:
            proc = subprocess.run(
                ["sinfo", "-p", partition, "-o", "%m %c %f", "--noheader"],
                capture_output=True, text=True, timeout=10
            )
            lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
            if not lines:
                return result
            # Handle heterogeneous partitions: use minimum memory (conservative)
            rows = [l.split(None, 2) for l in lines]
            min_mem = min(int(r[0]) for r in rows)
            min_cpu = min(int(r[1]) for r in rows)
            all_features = set(",".join(r[2] for r in rows if len(r) > 2).split(","))
            arch = ("spr" if "spr" in all_features else
                    "skx" if "skx" in all_features else
                    "knl" if "knl" in all_features else "generic")
            result = {
                "node_memory_mb": min_mem,
                "cpus_per_node": min_cpu,
                "cpu_arch": arch,
                "features": sorted(all_features),
            }
        except Exception:
            pass  # fall through to None defaults

    elif scheduler_type == "pbs":
        # parse `pbsnodes -a` or `qstat -q <partition>` similarly
        pass

    if cache is not None:
        cache[partition] = result
    return result
```

**Profile schema change**: Remove `mpi_ranks` and `node_memory_mb` as required
fields — they become optional overrides. The profile only needs routing info:

```json
{
  "stampede3_spr": {
    "launcher": { "kind": "scheduler", "scheduler_type": "slurm" },
    "resources": {
      "nodes": 1,
      "partition": "spr",
      "walltime": "24:00:00",
      "account": null
    }
  }
}
```

If `mpi_ranks` is omitted, `launch_nwchem_run` calls `query_partition_specs`
and `suggest_resources` (see §2) to derive it automatically. Users can still
override with `resource_overrides={"mpi_ranks": 32}`.

---

### 1.2 Local/Direct Runs: Read System State at Launch Time

**Problem**: Direct runner profiles hardcode `-np N` in their command template.
This ignores how many cores are actually free and how much RAM is available,
making local runs non-portable and potentially slow if the machine is loaded.

**Solution**: Add `get_local_resource_budget()` that reads live system state
via `psutil` (already a common dependency) and uses it as the hardware spec
for `suggest_resources`.

```python
# chemtools/runner.py  (new function)

def get_local_resource_budget() -> dict:
    """
    Return available CPU cores and memory on the local machine,
    accounting for current load so we don't saturate a busy system.
    """
    try:
        import psutil
        phys_cores = psutil.cpu_count(logical=False) or 1
        load_1min = psutil.getloadavg()[0]
        # Cores already consumed by other processes
        cores_in_use = min(int(load_1min + 0.5), phys_cores - 1)
        available_cores = max(1, phys_cores - cores_in_use)

        mem = psutil.virtual_memory()
        # Reserve 15% for OS + other processes
        available_mem_mb = int(mem.available / 1_000_000 * 0.85)
        total_mem_mb = int(mem.total / 1_000_000)

        return {
            "physical_cores": phys_cores,
            "available_cores": available_cores,
            "current_load_1min": load_1min,
            "total_mem_mb": total_mem_mb,
            "available_mem_mb": available_mem_mb,
            "cpu_arch": _detect_local_cpu_arch(),  # see below
        }
    except ImportError:
        # psutil not available: fall back to os module
        import os
        cores = os.cpu_count() or 1
        return {
            "physical_cores": cores,
            "available_cores": max(1, cores - 1),
            "current_load_1min": None,
            "total_mem_mb": None,
            "available_mem_mb": None,
            "cpu_arch": "generic",
        }


def _detect_local_cpu_arch() -> str:
    """Detect AVX-512 / AVX2 / generic from /proc/cpuinfo or platform."""
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        if "avx512f" in flags:
            return "avx512"
        if "avx2" in flags:
            return "avx2"
    except OSError:
        pass
    import platform
    machine = platform.machine().lower()
    return "arm" if "arm" in machine or "aarch" in machine else "generic"
```

**Integration in `launch_nwchem_run`**:

```python
# In runner.py: _resolve_resources() — called before render

def _resolve_resources(profile_payload, input_path, resource_overrides):
    kind = profile_payload["launcher"]["kind"]

    if kind == "scheduler":
        partition = profile_payload["resources"].get("partition")
        sched = profile_payload["launcher"].get("scheduler_type", "slurm")
        hw = query_partition_specs(partition, sched)
    else:
        hw = get_local_resource_budget()

    suggested = suggest_resources(input_path, hw)  # see §2

    # Merge: profile defaults < suggested < explicit overrides
    resources = {**profile_payload["resources"], **suggested}
    resources.update(resource_overrides or {})
    return resources, hw
```

---

## 2. Resource Recommendation Engine

### 2.1 New tool: `suggest_resources`

**Problem**: No tool exists to recommend MPI rank count. The existing
`suggest_memory` is heuristic-only and doesn't know about node RAM or ranks.
In practice, the Am-complex jobs ran 112 ranks for a ~700 basis-function system
— far above the efficient range — likely causing slow per-displacement SCFs.

**Solution**: `suggest_resources` combines basis-function scaling theory with
hardware specs (from §1) and optional run history (from §3).

```python
# chemtools/api_strategy.py  (new function)

# Empirical target: basis functions per MPI rank for good parallel efficiency.
# Below this, communication overhead dominates. Values are starting points;
# they should be updated as empirical data accumulates (see §3).
_BF_PER_RANK_TARGET = {
    "spr":     60,   # AVX-512, high memory bandwidth
    "skx":     80,   # AVX-512, standard SKX
    "avx512":  70,
    "avx2":    90,
    "knl":    120,   # KNL has high core count but weak single-core perf
    "generic": 80,
}

def suggest_resources(
    input_file: str,
    hw_specs: dict,                        # from query_partition_specs or get_local_resource_budget
    past_runs: list[dict] | None = None,   # from run history (§3)
    task_override: str | None = None,
) -> dict:
    """
    Recommend mpi_ranks and memory_per_rank_mb for a NWChem job.
    hw_specs comes from query_partition_specs() or get_local_resource_budget().
    """
    from .api_input import lint_nwchem_input
    summary = lint_nwchem_input(input_file).get("input_summary", {})
    elements = summary.get("elements", [])
    n_atoms = sum(1 for _ in elements)  # approximate; lint gives element list
    method = (summary.get("tasks") or [{}])[0].get("module", "dft")
    task = task_override or (summary.get("tasks") or [{}])[0].get("operation", "energy")

    # Estimate basis functions
    M = _estimate_basis_functions(elements, summary.get("basis_block"))

    # CPU-arch-aware parallelism target
    arch = hw_specs.get("cpu_arch", "generic")
    bf_per_rank = _BF_PER_RANK_TARGET.get(arch, 80)

    # Ranks from scaling model
    max_cores = hw_specs.get("cpus_per_node") or hw_specs.get("available_cores") or 1
    ranks_by_scaling = max(1, M // bf_per_rank)

    # Ranks from memory budget (must fit in node RAM)
    node_mem = hw_specs.get("node_memory_mb") or hw_specs.get("available_mem_mb")
    if node_mem:
        min_mem_per_rank = 400  # MB: absolute floor for NWChem to start
        ranks_by_memory = int(node_mem * 0.80 / min_mem_per_rank)
    else:
        ranks_by_memory = max_cores

    optimal_ranks = min(ranks_by_scaling, ranks_by_memory, max_cores)
    optimal_ranks = max(1, optimal_ranks)

    # Refine from empirical history if available
    rationale = f"BF/rank model: {M} BF / {bf_per_rank} target = {ranks_by_scaling} ranks"
    if past_runs:
        empirical = _refine_ranks_from_history(past_runs, M, method, arch, optimal_ranks)
        if empirical["adjusted"]:
            optimal_ranks = empirical["recommended_ranks"]
            rationale += f"; empirically adjusted: {empirical['reason']}"

    # Memory per rank
    if node_mem:
        mem_per_rank = int(node_mem * 0.80 / optimal_ranks)
    else:
        mem_per_rank = suggest_memory(
            n_atoms=n_atoms, basis="6-31g*", method=method
        )["recommended_total_mb"]

    return {
        "mpi_ranks": optimal_ranks,
        "memory_per_rank_mb": mem_per_rank,
        "estimated_basis_functions": M,
        "bf_per_rank_actual": round(M / optimal_ranks, 1),
        "cpu_arch": arch,
        "rationale": rationale,
    }
```

**Helper: `_estimate_basis_functions`**

Reuse the existing `_BASIS_SCALE` table from `suggest_memory`. Parse the basis
block from the lint summary to count contracted shells per element type, multiply
by atom count. This is already partially implemented — consolidate into one
shared function.

---

## 3. Run History Database

### 3.1 Append-only JSONL per working directory

**Problem**: No empirical data is collected from completed jobs. Every resource
decision starts from scratch, and the tool cannot improve its estimates over
time. The freq restart workflow required manually counting gradient evaluations
to estimate progress and walltime.

**Solution**: After each job completes (inside the watch loop), append a
structured record to `.chemtools_runs.jsonl` in the job directory.

```python
# chemtools/runner.py

def _record_run_outcome(
    job_dir: str,
    job_name: str,
    profile_name: str,
    hw_specs: dict,
    input_summary: dict,
    timing: dict,       # from watch result
    output_path: str,
) -> None:
    import json, re
    from datetime import datetime, timezone
    from pathlib import Path

    # Extract memory actually used from output
    peak_mem_mb = _parse_peak_memory_from_output(output_path)
    sec_per_gradient = _parse_sec_per_gradient(output_path)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_name": job_name,
        "profile": profile_name,
        "n_atoms": input_summary.get("n_atoms"),
        "elements": input_summary.get("elements"),
        "estimated_basis_functions": input_summary.get("estimated_bf"),
        "method": input_summary.get("method"),
        "task": input_summary.get("task"),
        "cpu_arch": hw_specs.get("cpu_arch"),
        "mpi_ranks": hw_specs.get("mpi_ranks_used"),
        "node_memory_mb": hw_specs.get("node_memory_mb"),
        "memory_per_rank_mb": hw_specs.get("memory_per_rank_mb"),
        "walltime_used_sec": timing.get("elapsed_sec"),
        "peak_memory_mb_actual": peak_mem_mb,
        "status": timing.get("final_status"),   # "completed" | "timelimit" | "oom" | "failed"
        "sec_per_gradient": sec_per_gradient,   # None if not a freq/gradient job
        "n_gradients_completed": timing.get("n_gradients"),
    }

    db_path = Path(job_dir) / ".chemtools_runs.jsonl"
    with db_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _parse_peak_memory_from_output(output_path: str) -> float | None:
    """Extract 'Heap Space remaining' lines to infer actual memory used."""
    import re
    try:
        text = open(output_path).read()
        # "Heap Space remaining (MW):    26.16   26160296"
        matches = re.findall(r"Heap Space remaining \(MW\):\s+([\d.]+)", text)
        if matches:
            # Initial heap minus minimum remaining = used
            pass  # implementation detail
    except OSError:
        pass
    return None


def _parse_sec_per_gradient(output_path: str) -> float | None:
    """
    Parse wall-time stamps from finite-difference gradient output.
    Returns average seconds between consecutive gradient completions.
    """
    import re
    try:
        text = open(output_path).read()
        times = [float(m) for m in re.findall(r"wall time:\s+([\d.]+)", text)]
        if len(times) >= 2:
            diffs = [times[i+1] - times[i] for i in range(len(times)-1) if times[i+1] > times[i]]
            if diffs:
                return sum(diffs) / len(diffs)
    except OSError:
        pass
    return None
```

### 3.2 Load history for `suggest_resources`

```python
def load_run_history(job_dir: str) -> list[dict]:
    import json
    from pathlib import Path
    db = Path(job_dir) / ".chemtools_runs.jsonl"
    if not db.exists():
        return []
    records = []
    for line in db.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _refine_ranks_from_history(
    past_runs: list[dict],
    M: int,
    method: str,
    arch: str,
    current_suggestion: int,
) -> dict:
    """
    Find past runs with similar basis-function count on the same arch.
    If a run with more ranks was slower (per-gradient) than one with fewer,
    suggest the more efficient count.
    """
    similar = [
        r for r in past_runs
        if r.get("cpu_arch") == arch
        and r.get("method") == method
        and r.get("estimated_basis_functions")
        and abs(r["estimated_basis_functions"] - M) / max(M, 1) < 0.3
        and r.get("sec_per_gradient")
        and r.get("mpi_ranks")
    ]
    if len(similar) < 2:
        return {"adjusted": False, "recommended_ranks": current_suggestion}

    # Find the rank count with best sec/gradient
    best = min(similar, key=lambda r: r["sec_per_gradient"])
    if best["mpi_ranks"] != current_suggestion:
        return {
            "adjusted": True,
            "recommended_ranks": best["mpi_ranks"],
            "reason": f"empirically {best['sec_per_gradient']:.0f}s/gradient at {best['mpi_ranks']} ranks "
                      f"(from {best['job_name']})",
        }
    return {"adjusted": False, "recommended_ranks": current_suggestion}
```

---

## 4. Freq Calculation Tools

### 4.1 New tool: `parse_nwchem_freq_progress`

**Problem**: No tool reports how far a finite-difference freq job has progressed.
Determining this required manually grepping for `gen_hess restart`, counting
`DFT ENERGY GRADIENTS`, and reading `iatom_start`/`ixyz_start` from the output.
This is needed every time a freq job is restarted (which for large systems with
48-hour walltime limits happens multiple times).

```python
# chemtools/nwchem_freq.py  (new function)

def parse_freq_progress(path: str, contents: str) -> dict:
    """
    Report finite-difference Hessian progress: displacements done vs. total,
    pace, and estimated remaining time and runs needed.
    """
    import re, math

    # Get total atom count from geometry block
    atoms_match = re.search(r"No\.\s+Tag.*?\n((?:\s+\d+\s+\w+.*\n)+)", contents)
    n_atoms = len(atoms_match.group(1).strip().splitlines()) if atoms_match else None
    n_total = 2 * 3 * n_atoms if n_atoms else None  # ±displacements × 3 coords × N atoms

    # Find where the current run started (gen_hess restart)
    restart_match = re.search(
        r"\*\*\*\* gen_hess restart \*\*\*\*.*?iatom_start\s*=\s*(\d+).*?ixyz_start\s*=\s*(\d+)",
        contents, re.DOTALL
    )
    if restart_match:
        iatom_start = int(restart_match.group(1))
        ixyz_start  = int(restart_match.group(2))
        n_done_before_this_run = (iatom_start - 1) * 6 + (ixyz_start - 1) * 2
    else:
        n_done_before_this_run = 0

    # Count gradients completed in this run + extract wall-time stamps
    gradient_times = re.findall(
        r"atom:\s*\d+\s*xyz:\s*\d+\([+-]\)\s+wall time:\s*([\d.]+)", contents
    )
    n_done_this_run = len(gradient_times)
    n_done_total = n_done_before_this_run + n_done_this_run

    # Pace from recent 10 gradients
    eta_sec = None
    sec_per_grad = None
    runs_needed = None
    if len(gradient_times) >= 2:
        times = [float(t) for t in gradient_times]
        recent = times[-min(10, len(times)):]
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1) if recent[i+1] > recent[i]]
        if diffs:
            sec_per_grad = sum(diffs) / len(diffs)
            remaining = (n_total - n_done_total) * sec_per_grad if n_total else None
            eta_sec = remaining
            runs_needed = math.ceil(remaining / (48 * 3600)) if remaining else None

    return {
        "n_atoms": n_atoms,
        "n_total_displacements": n_total,
        "n_done_cumulative": n_done_total,
        "n_done_this_run": n_done_this_run,
        "n_done_before_restart": n_done_before_this_run,
        "pct_complete": round(100 * n_done_total / n_total, 1) if n_total else None,
        "sec_per_gradient_recent": round(sec_per_grad, 1) if sec_per_grad else None,
        "estimated_remaining_hours": round(eta_sec / 3600, 1) if eta_sec else None,
        "runs_needed_at_48h_walltime": runs_needed,
        "fdrst_file": path.replace(".out", ".fdrst").replace(".nw", ".fdrst"),
    }
```

Expose as MCP tool `parse_nwchem_freq_progress(output_file)`.

---

### 4.2 New tool: `prepare_nwchem_freq_restart`

**Problem**: Restarting a freq job required manually verifying the input has
`restart`, checking the `.fdrst` exists, and resubmitting. These steps should
be one tool call.

```python
# chemtools/api_strategy.py  (new function)

def prepare_nwchem_freq_restart(
    input_file: str,
    output_file: str,
    profile: str,
    profiles_path: str | None = None,
) -> dict:
    """
    Validate that a freq restart is ready and return a submit-ready config.
    Checks: restart keyword present, .fdrst exists, reports progress.
    Does NOT submit — caller decides whether to call launch_nwchem_run.
    """
    import re
    from pathlib import Path

    nw_text = Path(input_file).read_text()
    out_text = Path(output_file).read_text() if Path(output_file).exists() else ""

    issues = []

    # Check restart keyword
    has_restart = bool(re.search(r"^\s*restart\b", nw_text, re.MULTILINE | re.IGNORECASE))
    if not has_restart:
        issues.append("Input is missing 'restart' keyword — NWChem will start from scratch")

    # Check .fdrst exists
    stem = re.search(r"^\s*restart\s+(\S+)", nw_text, re.MULTILINE | re.IGNORECASE)
    fdrst_name = (stem.group(1) if stem else Path(input_file).stem) + ".fdrst"
    fdrst_path = Path(input_file).parent / fdrst_name
    has_fdrst = fdrst_path.exists()
    if not has_fdrst:
        issues.append(f"Checkpoint file {fdrst_name} not found — restart will start from atom 1")

    # Parse progress
    progress = parse_freq_progress(output_file, out_text) if out_text else {}

    return {
        "ready_to_restart": len(issues) == 0,
        "issues": issues,
        "input_file": input_file,
        "fdrst_file": str(fdrst_path),
        "fdrst_exists": has_fdrst,
        "fdrst_size_kb": round(fdrst_path.stat().st_size / 1024, 1) if has_fdrst else None,
        "progress": progress,
        "suggested_profile": profile,
    }
```

---

## 5. Output Parsing Improvements

### 5.1 Compact defaults for `analyze_nwchem_imaginary_modes`

**Problem**: The tool returned 63K characters including full Cartesian
displacement vectors for all imaginary modes across all atoms. This exceeded
MCP token limits and required saving to a file and reading back.

**Solution**: Add a `detail` parameter defaulting to `"compact"`. Displacement
vectors should only be returned when `detail="full"` or for a specific mode.

```python
# chemtools/nwchem_freq.py  — modify analyze_imaginary_modes signature

def analyze_imaginary_modes(
    path: str,
    contents: str,
    significant_threshold_cm1: float = 20.0,
    top_atoms: int = 4,
    detail: str = "compact",   # NEW: "compact" | "full"
) -> dict:
    ...
    for mode in significant_modes:
        analysis = _analyze_single_mode(mode, top_atoms=top_atoms, ...)
        if detail == "compact":
            # Drop displacement_vectors from the mode dict before returning
            analysis.pop("displacement_vectors", None)
            analysis.pop("cartesian_displacements", None)
        result["modes"].append(analysis)
```

Compact output for 8 imaginary modes on a 76-atom system: ~3 KB vs. 63 KB.
Displacement vectors are only needed when calling
`displace_nwchem_geometry_along_mode`, not for diagnosis.

---

### 5.2 `imaginary_only` filter on `parse_nwchem_output` freq section

Add a filter parameter so the caller can get just the imaginary modes without
receiving all 228 modes:

```python
# chemtools/api_output.py  — in the freq section dispatcher

def parse_output(path, sections=None, ..., imaginary_only=False, top_n_modes=None):
    ...
    if "freq" in sections:
        freq_data = parse_freq(path, contents, ...)
        if imaginary_only:
            freq_data["modes"] = [m for m in freq_data["modes"]
                                  if m["frequency_cm1"] < 0]
        if top_n_modes:
            freq_data["modes"] = freq_data["modes"][:top_n_modes]
```

---

## 6. Workflow Connectivity Fixes

### 6.1 `draft_nwchem_optimization_followup_input` — accept freq-only source

**Problem**: The tool failed with "no optimization geometry frames were found"
when the source output was a single-point freq job. It only handled outputs
containing an optimization trajectory.

**Fix**: Fall back to the last printed geometry in the output (which is always
present in any NWChem job that completed geometry setup) when no optimization
frames are found.

```python
# chemtools/api_input.py  — in draft_optimization_followup_input

def _get_source_geometry(output_path, contents):
    # Try 1: last optimization step
    geom = _extract_last_optimization_geometry(contents)
    if geom:
        return geom, "optimization_trajectory"

    # Try 2: last printed geometry block (freq, single-point, etc.)
    geom = _extract_last_geometry_atoms(contents)
    if geom:
        return geom, "last_printed_geometry"

    raise ValueError(
        "No geometry found in output. Provide an output file from any "
        "completed NWChem job (optimization, freq, or single-point)."
    )
```

---

### 6.2 Close the displacement → new input pipeline

**Problem**: `displace_nwchem_geometry_along_mode` returns displaced
coordinates, but there is no tool to turn those coordinates + the original
input settings into a new `.nw` file ready to submit. The full input had to be
reconstructed manually from the output echo, copying basis sets, ECP, DFT
block, and COSMO settings by hand.

**Solution**: New tool `create_nwchem_input_from_displaced_geometry` that
combines the displaced geometry with all settings extracted from the original
output or input file.

```python
# chemtools/api_input.py  (new function)

def create_nwchem_input_from_displaced_geometry(
    source_output_file: str,      # original job output (for settings echo)
    displaced_geometry: dict,     # result of displace_nwchem_geometry_along_mode
    task_strategy: str,           # "optimize" | "optimize_then_freq" | "freq"
    base_name: str,
    output_dir: str,
    title: str | None = None,
    write_file: bool = True,
) -> dict:
    """
    Build a complete NWChem input from a displaced geometry + settings
    extracted from the source output's input echo block.
    Handles: geometry, basis, ECP, DFT/SCF settings, COSMO, memory.
    """
    ...
```

This replaces the manual `awk`/`sed` workflow used in the previous session.

---

## 7. Preflight Check Tool

**Problem**: Several failures could have been caught before submission:
- `movecs input` pointing to a nonexistent file (not caught by lint)
- Memory per rank exceeding node RAM (not caught until MA_init crash)
- Wrong MKL library on SPR nodes (environmental, but detectable)

**Solution**: New tool `preflight_check` that combines lint + resource
validation + file existence checks:

```python
# chemtools/api_strategy.py  (new function)

def preflight_check(
    input_file: str,
    profile: str,
    profiles_path: str | None = None,
) -> dict:
    """
    Run all pre-submission checks and return a pass/fail report.
    Combines: lint, memory ceiling, movecs file existence, env compatibility.
    """
    checks = []

    # 1. Lint
    lint = lint_nwchem_input(input_file)
    checks.append({
        "check": "lint",
        "passed": lint["status"] in ("ok", "warning"),
        "issues": lint["issues"],
    })

    # 2. movecs input file exists
    nw_text = open(input_file).read()
    for m in re.finditer(r"vectors\s+input\s+(\S+)", nw_text, re.IGNORECASE):
        movecs_path = Path(input_file).parent / m.group(1)
        checks.append({
            "check": f"movecs_exists:{m.group(1)}",
            "passed": movecs_path.exists(),
            "issues": [] if movecs_path.exists() else [
                {"level": "error", "message": f"vectors input file not found: {movecs_path}"}
            ],
        })

    # 3. Memory vs. node RAM
    hw = query_partition_specs(partition, scheduler_type)
    suggested = suggest_resources(input_file, hw)
    requested_mem_total = suggested["mpi_ranks"] * suggested["memory_per_rank_mb"]
    if hw.get("node_memory_mb"):
        ok = requested_mem_total < hw["node_memory_mb"] * 0.90
        checks.append({
            "check": "memory_ceiling",
            "passed": ok,
            "details": {
                "requested_mb": requested_mem_total,
                "node_memory_mb": hw["node_memory_mb"],
            },
            "issues": [] if ok else [
                {"level": "error",
                 "message": f"Memory request {requested_mem_total} MB exceeds "
                            f"90% of node RAM ({hw['node_memory_mb']} MB). "
                            f"Reduce memory_per_rank or mpi_ranks."}
            ],
        })

    all_passed = all(c["passed"] for c in checks)
    return {
        "ready_to_submit": all_passed,
        "checks": checks,
        "summary": f"{'PASS' if all_passed else 'FAIL'}: "
                   f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed",
    }
```

Expose as MCP tool `preflight_check(input_file, profile)`. Call it
automatically before `launch_nwchem_run` and block submission if any check is
`level: "error"`.

---

## 8. Output File Versioning

**Problem**: Every resubmission overwrote `.out`, `.err`, `.job`, and `.jobid`.
This caused real confusion when diagnosing errors — it was impossible to tell
whether the error was from the current run or a previous one. The `next_versioned_path`
tool already exists for input files but is not applied to outputs.

**Solution**: In `launch_nwchem_run`, when output files from a previous run
exist, rename them with a timestamp suffix before overwriting:

```python
# chemtools/runner.py  — before writing new .job and accepting output paths

def _archive_previous_output(job_dir: str, job_name: str) -> list[str]:
    """
    If {job_name}.out/.err/.job already exist, rename them to
    {job_name}.out.2026-04-17T22-35 so they are not overwritten.
    Returns list of archived paths.
    """
    from pathlib import Path
    from datetime import datetime
    archived = []
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
    for ext in (".out", ".err", ".job"):
        p = Path(job_dir) / f"{job_name}{ext}"
        if p.exists() and p.stat().st_size > 0:
            dest = p.with_name(f"{job_name}{ext}.{ts}")
            p.rename(dest)
            archived.append(str(dest))
    return archived
```

Add an `archive_previous_outputs: bool = True` parameter to `launch_nwchem_run`.

---

## 9. MCP Tool: `suggest_memory` — integrate with launch

**Problem**: `suggest_memory` exists as a standalone tool but is never called
automatically. The user has to know to call it separately, and it doesn't use
profile information (node RAM, rank count).

**Fix**: Call it automatically inside `launch_nwchem_run` if `memory` is not
already set in the input file's `memory` directive AND no override is provided:

```python
# In render_nwchem_run, after resolving resources:

if not _input_has_memory_directive(input_text):
    mem = suggest_memory(n_atoms, basis, method)
    per_rank = min(mem["recommended_total_mb"], 
                   int(hw["node_memory_mb"] * 0.80 / resolved_ranks))
    warnings.append(
        f"No memory directive found. Suggest: memory total {per_rank} mb noverify"
    )
```

Emit the suggestion as a warning in the launch result, not a hard block.

---

---

## 10. Smaller-Model Friendly Tool Design

The goal here is to move chemistry reasoning FROM the LLM INTO the tools so
that a cheaper, less capable model can drive the full NWChem workflow without
understanding NWChem internals. A tool is smaller-model friendly when calling
it and reading its output is sufficient to decide the next call.

### 10.1 Structured next-action returns on every analysis tool

Every tool that diagnoses or analyzes should return a `next_actions` list of
directly callable tool specs — not prose recommendations. The model just
executes `next_actions[0]`; no chemistry reasoning required.

```python
# Instead of:
{"message": "SCF oscillating, try level shifting"}

# Return:
{
  "diagnosis": "scf_oscillation",
  "next_actions": [
    {
      "priority": 1,
      "tool": "draft_nwchem_scf_stabilization_input",
      "params": {
          "input_file": "mol.nw",
          "strategy": "level_shift",
          "level": 0.5
      },
      "reason": "energy oscillating ±0.003 Eh for 20+ iterations"
    },
    {
      "priority": 2,
      "tool": "draft_nwchem_scf_stabilization_input",
      "params": {"input_file": "mol.nw", "strategy": "tighter_grid"},
      "reason": "fallback if level shift insufficient"
    }
  ]
}
```

Apply `next_actions` to: `analyze_nwchem_case`, `watch_nwchem_run`,
`check_nwchem_freq_plausibility`, `get_nwchem_run_status`,
`analyze_nwchem_imaginary_modes`. Also add a `confidence` field (0.0–1.0)
so the model knows when to proceed vs. ask the user for guidance.

### 10.2 New tool: `get_nwchem_workflow_state`

A single tool that encodes the complete decision tree for what to do next in
any NWChem calculation. Returns an explicit state enum and a pre-filled
`next_action`. A cheap model that can call tools only needs to loop:
call `get_nwchem_workflow_state` → execute `next_action` → repeat.

```python
# chemtools/api_strategy.py  (new function)

WORKFLOW_STATES = [
    "pending",               # not yet submitted
    "running_scf",           # SCF in progress
    "running_opt",           # geometry optimization in progress
    "running_freq",          # freq displacements in progress
    "freq_timelimited",      # freq hit walltime, fdrst exists, can restart
    "freq_complete",         # all displacements done
    "opt_converged",         # geometry optimized, ready for next step
    "opt_failed_convergence",# optimization not converging
    "scf_failed",            # SCF not converging
    "imaginary_modes",       # freq done but imaginary modes present
    "oom",                   # out of memory
    "completed",             # all tasks done, results valid
    "needs_user_input",      # tool cannot decide; escalate to human
]

def get_nwchem_workflow_state(
    input_file: str,
    output_file: str,
    profile: str,
    profiles_path: str | None = None,
) -> dict:
    """
    Determine the current workflow state and return the exact next tool call
    to advance the calculation. Encodes domain logic so the LLM does not
    need to reason about NWChem internals.
    """
    # Implementation checks (in order):
    # 1. Is job running? → "running_*"
    # 2. Did it timelimit? Check .err for CANCELLED DUE TO TIME LIMIT
    # 3. Did it OOM? Check .err / .out for MA_init failure
    # 4. Did SCF fail? Check for "did not converge" in output
    # 5. Did opt converge? Check for "Optimization converged" 
    # 6. Is freq done? Check for "P.Frequency" summary section
    # 7. Are there imaginary modes? Check freq summary
    # 8. Is fdrst present? → freq_timelimited is restartable
    ...
    return {
        "state": "freq_timelimited",
        "progress_pct": 41,
        "human_summary": "Freq hit 48h walltime at 41% complete. fdrst checkpoint valid.",
        "next_action": {
            "tool": "launch_nwchem_run",
            "params": {
                "input_file": input_file,
                "profile": profile,
                "resource_overrides": {"walltime": "48:00:00"},
                "auto_watch": False,
            },
            "reason": "fdrst exists and is recent; resubmit same input to continue"
        },
        "confidence": 0.97,
        "alternatives": [],   # other options if confidence < 0.8
    }
```

### 10.3 Protocol library (pre-encoded calculation recipes)

Pre-baked calculation protocols eliminate the need for the model to know what
steps a thermochemistry workflow requires. Protocols are first-class objects
the tool uses to generate the full multi-step input set.

```python
# chemtools/protocols.py  (new file)

PROTOCOLS = {
    "geometry_opt_dft": {
        "description": "Standard DFT geometry optimization",
        "steps": ["optimize"],
        "method": "dft",
        "functional": "b3lyp",
        "basis_rule": "suggest_basis_set(purpose='geometry')",
        "convergence": "default",
    },
    "thermochem_dft": {
        "description": "Full thermochemistry: optimize then freq",
        "steps": ["optimize", "freq"],
        "method": "dft",
        "functional": "b3lyp",
        "post_process": ["parse_nwchem_thermochem"],
        "checks": ["no_imaginary_modes"],
        "on_imaginary_modes": "displace_and_reopt",
    },
    "high_accuracy_sp": {
        "description": "DFT geometry then CCSD(T) single point",
        "steps": ["optimize@b3lyp/6-31g*", "single_point@ccsd(t)/aug-cc-pvtz"],
        "requires_prior": "geometry_opt_dft",
    },
    "binding_energy": {
        "description": "Complex + fragments → ΔE_bind with BSSE",
        "steps": ["optimize_complex", "optimize_fragments", "compute_reaction_energy"],
        "bsse_correction": "counterpoise",
    },
}

def plan_calculation(
    molecule_file: str,
    protocol: str,
    profile: str,
    output_dir: str,
) -> dict:
    """
    Given a molecule and a protocol name, generate all input files,
    workflow steps, and dependencies. Returns a workflow_id ready
    for advance_workflow().
    """
    ...
```

New MCP tool: `plan_calculation(molecule_file, protocol, profile, output_dir)`

### 10.4 Smaller-model design checklist

When adding any new tool, verify against this checklist:

| Principle | Description |
|---|---|
| Structured `next_actions` | Returns callable specs, not prose |
| Explicit `state` enum | Workflow position is unambiguous |
| Bounded output | Compact by default; full detail on request |
| Pre-filled params | Next tool call params are complete, not left for model to derive |
| `confidence` score | Model knows when to proceed vs. escalate |
| No NWChem expertise required | Domain decisions encoded in tool, not expected from model |

---

## 11. Multi-Run / Campaign Management

For running and tracking large numbers of related calculations.

### 11.1 Run registry (SQLite)

A persistent database in `~/.chemtools/registry.db` that every
`launch_nwchem_run` writes to automatically. Replaces the per-directory
`.chemtools_runs.jsonl` for cross-directory queries while keeping the JSONL
as a local backup.

```sql
-- Schema

CREATE TABLE runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name        TEXT NOT NULL,
    input_file      TEXT,
    output_file     TEXT,
    profile         TEXT,
    method          TEXT,
    functional      TEXT,
    basis           TEXT,
    n_atoms         INTEGER,
    elements        TEXT,       -- JSON array
    charge          INTEGER,
    multiplicity    INTEGER,
    status          TEXT,       -- pending|running|completed|timelimited|oom|failed
    submitted_at    TEXT,       -- ISO timestamp
    completed_at    TEXT,
    walltime_used_sec REAL,
    energy_hartree  REAL,
    imaginary_modes INTEGER,
    mpi_ranks       INTEGER,
    node_memory_mb  INTEGER,
    cpu_arch        TEXT,
    sec_per_gradient REAL,
    parent_run_id   INTEGER REFERENCES runs(id),  -- for restart chains
    campaign_id     INTEGER REFERENCES campaigns(id),
    tags            TEXT        -- JSON object for arbitrary metadata
);

CREATE TABLE campaigns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at  TEXT,
    tags        TEXT
);

CREATE TABLE workflows (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER REFERENCES campaigns(id),
    name        TEXT,
    protocol    TEXT,
    state       TEXT,
    created_at  TEXT
);
```

New MCP tools built on the registry:

```python
register_run(input_file, profile, campaign=None, tags=None) → run_id
update_run_status(run_id, status, energy=None, ...)
list_runs(campaign=None, status=None, method=None, elements=None) → [run summaries]
get_run_summary(job_name_or_id) → full record including restart chain
find_runs_by_molecule(elements, charge, mult) → matching runs
compare_runs([run_id1, run_id2, ...]) → side-by-side energy/geometry table
```

### 11.2 Campaign tools

A campaign groups related runs (e.g., 20 ligands, a basis set convergence
study, or a conformer scan). Campaigns give you aggregate status at a glance.

```python
# New MCP tools

create_campaign(name, description, tags=None) → campaign_id

get_campaign_status(campaign_id) → {
    "name": "Am-ligand-binding",
    "total_runs": 20,
    "completed": 12,
    "running": 3,
    "failed": 2,
    "pending": 3,
    "completion_pct": 60.0,
    "estimated_remaining_hours": 14.0,
    "energies_available": 12,      # runs with parsed energy
    "next_actions": [...]          # e.g., resubmit 2 failed jobs
}

get_campaign_energies(campaign_id) → {
    # sorted table of all completed run energies for easy comparison
    "runs": [
        {"job_name": "mol_A", "energy_hartree": -2362.91, "G_298K_hartree": -2362.75},
        ...
    ],
    "delta_energies_kcal_mol": {...}   # relative to lowest
}
```

### 11.3 Workflow DAG for multi-step calculations

Encode step dependencies so the tool knows what to launch when an upstream
job finishes, without the model having to track state manually.

```python
# chemtools/api_strategy.py  (new functions)

create_workflow(
    name: str,
    steps: list[dict],   # [{"id": "opt", "input": "mol_opt.nw", "profile": "spr"},
                         #  {"id": "freq", "depends_on": "opt",
                         #   "auto_input": "extract_geometry_and_generate_freq_input"}]
    campaign_id: int | None = None,
) → workflow_id

advance_workflow(workflow_id) → {
    "launched": ["freq", "sp_highlevel"],
    "waiting_on": [],
    "completed": ["opt"],
    "failed": [],
    "next_action": {
        "tool": "advance_workflow",
        "params": {"workflow_id": workflow_id},
        "when": "after_running_jobs_complete"
    }
}
```

`advance_workflow` is the single call the model makes on a loop. It checks
all running jobs, marks completed ones, launches any newly unblocked steps,
and returns the updated state. No manual dependency tracking by the model.

### 11.4 Batch input generation

For scanning over chemical or computational parameters across many inputs
from a single template.

```python
# New MCP tool

generate_input_batch(
    template_input: str,           # base .nw file
    vary: dict,                    # {"charge": [0,1,2], "mult": [5,6,7]}
    output_dir: str,
    naming_pattern: str = "{stem}_q{charge}_m{mult}",
    campaign: str | None = None,   # auto-register in campaign
) → {
    "generated": ["mol_q0_m5.nw", "mol_q0_m6.nw", ...],   # 9 files
    "campaign_id": 3,
    "ready_to_submit": True,
}
```

Supports varying: charge, multiplicity, basis set, functional, COSMO
dielectric, geometry (from a list of xyz files), or any keyword-value pair
in the input template.

---

## 12. Better NWChem Analysis Tools

### 12.1 Thermochemistry post-processor

NWChem prints ZPE, thermal corrections, entropy, and enthalpy in its freq
output but not as a clean machine-readable table. A dedicated parser that
handles temperature dependence and warns about imaginary modes:

```python
# chemtools/nwchem_freq.py  (new function)

def parse_nwchem_thermochem(
    output_file: str,
    T: float = 298.15,    # K
    P: float = 1.0,       # atm
) -> dict:
    return {
        "T_K": T,
        "P_atm": P,
        "E_scf_hartree": -2362.9112,
        "ZPE_hartree": 0.2281,
        "ZPE_kcal_mol": 143.2,
        "H_thermal_kcal_mol": 5.1,
        "S_cal_mol_K": 138.5,
        "G_corr_kcal_mol": -14.2,
        "H_298K_hartree": -2362.683,
        "G_298K_hartree": -2362.754,
        "imaginary_modes_count": 0,
        "warnings": [],
        "nwchem_directive_used": "freq",
    }
```

Expose as MCP tool `parse_nwchem_thermochem(output_file, T, P)`.

### 12.2 Reaction energy with solvation decomposition

Extend the existing `compute_reaction_energy` to break out gas-phase and
solvation contributions separately when COSMO was used:

```python
compute_reaction_energy(
    reactants=["am_aq.out", "ligand.out"],
    products=["complex.out"],
    include_solvation_decomposition=True,
    T=298.15,
) → {
    "delta_E_electronic_kcal_mol": -45.2,
    "delta_G_solvation_kcal_mol": 7.1,
    "delta_ZPE_kcal_mol": -1.8,
    "delta_H_thermal_kcal_mol": 0.4,
    "delta_TdS_kcal_mol": -8.9,
    "delta_G_298K_kcal_mol": -31.4,
    "warnings": [],
}
```

### 12.3 Geometry quality checker

Check bond lengths, angles, and coordination environment against expected
ranges before submitting an expensive freq job. Catches geometry transfer
errors (e.g., water molecules in saddle-point orientations from a related
system) before wasting compute time.

```python
# New MCP tool: check_nwchem_geometry_plausibility (already exists — extend it)

check_nwchem_geometry_plausibility(output_file) → {
    "overall": "suspicious",    # "ok" | "suspicious" | "likely_wrong"
    "bond_issues": [
        {
            "atom1": "Am(1)", "atom2": "O(14)",
            "distance_ang": 3.82,
            "expected_range_ang": [2.30, 2.70],
            "severity": "warning",
            "likely_cause": "water molecule orientation not optimized for Am"
        }
    ],
    "coordination_number": {"Am": {"actual": 9, "expected_range": [8, 10]}},
    "next_actions": [
        {
            "tool": "displace_nwchem_geometry_along_mode",
            "params": {"mode_number": 1, "amplitude_angstrom": 0.15},
            "reason": "suspicious geometry — reoptimize before freq"
        }
    ]
}
```

### 12.4 Electronic structure summary tool

A compact summary of the electronic structure — HOMO-LUMO gap, spin density
on the metal center, Mulliken charges — without dumping the full MO table.
Useful for checking that the electronic state is physically reasonable after
optimization.

```python
# New MCP tool

summarize_electronic_structure(output_file) → {
    "homo_lumo_gap_ev": 2.31,
    "metal_centers": [
        {
            "atom": "Am(1)",
            "mulliken_charge": 1.82,
            "mulliken_spin_density": 5.03,
            "expected_spin_density_for_mult6": 5.0,
            "spin_state_consistent": True,
        }
    ],
    "total_spin_density": 5.01,
    "expected_for_multiplicity": 5.0,
    "spin_contamination_warning": False,
    "next_actions": [],
}
```

### 12.5 Spin-state tracker across optimization

For open-shell systems (especially f-element complexes), the electronic state
can silently flip during geometry optimization. A tool that checks each
optimization step for spin-density continuity:

```python
# New MCP tool

track_spin_state_across_optimization(output_file) → {
    "n_steps": 42,
    "spin_density_by_step": [5.02, 5.01, 4.98, ...],
    "flip_detected": False,
    "recommendation": "spin state consistent throughout optimization",
}
```

If a flip is detected, `next_actions` points to
`draft_nwchem_scf_stabilization_input` with a spin-locking strategy.

### 12.6 Input diff and change tracking

When resubmitting a failed job with modifications, record what changed and why.
This feeds into the run registry so the restart chain is fully documented.

```python
# New MCP tool

create_nwchem_input_variant(
    source_input: str,
    changes: dict,          # {"memory": "800 mb", "dft.iterations": 200}
    reason: str,            # "OOM at 2000 mb; reducing for SPR 127 GB nodes"
    output_path: str | None = None,   # auto-versioned if None
) → {
    "output_file": "mol_v2.nw",
    "diff_summary": [
        {"key": "memory", "old": "2000 mb", "new": "800 mb"},
        {"key": "dft.iterations", "old": "1000", "new": "200"},
    ],
    "reason": "OOM at 2000 mb; reducing for SPR 127 GB nodes",
    "registered_as_variant_of": "mol_v1.nw",
}
```

---

## 13. Self-Improving / Adaptive Tools

### 13.1 Autopilot mode

A thin orchestration loop that any model (even a cheap one) can run. The model
calls `get_nwchem_workflow_state`, executes `next_action`, waits for
completion, and repeats. All domain intelligence lives in the tools.

```python
# Conceptual autopilot loop (implemented in the MCP layer)

while True:
    state = get_nwchem_workflow_state(input_file, output_file, profile)

    if state["state"] == "completed":
        break
    if state["state"] == "needs_user_input":
        notify_user(state["human_summary"])
        break
    if state["confidence"] < 0.7:
        notify_user(f"Low confidence ({state['confidence']:.0%}): {state['human_summary']}")
        break

    call_tool(state["next_action"]["tool"], state["next_action"]["params"])
    wait_for_job_completion()
```

This can handle: initial submission → SCF convergence issues → opt → freq →
freq restarts (multiple) → imaginary mode fixes → reopt → final freq →
thermochemistry post-processing. The model just runs the loop.

### 13.2 Calibration workflow for scaling models

A structured benchmark to calibrate `_BF_PER_RANK_TARGET` for a new
architecture or site. Runs the same job at multiple rank counts, records
timing, and updates the profile's scaling parameters.

```python
# New MCP tool

run_scaling_benchmark(
    input_file: str,
    profile: str,
    rank_counts: list[int] = [8, 16, 32, 48, 64],
    metric: str = "sec_per_gradient",   # or "walltime_total"
) → {
    "results": [
        {"ranks": 8,  "sec_per_gradient": 210.3, "efficiency": 1.00},
        {"ranks": 16, "sec_per_gradient": 108.1, "efficiency": 0.97},
        {"ranks": 32, "sec_per_gradient":  61.4, "efficiency": 0.85},
        {"ranks": 48, "sec_per_gradient":  52.2, "efficiency": 0.67},
        {"ranks": 64, "sec_per_gradient":  49.8, "efficiency": 0.52},
    ],
    "recommended_ranks": 32,   # elbow of the efficiency curve
    "bf_per_rank_optimal": 21, # M / recommended_ranks
    "suggested_profile_update": {
        "_BF_PER_RANK_TARGET.spr": 21
    },
}
```

---

## Summary: Implementation Priority

### Phase 1 — Fix active pain points (small effort, high impact)

| # | Change | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 1 | `query_partition_specs` + integrate into launch | `runner.py` | Medium | Critical — prevents OOM |
| 2 | `get_local_resource_budget` for direct runner | `runner.py` | Small | High |
| 3 | `suggest_resources` (BF/rank model) | `api_strategy.py` | Medium | High |
| 4 | `parse_nwchem_freq_progress` MCP tool | `nwchem_freq.py` | Small | High |
| 5 | `.chemtools_runs.jsonl` append in watch loop | `runner.py` | Medium | High over time |
| 6 | `analyze_nwchem_imaginary_modes` compact default | `nwchem_freq.py` | Small | Moderate |
| 7 | `imaginary_only` filter on `parse_nwchem_output` | `api_output.py` | Small | Moderate |
| 8 | `prepare_nwchem_freq_restart` MCP tool | `api_strategy.py` | Small | Moderate |
| 9 | `draft_nwchem_optimization_followup_input` fallback | `api_input.py` | Small | Moderate |
| 10 | `create_nwchem_input_from_displaced_geometry` | `api_input.py` | Large | High |
| 11 | `preflight_check` MCP tool | `api_strategy.py` | Medium | High |
| 12 | Archive previous outputs before resubmit | `runner.py` | Small | Moderate |
| 13 | Auto-call `suggest_memory` in launch if missing | `runner.py` | Small | Moderate |

### Phase 2 — Smaller-model friendliness

| # | Change | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 14 | Add `next_actions` to all analysis tools | `api_strategy.py`, `nwchem_freq.py`, `diagnostics.py` | Medium | Critical for cheap models |
| 15 | `get_nwchem_workflow_state` state machine | `api_strategy.py` | Large | Critical for cheap models |
| 16 | Protocol library + `plan_calculation` | `protocols.py` (new) | Large | High |
| 17 | `create_nwchem_input_variant` with diff tracking | `api_input.py` | Small | Moderate |

### Phase 3 — Campaign / scale management

| # | Change | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 18 | Run registry SQLite + `register_run` / `list_runs` | `registry.py` (new) | Large | High at scale |
| 19 | Campaign tools (`create_campaign`, `get_campaign_status`) | `registry.py` | Medium | High at scale |
| 20 | Workflow DAG + `advance_workflow` | `api_strategy.py` | Large | High for multi-step |
| 21 | `generate_input_batch` | `api_input.py` | Medium | High for scans |

### Phase 4 — Better analysis

| # | Change | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 22 | `parse_nwchem_thermochem` MCP tool | `nwchem_freq.py` | Small | High |
| 23 | Solvation decomposition in `compute_reaction_energy` | `api_strategy.py` | Medium | Moderate |
| 24 | Extend `check_nwchem_geometry_plausibility` | `api_strategy.py` | Medium | Moderate |
| 25 | `summarize_electronic_structure` | `diagnostics.py` | Medium | Moderate |
| 26 | `track_spin_state_across_optimization` | `diagnostics.py` | Medium | Moderate |
| 27 | `run_scaling_benchmark` | `api_strategy.py` | Medium | High long-term |

Phase 1 addresses bugs and friction from real sessions. Phase 2 enables a
cheap model to drive the full workflow autonomously. Phase 3 scales to
campaign-level work. Phase 4 deepens the analysis capabilities.

---

| # | Change | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 1 | `query_partition_specs` + integrate into launch | `runner.py` | Medium | Critical — prevents OOM at submit |
| 2 | `get_local_resource_budget` for direct runner | `runner.py` | Small | High for local use |
| 3 | `suggest_resources` tool (BF/rank model) | `api_strategy.py` | Medium | High |
| 4 | `parse_nwchem_freq_progress` MCP tool | `nwchem_freq.py` | Small | High for freq workflows |
| 5 | `.chemtools_runs.jsonl` append in watch loop | `runner.py` | Medium | High over time |
| 6 | `analyze_nwchem_imaginary_modes` compact default | `nwchem_freq.py` | Small | Moderate — fixes token limit |
| 7 | `imaginary_only` filter on `parse_nwchem_output` | `api_output.py` | Small | Moderate |
| 8 | `prepare_nwchem_freq_restart` MCP tool | `api_strategy.py` | Small | Moderate |
| 9 | `draft_nwchem_optimization_followup_input` fallback | `api_input.py` | Small | Moderate |
| 10 | `create_nwchem_input_from_displaced_geometry` | `api_input.py` | Large | High for saddle-point workflows |
| 11 | `preflight_check` MCP tool | `api_strategy.py` | Medium | High |
| 12 | Archive previous outputs before resubmit | `runner.py` | Small | Moderate |
| 13 | Auto-call `suggest_memory` in launch if missing | `runner.py` | Small | Moderate |

Items 1–5 address the most painful problems observed in actual sessions.
Items 6–9 are small changes to existing functions.
Items 10–13 are new capabilities that improve the overall workflow.

---

## 15. Notes on Empirical Calibration

The `_BF_PER_RANK_TARGET` values in `suggest_resources` are starting points
based on general NWChem parallelism knowledge. They should be updated as real
runs accumulate in `.chemtools_runs.jsonl`. A simple calibration workflow:

1. Run the same job at 8, 16, 32, 64 ranks; record `sec_per_gradient`
2. Plot or fit a simple 1/N scaling curve; find the elbow (where adding ranks
   stops helping)
3. Update `_BF_PER_RANK_TARGET[arch]` = M / elbow_ranks

This only needs to be done once per (architecture, method) combination and
produces much better recommendations than the static heuristic alone. The run
history database in §3 accumulates this data automatically over normal use.
