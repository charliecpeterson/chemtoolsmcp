"""NWChem resource sizing advisors (CPU, memory, walltime).

Six entry points covering "how much computing power does this need":

  * _analyze_job_size           Internal — extract atom count, basis
                                 scale, methods from an input file.
                                 Used by both this module's advisors
                                 and the HPC family (lazy-imported).
  * _basis_scale                Internal — convert a basis name to a
                                 rough cost-scaling factor.
  * suggest_resources           Recommend CPU cores and memory based
                                 on local machine specs + job analysis.
  * suggest_memory              Standalone memory recommendation.
  * check_memory_fit            Validate that a proposed memory
                                 directive fits on the target machine.
  * estimate_freq_walltime      Estimate walltime for a frequency
                                 calculation given system size + method.

All six share _analyze_job_size + _basis_scale plumbing.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text
from chemtools.programs.nwchem.parse.input import inspect_nwchem_input


def _analyze_job_size(input_file: str) -> dict[str, Any]:
    """Shared helper: inspect input and extract job-size metrics.

    Returns dict with: summary, all_elements, n_atoms, n_heavy, tasks,
    main_task, module, operation, is_freq, is_opt, is_tce, basis_name,
    basis_scale, n_bf.
    """
    from chemtools.programs.nwchem.parse.input import inspect_nwchem_input, inspect_all_nwchem_basis_blocks

    summary = inspect_nwchem_input(input_file)
    all_elements = summary.get("all_elements") or summary.get("elements", [])
    n_atoms = summary.get("atom_count") or len(all_elements) or 1
    n_heavy = sum(1 for e in all_elements if e != "H") if all_elements else n_atoms
    tasks = summary.get("tasks") or [{}]
    main_task = tasks[-1] if tasks else {}
    module = (main_task.get("module") or "dft").lower()
    operation = (main_task.get("operation") or "energy").lower()

    is_freq = operation in ("freq", "frequencies", "vib")
    is_opt = operation in ("optimize", "saddle")
    is_tce = module == "tce"

    basis_blocks = inspect_all_nwchem_basis_blocks(input_file)
    basis_name = ""
    if basis_blocks:
        basis_name = basis_blocks[0].get("default_library", "") or ""
    scale = _basis_scale(basis_name) if basis_name else 1.5
    n_bf = max(10, int(n_heavy * 15 * scale))

    return {
        "summary": summary, "all_elements": all_elements,
        "n_atoms": n_atoms, "n_heavy": n_heavy,
        "tasks": tasks, "main_task": main_task,
        "module": module, "operation": operation,
        "is_freq": is_freq, "is_opt": is_opt, "is_tce": is_tce,
        "basis_name": basis_name, "basis_scale": scale, "n_bf": n_bf,
    }


def _basis_scale(basis: str) -> float:
    # Lazy import — avoids a resources↔input_advisors circular import.
    from chemtools.programs.nwchem.strategy.input_advisors import _BASIS_SCALE
    b = basis.strip().lower()
    if b in _BASIS_SCALE:
        return _BASIS_SCALE[b]
    for key, scale in sorted(_BASIS_SCALE.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in b:
            return scale
    return 1.5


# Empirical target: basis functions per MPI rank for good parallel efficiency.
# Below this, communication overhead dominates.
_BF_PER_RANK_TARGET: dict[str, int] = {
    "spr":     60,   # AVX-512, high memory bandwidth (Stampede3 SPR)
    "skx":     80,   # AVX-512, standard Skylake (Stampede3 SKX)
    "icx":     75,   # AVX-512, Ice Lake — slightly faster than SKX (Stampede3 ICX)
    "avx512":  70,
    "avx2":    90,
    "knl":    120,   # KNL: high core count but weak single-core
    "generic": 80,
}


def suggest_resources(
    input_file: str,
    hw_specs: dict[str, Any],
) -> dict[str, Any]:
    """Recommend mpi_ranks and memory_per_rank_mb for a NWChem job.

    .. deprecated::
        For HPC jobs, use :func:`suggest_hpc_resources` instead — it is
        profile-aware, multi-node capable, and handles task-type-specific
        walltime and memory estimation.  This function only handles
        single-node rank/memory selection.

    Args:
        input_file: Path to the NWChem .nw input file.
        hw_specs: Hardware specs dict.
            Expected keys: cpus_per_node (or available_cores), node_memory_mb
            (or available_mem_mb), cpu_arch.
    """
    job = _analyze_job_size(input_file)
    n_atoms = job["n_atoms"]
    M = job["n_bf"]
    method = job["module"]
    basis_name = job["basis_name"]

    # CPU-arch-aware parallelism target
    arch = hw_specs.get("cpu_arch", "generic")
    bf_per_rank = _BF_PER_RANK_TARGET.get(arch, 80)

    # Ranks from scaling model
    max_cores = hw_specs.get("cpus_per_node") or hw_specs.get("available_cores") or 1
    ranks_by_scaling = max(1, M // bf_per_rank)

    # Ranks from memory budget
    node_mem = hw_specs.get("node_memory_mb") or hw_specs.get("available_mem_mb")
    if node_mem:
        min_mem_per_rank = 400  # MB: floor for NWChem to start
        ranks_by_memory = max(1, int(node_mem * 0.80 / min_mem_per_rank))
    else:
        ranks_by_memory = max_cores

    optimal_ranks = min(ranks_by_scaling, ranks_by_memory, max_cores)
    optimal_ranks = max(1, optimal_ranks)

    rationale = f"BF/rank model: {M} BF / {bf_per_rank} target = {ranks_by_scaling} ranks"

    # Memory per rank
    if node_mem:
        mem_per_rank = int(node_mem * 0.80 / optimal_ranks)
    else:
        mem_suggestion = suggest_memory(
            n_atoms=n_atoms, basis=basis_name or "6-31g*", method=method,
        )
        mem_per_rank = mem_suggestion["recommended_total_mb"]

    return {
        "mpi_ranks": optimal_ranks,
        "memory_per_rank_mb": mem_per_rank,
        "estimated_basis_functions": M,
        "bf_per_rank_actual": round(M / optimal_ranks, 1),
        "cpu_arch": arch,
        "max_cores_available": max_cores,
        "node_memory_mb": node_mem,
        "rationale": rationale,
    }


def suggest_memory(
    n_atoms: int,
    basis: str,
    method: str,
    n_heavy_atoms: int | None = None,
) -> dict[str, Any]:
    """Suggest NWChem memory settings for a calculation.

    Returns a memory string ready for NWChem's ``memory`` directive.

    Args:
        n_atoms: Total number of atoms.
        basis: Basis set name (used to scale memory estimate).
        method: Computational method: "scf", "dft", "mp2", "ccsd", "ccsd(t)".
        n_heavy_atoms: Number of non-hydrogen atoms (optional; uses n_atoms if omitted).

    Returns dict with 'nwchem_directive' and 'memory_string'.
    """
    eff = n_heavy_atoms if n_heavy_atoms is not None else n_atoms
    scale = _basis_scale(basis)
    m = method.strip().lower()

    # Estimated basis functions: ~15 per heavy atom at double-zeta baseline
    n_bf = max(10, int(eff * 15 * scale))

    # SCF/DFT: dominated by Fock matrix + AO integrals
    fock_mb = max(64, int(8 * n_bf ** 2 / 1e6))

    if m in ("scf", "dft", "hf", "rhf", "rohf", "uhf"):
        total_mb = max(500, fock_mb * 4)
    elif m == "mp2":
        # n_occ ~ n_bf/3 is a reliable heuristic for typical neutral molecules
        # (roughly: 1/3 of basis functions are occupied at double-zeta)
        n_occ = max(1, n_bf // 3)
        n_virt = max(1, n_bf - n_occ)
        t2_mb = max(256, int(8 * (n_occ * n_virt) ** 2 / 1e6 / 4))
        total_mb = max(1000, fock_mb * 2 + t2_mb * 3)
    elif m in ("ccsd", "ccsd(t)", "tce"):
        n_occ = max(1, n_bf // 3)
        n_virt = max(1, n_bf - n_occ)
        t2_mb = max(256, int(8 * (n_occ * n_virt) ** 2 / 1e6 / 4))
        total_mb = max(2000, fock_mb * 2 + t2_mb * 6)
    else:
        total_mb = max(1000, fock_mb * 4)

    # Round to nearest 500 mb, cap at 128 GB
    total_mb = min(((total_mb + 499) // 500) * 500, 128 * 1024)

    heap_mb = max(128, total_mb // 4)
    stack_mb = max(128, total_mb // 6)
    global_mb = max(256, total_mb - heap_mb - stack_mb)
    # Ensure total >= sum of sub-components (max() floors can push sum over total)
    total_mb = max(total_mb, heap_mb + stack_mb + global_mb)

    memory_string = f"total {total_mb} mb stack {stack_mb} mb heap {heap_mb} mb global {global_mb} mb"

    return {
        "n_atoms": n_atoms,
        "n_heavy_atoms": eff,
        "basis": basis,
        "method": m,
        "basis_scale_factor": scale,
        "estimated_basis_functions": n_bf,
        "recommended_total_mb": total_mb,
        "memory_string": memory_string,
        "nwchem_directive": f"memory {memory_string}",
        "notes": (
            "Estimates are heuristic. Increase if NWChem aborts with out-of-memory errors. "
            "For CCSD(T) memory is the dominant bottleneck — more is always better."
        ),
    }


# ---------------------------------------------------------------------------
# Memory fitness check (profile-aware)
# ---------------------------------------------------------------------------


def check_memory_fit(
    input_file: str,
    profile_resources: dict[str, Any] | None = None,
    nodes: int = 1,
    mpi_ranks: int = 1,
    node_memory_mb: int | None = None,
) -> dict[str, Any]:
    """Check if an NWChem input's memory directive fits the target node.

    Reads the ``memory total`` line from *input_file* and compares against
    the node capacity.  Returns warnings and a corrected memory string when
    the requested allocation would exceed available RAM.

    *profile_resources* is the ``resources`` dict from a runner profile.
    If provided, ``nodes``, ``mpi_ranks``, and ``node_memory_mb`` are read
    from it (explicit kwargs override).
    """
    pr = profile_resources or {}
    nodes = nodes if nodes != 1 else int(pr.get("nodes", nodes))
    mpi_ranks = mpi_ranks if mpi_ranks != 1 else int(pr.get("mpi_ranks", mpi_ranks))
    node_memory_mb = node_memory_mb or pr.get("node_memory_mb")

    # Read input to find memory directive
    text = Path(input_file).read_text(encoding="utf-8", errors="replace")
    mem_line = ""
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("memory "):
            mem_line = stripped
            break

    # Extract total MB from memory line
    requested_mb_per_rank = 0
    if mem_line:
        import re
        m = re.search(r"total\s+(\d+)\s*(mb|mw|gb)", mem_line)
        if m:
            val = int(m.group(1))
            unit = m.group(2)
            if unit == "gb":
                val *= 1024
            elif unit == "mw":
                val *= 8  # 1 MW = 8 MB (64-bit words)
            requested_mb_per_rank = val

    if not requested_mb_per_rank:
        return {
            "status": "no_memory_directive",
            "message": "No 'memory total' directive found in input. NWChem will use defaults.",
            "warnings": [],
        }

    ranks_per_node = max(1, mpi_ranks // max(1, nodes))
    total_requested_per_node = requested_mb_per_rank * ranks_per_node

    warnings: list[dict[str, Any]] = []
    safe_mb_per_rank: int | None = None

    if node_memory_mb:
        # Reserve 15% for OS + MPI runtime
        usable_mb = int(node_memory_mb * 0.85)
        if total_requested_per_node > usable_mb:
            safe_mb_per_rank = max(400, (usable_mb // ranks_per_node // 100) * 100)
            warnings.append({
                "code": "memory_exceeds_node",
                "severity": "error",
                "message": (
                    f"Requested {requested_mb_per_rank} MB/rank × {ranks_per_node} ranks "
                    f"= {total_requested_per_node} MB, but node has {node_memory_mb} MB "
                    f"(~{usable_mb} MB usable). Job will crash with MA_init error."
                ),
                "fix": f"memory total {safe_mb_per_rank} mb",
            })
        elif total_requested_per_node > usable_mb * 0.9:
            warnings.append({
                "code": "memory_tight",
                "severity": "warning",
                "message": (
                    f"Requested {total_requested_per_node} MB/node is within 10% of "
                    f"usable capacity ({usable_mb} MB). Consider reducing for safety."
                ),
            })

    return {
        "status": "error" if any(w["severity"] == "error" for w in warnings) else (
            "warning" if warnings else "ok"),
        "requested_mb_per_rank": requested_mb_per_rank,
        "ranks_per_node": ranks_per_node,
        "total_mb_per_node": total_requested_per_node,
        "node_memory_mb": node_memory_mb,
        "safe_mb_per_rank": safe_mb_per_rank,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Frequency walltime estimation
# ---------------------------------------------------------------------------


def estimate_freq_walltime(
    n_atoms: int,
    seconds_per_displacement: float | None = None,
    n_displacements: int | None = None,
    mpi_ranks: int = 1,
    nodes: int = 1,
    max_walltime_hours: float = 48.0,
) -> dict[str, Any]:
    """Estimate walltime needed for a numerical frequency calculation.

    NWChem numerical frequencies require 6*N_atoms gradient evaluations
    (central differences: +/- displacement for each Cartesian coordinate).
    Each gradient evaluation is roughly the cost of a single-point SCF.

    If *seconds_per_displacement* is not provided, uses a rough heuristic
    based on atom count and MPI parallelism.

    IMPORTANT: NWChem cannot checkpoint mid-frequency. If the job runs out
    of walltime, ALL progress is lost. The job must complete in one submission.
    """
    if n_displacements is None:
        n_displacements = n_atoms * 6  # central differences: ±Δ for x,y,z per atom

    if seconds_per_displacement is None:
        # Rough heuristic: ~5 min per displacement for a 20-atom DFT/6-31G* on 48 cores
        # Scale quadratically with atom count, inversely with MPI parallelism
        base_seconds = 300.0 * (n_atoms / 20.0) ** 1.5
        total_ranks = mpi_ranks * nodes
        # Diminishing returns past ~64 ranks for intra-displacement parallelism
        effective_speedup = min(total_ranks, 64) + max(0, total_ranks - 64) * 0.3
        seconds_per_displacement = base_seconds * (48.0 / max(1, effective_speedup))

    total_seconds = n_displacements * seconds_per_displacement
    total_hours = total_seconds / 3600.0

    fits_in_walltime = total_hours <= max_walltime_hours
    safety_margin = max_walltime_hours - total_hours if fits_in_walltime else 0

    # Estimate how many nodes needed to fit in max_walltime
    if not fits_in_walltime and nodes == 1:
        # Try scaling up nodes
        for n in [2, 3, 4, 6, 8]:
            total_ranks_n = mpi_ranks * n
            eff = min(total_ranks_n, 64) + max(0, total_ranks_n - 64) * 0.3
            scaled_seconds = (48.0 / max(1, eff)) * 300.0 * (n_atoms / 20.0) ** 1.5
            scaled_hours = (n_displacements * scaled_seconds) / 3600.0
            if scaled_hours <= max_walltime_hours * 0.9:
                suggested_nodes = n
                break
        else:
            suggested_nodes = None
    else:
        suggested_nodes = None

    result: dict[str, Any] = {
        "n_atoms": n_atoms,
        "n_displacements": n_displacements,
        "seconds_per_displacement": round(seconds_per_displacement, 1),
        "estimated_total_hours": round(total_hours, 1),
        "max_walltime_hours": max_walltime_hours,
        "fits_in_walltime": fits_in_walltime,
        "safety_margin_hours": round(safety_margin, 1),
        "mpi_ranks": mpi_ranks,
        "nodes": nodes,
        "cannot_checkpoint": True,
        "warning": (
            "NWChem CANNOT checkpoint numerical frequency calculations. "
            "If the job exceeds walltime, ALL progress is lost. "
            "Ensure sufficient walltime and consider multi-node to speed up."
        ),
    }
    if suggested_nodes:
        result["suggested_nodes"] = suggested_nodes
        result["suggestion"] = (
            f"Single-node estimate is {total_hours:.0f}h which exceeds "
            f"{max_walltime_hours:.0f}h walltime. Use {suggested_nodes} nodes "
            f"({mpi_ranks * suggested_nodes} total MPI ranks) to fit within walltime."
        )
    elif not fits_in_walltime:
        result["suggestion"] = (
            f"Estimated {total_hours:.0f}h exceeds {max_walltime_hours:.0f}h walltime. "
            f"Even with multi-node scaling this may not fit. Consider analytical "
            f"frequencies (if available) or a smaller basis set."
        )

    return result


# ---------------------------------------------------------------------------
# Relativistic correction advisor
# ---------------------------------------------------------------------------

# Elements where relativistic effects are chemically significant
# 3d TMs (Z>=21): notable core-level effects; DK-basis use makes X2C appropriate
# 4d/heavy main-group (Z>=37): scalar relativistic important for energetics
# 5d metals, lanthanides, actinides (Z>=57): mandatory
_REL_SIGNIFICANT_Z = 21   # 3d transition metals — recommend when DK basis detected
_REL_IMPORTANT_Z = 37     # 4d metals and heavier — strongly recommend
_REL_CRITICAL_Z = 57      # 5d metals, lanthanides, actinides — mandatory

# DK-quality basis sets (designed for relativistic calculations)
_DK_BASIS_PATTERNS = {
    "cc-pvdz-dk", "cc-pvtz-dk", "cc-pvqz-dk", "cc-pv5z-dk",
    "aug-cc-pvdz-dk", "aug-cc-pvtz-dk", "aug-cc-pvqz-dk",
    "cc-pwcvdz-dk", "cc-pwcvtz-dk", "cc-pwcvqz-dk",
    "x2c-svpall", "x2c-tzvpall", "x2c-qzvpall",
    "dyall-v2z", "dyall-v3z", "dyall-v4z",
    "sarc-dkh2",
}

# Relativistic methods available in NWChem
_REL_METHODS = {
    "x2c": {
        "nwchem_block": "relativistic\n  x2c\nend",
        "description": "Exact Two-Component (X2C) — recommended for production quality. "
                       "Decouples large and small components exactly at the 1-electron level. "
                       "Use with DK-family basis sets (cc-pVDZ-DK, cc-pVTZ-DK, etc.).",
        "cost": "moderate",
        "suitable_for": ["single_point", "optimization", "frequency", "mp2", "ccsd"],
    },
    "dkh2": {
        "nwchem_block": "relativistic\n  douglas-kroll 2\nend",
        "description": "Douglas-Kroll-Hess 2nd order (DKH2) — widely tested, good accuracy "
                       "for 4d/5d metals. Use with DK-family basis sets.",
        "cost": "moderate",
        "suitable_for": ["single_point", "optimization", "frequency"],
    },
    "dkh3": {
        "nwchem_block": "relativistic\n  douglas-kroll 3\nend",
        "description": "Douglas-Kroll-Hess 3rd order (DKH3) — higher-order correction over DKH2. "
                       "Minimal improvement over DKH2 in most cases.",
        "cost": "moderate",
        "suitable_for": ["single_point"],
    },
    "zora": {
        "nwchem_block": "relativistic\n  zora\nend",
        "description": "ZORA (Zeroth Order Regular Approximation) — lower cost but less rigorous. "
                       "Not recommended for high-accuracy work.",
        "cost": "low",
        "suitable_for": ["single_point", "optimization"],
    },
}



__all__ = [
    "suggest_resources",
    "suggest_memory",
    "check_memory_fit",
    "estimate_freq_walltime",
]
