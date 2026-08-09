"""Inspect local hardware and scheduler partition resources."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from typing import Any


_PARTITION_SPECS_CACHE: dict[str, dict[str, Any]] = {}


def query_partition_specs(
    partition: str,
    scheduler_type: str,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Query scheduler node memory, CPU count, architecture, and features."""
    effective_cache = cache if cache is not None else _PARTITION_SPECS_CACHE
    if partition in effective_cache:
        return effective_cache[partition]

    partition_specs: dict[str, Any] = {
        "node_memory_mb": None,
        "cpus_per_node": None,
        "cpu_arch": "generic",
        "features": [],
    }

    if scheduler_type == "slurm":
        if not shutil.which("sinfo"):
            return partition_specs
        try:
            completed = subprocess.run(
                [
                    "sinfo",
                    "-p",
                    partition,
                    "-o",
                    "%m %c %f",
                    "--noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = [
                line.strip()
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
            if not lines:
                return partition_specs
            rows = [line.split(None, 2) for line in lines]
            minimum_memory = min(
                int(row[0]) for row in rows if row[0].isdigit()
            )
            minimum_cpus = min(
                int(row[1])
                for row in rows
                if len(row) > 1 and row[1].isdigit()
            )
            features = set()
            for row in rows:
                if len(row) > 2:
                    features.update(row[2].split(","))
            architecture = (
                "spr"
                if "spr" in features
                else "skx"
                if "skx" in features
                else "knl"
                if "knl" in features
                else "generic"
            )
            partition_specs = {
                "node_memory_mb": minimum_memory,
                "cpus_per_node": minimum_cpus,
                "cpu_arch": architecture,
                "features": sorted(features),
            }
        except Exception:
            pass

    elif scheduler_type == "pbs":
        if not shutil.which("pbsnodes"):
            return partition_specs
        try:
            completed = subprocess.run(
                ["pbsnodes", "-a"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            memory_match = re.search(
                r"resources_available\.mem\s*=\s*(\d+)kb",
                completed.stdout,
                re.IGNORECASE,
            )
            cpu_match = re.search(
                r"resources_available\.ncpus\s*=\s*(\d+)",
                completed.stdout,
                re.IGNORECASE,
            )
            if memory_match:
                partition_specs["node_memory_mb"] = (
                    int(memory_match.group(1)) // 1024
                )
            if cpu_match:
                partition_specs["cpus_per_node"] = int(cpu_match.group(1))
        except Exception:
            pass

    effective_cache[partition] = partition_specs
    return partition_specs


def get_local_resource_budget() -> dict[str, Any]:
    """Return available CPU cores and memory on the local machine."""
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or 1
        load_1min = psutil.getloadavg()[0]
        cores_in_use = min(int(load_1min + 0.5), physical_cores - 1)
        available_cores = max(1, physical_cores - cores_in_use)
        memory = psutil.virtual_memory()
        return {
            "physical_cores": physical_cores,
            "available_cores": available_cores,
            "current_load_1min": load_1min,
            "total_mem_mb": int(memory.total / 1_000_000),
            "available_mem_mb": int(memory.available / 1_000_000 * 0.85),
            "cpu_arch": _detect_local_cpu_arch(),
        }
    except ImportError:
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
    """Detect AVX-512, AVX2, ARM, or generic local CPU support."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            flags = handle.read()
        if "avx512f" in flags:
            return "avx512"
        if "avx2" in flags:
            return "avx2"
    except OSError:
        pass
    machine = platform.machine().lower()
    return "arm" if "arm" in machine or "aarch" in machine else "generic"


__all__ = ["get_local_resource_budget", "query_partition_specs"]
