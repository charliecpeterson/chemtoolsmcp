"""Exercise the operator smoke runner with a bounded fake NWChem process."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import sys

from scripts.smoke_nwchem_execution import run_smoke


def test_smoke_runner_records_terminal_scientific_evidence(tmp_path: Path):
    fake_nwchem = tmp_path / "fake_nwchem.py"
    fake_nwchem.write_text(
        "print('NWChem Input Module')\n"
        "print('Starting SCF solution at 0.0')\n"
        "print('Total SCF energy = -1.116759310191')\n"
        "print('Task  times  cpu: 0.0s wall: 0.1s')\n",
        encoding="utf-8",
    )
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "profiles": {
                "smoke": {
                    "launcher": {"kind": "direct"},
                    "programs": {
                        "nwchem": {
                            "launcher_argv": [sys.executable],
                            "executable_argv": [str(fake_nwchem)],
                        },
                    },
                    "resources": {
                        "mpi_ranks": 1,
                        "omp_threads": 1,
                    },
                },
            },
        }),
        encoding="utf-8",
    )

    evidence_path, success = run_smoke(Namespace(
        profiles_path=profiles_path,
        profile="smoke",
        expect_executor="local",
        work_root=tmp_path / "runs",
        mpi_ranks=1,
        walltime="00:05:00",
        timeout_seconds=10.0,
        poll_interval_seconds=0.01,
    ))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    stdout = (
        "NWChem Input Module\n"
        "Starting SCF solution at 0.0\n"
        "Total SCF energy = -1.116759310191\n"
        "Task  times  cpu: 0.0s wall: 0.1s\n"
    ).encode()

    assert success is True
    assert evidence["success"] is True
    assert evidence["execution_record"]["status"] == "completed"
    assert evidence["execution_record"]["return_code"] == 0
    assert evidence["watch"]["overall_status"] == "completed_success"
    assert evidence["scientific_check"] == {
        "iteration_count": 0,
        "scf_status": "converged",
        "total_energy_hartree": -1.116759310191,
    }
    assert evidence["artifacts"]["stdout"]["size_bytes"] == len(stdout)
    assert evidence["artifacts"]["stdout"]["sha256"] == (
        hashlib.sha256(stdout).hexdigest()
    )
