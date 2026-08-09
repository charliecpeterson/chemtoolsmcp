"""Exercise the charged-atom drafter against a real NWChem executable.

The check pins the omitted-multiplicity refusal, runs an explicit O+ quartet,
and records input, execution, and parsed SCF evidence under scratch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from chemtools.programs.nwchem.input.tce import draft_nwchem_atom_input
from chemtools.programs.nwchem.strategy.diagnose import parse_scf


def check_atomic_input(arguments: argparse.Namespace) -> dict[str, object]:
    scratch = arguments.scratch.expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    refusal = None
    try:
        draft_nwchem_atom_input("O", "6-31g", charge=1)
    except ValueError as error:
        refusal = str(error)
    if refusal is None:
        raise ValueError("charged atom without multiplicity was not refused")

    drafted = draft_nwchem_atom_input(
        "O",
        "6-31g",
        charge=1,
        multiplicity=4,
        output_dir=str(scratch),
        write_file=True,
    )
    input_path = Path(drafted["written_file"])
    completed = subprocess.run(
        [
            arguments.mpi_launcher,
            "-np",
            str(arguments.mpi_ranks),
            arguments.nwchem,
            input_path.name,
        ],
        text=True,
        cwd=scratch,
        capture_output=True,
        timeout=arguments.timeout,
        check=False,
    )
    output_path = scratch / "o_plus.out"
    error_path = scratch / "o_plus.err"
    output_path.write_text(completed.stdout, encoding="utf-8")
    error_path.write_text(completed.stderr, encoding="utf-8")
    scf = parse_scf(str(output_path))
    success = (
        "multiplicity is required" in refusal
        and drafted["multiplicity_source"] == "provided"
        and drafted["occupation_control"]["status"] == "unconstrained"
        and completed.returncode == 0
        and scf["status"] == "converged"
        and scf["total_energy_hartree"] is not None
    )
    return {
        "schema_version": "chemtools.nwchem-atomic-input-check/1",
        "omitted_multiplicity_refusal": refusal,
        "draft": {
            key: drafted[key]
            for key in (
                "element",
                "charge",
                "multiplicity",
                "nopen",
                "multiplicity_source",
                "occupation_control",
                "warnings",
                "written_file",
            )
        },
        "execution": {
            "mpi_ranks": arguments.mpi_ranks,
            "return_code": completed.returncode,
            "stdout": str(output_path),
            "stderr": str(error_path),
        },
        "scf": scf,
        "success": success,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--nwchem", default="/apps/nwchem/bin/nwchem")
    parser.add_argument("--mpi-launcher", default="mpirun")
    parser.add_argument("--mpi-ranks", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    arguments = parser.parse_args()
    evidence = check_atomic_input(arguments)
    evidence_path = arguments.scratch.expanduser().resolve() / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "evidence": str(evidence_path),
        "return_code": evidence["execution"]["return_code"],
        "scf_status": evidence["scf"]["status"],
        "total_energy_hartree": evidence["scf"]["total_energy_hartree"],
        "success": evidence["success"],
    }, sort_keys=True))
    return 0 if evidence["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
