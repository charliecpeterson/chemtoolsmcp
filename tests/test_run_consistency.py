"""Exact input-output consistency contracts for guided run inspection."""

from __future__ import annotations

from pathlib import Path

from chemtools.application.run_inspection import inspect_run
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend
from chemtools.programs.nwchem.basis_consistency import (
    compare_basis_coverage,
)
from chemtools.programs.nwchem.electron_consistency import (
    normalize_wavefunction_class,
)
from chemtools.programs.nwchem.task_consistency import normalize_operation
from chemtools.programs.nwchem.xc_consistency import (
    canonical_xc_alias,
    compare_xc_functional,
)


FIXTURES = Path(__file__).parent / "golden" / "mcp" / "fixtures"


def test_nwchem_raman_operation_normalizes_to_frequency_task_kind():
    assert normalize_operation("raman") == "frequency"


def test_nwchem_task_operation_check_uses_resolved_default(tmp_path):
    input_path = tmp_path / "implicit-energy.nw"
    output_path = tmp_path / "implicit-energy.out"
    input_path.write_text(
        (
            "start implicit\n"
            "geometry units angstrom\n"
            "  H 0 0 0\n"
            "end\n"
            "task scf\n"
            "property; mulliken; end\n"
            "task scf property\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem Input Module\n"
            "NWChem SCF Module\n"
            "Total SCF energy = -0.5\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
            "NWChem Input Module\n"
            "NWChem Property Module\n"
            "NWChem SCF Module\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )
    checks = inspected["evidence"]["input_output_consistency"]["checks"]
    task_operations = next(
        check for check in checks if check["field"] == "task_operations"
    )

    assert task_operations == {
        "field": "task_operations",
        "status": "match",
        "input": ["energy", "property"],
        "output": ["energy", "property"],
    }


def test_nwchem_task_operation_check_abstains_before_module(tmp_path):
    input_path = tmp_path / "input-error.nw"
    output_path = tmp_path / "input-error.out"
    input_path.write_text(
        "start broken\ntask mm optimize\n",
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem Input Module\n"
            "There is an error in the input file\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )
    checks = inspected["evidence"]["input_output_consistency"]["checks"]
    task_operations = next(
        check for check in checks if check["field"] == "task_operations"
    )

    assert task_operations == {
        "field": "task_operations",
        "status": "not_checked",
        "reason": (
            "Input and output did not expose the same number of comparable "
            "task operations."
        ),
        "input": ["optimize"],
        "output": ["other"],
    }


def test_nwchem_wavefunction_labels_normalize_to_reference_class():
    assert normalize_wavefunction_class("RHF") == "closed_shell"
    assert normalize_wavefunction_class("closed shell.") == "closed_shell"
    for label in (
        "ROHF",
        "UHF",
        "ODFT",
        "RODFT",
        "open shell",
        "spin polarized.",
    ):
        assert normalize_wavefunction_class(label) == "open_shell"


def test_nwchem_supported_xc_aliases_normalize_conservatively():
    assert canonical_xc_alias("B3LYP") == "b3lyp"
    assert canonical_xc_alias("BHLYP") == "bhlyp"
    assert canonical_xc_alias("M06-2X") == "m06-2x"
    assert canonical_xc_alias("pbe0") == "pbe0"
    assert canonical_xc_alias("SCAN") == "scan"
    assert canonical_xc_alias("xpbe96") is None


def test_nwchem_component_xc_expression_is_not_alias_compared():
    comparison = compare_xc_functional(
        {
            "module": "dft",
            "xc": {
                "name": None,
                "tokens": ["xpbe96", "cpbe96"],
                "source": "explicit_expression",
            },
        },
        {
            "xc_functional_labels": ["PBE0"],
            "xc_functional_names": ["pbe0"],
        },
    )

    assert comparison["status"] == "not_checked"
    assert comparison["reason"] == (
        "The explicit XC expression is not a supported named alias."
    )


def test_nwchem_sparse_output_compares_only_available_task_evidence():
    output_path = FIXTURES / "nwchem_scf.out"
    input_path = FIXTURES / "nwchem_h2.nw"

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    assert inspected["evidence"]["input_output_consistency"] == {
        "status": "checked",
        "input_path": str(input_path.resolve()),
        "summary": {
            "match": 2,
            "mismatch": 0,
            "not_checked": 6,
        },
        "checks": [
            {
                "field": "input_deck",
                "status": "not_checked",
                "reason": (
                    "The output contains no complete NWChem input-deck echo."
                ),
            },
            {
                "field": "task_methods",
                "status": "match",
                "input": ["SCF"],
                "output": ["SCF"],
            },
            {
                "field": "task_operations",
                "status": "match",
                "input": ["energy"],
                "output": ["energy"],
            },
            {
                "field": "charge",
                "status": "not_checked",
                "reason": (
                    "No single explicit input charge and output charge were "
                    "available."
                ),
                "input": [],
                "output": [],
            },
            {
                "field": "multiplicity",
                "status": "not_checked",
                "reason": (
                    "No single explicit input multiplicity and output "
                    "multiplicity were available."
                ),
                "input": [1],
                "output": [],
            },
            {
                "field": "atom_count",
                "status": "not_checked",
                "reason": (
                    "A single input geometry and a single output atom count "
                    "were not both available."
                ),
                "input": 2,
                "output": [],
            },
            {
                "field": "geometry",
                "status": "not_checked",
                "reason": (
                    "The output contains no complete coordinate table."
                ),
            },
            {
                "field": "restart_artifacts",
                "status": "not_checked",
                "reason": (
                    "The input declares no external restart artifacts."
                ),
            },
        ],
    }
    assert inspected["uncertainty"] == []


def test_nwchem_echo_and_general_information_match(tmp_path):
    input_path = tmp_path / "h2.nw"
    output_path = tmp_path / "h2.out"
    input_text = (
        "start h2\n"
        "geometry units angstroms\n"
        "  H 0.0 0.0 0.0\n"
        "  H 0.0 0.0 0.74\n"
        "end\n"
        "charge 0\n"
        "dft\n"
        "  mult 1\n"
        "end\n"
        "task dft energy\n"
    )
    input_path.write_text(input_text, encoding="utf-8")
    output_path.write_text(
        (
            "================ echo of input deck ================\n"
            f"{input_text}"
            "====================================================\n"
            "Northwest Computational Chemistry Package (NWChem) 7.2.2\n"
            "NWChem DFT Module\n"
            "No. of atoms     : 2\n"
            "Charge           : 0\n"
            "Spin multiplicity: 1\n"
            "Output coordinates in angstroms\n"
            "No. Tag Charge X Y Z\n"
            "1 H 1.0 -0.37000000 0.00000000 0.00000000\n"
            "2 H 1.0  0.37000000 0.00000000 0.00000000\n"
            "Atomic Mass\n"
            "Total DFT energy = -1.1000000000\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 7,
        "mismatch": 0,
        "not_checked": 1,
    }
    assert [
        (check["field"], check["status"])
        for check in consistency["checks"]
    ] == [
        ("input_deck", "match"),
        ("task_methods", "match"),
        ("task_operations", "match"),
        ("charge", "match"),
        ("multiplicity", "match"),
        ("atom_count", "match"),
        ("geometry", "match"),
        ("restart_artifacts", "not_checked"),
    ]
    assert consistency["checks"][0]["input"] == {
        "normalized_sha256": (
            "a40c5aaf62e303c4a08da278b7fe106c59f9eacc5f2412cf26b82ff8e2f7dd3c"
        ),
    }
    assert consistency["checks"][0]["output"] == (
        consistency["checks"][0]["input"]
    )
    assert consistency["checks"][6] == {
        "field": "geometry",
        "status": "match",
        "input": {
            "atom_count": 2,
            "elements": ["H", "H"],
            "source_units": "angstrom",
        },
        "output": {
            "atom_count": 2,
            "elements": ["H", "H"],
            "source_units": "angstrom",
        },
        "basis": (
            "Element order and all pair distances in the first complete "
            "output coordinate table."
        ),
        "metrics": {
            "pair_count": 1,
            "max_pair_distance_delta_angstrom": 0.0,
            "tolerance_angstrom": 1e-05,
        },
    }
    assert inspected["uncertainty"] == []


def test_nwchem_single_task_electron_count_mismatch(tmp_path):
    input_path = tmp_path / "water.nw"
    output_path = tmp_path / "water.out"
    input_path.write_text(
        (
            "start water\n"
            "geometry units angstrom\n"
            "  O 0 0 0\n"
            "  H 0 0 1\n"
            "  H 0 1 0\n"
            "end\n"
            "charge 0\n"
            "dft; singlet; end\n"
            "task dft energy\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem DFT Module\n"
            "Wavefunction type: closed shell.\n"
            "No. of electrons : 9\n"
            "Alpha electrons  : 5\n"
            "Beta electrons   : 4\n"
            "Spin multiplicity: 1\n"
            "Total DFT energy = -75.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    electron_count = next(
        check
        for check in inspected["evidence"]["input_output_consistency"]["checks"]
        if check["field"] == "electron_count"
    )
    assert electron_count == {
        "field": "electron_count",
        "status": "mismatch",
        "input": {
            "expected": 10,
            "effective_nuclear_charge": 10,
            "molecular_charge": 0,
            "ecp_core_electrons": {},
        },
        "output": 9,
        "basis": (
            "Sum of effective nuclear charges after explicit ECP core "
            "replacement, minus molecular charge."
        ),
    }
    electron_spin_parity = next(
        check
        for check in inspected["evidence"]["input_output_consistency"]["checks"]
        if check["field"] == "electron_spin_parity"
    )
    assert electron_spin_parity["status"] == "mismatch"
    assert electron_spin_parity["input"]["compatible"] is True
    assert electron_spin_parity["output"]["compatible"] is False
    spin_occupations = next(
        check
        for check in inspected["evidence"]["input_output_consistency"]["checks"]
        if check["field"] == "spin_occupations"
    )
    assert spin_occupations["status"] == "mismatch"
    wavefunction_class = next(
        check
        for check in inspected["evidence"]["input_output_consistency"]["checks"]
        if check["field"] == "wavefunction_class"
    )
    assert wavefunction_class["status"] == "match"


def test_nwchem_basis_mode_and_explicit_ecp_replacement_checks(tmp_path):
    input_path = tmp_path / "thorium.nw"
    output_path = tmp_path / "thorium.out"
    input_path.write_text(
        (
            "start thorium\n"
            "geometry units angstrom\n"
            "  Th 0 0 0\n"
            "end\n"
            "basis spherical\n"
            "  Th library stuttgart\n"
            "end\n"
            "ecp\n"
            "  Th nelec 60\n"
            "end\n"
            "charge 0\n"
            "dft; singlet; end\n"
            "task dft energy\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            'Summary of "ao basis" -> "" (cartesian)\n'
            "------------------------------------------------------------------------------\n"
            "      Tag                 Description            Shells   Functions and Types\n"
            "---------------- ------------------------------  ------  ---------------------\n"
            "*                       user specified             on all atoms\n"
            "\n"
            'Summary of "ao basis" -> "ao basis" (cartesian)\n'
            "------------------------------------------------------------------------------\n"
            "      Tag                 Description            Shells   Functions and Types\n"
            "---------------- ------------------------------  ------  ---------------------\n"
            "Th1                    user specified             25       87   8s7p6d4f\n"
            "\n"
            'Summary of "ao basis" -> "ao basis" (cartesian)\n'
            "------------------------------------------------------------------------------\n"
            "      Tag                 Description            Shells   Functions and Types\n"
            "---------------- ------------------------------  ------  ---------------------\n"
            "Th1                    user specified             25       87   8s7p6d4f\n"
            "\n"
            "Th1 (Thorium) Replaces 60 electrons\n"
            "NWChem DFT Module\n"
            "AO basis - number of functions: 87\n"
            "           number of shells: 25\n"
            "No. of electrons : 30\n"
            "Total DFT energy = -1.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    checks = {
        check["field"]: check
        for check in inspected["evidence"]["input_output_consistency"][
            "checks"
        ]
    }
    assert checks["basis_mode"] == {
        "field": "basis_mode",
        "status": "mismatch",
        "input": {
            "name": "ao basis",
            "mode": "spherical",
            "mode_source": "explicit",
            "source": "input",
        },
        "output": {
            "mode": "cartesian",
            "function_counts": [87],
            "shell_counts": [25],
        },
        "basis": (
            "Input AO basis spherical/Cartesian selection compared with "
            "the runtime basis summary; shell and function counts are "
            "reported as output evidence."
        ),
    }
    assert checks["basis_coverage"] == {
        "field": "basis_coverage",
        "status": "match",
        "input": {"elements": ["Th"]},
        "output": {
            "elements": ["Th"],
            "unparsed_tags": [],
            "summaries": [{
                "mode": "cartesian",
                "rows": [{
                    "tag": "Th1",
                    "element": "Th",
                    "description": "user specified",
                    "shells": 25,
                    "functions": 87,
                    "types": "8s7p6d4f",
                }],
            }],
        },
        "missing_elements": [],
        "basis": (
            "Elements in the selected input geometry compared with "
            "the runtime AO basis tag rows; per-tag shell and function "
            "counts are reported as output evidence."
        ),
    }
    assert checks["ecp_replacements"] == {
        "field": "ecp_replacements",
        "status": "match",
        "input": {"Th": 60},
        "output": {"Th": 60},
        "basis": (
            "Explicit or bundled-library nelec values compared with "
            "NWChem's printed ECP electron replacements."
        ),
    }


def test_nwchem_basis_coverage_reports_missing_geometry_element():
    comparison = compare_basis_coverage(
        {"elements": ["Th", "F", "F"]},
        {
            "basis_summaries": [{
                "mode": "spherical",
                "rows": [{
                    "tag": "F",
                    "element": "F",
                    "description": "cc-pVDZ",
                    "shells": 6,
                    "functions": 14,
                    "types": "3s2p1d",
                }],
            }],
        },
    )

    assert comparison["status"] == "mismatch"
    assert comparison["input"] == {"elements": ["Th", "F"]}
    assert comparison["output"]["elements"] == ["F"]
    assert comparison["missing_elements"] == ["Th"]


def test_nwchem_basis_coverage_abstains_for_unmapped_tag():
    comparison = compare_basis_coverage(
        {"elements": ["Th"]},
        {
            "basis_summaries": [{
                "mode": "spherical",
                "rows": [{
                    "tag": "metal",
                    "element": None,
                    "description": "user specified",
                    "shells": 25,
                    "functions": 87,
                    "types": "8s7p6d4f",
                }],
            }],
        },
    )

    assert comparison["status"] == "not_checked"
    assert comparison["output"]["unparsed_tags"] == ["metal"]
    assert comparison["missing_elements"] == ["Th"]


def test_nwchem_task_xc_functionals_follow_sequential_dft_state(tmp_path):
    input_path = tmp_path / "functionals.nw"
    output_path = tmp_path / "functionals.out"
    input_path.write_text(
        (
            "start functionals\n"
            "geometry units angstrom\n"
            "  H 0 0 0\n"
            "end\n"
            "dft; xc pbe0; end\n"
            "task dft energy\n"
            "dft; mult 1; end\n"
            "task dft energy\n"
            "dft; xc b3lyp; end\n"
            "task dft energy\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem Input Module\n"
            "NWChem DFT Module\n"
            "PBE0 Method XC Functional\n"
            "Total DFT energy = -1.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
            "NWChem Input Module\n"
            "NWChem DFT Module\n"
            "B3LYP Method XC Potential\n"
            "Total DFT energy = -2.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
            "NWChem Input Module\n"
            "NWChem DFT Module\n"
            "B3LYP Method XC Potential\n"
            "Total DFT energy = -3.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )
    task_states = next(
        check
        for check in inspected["evidence"]["input_output_consistency"][
            "checks"
        ]
        if check["field"] == "task_states"
    )
    comparisons = [
        task["comparisons"]["xc_functional"]
        for task in task_states["tasks"]
    ]

    assert [comparison["status"] for comparison in comparisons] == [
        "match",
        "mismatch",
        "match",
    ]
    assert [comparison["input"]["name"] for comparison in comparisons] == [
        "pbe0",
        "pbe0",
        "b3lyp",
    ]
    assert [comparison["output"]["name"] for comparison in comparisons] == [
        "pbe0",
        "b3lyp",
        "b3lyp",
    ]


def test_nwchem_tddft_and_dplot_keep_top_level_task_boundaries(tmp_path):
    input_path = tmp_path / "h2o-td.nw"
    output_path = tmp_path / "h2o-td.out"
    input_path.write_text(
        (
            "start h2o-td\n"
            "geometry units angstrom\n"
            "  O 0 0 0\n"
            "end\n"
            "dft; xc bhlyp; end\n"
            "tddft; nroots 5; end\n"
            "task tddft energy\n"
            "dplot; output root-2.cube; end\n"
            "task dplot\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem Input Module\n"
            "Total DFT energy = -76.318598406731\n"
            "NWChem TDDFT Module\n"
            "BHLYP Method XC Functional\n"
            "Excited state energy = -76.011296959418\n"
            "Task times cpu: 11.8s wall: 13.9s\n"
            "NWChem Input Module\n"
            "Limits (a.u.) specified for the density plot:\n"
            "Output is written to : root-2.cube\n"
            "Task times cpu: 0.7s wall: 0.7s\n"
        ),
        encoding="utf-8",
    )

    backend = load_backend(BUILTIN_BACKENDS[0])
    parsed = backend.parser.parse_output(str(output_path))
    inspected = inspect_run(
        backend,
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )
    task_states = next(
        check
        for check in inspected["evidence"]["input_output_consistency"][
            "checks"
        ]
        if check["field"] == "task_states"
    )

    assert [task["kind"] for task in parsed["tasks"]] == [
        "energy",
        "property",
    ]
    assert [task["method"] for task in parsed["tasks"]] == [
        "TDDFT",
        "DPLOT",
    ]
    assert [task["line_range"] for task in parsed["tasks"]] == [
        (1, 6),
        (7, 10),
    ]
    assert parsed["tasks"][0]["energy_hartree"] == -76.011296959418
    assert task_states["status"] == "match"
    assert [
        task["operation"] for task in task_states["tasks"]
    ] == ["energy", "property"]
    assert task_states["tasks"][0]["comparisons"]["xc_functional"] == {
        "status": "match",
        "input": {
            "name": "bhlyp",
            "tokens": ["bhlyp"],
            "source": "explicit_alias",
        },
        "output": {
            "name": "bhlyp",
            "labels": ["BHLYP"],
        },
    }


def test_nwchem_tce_task_pairs_with_explicit_correlated_method(tmp_path):
    input_path = tmp_path / "correlated.nw"
    output_path = tmp_path / "correlated.out"
    input_path.write_text(
        (
            "start correlated\n"
            "geometry units angstrom\n"
            "  N 0 0 0\n"
            "  N 0 0 1.1\n"
            "end\n"
            "task scf energy\n"
            "tce; ccsd; end\n"
            "task tce energy\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem Input Module\n"
            "NWChem SCF Module\n"
            "Total SCF energy = -108.9\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
            "NWChem Input Module\n"
            "CCSD total energy / hartree = -109.2\n"
            "Task times cpu: 0.2s wall: 0.2s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )
    checks = inspected["evidence"]["input_output_consistency"]["checks"]
    task_methods = next(
        check for check in checks if check["field"] == "task_methods"
    )
    task_states = next(
        check for check in checks if check["field"] == "task_states"
    )

    assert task_methods == {
        "field": "task_methods",
        "status": "match",
        "input": ["SCF", "CCSD"],
        "output": ["SCF", "CCSD"],
    }
    assert [task["module"] for task in task_states["tasks"]] == [
        "scf",
        "tce",
    ]
    assert task_states["tasks"][1]["input_state"]["method"] == "CCSD"
    assert task_states["tasks"][1]["input_state"]["method_source"] == (
        "explicit_tce_keyword"
    )


def test_nwchem_task_electron_counts_include_explicit_ecp_cores(tmp_path):
    input_path = tmp_path / "thf4.nw"
    output_path = tmp_path / "thf4.out"
    input_path.write_text(
        (
            "start thf4\n"
            "geometry metal units angstrom\n"
            "  Th 0 0 0\n"
            "end\n"
            "geometry ligands units angstrom\n"
            "  F  2 0 0\n"
            "  F -2 0 0\n"
            "  F 0  2 0\n"
            "  F 0 -2 0\n"
            "end\n"
            "geometry complex units angstrom\n"
            "  Th 0 0 0\n"
            "  F  2 0 0\n"
            "  F -2 0 0\n"
            "  F 0  2 0\n"
            "  F 0 -2 0\n"
            "end\n"
            "ecp\n"
            "  Th nelec 60\n"
            "end\n"
            "dft; singlet; end\n"
            "set geometry metal\n"
            "charge 4\n"
            "task dft energy\n"
            "set geometry ligands\n"
            "charge -4\n"
            "task dft energy\n"
            "set geometry complex\n"
            "charge 0\n"
            "task dft energy\n"
            "set geometry metal\n"
            "charge 3\n"
            "dft; doublet; end\n"
            "task dft energy\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem Input Module\n"
            "NWChem DFT Module\n"
            "Wavefunction type: closed shell.\n"
            "No. of electrons : 26\n"
            "Alpha electrons  : 13\n"
            "Beta electrons   : 13\n"
            "Spin multiplicity: 1\n"
            "Total DFT energy = -1.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
            "NWChem Input Module\n"
            "NWChem DFT Module\n"
            "Wavefunction type: closed shell.\n"
            "No. of electrons : 40\n"
            "Alpha electrons  : 20\n"
            "Beta electrons   : 20\n"
            "Spin multiplicity: 1\n"
            "Total DFT energy = -2.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
            "NWChem Input Module\n"
            "NWChem DFT Module\n"
            "Wavefunction type: closed shell.\n"
            "No. of electrons : 66\n"
            "Alpha electrons  : 33\n"
            "Beta electrons   : 33\n"
            "Spin multiplicity: 1\n"
            "Total DFT energy = -3.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
            "NWChem Input Module\n"
            "NWChem DFT Module\n"
            "Wavefunction type: spin polarized.\n"
            "No. of electrons : 27\n"
            "Alpha electrons  : 14\n"
            "Beta electrons   : 13\n"
            "Spin multiplicity: 2\n"
            "Total DFT energy = -4.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    task_states = next(
        check
        for check in inspected["evidence"]["input_output_consistency"]["checks"]
        if check["field"] == "task_states"
    )
    assert [
        task["comparisons"]["electron_count"]["input"]["expected"]
        for task in task_states["tasks"]
    ] == [26, 40, 66, 27]
    assert [
        task["comparisons"]["electron_count"]["status"]
        for task in task_states["tasks"]
    ] == ["match", "match", "match", "match"]
    assert [
        task["comparisons"]["electron_spin_parity"]["status"]
        for task in task_states["tasks"]
    ] == ["match", "match", "match", "match"]
    assert [
        task["comparisons"]["spin_occupations"]["status"]
        for task in task_states["tasks"]
    ] == ["match", "match", "match", "match"]
    assert [
        task["comparisons"]["wavefunction_class"]["status"]
        for task in task_states["tasks"]
    ] == ["match", "match", "match", "match"]


def test_nwchem_input_electron_spin_parity_mismatch(tmp_path):
    input_path = tmp_path / "hydrogen.nw"
    output_path = tmp_path / "hydrogen.out"
    input_path.write_text(
        (
            "start hydrogen\n"
            "geometry units angstrom\n"
            "  H 0 0 0\n"
            "end\n"
            "charge 0\n"
            "dft; singlet; end\n"
            "task dft energy\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem DFT Module\n"
            "Wavefunction type: spin polarized.\n"
            "No. of electrons : 1\n"
            "Alpha electrons  : 1\n"
            "Beta electrons   : 0\n"
            "Spin multiplicity: 1\n"
            "Total DFT energy = -0.5\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    electron_spin_parity = next(
        check
        for check in inspected["evidence"]["input_output_consistency"]["checks"]
        if check["field"] == "electron_spin_parity"
    )
    assert electron_spin_parity["status"] == "mismatch"
    assert electron_spin_parity["input"] == {
        "status": "checked",
        "electron_count": 1,
        "multiplicity": 1,
        "compatible": False,
    }
    assert electron_spin_parity["output"] == electron_spin_parity["input"]
    spin_occupations = next(
        check
        for check in inspected["evidence"]["input_output_consistency"]["checks"]
        if check["field"] == "spin_occupations"
    )
    assert spin_occupations["status"] == "mismatch"
    wavefunction_class = next(
        check
        for check in inspected["evidence"]["input_output_consistency"]["checks"]
        if check["field"] == "wavefunction_class"
    )
    assert wavefunction_class["status"] == "mismatch"


def test_nwchem_electron_count_abstains_for_unresolved_ecp_library(tmp_path):
    input_path = tmp_path / "thorium.nw"
    output_path = tmp_path / "thorium.out"
    input_path.write_text(
        (
            "start thorium\n"
            "geometry units angstrom\n"
            "  Th 0 0 0\n"
            "end\n"
            "ecp\n"
            "  Th library stuttgart\n"
            "end\n"
            "charge 0\n"
            "dft; singlet; end\n"
            "task dft energy\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem DFT Module\n"
            "No. of electrons : 30\n"
            "Total DFT energy = -1.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    electron_count = next(
        check
        for check in inspected["evidence"]["input_output_consistency"]["checks"]
        if check["field"] == "electron_count"
    )
    assert electron_count["status"] == "not_checked"
    assert electron_count["reason"] == (
        "The active ECP library core-electron replacements are unresolved."
    )
    assert electron_count["input"]["unresolved_elements"] == ["Th"]
    assert electron_count["output"] == [30]


def test_nwchem_resolves_bundled_library_ecp_core_count(tmp_path):
    input_path = tmp_path / "thorium.nw"
    output_path = tmp_path / "thorium.out"
    input_path.write_text(
        (
            "start thorium\n"
            "geometry units angstrom\n"
            "  Th 0 0 0\n"
            "end\n"
            "ecp\n"
            "  Th library stuttgart_rsc_1997_ecp\n"
            "end\n"
            "charge 0\n"
            "dft; singlet; end\n"
            "task dft energy\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "Th (Thorium) Replaces 60 electrons\n"
            "NWChem DFT Module\n"
            "No. of electrons : 30\n"
            "Total DFT energy = -1.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    checks = {
        check["field"]: check
        for check in inspected["evidence"]["input_output_consistency"][
            "checks"
        ]
    }
    assert checks["electron_count"]["status"] == "match"
    assert checks["electron_count"]["input"]["expected"] == 30
    assert checks["electron_count"]["input"]["ecp_core_electrons"] == {
        "Th": 60,
    }
    source = checks["electron_count"]["input"][
        "ecp_core_electron_sources"
    ]["Th"]
    assert source["kind"] == "bundled_nwchem_library"
    assert source["family"] == "stuttgart_rsc_1997_ecp"
    assert source["file"].endswith(
        "basis_library/stuttgart_rsc_1997_ecp"
    )
    assert checks["ecp_replacements"]["status"] == "match"
    assert checks["ecp_replacements"]["input"] == {"Th": 60}
    assert checks["ecp_replacements"]["output"] == {"Th": 60}


def test_nwchem_named_ecp_selection_changes_task_electron_count(tmp_path):
    input_path = tmp_path / "thorium.nw"
    output_path = tmp_path / "thorium.out"
    input_path.write_text(
        (
            "start thorium\n"
            "geometry units angstrom\n"
            "  Th 0 0 0\n"
            "end\n"
            'ecp "small core"\n'
            "  Th nelec 60\n"
            "end\n"
            'ecp "large core"\n'
            "  Th nelec 78\n"
            "end\n"
            "charge 0\n"
            "dft; singlet; end\n"
            'set "ecp basis" "small core"\n'
            "task dft energy\n"
            'set "ecp basis" "large core"\n'
            "task dft energy\n"
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        (
            "NWChem Input Module\n"
            "Th (Thorium) Replaces 60 electrons\n"
            "NWChem DFT Module\n"
            "No. of atoms : 1\n"
            "No. of electrons : 30\n"
            "Charge : 0\n"
            "Spin multiplicity: 1\n"
            "Total DFT energy = -1.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
            "NWChem Input Module\n"
            "Th (Thorium) Replaces 78 electrons\n"
            "NWChem DFT Module\n"
            "No. of atoms : 1\n"
            "No. of electrons : 12\n"
            "Charge : 0\n"
            "Spin multiplicity: 1\n"
            "Total DFT energy = -2.0\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    task_states = next(
        check
        for check in inspected["evidence"]["input_output_consistency"][
            "checks"
        ]
        if check["field"] == "task_states"
    )
    assert [
        task["comparisons"]["electron_count"]["input"]["expected"]
        for task in task_states["tasks"]
    ] == [30, 12]
    assert [
        task["comparisons"]["electron_count"]["status"]
        for task in task_states["tasks"]
    ] == ["match", "match"]
    assert [
        task["comparisons"]["ecp_replacements"]["status"]
        for task in task_states["tasks"]
    ] == ["match", "match"]


def test_nwchem_observed_charge_excludes_echoed_input_charge(tmp_path):
    input_path = tmp_path / "charged.nw"
    output_path = tmp_path / "charged.out"
    input_text = (
        "start charged\n"
        "geometry\n"
        "  H 0 0 0\n"
        "end\n"
        "charge 0\n"
        "scf\n"
        "  singlet\n"
        "end\n"
        "task scf energy\n"
    )
    input_path.write_text(input_text, encoding="utf-8")
    output_path.write_text(
        (
            "================ echo of input deck ================\n"
            f"{input_text}"
            "====================================================\n"
            "Northwest Computational Chemistry Package (NWChem) 7.2.2\n"
            "NWChem SCF Module\n"
            "No. of atoms     : 1\n"
            "Charge           : 1\n"
            "Spin multiplicity: 1\n"
            "Total SCF energy = -0.5000000000\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {
        "match": 5,
        "mismatch": 1,
        "not_checked": 2,
    }
    assert consistency["checks"][0]["status"] == "match"
    assert consistency["checks"][3] == {
        "field": "charge",
        "status": "mismatch",
        "input": 0,
        "output": 1,
    }
    assert inspected["uncertainty"] == [{
        "code": "input_output_mismatch",
        "message": (
            "The explicit input disagrees with output evidence for: charge."
        ),
        "impact": (
            "Verify that the supplied input and related restart files belong "
            "to this output."
        ),
    }]


def test_nwchem_geometry_distance_mismatch_is_reported(tmp_path):
    input_path = tmp_path / "h2.nw"
    output_path = tmp_path / "h2.out"
    input_text = (
        "start h2\n"
        "geometry units angstrom\n"
        "  H 0.0 0.0 0.0\n"
        "  H 0.0 0.0 0.74\n"
        "end\n"
        "charge 0\n"
        "scf\n"
        "  singlet\n"
        "end\n"
        "task scf energy\n"
    )
    input_path.write_text(input_text, encoding="utf-8")
    output_path.write_text(
        (
            "================ echo of input deck ================\n"
            f"{input_text}"
            "====================================================\n"
            "Northwest Computational Chemistry Package (NWChem) 7.2.2\n"
            "NWChem SCF Module\n"
            "No. of atoms     : 2\n"
            "Charge           : 0\n"
            "Spin multiplicity: 1\n"
            "Output coordinates in angstroms\n"
            "No. Tag Charge X Y Z\n"
            "1 H 1.0 -0.40000000 0.00000000 0.00000000\n"
            "2 H 1.0  0.40000000 0.00000000 0.00000000\n"
            "Atomic Mass\n"
            "Total SCF energy = -1.1000000000\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {
        "match": 6,
        "mismatch": 1,
        "not_checked": 1,
    }
    assert consistency["checks"][6]["field"] == "geometry"
    assert consistency["checks"][6]["status"] == "mismatch"
    assert consistency["checks"][6]["metrics"] == {
        "pair_count": 1,
        "max_pair_distance_delta_angstrom": 0.06,
        "tolerance_angstrom": 1e-05,
    }
    assert inspected["uncertainty"] == [{
        "code": "input_output_mismatch",
        "message": (
            "The explicit input disagrees with output evidence for: geometry."
        ),
        "impact": (
            "Verify that the supplied input and related restart files belong "
            "to this output."
        ),
    }]


def test_nwchem_task_states_find_multi_state_charge_mismatch(tmp_path):
    input_path = tmp_path / "states.nw"
    output_path = tmp_path / "states.out"
    input_text = (
        "start states\n"
        "geometry cation units angstrom\n"
        "  H 0.0 0.0 0.0\n"
        "end\n"
        "geometry neutral units angstrom\n"
        "  H 0.0 0.0 -0.37\n"
        "  H 0.0 0.0  0.37\n"
        "end\n"
        "set geometry cation\n"
        "charge 1\n"
        "dft; mult 1; end\n"
        "task dft energy\n"
        "set geometry neutral\n"
        "charge 0\n"
        "dft; mult 1; end\n"
        "task dft energy\n"
    )
    input_path.write_text(input_text, encoding="utf-8")
    output_path.write_text(
        (
            "================ echo of input deck ================\n"
            f"{input_text}"
            "====================================================\n"
            "NWChem Input Module\n"
            'Geometry "cation" -> ""\n'
            "Output coordinates in angstroms\n"
            "No. Tag Charge X Y Z\n"
            "1 H 1.0 0.0 0.0 0.0\n"
            "Atomic Mass\n"
            'Geometry "neutral" -> ""\n'
            "Output coordinates in angstroms\n"
            "No. Tag Charge X Y Z\n"
            "1 H 1.0 0.0 0.0 -0.37\n"
            "2 H 1.0 0.0 0.0  0.37\n"
            "Atomic Mass\n"
            "NWChem DFT Module\n"
            "No. of atoms     : 1\n"
            "Charge           : 1\n"
            "Spin multiplicity: 1\n"
            "Total DFT energy = -0.5000000000\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
            "NWChem Input Module\n"
            "NWChem DFT Module\n"
            "No. of atoms     : 2\n"
            "Charge           : 1\n"
            "Spin multiplicity: 1\n"
            "Total DFT energy = -1.1000000000\n"
            "Task times cpu: 0.1s wall: 0.1s\n"
        ),
        encoding="utf-8",
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    task_states = next(
        check
        for check in consistency["checks"]
        if check["field"] == "task_states"
    )
    assert consistency["status"] == "mismatch"
    assert task_states["status"] == "mismatch"
    assert [task["status"] for task in task_states["tasks"]] == [
        "match",
        "mismatch",
    ]
    assert task_states["tasks"][0]["input_state"]["geometry"] == {
        "name": "cation",
        "block_index": 0,
        "source": "input",
    }
    assert task_states["tasks"][0]["comparisons"]["geometry"]["status"] == (
        "match"
    )
    assert task_states["tasks"][1]["comparisons"]["charge"] == {
        "status": "mismatch",
        "input": 0,
        "output": 1,
    }
    assert task_states["tasks"][1]["comparisons"]["atom_count"] == {
        "status": "match",
        "input": 2,
        "output": 2,
    }
    assert inspected["uncertainty"] == [{
        "code": "input_output_mismatch",
        "message": (
            "The explicit input disagrees with output evidence for: "
            "task_states."
        ),
        "impact": (
            "Verify that the supplied input and related restart files belong "
            "to this output."
        ),
    }]


def test_nwchem_missing_explicit_restart_artifact_is_a_mismatch(tmp_path):
    input_path = tmp_path / "restart.nw"
    database_path = tmp_path / "old.db"
    movecs_path = tmp_path / "old.movecs"
    input_path.write_text(
        (
            "restart old\n"
            "geometry\n"
            "  H 0 0 0\n"
            "  H 0 0 0.74\n"
            "end\n"
            "scf\n"
            "  vectors input old.movecs output new.movecs\n"
            "end\n"
            "task scf energy\n"
        ),
        encoding="utf-8",
    )
    database_path.write_bytes(b"database")
    movecs_path.write_bytes(b"movecs")

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        FIXTURES / "nwchem_scf.out",
        resolved_by="explicit",
        artifact_files=(input_path, database_path),
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {
        "match": 2,
        "mismatch": 1,
        "not_checked": 5,
    }
    assert consistency["checks"][-1] == {
        "field": "restart_artifacts",
        "status": "mismatch",
        "input": {
            "references": [
                {
                    "directive": "restart",
                    "declared": "old",
                    "path": str(database_path.resolve()),
                },
                {
                    "directive": "vectors_input",
                    "declared": "old.movecs",
                    "path": str(movecs_path.resolve()),
                },
            ],
        },
        "output": {
            "supplied_paths": sorted(
                (
                    str(input_path.resolve()),
                    str((FIXTURES / "nwchem_scf.out").resolve()),
                    str(database_path.resolve()),
                )
            ),
            "missing_paths": [str(movecs_path.resolve())],
        },
        "basis": (
            "Explicit related-artifact paths supplied to inspect_run."
        ),
    }
    assert inspected["uncertainty"] == [{
        "code": "input_output_mismatch",
        "message": (
            "The explicit input disagrees with output evidence for: "
            "restart_artifacts."
        ),
        "impact": (
            "Verify that the supplied input and related restart files belong "
            "to this output."
        ),
    }]
