from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from chemtools.runner import _compute_watch_sleep_seconds
from chemtools import (
    analyze_frontier_orbitals,
    analyze_imaginary_modes,
    check_nwchem_run_status,
    check_spin_charge_state,
    compare_nwchem_runs,
    review_nwchem_followup_outcome,
    review_nwchem_mcscf_followup_outcome,
    review_nwchem_mcscf_case,
    create_nwchem_input,
    create_nwchem_dft_input_from_request,
    create_nwchem_dft_workflow_input,
    diagnose_output,
    displace_geometry_along_mode,
    evaluate_cases,
    find_restart_assets,
    draft_nwchem_cube_input,
    draft_nwchem_frontier_cube_input,
    draft_nwchem_imaginary_mode_inputs,
    draft_nwchem_mcscf_input,
    draft_nwchem_mcscf_retry_input,
    draft_nwchem_optimization_followup_input,
    draft_nwchem_property_check_input,
    draft_nwchem_scf_stabilization_input,
    draft_nwchem_vectors_swap_input,
    inspect_input,
    inspect_runner_profiles,
    lint_nwchem_input,
    launch_nwchem_run,
    parse_cube,
    parse_freq,
    parse_mcscf_output,
    parse_mos,
    parse_population_analysis,
    parse_output,
    parse_scf,
    parse_tasks,
    parse_trajectory,
    prepare_nwchem_next_step,
    prepare_nwchem_run,
    render_basis_block,
    render_basis_block_from_geometry,
    render_ecp_block,
    render_nwchem_basis_setup,
    review_nwchem_progress,
    watch_nwchem_run,
    review_nwchem_input_request,
    review_nwchem_case,
    resolve_basis,
    resolve_basis_setup,
    resolve_ecp,
    suggest_vectors_swaps,
    summarize_cube,
    summarize_nwchem_case,
    summarize_output,
    suggest_nwchem_scf_fix_strategy,
    suggest_nwchem_mcscf_active_space,
    suggest_nwchem_state_recovery_strategy,
    tail_nwchem_output,
    terminate_nwchem_run,
)


ROOT = Path(__file__).resolve().parent.parent
ORBITRON_TESTS = ROOT / "orbitron" / "io" / "pipelines" / "tests"
BASIS_LIBRARY = ROOT / "nwchem-test" / "nwchem_basis_library"


class ChemToolsTests(unittest.TestCase):
    def _write_runner_profiles_json(self, directory: Path) -> Path:
        payload = {
            "schema_version": "1.0",
            "defaults": {
                "program": "nwchem",
                "input_extension": ".nw",
                "output_extension": ".out",
                "error_extension": ".err",
            },
            "profiles": {
                "test_local": {
                    "description": "Test direct launch",
                    "launcher": {
                        "kind": "direct",
                        "command": "printf launched > {output_file}",
                    },
                    "execution": {
                        "command_template": "{launcher}",
                        "working_directory": "{job_dir}",
                        "shell": "/bin/bash",
                    },
                    "resources": {
                        "mpi_ranks": 1,
                        "omp_threads": 1,
                    },
                    "env": {
                        "OMP_NUM_THREADS": "{omp_threads}",
                    },
                    "file_rules": {
                        "output_file": "{job_name}.out",
                        "error_file": "{job_name}.err",
                        "restart_prefix": "{job_name}",
                    },
                },
                "test_slurm": {
                    "description": "Test scheduler launch",
                    "launcher": {
                        "kind": "scheduler",
                        "submit_command": "sbatch",
                    },
                    "scheduler": {
                        "script_template": "#!/bin/bash\n#SBATCH --job-name={job_name}\n{module_block}\ncd {job_dir}\n{pre_run_block}\nsrun nwchem {input_file}\n",
                        "submit_script_name": "{job_name}.slurm",
                    },
                    "resources": {
                        "mpi_ranks": 4,
                        "omp_threads": 1,
                    },
                    "modules": {
                        "purge_first": True,
                        "load": ["nwchem"],
                    },
                    "hooks": {
                        "pre_run": ["echo ready"],
                    },
                    "file_rules": {
                        "output_file": "{job_name}.out",
                        "error_file": "{job_name}.err",
                        "restart_prefix": "{job_name}",
                    },
                },
            },
        }
        target = directory / "runner_profiles.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def _write_partial_optfreq_case(self, directory: Path) -> tuple[Path, Path]:
        input_path = directory / "partial_optfreq.nw"
        input_path.write_text(
            "\n".join(
                [
                    "start partial_optfreq",
                    'title "partial opt+freq test"',
                    "charge 0",
                    "multiplicity 3",
                    "geometry units angstroms",
                    "  Fe 0.0 0.0 0.0",
                    "  O  0.0 0.0 1.6",
                    "end",
                    'basis "ao basis" spherical',
                    "Fe library def2-svp",
                    "O  library def2-svp",
                    "end",
                    "dft",
                    "  xc b3lyp",
                    "end",
                    "task dft optimize",
                    "task dft freq",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        output_path = directory / "partial_optfreq.out"
        output_path.write_text(
            "\n".join(
                [
                    " NWChem Input Module",
                    ' title "partial opt+freq test"',
                    ' basis "ao basis" spherical',
                    " NWChem Geometry Optimization",
                    " Step 0",
                    " Output coordinates in angstroms",
                    " No.       Tag          Charge          X              Y              Z",
                    " 1         Fe          26.0000         0.000000       0.000000       0.000000",
                    " 2         O            8.0000         0.000000       0.000000       1.600000",
                    " Atomic Mass",
                    " @ Step       Energy      Delta E   Gmax     Grms     Xrms     Xmax   Walltime",
                    " @ ---- ---------------- -------- -------- -------- -------- -------- --------",
                    " @    0   -100.00000000  0.0D+00  0.10000  0.05000  0.00000  0.00000   10.0",
                    " Total DFT energy = -100.000000000000",
                    " Step 1",
                    " Output coordinates in angstroms",
                    " No.       Tag          Charge          X              Y              Z",
                    " 1         Fe          26.0000         0.000000       0.000000       0.010000",
                    " 2         O            8.0000         0.000000       0.000000       1.590000",
                    " Atomic Mass",
                    " @    1   -100.50000000 -5.0D-01  0.02000  0.01000  0.05000  0.08000   20.0",
                    " Total DFT energy = -100.500000000000",
                    " NWChem DFT Gradient Module",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return input_path, output_path

    def _write_divergent_opt_case(self, directory: Path) -> tuple[Path, Path]:
        input_path = directory / "divergent_opt.nw"
        input_path.write_text(
            "\n".join(
                [
                    "start divergent_opt",
                    'title "divergent optimization test"',
                    "charge 0",
                    "multiplicity 3",
                    "geometry units angstroms",
                    "  Fe 0.0 0.0 0.0",
                    "  O  0.0 0.0 1.6",
                    "end",
                    'basis "ao basis" spherical',
                    "Fe library def2-svp",
                    "O  library def2-svp",
                    "end",
                    "dft",
                    "  xc b3lyp",
                    "end",
                    "task dft optimize",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        output_path = directory / "divergent_opt.out"
        output_path.write_text(
            "\n".join(
                [
                    " NWChem Input Module",
                    ' title "divergent optimization test"',
                    ' basis "ao basis" spherical',
                    " NWChem Geometry Optimization",
                    " Step 0",
                    " Output coordinates in angstroms",
                    " No.       Tag          Charge          X              Y              Z",
                    " 1         Fe          26.0000         0.000000       0.000000       0.000000",
                    " 2         O            8.0000         0.000000       0.000000       1.600000",
                    " Atomic Mass",
                    " @ Step       Energy      Delta E   Gmax     Grms     Xrms     Xmax   Walltime",
                    " @ ---- ---------------- -------- -------- -------- -------- -------- --------",
                    " @    0   -100.00000000  0.0D+00  0.05000  0.02000  0.00000  0.00000   10.0",
                    " Total DFT energy = -100.000000000000",
                    " Step 1",
                    " Output coordinates in angstroms",
                    " No.       Tag          Charge          X              Y              Z",
                    " 1         Fe          26.0000         0.000000       0.000000       0.100000",
                    " 2         O            8.0000         0.000000       0.000000       1.800000",
                    " Atomic Mass",
                    " @    1   -101.00000000 -1.0D+00  0.08000  0.03000  0.10000  0.20000   20.0",
                    " Total DFT energy = -101.000000000000",
                    " Step 2",
                    " Output coordinates in angstroms",
                    " No.       Tag          Charge          X              Y              Z",
                    " 1         Fe          26.0000         0.000000       0.000000       0.300000",
                    " 2         O            8.0000         0.000000       0.000000       2.200000",
                    " Atomic Mass",
                    " @    2   -100.70000000  3.0D-01  0.12000  0.05000  0.20000  0.40000   30.0",
                    " Total DFT energy = -100.700000000000",
                    " Step 3",
                    " Output coordinates in angstroms",
                    " No.       Tag          Charge          X              Y              Z",
                    " 1         Fe          26.0000         0.000000       0.000000       0.800000",
                    " 2         O            8.0000         0.000000       0.000000       3.900000",
                    " Atomic Mass",
                    " @    3    -99.90000000  8.0D-01  0.22000  0.09000  0.50000  1.00000   40.0",
                    " Total DFT energy = -99.900000000000",
                    " Step 4",
                    " Output coordinates in angstroms",
                    " No.       Tag          Charge          X              Y              Z",
                    " 1         Fe          26.0000         0.000000       0.000000       1.600000",
                    " 2         O            8.0000         0.000000       0.000000       5.400000",
                    " Atomic Mass",
                    " @    4    -98.00000000  1.9D+00  0.60000  0.25000  0.90000  1.80000   50.0",
                    " Total DFT energy = -98.000000000000",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return input_path, output_path

    def _write_invalid_task_input(self, directory: Path) -> Path:
        input_path = directory / "invalid_tasks.nw"
        input_path.write_text(
            "\n".join(
                [
                    "start invalid_tasks",
                    'title "invalid task syntax test"',
                    "charge -3",
                    "multiplicity 6",
                    "geometry units angstroms",
                    "  Fe 0.0 0.0 0.0",
                    "  Cl 2.1 0.0 0.0",
                    "  Cl -1.05 1.82 0.0",
                    "  Cl -1.05 -1.82 0.0",
                    "end",
                    'basis "ao basis"',
                    "  * library def2-svp",
                    "end",
                    "dft",
                    "  xc b3lyp",
                    "end",
                    "task optimize",
                    "task frequency",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return input_path

    def test_nwchem_tasks_freq_mos_and_trajectory(self) -> None:
        path = ORBITRON_TESTS / "fixtures" / "nwchem" / "ammonium.out"

        tasks = parse_tasks(str(path))
        self.assertEqual(tasks["metadata"]["program"], "nwchem")
        self.assertTrue(tasks["generic_tasks"])

        freq = parse_freq(str(path))
        self.assertGreater(freq["mode_count"], 0)

        mos = parse_mos(str(path), top_n=3)
        self.assertGreater(mos["orbital_count"], 0)

        trajectory = parse_trajectory(str(path))
        self.assertGreater(trajectory["frame_count"], 0)
        self.assertEqual(trajectory["optimization_status"], "converged")

    def test_molpro_tasks_and_mos(self) -> None:
        path = ORBITRON_TESTS / "corpus" / "molpro" / "water.out"

        tasks = parse_tasks(str(path))
        self.assertEqual(tasks["metadata"]["program"], "molpro")
        self.assertTrue(tasks["generic_tasks"])

        mos = parse_mos(str(path), top_n=3)
        self.assertGreater(mos["orbital_count"], 0)

    def test_nwchem_mos_include_spin_channels_and_orbital_character(self) -> None:
        path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"

        mos = parse_mos(str(path), top_n=3)
        self.assertIn("alpha", mos["spin_channels"])
        self.assertIn("beta", mos["spin_channels"])
        self.assertEqual(mos["spin_channels"]["alpha"]["somo_count"], 5)
        self.assertEqual(mos["spin_channels"]["beta"]["somo_count"], 0)
        self.assertEqual(mos["somo_count"], 5)
        self.assertEqual(mos["spin_channels"]["alpha"]["somos"][0]["top_atom_contributions"][0]["element"], "C")
        self.assertIsNotNone(mos["spin_channels"]["alpha"]["somos"][0]["dominant_character"])

    def test_nwchem_population_analysis_parses_mulliken_and_lowdin(self) -> None:
        path = ROOT / "nwchem-test" / "train" / "cmcc3h2_s" / "cmcc3h2_s_1.out"

        population = parse_population_analysis(str(path))
        self.assertIn("mulliken", population["available_methods"])
        self.assertIn("lowdin", population["available_methods"])
        self.assertEqual(population["methods"]["mulliken"]["latest_total"]["atom_count"], 28)
        self.assertEqual(population["methods"]["lowdin"]["latest_spin"]["atom_count"], 28)
        self.assertAlmostEqual(population["methods"]["mulliken"]["latest_spin"]["population_sum"], 7.0, places=2)
        self.assertEqual(
            population["methods"]["mulliken"]["latest_total"]["largest_positive_sites"][0]["element"],
            "Cm",
        )

    def test_analyze_frontier_orbitals_flags_wrong_state_vs_good_metal_somo(self) -> None:
        bad_output = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        bad_input = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"
        good_output = ROOT / "nwchem-test" / "train" / "fecn6_fragment_guess" / "solution.out"

        bad = analyze_frontier_orbitals(str(bad_output), input_path=str(bad_input))
        bad_analysis = bad["analysis"]
        self.assertEqual(bad_analysis["assessment"], "metal_state_mismatch_suspected")
        self.assertEqual(
            bad_analysis["frontier_channels"]["alpha"]["somos"][0]["character_class"],
            "ligand_centered_pi",
        )

        good = analyze_frontier_orbitals(
            str(good_output),
            expected_metal_elements=["Fe"],
            expected_somo_count=1,
        )
        good_analysis = good["analysis"]
        self.assertEqual(good_analysis["assessment"], "state_consistent_with_expected_metal_open_shell")
        self.assertEqual(good_analysis["metal_like_somo_count"], 1)
        self.assertEqual(
            good_analysis["frontier_channels"]["alpha"]["somos"][0]["character_class"],
            "metal_centered_d",
        )

    def test_suggest_vectors_swaps_reproduces_fragment_guess_pattern(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_frag.out"

        suggestion = suggest_vectors_swaps(
            str(output_path),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
            vectors_input="hexaaquairon_frag.movecs",
            vectors_output="hexaaquairon_swap.movecs",
        )["suggestion"]

        self.assertTrue(suggestion["available"])
        self.assertEqual(
            [(pair["spin"], pair["from_vector"], pair["to_vector"]) for pair in suggestion["swap_pairs"]],
            [
                ("alpha", 73, 76),
                ("beta", 73, 76),
                ("alpha", 70, 78),
                ("alpha", 67, 80),
            ],
        )
        self.assertIn("vectors input hexaaquairon_frag.movecs", suggestion["vectors_block"])
        self.assertIn("swap alpha 73 76", suggestion["vectors_block"])
        self.assertIn("output hexaaquairon_swap.movecs", suggestion["vectors_block"])

    def test_cube_summary_handles_density_and_spin_density(self) -> None:
        density_path = ROOT / "nwchem-test" / "train" / "standard" / "chargedensity.cube"
        spin_path = ROOT / "nwchem-test" / "train" / "standard" / "h2o_spindens.cube"

        density_header = parse_cube(str(density_path))
        self.assertEqual(density_header["grid_shape"], [11, 11, 11])
        density = summarize_cube(str(density_path), top_atoms=2)
        self.assertEqual(density["dataset_kind"], "density")
        self.assertEqual(density["top_localized_atoms"][0]["element"], "N")
        self.assertGreater(density["absolute_integral"], 0.0)

        spin = summarize_cube(str(spin_path), top_atoms=2)
        self.assertEqual(spin["dataset_kind"], "spin_density")
        self.assertEqual(spin["value_range"], [0.0, 0.0])
        self.assertIsNone(spin["positive_lobe_center_angstrom"])

    def test_molcas_tasks(self) -> None:
        path = ORBITRON_TESTS / "corpus" / "molcas" / "benzene" / "benzene.out"

        tasks = parse_tasks(str(path))
        self.assertEqual(tasks["metadata"]["program"], "molcas")
        self.assertTrue(tasks["generic_tasks"])

    def test_parse_output_collects_errors_for_unsupported_sections(self) -> None:
        path = ORBITRON_TESTS / "corpus" / "molcas" / "benzene" / "benzene.out"
        # Explicitly request sections that are not supported for molcas to verify error collection
        payload = parse_output(str(path), sections=["tasks", "mos", "freq", "trajectory"])
        self.assertEqual(payload["metadata"]["program"], "molcas")
        self.assertIsNotNone(payload["tasks"])
        self.assertTrue(payload["errors"])

    def test_resolve_basis_and_render_block(self) -> None:
        resolved = resolve_basis("def2-svp", ["Fe", "C", "H"], str(BASIS_LIBRARY))
        self.assertTrue(resolved["all_elements_covered"])
        self.assertEqual(resolved["covered_elements"], ["Fe", "C", "H"])

        rendered = render_basis_block("def2-svp", ["Fe", "C", "H"], str(BASIS_LIBRARY))
        self.assertIn('basis "ao basis" spherical', rendered["text"])
        self.assertNotIn("library def2-svp", rendered["text"])
        self.assertIn("Fe    S", rendered["text"])
        self.assertIn("C    S", rendered["text"])

    def test_render_basis_block_from_geometry(self) -> None:
        path = ROOT / "nwchem-test" / "train" / "ferrocene" / "successful" / "ferrocene.nw"
        rendered = render_basis_block_from_geometry("def2-svp", str(path), str(BASIS_LIBRARY))
        self.assertNotIn("library def2-svp", rendered["text"])
        self.assertIn("Fe    S", rendered["text"])
        self.assertIn("C    S", rendered["text"])
        self.assertIn("H    S", rendered["text"])

    def test_resolve_render_and_create_mixed_basis_setup_with_ecp(self) -> None:
        geometry_path = ROOT / "nwchem-test" / "train" / "standard" / "uo2-test.nw"

        resolved_ecp = resolve_ecp("lanl2dz_ecp", ["U"], str(BASIS_LIBRARY))
        self.assertTrue(resolved_ecp["all_elements_covered"])
        self.assertEqual(resolved_ecp["covered_elements"], ["U"])

        rendered_ecp = render_ecp_block("lanl2dz_ecp", ["U"], str(BASIS_LIBRARY))
        self.assertIn("ecp", rendered_ecp["text"].lower())
        self.assertNotIn("library lanl2dz_ecp", rendered_ecp["text"])
        self.assertIn("U nelec", rendered_ecp["text"])

        setup = resolve_basis_setup(
            geometry_path=str(geometry_path),
            library_path=str(BASIS_LIBRARY),
            basis_assignments={"U": "lanl2dz_ecp"},
            ecp_assignments={"U": "lanl2dz_ecp"},
            default_basis="aug-cc-pvdz",
        )
        self.assertEqual(setup["elements"], ["U", "O"])
        self.assertTrue(setup["basis"]["all_elements_covered"])
        self.assertEqual(setup["basis"]["assignments"]["U"], "lanl2dz_ecp")
        self.assertEqual(setup["basis"]["assignments"]["O"], "aug-cc-pvdz")
        self.assertEqual(setup["ecp"]["elements_with_ecp"], ["U"])

        rendered_setup = render_nwchem_basis_setup(
            geometry_path=str(geometry_path),
            library_path=str(BASIS_LIBRARY),
            basis_assignments={"U": "lanl2dz_ecp"},
            ecp_assignments={"U": "lanl2dz_ecp"},
            default_basis="aug-cc-pvdz",
        )
        self.assertNotIn("library lanl2dz_ecp", rendered_setup["basis_block"]["text"])
        self.assertNotIn("library aug-cc-pvdz", rendered_setup["basis_block"]["text"])
        self.assertIn("U    S", rendered_setup["basis_block"]["text"])
        self.assertIn("O    S", rendered_setup["basis_block"]["text"])
        self.assertIn("U nelec", rendered_setup["ecp_block"]["text"])

        created = create_nwchem_input(
            geometry_path=str(geometry_path),
            library_path=str(BASIS_LIBRARY),
            basis_assignments={"U": "lanl2dz_ecp"},
            ecp_assignments={"U": "lanl2dz_ecp"},
            default_basis="aug-cc-pvdz",
            module="dft",
            task_operation="energy",
            charge=2,
            multiplicity=3,
            memory="1800 mb",
            start_name="uo2_mixed",
            module_settings=["xc pbe0", "maxiter 200"],
        )
        self.assertEqual(created["vectors_output"], "uo2_mixed.movecs")
        self.assertIn("vectors output uo2_mixed.movecs", created["input_text"])
        self.assertIn("mult 3", created["input_text"])
        self.assertIn("iterations 300", created["input_text"].lower())
        self.assertIn("\n  mulliken\n", created["input_text"].lower())
        self.assertIn("\n  direct\n", created["input_text"].lower())
        self.assertIn("\n  noio\n", created["input_text"].lower())
        self.assertIn("\n  grid nodisk\n", created["input_text"].lower())
        self.assertNotIn("library aug-cc-pvdz", created["input_text"])
        self.assertNotIn("library lanl2dz_ecp", created["input_text"])
        self.assertIn("O    S", created["input_text"])
        self.assertIn("U    S", created["input_text"])
        self.assertIn("U nelec", created["input_text"])
        self.assertIn("task dft energy", created["input_text"])

    def test_review_nwchem_input_request_requires_geometry_and_multiplicity_for_formula_only_transition_metal_case(self) -> None:
        review = review_nwchem_input_request(
            formula="Fe2O3",
            library_path=str(BASIS_LIBRARY),
            default_basis="cc-pvdz",
            module="dft",
            task_operations=["opt", "freq"],
            functional="b3lyp",
        )
        self.assertFalse(review["ready_to_create"])
        self.assertIn("Fe", review["transition_metals"])
        self.assertEqual(review["charge"], 0)
        missing = {item["field"] for item in review["missing_requirements"]}
        self.assertIn("geometry_source", missing)
        self.assertIn("multiplicity", missing)

    def test_create_nwchem_dft_workflow_input_builds_opt_and_freq(self) -> None:
        geometry_path = ROOT / "nwchem-test" / "train" / "standard" / "uo2-test.nw"

        drafted = create_nwchem_dft_workflow_input(
            geometry_path=str(geometry_path),
            library_path=str(BASIS_LIBRARY),
            basis_assignments={"U": "lanl2dz_ecp"},
            ecp_assignments={"U": "lanl2dz_ecp"},
            default_basis="aug-cc-pvdz",
            xc_functional="b3lyp",
            task_operations=["opt", "freq"],
            charge=2,
            multiplicity=3,
            start_name="uo2_optfreq",
            dft_settings=["maxiter 200"],
        )
        self.assertEqual(drafted["vectors_output"], "uo2_optfreq.movecs")
        self.assertIn("xc b3lyp", drafted["input_text"].lower())
        self.assertIn("mult 3", drafted["input_text"].lower())
        self.assertIn("iterations 300", drafted["input_text"].lower())
        self.assertIn("\n  mulliken\n", drafted["input_text"].lower())
        self.assertIn("\n  direct\n", drafted["input_text"].lower())
        self.assertIn("\n  noio\n", drafted["input_text"].lower())
        self.assertIn("\n  grid nodisk\n", drafted["input_text"].lower())
        self.assertIn("\ndriver\n  maxiter 300\nend\n", drafted["input_text"].lower())
        self.assertIn("task dft optimize", drafted["input_text"].lower())
        self.assertIn("task dft freq", drafted["input_text"].lower())
        self.assertIn("vectors output uo2_optfreq.movecs", drafted["input_text"].lower())

    def test_create_nwchem_dft_input_from_request_refuses_incomplete_transition_metal_request(self) -> None:
        payload = create_nwchem_dft_input_from_request(
            formula="Fe2O3",
            library_path=str(BASIS_LIBRARY),
            default_basis="cc-pvdz",
            xc_functional="b3lyp",
            task_operations=["opt", "freq"],
        )
        self.assertFalse(payload["ready_to_create"])
        self.assertFalse(payload["created"])
        self.assertEqual(payload["next_action"], "provide_missing_requirements")
        self.assertIsNone(payload["input_text"])
        missing = {item["field"] for item in payload["review"]["missing_requirements"]}
        self.assertIn("geometry_source", missing)
        self.assertIn("multiplicity", missing)

    def test_create_nwchem_dft_input_from_request_builds_deterministic_input(self) -> None:
        geometry_path = ROOT / "nwchem-test" / "train" / "standard" / "uo2-test.nw"

        payload = create_nwchem_dft_input_from_request(
            geometry_path=str(geometry_path),
            library_path=str(BASIS_LIBRARY),
            basis_assignments={"U": "lanl2dz_ecp"},
            ecp_assignments={"U": "lanl2dz_ecp"},
            default_basis="aug-cc-pvdz",
            xc_functional="b3lyp",
            task_operations=["opt", "freq"],
            charge=2,
            multiplicity=3,
            start_name="uo2_optfreq_request",
        )
        self.assertTrue(payload["ready_to_create"])
        self.assertTrue(payload["created"])
        self.assertIn("task dft optimize", payload["input_text"].lower())
        self.assertIn("task dft freq", payload["input_text"].lower())
        self.assertIn("iterations 300", payload["input_text"].lower())
        self.assertIn("\n  mulliken\n", payload["input_text"].lower())
        self.assertIn("vectors output uo2_optfreq_request.movecs", payload["input_text"].lower())

    def test_lint_nwchem_input_detects_missing_vectors_output(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.nw"

        lint = lint_nwchem_input(str(input_path), library_path=str(BASIS_LIBRARY))
        self.assertIn(lint["status"], {"warning", "error"})
        self.assertTrue(any(issue["code"] == "missing_vectors_output" for issue in lint["issues"]))

    def test_find_restart_assets_prefers_matching_stem_assets(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"

        assets = find_restart_assets(str(output_path))
        self.assertEqual(Path(assets["preferred"]["vectors_file"]).name, "hexaaquairon.movecs")
        self.assertEqual(Path(assets["preferred"]["database_file"]).name, "hexaaquairon.db")
        self.assertTrue(any(item["kind"] == "movecs" for item in assets["restart_candidates"]))

    def test_find_restart_assets_uses_related_tokens_when_no_exact_match(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.out"

        assets = find_restart_assets(str(output_path))
        self.assertEqual(Path(assets["preferred"]["vectors_file"]).name, "cu_h2o4_opt.movecs")
        self.assertEqual(Path(assets["preferred"]["database_file"]).name, "cu_h2o4_opt.db")

    def test_check_spin_charge_state_flags_wrong_state_case(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        state = check_spin_charge_state(
            str(output_path),
            input_path=str(input_path),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )
        self.assertEqual(state["assessment"], "suspicious")
        self.assertEqual(state["charge"], 1)
        self.assertEqual(state["multiplicity"], 6)

    def test_summarize_nwchem_case_includes_lint_assets_state_and_next_step(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        summary = summarize_nwchem_case(
            output_path=str(output_path),
            input_path=str(input_path),
            library_path=str(BASIS_LIBRARY),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )
        self.assertIn("Input lint:", summary["summary_text"])
        self.assertEqual(summary["spin_charge_state"]["assessment"], "suspicious")
        self.assertEqual(summary["next_step"]["selected_workflow"], "wrong_state_swap_recovery")
        self.assertEqual(Path(summary["restart_assets"]["preferred"]["vectors_file"]).name, "hexaaquairon.movecs")
        self.assertIn("compact_summary", summary)

    def test_summarize_nwchem_case_compact_mode_trims_nested_payloads(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        summary = summarize_nwchem_case(
            output_path=str(output_path),
            input_path=str(input_path),
            library_path=str(BASIS_LIBRARY),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
            compact=True,
        )
        self.assertNotIn("diagnosis_summary", summary)
        self.assertNotIn("restart_assets", summary["next_step"])
        self.assertEqual(summary["spin_charge_state"]["assessment"], "suspicious")
        self.assertEqual(summary["next_step"]["selected_workflow"], "wrong_state_swap_recovery")
        self.assertIn("prepared_artifact_summaries", summary["next_step"])

    def test_review_nwchem_case_returns_compact_summary_alias(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        review = review_nwchem_case(
            output_path=str(output_path),
            input_path=str(input_path),
            library_path=str(BASIS_LIBRARY),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )
        self.assertNotIn("diagnosis_summary", review)
        self.assertEqual(review["spin_charge_state"]["assessment"], "suspicious")
        self.assertEqual(review["next_step"]["selected_workflow"], "wrong_state_swap_recovery")
        self.assertIn("Prepared workflow:", review["summary_text"])

    def test_suggest_nwchem_scf_fix_strategy_for_failed_state_check_prefers_stabilization(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw"

        suggestion = suggest_nwchem_scf_fix_strategy(
            str(output_path),
            input_path=str(input_path),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )
        self.assertEqual(suggestion["strategy_family"], "scf_recovery")
        self.assertEqual(suggestion["primary_strategy"], "stabilization_restart")
        self.assertEqual(suggestion["strategies"][0]["name"], "stabilization_restart")
        self.assertIn("scf_open_shell", suggestion["strategies"][0]["docs_topics"])
        self.assertIn("fragment_guess", suggestion["strategies"][1]["docs_topics"])

    def test_suggest_nwchem_state_recovery_strategy_for_wrong_state_prefers_swap_then_fragment(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        suggestion = suggest_nwchem_state_recovery_strategy(
            str(output_path),
            input_path=str(input_path),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )
        self.assertEqual(suggestion["regime"], "metal_state_mismatch")
        self.assertEqual(suggestion["primary_strategy"], "vectors_swap_restart")
        self.assertEqual(suggestion["strategies"][0]["name"], "vectors_swap_restart")
        self.assertEqual(suggestion["strategies"][1]["name"], "fragment_guess_seed")
        self.assertIn("mcscf", suggestion["strategies"][2]["docs_topics"])

    def test_suggest_nwchem_mcscf_active_space_includes_somos_and_metal_d_candidates(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        suggestion = suggest_nwchem_mcscf_active_space(
            str(output_path),
            input_path=str(input_path),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )
        self.assertEqual(suggestion["frontier_assessment"], "metal_state_mismatch_suspected")
        self.assertTrue(suggestion["minimal_active_space"]["active_orbitals"] >= 8)
        self.assertIn(80, suggestion["minimal_active_space"]["vector_numbers"])
        self.assertIn(79, suggestion["minimal_active_space"]["vector_numbers"])
        self.assertGreaterEqual(len(suggestion["swap_in_candidates"]), 1)
        self.assertTrue(
            any(item["character_class"] == "metal_centered_d" for item in suggestion["minimal_active_space"]["orbitals"])
        )

    def test_parse_nwchem_mcscf_output_reports_input_parse_error_for_bad_state_syntax(self) -> None:
        output_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.out"

        parsed = parse_mcscf_output(str(output_path))
        self.assertEqual(parsed["status"], "failed")
        self.assertEqual(parsed["failure_mode"], "input_parse_error")
        self.assertEqual(parsed["settings"]["state"], "sextet")
        self.assertEqual(parsed["settings"]["multiplicity"], 6)

    def test_parse_nwchem_mcscf_output_extracts_converged_retry_details(self) -> None:
        output_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test_retry2.out"

        parsed = parse_mcscf_output(str(output_path))
        self.assertEqual(parsed["status"], "converged")
        self.assertEqual(parsed["settings"]["active_orbitals"], 8)
        self.assertEqual(parsed["settings"]["active_electrons"], 9)
        self.assertEqual(parsed["settings"]["hessian"], "exact")
        self.assertEqual(parsed["settings"]["maxiter"], 80)
        self.assertAlmostEqual(parsed["settings"]["initial_levelshift"], 0.6, places=6)
        self.assertGreater(parsed["iteration_count"], 10)
        self.assertLess(parsed["final_energy_hartree"], -2641.13)
        self.assertTrue(parsed["converged_ci_vector"])
        self.assertGreater(parsed["ci_vector"]["configuration_count"], 5)
        self.assertTrue(parsed["natural_occupations"])
        self.assertEqual(parsed["active_space_mulliken"]["atom_count"], 4)

    def test_review_nwchem_mcscf_case_summarizes_converged_retry(self) -> None:
        output_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test_retry2.out"
        input_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.nw"

        review = review_nwchem_mcscf_case(
            output_path=str(output_path),
            input_path=str(input_path),
            expected_metal_elements=["Fe"],
        )
        self.assertEqual(review["status"], "converged")
        self.assertEqual(review["convergence_review"]["assessment"], "converged_with_stiff_orbital_optimization")
        self.assertIn("CAS(9,8)", review["summary_text"])
        self.assertEqual(review["active_space_density_review"]["assessment"], "metal_participation_significant")
        self.assertEqual(review["recommended_next_action"], "use_mcscf_as_reference_or_seed_for_follow_up")

    def test_review_nwchem_mcscf_case_flags_parse_error(self) -> None:
        output_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.out"
        input_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.nw"

        review = review_nwchem_mcscf_case(
            output_path=str(output_path),
            input_path=str(input_path),
            expected_metal_elements=["Fe"],
        )
        self.assertEqual(review["status"], "failed")
        self.assertEqual(review["recommended_next_action"], "fix_mcscf_block_syntax_before_retry")
        self.assertIn("Next action: fix_mcscf_block_syntax_before_retry", review["summary_text"])

    def test_review_nwchem_mcscf_followup_outcome_bad_to_converged_retry(self) -> None:
        reference_output = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.out"
        candidate_output = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test_retry2.out"
        input_file = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.nw"

        payload = review_nwchem_mcscf_followup_outcome(
            str(reference_output),
            str(candidate_output),
            str(input_file),
            str(input_file),
            expected_metal_elements=["Fe"],
        )
        self.assertIn("MCSCF now converges", payload["comparison_headline"])
        self.assertEqual(payload["comparison"]["overall_assessment"], "improved")
        self.assertEqual(payload["comparison"]["candidate_summary"]["status"], "converged")
        self.assertIsNone(payload["candidate_next_step"])

    def test_review_nwchem_mcscf_followup_outcome_incomplete_candidate_prepares_retry(self) -> None:
        reference_output = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.out"
        candidate_output = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test_retry.out"
        input_file = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.nw"

        payload = review_nwchem_mcscf_followup_outcome(
            str(reference_output),
            str(candidate_output),
            str(input_file),
            str(input_file),
            expected_metal_elements=["Fe"],
        )
        self.assertEqual(payload["comparison"]["candidate_summary"]["status"], "incomplete")
        self.assertEqual(payload["comparison"]["overall_assessment"], "improved")
        self.assertIsNotNone(payload["candidate_next_step"])
        self.assertEqual(payload["candidate_next_step"]["retry_strategy"], "stronger_convergence_retry")

    def test_draft_nwchem_mcscf_retry_input_repairs_bad_state_syntax_case(self) -> None:
        output_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.out"
        input_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.nw"

        drafted = draft_nwchem_mcscf_retry_input(
            output_path=str(output_path),
            input_path=str(input_path),
            expected_metal_elements=["Fe"],
        )
        self.assertEqual(drafted["retry_strategy"], "syntax_cleanup_retry")
        self.assertEqual(drafted["resolved_settings"]["active_space_mode"], "minimal")
        self.assertEqual(drafted["resolved_settings"]["hessian"], "exact")
        self.assertEqual(drafted["resolved_settings"]["maxiter"], 80)
        self.assertAlmostEqual(drafted["resolved_settings"]["level"], 0.6, places=6)
        self.assertEqual(drafted["drafted_input"]["active_space"]["active_orbitals"], 8)
        self.assertEqual(drafted["drafted_input"]["active_space"]["active_electrons"], 9)
        self.assertIn("active 8", drafted["input_text"].lower())
        self.assertIn("actelec 9", drafted["input_text"].lower())
        self.assertIn("multiplicity 6", drafted["input_text"].lower())
        self.assertNotIn("state sextet", drafted["input_text"].lower())
        self.assertIn("task mcscf", drafted["input_text"].lower())

    def test_draft_nwchem_mcscf_retry_input_refines_stiff_converged_case(self) -> None:
        output_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test_retry2.out"
        input_path = ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.nw"

        drafted = draft_nwchem_mcscf_retry_input(
            output_path=str(output_path),
            input_path=str(input_path),
            expected_metal_elements=["Fe"],
        )
        self.assertEqual(drafted["retry_strategy"], "stiff_but_converged_refinement")
        self.assertEqual(drafted["resolved_settings"]["active_space_mode"], "minimal")
        self.assertEqual(drafted["resolved_settings"]["vectors_input"], "fecl3_mcscf_test.movecs")
        self.assertEqual(drafted["resolved_settings"]["hessian"], "exact")
        self.assertEqual(drafted["resolved_settings"]["maxiter"], 120)
        self.assertAlmostEqual(drafted["resolved_settings"]["level"], 0.6, places=6)
        self.assertIn("vectors input fecl3_mcscf_test.movecs", drafted["input_text"].lower())
        self.assertIn("output fecl3_mcscf_test_mcscf_retry.movecs", drafted["input_text"].lower())

    def test_draft_nwchem_mcscf_input_uses_recommended_active_space_and_task(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        drafted = draft_nwchem_mcscf_input(
            input_path=str(input_path),
            reference_output_path=str(output_path),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )
        self.assertEqual(drafted["active_space_mode"], "minimal")
        self.assertEqual(drafted["active_space"]["active_orbitals"], 10)
        self.assertEqual(drafted["active_space"]["active_electrons"], 11)
        self.assertEqual(drafted["task_lines"], ["task mcscf"])
        self.assertIn("mcscf", drafted["module_block_text"].lower())
        self.assertIn("active 10", drafted["module_block_text"].lower())
        self.assertIn("actelec 11", drafted["module_block_text"].lower())
        self.assertIn("multiplicity 6", drafted["module_block_text"].lower())
        self.assertIn("hessian exact", drafted["module_block_text"].lower())
        self.assertIn("maxiter 80", drafted["module_block_text"].lower())
        self.assertIn("thresh 1.0e-05", drafted["module_block_text"].lower())
        self.assertIn("level 0.6", drafted["module_block_text"].lower())
        self.assertIn("vectors input hexaaquairon.movecs", drafted["module_block_text"].lower())
        self.assertIn("output hexaaquairon_mcscf.movecs", drafted["module_block_text"].lower())
        self.assertTrue(drafted["reorder_plan"]["swap_pairs"])
        self.assertIn("task mcscf", drafted["input_text"].lower())

    def test_inspect_nwchem_input_and_parse_scf(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"
        output_path = ROOT / "nwchem-test" / "train" / "feo_scf_convergence" / "solution.out"

        inspected = inspect_input(str(input_path))
        self.assertEqual(inspected["charge"], 1)
        self.assertEqual(inspected["multiplicity"], 6)
        self.assertIn("Fe", inspected["transition_metals"])

        scf = parse_scf(str(output_path))
        self.assertEqual(scf["status"], "converged")
        self.assertGreater(scf["iteration_count"], 0)

    def test_diagnose_wrong_state_convergence(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"

        diagnosis = diagnose_output(str(output_path), input_path=str(input_path))
        self.assertEqual(diagnosis["failure_class"], "wrong_state_convergence")
        self.assertEqual(diagnosis["state_check"]["expected_somo_count"], 5)

    def test_diagnose_scf_nonconvergence(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw"
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.out"

        diagnosis = diagnose_output(str(output_path), input_path=str(input_path))
        self.assertEqual(diagnosis["failure_class"], "scf_nonconvergence")
        self.assertEqual(diagnosis["scf"]["status"], "failed")

    def test_summarize_output(self) -> None:
        input_path = ROOT / "nwchemaitest" / "uo2-test.nw"
        output_path = ROOT / "nwchemaitest" / "uo2-test.out"

        summary = summarize_output(str(output_path), input_path=str(input_path))
        self.assertEqual(summary["failure_class"], "no_clear_failure_detected")
        self.assertIn("Outcome:", summary["summary_text"])
        self.assertIn("projected modes", summary["summary_text"])
        self.assertIn("Thermochemistry:", summary["summary_text"])
        self.assertIn("Next action:", summary["summary_text"])

    def test_diagnose_interrupted_frequency_after_completed_optimization(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "cmcc3h2_s" / "cmcc3h2_s_1.out"

        diagnosis = diagnose_output(str(output_path))
        self.assertEqual(diagnosis["failure_class"], "post_optimization_frequency_interrupted")
        self.assertEqual(diagnosis["stage"], "frequency")
        self.assertEqual(diagnosis["task_outcome"], "incomplete")
        self.assertEqual(diagnosis["recommended_next_action"], "restart_frequency_from_last_geometry_without_reoptimizing")

    def test_summarize_interrupted_frequency_after_completed_optimization(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "cmcc3h2_s" / "cmcc3h2_s_1.out"

        summary = summarize_output(str(output_path))
        self.assertEqual(summary["failure_class"], "post_optimization_frequency_interrupted")
        self.assertIn("Diagnosis: post_optimization_frequency_interrupted", summary["summary_text"])
        self.assertIn("Next action: restart_frequency_from_last_geometry_without_reoptimizing", summary["summary_text"])

    def test_parse_frequency_prefers_projected_modes_and_extracts_thermochemistry(self) -> None:
        path = ROOT / "nwchem-test" / "train" / "nh3_inversion_freq" / "solution.out"

        freq = parse_freq(str(path))
        self.assertEqual(freq["preferred_kind"], "projected")
        self.assertEqual(freq["mode_count"], 12)
        self.assertEqual(freq["near_zero_mode_count"], 6)
        self.assertEqual(freq["significant_imaginary_mode_count"], 0)
        self.assertAlmostEqual(freq["thermochemistry"]["temperature_kelvin"], 298.15, places=2)
        self.assertAlmostEqual(
            freq["thermochemistry"]["zero_point_correction"]["hartree"], 0.034159, places=6
        )

    def test_parse_trajectory_reports_optimization_status_and_steps(self) -> None:
        path = ORBITRON_TESTS / "fixtures" / "nwchem" / "opt" / "h2o-opt.out"

        trajectory = parse_trajectory(str(path))
        self.assertEqual(trajectory["optimization_status"], "converged")
        self.assertEqual(trajectory["step_count"], 5)
        self.assertEqual(trajectory["frame_count"], 5)
        self.assertIsNotNone(trajectory["final_energy_hartree"])
        self.assertTrue(trajectory["criteria_met"]["all_met"])
        self.assertAlmostEqual(trajectory["thresholds"]["gmax"], 1.5e-05, places=10)

    def test_parse_frequency_can_include_projected_mode_displacements(self) -> None:
        path = ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "solution.out"

        freq = parse_freq(str(path), include_displacements=True)
        self.assertEqual(freq["preferred_kind"], "projected")
        self.assertEqual(freq["mode_count"], 12)
        self.assertEqual(len(freq["modes"][6]["displacements_cartesian"]), 4)
        self.assertEqual(freq["modes"][6]["displacements_cartesian"][0]["label"], "O")

    def test_analyze_imaginary_modes(self) -> None:
        path = ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "failed.out"

        analysis = analyze_imaginary_modes(str(path))
        self.assertEqual(analysis["significant_imaginary_mode_count"], 1)
        self.assertAlmostEqual(analysis["selected_mode"]["frequency_cm1"], -282.195, places=3)
        self.assertEqual(analysis["selected_mode"]["motion_type"], "torsion")
        self.assertEqual(analysis["stability_assessment"]["classification"], "likely_real_instability")
        self.assertEqual(len(analysis["selected_mode"]["dominant_atoms"]), 4)

    def test_analyze_imaginary_modes_detects_pyramidal_inversion(self) -> None:
        path = ROOT / "nwchem-test" / "train" / "nh3_inversion_freq" / "failed.out"

        analysis = analyze_imaginary_modes(str(path))
        self.assertEqual(analysis["selected_mode"]["motion_type"], "pyramidal_inversion")

    def test_displace_geometry_along_mode(self) -> None:
        path = ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "failed.out"

        displaced = displace_geometry_along_mode(str(path), amplitude_angstrom=0.12)
        self.assertAlmostEqual(displaced["selected_mode"]["frequency_cm1"], -282.195, places=3)
        self.assertEqual(displaced["equilibrium_geometry"]["atom_count"], 4)
        self.assertEqual(len(displaced["plus_geometry"]["atoms"]), 4)
        self.assertIn("geometry units angstrom", displaced["plus_geometry"]["geometry_block"])

    def test_draft_nwchem_imaginary_mode_inputs(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "failed.out"
        input_path = ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "failed.nw"

        drafted = draft_nwchem_imaginary_mode_inputs(
            str(output_path),
            str(input_path),
            amplitude_angstrom=0.12,
        )
        self.assertIn("noautosym", drafted["plus_input_text"].lower())
        self.assertIn("symmetry c1", drafted["plus_input_text"].lower())
        self.assertIn("geometry units angstrom noautosym", drafted["plus_input_text"].lower())
        self.assertEqual(drafted["follow_up_plan"]["strategy"], "optimize_then_freq")
        self.assertIn("task dft optimize", drafted["plus_input_text"].lower())
        self.assertIn("task dft freq", drafted["plus_input_text"].lower())
        self.assertEqual(drafted["plus_vectors_output"], "failed_mode1_plus.movecs")
        self.assertEqual(drafted["minus_vectors_output"], "failed_mode1_minus.movecs")
        self.assertIn("vectors input atomic output failed_mode1_plus.movecs", drafted["plus_input_text"].lower())
        self.assertIn("vectors input atomic output failed_mode1_minus.movecs", drafted["minus_input_text"].lower())

    def test_draft_nwchem_imaginary_mode_inputs_can_write_files(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "failed.out"
        input_path = ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "failed.nw"

        with tempfile.TemporaryDirectory() as tmpdir:
            drafted = draft_nwchem_imaginary_mode_inputs(
                str(output_path),
                str(input_path),
                amplitude_angstrom=0.12,
                output_dir=tmpdir,
                base_name="h2o2_fix",
                write_files=True,
            )
            self.assertIsNotNone(drafted["written_files"])
            self.assertTrue(Path(drafted["written_files"]["plus_file"]).is_file())
            self.assertTrue(Path(drafted["written_files"]["minus_file"]).is_file())

    def test_draft_nwchem_vectors_swap_input(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_frag.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        drafted = draft_nwchem_vectors_swap_input(
            str(output_path),
            str(input_path),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )

        self.assertEqual(drafted["suggestion"]["swap_pairs"][0]["from_vector"], 73)
        self.assertEqual(drafted["vectors_input"], "hexaaquairon_frag.movecs")
        self.assertEqual(drafted["vectors_output"], "hexaaquairon_swap.movecs")
        self.assertIn("start hexaaquairon_swap", drafted["input_text"].lower())
        self.assertIn("title \"hexaaquairon_swap: push metal-centered orbitals into somo positions\"", drafted["input_text"].lower())
        self.assertIn("iterations 500", drafted["input_text"].lower())
        self.assertIn("smear 0.001", drafted["input_text"].lower())
        self.assertIn("convergence damp 30", drafted["input_text"].lower())
        self.assertIn("swap alpha 73 76", drafted["input_text"].lower())
        self.assertIn("swap beta 73 76", drafted["input_text"].lower())
        self.assertIn("output hexaaquairon_swap.movecs", drafted["input_text"].lower())
        self.assertIn('print "mulliken"', drafted["input_text"].lower())
        self.assertNotIn("\ndriver\n", drafted["input_text"].lower())
        self.assertEqual(drafted["file_plan"]["input_file"], str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw"))

    def test_draft_nwchem_vectors_swap_input_can_write_file(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_frag.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        with tempfile.TemporaryDirectory() as tmpdir:
            drafted = draft_nwchem_vectors_swap_input(
                str(output_path),
                str(input_path),
                expected_metal_elements=["Fe"],
                expected_somo_count=5,
                output_dir=tmpdir,
                base_name="hexaaquairon_recover",
                write_file=True,
            )
            self.assertIsNotNone(drafted["written_file"])
            self.assertTrue(Path(drafted["written_file"]).is_file())
            self.assertTrue(drafted["written_file"].endswith("hexaaquairon_recover.nw"))

    def test_draft_nwchem_property_check_input(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw"

        drafted = draft_nwchem_property_check_input(
            str(input_path),
            vectors_input="hexaaquairon_swap.movecs",
            vectors_output="hexaaquairon_prop.movecs",
            base_name="hexaaquairon_prop",
            title="hexaaquairon Mulliken spin population",
        )

        self.assertIn("start hexaaquairon_prop", drafted["input_text"].lower())
        self.assertIn('title "hexaaquairon mulliken spin population"', drafted["input_text"].lower())
        self.assertIn("iterations 1", drafted["input_text"].lower())
        self.assertIn("convergence energy 1e3", drafted["input_text"].lower())
        self.assertIn("vectors input hexaaquairon_swap.movecs output hexaaquairon_prop.movecs", drafted["input_text"].lower())
        self.assertIn("\nproperty\n  mulliken\nend\n", drafted["input_text"].lower())
        self.assertIn("task dft property", drafted["input_text"].lower())
        self.assertNotIn("\ndriver\n", drafted["input_text"].lower())

    def test_draft_nwchem_property_check_input_auto_uses_energy_for_suspicious_state(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw"
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"

        drafted = draft_nwchem_property_check_input(
            str(input_path),
            reference_output_path=str(output_path),
            vectors_input="hexaaquairon_swap.movecs",
            vectors_output="hexaaquairon_prop.movecs",
            base_name="hexaaquairon_prop",
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )

        self.assertEqual(drafted["selected_task_operation"], "energy")
        self.assertIn("task dft energy", drafted["input_text"].lower())
        self.assertNotIn("\nproperty\n", drafted["input_text"].lower())
        self.assertIn("mulliken", drafted["input_text"].lower())
        self.assertIn("auto_strategy_downgraded_to_energy_due_to_unstable_or_suspicious_state", drafted["strategy_notes"])

    def test_draft_nwchem_property_check_input_can_write_file(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw"

        with tempfile.TemporaryDirectory() as tmpdir:
            drafted = draft_nwchem_property_check_input(
                str(input_path),
                vectors_input="hexaaquairon_swap.movecs",
                output_dir=tmpdir,
                base_name="hexaaquairon_prop_check",
                write_file=True,
            )
            self.assertIsNotNone(drafted["written_file"])
            self.assertTrue(Path(drafted["written_file"]).is_file())
            self.assertTrue(drafted["written_file"].endswith("hexaaquairon_prop_check.nw"))

    def test_draft_nwchem_scf_stabilization_input(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw"
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.out"

        drafted = draft_nwchem_scf_stabilization_input(
            str(input_path),
            reference_output_path=str(output_path),
        )

        self.assertIn("task dft energy", drafted["input_text"].lower())
        self.assertEqual(drafted["stabilization_strategy"], "state_check_recovery")
        self.assertIn("iterations 120", drafted["input_text"].lower())
        self.assertIn("smear 0.001", drafted["input_text"].lower())
        self.assertIn("convergence damp 40", drafted["input_text"].lower())
        self.assertIn("convergence ncydp 80", drafted["input_text"].lower())
        self.assertIn("vectors input hexaaquairon_prop.movecs output hexaaquairon_prop_stabilize.movecs", drafted["input_text"].lower())
        self.assertIn("reference_run_failed_before_meaningful_scf_progress", drafted["strategy_notes"])

    def test_draft_nwchem_scf_stabilization_input_adapts_to_oscillatory_scf(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw"
        oscillatory_output = """
 Start of synthetic SCF
 Maximum number of iterations: 5
 convergence    iter
 d= 0,ls=0.0,diis    1   -100.0   -1.0   1.0E-02   1.0E-03   0.1
 d= 0,ls=0.0,diis    2   -101.0    1.0   9.0E-03   1.0E-03   0.2
 d= 0,ls=0.0,diis    3   -100.5   -0.5   8.5E-03   1.0E-03   0.3
 d= 0,ls=0.0,diis    4   -101.2    0.7   8.1E-03   1.0E-03   0.4
 d= 0,ls=0.0,diis    5   -100.8   -0.4   7.8E-03   1.0E-03   0.5
 Calculation failed to converge
 """

        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as handle:
            handle.write(oscillatory_output)
            output_path = Path(handle.name)
        try:
            drafted = draft_nwchem_scf_stabilization_input(
                str(input_path),
                reference_output_path=str(output_path),
            )
        finally:
            output_path.unlink(missing_ok=True)

        self.assertEqual(drafted["stabilization_strategy"], "oscillation_control")
        self.assertIn("iterations 250", drafted["input_text"].lower())
        self.assertIn("smear 0.005", drafted["input_text"].lower())
        self.assertIn("convergence damp 80", drafted["input_text"].lower())
        self.assertIn("convergence ncydp 120", drafted["input_text"].lower())
        self.assertIn("reference_scf_showed_energy_oscillation", drafted["strategy_notes"])

    def test_draft_nwchem_optimization_followup_input_auto_selects_freq_only(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.out"
        input_path = ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.nw"

        drafted = draft_nwchem_optimization_followup_input(
            str(output_path),
            str(input_path),
        )

        self.assertEqual(drafted["follow_up_plan"]["strategy"], "freq_only")
        self.assertIn("task dft freq", drafted["input_text"].lower())
        self.assertNotIn("task dft optimize", drafted["input_text"].lower())
        self.assertEqual(drafted["selected_frame"]["step"], drafted["trajectory_summary"]["last_step"])
        self.assertEqual(drafted["vectors_output"], "cu-opt_freq_followup.movecs")
        self.assertIn("vectors output cu-opt_freq_followup.movecs", drafted["input_text"].lower())

    def test_draft_nwchem_cube_input(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw"

        drafted = draft_nwchem_cube_input(
            str(input_path),
            vectors_input="hexaaquairon_swap.movecs",
            orbital_vectors=[76, 79],
            density_modes=["spindens"],
            base_name="hexaaquairon_cubes",
        )

        self.assertEqual(drafted["dplot_block_count"], 3)
        self.assertIn("start hexaaquairon_cubes", drafted["input_text"].lower())
        self.assertIn("vectors hexaaquairon_swap.movecs", drafted["input_text"].lower())
        self.assertIn("orbitals view; 1; 76; output hexaaquairon_cubes_mo_076.cube", drafted["input_text"].lower())
        self.assertIn("orbitals view; 1; 79; output hexaaquairon_cubes_mo_079.cube", drafted["input_text"].lower())
        self.assertIn("spin spindens", drafted["input_text"].lower())
        self.assertIn("output hexaaquairon_cubes_spindens.cube", drafted["input_text"].lower())
        self.assertIn("task dplot", drafted["input_text"].lower())
        self.assertNotIn("task dft energy", drafted["input_text"].lower())
        self.assertEqual(len(drafted["file_plan"]["cube_files"]), 3)

    def test_draft_nwchem_cube_input_can_write_file(self) -> None:
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw"

        with tempfile.TemporaryDirectory() as tmpdir:
            drafted = draft_nwchem_cube_input(
                str(input_path),
                vectors_input="hexaaquairon_swap.movecs",
                orbital_vectors=[76],
                density_modes=["total"],
                output_dir=tmpdir,
                base_name="hexaaquairon_cube_job",
                write_file=True,
            )
            self.assertIsNotNone(drafted["written_file"])
            self.assertTrue(Path(drafted["written_file"]).is_file())
            self.assertTrue(drafted["written_file"].endswith("hexaaquairon_cube_job.nw"))

    def test_draft_nwchem_frontier_cube_input(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_frag.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw"

        drafted = draft_nwchem_frontier_cube_input(
            str(output_path),
            str(input_path),
            vectors_input="hexaaquairon_frag.movecs",
            include_density_modes=["spindens"],
            base_name="hexaaquairon_frontier",
        )

        self.assertTrue(drafted["frontier_requests"])
        self.assertIn("alpha somo 1", drafted["input_text"].lower())
        self.assertIn("alpha lumo", drafted["input_text"].lower())
        self.assertIn("beta homo", drafted["input_text"].lower())
        self.assertIn("beta lumo", drafted["input_text"].lower())
        self.assertIn("spin alpha", drafted["input_text"].lower())
        self.assertIn("spin beta", drafted["input_text"].lower())
        self.assertIn("output hexaaquairon_frontier_spindens.cube", drafted["input_text"].lower())
        self.assertIn("task dplot", drafted["input_text"].lower())
        self.assertNotIn("alpha homo (vector 80)", drafted["input_text"].lower())

    def test_prepare_nwchem_run_with_json_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            profiles_path = self._write_runner_profiles_json(tmp_path)
            input_path = tmp_path / "job.nw"
            input_path.write_text("start test\n", encoding="utf-8")

            profiles = inspect_runner_profiles(str(profiles_path))
            self.assertIn("test_local", profiles["profile_names"])

            rendered = prepare_nwchem_run(
                str(input_path),
                profile="test_slurm",
                profiles_path=str(profiles_path),
                job_name="chemjob",
                resource_overrides={"mpi_ranks": 8},
            )

            self.assertEqual(rendered["launcher_kind"], "scheduler")
            self.assertIn("module purge", rendered["submit_script_text"])
            self.assertIn("module load nwchem", rendered["submit_script_text"])
            self.assertIn("echo ready", rendered["submit_script_text"])
            self.assertIn("srun nwchem job.nw", rendered["submit_script_text"])
            self.assertTrue(rendered["submit_script_path"].endswith("chemjob.slurm"))

    def test_runner_profiles_can_come_from_environment_variable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            profiles_path = self._write_runner_profiles_json(tmp_path)
            input_path = tmp_path / "job.nw"
            input_path.write_text("start test\n", encoding="utf-8")

            previous = os.environ.get("CHEMTOOLS_RUNNER_PROFILES")
            os.environ["CHEMTOOLS_RUNNER_PROFILES"] = str(profiles_path)
            try:
                profiles = inspect_runner_profiles()
                self.assertEqual(Path(profiles["profiles_path"]).resolve(), profiles_path.resolve())
                self.assertIn("test_local", profiles["profile_names"])

                rendered = prepare_nwchem_run(
                    str(input_path),
                    profile="test_local",
                )
                self.assertEqual(Path(rendered["profiles_path"]).resolve(), profiles_path.resolve())
                self.assertIn("printf launched > job.out", rendered["command"])
            finally:
                if previous is None:
                    os.environ.pop("CHEMTOOLS_RUNNER_PROFILES", None)
                else:
                    os.environ["CHEMTOOLS_RUNNER_PROFILES"] = previous

    def test_launch_nwchem_run_direct_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            profiles_path = self._write_runner_profiles_json(tmp_path)
            input_path = tmp_path / "job.nw"
            input_path.write_text("start test\n", encoding="utf-8")

            launched = launch_nwchem_run(
                str(input_path),
                profile="test_local",
                profiles_path=str(profiles_path),
                job_name="chemjob",
            )

            self.assertTrue(launched["executed"])
            self.assertEqual(launched["status"], "started")
            output_path = tmp_path / "chemjob.out"
            for _ in range(20):
                if output_path.exists():
                    break
                time.sleep(0.05)
            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.read_text(encoding="utf-8"), "launched")
            os.waitpid(launched["process_id"], 0)

    def test_check_nwchem_run_status_and_tail(self) -> None:
        output_path = ROOT / "nwchemaitest" / "uo2-test.out"
        status = check_nwchem_run_status(output_path=str(output_path))
        self.assertEqual(status["overall_status"], "completed_success")
        self.assertEqual(status["output_summary"]["outcome"], "success")

        tail = tail_nwchem_output(str(output_path), lines=20)
        self.assertEqual(tail["requested_lines"], 20)
        self.assertGreater(tail["returned_line_count"], 0)
        self.assertTrue(tail["last_nonempty_line"])

    def test_watch_nwchem_run_returns_immediately_for_completed_output(self) -> None:
        output_path = ROOT / "nwchemaitest" / "uo2-test.out"

        watched = watch_nwchem_run(
            output_path=str(output_path),
            poll_interval_seconds=0.0,
            max_polls=1,
        )

        self.assertTrue(watched["terminal"])
        self.assertEqual(watched["stop_reason"], "terminal_status")
        self.assertEqual(watched["poll_count"], 1)
        self.assertEqual(watched["final_status"]["overall_status"], "completed_success")
        self.assertTrue(watched["adaptive_polling"])
        self.assertEqual(watched["history_limit"], 8)

    def test_watch_nwchem_run_treats_zombie_process_as_finished(self) -> None:
        output_path = ROOT / "nwchemaitest" / "uo2-test.out"
        if not hasattr(os, "fork"):
            self.skipTest("requires os.fork")
        process_id = os.fork()
        if process_id == 0:  # pragma: no cover
            os._exit(0)
        try:
            deadline = time.time() + 2.0
            status = check_nwchem_run_status(output_path=str(output_path), process_id=process_id)
            while status["process"]["status"] == "running" and time.time() < deadline:
                time.sleep(0.05)
                status = check_nwchem_run_status(output_path=str(output_path), process_id=process_id)

            self.assertIn(status["process"]["status"], {"exited", "zombie", "not_found"})
            self.assertEqual(status["overall_status"], "completed_success")

            watched = watch_nwchem_run(
                output_path=str(output_path),
                process_id=process_id,
                poll_interval_seconds=0.0,
                max_polls=2,
            )

            self.assertTrue(watched["terminal"])
            self.assertEqual(watched["stop_reason"], "terminal_status")
            self.assertEqual(watched["final_status"]["overall_status"], "completed_success")
        finally:
            try:
                os.waitpid(process_id, 0)
            except ChildProcessError:
                pass

    def test_compute_watch_sleep_seconds_adaptive_backoff(self) -> None:
        self.assertEqual(
            _compute_watch_sleep_seconds(
                base_interval_seconds=10.0,
                stable_poll_count=0,
                adaptive_polling=True,
                max_poll_interval_seconds=60.0,
            ),
            10.0,
        )
        self.assertEqual(
            _compute_watch_sleep_seconds(
                base_interval_seconds=10.0,
                stable_poll_count=1,
                adaptive_polling=True,
                max_poll_interval_seconds=60.0,
            ),
            20.0,
        )
        self.assertEqual(
            _compute_watch_sleep_seconds(
                base_interval_seconds=10.0,
                stable_poll_count=4,
                adaptive_polling=True,
                max_poll_interval_seconds=60.0,
            ),
            60.0,
        )
        self.assertEqual(
            _compute_watch_sleep_seconds(
                base_interval_seconds=10.0,
                stable_poll_count=4,
                adaptive_polling=False,
                max_poll_interval_seconds=60.0,
            ),
            10.0,
        )

    def test_check_nwchem_run_status_for_incomplete_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path, output_path = self._write_partial_optfreq_case(Path(tmpdir))

            status = check_nwchem_run_status(output_path=str(output_path), input_path=str(input_path))

            self.assertEqual(status["overall_status"], "completed_incomplete")
            self.assertEqual(status["output_summary"]["outcome"], "incomplete")
            self.assertEqual(status["output_summary"]["current_task_kind"], "optimization")
            self.assertEqual(status["output_summary"]["current_phase"], "optimization_in_progress")
            self.assertIn("Optimization incomplete", status["output_summary"]["status_line"])
            self.assertEqual(status["progress_summary"]["optimization_status"], "incomplete")
            self.assertIsInstance(status["progress_summary"]["optimization_last_step"], int)
            self.assertGreaterEqual(status["progress_summary"]["optimization_last_step"], 0)
            self.assertFalse(status["progress_summary"]["frequency_started"])
            self.assertEqual(status["progress_summary"]["frequency_status"], "not_started")
            self.assertEqual(status["progress_summary"]["requested_task_count"], 2)
            self.assertEqual(status["progress_summary"]["requested_tasks"][0]["kind"], "optimization")
            self.assertEqual(status["progress_summary"]["requested_tasks"][1]["kind"], "frequency")
            self.assertEqual(status["progress_summary"]["completed_requested_task_count"], 0)
            self.assertEqual(status["progress_summary"]["current_requested_task"]["kind"], "optimization")
            self.assertEqual(status["progress_summary"]["next_requested_task"]["kind"], "frequency")
            self.assertEqual(status["progress_summary"]["observed_current_task"]["kind"], "optimization")

    def test_parse_tasks_ignores_title_as_basis_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, output_path = self._write_partial_optfreq_case(Path(tmpdir))

            payload = parse_tasks(str(output_path))
            raw_tasks = payload["program_summary"]["raw"]["tasks"]

            self.assertEqual(raw_tasks[0]["kind"], "optimization")
            self.assertEqual(raw_tasks[0]["basis"], "ao basis")

    def test_review_nwchem_progress_returns_compact_task_aware_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path, output_path = self._write_partial_optfreq_case(Path(tmpdir))

            payload = review_nwchem_progress(
                output_path=str(output_path),
                input_path=str(input_path),
            )

            self.assertEqual(payload["overall_status"], "completed_incomplete")
            self.assertTrue(payload["progress_headline"].startswith("DFT optimization is still running"))
            self.assertEqual(payload["current_phase"], "optimization_in_progress")
            self.assertEqual(payload["current_requested_task"]["kind"], "optimization")
            self.assertEqual(payload["next_requested_task"]["kind"], "frequency")
            self.assertIn(payload["progress_headline"], payload["summary_text"])
            self.assertEqual(payload["intervention"]["assessment"], "continue")
            self.assertFalse(payload["intervention"]["should_terminate_process"])

    def test_review_nwchem_progress_flags_oscillatory_scf_for_intervention(self) -> None:
        oscillatory_output = """
 NWChem SCF Module
 Start of synthetic SCF
 convergence    iter
 d= 0,ls=0.0,diis    1   -100.0   -1.0   1.0E-02   1.0E-03   0.1
 d= 0,ls=0.0,diis    2   -101.0    1.0   9.0E-03   1.0E-03   0.2
 d= 0,ls=0.0,diis    3   -100.5   -0.5   8.5E-03   1.0E-03   0.3
 d= 0,ls=0.0,diis    4   -101.2    0.7   8.1E-03   1.0E-03   0.4
 d= 0,ls=0.0,diis    5   -100.8   -0.4   7.8E-03   1.0E-03   0.5
 d= 0,ls=0.0,diis    6   -101.4    0.6   7.5E-03   1.0E-03   0.6
 d= 0,ls=0.0,diis    7   -100.9   -0.5   7.2E-03   1.0E-03   0.7
 d= 0,ls=0.0,diis    8   -101.5    0.6   7.0E-03   1.0E-03   0.8
 d= 0,ls=0.0,diis    9   -101.0   -0.5   6.8E-03   1.0E-03   0.9
 d= 0,ls=0.0,diis   10   -101.6    0.6   6.6E-03   1.0E-03   1.0
 d= 0,ls=0.0,diis   11   -101.1   -0.5   6.5E-03   1.0E-03   1.1
 d= 0,ls=0.0,diis   12   -101.7    0.6   6.4E-03   1.0E-03   1.2
 """
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as handle:
            handle.write(oscillatory_output)
            output_path = Path(handle.name)
        try:
            payload = review_nwchem_progress(output_path=str(output_path))
        finally:
            output_path.unlink(missing_ok=True)

        self.assertEqual(payload["intervention"]["assessment"], "kill_recommended")
        self.assertTrue(payload["intervention"]["should_terminate_process"])
        self.assertEqual(payload["intervention"]["recommended_action"], "kill_and_change_scf_strategy")

    def test_review_nwchem_progress_flags_divergent_optimization_for_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path, output_path = self._write_divergent_opt_case(Path(tmpdir))

            payload = review_nwchem_progress(
                output_path=str(output_path),
                input_path=str(input_path),
            )

            self.assertEqual(payload["current_phase"], "optimization_in_progress")
            self.assertEqual(payload["intervention"]["assessment"], "kill_recommended")
            self.assertTrue(payload["intervention"]["should_terminate_process"])
            self.assertEqual(
                payload["intervention"]["recommended_action"],
                "kill_and_restart_from_better_geometry_or_guess",
            )
            self.assertTrue(
                any("recent optimization energies rose" in reason for reason in payload["intervention"]["reasons"])
            )
            self.assertTrue(
                any("pair-distance change" in reason or "atomic displacement" in reason for reason in payload["intervention"]["reasons"])
            )
            self.assertTrue(
                any(
                    "nearest-neighbor separation" in reason or "metal-ligand dissociation" in reason
                    for reason in payload["intervention"]["reasons"]
                )
            )
            self.assertTrue(
                any("Fe1-O2" in reason for reason in payload["intervention"]["reasons"])
            )
            self.assertEqual(
                payload["intervention"]["primary_geometry_alert"]["kind"],
                "metal_ligand_dissociation",
            )
            self.assertEqual(
                payload["intervention"]["primary_geometry_alert"]["pair_label"],
                "Fe1-O2",
            )
            self.assertEqual(
                payload["intervention"]["primary_geometry_alert"]["fragment_label"],
                "O2",
            )
            self.assertTrue(payload["intervention"]["geometry_alerts"])

    def test_terminate_nwchem_run(self) -> None:
        process_id = os.spawnv(os.P_NOWAIT, "/bin/sh", ["/bin/sh", "-c", "sleep 30"])
        try:
            payload = terminate_nwchem_run(process_id=process_id, signal_name="term")
            self.assertTrue(payload["sent"])
            self.assertEqual(payload["signal"], "SIGTERM")
            waited_pid, _ = os.waitpid(process_id, 0)
            self.assertEqual(waited_pid, process_id)
        finally:
            try:
                os.kill(process_id, 9)
            except ProcessLookupError:
                pass

    def test_inspect_input_accepts_multiplicity_keyword(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = self._write_invalid_task_input(Path(tmpdir))

            payload = inspect_input(str(input_path))

            self.assertEqual(payload["charge"], -3)
            self.assertEqual(payload["multiplicity"], 6)

    def test_lint_flags_invalid_task_syntax_for_moduleless_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = self._write_invalid_task_input(Path(tmpdir))

            payload = lint_nwchem_input(str(input_path))
            codes = {issue["code"] for issue in payload["issues"]}

            self.assertIn("invalid_task_syntax", codes)
            self.assertNotIn("multiplicity_not_set", codes)

    def test_compare_nwchem_runs(self) -> None:
        reference_output = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        candidate_output = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.out"

        comparison = compare_nwchem_runs(
            str(reference_output),
            str(candidate_output),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )

        self.assertIsNotNone(comparison["energy_delta_hartree"])
        self.assertLess(comparison["energy_delta_hartree"], 0.0)
        self.assertIn("candidate_is_lower_in_energy", comparison["improved_signals"])
        self.assertIn(comparison["overall_assessment"], {"improved", "mixed"})

    def test_review_nwchem_followup_outcome(self) -> None:
        reference_output = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.out"
        candidate_output = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.out"
        reference_input = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw"
        candidate_input = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw"

        payload = review_nwchem_followup_outcome(
            str(reference_output),
            str(candidate_output),
            str(reference_input),
            str(candidate_input),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )

        self.assertIn("SCF now converges", payload["comparison_headline"])
        self.assertEqual(payload["comparison"]["candidate_summary"]["scf_status"], "converged")
        self.assertIsNotNone(payload["candidate_next_step"])
        self.assertEqual(payload["candidate_next_step"]["selected_workflow"], "wrong_state_swap_recovery")

    def test_prepare_nwchem_next_step_for_wrong_state(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw"

        payload = prepare_nwchem_next_step(
            output_path=str(output_path),
            input_path=str(input_path),
            expected_metal_elements=["Fe"],
            expected_somo_count=5,
        )

        self.assertEqual(payload["selected_workflow"], "wrong_state_swap_recovery")
        self.assertTrue(payload["can_auto_prepare"])
        self.assertIn("swap_restart", payload["prepared_artifacts"])
        self.assertIn("property_check", payload["prepared_artifacts"])
        self.assertEqual(
            payload["prepared_artifact_summaries"]["swap_restart"]["vectors_output"],
            payload["prepared_artifacts"]["swap_restart"]["vectors_output"],
        )
        self.assertEqual(
            payload["prepared_artifacts"]["property_check"]["vectors_input"],
            payload["prepared_artifacts"]["swap_restart"]["vectors_output"],
        )
        self.assertEqual(
            payload["prepared_artifact_summaries"]["property_check"]["vectors_input"],
            payload["prepared_artifacts"]["swap_restart"]["vectors_output"],
        )
        self.assertEqual(
            payload["prepared_artifacts"]["property_check"]["selected_task_operation"],
            "energy",
        )

    def test_prepare_nwchem_next_step_for_imaginary_mode(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "failed.out"
        input_path = ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "failed.nw"

        payload = prepare_nwchem_next_step(
            output_path=str(output_path),
            input_path=str(input_path),
        )

        self.assertEqual(payload["selected_workflow"], "imaginary_mode_follow_up")
        self.assertTrue(payload["can_auto_prepare"])
        self.assertIn("imaginary_mode_restarts", payload["prepared_artifacts"])
        self.assertEqual(
            payload["prepared_artifact_summaries"]["imaginary_mode_restarts"]["plus_vectors_output"],
            "failed_imaginary_followup_mode1_plus.movecs",
        )
        self.assertEqual(
            payload["prepared_artifact_summaries"]["imaginary_mode_restarts"]["minus_vectors_output"],
            "failed_imaginary_followup_mode1_minus.movecs",
        )

    def test_prepare_nwchem_next_step_for_optimization_review(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.out"
        input_path = ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.nw"

        payload = prepare_nwchem_next_step(
            output_path=str(output_path),
            input_path=str(input_path),
        )

        self.assertEqual(payload["selected_workflow"], "optimization_follow_up")
        self.assertTrue(payload["can_auto_prepare"])
        self.assertIn("optimization_follow_up", payload["prepared_artifacts"])
        self.assertEqual(
            payload["prepared_artifacts"]["optimization_follow_up"]["follow_up_plan"]["strategy"],
            "freq_only",
        )
        self.assertEqual(
            payload["prepared_artifact_summaries"]["optimization_follow_up"]["vectors_output"],
            "cu-opt_freq_followup.movecs",
        )

    def test_prepare_nwchem_next_step_for_interrupted_frequency_without_input(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "cmcc3h2_s" / "cmcc3h2_s_1.out"

        payload = prepare_nwchem_next_step(
            output_path=str(output_path),
        )

        self.assertEqual(payload["selected_workflow"], "post_optimization_frequency_follow_up")
        self.assertFalse(payload["can_auto_prepare"])

    def test_prepare_nwchem_next_step_for_scf_nonconvergence(self) -> None:
        output_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.out"
        input_path = ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw"

        payload = prepare_nwchem_next_step(
            output_path=str(output_path),
            input_path=str(input_path),
        )

        self.assertEqual(payload["selected_workflow"], "scf_stabilization_restart")
        self.assertTrue(payload["can_auto_prepare"])
        self.assertIn("scf_stabilization", payload["prepared_artifacts"])
        self.assertEqual(
            payload["prepared_artifact_summaries"]["scf_stabilization"]["vectors_output"],
            payload["prepared_artifacts"]["scf_stabilization"]["vectors_output"],
        )
        self.assertIn("prepared_artifacts_ready_for_local_review", payload["notes"])

    def test_prepare_nwchem_next_step_for_clean_run(self) -> None:
        output_path = ROOT / "nwchemaitest" / "uo2-test.out"

        payload = prepare_nwchem_next_step(output_path=str(output_path))

        self.assertEqual(payload["selected_workflow"], "verification_only")
        self.assertFalse(payload["can_auto_prepare"])
        self.assertEqual(payload["prepared_artifacts"], {})

    def test_evaluate_cases_on_seeded_training_cases(self) -> None:
        payload = evaluate_cases(str(ROOT / "nwchem-test" / "train"))

        self.assertGreaterEqual(payload["case_count"], 4)
        self.assertEqual(payload["failed_case_count"], 0)
        case_ids = {result["case_id"] for result in payload["results"]}
        self.assertIn("nwchem_hexaaquairon_wrong_state_001", case_ids)
        self.assertIn("nwchem_h2o2_imaginary_mode_001", case_ids)
        self.assertIn("nwchem_cmcc3h2_postopt_freq_interrupt_001", case_ids)


if __name__ == "__main__":
    unittest.main()
