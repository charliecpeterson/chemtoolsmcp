from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = ROOT / "chemtools" / "mcp" / "nwchem.py"


def load_server_module():
    spec = importlib.util.spec_from_file_location("chemtools.mcp.nwchem", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ChemtoolsMCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = load_server_module()

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

    def test_tools_list_contains_expected_names(self) -> None:
        response, should_exit = self.server.handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        self.assertFalse(should_exit)
        names = [tool["name"] for tool in response["result"]["tools"]]
        self.assertIn("parse_nwchem_output", names)
        self.assertIn("analyze_nwchem_frontier_orbitals", names)
        self.assertIn("suggest_nwchem_vectors_swaps", names)
        self.assertIn("draft_nwchem_vectors_swap_input", names)
        self.assertIn("draft_nwchem_property_check_input", names)
        self.assertIn("draft_nwchem_scf_stabilization_input", names)
        self.assertIn("draft_nwchem_optimization_followup_input", names)
        self.assertIn("draft_nwchem_cube_input", names)
        self.assertIn("draft_nwchem_frontier_cube_input", names)
        self.assertIn("compare_nwchem_runs", names)
        self.assertIn("review_nwchem_mcscf_case", names)
        self.assertIn("review_nwchem_mcscf_followup_outcome", names)
        self.assertIn("terminate_nwchem_run", names)
        self.assertIn("inspect_nwchem_runner_profiles", names)
        self.assertIn("launch_nwchem_run", names)
        self.assertIn("get_nwchem_run_status", names)
        self.assertIn("review_nwchem_progress", names)
        self.assertIn("watch_nwchem_run", names)
        self.assertIn("tail_nwchem_output", names)
        self.assertIn("prepare_nwchem_next_step", names)
        self.assertIn("parse_nwchem_mos", names)
        self.assertIn("parse_nwchem_mcscf_output", names)
        self.assertIn("parse_nwchem_population_analysis", names)
        self.assertIn("parse_cube_file", names)
        self.assertIn("analyze_nwchem_case", names)
        self.assertIn("lint_nwchem_input", names)
        self.assertIn("find_nwchem_restart_assets", names)
        self.assertIn("render_nwchem_basis_block", names)
        self.assertIn("check_nwchem_spin_charge_state", names)
        self.assertIn("suggest_nwchem_recovery", names)
        self.assertIn("suggest_nwchem_mcscf_active_space", names)
        self.assertIn("draft_nwchem_mcscf_input", names)
        self.assertIn("draft_nwchem_mcscf_retry_input", names)
        self.assertIn("review_nwchem_input_request", names)
        self.assertIn("create_nwchem_dft_input_from_request", names)
        self.assertIn("create_nwchem_dft_workflow_input", names)
        self.assertIn("render_nwchem_ecp_block", names)
        self.assertIn("render_nwchem_basis_setup", names)
        self.assertIn("create_nwchem_input", names)
        self.assertIn("summarize_nwchem_output", names)
        self.assertIn("analyze_nwchem_imaginary_modes", names)

    def test_initialize(self) -> None:
        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        )
        self.assertFalse(should_exit)
        self.assertEqual(response["result"]["serverInfo"]["name"], "chemtools-nwchem")

    def test_diagnose_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "diagnose_nwchem_output",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        diag = payload.get("diagnosis", payload)
        self.assertEqual(diag["failure_class"], "wrong_state_convergence")

    def test_diagnose_interrupted_frequency_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "cmcc3h2_s" / "cmcc3h2_s_1.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 301,
                "method": "tools/call",
                "params": {
                    "name": "diagnose_nwchem_output",
                    "arguments": {
                        "output_file": output_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        diag = payload.get("diagnosis", payload)
        self.assertEqual(diag["failure_class"], "post_optimization_frequency_interrupted")

    def test_parse_mos_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "parse_nwchem_mos",
                    "arguments": {
                        "file_path": output_file,
                        "top_n": 3,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["spin_channels"]["alpha"]["somo_count"], 5)

    def test_parse_population_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "cmcc3h2_s" / "cmcc3h2_s_1.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "parse_nwchem_population_analysis",
                    "arguments": {
                        "file_path": output_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["methods"]["mulliken"]["latest_total"]["atom_count"], 28)

    def test_analyze_frontier_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {
                    "name": "analyze_nwchem_frontier_orbitals",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["analysis"]["assessment"], "metal_state_mismatch_suspected")

    def test_suggest_vectors_swaps_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_frag.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "suggest_nwchem_vectors_swaps",
                    "arguments": {
                        "output_file": output_file,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                        "vectors_input": "hexaaquairon_frag.movecs",
                        "vectors_output": "hexaaquairon_swap.movecs",
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertTrue(payload["suggestion"]["available"])
        self.assertEqual(payload["suggestion"]["swap_pairs"][0]["from_vector"], 73)

    def test_draft_vectors_swap_input_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_frag.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {
                    "name": "draft_nwchem_vectors_swap_input",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["vectors_input"], "hexaaquairon_frag.movecs")
        self.assertIn("swap alpha 73 76", payload["input_text"].lower())
        self.assertIn("output hexaaquairon_swap.movecs", payload["input_text"].lower())

    def test_draft_property_check_input_tool_call(self) -> None:
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "tools/call",
                "params": {
                    "name": "draft_nwchem_property_check_input",
                    "arguments": {
                        "input_file": input_file,
                        "vectors_input": "hexaaquairon_swap.movecs",
                        "vectors_output": "hexaaquairon_prop.movecs",
                        "base_name": "hexaaquairon_prop",
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["vectors_input"], "hexaaquairon_swap.movecs")
        self.assertIn("task dft property", payload["input_text"].lower())
        self.assertIn("property\n  mulliken\nend", payload["input_text"].lower())

    def test_draft_property_check_input_auto_uses_energy_for_suspicious_state_tool_call(self) -> None:
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw")
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13_1,
                "method": "tools/call",
                "params": {
                    "name": "draft_nwchem_property_check_input",
                    "arguments": {
                        "input_file": input_file,
                        "reference_output_file": output_file,
                        "vectors_input": "hexaaquairon_swap.movecs",
                        "vectors_output": "hexaaquairon_prop.movecs",
                        "expected_metals": ["Fe"],
                        "expected_somos": 5
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["selected_task_operation"], "energy")
        self.assertIn("task dft energy", payload["input_text"].lower())

    def test_draft_scf_stabilization_input_tool_call(self) -> None:
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw")
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13_2,
                "method": "tools/call",
                "params": {
                    "name": "draft_nwchem_scf_stabilization_input",
                    "arguments": {
                        "input_file": input_file,
                        "reference_output_file": output_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertIn("task dft energy", payload["input_text"].lower())
        self.assertEqual(payload["stabilization_strategy"], "state_check_recovery")
        self.assertIn("convergence damp 40", payload["input_text"].lower())

    def test_create_nwchem_input_tool_call(self) -> None:
        geometry_file = str(ROOT / "nwchem-test" / "train" / "standard" / "uo2-test.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 130,
                "method": "tools/call",
                "params": {
                    "name": "create_nwchem_input",
                    "arguments": {
                        "geometry_file": geometry_file,
                        "basis_assignments": {"U": "lanl2dz_ecp"},
                        "ecp_assignments": {"U": "lanl2dz_ecp"},
                        "default_basis": "aug-cc-pvdz",
                        "module": "dft",
                        "task_operation": "energy",
                        "charge": 2,
                        "multiplicity": 3,
                        "start_name": "uo2_mixed",
                        "module_settings": ["xc pbe0"],
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["vectors_output"], "uo2_mixed.movecs")
        self.assertIn("vectors output uo2_mixed.movecs", payload["input_text"])
        self.assertIn("iterations 300", payload["input_text"].lower())
        self.assertIn("\n  mulliken\n", payload["input_text"].lower())
        self.assertIn("\n  direct\n", payload["input_text"].lower())
        self.assertIn("\n  noio\n", payload["input_text"].lower())
        self.assertIn("\n  grid nodisk\n", payload["input_text"].lower())
        self.assertNotIn("library lanl2dz_ecp", payload["input_text"])
        self.assertNotIn("library aug-cc-pvdz", payload["input_text"])
        self.assertIn("U    S", payload["input_text"])
        self.assertIn("O    S", payload["input_text"])
        self.assertIn("U nelec", payload["input_text"])

    def test_review_nwchem_input_request_tool_call(self) -> None:
        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13001,
                "method": "tools/call",
                "params": {
                    "name": "review_nwchem_input_request",
                    "arguments": {
                        "formula": "Fe2O3",
                        "default_basis": "cc-pvdz",
                        "module": "dft",
                        "task_operations": ["opt", "freq"],
                        "functional": "b3lyp",
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertFalse(payload["ready_to_create"])
        missing = {item["field"] for item in payload["missing_requirements"]}
        self.assertIn("geometry_source", missing)
        self.assertIn("multiplicity", missing)

    def test_create_nwchem_dft_workflow_input_tool_call(self) -> None:
        geometry_file = str(ROOT / "nwchem-test" / "train" / "standard" / "uo2-test.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13002,
                "method": "tools/call",
                "params": {
                    "name": "create_nwchem_dft_workflow_input",
                    "arguments": {
                        "geometry_file": geometry_file,
                        "basis_assignments": {"U": "lanl2dz_ecp"},
                        "ecp_assignments": {"U": "lanl2dz_ecp"},
                        "default_basis": "aug-cc-pvdz",
                        "xc_functional": "b3lyp",
                        "task_operations": ["opt", "freq"],
                        "charge": 2,
                        "multiplicity": 3,
                        "start_name": "uo2_optfreq"
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["vectors_output"], "uo2_optfreq.movecs")
        self.assertIn("iterations 300", payload["input_text"].lower())
        self.assertIn("\n  mulliken\n", payload["input_text"].lower())
        self.assertIn("\n  direct\n", payload["input_text"].lower())
        self.assertIn("\n  noio\n", payload["input_text"].lower())
        self.assertIn("\n  grid nodisk\n", payload["input_text"].lower())
        self.assertIn("\ndriver\n  maxiter 300\nend\n", payload["input_text"].lower())
        self.assertIn("task dft optimize", payload["input_text"].lower())
        self.assertIn("task dft freq", payload["input_text"].lower())

    def test_create_nwchem_dft_input_from_request_tool_call(self) -> None:
        geometry_file = str(ROOT / "nwchem-test" / "train" / "standard" / "uo2-test.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13002_1,
                "method": "tools/call",
                "params": {
                    "name": "create_nwchem_dft_input_from_request",
                    "arguments": {
                        "geometry_file": geometry_file,
                        "basis_assignments": {"U": "lanl2dz_ecp"},
                        "ecp_assignments": {"U": "lanl2dz_ecp"},
                        "default_basis": "aug-cc-pvdz",
                        "xc_functional": "b3lyp",
                        "task_operations": ["opt", "freq"],
                        "charge": 2,
                        "multiplicity": 3,
                        "start_name": "uo2_optfreq_request"
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertTrue(payload["ready_to_create"])
        self.assertTrue(payload["created"])
        self.assertIn("task dft optimize", payload["input_text"].lower())
        self.assertIn("task dft freq", payload["input_text"].lower())

    def test_lint_nwchem_input_tool_call(self) -> None:
        input_file = str(ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1301,
                "method": "tools/call",
                "params": {
                    "name": "lint_nwchem_input",
                    "arguments": {
                        "input_file": input_file,
                        "library_path": str(ROOT / "nwchem-test" / "nwchem_basis_library"),
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertIn(payload["status"], {"warning", "error"})
        self.assertTrue(any(issue["code"] == "missing_vectors_output" for issue in payload["issues"]))

    def test_find_restart_assets_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1302,
                "method": "tools/call",
                "params": {
                    "name": "find_nwchem_restart_assets",
                    "arguments": {
                        "path": output_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(Path(payload["preferred"]["vectors_file"]).name, "hexaaquairon.movecs")

    def test_find_restart_assets_related_match_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13021,
                "method": "tools/call",
                "params": {
                    "name": "find_nwchem_restart_assets",
                    "arguments": {
                        "path": output_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(Path(payload["preferred"]["vectors_file"]).name, "cu_h2o4_opt.movecs")

    def test_check_spin_charge_state_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1303,
                "method": "tools/call",
                "params": {
                    "name": "check_nwchem_spin_charge_state",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["assessment"], "suspicious")

    def test_summarize_nwchem_case_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1304,
                "method": "tools/call",
                "params": {
                    "name": "summarize_nwchem_case",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "library_path": str(ROOT / "nwchem-test" / "nwchem_basis_library"),
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["spin_charge_state"]["assessment"], "suspicious")
        self.assertEqual(payload["next_step"]["selected_workflow"], "wrong_state_swap_recovery")

    def test_summarize_nwchem_case_compact_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1305,
                "method": "tools/call",
                "params": {
                    "name": "summarize_nwchem_case",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "library_path": str(ROOT / "nwchem-test" / "nwchem_basis_library"),
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                        "compact": True,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertNotIn("diagnosis_summary", payload)
        self.assertEqual(payload["next_step"]["selected_workflow"], "wrong_state_swap_recovery")

    def test_review_nwchem_case_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1306,
                "method": "tools/call",
                "params": {
                    "name": "review_nwchem_case",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "library_path": str(ROOT / "nwchem-test" / "nwchem_basis_library"),
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertNotIn("diagnosis_summary", payload)
        self.assertEqual(payload["spin_charge_state"]["assessment"], "suspicious")
        self.assertEqual(payload["next_step"]["selected_workflow"], "wrong_state_swap_recovery")

    def test_suggest_nwchem_scf_fix_strategy_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1307,
                "method": "tools/call",
                "params": {
                    "name": "suggest_nwchem_scf_fix_strategy",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["strategy_family"], "scf_recovery")
        self.assertEqual(payload["primary_strategy"], "stabilization_restart")
        self.assertEqual(payload["strategies"][0]["name"], "stabilization_restart")

    def test_suggest_nwchem_state_recovery_strategy_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1308,
                "method": "tools/call",
                "params": {
                    "name": "suggest_nwchem_state_recovery_strategy",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["regime"], "metal_state_mismatch")
        self.assertEqual(payload["primary_strategy"], "vectors_swap_restart")
        self.assertEqual(payload["strategies"][1]["name"], "fragment_guess_seed")

    def test_suggest_nwchem_mcscf_active_space_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1309,
                "method": "tools/call",
                "params": {
                    "name": "suggest_nwchem_mcscf_active_space",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["frontier_assessment"], "metal_state_mismatch_suspected")
        self.assertIn(79, payload["minimal_active_space"]["vector_numbers"])
        self.assertIn(80, payload["minimal_active_space"]["vector_numbers"])

    def test_parse_nwchem_mcscf_output_tool_call(self) -> None:
        output_file = str(ROOT / "nwchemaitest-2" / "fecl3_mcscf_test_retry2.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13095,
                "method": "tools/call",
                "params": {
                    "name": "parse_nwchem_mcscf_output",
                    "arguments": {
                        "file_path": output_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["status"], "converged")
        self.assertEqual(payload["settings"]["active_orbitals"], 8)
        self.assertTrue(payload["converged_ci_vector"])
        self.assertGreater(payload["precondition_warning_count"], 0)

    def test_review_nwchem_mcscf_case_tool_call(self) -> None:
        output_file = str(ROOT / "nwchemaitest-2" / "fecl3_mcscf_test_retry2.out")
        input_file = str(ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13096,
                "method": "tools/call",
                "params": {
                    "name": "review_nwchem_mcscf_case",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "expected_metals": ["Fe"],
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["status"], "converged")
        self.assertEqual(payload["recommended_next_action"], "use_mcscf_as_reference_or_seed_for_follow_up")
        self.assertEqual(payload["active_space_density_review"]["assessment"], "metal_participation_significant")

    def test_review_nwchem_mcscf_followup_outcome_tool_call(self) -> None:
        reference_output_file = str(ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.out")
        candidate_output_file = str(ROOT / "nwchemaitest-2" / "fecl3_mcscf_test_retry2.out")
        input_file = str(ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13098,
                "method": "tools/call",
                "params": {
                    "name": "review_nwchem_mcscf_followup_outcome",
                    "arguments": {
                        "reference_output_file": reference_output_file,
                        "candidate_output_file": candidate_output_file,
                        "reference_input_file": input_file,
                        "candidate_input_file": input_file,
                        "expected_metals": ["Fe"],
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["comparison"]["overall_assessment"], "improved")
        self.assertIsNone(payload["candidate_next_step"])
        self.assertIn("MCSCF now converges", payload["comparison_headline"])

    def test_draft_nwchem_mcscf_retry_input_tool_call(self) -> None:
        output_file = str(ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.out")
        input_file = str(ROOT / "nwchemaitest-2" / "fecl3_mcscf_test.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13097,
                "method": "tools/call",
                "params": {
                    "name": "draft_nwchem_mcscf_retry_input",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "expected_metals": ["Fe"],
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["retry_strategy"], "syntax_cleanup_retry")
        self.assertEqual(payload["resolved_settings"]["hessian"], "exact")
        self.assertIn("task mcscf", payload["input_text"].lower())
        self.assertNotIn("state sextet", payload["input_text"].lower())

    def test_draft_nwchem_mcscf_input_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1310,
                "method": "tools/call",
                "params": {
                    "name": "draft_nwchem_mcscf_input",
                    "arguments": {
                        "reference_output_file": output_file,
                        "input_file": input_file,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["active_space"]["active_orbitals"], 10)
        self.assertEqual(payload["task_lines"], ["task mcscf"])
        self.assertIn("actelec 11", payload["module_block_text"].lower())
        self.assertIn("hessian exact", payload["module_block_text"].lower())
        self.assertTrue(payload["reorder_plan"]["swap_pairs"])

    def test_draft_optimization_followup_input_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 131,
                "method": "tools/call",
                "params": {
                    "name": "draft_nwchem_optimization_followup_input",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["follow_up_plan"]["strategy"], "freq_only")
        self.assertIn("task dft freq", payload["input_text"].lower())
        self.assertEqual(payload["vectors_output"], "cu-opt_freq_followup.movecs")
        self.assertIn("vectors output cu-opt_freq_followup.movecs", payload["input_text"].lower())

    def test_draft_cube_input_tool_call(self) -> None:
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 14,
                "method": "tools/call",
                "params": {
                    "name": "draft_nwchem_cube_input",
                    "arguments": {
                        "input_file": input_file,
                        "vectors_input": "hexaaquairon_swap.movecs",
                        "orbital_vectors": [76],
                        "density_modes": ["spindens"],
                        "base_name": "hexaaquairon_cubes",
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertIn("task dplot", payload["input_text"].lower())
        self.assertIn("orbitals view; 1; 76; output hexaaquairon_cubes_mo_076.cube", payload["input_text"].lower())
        self.assertIn("output hexaaquairon_cubes_spindens.cube", payload["input_text"].lower())

    def test_draft_frontier_cube_input_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_frag.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 15,
                "method": "tools/call",
                "params": {
                    "name": "draft_nwchem_frontier_cube_input",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "vectors_input": "hexaaquairon_frag.movecs",
                        "include_density_modes": ["spindens"],
                        "base_name": "hexaaquairon_frontier",
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertIn("spin alpha", payload["input_text"].lower())
        self.assertIn("spin beta", payload["input_text"].lower())
        self.assertIn("output hexaaquairon_frontier_spindens.cube", payload["input_text"].lower())

    def test_prepare_nwchem_run_tool_call(self) -> None:
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 16,
                "method": "tools/call",
                "params": {
                    "name": "prepare_nwchem_run",
                    "arguments": {
                        "input_file": input_file,
                        "profile": "local",
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["launcher_kind"], "direct")
        self.assertIn("hexaaquairon_swap.nw", payload["command"])

    def test_inspect_runner_profiles_tool_call(self) -> None:
        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 17,
                "method": "tools/call",
                "params": {
                    "name": "inspect_nwchem_runner_profiles",
                    "arguments": {},
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertIn("local", payload["profile_names"])

    def test_check_run_status_and_tail_tool_calls(self) -> None:
        output_file = str(ROOT / "nwchemaitest" / "uo2-test.out")

        status_response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 18,
                "method": "tools/call",
                "params": {
                    "name": "check_nwchem_run_status",
                    "arguments": {
                        "output_file": output_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        status_payload = status_response["result"]["structuredContent"]
        self.assertEqual(status_payload["overall_status"], "completed_success")

        tail_response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 19,
                "method": "tools/call",
                "params": {
                    "name": "tail_nwchem_output",
                    "arguments": {
                        "output_file": output_file,
                        "lines": 10,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        tail_payload = tail_response["result"]["structuredContent"]
        self.assertEqual(tail_payload["requested_lines"], 10)
        self.assertGreater(tail_payload["returned_line_count"], 0)

    def test_check_run_status_for_incomplete_optimization_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path, output_path = self._write_partial_optfreq_case(Path(tmpdir))
            output_file = str(output_path)
            input_file = str(input_path)

            response, should_exit = self.server.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 19_1,
                    "method": "tools/call",
                    "params": {
                        "name": "check_nwchem_run_status",
                        "arguments": {
                            "output_file": output_file,
                            "input_file": input_file,
                        },
                    },
                }
            )
            self.assertFalse(should_exit)
            payload = response["result"]["structuredContent"]
            self.assertEqual(payload["overall_status"], "completed_incomplete")
            self.assertEqual(payload["output_summary"]["current_phase"], "optimization_in_progress")
            self.assertIn("Optimization incomplete", payload["output_summary"]["status_line"])
            self.assertIsInstance(payload["progress_summary"]["optimization_last_step"], int)
            self.assertGreaterEqual(payload["progress_summary"]["optimization_last_step"], 0)
            self.assertFalse(payload["progress_summary"]["frequency_started"])
            self.assertEqual(payload["progress_summary"]["requested_tasks"][0]["kind"], "optimization")
            self.assertEqual(payload["progress_summary"]["next_requested_task"]["kind"], "frequency")

    def test_review_nwchem_progress_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path, output_path = self._write_partial_optfreq_case(Path(tmpdir))
            output_file = str(output_path)
            input_file = str(input_path)

            response, should_exit = self.server.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 19_2,
                    "method": "tools/call",
                    "params": {
                        "name": "review_nwchem_progress",
                        "arguments": {
                            "output_file": output_file,
                            "input_file": input_file,
                        },
                    },
                }
            )
            self.assertFalse(should_exit)
            payload = response["result"]["structuredContent"]
            self.assertEqual(payload["overall_status"], "completed_incomplete")
            self.assertTrue(payload["progress_headline"].startswith("DFT optimization is still running"))
            self.assertEqual(payload["current_phase"], "optimization_in_progress")
            self.assertEqual(payload["intervention"]["assessment"], "continue")
            self.assertEqual(payload["current_requested_task"]["kind"], "optimization")
            self.assertEqual(payload["next_requested_task"]["kind"], "frequency")
            self.assertIn(payload["progress_headline"], payload["summary_text"])

    def test_review_nwchem_progress_tool_call_flags_divergent_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file, output_file = self._write_divergent_opt_case(Path(tmpdir))

            response, should_exit = self.server.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 19_3,
                    "method": "tools/call",
                    "params": {
                        "name": "review_nwchem_progress",
                        "arguments": {
                            "output_file": str(output_file),
                            "input_file": str(input_file),
                        },
                    },
                }
            )
            self.assertFalse(should_exit)
            payload = response["result"]["structuredContent"]
            self.assertEqual(payload["intervention"]["assessment"], "kill_recommended")
            self.assertEqual(
                payload["intervention"]["recommended_action"],
                "kill_and_restart_from_better_geometry_or_guess",
            )
            self.assertEqual(
                payload["intervention"]["primary_geometry_alert"]["kind"],
                "metal_ligand_dissociation",
            )
            self.assertEqual(
                payload["intervention"]["primary_geometry_alert"]["pair_label"],
                "Fe1-O2",
            )

    def test_watch_nwchem_run_tool_call(self) -> None:
        output_file = str(ROOT / "nwchemaitest" / "uo2-test.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 8201,
                "method": "tools/call",
                "params": {
                    "name": "watch_nwchem_run",
                    "arguments": {
                        "output_file": output_file,
                        "poll_interval_seconds": 0.0,
                        "max_polls": 1,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertTrue(payload["terminal"])
        self.assertEqual(payload["stop_reason"], "terminal_status")
        self.assertEqual(payload["final_status"]["overall_status"], "completed_success")
        self.assertIn("completed_success", payload["final_progress"]["overall_status"])
        self.assertTrue(payload["adaptive_polling"])
        self.assertEqual(payload["history_limit"], 8)

    def test_terminate_nwchem_run_tool_call(self) -> None:
        process_id = os.spawnv(os.P_NOWAIT, "/bin/sh", ["/bin/sh", "-c", "sleep 30"])
        try:
            response, should_exit = self.server.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 8202,
                    "method": "tools/call",
                    "params": {
                        "name": "terminate_nwchem_run",
                        "arguments": {
                            "process_id": process_id,
                            "signal_name": "term",
                        },
                    },
                }
            )
            self.assertFalse(should_exit)
            payload = response["result"]["structuredContent"]
            self.assertTrue(payload["sent"])
            self.assertEqual(payload["signal"], "SIGTERM")
            waited_pid, _ = os.waitpid(process_id, 0)
            self.assertEqual(waited_pid, process_id)
        finally:
            try:
                os.kill(process_id, 9)
            except ProcessLookupError:
                pass

    def test_compare_nwchem_runs_tool_call(self) -> None:
        reference_output = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        candidate_output = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 20,
                "method": "tools/call",
                "params": {
                    "name": "compare_nwchem_runs",
                    "arguments": {
                        "reference_output_file": reference_output,
                        "candidate_output_file": candidate_output,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertLess(payload["energy_delta_hartree"], 0.0)
        self.assertIn("candidate_is_lower_in_energy", payload["improved_signals"])

    def test_prepare_nwchem_next_step_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "tools/call",
                "params": {
                    "name": "prepare_nwchem_next_step",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["selected_workflow"], "wrong_state_swap_recovery")
        self.assertIn("swap_restart", payload["prepared_artifacts"])
        self.assertEqual(
            payload["prepared_artifact_summaries"]["swap_restart"]["vectors_output"],
            payload["prepared_artifacts"]["swap_restart"]["vectors_output"],
        )

    def test_review_nwchem_followup_outcome_tool_call(self) -> None:
        reference_output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.out")
        candidate_output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.out")
        reference_input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw")
        candidate_input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_swap.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 21_01,
                "method": "tools/call",
                "params": {
                    "name": "review_nwchem_followup_outcome",
                    "arguments": {
                        "reference_output_file": reference_output_file,
                        "candidate_output_file": candidate_output_file,
                        "reference_input_file": reference_input_file,
                        "candidate_input_file": candidate_input_file,
                        "expected_metals": ["Fe"],
                        "expected_somos": 5,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertIn("SCF now converges", payload["comparison_headline"])
        self.assertEqual(payload["comparison"]["candidate_summary"]["scf_status"], "converged")
        self.assertEqual(payload["candidate_next_step"]["selected_workflow"], "wrong_state_swap_recovery")

    def test_prepare_nwchem_next_step_for_scf_nonconvergence_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "failed" / "hexaaquairon_prop.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 21_1,
                "method": "tools/call",
                "params": {
                    "name": "prepare_nwchem_next_step",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["selected_workflow"], "scf_stabilization_restart")
        self.assertTrue(payload["can_auto_prepare"])
        self.assertIn("scf_stabilization", payload["prepared_artifacts"])

    def test_prepare_nwchem_next_step_for_optimization_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.out")
        input_file = str(ROOT / "nwchem-test" / "train" / "standard" / "cu-opt.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "tools/call",
                "params": {
                    "name": "prepare_nwchem_next_step",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["selected_workflow"], "optimization_follow_up")
        self.assertEqual(
            payload["prepared_artifacts"]["optimization_follow_up"]["follow_up_plan"]["strategy"],
            "freq_only",
        )
        self.assertEqual(
            payload["prepared_artifact_summaries"]["optimization_follow_up"]["vectors_output"],
            "cu-opt_freq_followup.movecs",
        )

    def test_prepare_nwchem_next_step_for_interrupted_frequency_without_input_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "cmcc3h2_s" / "cmcc3h2_s_1.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 23,
                "method": "tools/call",
                "params": {
                    "name": "prepare_nwchem_next_step",
                    "arguments": {
                        "output_file": output_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["selected_workflow"], "post_optimization_frequency_follow_up")
        self.assertFalse(payload["can_auto_prepare"])

    def test_summarize_cube_tool_call(self) -> None:
        cube_file = str(ROOT / "nwchem-test" / "train" / "standard" / "chargedensity.cube")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "summarize_cube_file",
                    "arguments": {
                        "file_path": cube_file,
                        "top_atoms": 2,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["dataset_kind"], "density")

    def test_render_basis_tool_call(self) -> None:
        input_file = str(ROOT / "nwchem-test" / "train" / "ferrocene" / "successful" / "ferrocene.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "render_nwchem_basis_from_input",
                    "arguments": {
                        "basis_name": "def2-svp",
                        "input_file": input_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertNotIn("library def2-svp", payload["text"])
        self.assertIn("Fe    S", payload["text"])

    def test_summarize_tool_call(self) -> None:
        output_file = str(ROOT / "nwchemaitest" / "uo2-test.out")
        input_file = str(ROOT / "nwchemaitest" / "uo2-test.nw")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "summarize_nwchem_output",
                    "arguments": {
                        "output_file": output_file,
                        "input_file": input_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["failure_class"], "no_clear_failure_detected")
        self.assertIn("Outcome:", payload["summary_text"])

    def test_analyze_imaginary_modes_tool_call(self) -> None:
        output_file = str(ROOT / "nwchem-test" / "train" / "h2o2_imaginary_freq" / "failed.out")

        response, should_exit = self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "analyze_nwchem_imaginary_modes",
                    "arguments": {
                        "output_file": output_file,
                    },
                },
            }
        )
        self.assertFalse(should_exit)
        payload = response["result"]["structuredContent"]
        self.assertEqual(payload["significant_imaginary_mode_count"], 1)


if __name__ == "__main__":
    unittest.main()
