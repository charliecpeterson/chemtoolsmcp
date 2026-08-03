"""Regression tests for the initial QMCPACK XML input review."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chemtools.application.input_review import (
    detect_input_backend,
    review_input,
)
from chemtools.application.run_inspection import inspect_run
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend
from chemtools.programs.qmcpack import QMCPACK
from chemtools.programs.qmcpack.input import (
    lint_qmcpack_input,
    parse_qmcpack_input,
    parse_qmcpack_ion_geometries,
)
from chemtools.programs.qmcpack.includes import inspect_xml_includes
from chemtools.programs.qmcpack.particles import (
    collect_ion_geometries,
    collect_particle_sets,
    electron_particle_count,
    non_electron_particle_sets,
)
from chemtools.programs.qmcpack.sidecars import (
    find_referenced_hdf5,
    inspect_pwscf_h5_reference,
)
from chemtools.programs.qmcpack.output import parse_qmcpack_output_text


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_QMCPACK_FIXTURES = Path(__file__).parent / "fixtures" / "qmcpack"


def _write_hdf5(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(_HDF5_SIGNATURE)
    return path


PARTICLES = """\
<?xml version="1.0"?>
<qmcsystem>
  <simulationcell/>
  <particleset name="ion0" size="1">
    <group name="Ce"><parameter name="valence">12</parameter></group>
  </particleset>
  <particleset name="e" random="yes">
    <group name="u" size="5"/>
    <group name="d" size="4"/>
  </particleset>
</qmcsystem>
"""


SIMULATION = """\
<?xml version="1.0"?>
<simulation>
  <project id="Ce_secp_ion3f1_prod" series="0"/>
  <include href="Ce_secp_ion3f1.ptcl.xml"/>
  <include href="Ce_secp_ion3f1.wfj.xml"/>
  <hamiltonian name="h0" target="e">
    <pairpot name="PseudoPot" type="pseudo">
      <pseudo elementType="Ce" href="Ce.secp-sl.xml"/>
    </pairpot>
  </hamiltonian>
  <loop max="6">
    <qmc method="linear" move="pbyp">
      <parameter name="blocks">40</parameter>
      <parameter name="timestep">0.5</parameter>
    </qmc>
  </loop>
  <qmc method="vmc" move="pbyp">
    <parameter name="timestep">0.02</parameter>
  </qmc>
  <qmc method="dmc" move="pbyp">
    <parameter name="timestep">0.005</parameter>
  </qmc>
</simulation>
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_qmcsystem_input_preserves_particle_sets(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "Ce.ptcl.xml", PARTICLES))

    assert parsed["format"] == "qmcpack-input/1"
    assert parsed["root"] == "qmcsystem"
    assert parsed["project"] is None
    assert parsed["particle_sets"] == [
        {
            "name": "ion0",
            "size": "1",
            "groups": [{
                "name": "Ce",
                "size": None,
                "parameters": {"valence": "12"},
            }],
        },
        {
            "name": "e",
            "size": None,
            "groups": [
                {"name": "u", "size": "5"},
                {"name": "d", "size": "4"},
            ],
        },
    ]


def test_parse_qmcpack_ion_geometry_reads_explicit_bohr_cell_and_positions(tmp_path):
    source = _write(tmp_path, "ions.xml", """\
<qmcsystem>
  <simulationcell>
    <parameter name="lattice" units="bohr">24 0 0 0 24 0 0 0 24</parameter>
    <parameter name="bconds">p p p</parameter>
  </simulationcell>
  <particleset name="ion0" size="1">
    <group name="Ce"><attrib name="position">12 12 12</attrib></group>
  </particleset>
</qmcsystem>
""")

    assert parse_qmcpack_ion_geometries(source) == [{
        "particle_set": "ion0",
        "cell": {
            "lattice": {
                "units": "bohr",
                "vectors": [[24.0, 0.0, 0.0], [0.0, 24.0, 0.0], [0.0, 0.0, 24.0]],
            },
            "boundary_conditions": ["p", "p", "p"],
        },
        "atoms": [{"label": "Ce", "coordinates": [12.0, 12.0, 12.0]}],
        "status": "complete",
    }]


def test_collect_particle_sets_includes_present_xml_children(tmp_path):
    source = _write(tmp_path, "main.xml", """\
<simulation>
  <include href="particles.xml"/>
</simulation>
""")
    _write(tmp_path, "particles.xml", PARTICLES)
    parsed = parse_qmcpack_input(source)

    assert collect_particle_sets(
        parsed,
        inspect_xml_includes(source, parsed),
    ) == parse_qmcpack_input(tmp_path / "particles.xml")["particle_sets"]


def test_collect_ion_geometries_includes_present_xml_children(tmp_path):
    source = _write(tmp_path, "main.xml", """\
<simulation><include href="ions.xml"/></simulation>
""")
    ions = _write(tmp_path, "ions.xml", """\
<qmcsystem>
  <simulationcell><parameter name="lattice" units="bohr">1 0 0 0 1 0 0 0 1</parameter></simulationcell>
  <particleset name="ion0" size="1"><group name="H"><attrib name="position">0 0 0</attrib></group></particleset>
</qmcsystem>
""")
    parsed = parse_qmcpack_input(source)

    assert collect_ion_geometries(parsed, inspect_xml_includes(source, parsed)) == [{
        "particle_set": "ion0",
        "cell": {
            "lattice": {
                "units": "bohr",
                "vectors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            "boundary_conditions": None,
        },
        "atoms": [{"label": "H", "coordinates": [0.0, 0.0, 0.0]}],
        "status": "complete",
        "source_path": str(ions),
    }]


def test_particle_count_summaries_use_hamiltonian_target(tmp_path):
    source = _write(tmp_path, "main.xml", """\
<simulation>
  <particleset name="ion0" size="1"><group name="Ce"/></particleset>
  <particleset name="e"><group name="u" size="5"/><group name="d" size="4"/></particleset>
  <hamiltonian name="h0" target="e"/>
</simulation>
""")
    parsed = parse_qmcpack_input(source)
    include_review = inspect_xml_includes(source, parsed)

    assert electron_particle_count(parsed, include_review) == {
        "status": "complete",
        "electron_count": 9,
        "hamiltonian_targets": ["e"],
        "matching_particle_sets": [parsed["particle_sets"][1]],
        "include_review_status": "not_applicable",
    }
    assert non_electron_particle_sets(parsed, include_review) == {
        "status": "complete",
        "particle_count": 1,
        "hamiltonian_targets": ["e"],
        "qmcpack_non_electron_particle_sets": [parsed["particle_sets"][0]],
        "include_review_status": "not_applicable",
    }


def test_find_referenced_hdf5_matches_resolved_path(tmp_path):
    sidecar = _write(tmp_path, "orbitals.h5", "HDF5")
    review = {"entries": [{"path": str(sidecar.resolve()), "status": "present"}]}

    assert find_referenced_hdf5(review, sidecar) == review["entries"]
    assert find_referenced_hdf5(review, tmp_path / "other.h5") == []


def test_parse_simulation_input_records_qmc_and_pseudopotential_references(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "Ce.xml", SIMULATION))

    assert parsed["project"] == {
        "id": "Ce_secp_ion3f1_prod",
        "series": "0",
    }
    assert parsed["includes"] == [
        "Ce_secp_ion3f1.ptcl.xml",
        "Ce_secp_ion3f1.wfj.xml",
    ]
    assert parsed["hamiltonians"] == [{
        "name": "h0",
        "target": "e",
        "pseudopotentials": [{
            "element": "Ce",
            "href": "Ce.secp-sl.xml",
        }],
    }]
    assert [block["method"] for block in parsed["qmc_blocks"]] == [
        "linear", "vmc", "dmc",
    ]
    assert parsed["qmc_blocks"][0]["parameters"] == {
        "blocks": "40",
        "timestep": "0.5",
    }
    assert parsed["qmc_blocks"][0]["costs"] == []


def test_parse_qmcpack_input_summarizes_dmc_campaign_shape(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "campaign.xml", """\
<simulation>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.005</parameter><parameter name="blocks">100</parameter><parameter name="targetWalkers">960</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.0025</parameter><parameter name="blocks">200</parameter><parameter name="targetWalkers">960</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">no</parameter><parameter name="timestep">0.0025</parameter><parameter name="blocks">200</parameter><parameter name="targetWalkers">960</parameter></qmc>
</simulation>
"""))

    assert parsed["dmc_campaign"] == {
        "dmc_blocks": [
            {
                "qmc_block_index": 0,
                "timestep": 0.005,
                "blocks": 100,
                "target_walkers": 960,
                "nonlocalmoves": True,
            },
            {
                "qmc_block_index": 1,
                "timestep": 0.0025,
                "blocks": 200,
                "target_walkers": 960,
                "nonlocalmoves": True,
            },
            {
                "qmc_block_index": 2,
                "timestep": 0.0025,
                "blocks": 200,
                "target_walkers": 960,
                "nonlocalmoves": False,
            },
        ],
        "production_protocol": {
            "status": "not_assessed",
            "reason": (
                "the reference production order needs linear optimization, VMC, "
                "and DMC blocks"
            ),
            "observed_methods": ["dmc", "dmc", "dmc"],
        },
        "tmove_ladder": {
            "status": "assessed",
            "blocks": [
                {
                    "qmc_block_index": 0,
                    "timestep": 0.005,
                    "blocks": 100,
                    "target_walkers": 960,
                    "nonlocalmoves": True,
                },
                {
                    "qmc_block_index": 1,
                    "timestep": 0.0025,
                    "blocks": 200,
                    "target_walkers": 960,
                    "nonlocalmoves": True,
                },
            ],
            "timesteps_strictly_decrease": True,
            "block_counts_strictly_increase": True,
            "matches_fblock_reference_timestep_ladder": False,
        },
        "no_tmove_control": {
            "blocks": [{
                "qmc_block_index": 2,
                "timestep": 0.0025,
                "blocks": 200,
                "target_walkers": 960,
                "nonlocalmoves": False,
            }],
            "matching_tmove_timestep": True,
            "middle_timestep_control": {
                "status": "not_assessed",
                "reason": (
                    "at least three distinct T-move time steps are needed to identify "
                    "an interior control point"
                ),
            },
            "matching_tmove_settings": {
                "status": "assessed",
                "comparisons": [{
                    "no_tmove_qmc_block_index": 2,
                    "tmove_qmc_block_indices": [1],
                    "block_count_match": True,
                    "steps_match": None,
                    "warmup_steps_match": None,
                    "target_walkers_match": True,
                    "move_match": None,
                    "checkpoint_match": None,
                    "all_declared_settings_match": None,
                }],
            },
        },
        "declared_target_walkers": [960],
    }


def test_parse_qmcpack_input_accepts_total_walkers_for_dmc_campaign(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "campaign.xml", """\
<simulation>
  <qmc method="dmc"><parameter name="total_walkers">2240</parameter></qmc>
</simulation>
"""))

    assert parsed["dmc_campaign"]["dmc_blocks"] == [{
        "qmc_block_index": 0,
        "timestep": None,
        "blocks": None,
        "target_walkers": 2240,
        "nonlocalmoves": None,
    }]
    assert parsed["dmc_campaign"]["declared_target_walkers"] == [2240]


def test_qmcpack_dmc_campaign_flags_an_endpoint_tmove_control(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "campaign.xml", """\
<simulation>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.005</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.0025</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.00125</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">no</parameter><parameter name="timestep">0.005</parameter></qmc>
</simulation>
"""))

    assert parsed["dmc_campaign"]["no_tmove_control"]["middle_timestep_control"] == {
        "status": "assessed",
        "matching_tmove_timesteps": [0.005],
        "interior_tmove_timesteps": [0.0025],
        "control_count_matches_reference": True,
        "all_controls_match_interior_tmove_timestep": False,
    }


def test_qmcpack_dmc_campaign_flags_multiple_tmove_controls(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "campaign.xml", """\
<simulation>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.005</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.0025</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.00125</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">no</parameter><parameter name="timestep">0.0025</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">no</parameter><parameter name="timestep">0.00125</parameter></qmc>
</simulation>
"""))

    assert parsed["dmc_campaign"]["no_tmove_control"]["middle_timestep_control"] == {
        "status": "assessed",
        "matching_tmove_timesteps": [0.0025, 0.00125],
        "interior_tmove_timesteps": [0.0025],
        "control_count_matches_reference": False,
        "all_controls_match_interior_tmove_timestep": False,
    }


def test_qmcpack_dmc_campaign_records_reference_production_order():
    parsed = parse_qmcpack_input(
        "notes/fblock/examples/qmcpack/Ce-ion-dmc-production.xml"
    )

    assert parsed["dmc_campaign"]["production_protocol"] == {
        "status": "assessed",
        "linear_qmc_block_indices": [0],
        "vmc_qmc_block_indices": [1],
        "dmc_qmc_block_indices": [2, 3, 4, 5, 6],
        "linear_before_vmc": True,
        "vmc_before_dmc": True,
        "linear_optimization_loop": {
            "status": "assessed",
            "loop_maxes": [6],
            "all_loop_maxes_in_reference_range": True,
        },
        "linear_optimization_settings": {
            "status": "assessed",
            "settings": [{
                "qmc_block_index": 0,
                "min_method": "OneShiftOnly",
                "energy_cost": 0.1,
                "unreweighted_variance_cost": 0.9,
            }],
            "all_settings_match_reference": True,
        },
        "matches_reference_order": True,
    }
    assert parsed["dmc_campaign"]["tmove_ladder"][
        "matches_fblock_reference_timestep_ladder"
    ] is True
    assert parsed["dmc_campaign"]["no_tmove_control"]["middle_timestep_control"] == {
        "status": "assessed",
        "matching_tmove_timesteps": [0.0025],
        "interior_tmove_timesteps": [0.0025, 0.00125],
        "control_count_matches_reference": True,
        "all_controls_match_interior_tmove_timestep": True,
    }
    assert parsed["dmc_campaign"]["no_tmove_control"]["matching_tmove_settings"] == {
        "status": "assessed",
        "comparisons": [{
            "no_tmove_qmc_block_index": 6,
            "tmove_qmc_block_indices": [3],
            "block_count_match": True,
            "steps_match": True,
            "warmup_steps_match": True,
            "target_walkers_match": True,
            "move_match": True,
            "checkpoint_match": True,
            "all_declared_settings_match": True,
        }],
    }


def test_qmcpack_production_protocol_flags_linear_loop_outside_reference_range(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "campaign.xml", """\
<simulation>
  <loop max="5"><qmc method="linear"/></loop>
  <qmc method="vmc"/>
  <qmc method="dmc"/>
</simulation>
"""))

    assert parsed["dmc_campaign"]["production_protocol"]["linear_optimization_loop"] == {
        "status": "assessed",
        "loop_maxes": [5],
        "all_loop_maxes_in_reference_range": False,
    }


def test_qmcpack_tmove_control_reports_mismatched_declared_settings(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "campaign.xml", """\
<simulation>
  <qmc method="dmc" move="pbyp" checkpoint="10"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.0025</parameter><parameter name="blocks">800</parameter><parameter name="steps">40</parameter><parameter name="warmupSteps">800</parameter><parameter name="total_walkers">2240</parameter></qmc>
  <qmc method="dmc" move="pbyp" checkpoint="10"><parameter name="nonlocalmoves">no</parameter><parameter name="timestep">0.0025</parameter><parameter name="blocks">800</parameter><parameter name="steps">40</parameter><parameter name="warmupSteps">900</parameter><parameter name="total_walkers">2240</parameter></qmc>
</simulation>
"""))

    assert parsed["dmc_campaign"]["no_tmove_control"]["matching_tmove_settings"] == {
        "status": "assessed",
        "comparisons": [{
            "no_tmove_qmc_block_index": 1,
            "tmove_qmc_block_indices": [0],
            "block_count_match": True,
            "steps_match": True,
            "warmup_steps_match": False,
            "target_walkers_match": True,
            "move_match": True,
            "checkpoint_match": True,
            "all_declared_settings_match": False,
        }],
    }


def test_qmcpack_tmove_ladder_accepts_optional_fine_reference_timestep(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "campaign.xml", """\
<simulation>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.005</parameter><parameter name="blocks">100</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.0025</parameter><parameter name="blocks">200</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.00125</parameter><parameter name="blocks">300</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.000625</parameter><parameter name="blocks">400</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.0003125</parameter><parameter name="blocks">500</parameter></qmc>
</simulation>
"""))

    assert parsed["dmc_campaign"]["tmove_ladder"][
        "matches_fblock_reference_timestep_ladder"
    ] is True


def test_qmcpack_production_protocol_flags_linear_settings_outside_reference(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "campaign.xml", """\
<simulation>
  <loop max="6">
    <qmc method="linear">
      <parameter name="MinMethod">SomeOtherMethod</parameter>
      <cost name="energy">0.2</cost>
      <cost name="unreweightedvariance">0.8</cost>
    </qmc>
  </loop>
  <qmc method="vmc"/>
  <qmc method="dmc"/>
</simulation>
"""))

    assert parsed["dmc_campaign"]["production_protocol"][
        "linear_optimization_settings"
    ] == {
        "status": "assessed",
        "settings": [{
            "qmc_block_index": 0,
            "min_method": "SomeOtherMethod",
            "energy_cost": 0.2,
            "unreweighted_variance_cost": 0.8,
        }],
        "all_settings_match_reference": False,
    }


def test_parse_qmcpack_output_preserves_completion_time_and_unique_warnings():
    parsed = parse_qmcpack_output_text("""\
                        QMCPACK 4.1.0
  QMCPACK WARNING old input form
  QMCPACK WARNING old input form
  Total Execution time = 1.8549e-02 secs
QMCPACK execution completed successfully
""")

    assert parsed == {
        "program_version": "4.1.0",
        "line_count": 5,
        "completion": {"success_marker": True, "line": 5},
        "total_execution_time_seconds": 0.018549,
        "last_total_execution_time_line": 4,
        "project": None,
        "sections": [],
        "warnings": [{
            "message": "old input form",
            "line": 2,
            "occurrences": 2,
        }],
        "minwalkers_threshold_warnings": [],
        "optimization_messages": [],
        "linear_optimization_steps": {},
    }


def test_qmcpack_output_preserves_distinct_project_labels_without_selecting_one():
    parsed = parse_qmcpack_output_text("""\
QMCPACK 4.1.0
Project = initial_run
Project = restarted_run
""")

    assert parsed["project"] is None
    assert parsed["project_labels"] == [
        {"id": "initial_run", "line": 2},
        {"id": "restarted_run", "line": 3},
    ]


def test_qmcpack_output_uses_the_trailing_banner_run_for_completion(tmp_path):
    output_path = _write(tmp_path, "restarted.out", """\
QMCPACK 4.0.0
Total Execution time = 2.0 secs
QMCPACK execution completed successfully
QMCPACK 4.1.0
Start VMC
""")

    parsed = parse_qmcpack_output_text(output_path.read_text())

    assert parsed["program_version"] == "4.1.0"
    assert parsed["completion"] == {"success_marker": False, "line": None}
    assert parsed["total_execution_time_seconds"] is None
    assert parsed["last_total_execution_time_line"] is None
    assert parsed["last_run"] == {"start_line": 4}

    inspected = inspect_run(QMCPACK, output_path, resolved_by="override")

    assert inspected["assessment"]["verdict"]["label"] == "incomplete"
    assert inspected["evidence"]["derived"]["qmcpack:last_run_start_line"] == 4


def test_qmcpack_output_scopes_run_evidence_to_the_trailing_banner(tmp_path):
    output_path = _write(tmp_path, "concatenated.out", """\
QMCPACK 4.0.0
QMCPACK WARNING old input form
Cost Function is Invalid. If this frequently, reduce the step size.
Start DMC
DMC Execution time = 1.0 secs
QMCPACK execution completed successfully
QMCPACK 4.1.0
Project = latest_run
ParticleSet 'e' contains 6 particles : u(4) d(2)
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    parsed = parse_qmcpack_output_text(output_path.read_text())

    assert parsed["warnings"] == []
    assert parsed["optimization_messages"] == []
    assert parsed["project"] == {"id": "latest_run", "line": 8}
    assert parsed["runtime_particle_sets"] == [{
        "name": "e",
        "particle_count": 6,
        "groups": [{"name": "u", "count": 4}, {"name": "d", "count": 2}],
        "group_particle_count": 6,
        "group_particle_count_matches": True,
        "line": 9,
    }]
    assert parsed["sections"] == [{
        "name": "VMC",
        "start_line": 10,
        "end_line": 11,
        "execution_time_seconds": 1.0,
    }]

    inspected = inspect_run(QMCPACK, output_path, resolved_by="override")

    assert inspected["assessment"]["verdict"]["label"] == "completed"
    assert inspected["evidence"]["derived"]["qmcpack:warning_count"] == 0
    assert "qmcpack:cost_function_invalid_count" not in inspected["evidence"]["derived"]
    assert [task["method"] for task in inspected["evidence"]["tasks"]] == ["VMC"]


def test_qmcpack_corpus_modern_dmc_excerpt():
    parsed = parse_qmcpack_output_text(
        (_QMCPACK_FIXTURES / "oxygen_atom_dmc_excerpt.out").read_text()
    )

    assert parsed["project"] == {"id": "O.q0.dmc", "line": 2}
    assert parsed["completion"] == {"success_marker": True, "line": 19}
    assert parsed["total_execution_time_seconds"] == 0.018549
    assert [section["name"] for section in parsed["sections"]] == ["VMC", "DMC"]
    assert parsed["runtime_particle_sets"][0]["groups"] == [
        {"name": "u", "count": 4},
        {"name": "d", "count": 2},
    ]
    assert parsed["input_parameter_corrections"] == [{
        "parameter": "blocks",
        "requested_value": 0.0,
        "corrected_value": 1.0,
        "occurrences": 1,
        "first_line": 11,
        "last_line": 11,
        "section": "VMC",
    }, {
        "parameter": "steps",
        "requested_value": 0.0,
        "corrected_value": 1.0,
        "occurrences": 1,
        "first_line": 12,
        "last_line": 12,
        "section": "VMC",
    }, {
        "parameter": "blocks",
        "requested_value": 0.0,
        "corrected_value": 1.0,
        "occurrences": 1,
        "first_line": 15,
        "last_line": 15,
        "section": "DMC",
    }, {
        "parameter": "steps",
        "requested_value": 0.0,
        "corrected_value": 1.0,
        "occurrences": 1,
        "first_line": 16,
        "last_line": 16,
        "section": "DMC",
    }]
    assert parsed["warnings"] == [{
        "message": (
            "!!!!!!! Deprecated input style: creating SPO set inside determinantset. "
            "Support for this usage will soon be removed. SPO sets should be "
            "built outside using sposet_collection."
        ),
        "line": 3,
        "occurrences": 1,
    }, {
        "message": (
            "!!!!!!! Deprecated input style: creating SPO set inside "
            "slaterdeterminant and nested determinant tags. Support for this "
            "usage will soon be removed. SPO sets should be built outside using "
            "sposet_collection."
        ),
        "line": 4,
        "occurrences": 2,
    }, {
        "message": (
            "twist attribute does't exist but twistnum attribute was found. "
            "This is potentially ambiguous. Specifying twist attribute is preferred."
        ),
        "line": 6,
        "occurrences": 1,
    }, {
        "message": (
            "Nrule was not determined from qmcpack input or pseudopotential file. "
            "Setting sensible default."
        ),
        "line": 7,
        "occurrences": 1,
    }, {
        "message": (
            "Input parameter \"blocks\" must be positive! Set to 1. "
            "User input value 0"
        ),
        "line": 11,
        "occurrences": 2,
    }, {
        "message": (
            "Input parameter \"steps\" must be positive! Set to 1. "
            "User input value 0"
        ),
        "line": 12,
        "occurrences": 2,
    }]

    inspected = inspect_run(
        QMCPACK,
        _QMCPACK_FIXTURES / "oxygen_atom_dmc_excerpt.out",
        resolved_by="explicit",
    )

    assert inspected["assessment"] == {
        "source": "parser_diagnosis",
        "verdict": {
            "label": "input_parameter_auto_corrected",
            "confidence": 0.98,
            "reasons": [
                "QMCPACK replaced invalid input values: blocks 0 -> 1 in VMC "
                "(1 occurrence(s)); steps 0 -> 1 in VMC (1 occurrence(s)); "
                "blocks 0 -> 1 in DMC (1 occurrence(s)); steps 0 -> 1 in DMC "
                "(1 occurrence(s))."
            ],
        },
    }
    assert inspected["evidence"]["derived"][
        "qmcpack:input_parameter_correction_count"
    ] == 4


def test_qmcpack_corpus_legacy_two_run_excerpt_uses_the_trailing_run():
    parsed = parse_qmcpack_output_text(
        (_QMCPACK_FIXTURES / "h_two_runs_excerpt.out").read_text()
    )

    assert parsed["last_run"] == {"start_line": 8}
    assert parsed["project"] == {"id": "H", "line": 9}
    assert parsed["total_execution_time_seconds"] == 13.31537555
    assert parsed["sections"] == [{
        "name": "VMC",
        "start_line": 12,
        "end_line": 13,
        "execution_time_seconds": 13.250650895,
    }]
    assert parsed["runtime_particle_sets"][0]["group_offsets"] == [0, 1]


def test_qmcpack_corpus_legacy_optimizer_excerpt():
    parsed = parse_qmcpack_output_text(
        (_QMCPACK_FIXTURES / "soci_optimizer_excerpt.out").read_text()
    )

    assert len(parsed["sections"]) == 2
    assert parsed["optimization_messages"] == [{
        "code": "effective_walkers_too_small",
        "message": (
            "ERROR CostFunction-> Number of Effective Walkers is too small "
            "1.7493767924e+00 NumWalkersEff/NumSamples 3.4167515477e-05"
        ),
        "occurrences": 3,
        "first_line": 6,
        "last_line": 11,
        "minimum_reported_effective_walkers": 1.0000038027,
        "sections": [
            {"name": "QMCFixedSampleLinearOptimize", "start_line": 5},
            {"name": "QMCFixedSampleLinearOptimize", "start_line": 10},
        ],
    }, {
        "code": "linear_optimization_failed_step",
        "message": "Failed Step. Largest LM parameter change:2.3898804172e+02",
        "occurrences": 2,
        "first_line": 8,
        "last_line": 12,
        "largest_reported_parameter_change": 2672.9859945,
    }]
    assert parsed["linear_optimization_steps"]["good"]["occurrences"] == 1
    assert parsed["runtime_particle_sets"][0]["group_offsets"] == [0, 4, 8]


def test_parse_qmcpack_output_does_not_treat_warning_as_version_banner():
    with pytest.raises(ValueError, match="version banner"):
        parse_qmcpack_output_text("QMCPACK WARNING old input form\n")


def test_parse_qmcpack_output_marks_total_time_without_claiming_success():
    parsed = parse_qmcpack_output_text("""\
QMCPACK 1.0.0
Total Execution time = 1.3405175831e+01 secs
""")

    assert parsed["completion"] == {"success_marker": False, "line": None}
    assert parsed["total_execution_time_seconds"] == 13.405175831
    assert parsed["last_total_execution_time_line"] == 2


def test_parse_qmcpack_output_ignores_nonfinite_numeric_records():
    parsed = parse_qmcpack_output_text("""\
QMCPACK 4.1.0
Effective weight of all the samples measured by correlated sampling is nan
QMCPACK WARNING Smaller than the user specified threshold "minwalkers" = inf
Total Execution time = inf secs
QMCPACK execution completed successfully
""")

    assert parsed["total_execution_time_seconds"] is None
    assert parsed["minwalkers_threshold_warnings"] == []


def test_parse_qmcpack_output_summarizes_minwalkers_threshold_warnings(tmp_path):
    parsed = parse_qmcpack_output_text("""\
QMCPACK 4.1.0
Effective weight of all the samples measured by correlated sampling is 1.0000e-01
  QMCPACK WARNING     Smaller than the user specified threshold "minwalkers" = 5.0000e-01
Effective weight of all the samples measured by correlated sampling is 2.5000e-02
  QMCPACK WARNING     Smaller than the user specified threshold "minwalkers" = 5.0000e-01
QMCPACK execution completed successfully
""")

    assert parsed["minwalkers_threshold_warnings"] == [{
        "threshold": 0.5,
        "occurrences": 2,
        "first_line": 3,
        "last_line": 5,
        "minimum_immediately_preceding_effective_weight": 0.025,
        "immediately_preceding_effective_weight_count": 2,
    }]

    path = _write(tmp_path, "O.opt.out", """\
QMCPACK 4.1.0
Effective weight of all the samples measured by correlated sampling is 5.0000e-02
QMCPACK WARNING     Smaller than the user specified threshold "minwalkers" = 5.0000e-01
QMCPACK execution completed successfully
""")
    inspected = inspect_run(QMCPACK, path, resolved_by="override")

    assert inspected["evidence"]["derived"] == {
        "n_tasks": 1,
        "qmcpack:success_marker": True,
        "qmcpack:success_marker_line": 4,
        "qmcpack:completion_evidence": "explicit_success_marker",
        "qmcpack:total_execution_time_seconds": None,
        "qmcpack:last_total_execution_time_line": None,
        "qmcpack:warning_count": 1,
        "qmcpack:minwalkers_warning_count": 1,
        "qmcpack:minwalkers_thresholds": [0.5],
        "qmcpack:minwalkers_minimum_preceding_effective_weight": 0.05,
    }


def test_qmcpack_output_records_invalid_cost_and_parameter_reversion(tmp_path):
    parsed = parse_qmcpack_output_text("""\
QMCPACK 4.1.0
Cost Function is Invalid. If this frequently, reduce the step size.
Cost Function is Invalid. If this frequently, reduce the step size.
Reverting to old Parameters
QMCPACK execution completed successfully
""")

    assert parsed["optimization_messages"] == [
        {
            "code": "cost_function_invalid",
            "message": "Cost Function is Invalid. If this frequently, reduce the step size.",
            "occurrences": 2,
            "first_line": 2,
            "last_line": 3,
        },
        {
            "code": "reverting_to_old_parameters",
            "message": "Reverting to old Parameters",
            "occurrences": 1,
            "first_line": 4,
            "last_line": 4,
        },
    ]

    path = _write(tmp_path, "O.opt.out", """\
QMCPACK 4.1.0
Cost Function is Invalid. If this frequently, reduce the step size.
Reverting to old Parameters
QMCPACK execution completed successfully
""")
    inspected = inspect_run(QMCPACK, path, resolved_by="override")

    assert inspected["evidence"]["derived"] == {
        "n_tasks": 1,
        "qmcpack:success_marker": True,
        "qmcpack:success_marker_line": 4,
        "qmcpack:completion_evidence": "explicit_success_marker",
        "qmcpack:total_execution_time_seconds": None,
        "qmcpack:last_total_execution_time_line": None,
        "qmcpack:warning_count": 0,
        "qmcpack:optimization_messages": [
            {
                "code": "cost_function_invalid",
                "message": "Cost Function is Invalid. If this frequently, reduce the step size.",
                "occurrences": 1,
                "first_line": 2,
                "last_line": 2,
            },
            {
                "code": "reverting_to_old_parameters",
                "message": "Reverting to old Parameters",
                "occurrences": 1,
                "first_line": 3,
                "last_line": 3,
            },
        ],
        "qmcpack:cost_function_invalid_count": 1,
        "qmcpack:reverted_to_old_parameters": True,
    }
    assert [diagnostic["message"] for diagnostic in inspected["evidence"]["diagnostics"]] == [
        "Cost Function is Invalid. If this frequently, reduce the step size.",
        "Reverting to old Parameters",
    ]
    assert inspected["assessment"] == {
        "source": "parser_diagnosis",
        "verdict": {
            "label": "optimization_cost_function_invalid",
            "confidence": 0.95,
            "reasons": [
                "QMCPACK reported an invalid optimization cost function 1 time(s).",
                "QMCPACK reverted to older optimization parameters 1 time(s).",
            ],
        },
    }


def test_qmcpack_optimizer_messages_retain_affected_sections():
    parsed = parse_qmcpack_output_text("""\
QMCPACK 4.1.0
Start QMCFixedSampleLinearOptimize
Cost Function is Invalid. If this frequently, reduce the step size.
Start QMCFixedSampleLinearOptimize
Cost Function is Invalid. If this frequently, reduce the step size.
Reverting to old Parameters
QMCPACK execution completed successfully
""")

    assert parsed["optimization_messages"] == [
        {
            "code": "cost_function_invalid",
            "message": "Cost Function is Invalid. If this frequently, reduce the step size.",
            "occurrences": 2,
            "first_line": 3,
            "last_line": 5,
            "sections": [
                {"name": "QMCFixedSampleLinearOptimize", "start_line": 2},
                {"name": "QMCFixedSampleLinearOptimize", "start_line": 4},
            ],
        },
        {
            "code": "reverting_to_old_parameters",
            "message": "Reverting to old Parameters",
            "occurrences": 1,
            "first_line": 6,
            "last_line": 6,
            "sections": [
                {"name": "QMCFixedSampleLinearOptimize", "start_line": 4},
            ],
        },
    ]


def test_qmcpack_output_records_legacy_optimizer_recovery_evidence(tmp_path):
    parsed = parse_qmcpack_output_text("""\
QMCPACK 1.0.0
ERROR CostFunction-> Number of Effective Walkers is too small 4.0
ERROR CostFunction-> Number of Effective Walkers is too small 1.0
ERROR Revertting to old Parameters
""")

    assert parsed["optimization_messages"] == [
        {
            "code": "effective_walkers_too_small",
            "message": (
                "ERROR CostFunction-> Number of Effective Walkers is too "
                "small 4.0"
            ),
            "occurrences": 2,
            "first_line": 2,
            "last_line": 3,
            "minimum_reported_effective_walkers": 1.0,
        },
        {
            "code": "reverting_to_old_parameters",
            "message": "ERROR Revertting to old Parameters",
            "occurrences": 1,
            "first_line": 4,
            "last_line": 4,
        },
    ]

    path = _write(tmp_path, "legacy-opt.out", """\
QMCPACK 1.0.0
ERROR CostFunction-> Number of Effective Walkers is too small 4.0
ERROR CostFunction-> Number of Effective Walkers is too small 1.0
ERROR Revertting to old Parameters
""")
    inspected = inspect_run(QMCPACK, path, resolved_by="override")

    assert inspected["evidence"]["derived"] == {
        "n_tasks": 1,
        "qmcpack:success_marker": False,
        "qmcpack:success_marker_line": None,
        "qmcpack:completion_evidence": "none",
        "qmcpack:total_execution_time_seconds": None,
        "qmcpack:last_total_execution_time_line": None,
        "qmcpack:warning_count": 0,
        "qmcpack:optimization_messages": [
            {
                "code": "effective_walkers_too_small",
                "message": (
                    "ERROR CostFunction-> Number of Effective Walkers is too "
                    "small 4.0"
                ),
                "occurrences": 2,
                "first_line": 2,
                "last_line": 3,
                "minimum_reported_effective_walkers": 1.0,
            },
            {
                "code": "reverting_to_old_parameters",
                "message": "ERROR Revertting to old Parameters",
                "occurrences": 1,
                "first_line": 4,
                "last_line": 4,
            },
        ],
        "qmcpack:effective_walkers_too_small_count": 2,
        "qmcpack:minimum_reported_effective_walkers": 1.0,
        "qmcpack:reverted_to_old_parameters": True,
    }
    assert [
        diagnostic["message"] for diagnostic in inspected["evidence"]["diagnostics"]
    ] == [
        "ERROR CostFunction-> Number of Effective Walkers is too small 4.0",
        "ERROR Revertting to old Parameters",
    ]
    assert inspected["assessment"] == {
        "source": "parser_diagnosis",
        "verdict": {
            "label": "optimization_effective_walkers_too_small",
            "confidence": 0.98,
            "reasons": [
                "QMCPACK reported too few effective walkers 2 time(s); the "
                "lowest reported value was 1.",
                "QMCPACK reverted to older optimization parameters 1 time(s).",
            ],
        },
    }


def test_qmcpack_output_records_failed_linear_optimization_steps(tmp_path):
    parsed = parse_qmcpack_output_text("""\
QMCPACK 4.1.0
  Failed Step. Largest LM parameter change:2.5e+02
  Failed Step. Largest LM parameter change:5.0e+01
  Good Step. Largest LM parameter change:1.0e+01
QMCPACK execution completed successfully
""")

    assert parsed["optimization_messages"] == [{
        "code": "linear_optimization_failed_step",
        "message": "Failed Step. Largest LM parameter change:2.5e+02",
        "occurrences": 2,
        "first_line": 2,
        "last_line": 3,
        "largest_reported_parameter_change": 250.0,
    }]
    assert parsed["linear_optimization_steps"] == {
        "failed": {
            "message": "Failed Step. Largest LM parameter change:2.5e+02",
            "occurrences": 2,
            "first_line": 2,
            "last_line": 3,
            "largest_reported_parameter_change": 250.0,
        },
        "good": {
            "message": "Good Step. Largest LM parameter change:1.0e+01",
            "occurrences": 1,
            "first_line": 4,
            "last_line": 4,
            "largest_reported_parameter_change": 10.0,
        },
    }

    path = _write(tmp_path, "linear-opt.out", """\
QMCPACK 4.1.0
  Failed Step. Largest LM parameter change:2.5e+02
  Good Step. Largest LM parameter change:1.0e+01
QMCPACK execution completed successfully
""")
    inspected = inspect_run(QMCPACK, path, resolved_by="override")

    assert inspected["evidence"]["derived"] == {
        "n_tasks": 1,
        "qmcpack:success_marker": True,
        "qmcpack:success_marker_line": 4,
        "qmcpack:completion_evidence": "explicit_success_marker",
        "qmcpack:total_execution_time_seconds": None,
        "qmcpack:last_total_execution_time_line": None,
        "qmcpack:warning_count": 0,
        "qmcpack:optimization_messages": [
            {
                "code": "linear_optimization_failed_step",
                "message": "Failed Step. Largest LM parameter change:2.5e+02",
                "occurrences": 1,
                "first_line": 2,
                "last_line": 2,
                "largest_reported_parameter_change": 250.0,
            },
        ],
        "qmcpack:linear_optimization_failed_step_count": 1,
        "qmcpack:largest_failed_linear_optimization_parameter_change": 250.0,
        "qmcpack:linear_optimization_good_step_count": 1,
        "qmcpack:largest_good_linear_optimization_parameter_change": 10.0,
    }
    assert inspected["assessment"]["verdict"]["label"] == "completed"
    assert inspected["evidence"]["diagnostics"] == [{
        "kind": "warning",
        "message": "Failed Step. Largest LM parameter change:2.5e+02",
        "line": 2,
        "file": str(path.resolve()),
    }]


def test_qmcpack_output_records_runtime_particle_sets(tmp_path):
    parsed = parse_qmcpack_output_text("""\
QMCPACK 4.1.0
  ParticleSet 'e' contains 6 particles :  u(4) d(2)
  ParticleSet 'ion0' contains 1 particles :  O(1)
QMCPACK execution completed successfully
""")

    assert parsed["runtime_particle_sets"] == [
        {
            "name": "e",
            "particle_count": 6,
            "groups": [{"name": "u", "count": 4}, {"name": "d", "count": 2}],
            "group_particle_count": 6,
            "group_particle_count_matches": True,
            "line": 2,
        },
        {
            "name": "ion0",
            "particle_count": 1,
            "groups": [{"name": "O", "count": 1}],
            "group_particle_count": 1,
            "group_particle_count_matches": True,
            "line": 3,
        },
    ]

    path = _write(tmp_path, "O.opt.out", """\
QMCPACK 4.1.0
  ParticleSet 'e' contains 6 particles :  u(4) d(2)
QMCPACK execution completed successfully
""")
    inspected = inspect_run(QMCPACK, path, resolved_by="override")

    assert inspected["evidence"]["derived"]["qmcpack:runtime_particle_sets"] == [
        {
            "name": "e",
            "particle_count": 6,
            "groups": [{"name": "u", "count": 4}, {"name": "d", "count": 2}],
            "group_particle_count": 6,
            "group_particle_count_matches": True,
            "line": 2,
        },
    ]


def test_qmcpack_output_records_legacy_particle_set_offsets():
    parsed = parse_qmcpack_output_text("""\
QMCPACK 1.0.0
ParticleSet e : 0 4 8
ParticleSet ion0 : 0 1 3
""")

    assert parsed["runtime_particle_sets"] == [
        {
            "name": "e",
            "particle_count": 8,
            "groups": [{"name": None, "count": 4}, {"name": None, "count": 4}],
            "group_offsets": [0, 4, 8],
            "group_particle_count": 8,
            "group_particle_count_matches": True,
            "line": 2,
        },
        {
            "name": "ion0",
            "particle_count": 3,
            "groups": [{"name": None, "count": 1}, {"name": None, "count": 2}],
            "group_offsets": [0, 1, 3],
            "group_particle_count": 3,
            "group_particle_count_matches": True,
            "line": 3,
        },
    ]


def test_qmcpack_output_drives_generic_run_inspection(tmp_path):
    path = _write(tmp_path, "O.dmc.out", """\
QMCPACK 4.1.0
QMCPACK WARNING old input form
  Total Execution time = 1.8549e-02 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(QMCPACK, path, resolved_by="override")

    assert inspected["program"] == {
        "name": "qmcpack",
        "version": "4.1.0",
        "resolved_by": "override",
    }
    assert inspected["assessment"]["verdict"]["label"] == "completed"
    assert inspected["evidence"]["tasks"] == [{
        "index": 0,
        "kind": "unknown",
        "name": "QMCPACK run",
        "method": "QMCPACK",
        "basis": None,
        "energy_hartree": None,
        "line_range": (1, 4),
        "outcome": "success",
        "has_usable_data": True,
        "selection_priority": 1,
    }]
    assert inspected["evidence"]["derived"] == {
        "n_tasks": 1,
        "qmcpack:success_marker": True,
        "qmcpack:success_marker_line": 4,
        "qmcpack:completion_evidence": "explicit_success_marker",
        "qmcpack:total_execution_time_seconds": 0.018549,
        "qmcpack:last_total_execution_time_line": 3,
        "qmcpack:warning_count": 1,
    }


def test_qmcpack_input_output_consistency_matches_optimizer_method(tmp_path):
    input_path = _write(tmp_path, "opt.in.xml", """\
<simulation>
  <qmc method="linear"/>
</simulation>
""")
    output_path = _write(tmp_path, "opt.out", """\
QMCPACK 4.1.0
Start QMCFixedSampleLinearOptimize
QMCFixedSampleLinearOptimize Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    assert inspected["evidence"]["input_output_consistency"] == {
        "status": "checked",
        "input_path": str(input_path.resolve()),
        "summary": {"match": 1, "mismatch": 0, "not_checked": 0},
        "checks": [{
            "field": "qmc_method:QMCFixedSampleLinearOptimize",
            "status": "match",
            "input": {
                "declared": True,
                "supported_sections": ["QMCFixedSampleLinearOptimize"],
            },
            "output": {
                "observed": True,
                "supported_sections": ["QMCFixedSampleLinearOptimize"],
                "basis": (
                    "Repeated log sections are internal optimizer iterations; "
                    "this check compares supported method presence only."
                ),
            },
        }],
    }


def test_qmcpack_input_output_consistency_detects_method_mismatch(tmp_path):
    input_path = _write(tmp_path, "dmc.in.xml", """\
<simulation>
  <qmc method="dmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {"match": 0, "mismatch": 2, "not_checked": 0}
    assert [check["field"] for check in consistency["checks"]] == [
        "qmc_method:DMC",
        "qmc_method:VMC",
    ]


def test_qmcpack_input_output_consistency_detects_project_label_mismatch(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <project id="Ce_dmc" series="0"/>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
Project = Pr_dmc
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {
        "match": 1,
        "mismatch": 1,
        "not_checked": 0,
    }
    assert consistency["checks"][-1] == {
        "field": "project_id",
        "status": "mismatch",
        "input": {"project_id": "Ce_dmc"},
        "output": {
            "project_id": "Pr_dmc",
            "line": 2,
            "basis": (
                "This compares QMCPACK's printed project label only; it does not "
                "establish input controls or output provenance."
            ),
        },
    }


def test_qmcpack_input_output_consistency_checks_scalar_filename_projects(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <project id="Ce_dmc" series="0"/>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
Project = Ce_dmc
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")
    matching_scalar = _write(
        tmp_path,
        "Ce_dmc.s000.scalar.dat",
        "# index LocalEnergy\n0 -10.0\n",
    )
    mismatched_scalar = _write(
        tmp_path,
        "Pr_dmc.s001.scalar.dat",
        "# index LocalEnergy\n0 -10.0\n",
    )

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path, matching_scalar, mismatched_scalar],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {
        "match": 3,
        "mismatch": 1,
        "not_checked": 0,
    }
    assert consistency["checks"][-2:] == [
        {
            "field": "scalar_filename_project:Ce_dmc.s000.scalar.dat",
            "status": "match",
            "artifact": {
                "path": str(matching_scalar.resolve()),
                "project_id": "Ce_dmc",
                "series_index": 0,
                "basis": (
                    "Filename identity does not establish the source QMC input "
                    "block or its controls."
                ),
            },
            "output": {
                "project_id": "Ce_dmc",
                "line": 2,
                "basis": (
                    "This compares the scalar filename project label with the "
                    "primary log label; it does not establish source-run or "
                    "QMC-block lineage."
                ),
            },
        },
        {
            "field": "scalar_filename_project:Pr_dmc.s001.scalar.dat",
            "status": "mismatch",
            "artifact": {
                "path": str(mismatched_scalar.resolve()),
                "project_id": "Pr_dmc",
                "series_index": 1,
                "basis": (
                    "Filename identity does not establish the source QMC input "
                    "block or its controls."
                ),
            },
            "output": {
                "project_id": "Ce_dmc",
                "line": 2,
                "basis": (
                    "This compares the scalar filename project label with the "
                    "primary log label; it does not establish source-run or "
                    "QMC-block lineage."
                ),
            },
        },
    ]


def test_qmcpack_input_output_consistency_explains_scalar_project_abstention(
    tmp_path,
):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")
    scalar = _write(
        tmp_path,
        "Ce_dmc.s000.scalar.dat",
        "# index LocalEnergy\n0 -10.0\n",
    )

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path, scalar],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 1,
        "mismatch": 0,
        "not_checked": 1,
    }
    assert consistency["checks"][-1] == {
        "field": "scalar_filename_project:Ce_dmc.s000.scalar.dat",
        "status": "not_checked",
        "artifact": {
            "path": str(scalar.resolve()),
            "project_id": "Ce_dmc",
            "series_index": 0,
            "basis": (
                "Filename identity does not establish the source QMC input "
                "block or its controls."
            ),
        },
        "reason": "The primary log has no unambiguous project label to compare.",
    }


def test_qmcpack_input_output_explains_unrecognized_scalar_filename(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
Project = Ce_dmc
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")
    scalar = _write(
        tmp_path,
        "unrecognized.scalar.dat",
        "# index LocalEnergy\n0 -10.0\n",
    )

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path, scalar],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 1,
        "mismatch": 0,
        "not_checked": 1,
    }
    assert consistency["checks"][-1] == {
        "field": "scalar_filename_project:unrecognized.scalar.dat",
        "status": "not_checked",
        "artifact": {
            "path": str(scalar.resolve()),
            "filename": "unrecognized.scalar.dat",
            "basis": (
                "The scalar filename does not match the supported "
                "project.sNNN.scalar.dat convention."
            ),
        },
        "reason": "The scalar filename has no recognized project label to compare.",
    }


def test_qmcpack_input_output_consistency_abstains_for_distinct_project_labels(
    tmp_path,
):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <project id="restarted_run" series="0"/>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
Project = initial_run
Project = restarted_run
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    assert inspected["evidence"]["derived"]["qmcpack:project_labels"] == [
        {"id": "initial_run", "line": 2},
        {"id": "restarted_run", "line": 3},
    ]
    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 1,
        "mismatch": 0,
        "not_checked": 0,
    }


def test_qmcpack_input_output_consistency_checks_direct_particle_set_counts(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <particleset name="e">
    <group name="u" size="4"/>
    <group name="d" size="2"/>
  </particleset>
  <particleset name="ion0" size="1">
    <group name="O" size="1"/>
  </particleset>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
ParticleSet 'e' contains 5 particles : u(3) d(2)
ParticleSet 'ion0' contains 1 particles : O(1)
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {
        "match": 4,
        "mismatch": 2,
        "not_checked": 0,
    }
    particle_set_checks = [
        check
        for check in consistency["checks"]
        if check["field"].startswith("particle_set:")
    ]
    assert particle_set_checks == [
        {
            "field": "particle_set:e",
            "status": "mismatch",
            "input": {
                "particle_count": 6,
                "basis": (
                    "The count is from a direct XML particle set or its explicit "
                    "group sizes."
                ),
            },
            "output": {
                "particle_count": 5,
                "line": 2,
                "basis": "The count is from QMCPACK's runtime particle-pool summary.",
            },
        },
        {
            "field": "particle_set:ion0",
            "status": "match",
            "input": {
                "particle_count": 1,
                "basis": (
                    "The count is from a direct XML particle set or its explicit "
                    "group sizes."
                ),
            },
            "output": {
                "particle_count": 1,
                "line": 3,
                "basis": "The count is from QMCPACK's runtime particle-pool summary.",
            },
        },
    ]


def test_qmcpack_input_output_explains_missing_runtime_particle_set(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <particleset name="e">
    <group name="u" size="4"/>
    <group name="d" size="2"/>
  </particleset>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
ParticleSet 'ion0' contains 1 particles : O(1)
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 1,
        "mismatch": 0,
        "not_checked": 1,
    }
    assert consistency["checks"][-1] == {
        "field": "particle_set:e",
        "status": "not_checked",
        "input": {
            "particle_count": 6,
            "basis": (
                "The count is from a direct XML particle set or its explicit "
                "group sizes."
            ),
        },
        "reason": "The primary log has no unambiguous matching runtime particle set.",
    }


def test_qmcpack_input_output_consistency_checks_named_particle_groups(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <particleset name="e">
    <group name="u" size="4"/>
    <group name="d" size="2"/>
  </particleset>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
ParticleSet 'e' contains 6 particles : u(3) d(3)
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "mismatch"
    assert consistency["summary"] == {
        "match": 2,
        "mismatch": 2,
        "not_checked": 0,
    }
    assert consistency["checks"][-2:] == [
        {
            "field": "particle_group:e:u",
            "status": "mismatch",
            "input": {
                "particle_count": 4,
                "basis": "The count is from a direct XML particle group.",
            },
            "output": {
                "particle_count": 3,
                "line": 2,
                "basis": (
                    "The count is from QMCPACK's named runtime particle-pool "
                    "group."
                ),
            },
        },
        {
            "field": "particle_group:e:d",
            "status": "mismatch",
            "input": {
                "particle_count": 2,
                "basis": "The count is from a direct XML particle group.",
            },
            "output": {
                "particle_count": 3,
                "line": 2,
                "basis": (
                    "The count is from QMCPACK's named runtime particle-pool "
                    "group."
                ),
            },
        },
    ]


def test_qmcpack_input_output_consistency_abstains_for_inconsistent_runtime_groups(
    tmp_path,
):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <particleset name="e">
    <group name="u" size="4"/>
    <group name="d" size="2"/>
  </particleset>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
ParticleSet 'e' contains 6 particles : u(4) d(1)
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 2,
        "mismatch": 0,
        "not_checked": 2,
    }
    assert consistency["checks"][-2:] == [
        {
            "field": "particle_group:e:u",
            "status": "not_checked",
            "input": {
                "particle_count": 4,
                "basis": "The count is from a direct XML particle group.",
            },
            "output": {
                "particle_count": 6,
                "group_particle_count": 5,
                "line": 2,
                "basis": (
                    "The named runtime group counts do not sum to QMCPACK's "
                    "printed particle-set total."
                ),
            },
            "reason": "The runtime group counts are internally inconsistent.",
        },
        {
            "field": "particle_group:e:d",
            "status": "not_checked",
            "input": {
                "particle_count": 2,
                "basis": "The count is from a direct XML particle group.",
            },
            "output": {
                "particle_count": 6,
                "group_particle_count": 5,
                "line": 2,
                "basis": (
                    "The named runtime group counts do not sum to QMCPACK's "
                    "printed particle-set total."
                ),
            },
            "reason": "The runtime group counts are internally inconsistent.",
        },
    ]


def test_qmcpack_input_output_consistency_explains_missing_runtime_group(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <particleset name="e">
    <group name="u" size="4"/>
    <group name="d" size="2"/>
  </particleset>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
ParticleSet 'e' contains 6 particles : u(4) x(2)
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 3,
        "mismatch": 0,
        "not_checked": 1,
    }
    assert consistency["checks"][-1] == {
        "field": "particle_group:e:d",
        "status": "not_checked",
        "input": {
            "particle_count": 2,
            "basis": "The count is from a direct XML particle group.",
        },
        "output": {
            "line": 2,
            "basis": "The runtime particle-pool summary has no matching named group.",
        },
        "reason": "The primary log has no matching runtime particle group.",
    }


def test_qmcpack_input_output_consistency_explains_repeated_input_group(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <particleset name="e">
    <group name="u" size="4"/>
    <group name="u" size="0"/>
    <group name="d" size="2"/>
  </particleset>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
ParticleSet 'e' contains 6 particles : u(4) d(2)
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 3,
        "mismatch": 0,
        "not_checked": 1,
    }
    assert consistency["checks"][-2:] == [
        {
            "field": "particle_group:e:u",
            "status": "not_checked",
            "input": {
                "particle_set_name": "e",
                "group_name": "u",
                "basis": "The direct XML declares this group name more than once.",
            },
            "reason": "The input particle-group declaration is ambiguous.",
        },
        {
            "field": "particle_group:e:d",
            "status": "match",
            "input": {
                "particle_count": 2,
                "basis": "The count is from a direct XML particle group.",
            },
            "output": {
                "particle_count": 2,
                "line": 2,
                "basis": (
                    "The count is from QMCPACK's named runtime particle-pool "
                    "group."
                ),
            },
        },
    ]


def test_qmcpack_input_output_consistency_explains_repeated_runtime_group(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <particleset name="e">
    <group name="u" size="4"/>
    <group name="d" size="2"/>
  </particleset>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
ParticleSet 'e' contains 6 particles : u(4) u(0) d(2)
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 3,
        "mismatch": 0,
        "not_checked": 1,
    }
    assert consistency["checks"][-2:] == [
        {
            "field": "particle_group:e:u",
            "status": "not_checked",
            "input": {
                "particle_count": 4,
                "basis": "The count is from a direct XML particle group.",
            },
            "output": {
                "line": 2,
                "basis": (
                    "The runtime particle-pool summary declares this group "
                    "name more than once."
                ),
            },
            "reason": "The runtime particle-group declaration is ambiguous.",
        },
        {
            "field": "particle_group:e:d",
            "status": "match",
            "input": {
                "particle_count": 2,
                "basis": "The count is from a direct XML particle group.",
            },
            "output": {
                "particle_count": 2,
                "line": 2,
                "basis": (
                    "The count is from QMCPACK's named runtime particle-pool "
                    "group."
                ),
            },
        },
    ]


def test_qmcpack_input_output_consistency_abstains_for_repeated_runtime_set(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <particleset name="e">
    <group name="u" size="4"/>
    <group name="d" size="2"/>
  </particleset>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
ParticleSet 'e' contains 6 particles : u(4) d(2)
Start VMC
VMC Execution time = 1.0 secs
ParticleSet 'e' contains 5 particles : u(3) d(2)
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 1,
        "mismatch": 0,
        "not_checked": 1,
    }
    assert consistency["checks"][-1] == {
        "field": "particle_set:e",
        "status": "not_checked",
        "input": {
            "particle_count": 6,
            "basis": (
                "The count is from a direct XML particle set or its explicit "
                "group sizes."
            ),
        },
        "reason": "The primary log has no unambiguous matching runtime particle set.",
    }


def test_qmcpack_input_output_consistency_abstains_for_repeated_input_set(tmp_path):
    input_path = _write(tmp_path, "vmc.in.xml", """\
<simulation>
  <particleset name="e">
    <group name="u" size="4"/>
    <group name="d" size="2"/>
  </particleset>
  <particleset name="e">
    <group name="u" size="3"/>
    <group name="d" size="2"/>
  </particleset>
  <qmc method="vmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "vmc.out", """\
QMCPACK 4.1.0
ParticleSet 'e' contains 6 particles : u(4) d(2)
Start VMC
VMC Execution time = 1.0 secs
QMCPACK execution completed successfully
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    consistency = inspected["evidence"]["input_output_consistency"]
    assert consistency["status"] == "checked"
    assert consistency["summary"] == {
        "match": 1,
        "mismatch": 0,
        "not_checked": 1,
    }
    assert consistency["checks"][-1] == {
        "field": "particle_set:e",
        "status": "not_checked",
        "input": {
            "particle_set_name": "e",
            "basis": "The direct XML declares this particle-set name more than once.",
        },
        "reason": "The input particle-set declaration is ambiguous.",
    }


def test_qmcpack_input_output_consistency_abstains_for_untimed_section(tmp_path):
    input_path = _write(tmp_path, "dmc.in.xml", """\
<simulation>
  <qmc method="dmc"/>
</simulation>
""")
    output_path = _write(tmp_path, "dmc.out", """\
QMCPACK 4.1.0
Start DMC
""")

    inspected = inspect_run(
        QMCPACK,
        output_path,
        resolved_by="override",
        artifact_files=[input_path],
    )

    assert inspected["evidence"]["input_output_consistency"] == {
        "status": "not_checked",
        "input_path": str(input_path.resolve()),
        "reason": "The output contains no supported QMC section.",
    }


def test_qmcpack_output_indexes_top_level_vmc_and_dmc_sections():
    parsed = parse_qmcpack_output_text("""\
QMCPACK 4.1.0
  Start VMC
  VMC Execution time = 0.1 secs
  Start DMC
  DMC Execution time = 0.2 secs
QMCPACK execution completed successfully
""")

    assert parsed["sections"] == [
        {
            "name": "VMC",
            "start_line": 2,
            "end_line": 3,
            "execution_time_seconds": 0.1,
        },
        {
            "name": "DMC",
            "start_line": 4,
            "end_line": 5,
            "execution_time_seconds": 0.2,
        },
    ]


def test_qmcpack_output_indexes_legacy_vmc_and_dmc_sections():
    parsed = parse_qmcpack_output_text("""\
QMCPACK 1.0.0
  Start VMCSingleOMP
  QMC Execution time = 0.1 secs
  Start DMCOMP
  QMC Execution time = 0.2 secs
""")

    assert parsed["sections"] == [
        {
            "name": "VMC",
            "start_line": 2,
            "end_line": 3,
            "execution_time_seconds": 0.1,
        },
        {
            "name": "DMC",
            "start_line": 4,
            "end_line": 5,
            "execution_time_seconds": 0.2,
        },
    ]


def test_qmcpack_output_attributes_legacy_timing_to_optimizer_driver():
    parsed = parse_qmcpack_output_text("""\
QMCPACK 1.0.0
Start QMCFixedSampleLinearOptimize
Start VMCSingleOMP
QMC Execution time = 6.2 secs
""")

    assert parsed["sections"] == [{
        "name": "QMCFixedSampleLinearOptimize",
        "start_line": 2,
        "end_line": 4,
        "execution_time_seconds": 6.2,
    }]


def test_lint_qmcpack_input_reports_incomplete_references_and_blocks():
    issues = lint_qmcpack_input("""\
<simulation>
  <include href=" "/>
  <override_variational_parameters/>
  <hamiltonian><pairpot><pseudo elementType="Ce"/></pairpot></hamiltonian>
  <qmc/>
  <qmc method="dmc"><parameter name="blocks">0</parameter><parameter name="steps">not-a-number</parameter></qmc>
  <particleset><group size="0"/></particleset>
</simulation>
""")

    assert [issue["message"] for issue in issues] == [
        "<include> requires a non-empty href attribute.",
        "<override_variational_parameters> requires a non-empty href attribute.",
        "<pseudo> requires a non-empty href attribute.",
        "<qmc> requires a method attribute.",
        '<qmc> parameter "blocks" must be a positive integer when it is present.',
        '<qmc> parameter "steps" is non-numeric and cannot be validated.',
        "<group> size must be a positive integer when it is present.",
    ]


def test_lint_qmcpack_input_warns_for_deprecated_nonlocalpp():
    issues = lint_qmcpack_input("""\
<simulation>
  <qmc method="linear"><parameter name="nonlocalpp">yes</parameter></qmc>
</simulation>
""")

    assert [issue["message"] for issue in issues] == [
        '<qmc> parameter "nonlocalpp" is deprecated and does not affect '
        "QMCPACK execution; remove it.",
    ]


def test_lint_qmcpack_input_checks_declared_dmc_control_values():
    issues = lint_qmcpack_input("""\
<simulation>
  <qmc method="dmc">
    <parameter name="timestep">0</parameter>
    <parameter name="warmupSteps">0</parameter>
    <parameter name="targetWalkers">not-a-number</parameter>
    <parameter name="total_walkers">-1</parameter>
    <parameter name="nonlocalmoves">sometimes</parameter>
  </qmc>
</simulation>
""")

    assert [issue["message"] for issue in issues] == [
        '<qmc> parameter "timestep" must be a positive finite number when it is present.',
        '<qmc> parameter "targetWalkers" is non-numeric and cannot be validated.',
        '<qmc> parameter "total_walkers" must be a positive integer when it is present.',
        '<qmc> parameter "nonlocalmoves" is not a recognized boolean; use yes/no, true/false, or 1/0.',
    ]


def test_qmcpack_supports_modern_and_legacy_warmup_step_spellings(tmp_path):
    parsed = parse_qmcpack_input(_write(tmp_path, "campaign.xml", """\
<simulation>
  <qmc method="dmc"><parameter name="nonlocalmoves">yes</parameter><parameter name="timestep">0.01</parameter><parameter name="warmupsteps">0</parameter></qmc>
  <qmc method="dmc"><parameter name="nonlocalmoves">no</parameter><parameter name="timestep">0.01</parameter><parameter name="warmupSteps">0</parameter></qmc>
</simulation>
"""))

    assert parsed["dmc_campaign"]["no_tmove_control"]["matching_tmove_settings"] == {
        "status": "assessed",
        "comparisons": [{
            "no_tmove_qmc_block_index": 1,
            "tmove_qmc_block_indices": [0],
            "block_count_match": None,
            "steps_match": None,
            "warmup_steps_match": True,
            "target_walkers_match": None,
            "move_match": None,
            "checkpoint_match": None,
            "all_declared_settings_match": None,
        }],
    }
    assert lint_qmcpack_input("""\
<simulation><qmc method="dmc"><parameter name="warmupsteps">0</parameter></qmc></simulation>
""") == []


def test_lint_qmcpack_input_rejects_conflicting_warmup_step_aliases():
    issues = lint_qmcpack_input("""\
<simulation>
  <qmc method="dmc">
    <parameter name="warmupSteps">10</parameter>
    <parameter name="warmupsteps">20</parameter>
  </qmc>
</simulation>
""")

    assert [issue["message"] for issue in issues] == [
        '<qmc> parameters "warmupSteps" and "warmupsteps" disagree; '
        "declare one spelling or matching values.",
    ]


def test_lint_qmcpack_input_rejects_conflicting_walker_count_aliases():
    issues = lint_qmcpack_input("""\
<simulation>
  <qmc method="dmc">
    <parameter name="targetWalkers">960</parameter>
    <parameter name="total_walkers">2240</parameter>
  </qmc>
</simulation>
""")

    assert [issue["message"] for issue in issues] == [
        '<qmc> parameters "targetWalkers" and "total_walkers" disagree; '
        "declare one target or matching values.",
    ]


def test_lint_qmcpack_input_warns_for_ambiguous_twistnum():
    issues = lint_qmcpack_input("""\
<qmcsystem>
  <determinantset twistnum="0"/>
</qmcsystem>
""")

    assert [issue["message"] for issue in issues] == [
        "<determinantset> has twistnum but no twist attribute; specify twist "
        "to avoid an ambiguous selection.",
    ]


def test_lint_qmcpack_input_warns_for_legacy_inline_slater_determinant():
    issues = lint_qmcpack_input("""\
<qmcsystem>
  <determinantset><slaterdeterminant/></determinantset>
</qmcsystem>
""")

    assert [issue["message"] for issue in issues] == [
        "<determinantset> contains a legacy inline <slaterdeterminant>; "
        "move SPO setup to a top-level <sposet_collection>.",
    ]


def test_lint_qmcpack_input_warns_when_variational_sidecar_overrides_coefficients():
    issues = lint_qmcpack_input("""\
<qmcsystem>
  <wavefunction>
    <jastrow><coefficients>0.1 -0.2</coefficients></jastrow>
    <override_variational_parameters href="optimized.vp.h5"/>
  </wavefunction>
</qmcsystem>
""")

    assert [issue["message"] for issue in issues] == [
        "An override_variational_parameters sidecar is present with inline "
        "<coefficients> values. The sidecar is authoritative; the inline "
        "values may be stale display values.",
    ]


def test_lint_qmcpack_input_rejects_invalid_xml_and_unknown_roots():
    assert lint_qmcpack_input("<simulation>") == [{
        "level": "error",
        "message": (
            "QMCPACK XML is not well formed at line 1, column 12: "
            "no element found: line 1, column 12."
        ),
        "line": None,
        "suggested_fix": None,
    }]
    assert lint_qmcpack_input("<workflow/>") == [{
        "level": "error",
        "message": (
            "QMCPACK input must use a <simulation> or <qmcsystem> root; "
            "found <workflow>."
        ),
        "line": None,
        "suggested_fix": None,
    }]


def test_qmcpack_backend_drives_generic_guided_review(tmp_path):
    path = _write(tmp_path, "Ce.xml", SIMULATION)
    _write(tmp_path, "Ce_secp_ion3f1.ptcl.xml", PARTICLES)
    _write(tmp_path, "Ce_secp_ion3f1.wfj.xml", "<wavefunction/>")
    backends = tuple(load_backend(spec) for spec in BUILTIN_BACKENDS)

    backend, resolved_by = detect_input_backend(backends, path)
    reviewed = review_input(backend, path, resolved_by=resolved_by)

    assert backend is QMCPACK
    assert resolved_by == "content"
    assert reviewed["program"] == {
        "name": "qmcpack",
        "resolved_by": "content",
    }
    assert reviewed["assessment"]["verdict"]["label"] == "checks_passed"
    assert reviewed["evidence"]["artifact_classification"] == {
        "status": "matched",
        "candidates": [{
            "kind": "qmcpack.input",
            "roles": ["primary_input"],
            "evidence": "inferred",
            "matched_by": "extension",
            "matched_value": ".xml",
        }],
    }


def test_qmcpack_review_lints_present_included_xml(tmp_path):
    input_path = _write(tmp_path, "simulation.xml", """\
<simulation>
  <include href="wavefunction.xml"/>
  <qmc method="vmc"/>
</simulation>
""")
    _write(tmp_path, "wavefunction.xml", """\
<wavefunction>
  <determinantset twistnum="0"><slaterdeterminant/></determinantset>
</wavefunction>
""")

    reviewed = review_input(QMCPACK, input_path, resolved_by="explicit")

    assert reviewed["assessment"]["verdict"]["label"] == "review_required"
    assert [issue["message"] for issue in reviewed["evidence"]["lint"]["issues"]] == [
        "Included QMCPACK XML 'wavefunction.xml': <determinantset> has twistnum "
        "but no twist attribute; specify twist to avoid an ambiguous selection.",
        "Included QMCPACK XML 'wavefunction.xml': <determinantset> contains a "
        "legacy inline <slaterdeterminant>; move SPO setup to a top-level "
        "<sposet_collection>.",
    ]


def test_qmcpack_rejects_missing_direct_include(tmp_path):
    input_path = _write(tmp_path, "Ce.xml", """\
<simulation>
  <include href="missing.ptcl.xml"/>
  <qmc method="vmc"/>
</simulation>
""")

    parsed = QMCPACK.parser.parse_input(str(input_path))
    reviewed = review_input(QMCPACK, input_path, resolved_by="explicit")

    assert parsed["include_review"] == {
        "status": "incomplete",
        "resolution": {
            "base_path": str(tmp_path.resolve()),
            "basis": "input_directory_assumption",
        },
        "entries": [{
            "href": "missing.ptcl.xml",
            "path": str((tmp_path / "missing.ptcl.xml").resolve()),
            "status": "missing",
        }],
    }
    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert reviewed["evidence"]["lint"]["issues"] == [{
        "level": "error",
        "message": (
            "Referenced XML include 'missing.ptcl.xml' was not found "
            "relative to the reviewed input."
        ),
        "line": None,
        "suggested_fix": (
            "Provide the included XML file in the input directory or correct "
            "the include href."
        ),
    }]


def test_qmcpack_finds_hdf5_references_in_nested_includes(tmp_path):
    input_path = _write(tmp_path, "simulation.xml", """\
<simulation>
  <include href="wavefunction.xml"/>
  <qmc method="vmc"/>
</simulation>
""")
    wavefunction = _write(tmp_path, "wavefunction.xml", """\
<wavefunction>
  <determinantset href="orbitals.pwscf.h5"/>
</wavefunction>
""")
    sidecar = _write_hdf5(tmp_path, "orbitals.pwscf.h5")

    parsed = QMCPACK.parser.parse_input(str(input_path))
    reviewed = review_input(QMCPACK, input_path, resolved_by="explicit")

    assert parsed["include_review"]["discovered_references"] == [{
        "kind": "determinantset",
        "href": "orbitals.pwscf.h5",
        "source_path": str(wavefunction.resolve()),
    }]
    assert parsed["hdf5_sidecar_review"]["entries"] == [{
        "href": "orbitals.pwscf.h5",
        "path": str(sidecar.resolve()),
        "reference_kinds": ["determinantset"],
        "source_path": str(wavefunction.resolve()),
        "status": "present",
        "freshness": "not_older_than_input",
        "size_bytes": 8,
        "modified_ns": sidecar.stat().st_mtime_ns,
        "hdf5_signature_offset": 0,
    }]
    assert reviewed["assessment"]["verdict"]["label"] == "checks_passed"


def test_qmcpack_rejects_include_cycles(tmp_path):
    input_path = _write(tmp_path, "simulation.xml", """\
<simulation>
  <include href="wavefunction.xml"/>
  <qmc method="vmc"/>
</simulation>
""")
    _write(tmp_path, "wavefunction.xml", """\
<wavefunction>
  <include href="simulation.xml"/>
</wavefunction>
""")

    parsed = QMCPACK.parser.parse_input(str(input_path))
    reviewed = review_input(QMCPACK, input_path, resolved_by="explicit")

    assert parsed["include_review"]["entries"][-1] == {
        "href": "simulation.xml",
        "path": str(input_path.resolve()),
        "status": "cycle",
    }
    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert reviewed["evidence"]["lint"]["issues"] == [{
        "level": "error",
        "message": (
            "Referenced XML include 'simulation.xml' creates an include cycle."
        ),
        "line": None,
        "suggested_fix": None,
    }]
def test_qmcpack_reviews_authoritative_variational_parameter_sidecars(tmp_path):
    input_path = _write(tmp_path, "optimized.xml", """\
<simulation>
  <override_variational_parameters href="optimized.vp.h5"/>
  <qmc method="vmc"/>
</simulation>
""")
    sidecar = _write_hdf5(tmp_path, "optimized.vp.h5")
    os.utime(sidecar, ns=(1_000_000_000, 1_000_000_000))
    os.utime(input_path, ns=(2_000_000_000, 2_000_000_000))

    parsed = QMCPACK.parser.parse_input(str(input_path))
    reviewed = review_input(QMCPACK, input_path, resolved_by="explicit")

    assert parsed["hdf5_sidecar_review"] == {
        "status": "reviewed",
        "resolution": {
            "base_path": str(tmp_path.resolve()),
            "basis": "input_directory_assumption",
        },
        "entries": [{
            "href": "optimized.vp.h5",
            "path": str(sidecar.resolve()),
            "reference_kinds": ["override_variational_parameters"],
            "status": "present",
            "freshness": "older_than_input",
            "size_bytes": 8,
            "modified_ns": 1_000_000_000,
            "hdf5_signature_offset": 0,
        }],
    }
    assert reviewed["assessment"]["verdict"]["label"] == "review_required"
    assert reviewed["evidence"]["lint"]["issues"] == [{
        "level": "warning",
        "message": (
            "HDF5 sidecar 'optimized.vp.h5' is older than the XML input; "
            "confirm that the reference still matches this input."
        ),
        "line": None,
        "suggested_fix": None,
    }]


def test_qmcpack_rejects_invalid_authoritative_variational_sidecar(tmp_path):
    input_path = _write(tmp_path, "optimized.xml", """\
<simulation>
  <override_variational_parameters href="optimized.vp.h5"/>
  <qmc method="vmc"/>
</simulation>
""")
    _write(tmp_path, "optimized.vp.h5", "not an HDF5 file")

    parsed = QMCPACK.parser.parse_input(str(input_path))
    reviewed = review_input(QMCPACK, input_path, resolved_by="explicit")

    assert parsed["hdf5_sidecar_review"]["entries"][0]["status"] == "invalid"
    assert (
        parsed["hdf5_sidecar_review"]["entries"][0]["hdf5_signature_offset"]
        is None
    )
    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert reviewed["evidence"]["lint"]["issues"] == [{
        "level": "error",
        "message": (
            "Authoritative variational-parameter sidecar 'optimized.vp.h5' "
            "does not contain an HDF5 signature."
        ),
        "line": None,
        "suggested_fix": (
            "Provide the HDF5 variational-parameter sidecar written by QMCPACK "
            "or remove the override and re-optimize the Jastrow."
        ),
    }]


def test_qmcpack_rejects_invalid_orbital_sidecar_as_conversion_reference(tmp_path):
    input_path = _write(tmp_path, "simulation.xml", """\
<simulation>
  <determinantset href="orbitals.pwscf.h5"/>
</simulation>
""")
    orbital_path = _write(tmp_path, "orbitals.pwscf.h5", "not an HDF5 file")

    parsed = QMCPACK.parser.parse_input(str(input_path))
    inspection = inspect_pwscf_h5_reference(
        parsed["hdf5_sidecar_review"],
        parsed["include_review"],
        orbital_path,
    )

    assert inspection["status"] == "not_ready"
    assert inspection["observed"]["matching_references"][0]["status"] == "invalid"


def test_qmcpack_rejects_missing_authoritative_variational_sidecar(tmp_path):
    input_path = _write(tmp_path, "optimized.xml", """\
<simulation>
  <override_variational_parameters href="missing.vp.h5"/>
  <qmc method="vmc"/>
</simulation>
""")

    reviewed = review_input(QMCPACK, input_path, resolved_by="explicit")

    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert reviewed["evidence"]["lint"]["issues"] == [{
        "level": "error",
        "message": (
            "Authoritative variational-parameter sidecar 'missing.vp.h5' "
            "is missing."
        ),
        "line": None,
        "suggested_fix": (
            "Provide the referenced vp.h5 file or remove the override and "
            "re-optimize the Jastrow."
        ),
    }]
