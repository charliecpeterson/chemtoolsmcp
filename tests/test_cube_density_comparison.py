"""Contracts for strict same-grid electron-density CUBE comparison."""

import math

import pytest

from chemtools.core.cube import (
    BOHR_TO_ANGSTROM,
    compare_cube_densities,
    compare_cube_orbitals,
)
from chemtools.core.cube_subspace import compare_cube_orbital_subspaces
from chemtools.mcp.dispatch import dispatch_tool, tool_definitions


def _write_cube(
    tmp_path,
    name,
    values,
    *,
    title="Electron density",
    comment="Total density on a common grid",
    atom_position=(0.0, 0.0, 0.0),
    nuclear_charge=1.0,
    origin=(0.0, 0.0, 0.0),
    z_vector=(0.0, 0.0, 1.0),
):
    path = tmp_path / name
    path.write_text(
        "\n".join((
            title,
            comment,
            f"1 {origin[0]} {origin[1]} {origin[2]}",
            "2 1.0 0.0 0.0",
            "2 0.0 1.0 0.0",
            f"2 {z_vector[0]} {z_vector[1]} {z_vector[2]}",
            f"1 {nuclear_charge} {atom_position[0]} {atom_position[1]} {atom_position[2]}",
            " ".join(str(value) for value in values),
            "",
        )),
        encoding="utf-8",
    )
    return path


def test_cube_density_comparison_normalizes_values_before_metrics(tmp_path):
    reference = _write_cube(tmp_path, "reference_density.cube", [1.0] * 8)
    candidate = _write_cube(
        tmp_path,
        "candidate_density.cube",
        [2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )

    comparison = compare_cube_densities(
        str(reference),
        str(candidate),
        reference_density_unit="electron_per_bohr3",
        candidate_density_unit="electron_per_bohr3",
    )

    inverse_bohr_volume = BOHR_TO_ANGSTROM ** -3
    assert comparison["status"] == "comparable"
    assert comparison["compatibility"]["findings"] == []
    assert comparison["reference"]["inferred_kind"] == "density"
    assert comparison["reference"]["sha256"]
    assert comparison["metrics"] == {
        "quadrature": "uniform_trapezoidal_grid",
        "reference_integrated_electrons": pytest.approx(1.0),
        "candidate_integrated_electrons": pytest.approx(1.0),
        "integrated_electron_difference": pytest.approx(0.0),
        "l1_difference_electrons": pytest.approx(0.25),
        "relative_l1_difference": pytest.approx(0.25),
        "l2_difference_electron_per_angstrom_1p5": pytest.approx(
            0.5 * math.sqrt(inverse_bohr_volume)
        ),
        "rms_density_difference_electron_per_angstrom3": pytest.approx(
            0.5 * inverse_bohr_volume
        ),
        "max_abs_density_difference_electron_per_angstrom3": pytest.approx(
            inverse_bohr_volume
        ),
    }


def test_cube_density_comparison_accepts_equivalent_declared_units(tmp_path):
    reference = _write_cube(tmp_path, "reference_density.cube", [1.0] * 8)
    density_in_angstrom3 = BOHR_TO_ANGSTROM ** -3
    candidate = _write_cube(
        tmp_path,
        "candidate_density.cube",
        [density_in_angstrom3] * 8,
    )

    comparison = compare_cube_densities(
        str(reference),
        str(candidate),
        reference_density_unit="electron_per_bohr3",
        candidate_density_unit="electron_per_angstrom3",
    )

    assert comparison["status"] == "comparable"
    assert comparison["metrics"]["l1_difference_electrons"] == pytest.approx(0.0)
    assert comparison["metrics"]["max_abs_density_difference_electron_per_angstrom3"] == pytest.approx(0.0)


def test_cube_density_comparison_accepts_different_cube_charge_headers(tmp_path):
    reference = _write_cube(tmp_path, "reference_density.cube", [1.0] * 8)
    candidate = _write_cube(
        tmp_path,
        "candidate_density.cube",
        [1.0] * 8,
        nuclear_charge=0.0,
    )

    comparison = compare_cube_densities(
        str(reference),
        str(candidate),
        reference_density_unit="electron_per_bohr3",
        candidate_density_unit="electron_per_bohr3",
    )

    assert comparison["status"] == "comparable"
    assert comparison["compatibility"]["warnings"] == [{
        "code": "cube_nuclear_charge_header_difference",
        "message": (
            "CUBE nuclear-charge header values differ, but atomic numbers "
            "and positions match."
        ),
    }]


def test_cube_density_comparison_keeps_unknown_field_identity_as_uncertainty(tmp_path):
    reference = _write_cube(
        tmp_path,
        "reference.cube",
        [1.0] * 8,
        title="Reference field",
        comment="Generated on a common grid",
    )
    candidate = _write_cube(
        tmp_path,
        "candidate.cube",
        [1.0] * 8,
        title="Candidate field",
        comment="Generated on a common grid",
    )

    comparison = compare_cube_densities(
        str(reference),
        str(candidate),
        reference_density_unit="electron_per_bohr3",
        candidate_density_unit="electron_per_bohr3",
    )

    assert comparison["status"] == "comparable"
    assert [warning["code"] for warning in comparison["compatibility"]["warnings"]] == [
        "reference_density_kind_not_identified",
        "candidate_density_kind_not_identified",
    ]


def test_cube_density_comparison_refuses_grid_geometry_and_field_mismatches(tmp_path):
    reference = _write_cube(tmp_path, "reference_density.cube", [1.0] * 8)
    candidate = _write_cube(
        tmp_path,
        "candidate_orbital.cube",
        [1.0] * 8,
        title="Molecular orbital",
        comment="Orbital field on a common grid",
        atom_position=(0.1, 0.0, 0.0),
        origin=(0.2, 0.0, 0.0),
        z_vector=(0.0, 0.0, 1.1),
    )

    comparison = compare_cube_densities(
        str(reference),
        str(candidate),
        reference_density_unit="electron_per_bohr3",
        candidate_density_unit="electron_per_bohr3",
    )

    assert comparison["status"] == "not_comparable"
    assert "metrics" not in comparison
    assert [finding["code"] for finding in comparison["compatibility"]["findings"]] == [
        "candidate_not_identified_as_density",
        "grid_origin_mismatch",
        "grid_vector_mismatch",
        "nuclear_geometry_mismatch",
    ]


def test_cube_density_comparison_rejects_extra_scalar_values(tmp_path):
    reference = _write_cube(tmp_path, "reference_density.cube", [1.0] * 8)
    candidate = _write_cube(tmp_path, "candidate_density.cube", [1.0] * 9)

    with pytest.raises(ValueError, match="has 9 values but expected 8"):
        compare_cube_densities(
            str(reference),
            str(candidate),
            reference_density_unit="electron_per_bohr3",
            candidate_density_unit="electron_per_bohr3",
        )


def test_cube_density_comparison_is_exposed_through_the_generic_mcp_surface(tmp_path):
    reference = _write_cube(tmp_path, "reference_density.cube", [1.0] * 8)
    candidate = _write_cube(tmp_path, "candidate_density.cube", [1.0] * 8)

    payload = dispatch_tool("compare_cube_densities", {
        "reference_cube": str(reference),
        "candidate_cube": str(candidate),
        "reference_density_unit": "electron_per_bohr3",
        "candidate_density_unit": "electron_per_bohr3",
    })

    assert payload["status"] == "comparable"
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "compare_cube_densities"
    )
    assert definition["inputSchema"]["required"] == [
        "reference_cube",
        "candidate_cube",
        "reference_density_unit",
        "candidate_density_unit",
    ]


def test_cube_orbital_comparison_aligns_a_sign_flip(tmp_path):
    reference = _write_cube(
        tmp_path,
        "reference_orbital.cube",
        [1.0] * 8,
        title="Molecular orbital",
        comment="Matched orbital on a common grid",
    )
    candidate = _write_cube(
        tmp_path,
        "candidate_orbital.cube",
        [-1.0] * 8,
        title="Molecular orbital",
        comment="Matched orbital on a common grid",
    )

    comparison = compare_cube_orbitals(
        str(reference),
        str(candidate),
        reference_orbital_label="alpha HOMO, orbital 5",
        candidate_orbital_label="alpha HOMO, orbital 5",
    )

    assert comparison["status"] == "comparable"
    assert comparison["comparison_scope"] == (
        "one_explicitly_matched_nondegenerate_orbital"
    )
    assert comparison["reference"]["orbital_label"] == "alpha HOMO, orbital 5"
    assert comparison["metrics"] == {
        "signed_normalized_overlap": pytest.approx(-1.0),
        "phase_alignment": "flip_candidate_sign",
        "phase_aligned_normalized_overlap": pytest.approx(1.0),
        "phase_aligned_l2_distance": pytest.approx(0.0),
    }


def test_cube_orbital_comparison_refuses_density_and_zero_norm(tmp_path):
    density = _write_cube(tmp_path, "reference_density.cube", [1.0] * 8)
    orbital = _write_cube(
        tmp_path,
        "candidate_orbital.cube",
        [1.0] * 8,
        title="Molecular orbital",
        comment="Matched orbital on a common grid",
    )

    wrong_kind = compare_cube_orbitals(
        str(density),
        str(orbital),
        reference_orbital_label="orbital 1",
        candidate_orbital_label="orbital 1",
    )

    assert wrong_kind["status"] == "not_comparable"
    assert wrong_kind["compatibility"]["findings"] == [{
        "code": "reference_not_identified_as_orbital",
        "message": (
            "Reference CUBE header and filename do not identify an orbital field."
        ),
        "observed": "density",
    }]

    zero_orbital = _write_cube(
        tmp_path,
        "zero_orbital.cube",
        [0.0] * 8,
        title="Molecular orbital",
        comment="Matched orbital on a common grid",
    )
    zero_norm = compare_cube_orbitals(
        str(orbital),
        str(zero_orbital),
        reference_orbital_label="orbital 1",
        candidate_orbital_label="orbital 1",
    )

    assert zero_norm["status"] == "not_comparable"
    assert zero_norm["compatibility"]["findings"] == [{
        "code": "orbital_grid_norm_zero",
        "reference_norm": pytest.approx(BOHR_TO_ANGSTROM ** 1.5),
        "candidate_norm": pytest.approx(0.0),
    }]


def test_cube_orbital_comparison_is_exposed_through_the_generic_mcp_surface(tmp_path):
    reference = _write_cube(
        tmp_path,
        "reference_orbital.cube",
        [1.0] * 8,
        title="Molecular orbital",
        comment="Matched orbital on a common grid",
    )
    candidate = _write_cube(
        tmp_path,
        "candidate_orbital.cube",
        [1.0] * 8,
        title="Molecular orbital",
        comment="Matched orbital on a common grid",
    )

    payload = dispatch_tool("compare_cube_orbitals", {
        "reference_cube": str(reference),
        "candidate_cube": str(candidate),
        "reference_orbital_label": "orbital 1",
        "candidate_orbital_label": "orbital 1",
    })

    assert payload["status"] == "comparable"
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "compare_cube_orbitals"
    )
    assert definition["inputSchema"]["required"] == [
        "reference_cube",
        "candidate_cube",
        "reference_orbital_label",
        "candidate_orbital_label",
    ]


def test_cube_orbital_subspace_comparison_is_phase_and_rotation_invariant(tmp_path):
    inverse_sqrt2 = 1.0 / math.sqrt(2.0)
    reference_one = _write_cube(
        tmp_path,
        "reference_one_orbital.cube",
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Reference degenerate subspace",
    )
    reference_two = _write_cube(
        tmp_path,
        "reference_two_orbital.cube",
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Reference degenerate subspace",
    )
    candidate_one = _write_cube(
        tmp_path,
        "candidate_one_orbital.cube",
        [inverse_sqrt2, inverse_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Candidate degenerate subspace",
    )
    candidate_two = _write_cube(
        tmp_path,
        "candidate_two_orbital.cube",
        [inverse_sqrt2, -inverse_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Candidate degenerate subspace",
    )

    comparison = compare_cube_orbital_subspaces(
        [
            {"path": str(reference_one), "orbital_label": "reference 1"},
            {"path": str(reference_two), "orbital_label": "reference 2"},
        ],
        [
            {"path": str(candidate_one), "orbital_label": "candidate a"},
            {"path": str(candidate_two), "orbital_label": "candidate b"},
        ],
    )

    assert comparison["status"] == "comparable"
    assert comparison["metrics"]["principal_overlap_singular_values"] == pytest.approx([1.0, 1.0])
    assert comparison["metrics"]["principal_angles_degrees"] == pytest.approx([0.0, 0.0])
    assert comparison["metrics"]["least_principal_overlap"] == pytest.approx(1.0)
    assert comparison["metrics"]["projection_frobenius_distance"] == pytest.approx(0.0)
    assert comparison["metrics"]["cross_overlap_matrix"]["reference_orbital_labels"] == [
        "reference 1",
        "reference 2",
    ]


def test_cube_orbital_subspace_comparison_refuses_rank_deficiency(tmp_path):
    reference_one = _write_cube(
        tmp_path,
        "reference_one_orbital.cube",
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Reference degenerate subspace",
    )
    reference_two = _write_cube(
        tmp_path,
        "reference_two_orbital.cube",
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Reference degenerate subspace",
    )
    candidate_one = _write_cube(
        tmp_path,
        "candidate_one_orbital.cube",
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Candidate degenerate subspace",
    )
    candidate_two = _write_cube(
        tmp_path,
        "candidate_two_orbital.cube",
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Candidate degenerate subspace",
    )

    comparison = compare_cube_orbital_subspaces(
        [
            {"path": str(reference_one), "orbital_label": "reference 1"},
            {"path": str(reference_two), "orbital_label": "reference 2"},
        ],
        [
            {"path": str(candidate_one), "orbital_label": "candidate a"},
            {"path": str(candidate_two), "orbital_label": "candidate b"},
        ],
    )

    assert comparison["status"] == "not_comparable"
    assert [finding["code"] for finding in comparison["compatibility"]["findings"]] == [
        "candidate_subspace_rank_deficient",
    ]


def test_cube_orbital_subspace_comparison_is_exposed_through_generic_mcp(tmp_path):
    reference_one = _write_cube(
        tmp_path,
        "reference_one_orbital.cube",
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Reference degenerate subspace",
    )
    reference_two = _write_cube(
        tmp_path,
        "reference_two_orbital.cube",
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        title="Molecular orbital",
        comment="Reference degenerate subspace",
    )
    payload = dispatch_tool("compare_cube_orbital_subspaces", {
        "reference_orbitals": [
            {"path": str(reference_one), "orbital_label": "reference 1"},
            {"path": str(reference_two), "orbital_label": "reference 2"},
        ],
        "candidate_orbitals": [
            {"path": str(reference_one), "orbital_label": "candidate a"},
            {"path": str(reference_two), "orbital_label": "candidate b"},
        ],
    })

    assert payload["status"] == "comparable"
    definition = next(
        item
        for item in tool_definitions()
        if item["name"] == "compare_cube_orbital_subspaces"
    )
    assert definition["inputSchema"]["required"] == [
        "reference_orbitals",
        "candidate_orbitals",
    ]
