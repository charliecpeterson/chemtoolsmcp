"""Periodic-aware structural checks for PWSCF trajectory frames."""

from __future__ import annotations

from chemtools.programs.qe.trajectory_analysis import analyze_pw_trajectory


def _frame(index, atoms, cell_length=10.0):
    return {
        "index": index,
        "role": "initial" if index == 0 else "last_attempted",
        "atoms": atoms,
        "cell": {
            "vectors_angstrom": [
                [cell_length, 0.0, 0.0],
                [0.0, cell_length, 0.0],
                [0.0, 0.0, cell_length],
            ],
            "periodic": [True, True, True],
        },
    }


def _trajectory(*frames):
    return {
        "status": "available",
        "frames": list(frames),
    }


def test_isolated_molecule_with_stable_bond_network_has_no_obvious_issue():
    hydrogen = [
        {"element": "H", "x": 4.0, "y": 5.0, "z": 5.0},
        {"element": "H", "x": 4.74, "y": 5.0, "z": 5.0},
    ]

    analysis = analyze_pw_trajectory(
        _trajectory(_frame(0, hydrogen), _frame(1, hydrogen))
    )

    assert analysis["scope"] == "isolated_molecule"
    assert analysis["initial"]["covalent_bond_count"] == 1
    assert analysis["final"]["fragment_count"] == 1
    assert analysis["verdict"] == {
        "status": "no_obvious_issue",
        "origin": None,
        "reasons": [
            "No close contacts, new fragmentation, or main-group "
            "overcoordination were detected by this heuristic."
        ],
        "findings": [],
    }


def test_bond_loss_and_new_fragments_are_trajectory_concerns():
    initial = [
        {"element": "H", "x": 4.0, "y": 5.0, "z": 5.0},
        {"element": "H", "x": 4.74, "y": 5.0, "z": 5.0},
    ]
    separated = [
        {"element": "H", "x": 3.5, "y": 5.0, "z": 5.0},
        {"element": "H", "x": 6.5, "y": 5.0, "z": 5.0},
    ]

    analysis = analyze_pw_trajectory(
        _trajectory(_frame(0, initial), _frame(1, separated))
    )

    assert analysis["evolution"]["covalent_bond_count_change"] == -1
    assert analysis["evolution"]["fragment_count_change"] == 1
    assert analysis["evolution"]["dangling_atom_count_change"] == 2
    assert analysis["verdict"] == {
        "status": "concerning",
        "origin": "trajectory",
        "reasons": [
            "The covalent-radius graph gains 1 disconnected fragment(s).",
            (
                "The final frame has 2 additional atom(s) without a "
                "covalent-radius neighbor."
            ),
        ],
        "findings": [
            {
                "code": "new_fragment",
                "origin": "trajectory",
                "message": (
                    "The covalent-radius graph gains 1 disconnected "
                    "fragment(s)."
                ),
            },
            {
                "code": "new_dangling_atom",
                "origin": "trajectory",
                "message": (
                    "The final frame has 2 additional atom(s) without a "
                    "covalent-radius neighbor."
                ),
            },
        ],
    }


def test_initial_close_contact_is_attributed_to_input_geometry():
    overlapping = [
        {"element": "C", "x": 5.0, "y": 5.0, "z": 5.0},
        {"element": "H", "x": 5.2, "y": 5.0, "z": 5.0},
    ]

    analysis = analyze_pw_trajectory(_trajectory(_frame(0, overlapping)))

    assert analysis["initial"]["minimum_pair_distance_angstrom"] == 0.2
    assert analysis["verdict"]["status"] == "concerning"
    assert analysis["verdict"]["origin"] == "input_geometry"
    assert analysis["verdict"]["reasons"] == [
        "The initial geometry has 1 pair(s) closer than 0.60 angstrom."
    ]
    assert analysis["verdict"]["findings"] == [{
        "code": "initial_close_contact",
        "origin": "input_geometry",
        "message": (
            "The initial geometry has 1 pair(s) closer than 0.60 angstrom."
        ),
    }]


def test_input_and_trajectory_findings_keep_separate_origins():
    initial = [
        {"element": "C", "x": 5.0, "y": 5.0, "z": 5.0},
        {"element": "H", "x": 5.2, "y": 5.0, "z": 5.0},
    ]
    separated = [
        {"element": "C", "x": 3.5, "y": 5.0, "z": 5.0},
        {"element": "H", "x": 6.5, "y": 5.0, "z": 5.0},
    ]

    analysis = analyze_pw_trajectory(
        _trajectory(_frame(0, initial), _frame(1, separated))
    )

    assert analysis["verdict"]["origin"] == "mixed"
    assert [
        (finding["code"], finding["origin"])
        for finding in analysis["verdict"]["findings"]
    ] == [
        ("initial_close_contact", "input_geometry"),
        ("new_fragment", "trajectory"),
        ("new_dangling_atom", "trajectory"),
    ]


def test_intermediate_frame_requires_an_increase_over_input_counts():
    stable = [
        {"element": "H", "x": 4.0, "y": 5.0, "z": 5.0},
        {"element": "H", "x": 4.74, "y": 5.0, "z": 5.0},
    ]
    compressed = [
        {"element": "H", "x": 4.0, "y": 5.0, "z": 5.0},
        {"element": "H", "x": 4.2, "y": 5.0, "z": 5.0},
    ]

    analysis = analyze_pw_trajectory(
        _trajectory(
            _frame(0, stable),
            _frame(1, compressed),
            _frame(2, stable),
        )
    )

    assert analysis["verdict"]["findings"] == [{
        "code": "intermediate_structural_concern",
        "origin": "trajectory",
        "message": "Frame 1 increases the close-contact count from 0 to 1.",
    }]


def test_extended_periodic_structure_avoids_molecular_connectivity_verdict():
    atoms = [
        {"element": "Fe", "x": 0.0, "y": 0.0, "z": 0.0},
        {"element": "O", "x": 2.0, "y": 2.0, "z": 2.0},
    ]

    analysis = analyze_pw_trajectory(
        _trajectory(_frame(0, atoms, cell_length=4.0))
    )

    assert analysis["scope"] == "metrics_only"
    assert analysis["initial"]["cell_volume_angstrom3"] == 64.0
    assert analysis["verdict"] == {
        "status": "not_assessed",
        "reasons": [
            "The structure lacks enough vacuum padding for molecular "
            "bond-network heuristics; use a periodic topology analysis."
        ],
    }


def test_minimum_image_distance_handles_skewed_cell():
    frame = _frame(
        0,
        [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {
                "element": "H",
                "x": 7.5,
                "y": 4.330127018922193,
                "z": 5.0,
            },
        ],
    )
    frame["cell"]["vectors_angstrom"] = [
        [10.0, 0.0, 0.0],
        [5.0, 8.660254037844386, 0.0],
        [0.0, 0.0, 10.0],
    ]

    analysis = analyze_pw_trajectory(_trajectory(frame))

    assert analysis["initial"]["minimum_pair_distance_angstrom"] == (
        7.071067811865
    )


def test_large_cell_change_is_observation_not_connectivity_verdict():
    atom = [{"element": "H", "x": 3.0, "y": 3.0, "z": 3.0}]

    analysis = analyze_pw_trajectory(
        _trajectory(
            _frame(0, atom, cell_length=10.0),
            _frame(1, atom, cell_length=7.0),
        )
    )

    assert analysis["evolution"]["large_cell_volume_change"] is True
    assert analysis["observations"] == [
        {
            "code": "large_cell_volume_change",
            "message": "The cell volume changes by -65.7 percent.",
            "impact": (
                "Check the starting cell, pressure, stress, and convergence "
                "path before treating the final cell as routine."
            ),
        }
    ]


def test_singular_cell_abstains_instead_of_crashing():
    frame = _frame(
        0,
        [{"element": "H", "x": 0.0, "y": 0.0, "z": 0.0}],
    )
    frame["cell"]["vectors_angstrom"][2] = [0.0, 0.0, 0.0]

    analysis = analyze_pw_trajectory(_trajectory(frame))

    assert analysis == {
        "schema": "qe-trajectory-structural-analysis/1",
        "scope": "not_assessed",
        "verdict": {
            "status": "not_assessed",
            "reasons": [
                "Periodic geometry metrics failed: the trajectory contains "
                "a singular periodic cell"
            ],
        },
    }


def test_frame_limit_abstains_before_unbounded_analysis():
    atom = [{"element": "H", "x": 5.0, "y": 5.0, "z": 5.0}]

    analysis = analyze_pw_trajectory(
        _trajectory(*(_frame(index, atom) for index in range(513)))
    )

    assert analysis == {
        "schema": "qe-trajectory-structural-analysis/1",
        "scope": "not_assessed",
        "verdict": {
            "status": "not_assessed",
            "reasons": [
                "Structural analysis requires 513 frames, exceeding "
                "the bounded limit of 512."
            ],
        },
    }


def test_pair_limit_abstains_before_quadratic_analysis():
    atoms = [
        {"element": "H", "x": 5.0, "y": 5.0, "z": 5.0}
        for _ in range(708)
    ]

    analysis = analyze_pw_trajectory(_trajectory(_frame(0, atoms)))

    assert analysis == {
        "schema": "qe-trajectory-structural-analysis/1",
        "scope": "not_assessed",
        "verdict": {
            "status": "not_assessed",
            "reasons": [
                "Structural analysis requires 250278 pair evaluations, "
                "exceeding the bounded limit of 250000."
            ],
        },
    }
