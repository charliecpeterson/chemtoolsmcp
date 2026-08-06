"""The angular census must agree with GRASP on every state it is asked about.

`_multiplet_contract` derives J/parity blocks and CSF counts from a state's
configuration alone, and `_validate_csf` REJECTS any GRASP artifact whose
blocks disagree. That makes it a gate: if the census were wrong it would
either wave through bad CSF lists or refuse good ones. The catalog's
J_blocks/ncsf come from real MCDHF runs, so comparing the two is a genuine
cross-check of the census against an independent computation, not a
restatement of it.
"""

from __future__ import annotations

import pytest

from chemtools.reference.fblock import load_fblock_catalog
from chemtools.reference.fblock_grasp import _multiplet_contract, catalog_parity


def _elements():
    catalog = load_fblock_catalog()
    elements = catalog.elements
    if not isinstance(elements, (list, tuple)):
        elements = list(elements.values())
    return elements


ALL_STATES = [
    (element.symbol, state)
    for element in _elements()
    for state in element.states
]
# validate_grasp_fblock_artifacts refuses a state with no confline before it
# ever reaches the census, so those are out of scope here by construction.
VALIDATABLE = [(sym, st) for sym, st in ALL_STATES if st.confline]


def test_catalog_is_fully_populated():
    assert len(ALL_STATES) == 633


@pytest.mark.parametrize(
    "symbol,state",
    VALIDATABLE,
    ids=[f"{symbol}.{state.slug}" for symbol, state in VALIDATABLE],
)
def test_census_reproduces_grasp_blocks(symbol, state):
    derived = _multiplet_contract(state)["blocks"]
    expected = [
        {"j": j, "parity": catalog_parity(state), "ncsf": ncsf}
        for j, ncsf in zip(state.j_blocks, state.ncsf)
    ]
    assert derived == expected


def test_confline_coverage_is_exactly_the_known_gap():
    """Y is the only element without conflines. Pin it so the gap can't spread.

    These 17 states carry real GRASP J_blocks and ncsf, so MCDHF was run for
    them -- only the confline was never recorded, which is what excludes them
    from artifact validation. If another element ever loses its confline this
    fails, and if Y gains them the count drops and this fails too; either way
    the change gets looked at rather than silently widening the blind spot.
    """
    missing = sorted(
        f"{symbol}.{state.slug}"
        for symbol, state in ALL_STATES
        if not state.confline
    )
    assert len(missing) == 17
    assert {name.split(".")[0] for name in missing} == {"Y"}
    assert len(VALIDATABLE) == 616


def test_closed_shell_census_is_a_single_j0_block():
    closed = [(s, st) for s, st in VALIDATABLE if st.slug.endswith("_closed")]
    assert closed, "catalog has no validatable closed anchors"
    for symbol, state in closed:
        assert _multiplet_contract(state)["blocks"] == [
            {"j": "0", "parity": "+", "ncsf": 1}
        ], f"{symbol}.{state.slug}"
