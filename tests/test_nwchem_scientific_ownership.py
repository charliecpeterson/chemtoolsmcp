"""NWChem policy and syntax decisions stay outside MCP handlers."""

from chemtools.mcp.tools import nwchem_analysis, nwchem_input
from chemtools.programs.nwchem.input.general import build_scf_module_settings


def test_scf_module_settings_use_program_owned_nwchem_syntax():
    assert build_scf_module_settings(
        "SCF",
        scf_type="uhf",
        nopen=2,
        maxiter=150,
        thresh=1e-7,
    ) == ["uhf", "nopen 2", "maxiter 150", "thresh 1.00e-07"]
    assert build_scf_module_settings("dft", nopen=2) is None


def test_nwchem_analysis_handlers_only_translate_policy_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nwchem_analysis,
        "suggest_nwchem_recovery",
        lambda **kwargs: calls.append(("recovery", kwargs))
        or {"kind": "recovery"},
    )
    monkeypatch.setattr(
        nwchem_analysis,
        "suggest_multiplicity_scan_from_source",
        lambda **kwargs: calls.append(("multiplicity", kwargs))
        or {"kind": "multiplicity"},
    )

    assert nwchem_analysis._handle_suggest_nwchem_recovery({
        "output_file": "run.out",
        "mode": "state",
    }) == {"kind": "recovery"}
    assert nwchem_analysis._handle_suggest_multiplicity_scan({
        "input_file": "run.nw",
        "output_dir": "scan",
    }) == {"kind": "multiplicity"}
    assert calls == [
        (
            "recovery",
            {
                "output_path": "run.out",
                "input_path": None,
                "expected_metal_elements": None,
                "expected_somo_count": None,
                "mode": "state",
            },
        ),
        (
            "multiplicity",
            {
                "input_file": "run.nw",
                "elements": None,
                "charge": None,
                "multiplicity": None,
                "metal_oxidation_states": None,
                "output_dir": "scan",
            },
        ),
    ]


def test_nwchem_input_handler_delegates_scf_directive_rendering(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nwchem_input,
        "build_scf_module_settings",
        lambda module, **kwargs: calls.append((module, kwargs)) or ["uhf"],
    )
    monkeypatch.setattr(
        nwchem_input,
        "create_nwchem_input",
        lambda **kwargs: kwargs,
    )

    drafted = nwchem_input._handle_create_nwchem_input({
        "geometry_file": "atoms.xyz",
        "basis_assignments": {"Fe": "def2-svp"},
        "module": "scf",
        "scf_type": "uhf",
    })

    assert drafted["module_settings"] == ["uhf"]
    assert calls == [(
        "scf",
        {
            "scf_type": "uhf",
            "nopen": None,
            "maxiter": None,
            "thresh": None,
        },
    )]
