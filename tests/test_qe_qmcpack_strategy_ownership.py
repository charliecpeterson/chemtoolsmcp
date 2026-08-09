"""QE-to-QMCPACK scientific readiness belongs to the program package."""

from chemtools.mcp.tools import qe_qmcpack
from chemtools.programs.qe import qmcpack


def test_conversion_readiness_pins_status_precedence():
    assert qmcpack.conversion_readiness([{"status": "pass"}]) == "ready"
    assert qmcpack.conversion_readiness([
        {"status": "pass"},
        {"status": "review_required"},
    ]) == "review_required"
    assert qmcpack.conversion_readiness([
        {"status": "not_ready"},
        {"status": "review_required"},
    ]) == "not_ready"


def test_readiness_handler_only_resolves_and_parses_input(monkeypatch, tmp_path):
    source = tmp_path / "si.in"
    source.write_text("&control\n/\n", encoding="utf-8")
    parsed = {"calculation": "scf"}
    calls = []

    monkeypatch.setattr(qe_qmcpack, "parse_pw_input", lambda path: parsed)
    monkeypatch.setattr(
        qe_qmcpack,
        "inspect_conversion_readiness",
        lambda path, parsed_input: calls.append((path, parsed_input))
        or {"readiness": "ready"},
    )

    result = qe_qmcpack._handle_check_qe_qmcpack_conversion_ready({
        "qe_input": str(source),
    })

    assert result == {"readiness": "ready"}
    assert calls == [(source.resolve(), parsed)]
