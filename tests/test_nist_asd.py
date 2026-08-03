"""Tests for bounded NIST ASD table retrieval and cache provenance."""

from urllib.parse import parse_qs, urlparse

import pytest

from chemtools.integrations import nist_asd


_LEVELS = (
    "Configuration\tTerm\tJ\tLevel (cm-1)\tUncertainty (cm-1)\tReference\n"
    '"2s2.2p4"\t"3P"\t"2"\t"0.000"\t"0"\t"L7288"\n'
    '"2s2.2p4"\t"3P"\t"1"\t"158.265"\t""\t""\n'
)
_IONIZATION = (
    "At. num\tSp. Name\tIon Charge\tIonization Energy (1/cm)\tUncertainty (1/cm)\t\n"
    '"8"\t"O I"\t"0"\t"109837.02"\t"0.06"\t\n'
)


def test_energy_level_fetch_caches_exact_tabular_response(tmp_path, monkeypatch):
    calls = []

    def fetch(url):
        calls.append(url)
        return _LEVELS

    monkeypatch.setattr(nist_asd, "_fetch_nist_table", fetch)

    fetched = nist_asd.fetch_nist_asd_reference(
        "energy_levels",
        " O   I ",
        row_limit=1,
        cache_directory=tmp_path,
    )
    cached = nist_asd.fetch_nist_asd_reference(
        "energy_levels",
        "O I",
        row_limit=2,
        cache_directory=tmp_path,
    )

    query = parse_qs(urlparse(calls[0]).query)
    assert query["spectrum"] == ["O I"]
    assert query["format"] == ["3"]
    assert fetched["status"] == "fetched"
    assert fetched["table"] == {
        "row_count": 2,
        "returned_row_count": 1,
        "truncated": True,
        "rows": [{
            "Configuration": "2s2.2p4",
            "Term": "3P",
            "J": "2",
            "Level (cm-1)": "0.000",
            "Uncertainty (cm-1)": "0",
            "Reference": "L7288",
        }],
    }
    assert cached["status"] == "cached"
    assert cached["table"]["returned_row_count"] == 2
    assert len(calls) == 1


def test_ionization_fetch_uses_fixed_asd_endpoint(tmp_path, monkeypatch):
    observed = {}

    def fetch(url):
        observed["url"] = url
        return _IONIZATION

    monkeypatch.setattr(nist_asd, "_fetch_nist_table", fetch)

    result = nist_asd.fetch_nist_asd_reference(
        "ionization_energies",
        "O I",
        cache_directory=tmp_path,
    )

    parsed = urlparse(observed["url"])
    assert parsed.path == "/cgi-bin/ASD/ie.pl"
    assert parse_qs(parsed.query)["spectra"] == ["O I"]
    assert result["table"]["rows"] == [{
        "At. num": "8",
        "Sp. Name": "O I",
        "Ion Charge": "0",
        "Ionization Energy (1/cm)": "109837.02",
        "Uncertainty (1/cm)": "0.06",
    }]


def test_refresh_bypasses_matching_cache(tmp_path, monkeypatch):
    calls = []

    def fetch(url):
        calls.append(url)
        return _LEVELS

    monkeypatch.setattr(nist_asd, "_fetch_nist_table", fetch)
    nist_asd.fetch_nist_asd_reference(
        "energy_levels", "O I", cache_directory=tmp_path
    )
    refreshed = nist_asd.fetch_nist_asd_reference(
        "energy_levels", "O I", refresh=True, cache_directory=tmp_path
    )

    assert refreshed["status"] == "fetched"
    assert len(calls) == 2


@pytest.mark.parametrize("kind, spectrum, row_limit", [
    ("lines", "O I", 1),
    ("energy_levels", "O I&x=1", 1),
    ("energy_levels", "O I", 501),
])
def test_request_rejects_unbounded_or_unsupported_inputs(
    tmp_path,
    kind,
    spectrum,
    row_limit,
):
    with pytest.raises(ValueError):
        nist_asd.fetch_nist_asd_reference(
            kind,
            spectrum,
            row_limit=row_limit,
            cache_directory=tmp_path,
        )


def test_request_refuses_non_tabular_nist_response(tmp_path, monkeypatch):
    monkeypatch.setattr(nist_asd, "_fetch_nist_table", lambda _: "<html>error</html>")

    with pytest.raises(nist_asd.NistAsdError, match="tab-delimited"):
        nist_asd.fetch_nist_asd_reference(
            "energy_levels", "O I", cache_directory=tmp_path
        )
