"""Bounded NIST Atomic Spectra Database retrieval with local provenance cache.

The client only calls fixed ASD energy-level and ionization-energy endpoints.
It returns parsed tabular evidence, never a bulk mirror of the database.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


NIST_ASD_REFERENCE_SCHEMA = "chemtools.nist-asd-reference/1"
NIST_ASD_CACHE_ENV = "CHEMTOOLS_NIST_ASD_CACHE"
MAX_NIST_ASD_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_NIST_ASD_ROWS = 500
_NIST_ASD_TIMEOUT_SECONDS = 20
_SPECTRUM_PATTERN = re.compile(r"[A-Za-z0-9 +\-]+\Z")


class NistAsdError(RuntimeError):
    """A NIST ASD request could not produce bounded tabular evidence."""


def fetch_nist_asd_reference(
    kind: str,
    spectrum: str,
    *,
    row_limit: int = 200,
    refresh: bool = False,
    cache_directory: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Fetch one NIST ASD spectrum and return bounded, cached table rows."""
    normalized_kind = _normalize_kind(kind)
    normalized_spectrum = _normalize_spectrum(spectrum)
    normalized_limit = _normalize_row_limit(row_limit)
    if not isinstance(refresh, bool):
        raise ValueError("refresh must be a boolean")

    url = _query_url(normalized_kind, normalized_spectrum)
    cache_path = _cache_path(
        normalized_kind,
        normalized_spectrum,
        _resolve_cache_directory(cache_directory),
    )
    cached = _load_cached(cache_path) if not refresh else None
    if cached is None:
        body = _fetch_nist_table(url)
        cached = {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "body": body,
        }
        _write_cached(cache_path, cached)
        cache_status = "fetched"
    else:
        cache_status = "cached"

    rows = _parse_tabular_response(cached["body"])
    returned_rows = rows[:normalized_limit]
    return {
        "schema_version": NIST_ASD_REFERENCE_SCHEMA,
        "status": cache_status,
        "query": {
            "kind": normalized_kind,
            "spectrum": normalized_spectrum,
            "row_limit": normalized_limit,
        },
        "source": {
            "database": "NIST Atomic Spectra Database",
            "url": cached["url"],
            "retrieved_at": cached["retrieved_at"],
            "sha256": hashlib.sha256(cached["body"].encode("utf-8")).hexdigest(),
        },
        "table": {
            "row_count": len(rows),
            "returned_row_count": len(returned_rows),
            "truncated": len(rows) > len(returned_rows),
            "rows": returned_rows,
        },
    }


def _normalize_kind(kind: str) -> str:
    if not isinstance(kind, str):
        raise ValueError("kind must be text")
    if kind not in {"energy_levels", "ionization_energies"}:
        raise ValueError("kind must be energy_levels or ionization_energies")
    return kind


def _normalize_spectrum(spectrum: str) -> str:
    if not isinstance(spectrum, str):
        raise ValueError("spectrum must be text")
    normalized = " ".join(spectrum.split())
    if not normalized or len(normalized) > 64 or not _SPECTRUM_PATTERN.fullmatch(normalized):
        raise ValueError(
            "spectrum must use NIST ASD element, charge-state, or isoelectronic notation"
        )
    return normalized


def _normalize_row_limit(row_limit: int) -> int:
    if isinstance(row_limit, bool) or not isinstance(row_limit, int):
        raise ValueError("row_limit must be an integer")
    if not 1 <= row_limit <= MAX_NIST_ASD_ROWS:
        raise ValueError(f"row_limit must be between 1 and {MAX_NIST_ASD_ROWS}")
    return row_limit


def _query_url(kind: str, spectrum: str) -> str:
    if kind == "energy_levels":
        endpoint = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
        query = {
            "spectrum": spectrum,
            "units": "0",
            "format": "3",
            "output": "0",
            "page_size": "15",
            "multiplet_ordered": "0",
            "conf_out": "on",
            "term_out": "on",
            "level_out": "on",
            "unc_out": "on",
            "j_out": "on",
            "lande_out": "on",
            "perc_out": "on",
            "biblio": "on",
            "temp": "",
        }
    else:
        endpoint = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"
        query = {
            "spectra": spectrum,
            "units": "0",
            "format": "3",
            "order": "0",
            "at_num_out": "on",
            "sp_name_out": "on",
            "ion_charge_out": "on",
            "el_name_out": "on",
            "seq_out": "on",
            "shells_out": "on",
            "conf_out": "on",
            "level_out": "on",
            "ion_conf_out": "on",
            "e_out": "0",
            "unc_out": "on",
            "biblio": "on",
        }
    return f"{endpoint}?{urllib.parse.urlencode(query)}"


def _resolve_cache_directory(
    cache_directory: str | os.PathLike[str] | None,
) -> Path:
    configured = cache_directory or os.environ.get(NIST_ASD_CACHE_ENV)
    if configured is not None:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".chemtools" / "nist-asd"


def _cache_path(kind: str, spectrum: str, directory: Path) -> Path:
    token = hashlib.sha256(f"{kind}\0{spectrum}".encode("utf-8")).hexdigest()
    return directory / f"{token}.json"


def _load_cached(path: Path) -> dict[str, str] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        raise NistAsdError(f"could not read NIST ASD cache: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"retrieved_at", "url", "body"}:
        raise NistAsdError("NIST ASD cache has an invalid format")
    if not all(isinstance(value, str) and value for value in payload.values()):
        raise NistAsdError("NIST ASD cache has invalid values")
    return payload


def _write_cached(path: Path, payload: dict[str, str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError as error:
        raise NistAsdError(f"could not write NIST ASD cache: {error}") from error


def _fetch_nist_table(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/plain,text/tab-separated-values;q=0.9,*/*;q=0.1",
            "User-Agent": "chemtools-mcp/0.1 NIST-ASD-reference-client",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_NIST_ASD_TIMEOUT_SECONDS) as response:
            body = response.read(MAX_NIST_ASD_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, OSError) as error:
        raise NistAsdError(f"NIST ASD request failed: {error}") from error
    if len(body) > MAX_NIST_ASD_RESPONSE_BYTES:
        raise NistAsdError("NIST ASD response exceeds the 4 MiB limit")
    return body.decode("utf-8", errors="replace")


def _parse_tabular_response(body: str) -> list[dict[str, str]]:
    if not body.strip() or "\t" not in body or body.lstrip().startswith("<"):
        raise NistAsdError("NIST ASD did not return tab-delimited reference data")
    reader = csv.reader(io.StringIO(body), delimiter="\t")
    try:
        headings = next(reader)
    except StopIteration as error:
        raise NistAsdError("NIST ASD response has no table header")
    if headings and not headings[-1]:
        headings.pop()
        trailing_delimiter = True
    else:
        trailing_delimiter = False
    headings = [heading.strip() for heading in headings]
    if not all(headings):
        raise NistAsdError("NIST ASD response has an invalid table header")
    rows = []
    for values in reader:
        if trailing_delimiter and len(values) == len(headings) + 1 and not values[-1]:
            values.pop()
        if len(values) != len(headings):
            raise NistAsdError("NIST ASD response has an uneven table row")
        normalized = {
            heading: value.strip()
            for heading, value in zip(headings, values)
        }
        if any(normalized.values()):
            rows.append(normalized)
    return rows


__all__ = [
    "MAX_NIST_ASD_ROWS",
    "NIST_ASD_CACHE_ENV",
    "NIST_ASD_REFERENCE_SCHEMA",
    "NistAsdError",
    "fetch_nist_asd_reference",
]
