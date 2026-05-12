"""Helpers for the bundled OpenMolcas basis library.

The library lives at chemtools/data/molcas/basis_library/ — one file per basis
set name (e.g. ANO-S, ANO-RCC, 6-31G, AUG-CC-PVDZ, ...). Each file is a
concatenation of per-element entries marked by a `/` line of the form

    /<ELEMENT>.<LIBRARY>.<AUTHOR>.<primitive>.<contraction>.

The way users name a basis in a Molcas input is the abbreviated label
`<ELEMENT>.<LIBRARY>...<contraction>.` — the `...` means "default author and
primitive set". Most agents will let us pick the contraction; we expose a
default-contraction picker plus a label-pass-through path for power users.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import chemtools as _chemtools_pkg


_DATA_DIR = Path(_chemtools_pkg.__file__).resolve().parent / "data" / "molcas" / "basis_library"

_LABEL_LINE_RE = re.compile(r"^/([A-Za-z]{1,3})\.([^.]+)\.([^.]*)\.([^.]+)\.([^.]+)\.")


@lru_cache(maxsize=None)
def list_basis_sets() -> list[str]:
    """All basis-set names (filenames) in the bundled library, sorted."""
    if not _DATA_DIR.is_dir():
        return []
    return sorted(
        p.name for p in _DATA_DIR.iterdir()
        if p.is_file() and not p.name.endswith(".README")
    )


@lru_cache(maxsize=None)
def list_elements_for_basis(basis_name: str) -> list[str]:
    """Return the element symbols defined in a given basis library file."""
    path = _DATA_DIR / basis_name
    if not path.is_file():
        return []
    elements: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _LABEL_LINE_RE.match(line)
        if not m:
            continue
        sym = m.group(1)
        sym = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
        if sym not in seen:
            seen.add(sym)
            elements.append(sym)
    return elements


@lru_cache(maxsize=None)
def list_contractions_for(basis_name: str, element: str) -> list[dict]:
    """All available `(author, primitive, contraction)` entries for one element."""
    path = _DATA_DIR / basis_name
    if not path.is_file():
        return []
    sym = element[0].upper() + element[1:].lower() if len(element) > 1 else element.upper()
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _LABEL_LINE_RE.match(line)
        if not m:
            continue
        e = m.group(1)
        e = e[0].upper() + e[1:].lower() if len(e) > 1 else e.upper()
        if e != sym:
            continue
        out.append(
            {
                "element": e,
                "library": m.group(2),
                "author": m.group(3),
                "primitive": m.group(4),
                "contraction": m.group(5),
                "label": line[1:].rstrip(),  # full label without leading '/'
            }
        )
    return out


def default_contraction(basis_name: str, element: str) -> str | None:
    """Pick a sensible default contraction for an element in a basis library.

    Strategy:
      * If only one entry exists, use it.
      * For ANO-* libraries, prefer the LARGEST contraction (most flexibility).
      * Otherwise prefer the FIRST entry (libraries are usually ordered with
        the canonical / most common contraction first).
    """
    entries = list_contractions_for(basis_name, element)
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]["contraction"]
    if basis_name.upper().startswith("ANO"):
        return max(entries, key=lambda e: _contraction_size(e["contraction"]))["contraction"]
    return entries[0]["contraction"]


def basis_label(basis_name: str, element: str, contraction: str | None = None) -> str:
    """Build a Molcas basis label like ``C.ANO-S...3s2p1d.``.

    If `contraction` is None, uses default_contraction(). Raises ValueError if
    the element isn't in the library.
    """
    if contraction is None:
        contraction = default_contraction(basis_name, element)
        if contraction is None:
            raise ValueError(f"No {basis_name} entry for element {element!r}")
    sym = element[0].upper() + element[1:].lower() if len(element) > 1 else element.upper()
    return f"{sym}.{basis_name}...{contraction}."


def get_inline_basis_block(basis_name: str, element: str, contraction: str | None = None) -> str:
    """Extract a per-element basis entry from the library and reformat it as a
    Molcas inline-basis block.

    The bundled basis-library files use a per-element entry header like
    ``/H.ANO-S.Pierloot.7s3p.4s3p.`` — that label is a LIBRARY reference and
    is not valid inline. For inline use, Molcas requires the magic prefix
    ``<element>    / inline`` followed by ``<charge>  <L_max>`` and the per-shell
    primitive blocks.

    This function:
      1. Locates the requested (element, basis, contraction) entry,
      2. Replaces the library header with ``<element>    / inline``,
      3. Strips the library-only ``Options / EndOptions`` block (those are
         library hints, not basis primitives — they are not valid inline),
      4. Preserves the citation + descriptive comments as ``*`` comment lines.

    Returns text ready to splice into a Molcas ``Basis set ... End of basis``
    block (caller appends the coordinate lines + ``End of basis``).
    """
    if contraction is None:
        contraction = default_contraction(basis_name, element)
        if contraction is None:
            raise ValueError(f"No {basis_name} entry for element {element!r}")
    sym = element[0].upper() + element[1:].lower() if len(element) > 1 else element.upper()
    path = _DATA_DIR / basis_name
    if not path.is_file():
        raise ValueError(f"Basis library file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    # The library file's per-entry header uses arbitrary case for the basis
    # name (e.g. `/Cr.ANO-rcc.Roos...` even when the filename is ANO-RCC).
    # Match case-insensitively on the `<element>.<basis>.` prefix.
    prefix_pattern = re.compile(
        rf"^/{re.escape(sym)}\.{re.escape(basis_name)}\.", re.M | re.I
    )
    contraction_suffix_re = re.compile(
        rf"\.{re.escape(contraction)}\.\s*$", re.I
    )
    header_match = None
    for m in prefix_pattern.finditer(text):
        line_end = text.find("\n", m.start())
        candidate = text[m.start():line_end].rstrip()
        if contraction_suffix_re.search(candidate):
            header_match = (m.start(), line_end)
            break
    if header_match is None:
        raise ValueError(
            f"Basis library {basis_name} has no entry for element {element!r} "
            f"with contraction {contraction!r}"
        )
    body_start = header_match[1] + 1
    next_entry = text.find("\n/", body_start)
    body_end = next_entry if next_entry != -1 else len(text)
    raw_body = text[body_start:body_end]

    return _convert_library_entry_to_inline(raw_body, sym)


_FIRST_DATA_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s+\d+\s*$")
_LBLOCK_HEADER_RE = re.compile(r"^\s*\*\s*[a-zA-Z]-type functions", re.I)
_NUM_RE = re.compile(r"-?\d+\.\d+(?:[EeDd][+-]?\d+)?|-?\d+")


def _convert_library_entry_to_inline(raw_body: str, sym: str) -> str:
    """Translate a library-format entry body into a Molcas inline-basis block.

    The bundled library entries have THREE pieces that don't belong inline:
      1. **Prologue**: citation + descriptive lines BEFORE the first numeric
         data — Molcas's inline parser would treat them as data. Convert to
         ``*``-prefixed comments.
      2. **Options/EndOptions block**: library hints (FockOperator,
         OrbitalEnergies). Strip it.
      3. **Per-L extras**: when Options enabled OrbitalEnergies or
         FockOperator, each ``* X-type functions`` block has trailing data
         AFTER the coefficient matrix (e.g. ``<n_orb_energies>`` then the
         orbital energies). With Options stripped, those extras become
         orphans that Molcas misreads as the next L-block's ``n_prim n_cont``.
         We walk each L-block and emit only the spec we need: ``n_prim
         n_cont`` + ``n_prim`` exponents + ``n_prim × n_cont`` coefficients.
    """
    lines = raw_body.splitlines()
    # Pass 1: strip Options/EndOptions block.
    pass1: list[str] = []
    in_options = False
    for line in lines:
        s = line.strip().lower()
        if not in_options and s == "options":
            in_options = True
            continue
        if in_options:
            if s == "endoptions":
                in_options = False
            continue
        pass1.append(line)

    # Pass 2: walk lines, separating prologue (before charge+L_max) from
    # the per-L blocks; for each L block, emit only n_prim+n_cont +
    # exponents + coefficients (drop trailing per-L extras).
    out: list[str] = []
    i = 0
    n = len(pass1)
    # Prologue: convert non-* non-numeric lines into * comments
    while i < n:
        line = pass1[i]
        if _FIRST_DATA_RE.match(line):
            # We hit the charge + L_max line — emit it and break.
            out.append(line)
            i += 1
            break
        s = line.strip()
        if not s:
            out.append(line)
        elif s.startswith("*"):
            out.append(line)
        else:
            out.append(f"* {s}")
        i += 1

    # L-blocks: each starts with a `* X-type functions` header (preserved as
    # a comment), followed by an `n_prim n_cont` integer pair, then the
    # exponents (n_prim floats), then the coefficient matrix (n_prim × n_cont
    # floats). Drop everything between the coefficient matrix and the next
    # L-block header.
    while i < n:
        line = pass1[i]
        if _LBLOCK_HEADER_RE.match(line):
            out.append(line)
            i += 1
            # Find the n_prim n_cont line (could be preceded by blank/comment lines)
            while i < n and not pass1[i].strip():
                out.append(pass1[i])
                i += 1
            if i >= n:
                break
            sizes_line = pass1[i]
            sizes_tokens = sizes_line.split()
            try:
                n_prim = int(sizes_tokens[0])
                n_cont = int(sizes_tokens[1])
            except (ValueError, IndexError):
                # Unexpected format — emit verbatim and move on
                out.append(sizes_line)
                i += 1
                continue
            out.append(sizes_line)
            i += 1
            # Read n_prim exponent values (one or more per line)
            collected = 0
            while collected < n_prim and i < n:
                tokens = _NUM_RE.findall(pass1[i])
                if not tokens or pass1[i].strip().startswith("*"):
                    out.append(pass1[i])
                    i += 1
                    continue
                out.append(pass1[i])
                collected += len(tokens)
                i += 1
            # Read n_prim * n_cont coefficient values
            n_coeffs_needed = n_prim * n_cont
            collected = 0
            while collected < n_coeffs_needed and i < n:
                tokens = _NUM_RE.findall(pass1[i])
                if not tokens or pass1[i].strip().startswith("*"):
                    out.append(pass1[i])
                    i += 1
                    continue
                out.append(pass1[i])
                collected += len(tokens)
                i += 1
            # Drop trailing per-L extras (orbital energies, Fock entries) until
            # we reach the next `* X-type functions` header or end of body.
            while i < n and not _LBLOCK_HEADER_RE.match(pass1[i]):
                i += 1
            continue
        # Lines that don't belong to any L-block are passed through (rare —
        # usually trailing whitespace / blank lines)
        out.append(line)
        i += 1

    return f"{sym}    / inline\n" + "\n".join(out).rstrip() + "\n"


def _contraction_size(contraction: str) -> int:
    """Crude 'flexibility' score of a contraction string like '7s6p3d'.

    Sums the digits before each shell letter. Larger == more flexible.
    """
    total = 0
    current = ""
    for ch in contraction:
        if ch.isdigit():
            current += ch
        elif ch.isalpha():
            if current:
                total += int(current)
                current = ""
    return total


def resolve_basis_assignments(
    basis_request: str | dict[str, str],
    elements_in_molecule: Iterable[str],
) -> dict[str, str]:
    """Map (basis spec, list of elements) → {element: full Molcas basis label}.

    Accepts:
      * String like ``"ANO-S"`` — picks default contraction per element.
      * String like ``"C.ANO-S...3s2p1d."`` — only valid for one element; will
        raise if molecule has multiple elements.
      * Dict like ``{"C": "ANO-S", "H": "ANO-S", "Fe": "ANO-RCC"}`` — per-element
        basis name, default contraction each.
      * Dict like ``{"C": "C.ANO-S...3s2p1d."}`` — fully-qualified labels.
    """
    elements = list(dict.fromkeys(elements_in_molecule))  # preserve order, dedupe
    out: dict[str, str] = {}
    if isinstance(basis_request, str):
        # Single basis name applied to all elements
        if "..." in basis_request:
            # Fully qualified label — only one element allowed
            if len(elements) != 1:
                raise ValueError(
                    "Fully-qualified basis label only valid for single-element systems; "
                    f"molecule has {len(elements)} elements: {elements}"
                )
            return {elements[0]: basis_request}
        for el in elements:
            try:
                out[el] = basis_label(basis_request, el)
            except ValueError as e:
                raise ValueError(
                    f"Basis {basis_request!r} has no entry for element {el!r} in the bundled library"
                ) from e
        return out
    if not isinstance(basis_request, dict):
        raise TypeError(f"basis must be a str or dict, got {type(basis_request).__name__}")
    # Dict path
    for el in elements:
        spec = basis_request.get(el)
        if not spec:
            raise ValueError(f"basis dict missing entry for element {el!r}")
        if "..." in spec:
            out[el] = spec
        else:
            out[el] = basis_label(spec, el)
    return out
