"""Parser for the Molcas `xmldump` file.

Each Molcas module appends a small XML record to ``$WorkDir/xmldump`` and
that file is copied back to the input directory at the end of the run. The
file is NOT well-formed XML at the top level — it's a concatenation of
``<module>...</module>`` fragments with no root element — so we wrap it
before handing to ``ElementTree``.

Tag inventory observed across our dogfood runs (Molcas 26.02 container):

  <module value="…">      …per-module wrapper. Modules seen: seward, scf,
                          rasscf, caspt2, mckinley, mclr, alaska, slapaf,
                          rassi.
  <method>                "rhf" / "uhf" / "rohf" (SCF only)
  <energy>                Total SCF energy (au) — only SCF emits this
  <kinetic>               Kinetic energy (au)
  <virial>                Virial coefficient (≈ 2 for HF)
  <spin>                  ⟨S²⟩ (SCF only)
  <potnuc>                Nuclear repulsion (au)
  <energy1el>             One-electron energy (au)
  <energy2el>             Two-electron energy (au)
  <nsym>                  Number of irreps
  <nbas>                  Number of basis functions
  <norb>                  Number of MOs
  <nocc>                  Number of occupied orbitals
  <FormalCharge>          Total molecular charge
  <dipole>                Dipole moment in Debye; child <v> per component

Most useful applications:

  * Track SCF energy progression across opt-loop iterations (each iter has
    its own <module value="scf"> entry).
  * Extract per-module dipole moments — useful for property analysis.
  * Cross-check the .log parser's energy extraction against a different
    source (xmldump is more robust to log text-format changes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import xml.etree.ElementTree as ET


_FLOAT_TAGS = {
    "energy", "kinetic", "virial", "spin", "potnuc",
    "energy1el", "energy2el", "FormalCharge",
}
_INT_TAGS = {"nsym", "nbas", "norb", "nocc"}


def _coerce(elem: ET.Element) -> Any:
    """Convert one tag's text to its semantic type."""
    text = (elem.text or "").strip()
    if not text and elem.findall("v"):
        # vector type (dipole)
        return [float(v.text.strip()) for v in elem.findall("v") if v.text]
    if not text:
        return None
    if elem.tag in _INT_TAGS:
        try:
            return int(text)
        except ValueError:
            return text
    if elem.tag in _FLOAT_TAGS:
        try:
            return float(text)
        except ValueError:
            return text
    # Strings like method "rhf" come with quotes
    return text.strip('"').strip()


def parse_xmldump(text: str) -> dict[str, Any]:
    """Parse a Molcas xmldump text into a structured dict.

    Returns
    -------
    dict with:
      modules         list of {module, energy, dipole, ...} dicts, one per
                      <module> entry (preserves order — useful for opt
                      iteration tracing).
      n_modules       int
      module_counts   dict of {module_name: count} (how many times each
                      module appeared).
      energy_trace    list of {module_index, module, energy_au} for entries
                      that carried an `<energy>` tag — useful for plotting
                      SCF-per-iteration during opt loops.
      final_module    the last module entry — closest analog of "final
                      run state".
    """
    # xmldump is a sequence of <module>...</module> fragments without a
    # root. Wrap it so ET.fromstring can parse.
    if not text.strip():
        return {"modules": [], "n_modules": 0, "module_counts": {}, "energy_trace": [], "final_module": None}
    wrapped = "<root>\n" + text + "\n</root>\n"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        # Some Molcas modules write malformed bits — fall back to a
        # regex-driven module split.
        return _fallback_regex_parse(text)

    modules: list[dict[str, Any]] = []
    for mod in root.findall("module"):
        rec: dict[str, Any] = {"module": (mod.get("value") or "").lower()}
        for child in mod:
            val = _coerce(child)
            if val is not None:
                # Some tags (like <method>) appear once; <dipole> is a vec.
                rec[child.tag] = val
        modules.append(rec)

    energy_trace = [
        {"module_index": i, "module": m["module"], "energy_au": m["energy"]}
        for i, m in enumerate(modules)
        if "energy" in m and isinstance(m["energy"], (int, float))
    ]
    counts: dict[str, int] = {}
    for m in modules:
        counts[m["module"]] = counts.get(m["module"], 0) + 1

    return {
        "modules": modules,
        "n_modules": len(modules),
        "module_counts": counts,
        "energy_trace": energy_trace,
        "final_module": modules[-1] if modules else None,
    }


def _fallback_regex_parse(text: str) -> dict[str, Any]:
    """Regex-only parser used when XML is malformed."""
    modules: list[dict[str, Any]] = []
    # Each module block starts with <module value="..."> and ends with </module>
    for m in re.finditer(
        r'<module value="(\w+)">([\s\S]*?)</module>', text,
    ):
        rec: dict[str, Any] = {"module": m.group(1).lower()}
        body = m.group(2)
        for tag_m in re.finditer(
            r'<(\w+)\b[^>]*>([\s\S]*?)</\1>', body,
        ):
            tag = tag_m.group(1)
            inner = tag_m.group(2)
            if "<v>" in inner:
                rec[tag] = [float(v) for v in re.findall(r'<v>\s*([+-]?\d+\.?\d*[Ee]?[+-]?\d*)\s*</v>', inner)]
            else:
                txt = inner.strip()
                try:
                    if tag in _INT_TAGS:
                        rec[tag] = int(txt)
                    elif tag in _FLOAT_TAGS:
                        rec[tag] = float(txt)
                    else:
                        rec[tag] = txt.strip('"')
                except ValueError:
                    rec[tag] = txt
        modules.append(rec)
    energy_trace = [
        {"module_index": i, "module": m["module"], "energy_au": m["energy"]}
        for i, m in enumerate(modules)
        if "energy" in m and isinstance(m["energy"], (int, float))
    ]
    counts: dict[str, int] = {}
    for m in modules:
        counts[m["module"]] = counts.get(m["module"], 0) + 1
    return {
        "modules": modules,
        "n_modules": len(modules),
        "module_counts": counts,
        "energy_trace": energy_trace,
        "final_module": modules[-1] if modules else None,
    }


def parse_xmldump_file(path: str) -> dict[str, Any]:
    """Parse a Molcas xmldump file at the given path. Returns the same
    shape as parse_xmldump(). Raises FileNotFoundError if absent."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"xmldump file not found: {path}")
    return parse_xmldump(p.read_text(encoding="utf-8", errors="replace"))
