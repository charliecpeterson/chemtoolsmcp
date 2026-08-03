# Orbitron structure-identity Python API

`inspect_structure_identity_with_orbitron` is a fixed, read-only companion
runtime operation for one molecular or coordination-structure file. It
preserves source path, size, SHA-256, Orbitron package version, and Python
version, then reports atom and bond counts, bond-order counts, and identifier
evidence.

Formula, InChI, InChIKey, and SMILES are each returned independently as
available or unavailable. This prevents one exporter limitation from hiding
the rest of the parsed structure. The result also preserves `Dative` bond
counts when Orbitron's canonical conversion assigns them, which is useful
evidence for reviewing a coordination model before an external conversion.

The operation neither edits input nor decides that a coordinate-derived bond
model, identifier, oxidation state, or coordination assignment is chemically
correct. It reports what Orbitron constructed from the source.

## Fixture

[`tests/fixtures/orbitron_identity/zncl2.xyz`](../tests/fixtures/orbitron_identity/zncl2.xyz)
is an Orbitron-owned zinc chloride fixture with SHA-256
`bdd9c6c2bf1e578bebd137cb33d02bdd3a3cdd032af6e03dc89957fc063ed8e8`.
Orbitron currently derives two `Dative` bonds, formula `Cl2Zn`, and all four
identifier forms from it. These are parser-contract expectations, not a zinc
chloride reference calculation.

Run the opt-in bridge check with the configured companion interpreter:

```bash
export CHEMTOOLS_SCIENCE_PYTHON=/path/to/chemtools-science/bin/python
.venv/bin/python scripts/check_orbitron_structure_identity_python_api.py
```
