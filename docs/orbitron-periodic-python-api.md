# Orbitron periodic Python API

`inspect_periodic_electronic_structure_with_orbitron` is the first Chemtools
operation backed by Orbitron's Python API rather than its CLI. It reads one
absolute local path through the configured companion interpreter, then returns
a compact summary of periodic electronic-structure data when Orbitron provides
it.

The response preserves the source path, size, SHA-256, Orbitron package
version, and Python version. Its periodic summary contains the Fermi energy,
total magnetization when present, band-gap value and directness, band sampling
and dimensions, and DOS energy range and dimensions. Raw eigenvalue curves,
DOS curves, and projections stay out of the MCP response. That avoids turning
a large visualization payload into unbounded tool output.

The tool reports parsed evidence only. It does not validate a band gap,
establish a band ordering, select a k-point path, judge smearing or basis
settings, or compare results between electronic-structure programs. Missing
periodic data returns `unavailable_data`; it is distinct from a loader refusal.

## Fixture

[`tests/fixtures/orbitron_periodic/vasprun_band_dos.xml`](../tests/fixtures/orbitron_periodic/vasprun_band_dos.xml)
is a small synthetic VASP input with two k-points, two bands, and three DOS
energies. The pinned source hash is
`7da9d64780e54b61fc779d9fae4d8714ef5071cd565b6d88ce0878443cd1f435`.
Its 7.0 eV gap and 1.2 eV Fermi level test the parsing contract only; they are
not silicon reference values.

Run the opt-in bridge check with the configured companion interpreter:

```bash
export CHEMTOOLS_SCIENCE_PYTHON=/path/to/chemtools-science/bin/python
.venv/bin/python scripts/check_orbitron_periodic_python_api.py
```
