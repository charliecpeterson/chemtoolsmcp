# Orbitron NBO Python API

`inspect_nbo_with_orbitron` reads one local output through the configured
companion interpreter and returns bounded Natural Bond Orbital evidence. The
response preserves source path, size, SHA-256, Orbitron version, and Python
version. It summarizes the complete orbital count, orbital-type counts,
occupancy range, per-atom NBO-entry counts, and at most twelve bonding-orbital
samples.

Samples are restricted to `BD`, `BD*`, `LP`, and `LP*` types. Each keeps its
source number, label, occupancy, and at most five contributing atoms with
relative weight and coefficient sign. Raw NBO tables and unbounded orbital
lists never cross the MCP boundary.

NBO labels, occupancies, weights, and signs are parsed evidence from the
source calculation. The operation does not infer a unique bonding model,
choose an oxidation state, or validate the underlying electronic-structure
method. Missing NBO data returns `unavailable_data`, distinct from a parser or
runtime refusal.

## Heavy-element regression source

The live check uses Orbitron's owned standalone NBO7 UO₂ fixture,
`io/pipelines/tests/fixtures/nbo/uo2-test.nbo`, at Orbitron revision
`d913197b`. The source hash is
`f29c9a3275223c1fa28eed396a61d282a1333ef3dd259e4be3ca689dfa086311`.
It stays in Orbitron's fixture corpus because it is an NBO program output;
Chemtools records its hash and checks the fixed bridge boundary without
duplicating third-party program text.

At that revision, the fixture has 142 orbitals: six `BD`, six `BD*`, nine
`CR`, fifteen `LV`, and 106 `RY*`. The first bounded samples are the six U–O
bonding orbitals followed by six U–O antibonding orbitals.

Run the opt-in bridge check with the configured companion interpreter:

```bash
export CHEMTOOLS_SCIENCE_PYTHON=/path/to/chemtools-science/bin/python
.venv/bin/python scripts/check_orbitron_nbo_python_api.py \
  --fixture /path/to/orbitron/io/pipelines/tests/fixtures/nbo/uo2-test.nbo
```
