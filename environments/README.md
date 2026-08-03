# Optional environments

`chemtools-science.yml` defines the companion interpreter used by fixed
Chemtools science operations. It keeps PySCF, RDKit, Open Babel, Basis Set
Exchange, and HDF5 inspection support out of the MCP server environment.
Install Orbitron's Python bridge explicitly after creating the environment so
its active-development version is deliberate.

Create it explicitly with Conda or micromamba, then point Chemtools at the
resulting interpreter:

```bash
micromamba create -f environments/chemtools-science.yml
export CHEMTOOLS_SCIENCE_PYTHON=/path/to/envs/chemtools-science/bin/python
```

Call `inspect_science_runtime` after setup. It verifies imports and reports
versions, but does not install packages or run a calculation. BSE is used by
`render_basis_set_with_bse` to render the environment's bundled BSE data into
an explicit program-format block; it does not clone or edit the upstream BSE
repository.

`fetch_nist_atomic_reference` does not use this environment. It queries only
the fixed NIST ASD energy-level and ionization-energy endpoints, then stores
the exact tab-delimited response, source URL, retrieval time, and SHA-256 in
`~/.chemtools/nist-asd` by default. Set `CHEMTOOLS_NIST_ASD_CACHE` to move that
local cache. It is a query cache, not a distributed ASD mirror.

For a local Orbitron checkout, install its Python bridge from
`extensions/python-bridge` into this environment. The bridge is native code,
so build it with the checkout's supported Rust toolchain. The verified linux-64
Conda resolution is committed as
[`chemtools-science-linux-64.explicit.txt`](../chemtools/data/science/chemtools-science-linux-64.explicit.txt).
It is package data so the science runner can hash it in every result. The
Orbitron bridge is installed separately, so each result also records its
version and the SHA-256 of its native extension when available.
