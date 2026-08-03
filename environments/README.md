# Optional environments

`chemtools-science.yml` defines the companion interpreter used by fixed
Chemtools science operations. It keeps PySCF, RDKit, Open Babel, and HDF5
inspection support out of the MCP server environment. Install Orbitron's Python bridge explicitly after
creating the environment so its active-development version is deliberate.

Create it explicitly with Conda or micromamba, then point Chemtools at the
resulting interpreter:

```bash
micromamba create -f environments/chemtools-science.yml
export CHEMTOOLS_SCIENCE_PYTHON=/path/to/envs/chemtools-science/bin/python
```

Call `inspect_science_runtime` after setup. It verifies imports and reports
versions, but does not install packages or run a calculation.

For a local Orbitron checkout, install its Python bridge from
`extensions/python-bridge` into this environment. The bridge is native code,
so build it with the checkout's supported Rust toolchain. The verified linux-64
Conda resolution is committed as
[`chemtools-science-linux-64.explicit.txt`](../chemtools/data/science/chemtools-science-linux-64.explicit.txt).
It is package data so the science runner can hash it in every result. The
Orbitron bridge is installed separately, so each result also records its
version and the SHA-256 of its native extension when available.
