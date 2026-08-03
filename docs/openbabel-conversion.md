# Open Babel molecular conversion

`convert_molecule_with_openbabel` converts one declared SMILES string or MDL
mol block to the other supported form in the optional companion runtime. It
does not accept file paths, Open Babel flags, conversion plugins, output paths,
or coordinate-generation requests.

Each conversion has two checks. Open Babel produces the requested text, then
RDKit parses and sanitizes both the submitted and converted form. The response
reports canonical SMILES, formula, atom and bond counts, formal charge,
radical electrons, fragment count, aromatic atom count, stereocenter count,
and stereo-bond count on both sides. `comparison.status` is `matched` only
when every reported field agrees; otherwise it is `different` and names each
difference. An output that RDKit cannot inspect is refused as
`uninspectable_output` and is not returned as a converted artifact.

The converted text is evidence, not a validated calculation model. In
particular, Open Babel can write a MOL block from SMILES without generating
coordinates. Chemtools records that as `coordinate_status: not_generated`.
Those zero coordinates describe neither a conformer nor a geometry fit for an
electronic-structure calculation. Coordinate generation needs a separate
operation with its own method, seed, and provenance contract.

For example, a chiral SMILES may produce a MOL block that remains readable but
does not retain the same RDKit canonical isomeric SMILES. The conversion stays
available for inspection, while `comparison.status: different` prevents a
caller from treating the result as a silent round trip.

## Scope

The first slice supports only `smiles` and `molblock`, with a 128 KiB input and
output-text limit. XYZ, SDF, file conversion, and 2D or 3D coordinate
generation remain deferred. They need format-specific decisions about units,
conformers, stereochemistry, multi-record input, and artifact ownership.

## Reviewed conversion fixtures

[`tests/fixtures/openbabel/conversion_cases.json`](../tests/fixtures/openbabel/conversion_cases.json)
records seven small version-scoped conversions: neutral water, charged acetate,
aromatic benzene, disconnected sodium chloride, a methyl radical, chiral
fluoroethanol, and a supplied ethanol mol block. The corpus pins both RDKit
evidence records, warning codes, converted-text markers, and the fields that
must differ.

Run the external check only with the explicitly configured companion runtime:

```bash
export CHEMTOOLS_SCIENCE_PYTHON=/path/to/chemtools-science/bin/python
.venv/bin/python scripts/check_openbabel_fixture_corpus.py
```

The checker returns zero only when all seven cases agree with their recorded
outcomes. It does not pin MOL text hashes because Open Babel writes a timestamp
in its MOL title line.

## Installed-format evidence

The local Open Babel 3.1.0 `obabel -H smi` help describes SMILES as a linear
text format for connectivity and chirality. Its `obabel -H mol` help describes
the MDL MOL reader and writer. This tool fixes those two format names in the
runner rather than exposing Open Babel's general format and option surface.
