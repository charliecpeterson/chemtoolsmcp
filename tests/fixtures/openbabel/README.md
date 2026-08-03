# Open Babel conversion fixtures

This corpus records fixed SMILES and MDL mol-block conversions through the
bounded Open Babel companion operation. Each case pins the submitted text and
the independent RDKit evidence expected before and after conversion. It does
not pin Open Babel's emitted mol-block bytes because its title line includes a
run timestamp.

The cases cover neutral, charged, aromatic, disconnected, radical, and chiral
SMILES, plus a mol-block-to-SMILES conversion. The chiral case is intentionally
different: Open Babel 3.1.0 writes a readable 0D mol block, but RDKit does not
recover its original canonical isomeric SMILES. That difference must remain
reported rather than be repaired or reclassified as a successful round trip.

The corpus was recorded with Open Babel 3.1.0 and RDKit 2025.09.5 in the
linux-4090 `chemtools-science` environment. It is version-scoped regression
evidence, not a general interchange-validation suite.
