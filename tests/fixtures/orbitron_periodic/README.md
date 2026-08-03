# Orbitron periodic Python-API fixture

This small synthetic VASP `vasprun.xml` contains two silicon atoms, a two-point
band path, and a three-point total density of states. It was copied from
Orbitron's owned Python-bridge test fixture at revision `34aa7c31` so Chemtools
can test its fixed periodic-summary boundary without an Orbitron checkout.

The fixture is parser-contract evidence. Its 7.0 eV gap and 1.2 eV Fermi level
are deliberately synthetic and must not be treated as silicon reference data.
