"""NWChem input file drafters and renderers.

Will hold the family-split drafter modules (scf.py, dft.py, tce.py, mcscf.py,
freq.py, property.py, ...) once api_input.py is broken up. For now contains
basis.py (basis/ECP block renderers, lifted from the legacy api_basis.py).

The eventual Drafter sub-protocol on chemtools.programs.nwchem.NWCHEM will be
assembled from these modules.
"""
