# NWChem-to-PySCF matched-reference fixtures

This corpus records small NWChem 7.2.2 calculations run through the local
Apptainer image on linux-4090. `cases.json` pins each input/output hash,
declared PySCF match settings, and selected SCF evidence. The paired tests exercise NWChem evidence
extraction and the matched-run refusal boundary; they do not treat an energy
agreement with PySCF as a correctness verdict.

Each input states charge, multiplicity, a Cartesian geometry with explicit
angstrom units, and one `sto-3g` library basis. The three converged cases cover
closed-shell H2, closed-shell water, and unrestricted triplet O2. The
water `maxiter 1` calculation records an intentional NWChem SCF failure so
the matched workflow can prove it does not launch PySCF from incomplete
reference evidence.

The outputs are fixed regression artifacts. Re-record them only with a noted
NWChem/container-version change and after reviewing the scientific outcome.
