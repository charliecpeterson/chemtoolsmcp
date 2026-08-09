# Curated knowledge cards

Each YAML file in `cards/` contains one scoped claim and its curation
metadata. The runtime loader reads this directory only. It never scans
`notes/`, which remain working records and source material.

The filename must match the card ID. Cards that still lack evidence across
their claimed scope stay `draft`; `accepted` cards must cite at least one
source and one regression test. The broad silent-success lesson remains a
draft, while its tested RMCDHF instance is accepted only for GRASP. The
direct-producer card is program-neutral but states its narrow provenance scope;
it does not claim full method independence. Starting-guess diversity also
depends on an explicit canonical class in recorded provenance. Chemtools does
not infer that class from a program name. The numeric invariant card is also a
draft: its generic sign and monotonicity checks have synthetic contract cases,
but accepted chemistry rules still need scoped reference values. Optimizer
sentinels have an accepted program-neutral check, but only when the valid
objective bounds are scientifically scoped and all values are finite. A
separate failure channel is required when no portable finite sentinel exists.
The accepted QMCPACK cards are deliberately narrow: they cover the
determinant-only VMC comparison, the post-Jastrow VMC energy gate, the
authoritative `vp.h5` parameter sidecar, and the tested heavy-open-f-ion DMC
reference protocol. They are reference checks for the stated workflow, not
general convergence claims for every QMCPACK calculation.
The accepted PySCF cards are also narrowly bounded to the companion molecular
single-point runner. They distinguish normal process completion from SCF
convergence and an electron-count/spin runtime refusal from both invalid
request syntax and an unconverged SCF. They do not endorse a chemical model or
electronic-state assignment.
Three NWChem policies migrated from the stale `chem-agent-package/` skill
prose remain drafts. They cover SCF trend classification, scoped imaginary-mode
interpretation, and explicit element-by-element basis coverage. Existing code
and tests support narrower pieces of these claims, but not their complete
scope, so they are excluded from default recommendations.
The `search_knowledge` MCP tool defaults to `accepted`. Other curation
states require an explicit status, and every result repeats its status, scope,
sources, checks, tests, and recommendation eligibility.
