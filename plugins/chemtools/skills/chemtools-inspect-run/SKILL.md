---
name: chemtools-inspect-run
description: Inspect completed, failed, or partial output from NWChem, OpenMolcas, DIRAC, GRASP, Quantum ESPRESSO, QMCPACK, or ORCA. Use when the user asks whether a run is usable, why it failed, what the output proves, or what evidence supports a next step.
---

# Inspect a chemistry run

Use the Chemtools `inspect_run` MCP tool to establish the scientific evidence
before recommending another calculation.

1. Require one primary output path. Do not scan its directory.
2. Pass only related artifacts the user identified, such as the matching input,
   stderr, checkpoint, or orbital file. Preserve their order.
3. Pass `program` only when supplied by the user or needed to resolve an
   ambiguity. Do not force a program that conflicts with a positive detector.
4. Call `inspect_run` and separate the response into execution or convergence
   verdict, scientific evidence, warnings, uncertainty, and next actions.
5. Distinguish facts parsed from the output from conclusions that remain
   conditional on missing input, geometry, method, basis, or state evidence.

Use `compare_runs` when the question depends on two outputs. Use
`plan_recovery` only for NWChem after the inspection establishes a failure or
instability and the matching input is available. Recovery planning returns
candidate text for review; it does not write or launch anything.

If the user supplies only a PID, scheduler ID, or launch ID, do not treat that
identifier as an output. Use `chemtools-monitor-run` for a Chemtools-owned
launch ID, or ask for the primary output path.
