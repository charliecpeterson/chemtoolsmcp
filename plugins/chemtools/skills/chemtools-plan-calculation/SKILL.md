---
name: chemtools-plan-calculation
description: Plan a computational chemistry calculation before input syntax is written. Use when the user wants ordered stages, assumptions, required scientific decisions, or a strategy for a new calculation rather than an input file or immediate launch.
---

# Plan a chemistry calculation

Use the Chemtools `plan_calculation` MCP tool. The operation is read-only and
does not inspect files, render program syntax, or launch a job.

1. Establish the required specification: program, concise system label,
   distinct elements, charge, multiplicity, and ordered stages.
2. Ask for missing charge, multiplicity, or stage intent instead of inventing
   it. Group closely related missing facts into one concise question.
3. Include method, functional, basis, ECP, relativistic treatment, geometry
   source and units, solvent, or state strategy only when the user supplied or
   explicitly settled them.
4. Call `plan_calculation` and report the ordered stage dependencies,
   decisions already resolved, remaining decisions, assumptions, uncertainty,
   and next action.
5. Keep energy ordering, electronic-state assignment, and method suitability
   conditional where the returned evidence does.

NWChem currently provides the calculation-planning backend. For another
program, return the tool's unsupported-capability result and do not fabricate
a generic plan under that program's name.

Call `draft_input` only after the required decisions are settled and the user
asks for native input text. Call `launch_run` only for an existing reviewed
input and an explicit execution request.
