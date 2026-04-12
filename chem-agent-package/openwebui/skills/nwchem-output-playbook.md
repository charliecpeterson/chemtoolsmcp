# nwchem-output-playbook

## Purpose
This skill adds NWChem-specific output-analysis behavior on top of `chemapps-core`, `chem-style`, and `nwchem-skill`.

## Required workflow
For any NWChem output analysis:
1. identify the final completed stage
2. identify the active failing stage
3. summarize the termination status
4. inspect convergence behavior if SCF/DFT is involved
5. inspect state quality if orbitals/occupations are available
6. recommend the smallest next change

## Stage classification
Classify the active issue as one of:
- input/parser problem
- SCF/DFT convergence problem
- wrong-state convergence
- optimization problem
- frequency interpretation problem
- restart/database problem
- runtime/resource problem

If more than one is plausible, rank them.

## SCF/DFT rules
- Do not treat `maxiter` as sufficient evidence that the fix is `maxiter`.
- Read the recent energy and density trend first.
- If the energy is stabilizing and the density error is slowly falling, a modest iteration increase may be reasonable.
- If the energy oscillates, stalls, or the orbital/state picture is wrong, recommend a strategy change instead of only more iterations.
- If the run converged to a chemically wrong state, label it as wrong-state convergence.

## Orbital and state rules
- If MO data is available, inspect occupancies and frontier orbitals before recommending state-sensitive fixes.
- For open-shell transition-metal cases, be explicit about whether the observed occupations match the intended qualitative picture.
- If the state looks wrong, recommend state-oriented fixes before numerical fine-tuning.

## Optimization rules
- Separate electronic-structure failure during an optimization from optimizer failure.
- If the optimization stopped only because the job was interrupted, prefer restart from the last useful structure.
- If an optimization ends in a stationary point with an unexpected imaginary mode, say whether the mode looks chemically meaningful or likely numerical.

## Frequency rules
- Do not call every imaginary frequency a failure.
- Distinguish:
  - intended transition-state-like mode
  - torsional/floppy small mode
  - clearly wrong minimum
- Explain what rerun or displaced-geometry test would confirm the interpretation.

## Restart rules
- Prefer restarting from the latest valid structure or vectors when the file evidence supports it.
- Do not recommend restart reuse blindly if the basis, state, or scientific target changed.

## Output format
Use headings when helpful:
- **Stage**
- **Evidence**
- **State Check**
- **Likely Cause**
- **Best Next Change**
- **After Rerun, Check**
