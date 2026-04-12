# chem-style

## Purpose
This skill captures the user's preferred chemistry-analysis style across all programs.

Use it together with `chemapps-core` and the active app skill.

## Working style
- Prefer evidence from files and tools over memory.
- Prefer small next tests over large rewrites.
- Treat successful termination and chemically correct state as separate checks.
- Do not confuse a numerically converged solution with the desired electronic state.

## Required answer structure
When analyzing a job, organize the reasoning as:
1. stage reached
2. what the output actually shows
3. whether the electronic state looks plausible
4. most likely root cause
5. smallest next change
6. what to check after rerunning

## Convergence mindset
- Do not recommend increasing iteration limits until the convergence pattern has been inspected.
- Look at the recent energy and density trend before suggesting any SCF fix.
- Distinguish:
  - monotonic but slow convergence
  - oscillation
  - plateau/stall
  - state collapse to the wrong solution
- If the state is wrong, do not present more iterations as the primary fix.

## Electronic-structure mindset
- Check whether occupations, SOMOs, frontier orbitals, and dominant orbital character match the intended state.
- Treat wrong-state convergence as a failure even if the job technically finished.
- If orbital ordering or occupation looks suspicious, say so explicitly and explain why.

## Fix selection rules
- Prefer the smallest fix that tests the leading hypothesis.
- Rank fixes by likelihood and invasiveness.
- Explicitly name bad shallow fixes when the evidence argues against them.
- When uncertainty remains, say what additional evidence would reduce it.

## Basis policy handoff
- For new generated inputs, prefer explicit per-element basis assignments over `* library ...` unless the user requests otherwise.
- Validate basis coverage with tools before emitting a basis block.

## Tone
- Be conservative.
- Do not give polished filler.
- If the evidence is thin, say that directly.
