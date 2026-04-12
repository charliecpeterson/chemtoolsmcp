# nwchem-basis-policy

## Purpose
This skill defines how the NWChem agent should choose and emit basis information for new or revised inputs.

## Policy
- Prefer explicit per-element basis assignments for newly generated inputs.
- Do not rely on `* library ...` by default for mixed-element systems unless the user requests a minimal shorthand input.
- Validate basis availability and element coverage with a tool before writing the block.

## Default generation rule
When drafting a new NWChem input:
1. inspect the geometry or atom list
2. identify the unique elements
3. resolve the requested basis against the configured local basis library
4. emit a basis block with one line per element

Preferred form:

```text
basis "ao basis" spherical
  Fe library def2-svp
  C  library def2-svp
  H  library def2-svp
end
```

## Safety rules
- Do not invent basis names.
- Do not assume a basis covers all atoms without checking.
- If heavy elements may need ECP treatment, say so explicitly rather than guessing.
- Do not invent `cd basis`, `xc basis`, or auxiliary basis choices unless documentation or user policy supports them.

## Output-analysis tie-in
- When reviewing an input or failed run, check whether basis assignment is incomplete, mixed unexpectedly, or inconsistent with the target method.
- If the user is doing basis stepping or projecting from a smaller basis, preserve that workflow explicitly.
