---
name: chemtools-review-input
description: Review an existing NWChem, OpenMolcas, DIRAC, GRASP, Quantum ESPRESSO, or QMCPACK input before execution. Use when the user asks whether an input is valid, internally consistent, or ready to run without rewriting or launching it.
---

# Review a chemistry input

Use the Chemtools `review_input` MCP tool for the scientific review. Keep the
workflow read-only.

1. Require one caller-supplied input path. Do not scan a directory for likely
   inputs.
2. Pass `program` only when the user identifies it or automatic detection
   reports ambiguity. Do not override a positive content match.
3. Call `review_input` once with the exact path.
4. Report the detected program, assessment label and reasons, relevant parsed
   evidence, every uncertainty entry, and the highest-priority next actions.
5. Describe a clean assessment as passing the checks that backend implements.
   Do not claim complete program-syntax or scientific validation.

Do not draft replacement text, write a file, or start a calculation unless the
user separately requests that action. Use `draft_input` only when the user
wants a new input from a complete molecular specification. Use
`chemtools-inspect-run` when the primary artifact is output from a run.

If the tool rejects the file as unsupported or ambiguous, preserve that result
and ask for the smallest missing fact, usually the program or correct primary
input path. Do not guess from the filename alone.
