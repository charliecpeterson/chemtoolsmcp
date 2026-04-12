# NWChem Project Memory

This project is for NWChem workflow support in Claude Code.

The intended split is:
- local file, parsing, input-generation, and run-control actions come from the local `chemtools` MCP server
- local NWChem syntax/examples/documentation lookups come from the local `nwchem-docs` MCP server over `/Users/charlie/test/mytest/nwchem-docs`

Follow @AGENTS.md for the detailed NWChem workflow and tool precedence.

## Claude Code Rules

- Use `chemtools` MCP tools as the primary source of truth for local NWChem files and workflows.
- Use the `nwchem-docs` MCP tools for NWChem syntax, options, examples, and documented behavior.
- For syntax-sensitive blocks such as `scf`, `dft`, `mcscf`, `tce`, `vectors`, and fragment-guess workflows, consult `nwchem-docs` before drafting or patching input text.
- Do not invent NWChem syntax, keywords, or options.
- Do not guess multiplicity for transition-metal or open-shell systems.
- For standard new DFT inputs, prefer `chemtools_create_nwchem_dft_input_from_request`.
- After creating or substantially revising an input, run `chemtools_lint_nwchem_input` before presenting it as final.
- Prefer explicit inlined basis and ECP blocks from the basis library rather than `library` shorthand.
- Prefer explicit `task <module> <operation>` syntax.
- Distinguish SCF/DFT convergence from geometry optimization convergence.
- Do not assume a later task started unless the output clearly shows it.
- For progress questions, use `chemtools_review_nwchem_progress` first and include both input and output files when available.
- If the user wants you to wait on a running local job, use `chemtools_watch_nwchem_run` instead of repeatedly polling manually.
- `chemtools_watch_nwchem_run` already uses adaptive polling and compact history by default, so do not add extra manual polling logic in chat.
- Use the `intervention` field from `chemtools_review_nwchem_progress` before deciding to stop a live run.
- Only use `chemtools_terminate_nwchem_run` when the intervention explicitly recommends killing the run or the user asks to stop it.
- After a follow-up run finishes, prefer `chemtools_review_nwchem_followup_outcome` before manually deciding whether the new run is actually better.
- Do not repeatedly tail long outputs for routine status checks.
- If a state-check/property-style restart fails with SCF nonconvergence, prefer a stabilization restart artifact before stopping at manual review.
- If `CHEMTOOLS_RUNNER_PROFILES` is configured, do not probe `nwchem` on PATH and do not guess the local launch command.
- Ask for geometry by default when it is missing.
- If the user explicitly accepts a guessed/example starting structure, you may draft a provisional geometry, but you must label it as heuristic and non-validated.
- For provisional geometries, state that the structure is only a starting guess and that a reference geometry is preferred.

## Typical Workflow

1. For a new NWChem input request:
   - `chemtools_create_nwchem_dft_input_from_request`
   - `chemtools_lint_nwchem_input`
   - if geometry is missing and the user explicitly accepts a guessed structure, draft a provisional geometry and then validate the resulting input
2. For "how is this job going?":
   - `chemtools_review_nwchem_progress`
   - `chemtools_watch_nwchem_run` if the user wants you to stay with the job until it finishes or fails
   - `chemtools_review_nwchem_followup_outcome` after a follow-up run completes and you need to assess whether it improved the case
3. For case diagnosis:
   - `chemtools_review_nwchem_case`
   - `chemtools_prepare_nwchem_next_step` if drafted follow-up artifacts are needed
   - `chemtools_draft_nwchem_scf_stabilization_input` if the current restart still fails electronically
4. For local execution:
   - `chemtools_inspect_nwchem_runner_profiles`
   - `chemtools_prepare_nwchem_run`
   - `chemtools_launch_nwchem_run`
5. For NWChem syntax or example lookup:
   - `nwchem-docs_lookup_nwchem_block_syntax`
   - `nwchem-docs_search_nwchem_docs`
   - `nwchem-docs_find_nwchem_examples`
   - `nwchem-docs_read_nwchem_doc_excerpt`

## Notes

- Treat deterministic tool output as the source of truth for local file facts, task state, basis/ECP rendering, and run configuration.
- Treat `nwchem-docs` as the source of truth for documented NWChem syntax when the local tools do not already encode the behavior.
