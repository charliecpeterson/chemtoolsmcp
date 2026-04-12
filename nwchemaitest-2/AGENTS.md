  For NWChem output analysis, follow-up preparation, and input drafting, use `chemtools` MCP tools as the primary workflow.
  For NWChem syntax, exact block forms, fragment-guess patterns, and examples, use `nwchem-docs` MCP tools before drafting unfamiliar input.

  Required order:
  1. `chemtools_review_nwchem_case`
  2. `chemtools_review_nwchem_progress` when the user asks how a running or incomplete job is going
  2a. `chemtools_watch_nwchem_run` when the user wants you to wait for completion or watch for failure without manual follow-up
  2b. `chemtools_review_nwchem_followup_outcome` after a follow-up run finishes and you need to decide whether the new run actually improved SCF behavior or state quality
  3. `chemtools_prepare_nwchem_next_step` if you want drafted follow-up artifacts
  3a. `chemtools_draft_nwchem_scf_stabilization_input` when a state-check/property-style restart still fails and you want a safer SCF retry artifact
  4. `chemtools_create_nwchem_dft_input_from_request` for standard new DFT inputs
  5. `chemtools_review_nwchem_input_request` only when you need to explain why a request is not ready
  6. `chemtools_create_nwchem_dft_workflow_input` only when you already have a fully explicit geometry+basis plan and need the lower-level deterministic creator directly
  7. `chemtools_diagnose_nwchem_output` if the job is questionable or electronically sensitive
  8. `chemtools_parse_nwchem_scf` only when SCF detail is needed
  9. Only then inspect raw output manually if needed
  10. For syntax-sensitive blocks such as `scf`, `mcscf`, `tce`, `vectors`, and fragment guesses:
    - `nwchem-docs_lookup_nwchem_block_syntax`
    - `nwchem-docs_search_nwchem_docs`
    - `nwchem-docs_find_nwchem_examples`
    - `nwchem-docs_read_nwchem_doc_excerpt`

  Do not treat the chemtools MCP tools as optional if they are available.
  Do not draft unfamiliar NWChem syntax from memory when the `nwchem-docs` MCP server is available.
  Do not rely primarily on grep/manual reading when a chemtools tool already provides structured output.
  For long or still-running jobs, do not repeatedly tail the output file.
  Use `chemtools_review_nwchem_progress` as the primary progress source, and include the input file when it is available so task progress is compared against the requested task list.
  If the user wants you to wait on a running local job, prefer `chemtools_watch_nwchem_run` over repeated manual status checks.
  `chemtools_watch_nwchem_run` already uses adaptive polling and compact history by default, so do not simulate extra manual polling in chat.
  Use the `intervention` field from `chemtools_review_nwchem_progress` before deciding to stop a live run.
  Only call `chemtools_terminate_nwchem_run` when the intervention assessment recommends killing the job or the user explicitly asks to stop it.
  Only call `chemtools_tail_nwchem_output` once if the status summary is ambiguous or the user explicitly asks for raw output details.
  If tailing is necessary, request a short tail only.
  The `chemtools` MCP server already has `CHEMTOOLS_RUNNER_PROFILES` configured for this project, so runner tools can omit an explicit profiles path unless a different profile file is required.
  If `CHEMTOOLS_RUNNER_PROFILES` is configured, do not probe `nwchem` on PATH, do not search Homebrew/system install locations, and do not guess how NWChem is launched.
  Use `chemtools_inspect_nwchem_runner_profiles` and `chemtools_prepare_nwchem_run` as the source of truth for local execution.
  Do not guess multiplicity for transition-metal or open-shell systems.
  For generated NWChem inputs, prefer explicit inlined basis and ECP blocks from the basis library rather than `library` shorthand.
  For `scf`, `mcscf`, `tce`, and fragment-guess inputs, verify the exact block syntax from `nwchem-docs` before writing or patching the file.
  If geometry is missing, ask for it explicitly by default.
  If the user explicitly asks for a guessed starting structure or says a rough/example geometry is acceptable, you may draft a provisional starting geometry.
  Any guessed geometry must be labeled clearly as heuristic, chemistry-based, and non-validated.
  For guessed geometries, state that optimization may fail or converge to an unintended structure and that a reference geometry is preferred.
  If geometry or spin state is missing, ask for it explicitly or state that you are drafting a provisional/example model.
  For formula-level or new-input requests, do not hand-write or patch NWChem input text when `chemtools_create_nwchem_dft_input_from_request` can be used.
  Treat the deterministic creator output as the source of truth for task syntax, basis/ECP rendering, default DFT settings, and movecs policy.
  After creating or substantially revising an input, validate it with `chemtools_lint_nwchem_input` before presenting it as final.
  Only hand-edit a generated NWChem input after the deterministic creator has been used and only when the creator cannot express the requested change directly.
  If the user explicitly allows a guessed starting geometry and no geometry-generation tool is available, you may draft a small provisional geometry yourself, then use the deterministic creator or lint step on the resulting input.
  If a property/state-check restart fails with SCF nonconvergence, prefer drafting a stabilization restart before falling back to manual review.
