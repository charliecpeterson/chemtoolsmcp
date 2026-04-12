# NWChem Agent Workspace

This directory is the active test workspace for the chemtoolsmcp NWChem AI agent.
The `chemtools` MCP server handles all parsing, input generation, and run control.
The `nwchem-docs` MCP server handles NWChem syntax reference and examples.

Follow @AGENTS.md for the detailed workflow and tool precedence.

## How NWChem Runs Here

- **NWChem version:** 7.2.2 inside an Apptainer container
- **Container:** `~/mycontainers/nwchem_7.2.2.sif`
- **Launch command:** `mpirun -np 15 apptainer exec ~/mycontainers/nwchem_7.2.2.sif nwchem`
- **Default profile:** `workstation_mpi` (15 MPI ranks, direct local launch)
- **Profiles config:** `runner_profiles.local.json` in this directory (set via `CHEMTOOLS_RUNNER_PROFILES`)
- Do not probe `nwchem` on PATH — all execution goes through the runner profiles

## What This Workspace Is

This is Charlie's iterative MCP development test environment. After each test run, the chemtools MCP tools are improved based on what worked and what didn't. If a tool gives a wrong answer, a confusing result, or is missing a case entirely, that's a signal to improve the tool — not to work around it manually.

The goal: the agent should never need to read raw NWChem output directly. If you find yourself trying to parse output manually, that's a gap in the tools that should be fixed.

## Claude Code Rules

- Use `chemtools` MCP tools as the primary source of truth for local files and workflows. Do not read NWChem output files directly when a tool already provides structured output.
- Use `nwchem-docs` for syntax, block options, examples, and documented NWChem behavior.
- For syntax-sensitive blocks (`scf`, `dft`, `mcscf`, `tce`, `vectors`, fragment-guess), consult `nwchem-docs` before drafting.
- Prefer `nwchem-docs_get_nwchem_topic_guide` first for `scf_open_shell`, `mcscf`, `fragment_guess`, and `tce`.
- Do not invent NWChem syntax from memory.
- Do not guess multiplicity for transition-metal or open-shell systems — always ask or derive from the output.
- For new DFT inputs: use `chemtools_create_nwchem_dft_workflow_input` or `chemtools_create_nwchem_input`.
- After creating or substantially revising an input: run `chemtools_lint_nwchem_input`.
- Prefer explicit inlined basis and ECP blocks over `library` shorthand.
- Prefer explicit `task <module> <operation>` syntax.
- For progress questions: use `chemtools_get_nwchem_run_status` — do not repeatedly tail output.
- To wait on a running job: use `chemtools_watch_nwchem_run` — do not manually poll.
- Before deciding to stop a live run: check the `progress` field from `chemtools_get_nwchem_run_status`.
- Only call `chemtools_terminate_nwchem_run` when the run clearly needs to stop or the user asks.
- After a follow-up run finishes: use `chemtools_compare_nwchem_runs` before deciding if it improved things.
- For SCF nonconvergence: use `chemtools_suggest_nwchem_recovery` (mode='scf') before inventing a retry plan.
- For suspicious electronic states: use `chemtools_suggest_nwchem_recovery` (mode='state') before inventing swaps or fragment guesses.
- For MCSCF planning: use `chemtools_suggest_nwchem_mcscf_active_space` before choosing an active space from memory.
- For a finished or failed MCSCF run: use `chemtools_review_nwchem_mcscf_case` before assuming the active space was good.
- When you need an MCSCF input: use `chemtools_draft_nwchem_mcscf_input` after reviewing the active-space suggestion.
- When retrying a failed MCSCF: use `chemtools_draft_nwchem_mcscf_retry_input` instead of hand-editing.
- After an MCSCF retry finishes: use `chemtools_review_nwchem_mcscf_followup_outcome`.
- For TCE freeze count: always inspect orbital ordering with `chemtools_parse_nwchem_movecs` or `chemtools_parse_nwchem_mos` before accepting the suggested count. Never use `freeze atomic`.
- If orbital ordering is wrong for TCE: use `chemtools_swap_nwchem_movecs` to fix the binary file before restarting. Do not use `vectors swap` in the SCF block — NWChem re-diagonalizes to canonical order.
- If geometry is missing: ask for it explicitly. If the user accepts a guessed structure, label it as heuristic and non-validated.

## Typical Workflows

### New input
1. `chemtools_create_nwchem_dft_workflow_input` (or `chemtools_create_nwchem_input` for non-DFT)
2. `chemtools_lint_nwchem_input`
3. `chemtools_launch_nwchem_run` with profile `workstation_mpi` (add `dry_run=true` to preview first)

### Check job progress
1. `chemtools_get_nwchem_run_status` (include both input and output files)
2. `chemtools_watch_nwchem_run` if staying with the job until done

### Diagnose a finished or failed run
1. `chemtools_analyze_nwchem_case` (default `detail='compact'` for routine triage; `detail='full'` for the complete report)
2. `chemtools_suggest_nwchem_recovery` if recovery is needed
3. `chemtools_prepare_nwchem_next_step` or the specific draft tools for the next artifact
4. `chemtools_compare_nwchem_runs` after the follow-up finishes

### TCE correlated wavefunction
1. Run SCF first, get a converged movecs file
2. `chemtools_parse_nwchem_movecs` — inspect orbital ordering
3. If ordering is wrong: `chemtools_swap_nwchem_movecs`, verify with `chemtools_parse_nwchem_movecs` again
4. `chemtools_draft_nwchem_tce_input` — generates explicit `freeze N`, warns about ordering issues
5. `chemtools_launch_nwchem_run`
6. `chemtools_parse_nwchem_tce_output` — read the result

### MCSCF
1. `chemtools_analyze_nwchem_case` on the DFT reference
2. `chemtools_suggest_nwchem_mcscf_active_space`
3. `nwchem-docs_get_nwchem_topic_guide` for `mcscf` syntax
4. `chemtools_draft_nwchem_mcscf_input`
5. After run: `chemtools_review_nwchem_mcscf_case`
6. If retry needed: `chemtools_draft_nwchem_mcscf_retry_input`, then `chemtools_review_nwchem_mcscf_followup_outcome`

### NWChem syntax lookup
- `nwchem-docs_get_nwchem_topic_guide`
- `nwchem-docs_lookup_nwchem_block_syntax`
- `nwchem-docs_search_nwchem_docs`
- `nwchem-docs_find_nwchem_examples`
- `nwchem-docs_read_nwchem_doc_excerpt`

## Notes

- Treat `chemtools` tool output as the source of truth for local file facts, task state, and run configuration.
- Treat `nwchem-docs` as the source of truth for NWChem syntax.
- If a tool gives a wrong or incomplete answer, that's a gap to fix in the MCP — not something to work around by reading output manually.
