# NWChem Agent Tool Precedence

Use `chemtools` MCP tools as the primary workflow. Use `nwchem-docs` for syntax reference.
Do not read raw NWChem output manually when a chemtools tool already provides structured output.

## Required Tool Order

### 1. Case review (always start here for any finished or failed run)
`chemtools_analyze_nwchem_case`
- Use `detail='compact'` for routine triage (default)
- Use `detail='full'` for the complete human-readable report
- Covers: diagnosis, lint, restart assets, spin-state check, and next-step planning in one call

### 2. Progress check (for running or incomplete jobs)
`chemtools_get_nwchem_run_status`
- Include both `output_file` and `input_file` so task progress is compared against the requested task list
- Check the `progress` field before deciding to intervene or stop a run

**2a.** `chemtools_watch_nwchem_run` — when the user wants you to wait for completion without manual polling  
**2b.** `chemtools_compare_nwchem_runs` — after a follow-up run finishes, to assess whether the new run actually improved SCF behavior or state quality. Pass `output_dir`/`base_name` to also write next-step artifacts.

### 3. Recovery strategy (when the run failed or the electronic state is suspicious)
`chemtools_suggest_nwchem_recovery`
- `mode='state'` — when the main issue is suspicious occupations, wrong SOMOs, or an electronically wrong state
- `mode='scf'` — when the main issue is SCF/DFT nonconvergence or unstable convergence behavior
- `mode='auto'` — when unsure (returns both sets of strategies)

### 4. MCSCF planning (when MCSCF is needed)
**4a.** `chemtools_suggest_nwchem_mcscf_active_space` — recommend which orbitals belong in the active space  
**4b.** `chemtools_review_nwchem_mcscf_case` — when an MCSCF run already exists and you need to judge whether the active space and convergence were acceptable  
**4c.** `chemtools_draft_nwchem_mcscf_input` — build a concrete MCSCF input from the recommended active space  
**4d.** `chemtools_draft_nwchem_mcscf_retry_input` — deterministic retry when an MCSCF run failed or converged stiffly  
**4e.** `chemtools_review_nwchem_mcscf_followup_outcome` — after MCSCF retry finishes, assess whether it improved convergence or active-space quality

### 5. Next-step artifact generation
`chemtools_prepare_nwchem_next_step` — if drafted follow-up artifacts are needed (swap restart, stabilization, etc.)  
`chemtools_draft_nwchem_scf_stabilization_input` — when a state-check restart still fails electronically

### 6. New inputs
`chemtools_create_nwchem_dft_workflow_input` — standard new DFT inputs (optimize, freq, single-point)  
`chemtools_create_nwchem_input` — for non-DFT modules or when you have a fully explicit geometry+basis plan  
Always follow with `chemtools_lint_nwchem_input` before presenting the input as final.

### 7. SCF detail (only when SCF iteration behavior is specifically needed)
`chemtools_parse_nwchem_scf`

### 8. Syntax reference (for unfamiliar blocks)
`nwchem-docs_get_nwchem_topic_guide` — for `scf_open_shell`, `mcscf`, `fragment_guess`, `tce`  
`nwchem-docs_lookup_nwchem_block_syntax`  
`nwchem-docs_search_nwchem_docs`  
`nwchem-docs_find_nwchem_examples`  
`nwchem-docs_read_nwchem_doc_excerpt`

### 9. Last resort
Only inspect raw output manually (tail, read) if no chemtools tool covers the specific information needed. If this happens, it's a gap to fix in the MCP.

---

## TCE-Specific Rules

For TCE (MP2, CCSD, CCSD(T)) workflows, orbital ordering must be verified before freezing:

1. `chemtools_parse_nwchem_movecs` — inspect binary movecs after SCF converges
2. Check ordering: metal 3s/3p should be lower in energy (lower MO index) than ligand 1s
3. If ordering is wrong: `chemtools_swap_nwchem_movecs` to patch the binary file directly
4. Verify swap with another `chemtools_parse_nwchem_movecs`
5. `chemtools_draft_nwchem_tce_input` — always emits explicit `freeze N`, warns if ordering is still wrong
6. After TCE run: `chemtools_parse_nwchem_tce_output`

**Never use `freeze atomic`.** **Never use `vectors swap` in the SCF block** — NWChem re-diagonalizes to canonical order on every SCF restart; only binary file patching (`swap_nwchem_movecs`) survives a restart.

---

## Execution Rules

- Default runner profile: `workstation_mpi` (15 MPI ranks, Apptainer container)
- To launch: `chemtools_launch_nwchem_run` with `profile='workstation_mpi'`
- To preview without running: add `dry_run=true`
- Do not probe `nwchem` on PATH, do not search system locations, do not guess the launch command
- Use `chemtools_inspect_nwchem_runner_profiles` if unsure which profiles are available

---

## Hard Rules

- Do not treat chemtools tools as optional when they are available
- Do not draft unfamiliar NWChem syntax from memory when `nwchem-docs` is available
- Do not rely on grep/manual reading when a chemtools tool already provides structured output
- Do not repeatedly tail long output files for routine status checks
- Do not guess multiplicity for transition-metal or open-shell systems
- Do not use `library` shorthand for basis sets — always use explicit inlined blocks from the basis library
- Do not hand-edit a generated NWChem input without first using the deterministic creator
- Do not assume a later task started unless the output clearly shows it
- If geometry is missing: ask for it. If the user accepts a guessed structure, label it clearly as heuristic and non-validated
- After any input is created or substantially revised: lint it with `chemtools_lint_nwchem_input`
