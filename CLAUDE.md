# chemtoolsmcp — MCP Development Workspace

This is the source repository for the chemtoolsmcp computational-chemistry
AI agent toolkit. Currently supports four QC programs (NWChem, OpenMolcas,
DIRAC, GRASP2018) plus a program-agnostic generic dispatch layer.

Work here is about **developing and improving the MCP** — adding tools,
fixing parsers, updating logic.

## Architecture

```
chemtools/
  core/                          Program-agnostic infrastructure
    registry.py                  QC-program plugin registry (auto-detect from output)
    program.py                   Parser / Drafter / Strategist / BinaryReader protocols
    runner.py                    Generic SLURM/PBS submit + render_*_run + watch + status
    run_registry.py              SQLite registry (runs, campaigns, workflows)
    eval.py                      Multi-program case evaluator (NWChem + Molcas + DIRAC + GRASP)
    types.py                     ParsedRun / TaskSummary / GeometryAtom typed dicts
    common.py, cube.py, ...      Shared utilities (program detection, cube parsing, etc.)
  programs/
    nwchem/                      NWChem plugin (parser, drafter, strategist, runner)
      parse/, input/, strategy/, binary/, runner.py, docs.py, output.py, ...
    molcas/                      OpenMolcas plugin (CASSCF / CASPT2 / RASSI / SLAPAF)
      parse/, input/, strategy/, binary/, runtime.py, docs.py, scheduler.py
    dirac/                       DIRAC plugin (4c / X2C / AOC / KPSELE / Cm-class)
      parse/, input/, strategy/, binary/, runtime.py, docs.py, basis.py, scheduler.py
    grasp/                       GRASP2018 plugin (multi-exe workflows + scheduler)
      parse/, input/, strategy/, runtime.py, docs.py, scheduler.py, _plugin_parser.py
  data/                          Bundled per-program data (basis libraries + docs)
    nwchem/basis_library/        608 files
    nwchem/docs/                 29 files
    molcas/basis_library/        91 files
    molcas/docs/                 133 files
    dirac/docs/                  179 files
    grasp/docs/                  15 files
  mcp/
    cli.py                       Entry point — main() / serve() / arg parsing
    server.py                    JSON-RPC transport (read_message / write_message)
    dispatch.py                  tool_definitions() aggregator + dispatch_tool + handle_request
    decorator.py                 @_tool decorator + shared registries (_TOOL_REGISTRY etc.)
    modes.py                     Mode + program-filter logic (filter_tools, resolve_mode)
    nwchem.py                    Back-compat shim re-exporting from cli + decorator
    tools/                       Per-program MCP tool modules
      generic.py                 36 program-agnostic tools (parse_output, register_run, ...)
      nwchem.py                  97 NWChem tools
      molcas.py                  45 Molcas tools
      dirac.py                   38 DIRAC tools
      grasp.py                   37 GRASP tools

test_phase1/                     Test suite (gitignored — local training corpus)
  test_modes.py                  Capability filter + mode tests (65)
  test_phase2-6.py               NWChem feature suites (52 + 70 + 66 + 25 + 40)
  test_{molcas,dirac,grasp}_parsers.py  Per-program parser tests (42 + 39 + 39)
examples/
  local_workstation/             Example local runner_profiles.yaml (apptainer containers)
  tacc_stampede3/                Example TACC scheduler profiles (all 4 programs)
  molcas/, dirac/, grasp/        Eval case bundles (case.json + outputs, gitignored)
```

## MCP Tool Architecture

- Domain logic lives in `chemtools/programs/<program>/` and `chemtools/core/`
- MCP handlers in `chemtools/mcp/tools/<program>.py` — one `@_tool(name)` decorated function per tool
- Tool naming convention: `verb_<program>_noun` where verb ∈ {parse, analyze, draft, create, suggest, launch, get, watch, inspect, lint, find, compare, review, render, swap, register, update, list, advance, generate, detect, estimate, compute, run, plan, apply, terminate, summarize, validate, check, extract, refine, prepare, evaluate, displace, init, append, search, lookup, read, basis, append, try}
- **Current tool count: 255** (99 NWChem + 45 Molcas + 38 DIRAC + 37 GRASP + 36 generic). Generics auto-dispatch via `registry.resolve()` and serve any program. The active mode + `--programs` + `--toolset` filters determine how many are actually exposed in a session (e.g. `--programs nwchem --toolset triage` → 12).
- Tools are tagged with a capability (`needs=`) on the `@_tool` decorator; the active server mode filters which tools are exposed. See **Server modes** below.

### NWChem tool categories (97 tools)

| Category | Count | Examples |
|----------|-------|---------|
| Input drafting | 17 | `create_nwchem_input`, `create_nwchem_dft_input_from_request`, `draft_nwchem_tce_input` |
| Output parsing | 16 | `parse_nwchem_output`, `parse_nwchem_tce_output`, `parse_nwchem_thermochem`, `parse_nwchem_freq_progress`, `parse_nwchem_tasks`, `parse_nwchem_trajectory` |
| Analysis & diagnosis | 11 | `analyze_nwchem_case`, `check_nwchem_spin_charge_state`, `summarize_nwchem_output`, `preflight_check`, `review_nwchem_progress` |
| Strategy & suggestions | 12 | `suggest_basis_set`, `suggest_nwchem_recovery`, `suggest_spin_state`, `suggest_nwchem_resources`, `suggest_resources` |
| Resource & HPC | 6 | `suggest_nwchem_resources`, `suggest_nwchem_partition`, `detect_nwchem_hpc_accounts`, `check_nwchem_memory_fit`, `estimate_nwchem_freq_walltime`, `render_job_script` |
| Basis & ECP | 4 | `render_nwchem_basis_setup`, `basis_library_summary` |
| Job management | 7 | `launch_nwchem_run`, `watch_nwchem_run`, `watch_multiple_runs`, `terminate_nwchem_run`, `get_nwchem_run_status`, `tail_nwchem_output` |
| Registry & campaigns | 9 | `register_nwchem_run`, `create_nwchem_campaign`, `get_nwchem_campaign_energies`, `generate_nwchem_input_batch` |
| Workflow & protocols | 7 | `plan_nwchem_calculation`, `plan_nwchem_workflow`, `get_nwchem_workflow_state`, `prepare_nwchem_next_step`, `advance_nwchem_workflow`, `create_nwchem_workflow`, `list_nwchem_protocols` |
| Geometry | 5 | `extract_nwchem_geometry`, `inspect_nwchem_geometry`, `displace_nwchem_geometry_along_mode` |
| Session & versioning | 3 | `init_session_log`, `append_session_log`, `next_versioned_path` |
| TCE (correlated methods) | 6 | `parse_nwchem_movecs`, `swap_nwchem_movecs`, `validate_nwchem_tce_setup` |
| Documentation | 7 | `search_nwchem_docs`, `lookup_nwchem_block_syntax`, `find_nwchem_examples`, `get_nwchem_topic_guide`, `read_nwchem_doc_excerpt`, `list_nwchem_docs`, `search_nwchem_forum` |
| Evaluation | 2 | `evaluate_nwchem_case`, `evaluate_nwchem_cases` |

### Molcas / OpenMolcas tools (45)

Adds 4 scheduler-submit tools to the existing 40: `launch_molcas_run`,
`get_molcas_run_status`, `watch_molcas_run`, `terminate_molcas_run` —
parallel to the NWChem pattern, all tagged `needs="executable"`. Plus
`summarize_molcas_outputs` — bulk one-row-per-file triage over a directory /
glob / list (the Molcas counterpart of `summarize_nwchem_outputs`).

| Tool | Purpose |
|------|---------|
| `draft_molcas_input` | Render a full Molcas input deck (MOLCAS_MEM + SEWARD + SCF + RASSCF + CASPT2 chain). Methods: HF/SCF/DFT/CASSCF/RASSCF/CASPT2/RASPT2/MS-CASPT2/XMS-CASPT2/RMS-CASPT2/XDW-CASPT2 |
| `lint_molcas_input` | Validate Molcas input string. Catches block-pair issues, unknown basis libraries, RASSCF/CASPT2 Frozen mismatches, missing Nactel, suspicious LumOrb without orbital provenance |
| `compute_molcas_active_space_partition` | CAS(M,N) → per-symmetry RASSCF directives (Nactel, Frozen, Inactive, Ras1/2/3, Secondary) |
| `list_molcas_basis_sets` | Enumerate bundled basis sets; filter by element; report contractions for a (basis, element) pair |
| `parse_molcas_output` | Deep parse: per-module SCF / RASSCF / CASPT2 details + active-space summary + warnings |
| `parse_molcas_tasks` | Cheap module-boundary task index |
| `parse_molcas_xmldump` | Parser for the XML dump Molcas writes alongside the .log. Per-module structured data: SCF total energy, kinetic / virial / one-electron / two-electron / nuclear-repulsion energies, basis counts, dipole moment, formal charge, spin. Returns module_counts (workflow-structure verification) + energy_trace (SCF energy per iter, for opt-loop progression). Robust to .log text-format changes. |
| `summarize_molcas_output` | **Single-dispatch summary** of a Molcas .out / .log. Flat structured dict: method, primary_energy_au, modules_run, active_space, geometry + bond_lengths (small systems), frequencies_cm1, imaginary_frequencies_cm1, ZPVE, thermochem at 298.15 K (if freq run), CASPT2 reference weight (if CASPT2), warnings. Saves the agent from chaining 4-5 parse_* tools to answer "what's in this run?". |
| `analyze_molcas_case` | **Quality dispatcher**: runs `summarize_molcas_output` + `validate_molcas_caspt2_setup` (if CASPT2) + `analyze_molcas_active_space` (if RASSCF) + cross-checks (charge/spin parity, imaginary frequencies, warning count). Returns the full summary plus verdict (healthy / caution / problematic) + issues list (severity + message + hint) + next_actions. Use this as the first call when reviewing any Molcas run for quality (not just failures — that's `suggest_molcas_recovery`). Smart imaginary-mode filter: \|f\| < 50 cm⁻¹ tagged as projection artifacts (info), \|f\| ≥ 50 cm⁻¹ as real (info for 1 = TS, caution for multiple). |
| `get_molcas_orbitals` | Last `++ Molecular orbitals:` block — RASSCF NOs win over SCF MOs |
| `parse_molcas_inporb` | Read INPORB / RasOrb / ScfOrb / GssOrb / LprOrb / SpdOrb files |
| `parse_molcas_frequencies` | Last `Harmonic frequencies in cm-1` block from MCLR or numerical-grad. Per-symmetry modes + IR intensity + reduced mass + per-atom displacements; imaginary modes as negative floats |
| `parse_molcas_thermochem` | Per-temperature ZPVE + S + U + H + G (kcal/mol + au); 298.15 K row hoisted under `standard_298_15` |
| `extract_molcas_geometry` | Single geometry snapshot — SLAPAF converged geometry preferred, else last `Cartesian coordinates` block |
| `inspect_molcas_geometry` | Geometric measurement / sanity check. Accepts a geometry from output_file (converged) / input_file / explicit atoms list and reports formula, bond_lengths (annotated `within_covalent_sum` per Mercury-style detection), bond_angles through bonded triples, close_contacts (overlap warning), center_of_mass, fragment count (disconnection detection), and optional user-specified distance / angle / dihedral measurements. Internally normalizes coordinates to Å so bond detection works regardless of source units (Molcas outputs are often in bohr). |
| `parse_molcas_trajectory` | SLAPAF Energy Statistics + per-iteration geometries (cumulative table de-duplicated) |
| `parse_molcas_rassi` | RASSI state-interaction: input states, spin-free + spin-orbit eigenstates (rel + abs energies), SO composition (SF state contributions per SO state), SOC matrix elements above SOCOupling threshold, dipole oscillator strengths in SF + SO bases, NRNATO natural-orbital occupations. Includes SOC stabilization roll-up in cm-1 |
| `analyze_molcas_active_space` | NO-occupation classification → healthy/marginal/poor verdict + promote/demote orbital lists + next_actions |
| `validate_molcas_caspt2_setup` | Reference-weight, IPEA / shift, intruder-state checks → healthy/caution/unreliable verdict |
| `list_molcas_docs` | List 133 bundled OpenMolcas doc files |
| `search_molcas_docs` | Search docs (programs / tutorials / users_guide / advanced_examples) |
| `lookup_molcas_module` | Pull the docs page for a Molcas module (rasscf, caspt2, alaska, ...) |
| `read_molcas_doc_excerpt` | Read a slice of a bundled doc by relative path |
| `get_molcas_topic_guide` | Curated guidance for high-value topics (rasscf_active_space, caspt2_setup, ipea_shift, xms_caspt2, alaska_gradients, mclr_freq, rassi_state_interaction, inporb_format, scf_setup) |
| `prepare_molcas_launch` | Build safe `pymolcas` command + env. Auto-downgrades `-np` to 1 for inputs containing `&CASPT2` when `execution.parallel_caspt2_supported=False` (broken GA builds); always isolates scratch via unique `MOLCAS_PROJECT` to prevent `RunHdr%nProcs/=nProcs` cross-run aborts. Does not execute — returns the command for the caller to run. |
| `swap_molcas_inporb_orbitals` | Swap orbital pairs within a symmetry block of an INPORB / RasOrb file — swaps MO coefficients + occupation + energy + typeindex. Used for active-space tuning workflows (initial RASSCF → inspect orbital character → swap wrong-class orbital with a better candidate → re-run RASSCF with FILEORB). Bit-exact round-trip. |
| `suggest_molcas_orbital_swaps` | Character-aware swap suggester. Walks the LAST '++ Molecular orbitals:' block, classifies each orbital's space (inactive/active/secondary) from the RASSCF orbital_specs, and matches dominant AO against a target pattern (e.g. 'Cr' + '3d'). Returns suggested (active_orb, swap_with) pairs ready to feed into `swap_molcas_inporb_orbitals`. Diagnostic only — final swap choice still requires chemistry judgment. |
| `refine_molcas_active_space` | **Thick orchestrator** that closes the active-space-tuning loop in ONE call: parses an existing RASSCF .out, runs occupation + character analyses, applies the suggested swaps to the RasOrb, generates a refined input with FILEORB injected, and returns a launch plan + next_actions envelope. Replaces the manual chain `parse_molcas_output → suggest_molcas_orbital_swaps → swap_molcas_inporb_orbitals → text-edit input → prepare_molcas_launch`. Supports dry-run (`apply_swaps=False`) for inspection-only. |
| `prepare_molcas_casscf_setup` | **Thick orchestrator** for fresh CASSCF / CASPT2 / MS-CASPT2 calculations. Takes molecule + method spec + (cas_active_electrons, cas_active_orbitals) OR `chemistry_hint='valence_d'`/`'frontier_pair'`. Drafts the input, lints, computes the active-space partition, optionally writes input + builds launch plan, returns a Diagnosis envelope. Auto-corrects parity mismatches in TM hints (e.g. Cr⁰ 3d⁵ + spin-dictated 4s electron). |
| `prepare_molcas_caspt2_chain` | **Thick orchestrator** that takes a converged RASSCF .out and chains CASPT2 on top. Auto-picks SS vs MS variant from `n_roots`, sets IPEA=0.25 by default, emits imaginary shift 0.1 if active-space verdict is 'marginal', mirrors RASSCF Frozen. Short-circuits to `verdict='needs_active_space_refinement'` if RASSCF active space is 'poor' (points at `refine_molcas_active_space` first). The continuation reads previous RASSCF orbitals via FILEORB so RASSCF re-converges in ~4 iters. |
| `prepare_molcas_excited_states` | **Thick orchestrator** for multi-state excited-state workflows. Chains SEWARD + SCF + RASSCF over `n_singlets` singlets + RASSCF over `n_triplets` triplets + per-group MS-CASPT2 + optional RASSI for SOC. Handles the EMIL JobIph plumbing: `>>COPY $Project.JobIph JOBxxx` after each RASSCF, then `>>COPY JOBxxx $Project.JobIph` before each per-group CASPT2 (without this swap, all CASPT2 groups silently read the last RASSCF's wave function). Generates the right RASSI block format (no `Title` keyword — Molcas RASSI rejects it). |
| `prepare_molcas_opt_freq_workflow` | **Thick orchestrator** for geometry-optimization + analytic-frequency runs. Wraps SEWARD + (SCF on iter 1 only) + RASSCF (CASSCF mode) + ALASKA + SLAPAF in an EMIL `>>> Do while <<<` ... `>>> ENDDO <<<` loop, followed by MCKINLEY + MCLR for analytic Hessian + harmonic frequencies. Supports SCF/HF/CASSCF/RASSCF, minimum or transition-state search (`transition_state=True` → adds `TS` to SLAPAF), freq-only single points (`do_optimization=False`), numerical-gradient fallback, and `iroot_freq` for state-averaged frequencies. |
| `prepare_molcas_irc_workflow` | **Thick orchestrator** for intrinsic reaction coordinate (IRC) analysis. Takes a converged TS geometry + reaction vector (parsed from a prior TS opt+freq `.log` via `ts_output_file`, OR passed explicitly as `reaction_vector=[[x,y,z],...]`) and emits GATEWAY + Do-while loop with SEWARD + SCF/RASSCF + ALASKA + SLAPAF IRC. Follows the imaginary-mode reaction coordinate in both directions until energy rises or NIRC points are reached. Produces `$Project.mep.molden` trajectory + final endpoint geometry — used to verify the TS connects the expected reactant + product. **Important: pass TS coords in bohr if you pulled them from the prior log's "Nuclear coordinates for the next iteration" section** (set `geometry_units='bohr'`). |
| `prepare_molcas_scan_workflow` | **Thick orchestrator** for constrained-geometry PES scans (bond / angle / dihedral). For each value in `scan_coordinate.values`, generates a Molcas input that fixes the chosen coordinate via the GATEWAY `Constraint` block + optimizes everything else via SLAPAF. Returns N inputs + sequential launch plan + per-point lint. With `chain_orbitals=True` (default), each point after the first reads the previous's RasOrb/ScfOrb via FILEORB → faster convergence and continuous PES (no orbital flipping between points). Scan-coord spec accepts either `atom_labels=['C1','H1']` (auto-generated labels) or `atom_indices=[2,3]` (1-based into the atoms list). **Watch out:** scans that traverse a bent↔linear transition (e.g. H-C-N as r(C-H) grows) trip SLAPAF's `BMtrx_internal: nq < nQQ` error — use the angle coordinate or perturb the geometry off-axis for those cases. |
| `compute_molcas_reaction_energy` | Post-hoc reaction-energy calculator. Takes signed stoichiometric coefficients on converged outputs (products + reactants) and returns ΔE in au / kcal/mol / eV. For atomization (1 molecule reactant, N atomic products) auto-emits `binding_energy_*` and `is_bound` fields. Honors `energy_kind` (primary / scf / rasscf / caspt2 / ms_caspt2 / rassi_sf / rassi_so) so the agent can force consistent level across species. `include_thermochem=True` adds ΔZPVE, D_0, ΔH(T), ΔG(T), ΔS(T): pulls ZPVE + thermal_H + thermal_G + entropy from each species' Molcas thermochem block (MCLR freq); for monoatomic species lacking thermochem, falls back to ideal-gas Sackur-Tetrode translational entropy + electronic-degeneracy entropy from the spin multiplicity (atomic mass table covers Z=1..30). |
| `check_molcas_active_space_consistency` | Diagnostic for multireference reaction energies. Compares a molecule's CAS dimensions (n_active_electrons, n_active_orbitals) to the SUM of its dissociation-fragment CASes. Verdicts: `consistent` / `molecule_undersized` / `fragments_undersized` / `char_mismatch`. If undersized, returns `suggested_cas=(M,N)` ready to feed into `prepare_molcas_casscf_setup`. Optional character check counts e.g. 'Cr 3d' active orbitals in molecule vs. fragments. Catches the textbook "CASSCF says CrO is unbound" trap before computing the energy. |
| `prepare_molcas_atomization` | **Thick orchestrator** for atomization / binding-energy workflows. Generates the molecule input + one input per unique atomic element at consistent CAS theory. Auto-sums molecule's CAS to span the atomic fragments by default (so `check_molcas_active_space_consistency` passes by construction). Applies `Relativistic R02O02` uniformly when any element requires DKH (TMs Z>=19). Drops the `&SCF` block on high-spin TM atoms (Cr ⁷S, Mn ⁶S, Fe ⁵D, V ⁴F, Co ⁴F) where Molcas ROHF won't converge from GuessOrb — RASSCF starts from GuessOrb directly. Supports `method='CASSCF'` (electronic only) or `method='CASPT2'`/`'MS-CASPT2'` (chain CASPT2 on every species at matching level). `imaginary_shift` defaults to 0.1 for TM systems (intruder protection); the orchestrator bumps RASSCF iterations to (100, 50) to handle TM convergence cliffs. Bundled ground-state table (Z=1..30) supplies multiplicity + recommended CAS per element. Returns launch plans + `next_actions` that chain through `check_molcas_active_space_consistency` + `compute_molcas_reaction_energy` with the right `energy_kind` per method. |
| `suggest_molcas_recovery` | Failure-mode classifier + step-by-step recovery suggester. Takes a Molcas `.out`/`.log` and walks a priority-ordered rule engine covering 11 traps surfaced by real dogfooding: `seward_angstrom_symmetry`, `missing_basis_in_loop`, `scf_single_electron` (H-atom-style abort), `scf_no_convergence` (TM atom from GuessOrb), `rasscf_no_convergence` (iter budget too small), `caspt2_intruder` (small denominators), `caspt2_low_ref_weight` (caution band), `jobiph_missing` (excited-states COPY plumbing), `ga_segfault` (parallel CASPT2 on broken GA build), `memory_exceeded`, `slapaf_no_convergence`, `nactel_parity`. Returns failure_class + severity + root_cause + fix_recipe + next_actions chained into the right orchestrator (e.g. caspt2_intruder → `prepare_molcas_caspt2_chain(imaginary_shift=0.1)`). |
| `apply_molcas_recovery` | Mechanical-fix applicator that pairs with `suggest_molcas_recovery` to close the auto-fix loop. Takes a failed input + (output_file OR pre-computed recovery dict) and regex-edits the input: drop &SCF block (scf_no_convergence, scf_single_electron); bump RASSCF Iteration to (100,50) (rasscf_no_convergence); add `Imaginary 0.1` to &CASPT2 (caspt2_intruder, caspt2_low_ref_weight); 2x MOLCAS_MEM (memory_exceeded); replace opening &SEWARD with &GATEWAY (missing_basis_in_loop); bump SLAPAF Iterations (slapaf_no_convergence). Writes `{stem}_recovered.input`. For non-mechanical failure classes (jobiph_missing, ga_segfault, nactel_parity, seward_angstrom_symmetry), returns verdict=`manual_intervention_required` with the orchestrator path to use. End-to-end validated on three real failures (Cr SCF non-converge, H 1-electron SCF, CrO CASPT2 ref_weight 0.0002) — each recovered automatically and converged on rerun. |
| `try_molcas_run_with_recovery` | **Local-mode** full auto-retry loop. Spawns pymolcas as a subprocess; on failure, calls `suggest_molcas_recovery` + `apply_molcas_recovery` + re-runs, up to `max_retries` times. Returns verdict=converged / max_retries_exhausted / non_recoverable_failure with the full attempt history + recovery_history. The only Molcas tool that directly executes pymolcas (others return launch plans). Capability tag: `executable` — hidden in `analysis` mode. |

Bundled data:
- 133 Markdown docs at `chemtools/data/molcas/docs/` (programs, tutorials, users_guide, advanced_examples, installation, overview)
- 91 basis library files at `chemtools/data/molcas/basis_library/` (3-21G, 6-31G family, ANO-RCC, ANO-L, ANO-S, ANO-XS, AUG-CC-PVxZ, etc.)

Plugin layout (`chemtools/programs/molcas/`):
- `parse/output.py` — module-boundary task index + `parse_output_full` orchestrator
- `parse/scf.py`, `parse/rasscf.py`, `parse/caspt2.py`, `parse/mos.py` — module-specific parsers
- `parse/freq.py`, `parse/thermochem.py`, `parse/geometry.py` — MCLR freq + thermochem + SLAPAF trajectory parsers
- `parse/rassi.py` — RASSI module parser (spin-free + spin-orbit eigenstates, SOC matrix, dipole strengths)
- `binary/orbitals.py` — INPORB / RasOrb file reader (named-section format)
- `strategy/active_space.py` — `analyze_active_space` and `validate_caspt2_setup`
- `input/seward.py`, `input/scf.py`, `input/rasscf.py`, `input/caspt2.py` — block-level builders
- `input/draft.py` — `draft_molcas_input` orchestrator (InputSpec → full deck)
- `input/lint.py` — `lint_molcas_input` (block pairs, basis labels, RASSCF↔CASPT2 Frozen consistency, Nactel sanity)
- `input/basis_library.py` — bundled basis library reader (default contractions, label builder)
- `docs.py` — bundled docs accessor
- `runtime.py` — launch-helper (`prepare_launch` returns safe pymolcas command + env, with CASPT2 -np guard rail and scratch isolation)
- `_plugin_parser.py`, `_plugin_binary.py`, `_plugin_drafter.py` — sub-protocol implementations

### DIRAC tools (38)

Adds 4 scheduler-submit tools to the existing 34: `launch_dirac_run`
(takes a `mol_file` argument), `get_dirac_run_status`, `watch_dirac_run`,
`terminate_dirac_run`.

| Category | Tools |
|----------|-------|
| Atomic SCF input | `prepare_dirac_atomic_start`, `prepare_dirac_core_ionization`, `prepare_dirac_cm_class_workflow`, `prepare_dirac_x2c_bootstrap` |
| Molecular input | `prepare_dirac_molecular_scf`, `draft_dirac_input`, `draft_dirac_mol`, `draft_initial_geometry` |
| Output parsing | `parse_dirac_output`, `parse_dirac_vecpop`, `parse_dirac_hessian`, `parse_dirac_spinor_spectrum`, `parse_dirac_cosci_energies` |
| HDF5 / binary | `read_dirac_orbitals`, `read_dirac_mo_coefficients`, `read_dirac_h5_geometry`, `read_dirac_h5_metadata` |
| Frequency | `compute_dirac_harmonic_frequencies` |
| Reorder / geometry | `draft_dirac_reorder_block`, `apply_dirac_reorder_to_input` |
| Analysis | `analyze_dirac_open_shell_quality`, `summarize_dirac_run` |
| Basis | `list_dirac_basis_sets`, `suggest_dirac_basis` |
| Documentation | `get_dirac_topic_guide`, `search_dirac_docs`, `read_dirac_doc_excerpt`, `lookup_dirac_section` |
| Strategy | `suggest_relativistic_correction` |

Key constraints:
- **4c is the default Hamiltonian** — `use_x2c=False` everywhere. X2C has a convergence pathology in DIRAC 25 + dyall.2zp for Z≥96 (oscillates at a wrong fixed-point, not user-tunable).
- **4c→X2C bootstrap does NOT work** — 4c CHECKPOINT.h5 is incompatible with X2C orbital space; DIRAC silently ignores it.
- **Cm (Z=96)+ direct AOC fails in DIRAC 25** — use Pu as the surrogate reference (`prepare_dirac_cm_class_workflow`).
- **`--outcmo` fails for 4c** — use `--get="CHECKPOINT.h5"` to retrieve checkpoints from 4c runs.
- **Diffuse basis families (av*, acv*, aae*) exclude f-block elements** — `suggest_dirac_basis` handles this automatically.

Plugin layout (`chemtools/programs/dirac/`):
- `parse/output.py` — SCF iteration trace, total energy, spinor eigenvalue spectrum, MULPOP detail, COSCI state table
- `parse/vecpop.py` — per-spinor population analysis
- `parse/inp.py` — DIRAC `.inp` file parser
- `parse/mol.py` — DIRAC `.mol` geometry file parser
- `binary/` — HDF5 checkpoint reader (geometry, MO coefficients, metadata)
- `input/atomic_start.py` — AOC atomic SCF input + KPSELE block builder (4c default)
- `input/core_ionization.py` — core-IP input builder
- `basis.py` — Dyall basis catalog (25 families, element coverage, f-block diffuse caveat)
- `docs.py` — bundled DIRAC docs accessor
- `runtime.py` — pam-dirac launch helper
- `strategy/` — open-shell quality analysis, Cm-class workflow routing
- `_plugin_parser.py`, `_plugin_binary.py` — sub-protocol implementations

### GRASP2018 tools (37)

Includes 26 original tools (per-exe runners, planners, parsers, session log),
7 parity tools (analyze_grasp_case, suggest_grasp_recovery, docs tools +
topic guides), and 4 scheduler-submit tools (`launch_grasp_workflow_run`
takes a `workflow_script_path` rather than a single input file, since
GRASP workflows are multi-exe shell scripts).

GRASP is structurally different from NWChem/Molcas/DIRAC: ~50 small executables run sequentially, each prompted via stdin (no input file). Tools wrap individual executables, plan workflows, and parse the `name.{w,c,m,sum,lsj.lbl}` files produced by `rsave`.

| Category | Tools |
|----------|-------|
| Per-exe runners (executable cap) | `run_grasp_rnucleus`, `run_grasp_rcsfgenerate`, `run_grasp_rangular`, `run_grasp_rwfnestimate`, `run_grasp_rmcdhf`, `run_grasp_rsave`, `run_grasp_jj2lsj`, `run_grasp_rlevels`, `run_grasp_hf`, `run_grasp_rwfnmchfmcdf`, `run_grasp_rci`, `run_grasp_exe` (escape hatch) |
| Workflow planners (any mode) | `plan_grasp_dhf_workflow`, `plan_grasp_nonrel_limit_workflow`, `plan_grasp_restart_from_workflow`, `plan_grasp_hf_bootstrap_workflow` |
| Workflow runner | `run_grasp_workflow` (executes a plan end-to-end) |
| Parsers | `parse_grasp_levels`, `summarize_grasp_terms`, `compare_grasp_levels`, `parse_grasp_lsjlbl`, `parse_grasp_sum`, `parse_grasp_rmcdhf_log` |
| Container + session log | `get_grasp_container`, `read_grasp_session_log`, `append_grasp_session_note` |

Key constraints:
- **Container path**: resolved via `CHEMTOOLS_GRASP_CONTAINER` env var (default `~/mycontainers/grasp2018.sif`). Run `get_grasp_container` to verify.
- **Per-run session log**: every `run_grasp_*` call appends a markdown block to `<working_dir>/grasp_session.md` recording the command, stdin, key stdout, and elapsed time — replayable trail.
- **Block-level selections**: `rmcdhf` prompts for ASF serial numbers per block. Pass one entry per block (e.g. `['1','1-2','1']` for 3 blocks). Mismatched length crashes with "End of file".
- **High-Z bootstrap**: for Z≥80 atoms (Cf, Bk, Th), `plan_grasp_hf_bootstrap_workflow` adds a non-rel `hf` + `rwfnmchfmcdf` step before `rwfnestimate` because Thomas-Fermi alone diverges.
- **Non-rel limit**: `plan_grasp_nonrel_limit_workflow` sets c=2000 au, suppressing all relativistic effects — useful for isolating relativistic contributions to a property.
- **rcsfgenerate output**: writes `rcsf.out`; the runner auto-copies it to `rcsf.inp` via `copy_to_inp=True` so the next step (rangular) finds it.

Plugin layout (`chemtools/programs/grasp/`):
- `runtime.py` — apptainer wrapper, stdin heredoc execution, session log writer
- `parse/rlevels.py` — energy-level table (No / Pos / J / Parity / E_au / E_cm-1 / splitting / config) + term grouping + DHF-vs-NR comparison
- `parse/lsjlbl.py` — LSJ-coupled composition per ASF
- `parse/sum_file.py` — `name.sum` summary (nucleus, c, grid, subshells, eigenenergies)
- `parse/rmcdhf_log.py` — SCF iteration trace from rmcdhf stdout
- `input/heredoc.py` — typed stdin builders for each exe (rnucleus, rcsfgenerate, rangular, rwfnestimate, rmcdhf, jj2lsj, hf, rwfnmchfmcdf, rci)
- `strategy/workflows.py` — DHF / non-rel / restart-from-w / hf-bootstrap planners
- `strategy/runner.py` — execute a workflow plan end-to-end with stop-on-failure
- `data/grasp/docs/` — bundled GRASP2018 manual (12 markdown files, 4 parts)

## Eval Framework

`chemtools/core/eval.py` — multi-program case evaluator. Dispatches by
the `program` field in `case.json` and calls program-specific checks.

A **case** is a directory with a `case.json` (or `*.case.json`) file that
specifies input/output files and `eval_expectations`. The expectations
vary by program:

| Program | Checks |
|---|---|
| NWChem | `diagnosis_failure_class`, `diagnosis_stage`, `recommended_next_action`, `workflow`, `can_auto_prepare` |
| Molcas | `primary_energy_au` (±tolerance), `modules_run` (presence), `converged`, `verdict` |
| DIRAC  | `scf_energy_au` (±tolerance), `converged`, `n_occupied_spinors`, `n_cosci_states` |
| GRASP  | `ground_energy_au` (±tol), `speed_of_light_au`, `atomic_number`, `n_subshells`, `n_levels`, `is_nonrel_limit`, `file_kind` |

Case files live under:

```
nwchem-test/train/          (4 NWChem cases — h2o2 imaginary freq, Cu opt,
                             failed SCF, closed-shell organic)
examples/molcas/<system>/case.json   (5 cases — acrolein CASSCF+CASPT2,
                             thiophene multi-root, cyclo-freq opt+MCLR,
                             PbO MS-CASPT2, benzene SCF+MBPT2)
examples/dirac/<system>/case.json    (3 cases — H2O / CO / N2 4c-DHF)
examples/grasp/<system>/case.json    (3 cases — Li DHF, Li non-rel limit,
                             Si 3p² ground term)
```

All 15 cases pass. The case-file directories are **gitignored** (local
training corpus), but the eval framework code is tracked.

MCP tools: `evaluate_nwchem_case(case_path)` and `evaluate_nwchem_cases(path)`
(NWChem-named for back-compat but program-agnostic — both dispatch by the
`program` field in case.json).

## How to Add a New Tool

1. Write the domain function in the appropriate `chemtools/programs/<program>/`
   submodule (parser, drafter, strategy, runtime). For generic tools, put it
   in `chemtools/core/`.
2. Add a tool-definition dict to `<program>_tool_definitions()` in
   `chemtools/mcp/tools/<program>.py` — schema + description.
3. Add a `@_tool("tool_name", program="<program>", needs="<cap>")` handler
   function in the same file that calls the library. Capability tag drives
   which server modes expose it (none / registry / runner_profile /
   executable_or_scheduler / executable / scheduler).
4. Verify:
   ```bash
   python3 -c "from chemtools.mcp.tools.nwchem import tool_definitions; print(len(tool_definitions()), 'tools')"
   chemtools --list-tools | grep your_new_tool
   ```

## How to Improve an Existing Tool

The iterative workflow: run a NWChem job locally, see what the agent does wrong or can't do, fix the tool.

Common patterns:
- Parser misses a new output format → fix regex in `nwchem_tasks.py` or `nwchem_tce.py`
- Strategy tool gives wrong advice → improve heuristics in `api_strategy.py`
- Input drafter generates bad syntax → fix in `api_input.py`, check with lint
- New NWChem module/feature → add parser + MCP tool following the pattern above

## Key Design Rules

- **Never add `freeze atomic` to TCE inputs** — always compute and emit explicit `freeze N`
- **Always inspect orbital ordering before TCE freeze** — `parse_nwchem_movecs` first, then decide
- **Binary movecs swap required for reordering** — `vectors swap` in SCF block doesn't survive re-diagonalization
- **MCP handlers are thin** — all logic lives in `chemtools/`, handlers just translate arguments
- **Explicit basis blocks** — generate explicit per-element basis text from the library, not `library` shorthand
- **Lint after drafting** — every input tool should be followed by lint in the workflow
- **Never overwrite input files** — always call `next_versioned_path` before writing a modified `.nw` file; the first version stays as-is and revisions become `_v2.nw`, `_v3.nw`, etc.
- **Always start a session log** — call `init_session_log` at the beginning of any multi-step workflow; append entries with `append_session_log` after each action, decision, or error; write a `summary` entry at the end
- **Parallel job monitoring** — submit jobs with `auto_watch=false`, then call `watch_multiple_runs` (not `watch_nwchem_run` in a loop) to block until all finish simultaneously
- **Register runs in the registry** — call `register_nwchem_run` when submitting jobs, `update_nwchem_run_status` after completion; this enables campaign tracking and energy tables across sessions
- **Use campaigns for related runs** — create a campaign first (`create_nwchem_campaign`), then link runs via `campaign_id`; use `get_nwchem_campaign_energies` for sorted energy tables with relative energies in kcal/mol
- **Workflow DAGs for multi-step protocols** — use `create_nwchem_workflow` for dependent steps (opt→freq), then `advance_nwchem_workflow` to find ready-to-launch steps
- **Registry is SQLite at `~/.chemtools/registry.db`** — uses stdlib `sqlite3`, no external dependency; override with `CHEMTOOLS_REGISTRY_DB` env var for testing

### Molcas / multi-reference workflow rules

- **Always check active-space quality before CASPT2** — call `analyze_molcas_active_space` after RASSCF; do not run CASPT2 on a `poor` verdict
- **Reference weight ≥ 0.70 is the trust threshold** — `validate_molcas_caspt2_setup` returns `unreliable` below that; agent should redraft the active space, not the CASPT2 input
- **Real intruders need both small denominator AND large coefficient** — large coefficient alone is normal chemistry; small denominator alone is a benign near-degeneracy
- **IPEA shift defaults to 0.25 from Molcas 6.4** — `MOLCAS_NEW_DEFAULTS=YES` switches to 0.0; `parse_molcas_output` reports the actual value used so the agent can flag mismatches with the user's intent
- **MS / XMS / RMS / XDW for state-mixing** — when SS-CASPT2 has multiple closely-spaced states, `validate_molcas_caspt2_setup` emits the `multistate_hint` warning
- **Last `++ Molecular orbitals:` block wins** — `get_molcas_orbitals` automatically returns the RASSCF NOs (which override SCF MOs that appeared earlier in the same task); use this to label active orbitals via the `dominant_aos` field
- **Energy roll-up follows SO-RASSI > RASSI SF > MS-CASPT2 > CASPT2 > RASSCF root 1 > SCF** — `parse_molcas_output` returns `energy_summary.primary_energy_hartree` with the chosen label; SO ground state wins whenever a RASSI SPINorbit run is present
- **Internal pymolcas modules are filtered** — `last_energy`, `last_atoms`, and `emil` never show up in the task list
- **CAS dimensions across reactants must match** — for atomization / binding / dissociation at CASSCF, the molecule's CAS must be at least as large as the sum of the dissociation fragments' CASes (e.g. CrO needs CAS(10,9) to dissociate into Cr CAS(6,6) + O CAS(4,3) cleanly). Run `check_molcas_active_space_consistency` BEFORE `compute_molcas_reaction_energy` — a molecule-undersized verdict explains unphysical results like CASSCF-says-CrO-unbound by 13 kcal/mol
- **DKH is required for absolute energies on transition metals** — ANO-RCC is contracted for relativistic eigenfunctions; without `Relativistic R02O02` in SEWARD, Cr atom comes out ~9 au too high. The relativistic correction mostly cancels between reactant and product for binding energies, but mixing DKH and non-DKH across species is a bug. Either both or neither.
- **TM atomic ROHF often won't converge from GuessOrb** — for Cr ⁷S(3d⁵4s¹), skip SCF and run RASSCF directly (Molcas GuessOrb starting orbitals reach the right CASSCF minimum after a few iters). Then run `suggest_molcas_orbital_swaps` to confirm 3d-character orbitals are in active before trusting the energy.

## Runner Profiles

Runner profiles tell the agent how to launch, monitor, and cancel NWChem jobs. They are
**per-machine configuration** (not checked into this repo). Set `CHEMTOOLS_RUNNER_PROFILES`
to point at your local YAML or JSON file.

Example files in this repo:
- `chemtools/runner_profiles.example.yaml` — canonical reference with all profile types
- `chemtools/runner_profiles.example.json` — auto-synced JSON copy (same content)
- `chemtools/runner_profiles.local.example.json` — minimal template for local customization

### Local profiles (`launcher.kind: "direct"`)

NWChem runs as a foreground subprocess on the same machine as the agent. The agent
monitors the process by PID and tails the output file.

```yaml
local_mpirun:
  launcher:
    kind: "direct"
    command: "mpirun -np {mpi_ranks} /path/to/nwchem"
  execution:
    command_template: "{launcher} {input_file} > {output_file} 2> {error_file}"
  resources:
    mpi_ranks: 14
```

### HPC / scheduler profiles (`launcher.kind: "scheduler"`)

NWChem is submitted to a queue. The agent submits via `sbatch`/`qsub`, writes
`{job_name}.jobid` alongside the input, and monitors via the scheduler's status command.

**Key fields:**

| Field | Purpose |
|---|---|
| `launcher.submit_command` | `sbatch`, `qsub`, `bsub` |
| `launcher.scheduler_type` | `slurm`, `pbs`, `lsf` — drives state mapping |
| `launcher.job_id_regex` | Regex to extract job ID from submit stdout |
| `launcher.status_command` | e.g. `squeue -j {job_id} -h -o %T` (returns state only) |
| `launcher.cancel_command` | e.g. `scancel {job_id}` |
| `execution.nwchem_executable` | Full path to the NWChem binary |
| `execution.mpi_launch` | Full MPI launch prefix: `ibrun` (TACC), `srun`, `mpirun -np 48` |
| `resources.nodes/mpi_ranks/walltime/partition/account` | Default job resources |
| `resources.account_command` | Shell command to discover allocations (e.g. `/usr/local/etc/taccinfo` on TACC) |
| `resources.cores_per_node` | Physical cores per node — enables auto rank selection |
| `resources.node_memory_mb` | Total RAM per node in MB — enables memory ceiling checks |
| `resources.max_nodes` | Max nodes available for jobs — enables multi-node suggestions |
| `resources.max_walltime` | Max walltime the queue allows (e.g. `"48:00:00"`) |
| `resources.cpu_arch` | CPU microarchitecture (`skx`, `icx`, `spr`, `avx2`) — tunes BF/rank |
| `scheduler.script_template` | Shell script with `{placeholder}` substitutions |
| `scheduler.submit_script_name` | Filename for the generated script, e.g. `{job_name}.job` |
| `modules.load` | List of `module load` commands to include in the script |
| `hooks.pre_run` | Shell commands inserted before the NWChem launch line |

**Template placeholders available in `script_template`:**
`{job_name}`, `{output_file}`, `{error_file}`, `{nodes}`, `{mpi_ranks}`, `{omp_threads}`,
`{walltime}`, `{partition}`, `{account}`, `{account_line}` (the full `#SBATCH -A ...` line or
empty string), `{nwchem_executable}`, `{mpi_launch}`, `{module_block}`, `{pre_run_block}`,
`{job_dir}`, `{input_file}`.

**TACC Stampede3 example** (profiles `stampede3_skx` / `stampede3_icx` / `stampede3_spr` / `stampede3_skx_dev` in the example file):
```yaml
stampede3_skx:
  launcher: { kind: "scheduler", submit_command: "sbatch", ... }
  scheduler: { script_template: "...", submit_script_name: "{job_name}.job" }
  execution:
    nwchem_executable: "/path/to/nwchem"
    mpi_launch: "ibrun"
  resources:
    # --- Defaults (overridden by suggest_nwchem_resources) ---
    nodes: 1
    mpi_ranks: 48
    partition: "skx"
    walltime: "24:00:00"
    account: null
    account_command: "/usr/local/etc/taccinfo"  # auto-detect allocation
    # --- Hardware description (static) ---
    cores_per_node: 48
    node_memory_mb: 192000
    max_nodes: 256
    max_walltime: "48:00:00"
    cpu_arch: "skx"
```

### How HPC monitoring works

1. `launch_nwchem_run` submits the job, parses the job ID, writes `{job_name}.jobid`
2. `get_nwchem_run_status` / `watch_nwchem_run` auto-detect the `.jobid` file from the
   input/output path — no need to pass `job_id` explicitly
3. Scheduler state (PENDING/RUNNING/COMPLETED/FAILED/etc.) is mapped to normalized status:
   `queued`, `running`, `completed`, `failed`, `cancelled`
4. Output file is tailed in parallel with scheduler polling — slow-phase detection works
   the same as local runs
5. `terminate_nwchem_run` accepts `job_id + profile` for HPC cancel (calls `scancel`/`qdel`)

### Auto resource selection

The `suggest_nwchem_resources` tool analyzes an input file against a profile's hardware
specs and recommends optimal nodes, MPI ranks, walltime, and memory directive. This
replaces manual guessing and prevents common HPC failures (OOM, walltime exceeded).

Profiles should describe the machine with these fields in `resources`:
- `cores_per_node` — physical cores per node
- `node_memory_mb` — total RAM per node in MB
- `max_nodes` — max nodes available for jobs
- `max_walltime` — max walltime the queue allows (e.g. `"48:00:00"`)
- `cpu_arch` — CPU microarchitecture (`skx`, `spr`, `avx2`, etc.)

The advisor handles:
- **Small molecules**: reduces ranks to avoid communication overhead
- **Numerical frequencies**: estimates 6*N_atoms displacements, scales to multi-node
  if needed, warns about no checkpoint capability
- **TCE**: scales nodes for memory when correlation memory exceeds single node
- **Walltime**: task-type-aware estimates with safety margins

### Agent workflow for HPC (single job)

```
init_session_log(log_path=..., session_title=...)  → start running doc
inspect_runner_profiles                             → verify profile is available
suggest_nwchem_resources(input_file, profile)        → get optimal resource_overrides
render_job_script(profile=..., resource_overrides=.) → preview the .job script
lint_nwchem_input                                   → check input is correct
launch_nwchem_run(auto_watch=true, resource_overrides=.) → sbatch + block until done
append_session_log(entry_type="result", ...)        → record outcome
analyze_nwchem_case                                 → diagnosis
append_session_log(entry_type="summary", ...)       → final summary
```

### Agent workflow for HPC (parallel jobs)

```
init_session_log(...)                               → start running doc
# For each job:
next_versioned_path(path="mol.nw")                  → get safe output path
lint_nwchem_input                                   → validate
launch_nwchem_run(auto_watch=false)                 → submit all jobs first
# After all submitted:
watch_multiple_runs(jobs=[...])                     → block until all done
# Analyze each result
append_session_log(entry_type="summary", ...)       → final summary
```

### Agent workflow for campaigns (e.g. ligand screen)

```
init_session_log(...)                               → start running doc
create_nwchem_campaign(name="ligand_screen")         → get campaign_id
generate_nwchem_input_batch(template, vary={...})    → create all inputs
# For each generated input:
register_nwchem_run(campaign_id=..., ...)            → track in registry
lint_nwchem_input                                    → validate
launch_nwchem_run(auto_watch=false)                  → submit
# After all submitted:
watch_multiple_runs(jobs=[...])                      → block until done
# After completion:
update_nwchem_run_status(run_id=..., status=..., energy_hartree=...) → record results
get_nwchem_campaign_energies(campaign_id=...)         → sorted energy table
append_session_log(entry_type="summary", ...)         → final summary
```

### Agent workflow for multi-step protocols (e.g. opt→freq)

```
init_session_log(...)                               → start running doc
plan_nwchem_calculation(protocol="thermochem_dft")   → get step plan
create_nwchem_workflow(steps=[...])                  → create DAG
advance_nwchem_workflow(workflow_id=...)              → find ready steps
# Launch ready step, wait, update status
advance_nwchem_workflow(workflow_id=...)              → next ready steps
# Repeat until workflow is done
append_session_log(entry_type="summary", ...)         → final summary
```

## Server modes

The MCP server runs in one of three modes; the mode determines which tools are exposed
at `tools/list` time and gated at `tools/call` time. This lets the same package serve a
laptop user doing post-hoc analysis, a workstation user running NWChem locally, and an
HPC user submitting to a scheduler — without the agent ever seeing tools it cannot use.

| Mode | Tools visible | Use when |
|---|---|---|
| `analysis` | 209 | No executable available; post-hoc parsing (NWChem + Molcas + DIRAC + GRASP), drafting, planning, registry tracking of runs done elsewhere |
| `local` | 249 | All 4 programs run as subprocesses on this machine (profile with `launcher.kind: "direct"`) |
| `hpc` | 252 | All 4 programs submitted to a scheduler (profile with `launcher.kind: "scheduler"`) — full submit/watch/terminate tooling for each |

### Selecting a mode

Priority order:
1. `chemtools --mode {analysis|local|hpc}` CLI flag
2. `CHEMTOOLS_MODE` env var
3. **Auto-detect** (default):
   - `CHEMTOOLS_RUNNER_PROFILES` not set → `analysis`
   - profiles file has any `launcher.kind: "scheduler"` → `hpc`
   - profiles file has only `direct` profiles → `local`
   - profiles file unreadable or empty → `analysis` (logged)

Auto-detect means most users never configure mode explicitly; the existing
`CHEMTOOLS_RUNNER_PROFILES` env var carries the signal.

### Selecting which programs are loaded

Tools are tagged with a program (`nwchem`, `molcas`, `dirac`, `grasp`,
or `generic`). The `--programs` filter restricts which subset is exposed
at `tools/list` time so a session loads only the tools it needs
(context-cost matters as we add more programs). Priority order:

1. `chemtools --programs molcas` (or `--programs nwchem,molcas`) CLI flag
2. `CHEMTOOLS_PROGRAMS` env var (comma-separated)
3. No filter (all programs visible) — current behaviour when nothing is set

Generic tools (e.g. `compute_reaction_energy`, `init_session_log`,
`render_job_script`) are always visible regardless of filter — they're
program-agnostic by design.

The filter is applied in the live server's `tools/list` **and** gated at
`tools/call` (an out-of-filter tool is refused with an explanatory error), not
just in `--list-tools`. `CHEMTOOLS_PROGRAMS=nwchem` takes the analysis-mode
surface from 211 to ~119 tools.

**Per-session program selection without splitting the package:** register the
same `chemtools` binary multiple times, one per program, and enable whichever
you need that session (Claude Code `/mcp`). The shared core (registry, eval,
runner, generics) stays in one codebase.
```
"mcpServers": {
  "chem-nwchem": { "command": "chemtools", "env": {"CHEMTOOLS_PROGRAMS": "nwchem"} },
  "chem-molcas": { "command": "chemtools", "env": {"CHEMTOOLS_PROGRAMS": "molcas"} }
}
```

### Selecting a curated tool subset (small models)

The `--toolset` filter (env `CHEMTOOLS_TOOLSET`) further trims the surface to an
exact tool-name allowlist — a preset name or a comma-separated list — applied
after the mode + program filters. This is the lever for small models (Haiku /
Llama) and focused work, where 100+ tools is too many to choose among.

- Bundled preset **`triage`** = the 12-tool output-assessment set
  (`summarize_nwchem_outputs`, `analyze_nwchem_case`, `summarize_nwchem_output`,
  `parse_nwchem_output`, `suggest_nwchem_recovery`, `suggest_nwchem_multiplicity_scan`,
  `analyze_nwchem_frontier_orbitals`, `check_nwchem_spin_charge_state`,
  `extract_nwchem_geometry`, `parse_nwchem_thermochem`, `compare_nwchem_runs`,
  `get_server_mode`).
- Bundled preset **`molcas-triage`** = the 11-tool Molcas counterpart
  (`summarize_molcas_outputs`, `analyze_molcas_case`, `summarize_molcas_output`,
  `parse_molcas_output`, `suggest_molcas_recovery`, `analyze_molcas_active_space`,
  `validate_molcas_caspt2_setup`, `parse_molcas_frequencies`,
  `parse_molcas_thermochem`, `extract_molcas_geometry`, `get_server_mode`).
- `CHEMTOOLS_PROGRAMS=nwchem CHEMTOOLS_TOOLSET=triage` → 12 tools (vs ~250);
  `CHEMTOOLS_PROGRAMS=molcas CHEMTOOLS_TOOLSET=molcas-triage` → 11.
- Unlike the program filter, the toolset is an **exact** allowlist: generics are
  not auto-included; list them by name if wanted. Add new presets in
  `chemtools/mcp/modes.py:TOOLSETS`.

### Batch output triage

`summarize_nwchem_outputs(path|paths, pattern, recursive, limit)` assesses many
outputs in one call: a directory, glob, file, or list → one compact row per file
(method, stage, status, energy, failure_class, verdict, headline) plus roll-up
counts by verdict and failure_class. Use this instead of one `analyze_nwchem_case`
call per file when triaging a batch; drill into flagged files afterward.

### Capability tags

Each tool is tagged via `@_tool("name", needs="...")`. Valid tags:

| Tag | Modes exposing it | Tools |
|---|---|---|
| `none` (default) | analysis, local, hpc | 191 pure-Python tools (parsing, drafting, suggest, docs, eval, planners) |
| `registry` | analysis, local, hpc | 18 SQLite registry/campaign/workflow tools |
| `runner_profile` | local, hpc | 2 profile inspection/validation tools |
| `executable_or_scheduler` | local, hpc | 5 resource advisors that adapt to `launcher.kind` |
| `executable` | local, hpc | 33 job-execution tools (per-exe runners + launch/watch/terminate for all 4 programs) |
| `scheduler` | hpc | 3 scheduler-only tools (`render_job_script`, `detect_nwchem_hpc_accounts`, `suggest_nwchem_partition`) |

To add a new tool: tag it on the decorator. `needs="none"` is the default and is
also the right answer for the majority of tools.

### CLI debugging

```
chemtools --show-mode                          # mode + reason + program filter + blocked tools (JSON)
chemtools --list-tools                         # tool names visible under the active filters
chemtools --mode analysis                      # force analysis mode (post-hoc work)
chemtools --programs molcas                    # only Molcas tools (+ generics) loaded
chemtools --mode local --programs nwchem,molcas  # local executable, both programs
chemtools --programs nwchem --toolset triage   # 12-tool assessment set (small models)
```

The legacy binary `chemtools-nwchem` still works as an alias for `chemtools`
(prints a deprecation hint on stderr).

At runtime, agents can call `get_server_mode` to introspect which mode they are in
and which tools are blocked — useful when a tool fails with a "not available in mode"
error or before recommending a workflow that needs HPC submission.

## Development Environment

- Install in editable mode: `pip install -e .`
- Entry points: `chemtools` (primary), `chemtools-nwchem` (legacy alias), `chemtools-nwchem-docs`
- Basis library: bundled at `chemtools/data/nwchem/basis_library/` (auto-detected after install)
- NWChem docs: bundled at `chemtools/data/nwchem/docs/` (29 text files, always available)
- Runner profiles: set `CHEMTOOLS_RUNNER_PROFILES` to your local YAML/JSON file
- Server mode: set `CHEMTOOLS_MODE` or pass `--mode` (see **Server modes** above; defaults to auto-detect)
