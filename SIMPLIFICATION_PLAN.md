# Chemtools simplification plan

Status: active

This task list turns Chemtools into a small, portable computational-chemistry
assistant for personal use and occasional use by a coworker. It preserves the
tested parsers, input builders, scientific checks, execution support, and
curated chemistry knowledge. It reduces the default AI-facing interface and
removes compatibility layers that no longer serve a real caller.

The existing [project plan](PROJECT_PLAN.md) and accepted ADRs remain the
record of earlier design decisions. This file is the working queue for the
simplification effort.

## Target product

- One installable Python distribution and one repository.
- One `chemtools` command with a local stdio MCP server.
- A default guided surface of no more than twelve user-intent tools.
- Program-specific functions retained as tested Python APIs or focused
  developer tools when they still have a real use.
- Portable runner profiles for local machines and Slurm systems.
- An optional thin plugin containing workflow skills and MCP configuration.

The Python library is the source of scientific behavior. MCP handlers adapt
that behavior for agents; plugin skills describe when and how to use it.

## Guardrails

- Preserve scientific behavior with exact tests before moving code.
- Do not rewrite working parsers merely to make the directory tree uniform.
- Add a public MCP tool only for a distinct user intent.
- Keep program syntax and scientific interpretation in program packages.
- Keep execution policy and process control out of parsers.
- Remove compatibility code only after its callers are identified and migrated.
- Keep external executables and large scientific runtimes optional.

## Current baseline

- [x] Record the 2026-08-06 default-suite result: 1,649 tests passed.
- [x] Record the live MCP surface: 337 public definitions, 283 visible in the
      default analysis mode, and 15 hidden compatibility aliases.
- [x] Identify the current guided surface: `review_input`, `draft_input`,
      `inspect_run`, `compare_runs`, `plan_calculation`, `plan_recovery`,
      `launch_run`, `monitor_run`, `visualize`, and `search_knowledge`.
- [x] Confirm that the main structural debt is already documented: inward
      facade imports, module-global MCP composition, manual schema/handler
      pairs, and temporary compatibility shims.
- [x] Identify the stale `chem-agent-package/` tree as replacement or removal
      work rather than a second packaging foundation.
- [x] Verify the first NWChem behavior-lock slice against the external corpus:
      1,673 tests passed on 2026-08-06.
- [x] Verify the guided drafting and workflow-mapping slice against the same
      corpus: 1,690 tests passed on 2026-08-06.
- [x] Verify the guided recovery-planning slice against the same corpus:
      1,703 tests passed on 2026-08-06.
- [x] Build the 0.1.0 wheel, install it with declared dependencies in a clean
      Python 3.9 environment, and pass the installed-copy guided smoke check
      on 2026-08-06. The check also caught and fixed omitted profile and case
      data in the wheel manifest.
- [x] Audit runtime imports and verify the dependency boundary on 2026-08-06:
      NumPy and PyYAML are the base dependencies; DIRAC HDF5 is the `dirac`
      extra; the fixed science runtime remains a separate environment. Clean
      installs passed both the no-h5py base check and the h5py-backed DIRAC
      checkpoint check.
- [x] Bundle and validate a standalone generic Slurm profile with no
      user-specific paths; the complete external-corpus suite passed 1,704
      tests on 2026-08-06.
- [x] Add one installed-wheel setup guide and a CLI path for printing bundled
      local and Slurm profiles; the complete external-corpus suite passed
      1,707 tests on 2026-08-06.
- [x] Verify a clean-user installation boundary on 2026-08-07: an isolated
      user site, home, and config directory loaded all dependencies from the
      temporary install, negotiated the stdio MCP protocol, listed six guided
      tools, and inspected the real ferrocene solution output successfully.
- [x] Pin realistic prompt-to-tool choices for all eleven retained guided
      intents, including explicit unavailable states for planned tools; the
      complete external-corpus suite passed 1,710 tests on 2026-08-07.
- [x] Add the read-only `plan_calculation` application contract and NWChem
      provider. The UO2 optimize-frequency case now exposes its ordered stages,
      unresolved method, basis/ECP, relativistic, geometry, and open-shell
      decisions without rendering input or creating files; the complete
      external-corpus suite passed 1,719 tests on 2026-08-07. A rebuilt wheel
      also listed seven guided tools and returned the same planning verdict
      through an installed stdio MCP exchange outside the source tree.
- [x] Add the guided `launch_run` application contract and initial NWChem
      provider. The first call prepares an exact plan without writing files;
      the second requires a token bound to the input, target, resources,
      command, configured-environment fingerprint, artifact paths, and Slurm
      script. Existing artifacts block launch rather than being overwritten or
      archived. The complete external-corpus suite passed 1,731 tests on
      2026-08-07. A rebuilt wheel listed eight guided tools and returned an
      `awaiting_approval` launch plan through installed stdio MCP without
      creating output files.
- [x] Add the read-only `monitor_run` application contract over owned launch
      records. It accepts only a launch ID from the same server process,
      refreshes retained local or target-owned Slurm state, observes only
      recorded artifact paths, and uses a declared backend progress provider
      when available. Missing scheduler records remain unresolved rather than
      implying completion. The complete external-corpus suite passed 1,742
      tests on 2026-08-07. A rebuilt wheel listed nine guided tools and the
      installed stdio server rejected an unknown launch ID as unowned.
- [x] Rename the canonical guided knowledge tool to `search_knowledge` and
      retain `search_knowledge_cards` as a hidden behavior-preserving alias.
      Custom toolset filters normalize the old name to the canonical one, so
      saved configurations keep working without advertising both choices.
      The complete external-corpus suite passed 1,744 tests on 2026-08-07. A
      rebuilt wheel returned an accepted knowledge card through the canonical
      name in an installed stdio MCP exchange.
- [x] Promote the existing fixed Orbitron render operation to the canonical
      guided `visualize` intent and retain `render_with_orbitron` as a hidden
      behavior-preserving alias. The public contract still accepts only one
      caller-supplied local path and returns either a validated fixed-size PNG
      or an explicit optional-runtime outcome. The complete external-corpus
      suite passed 1,746 tests on 2026-08-07. A rebuilt wheel listed ten guided
      tools and returned `orbitron_unavailable` through `visualize` when the
      optional executable was deliberately absent.
- [x] Select and pin a bounded non-NWChem review queue: two exploratory cases
      each for Molcas, DIRAC, GRASP, QE, and QMCPACK. Exact inputs, outputs,
      scripts, and scalar data now pass containment, size, and SHA-256 checks,
      while scientific expectations and provenance remain explicitly pending.
      The complete external-corpus suite passed 1,748 tests on 2026-08-07.

## Phase 1: lock the useful behavior

- [x] Select the first five real NWChem workflows. The initial audit is in
      [notes/nwchem-behavior-lock-audit.md](notes/nwchem-behavior-lock-audit.md).
- [x] Pin their external inputs and outputs in
      [chemtools/data/reference_cases/nwchem_behavior_cases.json](chemtools/data/reference_cases/nwchem_behavior_cases.json)
      and verify containment, size, and SHA-256 before parser access.
- [x] Select two review candidates each for Molcas, DIRAC, GRASP, QE, and
      QMCPACK. Their artifacts are pinned in
      `chemtools/data/reference_cases/non_nwchem_review_cases.json`; all remain
      exploratory until
      the scientific and provenance questions in the review queue are settled.
- [x] For each selected workflow, record the input files, expected scientific
      facts, expected uncertainty, and expected next action.
- [x] Add or retain golden tests at the application-service boundary.
- [x] Keep only representative MCP golden tests; avoid mirroring every library
      test at the protocol layer. The eleven cases cover every current program,
      generic dispatch, and the guided review, inspection, drafting, recovery,
      and launch boundaries without duplicating the application matrices.
- [x] Add fresh-install smoke tests for importing the package and listing the
      guided tools.
- [x] Decide which direct Python APIs the personal workflow still uses.
      Supported direct Python use goes through focused application, program,
      execution, integration, persistence, and reference modules. A scan of
      the other maintained repositories under `/home/charlie/projects` and
      `/home/charlie/mcps`, plus the shared scripts directory, found no use of
      the 121-name top-level facade or `api*` aggregators. Those broad imports
      are compatibility-only through the final compatibility release; see
      [notes/compatibility-surface-audit.md](notes/compatibility-surface-audit.md).
- [x] Tag one final compatibility release before intentionally breaking unused
      Python or MCP entry points. Release preparation now has a canonical
      `chemtools` MCP server identity, an accurate warning from the
      `chemtools-nwchem` command, and a concise migration note for executable,
      MCP, and Python callers. Focused checks passed 33 tests and the full
      suite passed 1,891 tests against the external corpus. Base and
      DIRAC-extra isolated installs of wheel SHA-256
      `aecf564b9f9677d6de1c153c2c86a245568926780caa4e67eeaacee4b49b3e3e`
      reported `chemtools` through MCP, listed the eleven guided tools, and
      preserved the compatibility imports. Annotated tag `v0.1.0` points to
      commit `c9a4298`; all 1,891 tests also passed from a clean detached
      worktree at that tag.

Workflow-lock evidence (2026-08-07): the application test matrix now pins
`review_input` and `inspect_run` verdicts, uncertainty codes, and next actions
for every input/output pair in the five cases. Recovery planning reuses the
input/output consistency boundary and suppresses automatic drafts for a
confirmed source mismatch. The complete external-corpus suite passed 1,783
tests. Base and DIRAC-extra installed-wheel checks passed for wheel SHA-256
`9af59c51c8a73567ddc3bdcb45f7dcd9f1d366bd6fb2801f01065f904a8f281c`.

Start with five NWChem cases because that is the original and broadest backend.
The initial set comes from `nwchem/hard_cases` in the external corpus:

| Case | Behavior to preserve |
| --- | --- |
| `01_fecn6_lowspin_fragment` | High-spin atomic-guess and low-spin fragment initialization, conditional state comparison, and frontier-orbital interpretation |
| `02_hexaaquairon_swap_chain` | Fragment/swap chain provenance and state comparison; expected filenames remain under scientific review |
| `03_feo_scf_convergence` | Compare converged spin states and retain the energy ordering instead of judging either run alone |
| `04_ferrocene_basis_stepping` | Detect unstable atomic-guess and standalone small-basis trajectories even when they converge; distinguish the controlled projected-and-damped solution |
| `05_crco6_freq_restart` | Distinguish a saddle with an imaginary mode from a verified minimum after displacement and restart |

The file inventory, hashes, current outputs, and unresolved provenance questions
are recorded in [`notes/nwchem-behavior-lock-audit.md`](notes/nwchem-behavior-lock-audit.md).
MCSCF and TCE cases are deferred until this first contract pattern works.

Prefer cases with decisions that depended on expert interpretation. Small H2
and H2O fixtures remain useful parser tests, but they do not substitute for
these behavior-lock cases.

Exit criterion: the calculations that matter in practice are pinned well
enough that layout changes cannot silently remove chemistry knowledge.

## Phase 2: finish the guided interface

The guided names marked as application services are implemented. The remaining
names are provisional
until they are exercised against real prompts and files.

| Intent | Proposed tool | Current source | Task |
| --- | --- | --- | --- |
| Review an input | `review_input` | Guided application service | Keep and refine |
| Inspect a completed, failed, or partial run | `inspect_run` | Guided application service | Keep and refine |
| Retrieve a chemistry rule or known trap | `search_knowledge` | Guided knowledge-card search with the former name retained as a hidden alias | Keep accepted cards as the default and expose curation state in every result |
| Plan a calculation | `plan_calculation` | Guided application service over declared planning providers | Keep strategy separate from input rendering and execution DAGs |
| Create an input | `draft_input` | Guided application service over declared input drafters | Keep the common specification bounded; use recovery contracts for specialized rewrites |
| Plan recovery from a failure | `plan_recovery` | Guided application service over diagnosis and recovery providers | Keep read-only and require explicit target state for state-changing interpretations |
| Compare calculations | `compare_runs` | Guided application service | Keep the energy ordering conditional on explicit comparability evidence |
| Launch an approved calculation | `launch_run` | Guided application service over declared launch-plan providers | Keep the two-call approval token bound to the exact input and rendered plan |
| Monitor an owned calculation | `monitor_run` | Guided application service over owned execution records and declared progress providers | Keep arbitrary PIDs, scheduler IDs, and artifact paths outside the guided contract |
| Inspect or render a structure | `visualize` | Guided Orbitron rendering with the former name retained as a hidden alias | Keep bounded local-path operations |
| Find a relevant validated case | `find_reference_case` | Packaged reference manifests | Default to validated cases; require explicit status for exploratory curation candidates |

- [x] Write one realistic prompt and expected tool choice for every retained
      intent.
- [x] Implement missing intents through application services, reusing current
      program functions. `find_reference_case` now searches only packaged,
      schema-validated manifest metadata. It returns required artifact paths,
      sizes, and SHA-256 pins while keeping scientific status separate. Retrieval
      defaults to `validated_reference`; exploratory and shelved cases require
      an explicit status. It does not scan or open the external corpus.
- [x] Return normalized assessment, evidence, domain findings, uncertainty,
      and next actions from each guided analysis tool. `review_input`,
      `inspect_run`, `compare_runs`, and `plan_recovery` share the versioned
      `assessment`, `evidence`, `uncertainty`, and `next_actions` envelope.
      Domain findings remain at documented evidence paths instead of being
      copied into an ambiguous generic list. Inspection now adds the common
      `action` field while retaining provider `tool` and `params` metadata
      during the compatibility window. The complete external-corpus suite
      passed 1,794 tests, and the installed MCP exchange pinned the normalized
      action for wheel SHA-256
      `576fa1263ddc60ca1aa51018e4cf346895b3a969a301a7d23e15e3f72363b9ea`.
- [x] Mark read-only and state-changing tools with accurate MCP annotations.
- [x] Add output schemas for guided tools where the result is stable enough to
      promise publicly. The SDK path now returns identical
      `structuredContent` and serialized text. Each of the eleven contracts
      pins its existing version and stable top-level fields while leaving
      nested scientific evidence open for compatible additions. Structured
      boundary errors have a schema alternative and retain plain text for old
      clients. The installed official client listed and called all eleven
      tools, validating every result. The external-corpus suite passed 1,843
      tests. Base, DIRAC-extra, and isolated user-site checks passed for wheel
      SHA-256
      `b23849a333125cad1301cb7f39bf0e3f2cb713822213193dfa4b38708a1d3584`.
- [x] Make the guided surface the default after it covers the personal
      workflows selected in Phase 1. The five pinned NWChem workflows pass the
      guided behavior lock against the external corpus.
- [x] Keep low-level tools behind explicit program and developer toolsets.
      `--toolset developer` and `--toolset full` select the complete surface;
      an unset toolset selects the eleven guided tools.

      Evidence (2026-08-07): all five NWChem behavior-lock workflows passed
      through the guided application APIs, and the complete external-corpus
      suite passed 1,828 tests. Base, DIRAC-extra, and isolated user-site
      installs of wheel SHA-256
      `9fa3be69e1b222d0526577c12e44d1c272fc73a37c819879615ef7f77e3edf0e`
      returned the eleven-tool default, loaded the packaged manifests, found
      two explicitly requested exploratory NWChem cases, and retained the
      approval-gated launch and owned-monitoring behavior.

Exit criterion: ordinary work does not require exposing the full low-level
registry to the model.

## Phase 3: simplify ownership and composition

- [x] Split the 1,106-line guided MCP module along its existing ownership
      boundary. `guided.py` now contains only the eight runtime adapters, while
      `_guided_definitions.py` owns their public descriptions, input schemas,
      and annotations. Tool names, ordering, schemas, handlers, and results are
      unchanged; the generated inventory remains byte-for-byte current. The
      complete external-corpus suite still passed 1,748 tests, and a rebuilt
      wheel loaded the separated module and returned the same ten guided tools.
- [x] Remove inward compatibility-facade imports from `core/eval.py`, the
      generic MCP module, and every program package. Callers now import their
      focused owners, while lazy imports remain only where module cycles
      require them. The generic module no longer changes `sys.path` to prefer
      a source checkout. Documentation and basis accessors resolve data from
      their owning modules. An AST contract scans every core and program
      module, and exact alias tests protect the public `chemtools`,
      `api_input`, and `api_strategy` exports. `_nwchem_base.py` remains a
      separate MCP compatibility facade for handler-family migration. The
      complete external-corpus suite passed 1,757 tests on 2026-08-07. A
      rebuilt wheel verified the public aliases, package data, and unchanged
      ten-tool guided surface outside the checkout.
- [x] Move remaining scientific decisions out of MCP handlers and into program
      providers or application services.
  - [x] Move the legacy NWChem next-action table into
        `programs/nwchem/strategy/legacy_next_actions.py`. The old MCP module
        is now an exact import alias for direct Python callers.
  - [x] Move DIRAC open-shell occupation analysis, single-run summaries, and
        spinor filtering into the DIRAC package. Tests pin the 1e-6 energy
        comparison and the occupied-spinor cutoff.
  - [x] Make QE-to-QMCPACK readiness reduction program-owned and use the same
        status precedence for all fourteen MCP inspection responses.
  - [x] Move Molcas task selection for orbital extraction, active-space
        analysis, CASPT2 review, and character-based swap suggestions into
        its parser and strategy modules. The backend parser and MCP tool now
        share one orbital-selection implementation.
  - [x] Move generic geometry normalization and inspection into the
        application layer, including element-key normalization and bohr to
        angstrom conversion.
  - [x] Review the residual low-level handlers listed in
        `notes/mcp-scientific-ownership-audit.md`; keep transport validation
        and compatibility translation in MCP, and move only parser or
        scientific policy that still remains there.

      The final pass moved generic recovery source agreement, Molcas geometry
      and RASSI selection, NWChem recovery-mode aggregation, multiplicity
      inference, and SCF directive rendering below MCP. The remaining handler
      branches are transport limits, optional-argument checks, execution
      ownership, or legacy response formatting.

      Evidence (2026-08-07): the external-corpus suite passed 1,816 tests and
      the generated tool inventory remained current. Installed-wheel evidence
      is recorded below after each package rebuild.
- [x] Remove imports from `core` and program packages through the top-level
      compatibility facade.
- [x] Replace module-global MCP filters and execution-service state with one
      composition object created by the CLI. `ServerState` now carries mode,
      program and tool filters, and a process-owned execution service through
      `serve`, request gating, and handler dispatch. Direct Python setters use
      one fallback compatibility state. State-isolation tests pin distinct
      execution ownership, all 77 execution and monitoring tests pass, and the
      complete external-corpus suite passed 1,788 tests on 2026-08-07. A clean
      installed-wheel check passed with the unchanged ten-tool guided surface
      for wheel SHA-256
      `b82aaa95d9caaf9c10d4e2feef7e7ccbea1b833bfad0adab8472aabf39eaac55`.
- [x] Generate or colocate each guided tool's schema with its handler so names,
      schemas, and implementations cannot drift independently. The eight
      separated contracts now bind to handlers by the existing
      `_handle_<tool name>` convention and fail on missing, duplicate, or
      unmatched registrations. `visualize` and `search_knowledge` already
      colocate definitions and handlers in their owning modules. The public
      order, definitions, and ten-tool preset are unchanged. The complete
      external-corpus suite passed 1,792 tests, and a clean installed-wheel
      check passed for wheel SHA-256
      `e571597b00bfd343bd0f082b1c4a01c54fc2a6d8a1ea3f1f34d29b44ebe728d8`.
- [x] Replace the hand-written protocol loop with a pinned official MCP SDK
      after the guided contract is stable. This now precedes guided output
      schemas because the SDK migration must own protocol negotiation,
      `structuredContent`, and backwards-compatible serialized text together.
      Chemtools now pins `mcp==2.0.0`, requires Python 3.10 or newer, and uses
      the low-level SDK server so the existing registry, filters, and exact
      input schemas remain authoritative. The SDK owns stdio framing and
      negotiation; successful results include the same dictionary as
      `structuredContent` and compact JSON text. The external-corpus suite
      passed 1,840 tests. Base, DIRAC-extra, and isolated user-site checks all
      passed against the installed command and official client for wheel
      SHA-256
      `39083a7b612c23fbb0fe0924d6f6351d12f2f073084e8e2f4924979e83ba357f`.
- [x] Preserve the program backend, artifact, execution, and persistence
      boundaries already established by the accepted ADRs.
  - [x] Move case evaluation from `core/eval.py` to
        `application/evaluation.py` and leave exact compatibility imports.
  - [x] Move the version 1 render and launch implementation from
        `core/runner.py` to `execution/legacy_runner.py`; migrate every
        first-party caller and retain the old path as an import-only facade.
  - [x] Enforce package direction with AST tests: internal layers cannot import
        MCP, core implementation cannot depend outward, and program packages
        cannot depend on application or reference layers. Execution cannot
        import program packages, and persistence can depend only on core and
        persistence modules.
  - [x] Move shared SQLite schema, run, artifact, and launch stores into
        `persistence`; move the combined run services and legacy artifact
        projection into `application`; retain exact compatibility imports at
        the old paths.
        Evidence (2026-08-07): 162 focused persistence and execution tests
        passed, followed by all 1,839 tests with the external corpus. The tool
        inventory remained current. The rebuilt wheel passed base, DIRAC-extra,
        and isolated user-site checks with SHA-256
        `9df0894884238f1fe247486960ae8f536e0ad52f46175823febce94ac5a64374`.
  - [x] Remove the NWChem dependency from `execution/legacy_status.py` by
        injecting the existing progress reader from
        `programs/nwchem/legacy_status.py`. Generic and NWChem response tests
        pin both sides of the boundary. The focused boundary and monitoring
        sets passed 45 tests, followed by all 1,842 tests with the external
        corpus; the tool inventory remained current. The rebuilt wheel passed
        base, DIRAC-extra, and isolated user-site checks with SHA-256
        `2ff2781e2e7b128abeb4bd9d1878f002ec37c42c9d7d8f2f0941fa82d424a212`.

Exit criterion: dependencies point from MCP to application services to core
contracts, with concrete program and execution adapters composed at startup.

## Phase 4: remove obsolete surfaces

- [x] Remove the NWChem evaluation, bundled-documentation, and forum-search
      handlers from the `_nwchem_base.py` wildcard namespace. The family now
      imports its focused owners directly and loads without importing the
      broad base module. NWChem-specific tool registration lives in one small
      shared module used by all handler families.
- [x] Remove the NWChem text and binary output-parsing handlers from the
      `_nwchem_base.py` wildcard namespace. The family imports output, input,
      strategy, and binary owners explicitly, retains the shared next-action
      projection, and loads without importing the broad base module. With the
      documentation-family extraction, the complete external-corpus suite
      passed 1,761 tests on 2026-08-07, and a rebuilt wheel preserved the
      guided MCP and compatibility-alias checks.
- [x] Remove the NWChem input, scientific-analysis, and job-management
      handlers from the `_nwchem_base.py` wildcard namespace. Package-aware
      basis-path resolution also replaces the base module's source-tree
      `sys.path` mutation. All five handler families now use direct owners.
- [x] Compose NWChem tools through `_nwchem_provider.py` in the built-in
      catalog. Normal MCP startup registers the five focused families and
      schemas without importing the legacy `nwchem.py` aggregator or
      `_nwchem_base.py`; both remain available only for Python compatibility.
      The complete external-corpus suite passed 1,762 tests on 2026-08-07.
      An installed-wheel check confirmed the focused provider path, public
      aliases, package data, and unchanged ten-tool guided surface.
- [x] Inventory imports and configuration files that still use legacy CLI,
      Python, and MCP names. Inventory schema version 2 separates 328
      canonical tools, 9 advertised legacy tools, 15 hidden MCP aliases, 2
      executable aliases, and 7 Python import shims. Maintained MCP examples
      now launch `chemtools`; removal blockers are recorded in
      [notes/compatibility-surface-audit.md](notes/compatibility-surface-audit.md).
      The complete external-corpus suite passed 1,763 tests on 2026-08-07,
      and the installed-wheel check retained the focused NWChem provider.
- [x] Migrate first-party NWChem recovery actions from three hidden aliases to
      canonical tool names. State and SCF recommendations now pass an explicit
      `mode` to `suggest_nwchem_recovery`, and review recommendations use
      `analyze_nwchem_case`. An AST contract rejects deprecated names in
      first-party recommendation fields, and behavior tests pin both mode
      translations. The complete external-corpus suite passed 1,767 tests on
      2026-08-07. Base and DIRAC-extra installed-wheel checks passed for wheel
      SHA-256 `7030d4418f91ad1a61cb5c49d40f40002f5e25b8d74c0fc31de60f4924005235`.
- [x] Replace the hidden alias tuple table with the validated metadata registry
      required by ADR 004. Registration rejects missing targets, canonical
      collisions, alias chains, and availability broader than the target.
      Dispatch, custom toolset normalization, and inventory generation use the
      registry while `_TOOL_ALIASES` remains as a Python compatibility view.
      Historical schemas and effects stay explicitly unverified instead of
      being inferred from current handlers. The complete external-corpus suite
      passed 1,779 tests on 2026-08-07. Base and DIRAC-extra installed-wheel
      checks passed for wheel SHA-256
      `0c6bd08dcacbb44768e54ffd922b993d711a3c36397e2d808c6f19f2603ed6c7`.
- [ ] Remove hidden aliases after the final compatibility release and migration
      window.
- [ ] Remove the NWChem wildcard and dynamic `__getattr__` compatibility shim.
- [ ] Remove legacy runner-profile and status adapters after named targets cover
      the retained workflows. See
      [notes/legacy-execution-adapter-audit.md](notes/legacy-execution-adapter-audit.md).
  - [x] Move profile loading and typed target conversion into the canonical
        `execution/profiles.py` owner. Keep `execution/legacy_profiles.py` as
        an exact import facade with no first-party runtime caller. Focused
        profile and execution checks passed 127 tests, followed by all 1,854
        tests with the external corpus. An installed copy of wheel SHA-256
        `0d25226a195fccac80f88bb3dd5ad5a6744679e43606ee98774d4b197b938e23`
        loaded both paths as identical objects, negotiated MCP `2025-11-25`,
        listed the eleven guided tools, inspected the representative NWChem
        output, and prepared an approval-gated launch without writing output.
  - [x] Prepare guided NWChem plans without `render_calculation_run`. The
        provider now merges version 1 resource settings, expands the retained
        profile context, and builds the typed plan directly. It produced the
        same prepared plan and executor rendering as the former path across
        all seven supported bundled local and Slurm profiles. Focused launch,
        model, and boundary checks passed 30 tests, followed by all 1,855 tests
        with the external corpus. Installed wheel SHA-256
        `f6fbc933a82c3e80ed5c47c0c2c6417316799c7dc784415bd064a714a105488f`
        negotiated MCP `2025-11-25`, listed the eleven guided tools, inspected
        the representative output, and prepared an approval-gated launch.
  - [x] Retain low-level NWChem, QE, QMCPACK, Molcas, DIRAC, and GRASP launch
        calls behind explicit program or developer toolsets. QE and QMCPACK
        have no guided execution replacement, and the Molcas, DIRAC, and GRASP
        adapters still use their scheduler wrappers for previews or unowned
        status. Remove a call only after a guided provider passes accepted
        reference cases or the owner explicitly drops execution for that
        program. See
        [notes/low-level-execution-retention-audit.md](notes/low-level-execution-retention-audit.md).
  - [x] Move compatibility output archival into the focused
        `execution/legacy_archive.py` owner. All six program application
        adapters import that owner directly, while `legacy_runner.py` retains
        exact compatibility imports for old Python callers. Execution and
        import-boundary checks passed 69 tests, followed by all 1,856 tests
        with the external corpus. Installed wheel SHA-256
        `63e6dd6e97a688293c6c37ba8ab8ae417405b5fc96c25e90c7bf5c742338158e`
        preserved both archive compatibility imports and passed the guided MCP
        exchange.
  - [x] Move local hardware and scheduler partition discovery into
        `execution/resource_inspection.py`. The generic resource tool and
        NWChem preflight import that focused owner, while `legacy_runner.py`
        retains exact compatibility imports. Focused resource, workflow, MCP,
        and import-boundary checks passed 69 tests, followed by all 1,859 tests
        with the external corpus. Installed wheel SHA-256
        `0fc350cf33074453e3dc945b4fe4da32f908c30f7eaa59d92c38b94a8d55a416`
        included the focused module, preserved the old import identities, and
        passed the guided MCP exchange.
  - [x] Decide whether arbitrary unowned PID, scheduler-ID, PBS, and LSF
        inspection remains a direct Python workflow. Keep read-only file
        inspection and Slurm job attachment through an explicit profile and
        job ID. Retire arbitrary PID, PBS, and LSF inspection after the final
        compatibility release and migration window. Keep cancellation limited
        to launches owned by the execution service; see
        [notes/unowned-status-scope-audit.md](notes/unowned-status-scope-audit.md).
  - [ ] Remove legacy response projection after its six low-level program
        adapters leave or version their response contracts.
- [x] Remove `chem-agent-package/` without preserving its hard-coded paths or
      obsolete tool inventory. The runtime/client audit found no maintained
      caller and no unique implementation. With owner approval, all 16 tracked
      files were deleted together. Its three useful NWChem policies remain as
      explicit draft knowledge cards. See
      [notes/chem-agent-package-knowledge-audit.md](notes/chem-agent-package-knowledge-audit.md).
      Focused knowledge and plugin checks passed 36 tests, and the complete
      external-corpus suite passed 1,852 tests on 2026-08-07.
- [x] Delete superseded planning prose and source comments that describe
      completed migrations rather than current behavior. Runtime Python no
      longer contains the audited phase labels, migration notes, planned-file
      claims, or TODO comments. The module map now records current ownership
      and six remaining compatibility seams instead of the completed
      phase-by-phase migration instructions. Focused ownership, inventory,
      knowledge, and plugin checks passed 97 tests, and the complete
      external-corpus suite passed 1,852 tests on 2026-08-07.
- [x] Regenerate the tool inventory and update all setup examples. Inventory
      schema 3 is checked against the live registry, README totals now follow
      the generated counts, and the maintained workstation and Stampede3
      examples use the canonical eleven-tool guided surface. Personal account
      and executable paths were removed from the Stampede3 configuration.
      Regression tests pin the canonical command, guided workflow names, live
      counts, and absence of known personal path prefixes. The complete
      external-corpus suite passed 1,845 tests on 2026-08-07.

Exit criterion: every compatibility module has a known active caller, and
modules without one are gone.

## Phase 5: portable packaging

- [x] Define minimal base dependencies and program-specific optional extras.
- [x] Build a wheel and install it into a clean environment.
- [x] Verify analysis-only use without chemistry executables or scheduler
      access.
- [x] Verify optional HDF5 and companion-science features through their extras
      or configured external environment.
- [x] Provide one local profile example and one Slurm profile example without
      machine-specific absolute paths.
- [x] Document installation, `chemtools --show-mode`, guided MCP setup, runner
      configuration, and troubleshooting in one short path.
- [x] Test installation and one real workflow on a coworker's machine or clean
      user account.

Exit criterion: a new user can install the package, connect an MCP client, and
inspect a representative output without editing the source tree.

Exit evidence (2026-08-07): the rebuilt wheel passed base and DIRAC-extra
virtual-environment installs plus an isolated user-site install with an empty
home and config directory. The installed stdio server negotiated MCP
`2024-11-05`, returned the eleven guided tools, inspected the external ferrocene
solution output, prepared an approval-gated NWChem launch without creating
output files, and rejected an unknown monitoring identifier as unowned through
`tools/call`. The current wheel also exercised the canonical
`search_knowledge` name, packaged reference search, and the optional-runtime
failure path for `visualize` through the installed server. Its SHA-256 is
`9fa3be69e1b222d0526577c12e44d1c272fc73a37c819879615ef7f77e3edf0e`.

## Phase 6: optional plugin

- [x] Create a thin plugin only after the guided MCP surface is stable.
- [x] Bundle focused skills for input review, run inspection, calculation
      planning, and monitoring.
- [x] Configure the plugin to launch the guided `chemtools` MCP command.
- [x] Keep installation of the Python package explicit and independently
      testable.
- [x] Test direct and indirect prompts, follow-ups, unsupported requests, and
      approval behavior for execution tools. The plugin is installed as
      `chemtools@personal` from `~/.agents/plugins/marketplace.json`, using the
      ignored repository-local `venv/`. Separate Codex processes selected the
      inspection skill for direct and indirect requests, retained exact paths
      across an inspection-to-recovery follow-up, refused arbitrary Slurm
      cancellation without a tool call, and stopped a launch preparation at
      Codex's approval boundary without retrying or writing files. See
      [notes/plugin-fresh-session-evaluation.md](notes/plugin-fresh-session-evaluation.md).

Plugin evidence (2026-08-07): `plugins/chemtools` contains only the plugin
manifest, one guided stdio MCP configuration, four concise workflow skills,
their UI metadata, and the prompt-routing contract. It does not contain Python
scientific code or package dependencies. The official plugin validator and all
four skill validators passed. An installed `chemtools --toolset guided`
command listed the exact eleven intended tools, focused plugin tests passed,
and the complete external-corpus suite passed 1,849 tests.

Exit criterion: the plugin improves installation and tool selection without
duplicating scientific logic or becoming the only supported distribution.

## Explicit deferrals

- Public hosted MCP service and multi-user authentication.
- Web UI or MCP Apps UI.
- Public plugin-directory submission.
- Dynamic third-party chemistry backends.
- Separate repositories or distributions for each supported program.
- Observability infrastructure beyond useful local logs and execution records.

## Definition of done

- The default model-facing surface contains at most twelve intent-level tools.
- Representative personal workflows retain pinned scientific facts and useful
  next actions.
- The complete default test suite passes from a clean checkout.
- Package installation and guided MCP startup work outside the source tree.
- No setup documentation contains checkout-specific absolute paths.
- Low-level tools, if retained, require an explicit developer configuration.
- A coworker can install and use the package from one concise setup guide.

## Next implementation slice

- [x] Create this focused task list.
- [x] Correct stale top-level tool counts and the misleading guided-test name.
- [x] Select the first five real NWChem workflows for the behavior lock.
- [x] Map those workflows to `review_input`, `inspect_run`, `draft_input`, and
      `plan_recovery` responses. The exact application matrix is recorded in
      [notes/guided-nwchem-workflow-map.md](notes/guided-nwchem-workflow-map.md),
      including the source-consistency block for the mismatched hexaaquairon
      swap stage.
- [x] Add the first application-level `compare_runs` contract and pin the FeO
      triplet/quintet energy ordering without claiming a ground state from
      energy alone.
- [x] Preserve converged NWChem SCF excursions as diagnosis warnings and pin
      the failed, standalone small-basis, and controlled ferrocene paths.
- [x] Pin the Cr(CO)6 saddle-to-minimum verdict transition and count only
      significant imaginary modes in normalized NWChem evidence.
- [x] Correct unrestricted SOMO labeling and pin the Fe(CN)6 high-spin to
      low-spin comparison, including the Fe-centered t2g frontier character.
- [x] Add a read-only `draft_input` application contract for NWChem and
      OpenMolcas, including deterministic NWChem names and immediate linting.
- [x] Record current guided routing and recovery gaps for all five cases in
      [notes/guided-nwchem-workflow-map.md](notes/guided-nwchem-workflow-map.md).
- [x] Add a read-only `plan_recovery` contract. Pin target-state rebuild for
      Fe(CN)6 and plus/minus imaginary-mode candidates for Cr(CO)6 without
      writing files.
- [x] Repair the NWChem diagnosis-recovery provider boundary so it no longer
      imports a nonexistent facade function.
- [x] Add representative guided drafting and recovery MCP golden cases.
- [x] Add optional stability hardening for converged NWChem runs with severe
      SCF excursions. Reuse a declared converged checkpoint for simple inputs,
      keep smaller-basis projection as a reviewed fallback, and refuse to
      rewrite multi-stage or fragment inputs automatically.
- [x] Add the two-call guided `launch_run` contract with exact approval,
      conflict refusal, analysis-only preparation, owned execution, and an
      initial NWChem launch provider.
- [x] Add guided `monitor_run` for process-owned launches with normalized
      process or scheduler state, recorded artifact metadata, scientific
      progress, explicit uncertainty, and no cancellation or restart effects.
- [x] Advertise the intent-level `search_knowledge` name and move the former
      card-storage name into the hidden compatibility alias registry.
- [x] Replace the separate module-global MCP filters and execution service with
      one CLI-created `ServerState`, while retaining the old direct Python
      setters against a single fallback compatibility state.
- [x] Bind every separated guided definition to exactly one handler and retain
      the already-colocated `visualize` and `search_knowledge` contracts.
- [x] Bring the maintained setup examples in line with the guided default,
      remove personal paths from the Stampede3 example, and bind README counts
      to the generated inventory. The external-corpus suite passed 1,845 tests.
- [x] Add the thin `plugins/chemtools` bundle with four guided workflow skills,
      an installed-command MCP configuration, plugin and skill validation, and
      a prompt-routing evaluation contract. The external-corpus suite passed
      1,849 tests.
