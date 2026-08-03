# Phase 0 ADR review

Review date: 2026-07-30

Disposition: accepted after changes. All eleven contract gaps were resolved
in ADRs 001 through 005, and the cross-document checks passed.

The review checked the ADRs against the live program plugins, registry,
execution code, SQLite schema, generated MCP inventory, package-data rules,
external corpus, module map, and official MCP specifications.

## Resolved findings

The findings are retained below as the review record. The accepted ADR text
now contains each required constraint.

1. **ADR 001, "One catalog owns built-in membership": place the catalog at
   the composition boundary.** The proposed `BuiltinBackendSpec` contains both
   program modules and MCP tool modules. That is valid only if the catalog
   belongs to the CLI composition layer. Putting it in `core` would violate
   the module map's rule that core does not depend on MCP. The ADR must name
   the owning module and state that core receives registered backends without
   importing MCP metadata.

2. **ADR 002, "Programs build launch plans": define final command assembly.**
   `LaunchPlan.argv`, `ProgramInstallation.executable`, and
   `ProgramInstallation.launcher_argv` currently overlap. The text alternates
   between programs building `argv` and executors applying the configured
   launcher. Pick one rule. The smallest clear model is for the program
   adapter to produce program arguments while the executor combines the
   target-owned launcher, executable or container prefix, and those arguments.

3. **ADR 002, execution permission: enforce the gate below MCP and scope
   cancellation.** Hiding and rejecting tools at the MCP dispatcher does not
   protect direct Python callers. The application execution service must
   enforce the same gate. By default, cancellation should accept only a run
   recorded as launched through the same Chemtools registry and target.
   Cancelling an arbitrary PID or scheduler job ID requires a separate,
   explicit policy.

4. **ADR 002, allowed work roots: specify resolved-path containment.**
   Launch, staging, script writes, and output writes must resolve symlinks and
   reject paths outside the selected target's allowed roots. A lexical prefix
   check is insufficient. This check belongs in the application execution
   boundary so MCP and Python callers receive the same behavior.

5. **ADR 003, `RunArtifacts`: include observations in the model.** The ADR
   says each artifact has point-in-time observations and the JSON envelope
   includes them, but the shown `RunArtifacts` type contains only artifacts,
   expectations, and provenance. Add observations to the aggregate or name a
   separate aggregate that owns them.

6. **ADR 003, `ProvenanceRecord`: identify exact output snapshots.** Inputs
   use `ArtifactSnapshotRef`, while outputs are bare artifact IDs. That loses
   the exact bytes produced by a copy, conversion, or merge. Record output
   snapshot references after observing the outputs, or state how each event
   resolves to immutable output observations.

7. **ADR 004, alias availability: preserve the alias's old scope.** The ADR
   says program ownership and availability come from the canonical target.
   That breaks a program-specific alias whose target is generic. For example,
   an NWChem registry alias can resolve to a generic tool and become callable
   under a Molcas-only filter. Store the old program and availability contract
   in `ToolAlias`, then require it to be no broader than the canonical target.

8. **ADR 004, protocol behavior: pin error and metadata rules by negotiated
   version.** Chemtools currently supports MCP `2024-11-05`. That
   specification treats an unknown tool or malformed call as a protocol
   error. The current dispatcher returns several of those as tool execution
   errors. ADR 004 must distinguish protocol errors from tool execution
   errors before requiring aliases to preserve error shape. It must also state
   that deprecation `_meta` is omitted until Chemtools negotiates a protocol
   revision whose `CallToolResult` defines that field. The
   [2024-11-05 tools specification](https://modelcontextprotocol.io/specification/2024-11-05/server/tools)
   and the
   [2025-11-25 schema](https://modelcontextprotocol.io/specification/2025-11-25/schema)
   provide the two relevant contracts.

9. **ADR 005, manifest dimensions: move storage and redistribution to each
   artifact.** The ADR says every artifact has a storage tier, then the
   example assigns one tier to the whole case. A real case can combine a
   committed input with an external checkpoint or output. Put
   `storage_tier`, `redistribution`, source, attribution, and license evidence
   on each artifact. A case-level value may remain only as a default when all
   artifacts share it.

10. **ADR 005, f-block dataset: decide how installed Chemtools can read the
    canonical data.** Phase 5 resolved this issue by moving the one canonical
    copy to `chemtools/data/fblock`, adding typed package-resource access, and
    removing the notes-tree copy.

11. **ADR 005, redistribution evidence: add the fields required by the
    policy.** The text requires attribution and a license basis for
    `redistribution="allowed"`, but the manifest example has only `source`.
    Add explicit attribution, license identifier or terms, and permission
    evidence fields, with validation tied to the redistribution state.

## Worth considering

1. **ADR 001, `file_extensions`:** align its mapping keys with ADR 003's
   artifact roles and program-specific kinds. The current plugins use a mix
   of both.

2. **ADR 002, effect metadata:** `changes_execution_state` is enough for the
   first migration, but file writes remain governed by scattered tool
   arguments. Record that a later effect split may be needed when guided tools
   combine drafting, staging, and launch.

3. **ADR 003, directory observations:** define the algorithm and version in
   any directory manifest hash. A plain hash value is not portable unless
   path ordering, entry types, relative names, and file hashes are specified.

4. **ADR 004, removal evidence:** change "no known local usage remains" to
   "no known first-party usage remains." Saved third-party configurations
   cannot be proven absent without telemetry, and the ADR correctly rejects
   remote telemetry.

5. **ADR 005, changed-during-use detection:** a verified external path can
   change between hashing and an external program opening it. For the initial
   local development corpus, record this as a known limitation. Add a
   post-use observation only if concurrent mutation becomes a real failure
   mode.

## Accuracy checks

- ADR 001's provider table matches the live plugin objects: NWChem wires all
  five providers, Molcas wires parser, drafter, and binary, DIRAC wires parser
  and binary, and GRASP wires only its parser.
- ADR 004's counts match the live registry: 265 advertised definitions, 13
  hidden dispatcher aliases, and 9 advertised legacy definitions. The 22
  compatibility names therefore produce 278 callable names.
- ADR 005's current corpus facts were rechecked. The external corpus contains
  7,953 files and 61,458,493,920 bytes of directory content. The f-block
  atomic library contains 106 files totaling 703,743 bytes; its largest file
  is 460,699 bytes and names 31 elements.
- The server name remains `chemtools-nwchem`, its package version remains
  `0.1.0`, and its only supported MCP protocol revision is `2024-11-05`.
- MCP version negotiation in ADR 004 matches the official
  [lifecycle rule](https://modelcontextprotocol.io/specification/2024-11-05/basic/lifecycle):
  echo a supported requested version, otherwise return another version the
  server supports.

## What is working

The five ADRs divide the problem along real ownership seams. Program
capabilities, execution, artifact history, public compatibility, and
reference curation remain separate decisions, and their migration orders
mostly line up with the module map.

The migrations preserve current behavior while moving one boundary at a
time. That is appropriate for a 265-tool public surface with thin existing
tests.

The scientific distinctions in ADRs 003 and 005 are sound. Provenance does
not imply validity, agreement does not imply scientific truth, and a reviewed
failure cannot become a recommended workflow by being committed to Git.

## Acceptance result

The eleven findings were resolved, followed by these checks:

1. The five ADRs were rechecked against each other and the module map.
2. Each ADR moved from `Proposed` to `Accepted`.
3. The ADR index and project plan were updated.
4. Phase 1 starts with ADR 001's catalog and capability tests.
