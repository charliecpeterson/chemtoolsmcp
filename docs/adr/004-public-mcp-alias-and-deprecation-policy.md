# ADR 004: Public MCP alias and deprecation policy

Status: Accepted

Date: 2026-07-30

Chemtools will expose one canonical MCP tool definition for each operation.
Renamed tools remain callable through a validated compatibility registry but
are omitted from `tools/list`. An alias must preserve the old request,
behavior, result, error, and effect contract. If it cannot, the old tool
remains a separate deprecated handler until its removal boundary.

Aliases will not disappear during the phase that introduces their
replacement. Removal requires a declared breaking release, compatibility
evidence, migrated first-party callers, and release notes.

## Context

Chemtools currently has two MCP alias mechanisms.

The dispatcher contains 13 hidden aliases:

- They are omitted from `tools/list`.
- A direct `tools/call` resolves them to a canonical handler.
- Four aliases translate arguments before dispatch.
- They inherit mode and program checks from the target.
- They have no stored old input schema, lifecycle metadata, result adapter,
  call warning, or removal version.

Nine more legacy NWChem-prefixed registry and workflow tools are ordinary
tool definitions. They appear in `tools/list`, have separate handlers and
schemas, and describe themselves as legacy tools. Most call the same shared
functions as their generic replacements. Some also preserve behavior, such
as adding `program="nwchem"` when the caller omits it.

The generated inventory reports 265 public definitions and 13 aliases. The
more accurate current breakdown is:

| Surface | Count |
| --- | ---: |
| Canonical definitions | 256 |
| Advertised legacy definitions | 9 |
| Hidden dispatcher aliases | 13 |
| Total callable names | 278 |

The difference matters. An assistant sees nine obsolete choices, while the
inventory omits them from its alias count. There is no single place to answer
which name is canonical, how long an alias remains supported, or whether its
old schema is still tested.

The `chemtools-nwchem` command is another compatibility name. Its help text
says it will remain for one release, but no version marks the start or end of
that period. The Python import shim at `chemtools.mcp.nwchem` has similar
compatibility intent but is not part of the MCP tool namespace.

The initialize handler initially echoed any client-requested MCP protocol
version. It did not check whether Chemtools supported that version. Phase 0
now pins support to `2024-11-05` and returns that version when a client
requests an unsupported version. This resolves the prerequisite for
protocol-dependent deprecation metadata without claiming support for later
protocol revisions.

The current tool surface will change again when guided operations such as
`review_input`, `inspect_run`, and `launch_run` replace narrow public choices.
Without a policy, each cleanup can either break saved agent configurations or
leave another permanent duplicate in `tools/list`.

## Decision

### One registry owns every public tool name

The built-in catalog from ADR 001 will provide canonical tool definitions.
A separate compatibility registry will own all non-canonical names:

```python
@dataclass(frozen=True)
class ToolAlias:
    name: str
    target: str
    input_schema: Mapping[str, JsonValue]
    translate_arguments: ArgumentAdapter
    translate_result: ResultAdapter | None
    availability: CompatibilityAvailability
    effects: ToolEffects
    deprecated_since: str
    remove_after: str | None
    reason: str
```

The registry includes the current 13 dispatcher aliases and the nine
advertised NWChem legacy tools. The latter become hidden aliases after their
compatibility adapters and golden cases exist.

Canonical definitions and aliases cannot share a name. An alias points
directly to a canonical tool, never to another alias. Registration fails for
a missing target, duplicate name, alias chain, or cycle.

The registry also validates that:

- The old input schema is present and valid.
- Argument and result adapters are deterministic and perform no I/O.
- The alias records the program, target, permission, and toolset restrictions
  of the old name.
- Alias availability preserves the old contract exactly and is no broader
  than the canonical target's availability. Registration fails if the target
  cannot cover that old scope.
- Alias effects match both the old contract and the canonical target.
- A compatibility test exercises the old name through `tools/call`.

The registry is metadata and small adapters. It will not become a second tool
framework.

### The public contract includes behavior and effects

A tool's public contract includes:

- Tool name.
- Input schema, required fields, defaults, and accepted units.
- Operation semantics and program ownership.
- Output envelope, fields, units, and selection rules.
- Error shape and error conditions.
- Read, write, execution, and cancellation effects.
- Target, program, and toolset visibility requirements.

Descriptions may clarify the contract but cannot redefine it silently.

An alias is valid only when a caller using the old contract receives the same
observable behavior. Argument translation may rename fields, insert historical
defaults, or select a canonical operation mode. A result adapter may preserve
an old response shape. If preserving the result would hide a scientific,
security, or state-change difference, the names represent separate operations
and must remain separate handlers.

Examples from the current surface:

- `list_nwchem_runs` may target `list_runs` only if its adapter inserts
  `program="nwchem"` when absent.
- `suggest_nwchem_scf_fix_strategy` may target
  `suggest_nwchem_recovery` because its adapter selects `mode="scf"`.
- A future `launch_nwchem_run` alias may target `launch_run` only after the
  target adapter preserves execution permission, default target selection,
  artifact expectations, and result fields.

Changing a read-only tool into a state-changing operation cannot be handled
as an ordinary alias.

### Canonical names follow operation ownership

Tool names use lowercase `snake_case`, matching the existing surface and the
MCP character guidance. Names remain case-sensitive.

Program-neutral application services use intent names:

- `review_input`
- `inspect_run`
- `plan_calculation`
- `launch_run`
- `monitor_run`
- `compare_runs`
- `visualize`
- `search_knowledge`
- `find_reference_case`

Program-specific developer tools retain the program in the name when their
semantics or artifact format belong to that program, such as
`parse_nwchem_movecs` or `validate_molcas_caspt2_setup`.

A tool is not renamed only to make its wording more consistent. Renaming is
reserved for a real ownership correction, a misleading public contract, a
collision, or replacement by a higher-level operation. Version suffixes such
as `_v2` are avoided unless incompatible tool contracts must coexist during a
migration.

### Canonical tools are the only names advertised

`tools/list` returns canonical tool definitions only. Hidden aliases remain
callable for clients that cached or saved the old name. This matches the
existing behavior of the 13 dispatcher aliases and removes duplicate choices
from the model-facing surface.

Program, target, permission, and toolset filters resolve aliases before
checking the intersection of alias and target availability. A
program-specific alias that points to a program-neutral target therefore
remains program-specific. Toolset configuration also normalizes alias names
to their canonical targets at startup without discarding the alias contract.
An existing
`CHEMTOOLS_TOOLSET=diagnose_nwchem_output` configuration therefore continues
to permit the corresponding canonical operation.

The server does not add a switch that republishes every deprecated alias.
Generated inventory and release notes are the discovery surface for
compatibility names.

### Deprecation warnings stay outside scientific payloads

Calling an alias returns the legacy result shape. Chemtools will not insert a
`warnings` field into chemistry data merely to announce a rename.

The dispatcher emits one warning per alias per server process to stderr and
the configured local MCP log. It records the alias, canonical target,
deprecation version, and earliest removal version. There is no remote
telemetry.

When the negotiated MCP protocol accepts result metadata, the tool result also
contains:

```json
{
  "_meta": {
    "dev.charlespeterson.chemtools/deprecation": {
      "alias": "diagnose_nwchem_output",
      "replacement": "analyze_nwchem_case",
      "deprecated_since": "0.x",
      "remove_after": null
    }
  }
}
```

MCP defines `_meta` on tool results and reserves namespaced keys for
implementation metadata. Clients may ignore custom metadata, so stderr,
inventory, tests, and release notes remain required. The metadata is a notice,
not the compatibility mechanism.

Result metadata is emitted only when the negotiated protocol revision defines
`_meta` on `CallToolResult`. Chemtools currently negotiates only
`2024-11-05`, whose call result does not provide that field, so this revision
uses stderr, local logs, inventory, and release notes without injecting
`_meta`. Supporting a later revision requires its own schema and golden cases
before metadata is enabled.

### Protocol errors follow the negotiated revision

Alias resolution does not blur JSON-RPC errors and tool execution errors. For
the current `2024-11-05` revision, an unknown tool or a malformed
`tools/call` request returns a protocol-level JSON-RPC error. A valid call
whose tool rejects input or fails during its operation returns a tool result
with `isError: true`. An alias preserves the old semantic error shape inside
that tool result, but it cannot convert a malformed request into a successful
protocol response.

Protocol behavior is selected from the version negotiated during
initialization. Chemtools will not return fields from a later schema merely
because a client can ignore them.

### Deprecation and removal are separate states

Aliases move through three states:

1. `callable_deprecated`: hidden from discovery, fully functional, and
   warning on use.
2. `removed_tombstone`: hidden from discovery and returns a specific
   `deprecated_tool_removed` execution error naming the replacement.
3. Deleted: the name is unknown to the server.

An alias cannot enter `removed_tombstone` until all of these are true:

- The canonical replacement has shipped in at least two tagged releases.
- Golden tests prove the alias contract or document an intentional
  incompatibility.
- Repository examples, documentation, toolsets, and maintained MCP
  configurations use the canonical name.
- The generated inventory declares the planned removal version.
- Release notes identify the removal as breaking.
- No known first-party usage remains, based on repository search and optional
  local MCP logs.

Removal occurs only in a release explicitly marked as a compatibility
boundary. Time passing by itself is not enough. If Chemtools has no tagged
release cadence, the alias stays callable.

A tombstone remains for one tagged release, then may be deleted. A dangerous
or scientifically incorrect alias may be disabled sooner, but the server must
return a specific error explaining the safety reason and replacement.

### Changes are classified before release

The following changes are compatible when tests prove existing calls retain
their meaning:

- Description corrections.
- Accepting an additional optional argument without changing old defaults.
- Accepting a broader input value while preserving prior values.
- Adding a canonical name while retaining the old name as an alias.

The following changes require an alias, a versioned result envelope, or a
declared breaking release:

- Renaming or removing a tool.
- Adding a required argument.
- Changing a default, unit, selection rule, or program scope.
- Removing or renaming a result field.
- Changing a field type or error shape.
- Changing read, write, execution, cancellation, or remote-access effects.
- Narrowing availability under program, target, or permission filters.

Adding result fields is treated conservatively because current golden tests
and external callers may compare exact keys. A versioned result envelope is
preferred when a guided tool needs a materially different shape.

### Inventory is the machine-readable contract ledger

The generated inventory will distinguish canonical definitions from aliases:

```json
{
  "summary": {
    "canonical_tool_count": 256,
    "callable_alias_count": 22,
    "total_callable_name_count": 278
  },
  "aliases": [
    {
      "name": "list_nwchem_runs",
      "target": "list_runs",
      "state": "callable_deprecated",
      "advertised": false,
      "deprecated_since": "0.x",
      "remove_after": null
    }
  ]
}
```

The initial `0.x` values are placeholders until the project chooses the first
tag governed by this ADR. Implementation must replace them with exact package
versions rather than committing fictional history.

For every canonical tool, the inventory records its input schema, output
schema when available, owner, capabilities, effects, and visibility. Stable
schema fingerprints make intentional contract changes visible in review.
Counts in the README come from `canonical_tool_count`, with alias counts shown
separately.

Current hidden aliases lack stored historical schemas. Before consolidating
them, Phase 0 or Phase 4 must recover the last advertised schema from Git
history, documentation, or fixtures. If that evidence is absent, the
inventory marks the alias contract `unverified` instead of inventing one.

### CLI and Python compatibility are tracked separately

`chemtools` is the canonical executable. `chemtools-nwchem` follows the same
versioned deprecation ledger as MCP aliases, but it is reported under
`entrypoint_aliases`, not `tool_aliases`.

The canonical MCP `serverInfo.name` becomes `chemtools` when the guided
multi-program server ships. The old executable remains usable through its
declared compatibility period.

Python import shims such as `chemtools.mcp.nwchem` are recorded under a
separate Python compatibility table. An internal module move does not create
an MCP alias unless a public tool name also changes.

Package version, `serverInfo.version`, inventory version, and deprecation
versions must come from one version source. The policy cannot work while
several constants can drift.

## Migration

The migration is metadata-first:

1. Add `ToolAlias` metadata and validation beside the current alias map.
2. Import the 13 dispatcher aliases without changing dispatch behavior.
3. Add exact alias-call tests for argument translation, result shape,
   protocol versus execution errors, effects, and the intersection of alias
   and target availability.
4. Represent the nine advertised NWChem legacy tools as aliases with their
   current schemas and behavior adapters.
5. Hide those nine names from `tools/list` only after their direct-call golden
   cases pass.
6. Normalize alias names in toolset configuration.
7. Extend the inventory with canonical, advertised-legacy, hidden-alias, and
   total-callable counts during the transition.
8. Keep the Phase 0 supported-version set explicit and add a later protocol
   revision only after its initialization and tool behavior are tested.
9. Add one-per-process warnings. Add namespaced result metadata only for a
   negotiated protocol revision whose `CallToolResult` schema defines it.
10. Move the CLI alias, server identity, and Python shim into their separate
   compatibility sections.
11. Apply the same registry when guided Phase 4 tools replace narrow public
    names.

No alias is removed in this migration.

## Consequences

The model-facing tool list becomes smaller without breaking saved tool calls.
The inventory reports every callable name and stops counting legacy
definitions as canonical tools.

Compatibility code becomes visible and testable. Argument adapters preserve
historical defaults, while result adapters are used only when they can
preserve the old contract honestly.

Aliases may remain longer than a calendar-based cleanup would prefer. That is
the cost of having no external usage telemetry and no fixed release cadence.
The hidden registry keeps that cost out of the assistant's tool-selection
surface.

Schema fingerprints and exact output tests add maintenance work. They also
make a unit change, default change, or new side effect visible before it
reaches a saved agent workflow.

## Alternatives rejected

### Keep aliases as ordinary tool definitions

This preserves discovery for old names but presents duplicate choices to the
assistant and inflates tool counts. The canonical tool should be the only
advertised choice.

### Delete old names when the new tool ships

Saved configurations and clients may call a tool without refreshing
`tools/list`. Removing the old name in the same release turns an internal
cleanup into an avoidable break.

### Keep aliases forever

Permanent aliases still need tests and can preserve unsafe or misleading
behavior indefinitely. A measured removal path is better than either
immediate deletion or an unbounded promise.

### Put deprecation warnings inside every payload

This changes result schemas and contaminates scientific data with transport
lifecycle information. Result metadata, logs, inventory, and release notes
carry the notice without altering the payload.

### Allow aliases to target aliases

Chains make defaults, permissions, and removal order difficult to reason
about. Every alias resolves directly to one canonical operation.

### Use MCP protocol versions as the removal clock

Protocol versions describe transport and feature compatibility, not the
Chemtools release in which a tool name changed. Package versions govern this
policy.

## Protocol references

- [MCP 2024-11-05 tool error handling](https://modelcontextprotocol.io/specification/2024-11-05/server/tools#error-handling)
- [MCP 2024-11-05 version negotiation](https://modelcontextprotocol.io/specification/2024-11-05/basic/lifecycle#initialization)
- [MCP 2025-11-25 tool results](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- [MCP 2025-11-25 `_meta` rules](https://modelcontextprotocol.io/specification/2025-11-25/basic/index#_meta)
- [MCP 2025-11-25 schema](https://modelcontextprotocol.io/specification/2025-11-25/schema)

## Acceptance checks

Phase 0 accepted this decision with these conditions:

- Only canonical tools appear in `tools/list`.
- All 22 current compatibility names move into one validated registry.
- An alias preserves arguments, defaults, results, errors, and effects.
- Program-specific aliases cannot gain the broader scope of a generic target.
- Behavior-changing replacements remain separate handlers.
- Toolset and availability checks resolve aliases consistently.
- Deprecation notices do not alter scientific payloads.
- Protocol and tool execution errors follow the negotiated MCP revision.
- Removal waits for two tagged releases and an explicit breaking boundary.
- Inventory counts canonical tools and aliases separately.
- CLI entry points and Python import shims use separate compatibility ledgers.
