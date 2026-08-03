# ADR 001: Optional program capabilities and one built-in backend catalog

Status: Accepted
Date: 2026-07-30

Chemtools will declare program support as operation-level capabilities and
load every built-in program through one explicit catalog. Provider objects may
remain coarse, but their presence will no longer imply support for every
method they contain.

## Context

The current `Program` protocol requires `parser`, `drafter`, and `strategist`.
It allows `binary` and `examples` to be absent. The four implementations do
not match that contract:

| Program | Parser | Drafter | Strategist | Binary | Examples |
| --- | --- | --- | --- | --- | --- |
| NWChem | Yes | Yes | Yes | Yes | Yes |
| Molcas | Yes | Yes | No | Yes | No |
| DIRAC | Yes | No | No | Yes | No |
| GRASP | Yes | No | No | No | No |

Each plugin class initially assigns every provider field to `None`, then wires
some fields after construction. The broad parser protocol creates a second
kind of false support: DIRAC, Molcas, and GRASP parser objects implement
methods that raise `NotImplementedError` for unsupported operations.

Callers cannot answer "does this program support frequency parsing?" from the
type or registry. They infer support from a provider field, call a method and
catch an exception, or bypass the plugin and import program code directly.
Generic MCP handlers currently dereference `plugin.parser` directly for
output, geometry, frequency, trajectory, and thermochemistry operations.

Program membership also has several sources:

- Program package imports register plugin instances as a side effect.
- `chemtools.mcp.modes.KNOWN_PROGRAMS` lists names by hand.
- `chemtools.mcp.dispatch` imports five tool modules and concatenates their
  definition lists by hand.
- MCP decorators attach another program name to each tool.

This makes partial support hard to report truthfully and makes adding Quantum
ESPRESSO or QMCPACK a multi-file registration change.

## Decision

### Capabilities describe operations

Add `ProgramCapability` as a string-valued enum. A capability states that a
backend can complete one public operation with its documented result shape.
It does not state that a provider object exists.

The initial set will cover operations consumed by current generic tools and
core workflows:

| Capability value | Meaning |
| --- | --- |
| `output.parse` | Return the compact cross-program parsed-run shape |
| `output.task_index` | Return task summaries without a full parse |
| `output.geometry` | Extract a geometry snapshot |
| `output.orbitals` | Extract orbital information |
| `output.frequencies` | Extract vibrational frequencies |
| `output.trajectory` | Extract an optimization or dynamics trajectory |
| `output.thermochemistry` | Extract thermochemical quantities |
| `input.parse` | Parse a program input |
| `input.draft` | Draft an input from a declared specification |
| `input.lint` | Check an input without executing it |
| `input.patch` | Apply a structured input change |
| `binary.read` | Read at least one declared binary artifact kind |
| `binary.write` | Write at least one declared binary artifact kind |
| `diagnosis.run` | Produce a chemistry or run diagnosis |
| `diagnosis.recovery` | Suggest ordered recovery actions |
| `resources.estimate` | Recommend program resources for a target profile |
| `progress.inspect` | Summarize an in-progress output |
| `run.consistency` | Compare one explicit primary input with output and related-artifact evidence |
| `examples.read` | List and read curated examples |

New enum members require a real consumer and an implementation. Future
concepts such as visualization, knowledge lookup, and artifact conversion
will not be added until the corresponding service exists.

Capabilities are finer than provider fields because a program may support
geometry but not frequencies, or binary reads but not writes. They remain
coarser than format variants. For example, supported binary kinds belong in
the binary provider's own metadata rather than separate enum members.

### Backends expose optional providers

Replace the current required `Program` shape with a backend object whose
required fields cover identity, declared capabilities, file roles, and
detection:

```python
@dataclass(frozen=True)
class ProgramBackend:
    name: str
    capabilities: frozenset[ProgramCapability]
    artifact_kinds: Mapping[str, ArtifactKindSpec]
    detector: ProgramDetector
    parser: OutputAdapter | None = None
    inputs: InputAdapter | None = None
    binary: BinaryAdapter | None = None
    diagnostics: DiagnosticAdapter | None = None
    resources: ResourceAdvisor | None = None
    progress: ProgressAdapter | None = None
    consistency: RunConsistencyAdapter | None = None
    examples: ExamplesCorpus | None = None
```

These fields group related code. They are not capability declarations. One
object may satisfy several capabilities, which preserves the current compact
implementations and avoids a class for each operation.

`ArtifactKindSpec` names a program-specific artifact kind, its accepted
extensions or fixed filenames, its default artifact roles, and whether its
content is known to be `text`, `binary`, or `unknown`. `unknown` is the safe
default for third-party backends and for a kind that groups files with
different formats. Core code does not treat mapping keys or filename suffixes
as artifact roles or permission to read a file as text. ADR 003 defines the
shared role vocabulary.

The exact provider protocols can stay small and evolve from current call
sites. Phase 1 will not split a working parser merely to match this sketch.

### Registration validates declarations

The registry will validate each backend at its boundary:

1. The backend name is normalized and unique.
2. Every declared capability has a provider with the required callable.
3. Provider fields may exist without advertising every method they happen to
   expose.
4. No generic handler may call an optional operation without requiring its
   capability first.

Duplicate registration will become an error. Silent replacement hides import
order mistakes and makes the catalog unreliable.

The registry will expose:

```python
backend.supports(ProgramCapability.OUTPUT_FREQUENCIES)
backend.require(ProgramCapability.OUTPUT_FREQUENCIES)
```

`require` returns the backend for fluent internal use or raises
`UnsupportedCapabilityError`. That error carries the program, requested
capability, and sorted available capabilities. The MCP boundary converts it
to a stable error object:

```json
{
  "error": "unsupported_capability",
  "program": "grasp",
  "capability": "output.frequencies",
  "available_capabilities": ["output.orbitals", "output.parse", "output.task_index"]
}
```

`NotImplementedError` will no longer represent normal program support
differences at the MCP boundary. It remains appropriate inside a provider for
an optional variant that the provider's own metadata excludes.

### One catalog owns built-in membership

Add an explicit catalog for the built-in backends:

```python
@dataclass(frozen=True)
class BuiltinBackendSpec:
    name: str
    program_module: str
    backend_attribute: str
    tools_module: str
    definitions_attribute: str
```

The catalog will contain one record for each built-in program. It will drive:

- Program loading and registry population.
- Known program names used by CLI validation.
- Program tool-module imports.
- Tool-definition aggregation.

This catalog is composition metadata. It belongs at the CLI/MCP composition
boundary, either in `chemtools.mcp.cli` or an adjacent composition module,
because its records refer to both program packages and MCP tool modules.
Composition code loads a backend and passes the resulting object to the core
registry, then loads that backend's MCP definitions. `chemtools.core` accepts
backend objects and never imports the catalog or MCP metadata. A headless
Python caller may register an explicit set of backend objects without loading
any MCP modules.

Generic tools are not a program backend. Their definition provider remains a
single explicit entry beside the backend loop.

The catalog uses fixed import paths committed in this repository. Dynamic
entry points, filesystem discovery, and third-party backend loading are
deferred. Chemtools needs one dependable built-in registration path before it
needs an extension mechanism.

Tool declarations remain in their current MCP modules during Phase 1. This
ADR removes repeated membership lists; it does not generate tools from
capabilities or move MCP schemas into program backends.

## Migration

Phase 1 will make this change in small behavior-preserving steps:

1. Add `ProgramCapability`, `UnsupportedCapabilityError`, the backend data
   shape, and validation tests without removing the current protocol.
2. Add the built-in catalog at the composition boundary and point it at the
   current program and MCP tool modules. Keep core registry imports
   independent of MCP metadata.
3. Derive CLI program names and dispatch aggregation from the catalog.
4. Wrap the four current plugins as backends and declare only operations
   covered by tests.
5. Gate generic handlers through `supports` or `require`, then add exact
   unsupported-capability tests.
6. Remove import-time self-registration after the catalog owns loading.
7. Remove the old broad `Program` protocol and placeholder provider fields
   after all call sites use the new boundary.

Compatibility aliases, MCP tool names, response schemas, and current mode
filtering remain unchanged during this migration. The generated inventory and
golden MCP cases must pass after each step.

## Consequences

Adding a backend will require one catalog record, one backend object, and only
the providers it actually supports. Quantum ESPRESSO can begin with detection,
output parsing, input parsing, and input drafting. QMCPACK can begin with
artifact parsing and diagnosis without pretending to support molecular
geometry or frequency operations.

Generic handlers gain a predictable refusal path. An assistant can inspect
available capabilities before choosing a tool, and an unsupported request
will name the missing operation.

Capabilities add a declaration that must remain synchronized with provider
code. Registration validation and exact per-backend tests pay that cost at
startup and in CI.

The built-in catalog becomes a central composition file, but it contains
membership and import metadata only. Chemistry logic, schemas, and provider
implementations remain in their program packages, and core remains usable
without importing MCP code.

## Alternatives rejected

### Keep broad protocols and catch `NotImplementedError`

This preserves the current ambiguity. Support remains discoverable only by
calling a method, and normal capability differences look like runtime
failures.

### Infer support from method presence

Current parser objects contain methods that deliberately refuse operations.
Method presence would report false support and cannot express binary read
versus write capability.

### Declare only component-level capabilities

Flags such as `parser` or `binary` are too coarse. GRASP has a parser but no
vibrational frequency parser. DIRAC and Molcas can read binary artifacts while
some write operations remain unsupported.

### Split every operation into its own provider class

That would fragment small backends and add indirection without changing
behavior. Operation-level declarations provide truthful support while related
methods can remain on one implementation object.

### Discover third-party backends dynamically

Entry points or directory scanning would add packaging, trust, versioning, and
failure-isolation questions before the built-in boundary is settled. This can
be reconsidered after Quantum ESPRESSO and QMCPACK have exercised the catalog.

## Acceptance checks

Phase 0 accepted this decision with these conditions:

- Capabilities describe public operations rather than provider presence.
- The initial capability list matches current consumers.
- One fixed built-in catalog should own program membership.
- The catalog lives at the composition boundary and core does not import MCP
  metadata.
- Program artifact kinds map accepted names to the roles defined in ADR 003.
- Dynamic third-party loading remains deferred.
- Execution-target selection stays outside this ADR and is decided in ADR
  002.
