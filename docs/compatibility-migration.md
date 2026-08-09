# Compatibility migration

Release `v0.1.0` preserves the old MCP names, executable aliases, and Python
import facades for one final migration window. New work should use the
canonical command, guided MCP tools, and focused Python modules.

## MCP setup

Replace `chemtools-nwchem` with `chemtools` in MCP client configuration. The
old command still starts the same server during the compatibility window and
prints a warning on stderr. The server now reports `chemtools` as its MCP
identity.

The separate `chemtools-nwchem-docs` command is also temporary. Its six tools
already exist in the main server. Configure one `chemtools` server instead.

Use the eleven guided tool names for agent workflows. Older MCP names remain
callable during the window, even when they are absent from `tools/list`. The
[generated tool inventory](tool-inventory.md) records each old name and its
replacement.

## Python imports

Direct Python code should import the module that owns the operation:

| Old import | Supported location |
| --- | --- |
| `chemtools` | `chemtools.application`, `chemtools.execution`, `chemtools.integrations`, `chemtools.persistence`, `chemtools.programs.<program>`, or `chemtools.reference` |
| `chemtools.api` | The focused module above that owns the function |
| `chemtools.api_input` | `chemtools.programs.nwchem.input` or the specific input module |
| `chemtools.api_strategy` | `chemtools.programs.nwchem.strategy` or the specific strategy module |
| `chemtools.mcp.nwchem` | `chemtools.mcp.cli` or `chemtools.mcp.dispatch` |
| `chemtools.mcp.tools.nwchem` | `chemtools.mcp.tools._nwchem_provider` or a focused handler module |
| `chemtools.execution.executors` | `chemtools.execution` or the specific executor module |
| `chemtools.execution.legacy_profiles` | `chemtools.execution.profiles` |
| `chemtools.execution.legacy_status` | `chemtools.execution.external_status` for file or explicit Slurm inspection |
| `chemtools.programs.nwchem.legacy_status` | `chemtools.programs.nwchem.external_status` |

Do not replace the old broad facade with another broad facade. Existing
scripts in this repository show the intended focused import paths.

`chemtools.mcp.tools.nwchem` was removed from post-`v0.1.0` development. The
`v0.1.0` tag is the final release that provides its wildcard exports and
dynamic handler lookup. Import `_nwchem_provider` only for MCP composition;
ordinary code should import the focused handler module that owns the operation.

`chemtools.execution.legacy_profiles` was also removed after `v0.1.0`.
Its exports were exact identities from `chemtools.execution.profiles`, which
remains the owner of the unchanged version 1 profile schema and conversion
functions.

The two `legacy_status` modules were replaced after `v0.1.0`. Their retained
contract is read-only file inspection and external Slurm attachment through an
explicit profile and job ID. Arbitrary PID probing, PBS and LSF status parsing,
`.jobid` inference, and direct Python cancellation wrappers were removed.
Cancellation through MCP still requires a launch owned by the current
execution service.

## Removal gate

The compatibility names stay intact in `v0.1.0`. The generated inventory uses
`0.1.0` as their exact `deprecated_since` value and leaves `remove_after`
unset. Removal happens in a later declared breaking release, after the
migration window and the contract checks recorded in ADR 004.
