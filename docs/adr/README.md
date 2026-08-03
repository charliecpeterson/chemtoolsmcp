# Architecture decisions

Architecture decision records capture choices that are costly to reverse. A
record stays `Proposed` until the Phase 0 review accepts or rejects it.
Accepted records guide implementation; they do not imply that the migration
has already happened.

The [Phase 0 review](phase-0-review.md) accepted all five records after its
eleven findings were resolved.

| ADR | Status | Decision |
| --- | --- | --- |
| [001](001-optional-program-capabilities-and-builtin-catalog.md) | Accepted | Optional program capabilities and one built-in backend catalog |
| [002](002-execution-targets-replace-behavior-modes.md) | Accepted | Named execution targets and an explicit execution gate replace behavior modes |
| [003](003-runs-are-artifact-collections-with-provenance.md) | Accepted | Runs contain typed artifacts, observations, and append-only provenance |
| [004](004-public-mcp-alias-and-deprecation-policy.md) | Accepted | Canonical MCP tools, hidden compatibility aliases, and measured removal gates |
| [005](005-reference-corpus-boundaries.md) | Accepted | Committed fixtures, scientific datasets, and manifest-selected external references have separate curation rules |
