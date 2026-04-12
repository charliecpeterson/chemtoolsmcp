# Chemtools MCP Server

This directory contains a pure-Python stdio MCP server for the shared `chemtools` package.

Server entrypoint:

- [chemtools_mcp_server.py](./chemtools_mcp_server.py)

It exposes these tools:

- `review_nwchem_progress`
- `parse_nwchem_output`
- `parse_nwchem_scf`
- `inspect_nwchem_input`
- `inspect_nwchem_geometry`
- `resolve_nwchem_basis`
- `render_nwchem_basis_from_elements`
- `render_nwchem_basis_from_input`
- `diagnose_nwchem_output`
- `summarize_nwchem_output`
- `draft_nwchem_optimization_followup_input`
- `prepare_nwchem_next_step`

## Run manually

```bash
python3 /Users/charlie/test/mytest/chem-agent-package/mcp/chemtools_mcp_server.py
```

The server uses stdio and is intended to be launched by an MCP client.

## Basis library path

By default the server uses:

```text
/Users/charlie/test/mytest/nwchem-test/nwchem_basis_library
```

Override that with:

```bash
CHEMTOOLS_BASIS_LIBRARY=/path/to/nwchem_basis_library
```

## Debug logging

To log server activity to a file, set:

```bash
CHEMTOOLS_MCP_LOG=/tmp/chemtools-mcp.log
```

This logs server startup plus `initialize`, `tools/list`, and `tools/call` handling so you can tell whether a client timeout happens before or during a request.

## Recommended use

- Use this MCP server for deterministic chemistry parsing, diagnosis, and basis rendering.
- Use `review_nwchem_progress` first for “how is the job going?” questions instead of repeatedly tailing long outputs.
- Keep local job-control tools separate.
- Let Open WebUI or OpenCode decide when to call the tools.
