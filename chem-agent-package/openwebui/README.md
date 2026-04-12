# Open WebUI Package

This directory contains the Open WebUI-facing pieces of the chemistry assistant bundle.

## Files

- [skills](./skills)
  - Markdown source for new skill content
- [skill-exports](./skill-exports)
  - Open WebUI importable JSON exports generated from the markdown files
- [tools/chemtools_openwebui.py](./tools/chemtools_openwebui.py)
  - Python tool wrapper around the shared `chemtools` package
- [model-bundles.example.yaml](./model-bundles.example.yaml)
  - Suggested wrapper split across `core`, `parse`, and `agent`

## Skill export workflow

Generate the JSON exports with:

```bash
python3 /Users/charlie/test/mytest/chem-agent-package/openwebui/export_skill_templates.py
```

This reads every markdown file in [skills](./skills) and writes matching JSON exports to [skill-exports](./skill-exports).

## Tool wiring

The simplest Open WebUI tool setup is to expose functions from:

- [tools/chemtools_openwebui.py](./tools/chemtools_openwebui.py)

Recommended initial tool set:

- `review_nwchem_progress`
- `parse_nwchem_output`
- `parse_nwchem_scf`
- `inspect_nwchem_input_geometry`
- `inspect_nwchem_input`
- `resolve_nwchem_basis`
- `render_nwchem_basis_from_input`
- `diagnose_nwchem_output`
- `summarize_nwchem_output`
- `draft_nwchem_optimization_followup_input`
- `prepare_nwchem_next_step`

Keep local job submission tools outside Open WebUI unless the WebUI environment already has safe access to the target working directory and scheduler.

## MCP option

If you prefer MCP over direct Python tool registration, use:

- [../mcp/chemtools_mcp_server.py](../mcp/chemtools_mcp_server.py)

That keeps the deterministic chemistry tool layer portable across Open WebUI, OpenCode, and other MCP-capable clients.
