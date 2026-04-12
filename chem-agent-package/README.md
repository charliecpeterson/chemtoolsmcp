# Chemistry Agent Package

This directory is the packaging layer around the standalone `chemtools` parser/basis helpers.

It is split by runtime:

- [mcp](./mcp)
  - Portable stdio MCP server for parser, diagnosis, and basis tools
- [openwebui](./openwebui)
  - Prompt and skill assets for Open WebUI model wrappers
- [opencode](./opencode)
  - Local agent instructions for filesystem-aware execution

The intended architecture is:

1. Open WebUI owns the chemistry wrapper:
   - app docs KB
   - case KB
   - chemistry style skills
   - parsing/basis tools when desired
2. OpenCode owns local execution:
   - file reads/writes
   - job submission
   - local runner profiles

The shared deterministic tool layer stays in:

- [chemtools](/Users/charlie/test/mytest/chemtools)
- [chemtools_tool.py](/Users/charlie/test/mytest/chemtools_tool.py)
