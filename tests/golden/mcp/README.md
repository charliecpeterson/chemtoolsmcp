# MCP golden cases

These cases pin representative `tools/call` behavior before the architecture
changes. The set contains one analysis-safe parser call for NWChem, Molcas,
DIRAC, and GRASP, plus generic auto-detection, guided input review, and guided
run-inspection calls.

Each case commits:

- The JSON-RPC request.
- A small synthetic output fixture.
- The exact top-level tool payload keys.
- A subset of stable chemistry and parser facts.

The tests also pin the MCP success envelope and require one JSON text content
item. Paths, file sizes, byte offsets, and other fixture-dependent metadata
remain outside the expected subset unless they are part of the behavior under
test.

Add a case only when it protects a distinct public behavior. Large scientific
reference calculations belong in the external reference manifest rather than
this directory.
