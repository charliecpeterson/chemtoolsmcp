# Standalone Chemtools

This package is a standalone Python port of selected parsing logic from Orbitron.

It does **not** import or depend on Orbitron at runtime.

Orbitron was used only as the reference implementation while porting the parser logic.

## Current support

- `parse-tasks`
  - NWChem
  - Molpro
  - Molcas
- `parse-mos`
  - NWChem
  - Molpro
- `parse-freq`
  - NWChem
- `parse-trajectory`
  - NWChem optimization outputs
- `parse-output`
  - Aggregates the above and reports unsupported sections in `errors`
- `parse-scf`
  - Extracts the NWChem SCF/DFT iteration table, max-iteration setting, and convergence pattern
- `diagnose-nwchem`
  - Produces a first structured diagnosis for SCF failure or likely wrong-state convergence
- `summarize-nwchem`
  - Produces a concise human-readable summary plus the structured diagnosis payload
- `review-nwchem-progress`
  - Produces a compact task-aware progress summary for a running or incomplete NWChem job
- `basis-library-summary`
  - Lists basis-set files in a local NWChem basis library
- `resolve-basis`
  - Validates that a requested basis exists and covers the requested elements
- `get-basis-blocks`
  - Extracts raw per-element basis blocks from the library file
- `inspect-nwchem-geometry`
  - Reads the unique element list from the first NWChem geometry block
- `inspect-nwchem-input`
  - Reads geometry elements, charge, multiplicity, and task lines from an input
- `render-nwchem-basis`
  - Builds an explicit per-element NWChem `basis` block using validated library names

## CLI

Run from this directory:

```bash
python3 chemtools_tool.py parse-tasks /path/to/file.out
python3 chemtools_tool.py parse-mos /path/to/file.out --top-n 5
python3 chemtools_tool.py parse-freq /path/to/file.out
python3 chemtools_tool.py parse-trajectory /path/to/file.out
python3 chemtools_tool.py parse-output /path/to/file.out --pretty
python3 chemtools_tool.py parse-scf /path/to/file.out --pretty
python3 chemtools_tool.py diagnose-nwchem /path/to/job.out --input-file /path/to/job.nw --pretty
python3 chemtools_tool.py summarize-nwchem /path/to/job.out --input-file /path/to/job.nw --pretty
python3 chemtools_tool.py review-nwchem-progress --output-file /path/to/job.out --input-file /path/to/job.nw --pretty
python3 chemtools_tool.py resolve-basis def2-svp --elements Fe,C,H --library-path /path/to/nwchem_basis_library --pretty
python3 chemtools_tool.py render-nwchem-basis def2-svp --geometry-file /path/to/job.nw --library-path /path/to/nwchem_basis_library
```

## OpenWebUI tool shape

The simplest OpenWebUI tool wrapper is a Python function that shells out to:

```bash
python3 /absolute/path/to/chemtools_tool.py parse-output /absolute/path/to/file.out
```

or a narrower command like:

```bash
python3 /absolute/path/to/chemtools_tool.py parse-mos /absolute/path/to/file.out --top-n 5
```

For diagnosis:

```bash
python3 /absolute/path/to/chemtools_tool.py diagnose-nwchem /absolute/path/to/job.out \
  --input-file /absolute/path/to/job.nw \
  --pretty
```

For a concise summary:

```bash
python3 /absolute/path/to/chemtools_tool.py summarize-nwchem /absolute/path/to/job.out \
  --input-file /absolute/path/to/job.nw \
  --pretty
```

For a running or incomplete job progress review:

```bash
python3 /absolute/path/to/chemtools_tool.py review-nwchem-progress \
  --output-file /absolute/path/to/job.out \
  --input-file /absolute/path/to/job.nw \
  --pretty
```

For explicit basis blocks:

```bash
python3 /absolute/path/to/chemtools_tool.py render-nwchem-basis def2-svp \
  --geometry-file /absolute/path/to/job.nw \
  --library-path /absolute/path/to/nwchem_basis_library
```

This intentionally emits explicit per-element `library` assignments like:

```text
basis "ao basis" spherical
  Fe library def2-svp
  C  library def2-svp
  H  library def2-svp
end
```

That matches the style used in your own NWChem training inputs more closely than relying on `* library ...` for new generated inputs.

## MCP direction

The MCP server now lives at:

- [../chem-agent-package/mcp/chemtools_mcp_server.py](/Users/charlie/test/mytest/chem-agent-package/mcp/chemtools_mcp_server.py)

The intended exposed tools are:

- `parse_tasks(file)`
- `parse_mos(file, top_n=5, include_coefficients=false)`
- `parse_freq(file)`
- `parse_trajectory(file, include_positions=false)`
- `parse_output(file, sections=[...])`
- `parse_scf(file)`
- `review_nwchem_progress(output_file, input_file=None, ...)`
- `diagnose_output(output_file, input_file=None, expected_metal_elements=None, expected_somo_count=None)`
- `summarize_output(output_file, input_file=None, expected_metal_elements=None, expected_somo_count=None)`
- `resolve_basis(basis_name, elements, library_path)`
- `render_basis_block(basis_name, elements, library_path, block_name="ao basis")`

That keeps the parsing logic independent from the transport layer.

## Training case format

Use [case_schema.json](/Users/charlie/test/mytest/chemtools/case_schema.json) as the structured format for labeled examples.

Use [case_template.json](/Users/charlie/test/mytest/chemtools/case_template.json) as the per-case starter file inside directories like:

```text
nwchem-test/train/<case-name>/
  failed.nw
  failed.out
  solution.nw
  solution.out
  case.json
  NOTES.md
```

The schema is designed so the same case can help:

- parser regression tests
- agent diagnosis behavior
- future eval cases

## Runner profiles

Use [runner_profiles.example.json](/Users/charlie/test/mytest/chemtools/runner_profiles.example.json) as the default no-dependency profile source, or [runner_profiles.example.yaml](/Users/charlie/test/mytest/chemtools/runner_profiles.example.yaml) if you prefer YAML.
For workstation-first setup, start from [runner_profiles.local.example.json](/Users/charlie/test/mytest/chemtools/runner_profiles.local.example.json) and replace the placeholder launcher commands with the exact way NWChem is run on your machine.

Use runner profiles to separate:

- parser/agent logic
- machine-specific NWChem launch details

The intended pattern is:

- one stable tool like `prepare_nwchem_run(input_file, profile, ...)`
- one launch tool like `launch_nwchem_run(input_file, profile, ...)`
- one workflow tool like `prepare_nwchem_next_step(output_file, input_file, ...)`
- many environment-specific profiles such as `local`, `slurm_cpu`, and `pbs_cpu`

That way you do not rewrite the tool when you move between workstation and cluster environments.

If you want the tools to pick up a personal runner profile file automatically, set `CHEMTOOLS_RUNNER_PROFILES` to that file path. When this environment variable is set, the runner layer will use it whenever `profiles_path` is omitted.

Example:

```bash
export CHEMTOOLS_RUNNER_PROFILES=/Users/charlie/test/mytest/nwchemaitest/runner_profiles.local.json
```

### Local-first testing

1. Copy [runner_profiles.local.example.json](/Users/charlie/test/mytest/chemtools/runner_profiles.local.example.json) to your own editable file.
2. Replace `/path/to/nwchem` or `/absolute/path/to/run_nwchem_local.sh` with the real local launch command.
3. Preview the rendered launch without running anything:

```bash
python3 /Users/charlie/test/mytest/chemtools_tool.py inspect-runner-profiles \
  --profiles-path /absolute/path/to/runner_profiles.local.json --pretty

python3 /Users/charlie/test/mytest/chemtools_tool.py prepare-nwchem-run \
  /absolute/path/to/job.nw \
  --profile workstation_serial \
  --profiles-path /absolute/path/to/runner_profiles.local.json \
  --pretty
```

4. Only after the preview looks correct, launch with the same profile:

```bash
python3 /Users/charlie/test/mytest/chemtools_tool.py launch-nwchem-run \
  /absolute/path/to/job.nw \
  --profile workstation_serial \
  --profiles-path /absolute/path/to/runner_profiles.local.json \
  --pretty
```
