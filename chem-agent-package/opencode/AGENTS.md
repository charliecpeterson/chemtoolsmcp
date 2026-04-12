# NWChem Local Agent

Use this agent file when OpenCode is pointed at an Open WebUI chemistry wrapper such as `openai-nwchem-core`.

## Ownership split
- Open WebUI wrapper owns:
  - chemistry style
  - app docs knowledge
  - case knowledge
  - optional chemistry parser/basis tools
- OpenCode owns:
  - local filesystem access
  - job directories
  - file edits
  - run submission
  - output polling

## Local workflow
When working on an NWChem case:
1. inspect the current input and output files
2. if the user asks how a running or incomplete job is going, use `review-nwchem-progress` first with both output and input
3. for standard new DFT inputs, use `create-nwchem-dft-from-request` instead of hand-writing text
4. after creating or substantially revising an input, run `lint-nwchem-input` before presenting it as final
5. use `prepare-nwchem-next-step` when you want a diagnosis plus the most likely drafted follow-up artifact
6. otherwise review the case with `review-nwchem-case` before relying on raw-log reading
7. if the job is electronic-structure sensitive, run `diagnose-nwchem` using both input and output
8. if drafting or revising a basis block, use `render-nwchem-basis`
9. keep changes minimal and localized
10. if rerunning, use a named runner profile instead of inventing a shell command
11. if a local runner profiles file is configured through `CHEMTOOLS_RUNNER_PROFILES`, prefer it instead of passing `--profiles-path` repeatedly

## Commands
Parse a job:

```bash
python3 /Users/charlie/test/mytest/chemtools_tool.py review-nwchem-progress --output-file /absolute/path/to/job.out --input-file /absolute/path/to/job.nw --pretty
python3 /Users/charlie/test/mytest/chemtools_tool.py create-nwchem-dft-from-request --geometry-file /absolute/path/to/job.nw --library-path /Users/charlie/test/mytest/nwchem-test/nwchem_basis_library --default-basis def2-svp --xc-functional b3lyp --tasks opt,freq --multiplicity 3 --pretty
python3 /Users/charlie/test/mytest/chemtools_tool.py prepare-nwchem-next-step /absolute/path/to/job.out --input-file /absolute/path/to/job.nw --pretty
python3 /Users/charlie/test/mytest/chemtools_tool.py review-nwchem-case /absolute/path/to/job.out --input-file /absolute/path/to/job.nw --pretty
python3 /Users/charlie/test/mytest/chemtools_tool.py parse-output /absolute/path/to/job.out --pretty
python3 /Users/charlie/test/mytest/chemtools_tool.py summarize-nwchem /absolute/path/to/job.out --input-file /absolute/path/to/job.nw --pretty
python3 /Users/charlie/test/mytest/chemtools_tool.py diagnose-nwchem /absolute/path/to/job.out --input-file /absolute/path/to/job.nw --pretty
```

Render an explicit basis block from the current input geometry:

```bash
python3 /Users/charlie/test/mytest/chemtools_tool.py render-nwchem-basis def2-svp \
  --geometry-file /absolute/path/to/job.nw \
  --library-path /Users/charlie/test/mytest/nwchem-test/nwchem_basis_library
```

## Behavior rules
- Do not duplicate large chemistry prompts locally if the selected Open WebUI model already carries them.
- Do not use both Open WebUI job-control tools and local OpenCode job-control tools in the same workflow unless there is a clear reason.
- Treat successful completion and correct state as separate checks.
- Do not recommend more SCF iterations before inspecting the convergence pattern.
- Do not tail long outputs repeatedly for status questions when `review-nwchem-progress` already provides a compact task-aware summary.
- For formula-level or new-input requests, prefer `create-nwchem-dft-from-request` instead of hand-writing NWChem input text.
- Only hand-edit a generated input if `create-nwchem-dft-from-request` cannot express the requested structure directly.

## Future runner contract
Use stable interfaces like:

```text
inspect_nwchem_runner_profiles()
prepare_nwchem_run(input_file, profile, resources_override=None, env_overrides=None)
launch_nwchem_run(input_file, profile, resources_override=None, env_overrides=None)
```

The command details should live in runner profiles, not in the agent prompt.
