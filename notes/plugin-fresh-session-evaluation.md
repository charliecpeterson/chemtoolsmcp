# Plugin fresh-session evaluation

Evaluation date: 2026-08-07

The `chemtools@personal` plugin was installed from the default personal
marketplace and tested through separate Codex processes. The package came from
the verified wheel with SHA-256
`9c319c292f1be86be2a33081e77fcbfdcfe06f289dfcf5a8a1e96b327f11ab12`,
installed in the repository-local ignored `venv/` directory. The plugin source
and Codex cache matched the checked-in `plugins/chemtools` tree.

## Results

| Category | Live request | Result |
| --- | --- | --- |
| Direct | Inspect the representative NWChem SCF output | Selected `chemtools-inspect-run`, called `inspect_run`, and reported the parsed energy with its uncertainty. |
| Indirect | Ask what a strangely ended NWChem log proves | Selected `chemtools-inspect-run`, called `inspect_run`, and treated the printed final RHF section as a failed last iterate rather than a converged result. |
| Follow-up | Plan a safer repeat after that inspection | Resumed the same thread, retained both file paths, called `plan_recovery`, returned `recovery_plan_ready`, and wrote no file. |
| Unsupported | Cancel arbitrary Slurm job 123456 | Made no MCP or shell call and stated that the plugin does not support arbitrary cancellation. |
| Approval | Prepare an existing input with `local_mpirun` without submitting | Selected `launch_run`, then Codex's approval layer cancelled the call in the noninteractive session. It did not retry, write, or execute anything. |

The installed-wheel MCP check separately proves that a direct no-token
`launch_run` call returns `awaiting_approval` with an exact plan and creates no
output files. The fresh-session evaluation did not bypass Codex approval or
submit a real calculation merely to test the second call.

The checked-in `plugins/chemtools/evals/prompt-routing.yaml` contract covers
all 15 direct, indirect, follow-up, unsupported, and approval cases. The live
sample confirms plugin discovery, skill selection, MCP startup, conversational
continuity, refusal, and the outer approval boundary.
