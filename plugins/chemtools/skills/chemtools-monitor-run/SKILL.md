---
name: chemtools-monitor-run
description: Refresh execution state and scientific progress for a calculation launched by the active Chemtools MCP process. Use when the user provides a Chemtools launch ID and asks whether its local process or Slurm job is queued, running, complete, or making scientific progress.
---

# Monitor a Chemtools-owned run

Use the Chemtools `monitor_run` MCP tool for one explicit status refresh.

1. Require the complete `launch_id` returned by `launch_run` from the same
   active MCP server process.
2. Do not substitute a PID, Slurm job ID, output path, or guessed identifier.
3. Call `monitor_run` once and report execution state, recorded artifacts,
   backend scientific progress, uncertainty, and the highest-priority next
   action as separate facts.
4. Preserve unresolved scheduler or accounting state. Do not infer completion
   merely because a job is absent from the live queue.
5. For repeated checks, refresh only at the interval or follow-up requested by
   the user. Do not create an unbounded polling loop.

This workflow never submits, restarts, cancels, edits, or relaunches a run. If
the identifier is unowned after a server restart, explain that ownership was
process-local. Ask for the primary output path and use `chemtools-inspect-run`
when the user wants a scientific verdict from an existing file.
