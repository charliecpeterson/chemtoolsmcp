# ADR 002: Execution targets replace local and HPC behavior modes

Status: Accepted

Date: 2026-07-30

Chemtools will replace its global `analysis`, `local`, and `hpc` behavior
modes with two independent settings:

1. A global permission gate for operations that change process or scheduler
   state.
2. A named execution target that describes where and how an approved
   calculation runs.

Analysis tools will no longer depend on an execution mode. Programs will build
launch plans, while executors will run those plans without containing
chemistry-software rules.

## Context

The current mode model combines three separate questions:

- Which tools should be visible?
- Is Chemtools allowed to launch or cancel work?
- Is a configured runner direct or scheduler-backed?

`chemtools.mcp.modes` answers all three through capability tags:

| Mode | Allowed execution shape |
| --- | --- |
| `analysis` | No runner-profile or execution-tagged tools |
| `local` | Direct runner profiles and tools tagged `executable` |
| `hpc` | All local capabilities plus tools tagged `scheduler` |

If no mode is explicit, the server inspects
`CHEMTOOLS_RUNNER_PROFILES`. One scheduler profile changes the whole server
to `hpc`, including any direct profiles in the same file. Merely configuring
a profile therefore changes both tool visibility and execution permission.

The `needs` tags also hide read-only work. Profile inspection, resource
advice, memory-fit checks, script rendering, account detection, and partition
advice can be unavailable in analysis mode even though they do not launch or
cancel a calculation. At the same time, `executable` groups several different
effects under one label.

Execution had a second coupling when this decision was recorded. Shared
runner algorithms lived under NWChem-named functions such as `run_nwchem`
and `watch_nwchem_run`, which Molcas, DIRAC, and GRASP called through thin
wrappers. The canonical entry points are now program-neutral; the old names
remain compatibility aliases.

The current profile format contains useful target information:

- Direct or scheduler-backed launch.
- Submit, status, and cancellation commands.
- Scheduler script templates.
- Resource and hardware defaults.
- Modules, environment values, hooks, and file rules.
- Program executable or container settings.

It also mixes target-wide settings with NWChem-specific fields and permits
shell command templates. Adding Quantum ESPRESSO and QMCPACK to that shape
would copy more program knowledge into scheduler configuration.

Finally, `local` and `hpc` do not describe execution precisely. A workstation
may use MPI or a container. A process running inside an existing Slurm
allocation is direct execution even though it is on a cluster. The executor
kind is the useful distinction.

## Decision

### Execution permission is explicit and defaults to off

Add one server setting:

```text
enable_execution = false
```

The setting controls operations that change external process or scheduler
state:

- Starting a local process.
- Submitting a scheduler job.
- Cancelling a local process or scheduler job.

It does not control pure analysis, target inspection, resource advice,
launch-plan rendering, scheduler-script rendering, status queries, progress
inspection, or output parsing. Those operations may require a configured
target, but they do not require execution permission.

Writing an input draft or rendered script is also outside this gate. Those
tools retain their own path, overwrite, and dry-run controls. The execution
gate protects process and scheduler state rather than serving as a general
filesystem permission system.

The default is false even when valid targets exist. Target configuration must
never enable execution as a side effect.

At `tools/list` time, state-changing tools are hidden when execution is
disabled or no compatible target exists. Dispatch repeats the check so a
client cannot call a hidden tool by name. A refused call returns a structured
error:

```json
{
  "error": "execution_disabled",
  "operation": "launch",
  "target": "linux4090"
}
```

The application service below MCP dispatch enforces the same gate. Direct
Python callers use that service rather than calling an executor directly, so
MCP visibility is not the security boundary. The service also owns a run
registry. By default, cancellation is limited to a process or scheduler job
recorded as launched by this Chemtools instance for the same target. Accepting
an arbitrary PID or job ID would require a separate, explicit administration
policy and is outside this decision.

Invalid target configuration disables target-dependent tools without
preventing analysis-only startup.

### Named targets describe the execution environment

An `ExecutionTarget` owns machine and scheduler facts:

```python
@dataclass(frozen=True)
class ExecutionTarget:
    name: str
    executor: ExecutorKind
    allowed_work_roots: tuple[Path, ...]
    hardware: HardwareDescription
    scheduler: SchedulerDefaults | None
    programs: Mapping[str, ProgramInstallation]
```

The initial executor kinds are `local` and `slurm`. PBS and LSF remain in the
legacy profile adapter, but they will not become first-class executor kinds
until there is a real target, fixture, and smoke-test procedure for each.

A target may contain several program installations. Containers, MPI
launchers, and program-specific environment settings belong to each
installation. They are runtime choices for a program, not executor kinds.

The eventual configuration syntax will follow the repository's selected
configuration format. This illustrative shape records the required
separation:

```toml
[chemtools]
enable_execution = false
default_target = "linux4090"

[targets.linux4090]
executor = "local"
allowed_work_roots = ["/path/to/calculations"]

[targets.linux4090.programs.nwchem]
executable_argv = ["/path/to/nwchem"]
launcher_argv = ["mpirun", "-np", "{mpi_ranks}"]

[targets.stampede3]
executor = "slurm"
allowed_work_roots = ["/path/to/scratch"]

[targets.stampede3.scheduler]
submit_argv = ["sbatch", "{script_file}"]
status_argv = ["squeue", "-j", "{job_id}", "-h", "-o", "%T"]
cancel_argv = ["scancel", "{job_id}"]

[targets.stampede3.programs.nwchem]
executable_argv = ["/path/to/nwchem"]
launcher_argv = ["ibrun"]
```

Machine paths belong in ignored local configuration or environment values
referenced by that configuration. The repository will commit a portable
example, not working paths from `linux-4090`, Stampede3, or another machine.

Each state-changing request selects a configured target by name. If the
request omits the target, the server may use `default_target`. It does not
pick a target by scanning launcher kinds or executable names.

An MCP request may choose:

- A configured program and target.
- Typed resource fields supported by that target.
- A working directory under one of the target's allowed roots.
- Environment overrides whose keys the target explicitly permits.

It may not provide an arbitrary executable, command template, submit command,
shell hook, or scheduler script. Those values cross the local trust boundary
and must come from machine-owned configuration.

Allowed roots are a filesystem boundary, not a string-prefix check. The
application service resolves configured roots and every requested working,
staging, script, and output path through existing parents before checking
containment. It rejects symlinks that escape an allowed root and rechecks
containment after creating a directory or opening a destination. Filesystem
write tools that do not require a target retain their separate path and
overwrite policies.

### Targets are local to the MCP host

An execution target describes an environment directly reachable by the
running Chemtools server. A Stampede3 target is valid when Chemtools runs on
Stampede3 and can call its local Slurm commands. A `linux-4090` server does
not gain remote submission merely because its configuration contains a
target named `stampede3`.

SSH transport, remote file staging, credential handling, and remote path
mapping are deferred. If they are added, they require a separate executor and
security decision. They will not be hidden inside `SlurmExecutor`.

Orbitron also remains outside the execution-target model. Its current
read-only CLI adapter is an optional integration with its own executable
resolution and provenance. A target may later supply machine-local
integration paths, but invoking Orbitron does not turn it into a chemistry
calculation backend.

### Programs build launch plans

A program runtime adapter translates a reviewed calculation into a typed
launch plan:

```python
@dataclass(frozen=True)
class LaunchPlan:
    program: str
    program_arguments: tuple[str, ...]
    environment: Mapping[str, str]
    working_directory: Path
    staged_files: tuple[StagedFile, ...]
    expected_artifacts: tuple[ExpectedArtifact, ...]
    resources: ResourceRequest
    progress_detector: ProgressDetector | None
```

The adapter owns:

- Program argument syntax.
- Input and output filenames.
- Required auxiliary files.
- Program environment values.
- Expected artifacts and progress markers.

The target contributes the selected installation and resource defaults.
`ProgramInstallation` owns trusted `launcher_argv` and `executable_argv`
arrays. The executor constructs the effective command in this order:

```text
launcher_argv + executable_argv + program_arguments
```

The launcher may contain typed resource placeholders defined by the target.
The program adapter cannot replace either configured prefix or escape the
target's allowed working roots. This division also represents a container
installation without giving the program adapter control of the container
runtime.

ADR 001 does not yet declare an execution capability because Phase 1 has no
runtime-adapter consumer. When the Phase 3 launch service is implemented,
`execution.plan` will be added as an operation-level capability and declared
only by backends with tested launch-plan support.

### Executors do not contain program rules

`LocalExecutor` will:

- Assemble and execute the effective argument array without a shell.
- Apply the approved environment and working directory.
- Capture stdout, stderr, exit status, timing, and expected artifact state.
- Enforce configured timeout and cancellation behavior.

`SlurmExecutor` will:

- Convert the same launch plan and resource request into a batch script.
- Apply scheduler directives and the same target-owned command assembly.
- Submit, query, and cancel through configured argument arrays.
- Record the scheduler job ID, rendered script, commands, and state history.

A scheduler script is intentionally a shell document because that is the
batch interface. Its executable lines come from the launch plan and trusted
target configuration. Request arguments are quoted or rendered through typed
fields rather than interpolated into an open command string.

The canonical executors will not use `shell=True` for direct launch, submit,
status, or cancellation commands. Existing command strings and hooks are
accepted only by the legacy profile adapter during migration.

Execution results record enough provenance to reproduce the boundary:
program, target, executor kind, effective resource request, executable
identity when available, launch arguments, environment keys set by
Chemtools, working directory, timestamps, and scheduler job ID.

### Tool access uses effect and target requirements

Replace the mode-oriented `needs` values with two independent pieces of tool
metadata:

```python
requires_target: bool
changes_execution_state: bool
```

Program and toolset filtering remains independent. Executor compatibility is
checked against the resolved target and program installation rather than
encoded as a global server mode.

Examples:

| Operation | Requires target | Changes execution state |
| --- | --- | --- |
| Parse output | No | No |
| Draft input | No | No |
| Inspect configured targets | No | No |
| Advise resources for a target | Yes | No |
| Render a launch plan or batch script | Yes | No |
| Query or watch job status | Yes | No |
| Launch or submit | Yes | Yes |
| Cancel | Yes | Yes |

This is deliberately smaller than a general permission framework. More
effect classes will be added only if a real operation cannot be represented
truthfully by these fields. Filesystem writes retain separate path and
overwrite policies; Phase 4 may split write effects further if one guided
operation combines drafting, staging, and launch.

## Migration

The migration will preserve current MCP behavior long enough to compare the
old and new paths:

1. Add target and launch-plan models beside the current runner profiles.
2. Add a schema-versioned adapter that reads current version 1 profiles as
   targets without rewriting the user's file.
3. Move the generic algorithms behind `LocalExecutor` and `SlurmExecutor`,
   retaining the NWChem-named functions as compatibility wrappers.
4. Convert one NWChem direct profile and one Slurm profile to launch plans and
   compare rendered commands, scripts, job IDs, and status normalization.
5. Add exact tests for command assembly, symlink escape rejection, the
   application-service execution gate, and registry-bound cancellation.
6. Add Molcas, DIRAC, and GRASP launch-plan adapters without copying scheduler
   code.
7. Replace `needs` filtering with target and state-change metadata.
8. Add Quantum ESPRESSO and QMCPACK execution only after their program
   adapters and artifact contracts exist.
9. Remove the old mode resolver and profile adapter after their documented
   deprecation window and compatibility fixtures are complete.

During the compatibility window:

- `CHEMTOOLS_MODE=analysis` maps to execution disabled.
- `CHEMTOOLS_MODE=local` maps to execution enabled through legacy direct
  profiles.
- `CHEMTOOLS_MODE=hpc` maps to execution enabled through the current profile
  set. It remains a superset, so a mixed file can still contain direct and
  scheduler profiles.
- Auto-detection from `CHEMTOOLS_RUNNER_PROFILES` remains only on the legacy
  path and emits a startup warning that profile presence enabled execution.
- A version 2 target configuration is authoritative. If legacy mode settings
  imply a conflicting execution gate, startup fails closed with a
  configuration error.
- With neither new nor legacy execution configuration, execution is disabled.

The generated tool inventory and golden MCP responses remain the behavior
baseline. New target metadata may be added to introspection responses, but
current fields and tool names follow the alias policy decided in ADR 004.

## Consequences

The server can load several program installations and target types without
changing its analysis surface. Read-only target planning remains available
while execution is disabled, which lets an assistant review the exact launch
before the user enables state changes.

The same program launch plan can be rendered for a workstation or Slurm.
Scheduler code no longer needs NWChem, Molcas, DIRAC, GRASP, Quantum ESPRESSO,
or QMCPACK branches.

Configuration becomes more explicit. Users must name allowed work roots,
program installations, and any permitted environment overrides. That costs
some setup but prevents an MCP request from turning a scientific runner into
an arbitrary shell interface.

Legacy PBS and LSF support remains usable during migration but is not part of
the first canonical executor implementation. This avoids claiming tested
support for systems that do not yet have representative targets.

## Alternatives rejected

### Keep the three behavior modes

Modes continue to hide read-only tools and tie permission to target shape.
They also cannot describe several configured targets or direct execution on a
cluster without special cases.

### Infer permission from configured profiles

Configuration proves that a target exists. It does not prove that the user
intends the current MCP session to submit or cancel work. Permission must be
explicit and default to off.

### Build one runner per program

That would repeat scheduler submission, status normalization, cancellation,
timeout, and provenance code for every backend. Program-specific command
construction belongs in the launch-plan adapter.

### Treat every scheduler as one generic HPC executor now

The current code names Slurm, PBS, and LSF, but only Slurm has the planned
target and smoke-test coverage. Promoting all three would preserve a broad
claim without the evidence needed to maintain it.

### Add remote SSH targets

Remote execution introduces credentials, transport failures, file
synchronization, path identity, and host verification. It is a separate
system from a local scheduler executor and is not needed for the initial
architecture.

### Accept commands from MCP requests

This would expose a general command runner under chemistry-shaped tool names.
Configured program installations and typed resource overrides cover the
intended workflows with a much smaller trust surface.

## Acceptance checks

Phase 0 accepted this decision with these conditions:

- Execution permission and target selection are independent.
- Profile presence never enables execution in the final architecture.
- Read-only target planning does not require execution permission.
- Programs own launch-plan construction and executors own process or
  scheduler mechanics.
- Initial canonical executors are local and Slurm.
- Targets describe environments reachable from the running MCP host.
- MCP requests cannot supply arbitrary executables or command templates.
- Program adapters emit arguments; targets own launcher and executable
  arrays; executors assemble the effective command.
- The application service enforces execution permission, path containment,
  and registry-bound cancellation below MCP dispatch.
- Legacy modes and profiles have a measured, fail-closed migration path.
