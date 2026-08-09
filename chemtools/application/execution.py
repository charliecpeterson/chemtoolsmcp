"""Coordinate permission, rendering, launch records, and cancellation.

The service checks permission before process or scheduler changes. Cancellation
requires a launch record from this service instance and the same target used
for launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from uuid import UUID, uuid4

from chemtools.application.execution_policy import (
    EXECUTION_OPERATIONS,
    ExecutionDecision,
    ExecutionDisabledError,
    ExecutionOperation,
    LaunchCancellationError,
    LaunchStatusError,
)
from chemtools.core.execution import (
    ExecutionLaunchRecord,
    ExecutionTarget,
    LaunchPlan,
    LocalCancellationResult,
    LocalLaunchResult,
    LocalStatusResult,
    LocalSynchronousResult,
    RecordedCancellation,
    RecordedLaunch,
    RecordedLocalStatus,
    RecordedSlurmStatus,
    RecordedSynchronousRun,
    RenderedCommand,
    RenderedSlurmScript,
    SlurmCancellationResult,
    SlurmStatusResult,
    SlurmSubmissionResult,
)
from chemtools.execution import LocalExecutor, SlurmExecutor
from chemtools.persistence.launches import (
    UnknownLaunchRecordError,
    create_launch_record,
    load_launch_record,
    update_launch_record,
)

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ExecutionService:
    enable_execution: bool = False
    registry_db_path: str | Path | None = None
    instance_id: str = field(default_factory=lambda: str(uuid4()))
    _local_executor: LocalExecutor = field(
        default_factory=LocalExecutor,
        init=False,
        repr=False,
        compare=False,
    )
    _slurm_executor: SlurmExecutor = field(
        default_factory=SlurmExecutor,
        init=False,
        repr=False,
        compare=False,
    )
    _launch_targets: dict[str, ExecutionTarget] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.enable_execution, bool):
            raise TypeError("enable_execution must be a boolean")
        try:
            normalized = str(UUID(self.instance_id))
        except ValueError as exc:
            raise ValueError(
                "instance_id must be a canonical UUID string"
            ) from exc
        if normalized != self.instance_id:
            raise ValueError(
                "instance_id must be a canonical UUID string"
            )

    def check(
        self,
        operation: ExecutionOperation,
        target: ExecutionTarget,
    ) -> ExecutionDecision:
        if operation not in EXECUTION_OPERATIONS:
            raise ValueError(
                "execution operation must be 'launch' or 'cancel'"
            )
        if self.enable_execution:
            return ExecutionDecision(
                allowed=True,
                operation=operation,
                target=target.name,
            )
        return ExecutionDecision(
            allowed=False,
            operation=operation,
            target=target.name,
            error="execution_disabled",
        )

    def require(
        self,
        operation: ExecutionOperation,
        target: ExecutionTarget,
    ) -> ExecutionDecision:
        decision = self.check(operation, target)
        if not decision.allowed:
            raise ExecutionDisabledError(decision)
        return decision

    def render(
        self,
        plan: LaunchPlan,
        target: ExecutionTarget,
    ) -> RenderedCommand | RenderedSlurmScript:
        if target.executor == "local":
            return self._local_executor.render(plan, target)
        if target.executor == "slurm":
            return self._slurm_executor.render(plan, target)
        raise ValueError(f"unsupported executor kind: {target.executor!r}")

    def _pending_record(
        self,
        plan: LaunchPlan,
        target: ExecutionTarget,
        rendered: RenderedCommand | RenderedSlurmScript,
    ) -> ExecutionLaunchRecord:
        if isinstance(rendered, RenderedSlurmScript):
            command = rendered.command
            script_path = rendered.script_path
        else:
            command = rendered
            script_path = None
        if plan.stdin_text is None:
            stdin_sha256 = None
            stdin_size_bytes = None
        else:
            stdin_bytes = plan.stdin_text.encode("utf-8")
            stdin_sha256 = hashlib.sha256(stdin_bytes).hexdigest()
            stdin_size_bytes = len(stdin_bytes)
        created_at = _utc_now()
        return ExecutionLaunchRecord(
            launch_id=str(uuid4()),
            instance_id=self.instance_id,
            target=target.name,
            executor=target.executor,
            program=plan.program,
            working_directory=command.working_directory,
            argv=command.argv,
            environment_keys=tuple(sorted(command.environment)),
            resources=plan.resources,
            status="pending",
            created_at=created_at,
            updated_at=created_at,
            staged_files=command.staged_files,
            stdout_path=command.stdout_path,
            stderr_path=command.stderr_path,
            script_path=script_path,
            stdin_sha256=stdin_sha256,
            stdin_size_bytes=stdin_size_bytes,
        )

    def launch(
        self,
        plan: LaunchPlan,
        target: ExecutionTarget,
    ) -> RecordedLaunch:
        self.require("launch", target)
        rendered = self.render(plan, target)
        pending = self._pending_record(plan, target, rendered)
        create_launch_record(pending, self.registry_db_path)
        self._launch_targets[pending.launch_id] = target
        try:
            if target.executor == "local":
                result = self._local_executor.launch(plan, target)
            elif target.executor == "slurm":
                result = self._slurm_executor.submit(plan, target)
            else:
                raise ValueError(
                    f"unsupported executor kind: {target.executor!r}"
                )
        except Exception as exc:
            self._launch_targets.pop(pending.launch_id, None)
            failed = replace(
                pending,
                status="launch_failed",
                updated_at=_utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            update_launch_record(failed, self.registry_db_path)
            raise

        if isinstance(result, LocalLaunchResult):
            completed = replace(
                pending,
                status="started",
                process_id=result.process_id,
                updated_at=_utc_now(),
            )
        elif isinstance(result, SlurmSubmissionResult):
            completed = replace(
                pending,
                status=result.status,
                job_id=result.job_id,
                return_code=result.return_code,
                updated_at=_utc_now(),
                error=(
                    result.stderr
                    if result.status == "submit_failed"
                    else None
                ),
            )
        else:
            raise TypeError("executor returned an unknown launch result")
        update_launch_record(completed, self.registry_db_path)
        if completed.status not in ("started", "submitted"):
            self._launch_targets.pop(completed.launch_id, None)
        return RecordedLaunch(record=completed, result=result)

    def run_to_completion(
        self,
        plan: LaunchPlan,
        target: ExecutionTarget,
    ) -> RecordedSynchronousRun:
        self.require("launch", target)
        if target.executor != "local":
            raise ValueError(
                "synchronous execution requires a local target"
            )
        rendered = self.render(plan, target)
        if isinstance(rendered, RenderedSlurmScript):
            raise TypeError("local target rendered a Slurm script")
        pending = self._pending_record(plan, target, rendered)
        create_launch_record(pending, self.registry_db_path)
        try:
            result = self._local_executor.run_to_completion(
                plan,
                target,
            )
        except Exception as exc:
            failed = replace(
                pending,
                status="launch_failed",
                updated_at=_utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            update_launch_record(failed, self.registry_db_path)
            raise
        if not isinstance(result, LocalSynchronousResult):
            raise TypeError(
                "local executor returned an unknown synchronous result"
            )
        if result.status == "completed":
            error = None
        elif result.stderr:
            error = result.stderr
        elif result.status == "timed_out":
            error = "execution timed out"
        else:
            error = f"process exited with return code {result.return_code}"
        completed = replace(
            pending,
            status=result.status,
            return_code=result.return_code,
            elapsed_seconds=result.elapsed_seconds,
            updated_at=result.completed_at,
            error=error,
        )
        update_launch_record(completed, self.registry_db_path)
        return RecordedSynchronousRun(
            record=completed,
            result=result,
        )

    def get_launch_record(
        self,
        launch_id: str,
    ) -> ExecutionLaunchRecord:
        return load_launch_record(launch_id, self.registry_db_path)

    def _matching_launch_ids(
        self,
        *,
        process_id: int | None = None,
        job_id: str | None = None,
        target_name: str | None = None,
    ) -> list[str]:
        matches = []
        for launch_id, target in self._launch_targets.items():
            if target_name is not None and target.name != target_name:
                continue
            record = load_launch_record(
                launch_id,
                self.registry_db_path,
            )
            if process_id is not None and record.process_id != process_id:
                continue
            if job_id is not None and record.job_id != job_id:
                continue
            matches.append(launch_id)
        return matches

    def refresh_local_status(
        self,
        launch_id: str,
    ) -> RecordedLocalStatus:
        try:
            record = load_launch_record(
                launch_id,
                self.registry_db_path,
            )
        except UnknownLaunchRecordError as exc:
            raise LaunchStatusError({
                "error": "launch_not_owned",
                "launch_id": launch_id,
            }) from exc
        if (
            record.instance_id != self.instance_id
            or launch_id not in self._launch_targets
        ):
            raise LaunchStatusError({
                "error": "launch_not_owned",
                "launch_id": launch_id,
            })
        if record.executor != "local" or record.process_id is None:
            raise LaunchStatusError({
                "error": "launch_not_local",
                "launch_id": launch_id,
            })
        if record.status in ("completed", "failed"):
            result = LocalStatusResult(
                process_id=record.process_id,
                status=record.status,
                return_code=record.return_code,
                checked_at=_utc_now(),
            )
            return RecordedLocalStatus(record=record, result=result)
        if record.status != "started":
            raise LaunchStatusError({
                "error": "launch_not_statusable",
                "launch_id": launch_id,
                "status": record.status,
            })

        result = self._local_executor.status(record.process_id)
        if result.status == "running":
            return RecordedLocalStatus(record=record, result=result)
        error = (
            None
            if result.status == "completed"
            else f"process exited with return code {result.return_code}"
        )
        updated = replace(
            record,
            status=result.status,
            return_code=result.return_code,
            elapsed_seconds=max(
                0.0,
                (result.checked_at - record.created_at).total_seconds(),
            ),
            updated_at=result.checked_at,
            error=error,
        )
        update_launch_record(updated, self.registry_db_path)
        return RecordedLocalStatus(record=updated, result=result)

    def refresh_local_status_external(
        self,
        process_id: int,
    ) -> RecordedLocalStatus:
        launch_id = self.resolve_local_launch_external(process_id)
        return self.refresh_local_status(launch_id)

    def resolve_local_launch_external(
        self,
        process_id: int,
    ) -> str:
        matches = self._matching_launch_ids(process_id=process_id)
        if len(matches) != 1:
            raise LaunchStatusError({
                "error": (
                    "launch_not_owned"
                    if not matches
                    else "launch_identifier_ambiguous"
                ),
                "identifier": str(process_id),
            })
        return matches[0]

    def refresh_slurm_status(
        self,
        launch_id: str,
    ) -> RecordedSlurmStatus:
        try:
            record = load_launch_record(
                launch_id,
                self.registry_db_path,
            )
        except UnknownLaunchRecordError as exc:
            raise LaunchStatusError({
                "error": "launch_not_owned",
                "launch_id": launch_id,
            }) from exc
        target = self._launch_targets.get(launch_id)
        if record.instance_id != self.instance_id or target is None:
            raise LaunchStatusError({
                "error": "launch_not_owned",
                "launch_id": launch_id,
            })
        if record.executor != "slurm" or record.job_id is None:
            raise LaunchStatusError({
                "error": "launch_not_slurm",
                "launch_id": launch_id,
            })
        if record.status in (
            "completed",
            "failed",
            "timed_out",
            "cancelled",
        ):
            raw_state = None
            result_status = record.status
            error_prefix = "Slurm job state "
            if record.error and record.error.startswith(error_prefix):
                raw_state = record.error.removeprefix(error_prefix)
                if raw_state == "OUT_OF_MEMORY":
                    result_status = "out_of_memory"
            result = SlurmStatusResult(
                job_id=record.job_id,
                query_argv=(),
                source="record",
                status=result_status,
                raw_state=raw_state,
                query_return_code=None,
                stdout="",
                stderr="",
                checked_at=_utc_now(),
                job_exit_code=record.return_code,
                elapsed_seconds=record.elapsed_seconds,
                error=record.error,
            )
            return RecordedSlurmStatus(record=record, result=result)
        if record.status not in ("submitted", "cancel_failed"):
            raise LaunchStatusError({
                "error": "launch_not_statusable",
                "launch_id": launch_id,
                "status": record.status,
            })

        result = self._slurm_executor.status(record.job_id, target)
        terminal_status = {
            "completed": "completed",
            "failed": "failed",
            "out_of_memory": "failed",
            "timed_out": "timed_out",
            "cancelled": "cancelled",
        }.get(result.status)
        if terminal_status is None:
            return RecordedSlurmStatus(record=record, result=result)
        error = None
        if result.status != "completed":
            error = f"Slurm job state {result.raw_state}"
        updated = replace(
            record,
            status=terminal_status,
            return_code=result.job_exit_code,
            elapsed_seconds=result.elapsed_seconds,
            updated_at=result.checked_at,
            error=error,
        )
        update_launch_record(updated, self.registry_db_path)
        return RecordedSlurmStatus(record=updated, result=result)

    def refresh_slurm_status_external(
        self,
        job_id: str,
        *,
        target_name: str | None = None,
    ) -> RecordedSlurmStatus:
        launch_id = self.resolve_slurm_launch_external(
            job_id,
            target_name=target_name,
        )
        return self.refresh_slurm_status(launch_id)

    def resolve_slurm_launch_external(
        self,
        job_id: str,
        *,
        target_name: str | None = None,
    ) -> str:
        matches = self._matching_launch_ids(
            job_id=job_id,
            target_name=target_name,
        )
        if len(matches) != 1:
            raise LaunchStatusError({
                "error": (
                    "launch_not_owned"
                    if not matches
                    else "launch_identifier_ambiguous"
                ),
                "identifier": job_id,
            })
        return matches[0]

    def cancel(
        self,
        launch_id: str,
        target: ExecutionTarget,
        *,
        signal_name: str = "term",
    ) -> RecordedCancellation:
        self.require("cancel", target)
        try:
            record = load_launch_record(
                launch_id,
                self.registry_db_path,
            )
        except UnknownLaunchRecordError as exc:
            raise LaunchCancellationError({
                "error": "launch_not_owned",
                "launch_id": launch_id,
                "target": target.name,
            }) from exc
        if record.instance_id != self.instance_id:
            raise LaunchCancellationError({
                "error": "launch_not_owned",
                "launch_id": launch_id,
                "target": target.name,
            })
        if record.target != target.name or record.executor != target.executor:
            raise LaunchCancellationError({
                "error": "launch_target_mismatch",
                "launch_id": launch_id,
                "target": target.name,
                "recorded_target": record.target,
            })
        if record.status not in ("started", "submitted", "cancel_failed"):
            raise LaunchCancellationError({
                "error": "launch_not_cancelable",
                "launch_id": launch_id,
                "target": target.name,
                "status": record.status,
            })
        if launch_id not in self._launch_targets:
            raise LaunchCancellationError({
                "error": "launch_not_owned",
                "launch_id": launch_id,
                "target": target.name,
            })

        launch_target = self._launch_targets[launch_id]
        if record.executor == "local" and record.process_id is not None:
            result = self._local_executor.cancel(
                record.process_id,
                signal_name,
            )
        elif record.executor == "slurm" and record.job_id is not None:
            result = self._slurm_executor.cancel(
                record.job_id,
                launch_target,
            )
        else:
            raise LaunchCancellationError({
                "error": "launch_not_cancelable",
                "launch_id": launch_id,
                "target": target.name,
                "status": record.status,
            })
        if isinstance(result, LocalCancellationResult):
            cancellation_error = result.error
        elif isinstance(result, SlurmCancellationResult):
            cancellation_error = (
                result.stderr
                if result.status == "cancel_failed"
                else None
            )
        else:
            raise TypeError("executor returned an unknown cancellation result")
        updated = replace(
            record,
            status=result.status,
            updated_at=_utc_now(),
            error=cancellation_error,
        )
        update_launch_record(updated, self.registry_db_path)
        if updated.status == "cancelled":
            self._launch_targets.pop(launch_id, None)
        return RecordedCancellation(record=updated, result=result)

    def cancel_recorded(
        self,
        launch_id: str,
        *,
        signal_name: str = "term",
    ) -> RecordedCancellation:
        target = self._launch_targets.get(launch_id)
        if target is None:
            raise LaunchCancellationError({
                "error": "launch_not_owned",
                "launch_id": launch_id,
            })
        return self.cancel(
            launch_id,
            target,
            signal_name=signal_name,
        )

    def cancel_external(
        self,
        *,
        process_id: int | None = None,
        job_id: str | None = None,
        target_name: str | None = None,
        signal_name: str = "term",
    ) -> RecordedCancellation:
        if (process_id is None) == (job_id is None):
            raise ValueError(
                "provide exactly one of process_id or job_id"
            )
        matches = self._matching_launch_ids(
            process_id=process_id,
            job_id=job_id,
            target_name=target_name,
        )
        if len(matches) != 1:
            identifier = (
                str(process_id)
                if process_id is not None
                else str(job_id)
            )
            raise LaunchCancellationError({
                "error": (
                    "launch_not_owned"
                    if not matches
                    else "launch_identifier_ambiguous"
                ),
                "identifier": identifier,
            })
        return self.cancel_recorded(
            matches[0],
            signal_name=signal_name,
        )


__all__ = [
    "ExecutionDecision",
    "ExecutionDisabledError",
    "ExecutionOperation",
    "ExecutionService",
    "LaunchCancellationError",
    "LaunchStatusError",
]
