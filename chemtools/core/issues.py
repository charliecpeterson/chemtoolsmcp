"""Severity-tracking issue collector for case-analysis / recovery / lint.

Both `case_analysis.analyze_*_case` and `recovery.suggest_*_recovery` need
to accumulate a list of (severity, message, hint) records and report the
"worst" level seen as the final verdict. This module centralizes that.

Severity levels (low → high):

  info        — informational; doesn't change verdict
  caution     — meaningful concern; pushes verdict to "caution"
  problematic — fatal-ish issue; pushes verdict to "problematic"
"""

from __future__ import annotations

from typing import Any


SEVERITY_LEVELS: dict[str, int] = {"info": 0, "caution": 1, "problematic": 2}
VERDICT_BY_SEVERITY: dict[str, str] = {
    "info": "healthy",
    "caution": "caution",
    "problematic": "problematic",
}


class IssueCollector:
    """Accumulate (severity, message, hint) records with verdict tracking.

    Example:
        coll = IssueCollector()
        coll.add("caution", "ref weight in caution band",
                 hint="consider tighter active space")
        coll.add("problematic", "active space verdict 'poor'")
        print(coll.verdict)   # "problematic"
        print(coll.issues)    # list of dicts
    """

    def __init__(self) -> None:
        self.issues: list[dict[str, str]] = []
        self._worst: str = "info"

    def add(
        self,
        severity: str,
        message: str,
        *,
        hint: str | None = None,
        **extra: Any,
    ) -> None:
        if severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"unknown severity {severity!r}; expected one of {sorted(SEVERITY_LEVELS)}"
            )
        record: dict[str, Any] = {"severity": severity, "message": message}
        if hint is not None:
            record["hint"] = hint
        if extra:
            record.update(extra)
        self.issues.append(record)
        if SEVERITY_LEVELS[severity] > SEVERITY_LEVELS[self._worst]:
            self._worst = severity

    def bump(self, severity: str) -> None:
        """Raise the verdict floor without adding an issue record. Useful
        when another subsystem (e.g. a recovery dispatch) is what generates
        the message but this collector still tracks the cumulative verdict.
        """
        if severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"unknown severity {severity!r}; expected one of {sorted(SEVERITY_LEVELS)}"
            )
        if SEVERITY_LEVELS[severity] > SEVERITY_LEVELS[self._worst]:
            self._worst = severity

    @property
    def worst_severity(self) -> str:
        return self._worst

    @property
    def verdict(self) -> str:
        """Translate the worst-severity to the verdict label used by
        analyze_*_case results: 'healthy' / 'caution' / 'problematic'."""
        return VERDICT_BY_SEVERITY[self._worst]

    def __len__(self) -> int:
        return len(self.issues)

    def __bool__(self) -> bool:
        return bool(self.issues)
