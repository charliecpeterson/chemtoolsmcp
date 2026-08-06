"""Parse explicit f-block shell configurations and derive electron counts.

Catalog slugs are identifiers, not configuration syntax. This module only
accepts the parenthesized configuration fields carried by the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_SHELL_RE = re.compile(
    r"(?P<n>[1-9][0-9]*)(?P<orbital>[spdfgh])"
    r"\((?P<electrons>[0-9]+)(?P<inactive>,i)?\)"
)
_CAPACITY = {"s": 2, "p": 6, "d": 10, "f": 14, "g": 18, "h": 22}
_CORE_ELECTRONS = {"Ar": 18, "Kr": 36, "Xe": 54}


@dataclass(frozen=True)
class ShellOccupancy:
    principal: int
    orbital: str
    electrons: int
    inactive: bool

    @property
    def label(self) -> str:
        return f"{self.principal}{self.orbital}"

    def to_dict(self) -> dict[str, object]:
        return {
            "shell": self.label,
            "principal": self.principal,
            "orbital": self.orbital,
            "electrons": self.electrons,
            "inactive": self.inactive,
        }


def parse_shell_configuration(value: str) -> tuple[ShellOccupancy, ...]:
    """Parse the catalog's explicit ``4f(1)`` / ``4f(1,i)`` syntax."""
    if not isinstance(value, str) or not value:
        raise ValueError("configuration must be a non-empty string")
    shells: list[ShellOccupancy] = []
    position = 0
    while position < len(value):
        match = _SHELL_RE.match(value, position)
        if match is None:
            raise ValueError(
                f"invalid shell configuration at offset {position}: {value!r}"
            )
        orbital = match.group("orbital")
        electrons = int(match.group("electrons"))
        if electrons > _CAPACITY[orbital]:
            raise ValueError(
                f"occupancy {electrons} exceeds {orbital}-shell capacity "
                f"{_CAPACITY[orbital]}"
            )
        shell = ShellOccupancy(
            principal=int(match.group("n")),
            orbital=orbital,
            electrons=electrons,
            inactive=match.group("inactive") is not None,
        )
        if any(existing.label == shell.label for existing in shells):
            raise ValueError(f"duplicate shell {shell.label!r} in {value!r}")
        shells.append(shell)
        position = match.end()
    return tuple(shells)


def encoded_electron_count(confline: str, core: str) -> int:
    """Count all electrons represented by a complete GRASP confline."""
    try:
        core_electrons = _CORE_ELECTRONS[core]
    except KeyError:
        raise ValueError(f"unsupported GRASP core {core!r}") from None
    return core_electrons + sum(
        shell.electrons for shell in parse_shell_configuration(confline)
    )


def occupancy_projection(
    *,
    config: str,
    confline: str,
    core: str | None,
) -> dict[str, object]:
    """Return explicit shell and angular occupancy without parsing a slug."""
    if confline:
        shells = parse_shell_configuration(confline)
        electron_count = encoded_electron_count(confline, core or "")
        source = "grasp_confline"
        complete = True
    elif config != "closed":
        shells = parse_shell_configuration(config)
        electron_count = None
        source = "configuration"
        complete = False
    else:
        shells = ()
        electron_count = None
        source = "unavailable"
        complete = False
    represented_angular = {
        orbital: sum(
            shell.electrons for shell in shells if shell.orbital == orbital
        )
        for orbital in _CAPACITY
    }
    return {
        "source": source,
        "complete": complete,
        "core": core,
        "omitted_core_electrons": _CORE_ELECTRONS.get(core or ""),
        "electron_count": electron_count,
        "shells": [shell.to_dict() for shell in shells],
        "represented_angular_electrons": represented_angular,
    }


__all__ = [
    "ShellOccupancy",
    "encoded_electron_count",
    "occupancy_projection",
    "parse_shell_configuration",
]
