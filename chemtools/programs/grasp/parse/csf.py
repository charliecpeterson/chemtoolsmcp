"""Validated loading of GRASP block-structured configuration state functions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat


MAX_GRASP_CSF_BYTES = 64 * 1024 * 1024
MAX_GRASP_CSF_LINE_CHARS = 256
_ANGULAR_LABELS = "spdfghiklm"
_SUBSHELL_RE = re.compile(r"(?<!\S)([1-9]\d?[spdfghiklm]-?)(?!\S)")
_OCCUPATION_RE = re.compile(
    r"([1-9]\d?[spdfghiklm]-?)\s*\(\s*(\d+)\s*\)"
)
_SYMMETRY_RE = re.compile(r"(\d+)(?:/(2))?([+-])\s*$")


@dataclass(frozen=True)
class CsfEntry:
    block_index: int
    index_within_block: int
    global_index: int
    occupations: tuple[tuple[str, int], ...]
    occupation_line: str
    subshell_quantum_numbers: str
    coupling_and_symmetry: str
    two_j: int
    j_label: str
    parity: str

    def summary(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "index_within_block": self.index_within_block,
            "global_index": self.global_index,
            "configuration": " ".join(
                f"{subshell}({electrons})"
                for subshell, electrons in self.occupations
            ),
            "occupations": [
                {"subshell": subshell, "electrons": electrons}
                for subshell, electrons in self.occupations
            ],
            "subshell_quantum_numbers": self.subshell_quantum_numbers,
            "coupling_and_symmetry": self.coupling_and_symmetry,
            "two_j": self.two_j,
            "j_label": self.j_label,
            "parity": self.parity,
            "source_lines": [
                self.occupation_line,
                self.subshell_quantum_numbers,
                self.coupling_and_symmetry,
            ],
        }


@dataclass(frozen=True)
class CsfBlock:
    index: int
    two_j: int
    j_label: str
    parity: str
    entries: tuple[CsfEntry, ...]


@dataclass(frozen=True)
class CsfDocument:
    source: Path
    size_bytes: int
    sha256: str
    core_subshells: tuple[str, ...]
    peel_subshells: tuple[str, ...]
    electron_count: int
    blocks: tuple[CsfBlock, ...]

    @property
    def csf_count(self) -> int:
        return sum(len(block.entries) for block in self.blocks)

    @property
    def orbital_count(self) -> int:
        return len(self.core_subshells) + len(self.peel_subshells)


def load_grasp_csf_list(path: str | Path) -> CsfDocument:
    source = Path(path).expanduser().resolve()
    try:
        stream = source.open("rb")
    except OSError as error:
        raise ValueError(f"cannot open GRASP CSF file {source}: {error}") from error

    with stream:
        initial_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ValueError(f"GRASP CSF path is not a regular file: {source}")
        if initial_stat.st_size > MAX_GRASP_CSF_BYTES:
            raise ValueError(f"GRASP CSF file exceeds {MAX_GRASP_CSF_BYTES} bytes")
        content = stream.read()
        final_stat = os.fstat(stream.fileno())
        if (
            final_stat.st_size != initial_stat.st_size
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
        ):
            raise ValueError("GRASP CSF file changed during inspection")

    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("GRASP CSF file must contain ASCII text") from error
    lines = text.splitlines()
    if any(len(line) > MAX_GRASP_CSF_LINE_CHARS for line in lines):
        raise ValueError(
            f"GRASP CSF lines must not exceed {MAX_GRASP_CSF_LINE_CHARS} characters"
        )
    core_marker = _marker_index(lines, "Core subshells:")
    peel_marker = _marker_index(lines, "Peel subshells:")
    csf_marker = _marker_index(lines, "CSF(s):")
    if core_marker != 0 or not core_marker < peel_marker < csf_marker:
        raise ValueError("GRASP CSF headers are out of order")

    core_subshells = _parse_subshell_section(
        lines[core_marker + 1:peel_marker],
        "core",
    )
    peel_subshells = _parse_subshell_section(
        lines[peel_marker + 1:csf_marker],
        "peel",
    )
    if not peel_subshells:
        raise ValueError("GRASP CSF file declares no peel subshells")
    all_subshells = core_subshells + peel_subshells
    if len(all_subshells) != len(set(all_subshells)):
        raise ValueError("GRASP CSF core and peel subshells must be distinct")

    blocks, electron_count = _parse_blocks(
        lines[csf_marker + 1:],
        core_subshells,
        peel_subshells,
    )
    return CsfDocument(
        source=source,
        size_bytes=initial_stat.st_size,
        sha256=hashlib.sha256(content).hexdigest(),
        core_subshells=core_subshells,
        peel_subshells=peel_subshells,
        electron_count=electron_count,
        blocks=blocks,
    )


def _marker_index(lines: list[str], marker: str) -> int:
    matches = [index for index, line in enumerate(lines) if line.strip() == marker]
    if len(matches) != 1:
        raise ValueError(f"GRASP CSF file must contain one {marker!r} header")
    return matches[0]


def _parse_subshell_section(
    lines: list[str],
    section: str,
) -> tuple[str, ...]:
    text = " ".join(lines)
    subshells = tuple(_SUBSHELL_RE.findall(text))
    remainder = _SUBSHELL_RE.sub("", text)
    if remainder.strip():
        raise ValueError(
            f"GRASP CSF {section} subshell section contains invalid text"
        )
    if len(subshells) != len(set(subshells)):
        raise ValueError(f"GRASP CSF {section} subshells contain duplicates")
    return subshells


def _parse_blocks(
    lines: list[str],
    core_subshells: tuple[str, ...],
    peel_subshells: tuple[str, ...],
) -> tuple[tuple[CsfBlock, ...], int]:
    declared = set(core_subshells + peel_subshells)
    core_electrons = sum(_subshell_capacity(label) for label in core_subshells)
    blocks: list[CsfBlock] = []
    current: list[CsfEntry] = []
    electron_count: int | None = None
    cursor = 0
    global_index = 0

    while cursor < len(lines):
        line = lines[cursor]
        if line.startswith(" *"):
            if not current:
                raise ValueError("GRASP CSF file contains an empty block")
            blocks.append(_finish_block(len(blocks) + 1, current))
            current = []
            cursor += 1
            continue
        if cursor + 2 >= len(lines):
            raise ValueError("GRASP CSF file ends inside a three-line CSF")

        occupation_line = line.rstrip()
        quantum_line = lines[cursor + 1].rstrip()
        coupling_line = lines[cursor + 2].rstrip()
        occupations = _parse_occupations(occupation_line, declared)
        two_j, j_label, parity = _parse_symmetry(coupling_line)
        csf_electrons = core_electrons + sum(
            electrons for _, electrons in occupations
        )
        if electron_count is None:
            electron_count = csf_electrons
        elif csf_electrons != electron_count:
            raise ValueError(
                "GRASP CSF configurations do not have one electron count"
            )

        global_index += 1
        current.append(
            CsfEntry(
                block_index=len(blocks) + 1,
                index_within_block=len(current) + 1,
                global_index=global_index,
                occupations=occupations,
                occupation_line=occupation_line,
                subshell_quantum_numbers=quantum_line,
                coupling_and_symmetry=coupling_line,
                two_j=two_j,
                j_label=j_label,
                parity=parity,
            )
        )
        cursor += 3

    if not current:
        if blocks:
            raise ValueError("GRASP CSF file ends with a block delimiter")
        raise ValueError("GRASP CSF file contains no configurations")
    blocks.append(_finish_block(len(blocks) + 1, current))
    assert electron_count is not None
    return tuple(blocks), electron_count


def _parse_occupations(
    line: str,
    declared: set[str],
) -> tuple[tuple[str, int], ...]:
    matches = tuple(
        (match.group(1), int(match.group(2)))
        for match in _OCCUPATION_RE.finditer(line)
    )
    if not matches:
        raise ValueError("GRASP CSF occupation line contains no subshells")
    remainder = _OCCUPATION_RE.sub("", line)
    if remainder.strip():
        raise ValueError("GRASP CSF occupation line contains invalid text")
    labels = tuple(label for label, _ in matches)
    if len(labels) != len(set(labels)):
        raise ValueError("GRASP CSF occupation line repeats a subshell")
    for label, electrons in matches:
        if label not in declared:
            raise ValueError(f"GRASP CSF uses undeclared subshell {label}")
        if not 1 <= electrons <= _subshell_capacity(label):
            raise ValueError(
                f"GRASP CSF occupation for {label} is invalid: {electrons}"
            )
    return matches


def _parse_symmetry(line: str) -> tuple[int, str, str]:
    match = _SYMMETRY_RE.search(line)
    if match is None:
        raise ValueError("GRASP CSF coupling line has no final J/parity")
    numerator = int(match.group(1))
    if match.group(2):
        two_j = numerator
        j_label = f"{numerator}/2"
    else:
        two_j = 2 * numerator
        j_label = str(numerator)
    return two_j, j_label, match.group(3)


def _finish_block(index: int, entries: list[CsfEntry]) -> CsfBlock:
    symmetries = {(entry.two_j, entry.j_label, entry.parity) for entry in entries}
    if len(symmetries) != 1:
        raise ValueError(f"GRASP CSF block {index} contains mixed symmetries")
    two_j, j_label, parity = symmetries.pop()
    return CsfBlock(
        index=index,
        two_j=two_j,
        j_label=j_label,
        parity=parity,
        entries=tuple(entries),
    )


def _subshell_capacity(label: str) -> int:
    angular_label = label.rstrip("-")[-1]
    angular_momentum = _ANGULAR_LABELS.index(angular_label)
    if angular_momentum == 0:
        return 2
    if label.endswith("-"):
        return 2 * angular_momentum
    return 2 * angular_momentum + 2


__all__ = [
    "CsfBlock",
    "CsfDocument",
    "CsfEntry",
    "MAX_GRASP_CSF_BYTES",
    "MAX_GRASP_CSF_LINE_CHARS",
    "load_grasp_csf_list",
]
