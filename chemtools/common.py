from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def parse_scientific_float(value: str) -> float | None:
    try:
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def parse_float_after_delimiter(line: str, delimiter: str) -> float | None:
    if delimiter not in line:
        return None
    tail = line.split(delimiter, 1)[1].split()
    if not tail:
        return None
    return parse_scientific_float(tail[0])


def split_tokens(line: str) -> list[str]:
    return line.split()


def detect_program(contents: str) -> str | None:
    upper = contents.upper()
    if "PROGRAM SYSTEM MOLPRO" in upper or "***  PROGRAM SYSTEM MOLPRO  ***" in upper:
        return "molpro"
    if (
        "THIS RUN OF MOLCAS IS USING THE PYMOLCAS DRIVER" in upper
        or "OPENMOLCASOP" in upper
        or "DEFINITIONS: _MOLCAS_" in upper
    ):
        return "molcas"
    if "NORTHWEST COMPUTATIONAL CHEMISTRY PACKAGE" in upper or "NWCHEM" in upper:
        return "nwchem"
    return None


def json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def make_metadata(path: str | Path, contents: str, program: str | None = None) -> dict[str, Any]:
    detected = program or detect_program(contents)
    return {
        "file": normalize_path(path),
        "program": detected,
    }


LABEL_RE = re.compile(r"^\s*\*\*\*,\s*(.+)")
PROGRAM_RE = re.compile(r"(?i)^\s*PROGRAM \*\s*([^( \n]+.*?)(?:\s*\(|$)")
BASIS_RE = re.compile(r"^\s*SETTING\s+BASIS\s*=\s*(.+)")
CHARGE_RE = re.compile(r"^\s*CHARGE\s*=\s*([+-]?\d+(?:\.\d+)?)")
METHOD_RE = re.compile(r"^\s*\{?\s*([A-Za-z0-9\-\+]+)(?:\s*;|\s|$)")


# Canonical element lookup tables shared across modules
ATOMIC_SYMBOLS: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
    79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
    86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U",
}
ELEMENT_TO_Z: dict[str, int] = {sym: z for z, sym in ATOMIC_SYMBOLS.items()}

