#!/usr/bin/env python3
"""Compatibility shim — server moved to chemtools/mcp/nwchem.py.

Prefer using the `chemtools-nwchem` entry point (after pip install -e .)
or point your config directly at chemtools/mcp/nwchem.py.
"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "chemtools" / "mcp" / "nwchem.py"),
    run_name="__main__",
)
