from __future__ import annotations

# nwchem.py — re-export shim for backward compatibility
# Code that does `from . import nwchem; nwchem.parse_tasks(...)` still works.
# Code that does `from .nwchem import parse_tasks` also works.
from chemtools.programs.nwchem.parse.tasks import *  # noqa: F401,F403
from chemtools.programs.nwchem.parse.mos import *  # noqa: F401,F403
from chemtools.programs.nwchem.parse.freq import *  # noqa: F401,F403
