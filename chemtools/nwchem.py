from __future__ import annotations

# nwchem.py — re-export shim for backward compatibility
# Code that does `from . import nwchem; nwchem.parse_tasks(...)` still works.
# Code that does `from .nwchem import parse_tasks` also works.
from .nwchem_tasks import *  # noqa: F401,F403
from .nwchem_mos import *  # noqa: F401,F403
from .nwchem_freq import *  # noqa: F401,F403
