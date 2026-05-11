"""NWChem Drafter sub-protocol implementation.

Adapter layer that bridges the program-neutral `InputSpec` from
`chemtools.core.types` to the existing NWChem input drafters in
`chemtools.api_input`. Three responsibilities:

  * `draft_input(spec)`  — render an NWChem .nw file from an InputSpec
  * `lint_input(text)`   — validate input text, return LintIssue records
  * `patch_input(text, change)` — apply structured changes (TODO: not yet
    implemented; the existing API exposes specific draft_*_variant tools
    that will be unified here when api_input.py is split into families).

`draft_input` handles the InputSpec → create_nwchem_input translation:
  - inline atoms are written to a temp .xyz file for load_geometry_source
  - `method` ("DFT" / "CCSD(T)" / "CASSCF" / ...) maps to an NWChem module
  - `basis` (str or per-element dict) maps to basis_assignments
  - `functional` becomes an `xc` line in the DFT module settings
  - `program_options` provides escape hatches (library_path, extra_blocks,
    module_settings, vectors_input/output, memory, start_name)

Unsupported InputSpec fields (solvent, ecp at the spec top level, ...) are
either passed through where they fit or ignored with a warning in the
returned dict.
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Any

from chemtools.core.types import InputSpec, LintIssue


# Map InputSpec.method to NWChem's `task <module>` value. Method strings are
# matched case-insensitively against the prefix patterns in this list.
_METHOD_TO_MODULE: list[tuple[str, str]] = [
    # Most specific first.
    ("ccsd(t)",  "tce"),
    ("ccsdt",    "tce"),
    ("ccsd",     "tce"),
    ("mp2",      "tce"),
    ("eom-ccsd", "tce"),
    ("casscf",   "mcscf"),
    ("mcscf",    "mcscf"),
    ("tddft",    "tddft"),
    ("dft",      "dft"),
    ("hf",       "scf"),
    ("scf",      "scf"),
]


def _method_to_module(method: str) -> str:
    """Resolve a free-form method string to an NWChem module name."""
    m = method.strip().lower()
    for prefix, module in _METHOD_TO_MODULE:
        if m.startswith(prefix):
            return module
    raise ValueError(f"Unrecognized method {method!r}; provide program_options['module'] explicitly")


def _atoms_to_xyz_text(atoms: list[dict[str, Any]], title: str = "") -> str:
    """Render a list of GeometryAtom dicts to xyz file text."""
    lines = [str(len(atoms)), title]
    for a in atoms:
        elem = a["element"]
        x, y, z = a["x"], a["y"], a["z"]
        lines.append(f"{elem:<3s} {x:>16.10f} {y:>16.10f} {z:>16.10f}")
    return "\n".join(lines) + "\n"


def _default_basis_library_path() -> str:
    """Bundled NWChem basis library."""
    import chemtools
    return str(Path(chemtools.__file__).resolve().parent / "data" / "nwchem" / "basis_library")


class _NwchemDrafter:
    """Implements chemtools.core.program.Drafter for NWChem."""

    def draft_input(self, spec: InputSpec) -> str:
        """Render a complete .nw input file from a program-neutral InputSpec."""
        # Lazy import — api_input is still flat; later it splits into
        # programs/nwchem/input/{scf,dft,tce,...}.py and this becomes cleaner.
        from chemtools.api_input import create_nwchem_input

        # ---- 1. Geometry: inline atoms or program_options['geometry_path'] ----
        program_options = spec.get("program_options") or {}
        geom_path = program_options.get("geometry_path")
        atoms = spec.get("atoms") or []
        cleanup_tempfile = None
        if not geom_path:
            if not atoms:
                raise ValueError(
                    "InputSpec needs either atoms=[...] or "
                    "program_options['geometry_path']=..."
                )
            # Write inline atoms to a temp .xyz file.
            title = spec.get("title", "")
            xyz_text = _atoms_to_xyz_text(atoms, title=title)
            fh = tempfile.NamedTemporaryFile(
                mode="w", suffix=".xyz", delete=False, encoding="utf-8"
            )
            fh.write(xyz_text)
            fh.close()
            geom_path = fh.name
            cleanup_tempfile = fh.name

        # ---- 2. Method → NWChem module + task operation ----
        method = spec.get("method", "DFT")
        module = program_options.get("module") or _method_to_module(method)
        task_operation = spec.get("task", "energy")

        # ---- 3. Basis: str or per-element dict ----
        basis = spec.get("basis", "def2-TZVP")
        if isinstance(basis, str):
            elements = sorted({a["element"] for a in atoms}) if atoms else []
            basis_assignments = {e: basis for e in elements}
            default_basis = basis
        else:
            basis_assignments = dict(basis)
            default_basis = None

        # ---- 4. ECP, functional, module-specific settings ----
        ecp = spec.get("ecp") or {}
        module_settings: list[str] = list(program_options.get("module_settings") or [])
        if module == "dft":
            functional = spec.get("functional")
            if functional:
                module_settings.insert(0, f"xc {functional}")
        extra_blocks = list(program_options.get("extra_blocks") or [])

        # ---- 5. Library path ----
        library_path = (
            program_options.get("library_path")
            or _default_basis_library_path()
        )

        try:
            result = create_nwchem_input(
                geometry_path=geom_path,
                library_path=library_path,
                basis_assignments=basis_assignments,
                module=module,
                task_operation=task_operation,
                ecp_assignments=ecp,
                default_basis=default_basis,
                charge=spec.get("charge"),
                multiplicity=spec.get("multiplicity"),
                module_settings=module_settings,
                extra_blocks=extra_blocks,
                memory=program_options.get("memory"),
                title=spec.get("title") or program_options.get("title"),
                start_name=program_options.get("start_name"),
                vectors_input=program_options.get("vectors_input"),
                vectors_output=program_options.get("vectors_output"),
                write_file=False,
                inline_blocks=True,
            )
        finally:
            if cleanup_tempfile:
                try:
                    os.unlink(cleanup_tempfile)
                except OSError:
                    pass

        return result.get("input_text") or result.get("text") or ""

    def lint_input(self, text: str) -> list[LintIssue]:
        """Validate input text. Writes to a temp file so we can reuse
        the path-based lint_nwchem_input, then translates its issue list
        into the LintIssue TypedDict shape."""
        from chemtools.api_input import lint_nwchem_input

        fh = tempfile.NamedTemporaryFile(
            mode="w", suffix=".nw", delete=False, encoding="utf-8"
        )
        fh.write(text)
        fh.close()
        try:
            result = lint_nwchem_input(fh.name)
        finally:
            try:
                os.unlink(fh.name)
            except OSError:
                pass

        issues_out: list[LintIssue] = []
        for issue in result.get("issues") or []:
            level = issue.get("level") or issue.get("severity") or "warning"
            if level not in {"error", "warning", "info"}:
                level = "warning"
            issues_out.append({
                "level": level,
                "message": issue.get("message") or issue.get("code") or "",
                "line": issue.get("line"),
                "suggested_fix": (issue.get("details") or {}).get("suggested_fix"),
            })
        return issues_out

    def patch_input(self, text: str, change: dict[str, Any]) -> str:
        """Apply a structured change to an existing NWChem input.

        Delegates to `create_nwchem_input_variant`, which understands a set
        of dotted keyword paths:

          * "memory"                     -> replaces the `memory` line
          * "charge"                     -> top-level charge directive
          * "mult"                       -> SCF/DFT multiplicity (nopen)
          * "task"                       -> replaces the last task line
          * "dft.xc"                     -> functional inside the dft block
          * "dft.iterations"             -> dft iter count
          * "dft.convergence energy"     -> dft conv. threshold
          * "scf.maxiter"                -> scf iter count
          * any other "block.keyword"    -> best-effort in-block replace

        `change` is expected to map these keys to string values; an
        optional `change["reason"]` (without dots) is treated as commentary
        rather than a directive change.

        For the bigger structural rewrites that the legacy variant tools
        handle (vectors swap, SCF stabilization, property check,
        optimization follow-up), the agent should call those tools
        directly — they re-render the module body, not just a keyword.
        """
        # Lazy import — keeps the Drafter sub-protocol importable even if
        # input/ is still being split.
        from chemtools.programs.nwchem.input.general import create_nwchem_input_variant

        # Separate the optional "reason" key from the actual changes.
        change = dict(change or {})
        reason = change.pop("reason", "") if isinstance(change.get("reason"), str) else ""

        # Write the input text to a temp file so the variant tool can read it
        # via load_geometry_source / inspect_nwchem_input, then ask it not to
        # write the patched result back to disk — we want the text back, not
        # a versioned file.
        fh = tempfile.NamedTemporaryFile(
            mode="w", suffix=".nw", delete=False, encoding="utf-8"
        )
        fh.write(text)
        fh.close()
        try:
            result = create_nwchem_input_variant(
                source_input=fh.name,
                changes=change,
                reason=reason,
                output_path=None,
                write_file=False,
            )
        finally:
            try:
                os.unlink(fh.name)
            except OSError:
                pass

        return result.get("input_text") or text


NWCHEM_DRAFTER = _NwchemDrafter()


__all__ = ["NWCHEM_DRAFTER"]
