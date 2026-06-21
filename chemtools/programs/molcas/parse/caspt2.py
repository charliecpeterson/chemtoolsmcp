"""Parser for the Molcas CASPT2 module output block.

Goes well beyond what orbitron extracts: per-state final results (reference
energy, E2 non-variational, shift correction, E2 variational, total energy,
residual norm, reference weight, correlation energy contributions), CASPT2
specifications (SS / MS / XMS / RMS / XDW, IPEA shift, real shift, imaginary
shift, sigma-p regularization), Total CASPT2 energies table, MS-CASPT2
effective-Hamiltonian results + eigenvectors, intruder-state warnings, and
quality flags (low reference weight, near-zero denominators).
"""

from __future__ import annotations

import re
from typing import Any


_FLOAT_RE = r"-?\d+\.\d+(?:[Ee][+-]?\d+)?"
_SCI_RE = r"-?\d+\.\d+(?:[EeDd][+-]?\d+)?"


_CASPT2_TYPE_RE = re.compile(r"Type of calculation\s+(\S[^\n]*?)\s*$", re.M)
_FOCK_RE = re.compile(r"Fock operator\s+(\S[^\n]*?)\s*$", re.M)
_H0_TYPE_RE = re.compile(r"0th-order Hamiltonian\s+(\S[^\n]*?)\s*$", re.M)
_IPEA_RE = re.compile(r"IPEA shift\s+(" + _FLOAT_RE + r")")
_REAL_SHIFT_RE = re.compile(r"^\s*Real shift\s+(" + _FLOAT_RE + r")", re.M)
_IMAG_SHIFT_RE = re.compile(r"Imaginary shift\s+(" + _FLOAT_RE + r")")
_SIGMA_P_RE = re.compile(r"Sigma[- ]?p regularization\s+(\S[^\n]*?)\s*$", re.M | re.I)

_GROUP_HEADER_RE = re.compile(r"^\+\+\s*CASPT2 computation for group\s+(\d+)\s*$", re.M)
_FINAL_BLOCK_RE = re.compile(
    r"FINAL CASPT2 RESULT:(?P<body>.*?)(?=\n\n|\+\+|\Z)", re.DOTALL
)

_REF_ENERGY_RE = re.compile(r"Reference energy:\s+(" + _FLOAT_RE + r")")
_E2_NV_RE = re.compile(r"E2 \(Non-variational\):\s+(" + _FLOAT_RE + r")")
_SHIFT_CORR_RE = re.compile(r"Shift correction:\s+(" + _FLOAT_RE + r")")
_E2_VAR_RE = re.compile(r"E2 \(Variational\):\s+(" + _FLOAT_RE + r")")
_TOTAL_E_RE = re.compile(r"Total energy:\s+(" + _FLOAT_RE + r")")
_RESIDUAL_RE = re.compile(r"Residual norm:\s+(" + _FLOAT_RE + r")")
_REF_WEIGHT_RE = re.compile(r"Reference weight:\s+(" + _FLOAT_RE + r")")
_AVO_RE = re.compile(r"Active & Virtual Only:\s+(" + _FLOAT_RE + r")")
_OIE_RE = re.compile(r"One Inactive Excited:\s+(" + _FLOAT_RE + r")")
_TIE_RE = re.compile(r"Two Inactive Excited:\s+(" + _FLOAT_RE + r")")

# Per-root summary table:
#   ::    CASPT2 Root  1     Total energy: -20935.86226486
_CASPT2_ROOT_RE = re.compile(
    r"::\s*CASPT2\s+Root\s+(\d+)\s+Total energy:\s+(" + _FLOAT_RE + r")"
)
_MS_ROOT_RE = re.compile(
    r"::\s*MS-CASPT2\s+Root\s+(\d+)\s+Total energy:\s+(" + _FLOAT_RE + r")"
)

# Variance of |WF0> diagnostic
_VARIANCE_RE = re.compile(r"Variance of \|WF0>:\s+(" + _FLOAT_RE + r")")


def parse_caspt2(text: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "module": "caspt2",
        "specifications": _parse_specifications(text),
        "per_group_results": _parse_groups(text),
        "ss_root_energies": [
            {"root": int(m.group(1)), "energy_hartree": float(m.group(2))}
            for m in _CASPT2_ROOT_RE.finditer(text)
        ],
        "ms_root_energies": [
            {"root": int(m.group(1)), "energy_hartree": float(m.group(2))}
            for m in _MS_ROOT_RE.finditer(text)
        ],
    }
    info["effective_hamiltonian"] = _parse_effective_hamiltonian(text)
    info["intruder_state_report"] = _parse_intruder_report(text)
    info["warnings"] = _classify_warnings(text, info["intruder_state_report"], info["per_group_results"])
    return info


def _parse_specifications(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if (m := _CASPT2_TYPE_RE.search(text)):
        out["calculation_type"] = m.group(1).strip()
    if (m := _FOCK_RE.search(text)):
        out["fock_operator"] = m.group(1).strip()
    if (m := _H0_TYPE_RE.search(text)):
        out["zeroth_order_hamiltonian"] = m.group(1).strip()
    if (m := _IPEA_RE.search(text)):
        out["ipea_shift"] = float(m.group(1))
    if (m := _REAL_SHIFT_RE.search(text)):
        out["real_shift"] = float(m.group(1))
    if (m := _IMAG_SHIFT_RE.search(text)):
        out["imaginary_shift"] = float(m.group(1))
    if (m := _SIGMA_P_RE.search(text)):
        out["sigma_p_regularization"] = m.group(1).strip()
    return out


def _parse_groups(text: str) -> list[dict[str, Any]]:
    """Each ++ CASPT2 computation for group N produces one Final block."""
    groups: list[dict[str, Any]] = []
    headers = list(_GROUP_HEADER_RE.finditer(text))
    for idx, header in enumerate(headers):
        block_start = header.end()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[block_start:block_end]
        final = _FINAL_BLOCK_RE.search(block)
        if not final:
            groups.append({"group": int(header.group(1)), "converged": False})
            continue
        body = final.group("body")
        info: dict[str, Any] = {"group": int(header.group(1)), "converged": True}
        for key, pattern in {
            "reference_energy": _REF_ENERGY_RE,
            "e2_non_variational": _E2_NV_RE,
            "shift_correction": _SHIFT_CORR_RE,
            "e2_variational": _E2_VAR_RE,
            "total_energy": _TOTAL_E_RE,
            "residual_norm": _RESIDUAL_RE,
            "reference_weight": _REF_WEIGHT_RE,
            "correlation_active_virtual_only": _AVO_RE,
            "correlation_one_inactive_excited": _OIE_RE,
            "correlation_two_inactive_excited": _TIE_RE,
        }.items():
            m = pattern.search(body)
            if m:
                info[key] = float(m.group(1))
        # Variance diagnostic from before the final block
        var_m = _VARIANCE_RE.search(block[: final.start()])
        if var_m:
            info["wave_function_variance"] = float(var_m.group(1))
        groups.append(info)
    return groups


def _parse_effective_hamiltonian(text: str) -> dict[str, Any] | None:
    """Extract the MS / XMS effective Hamiltonian matrix and eigenvectors.

    Section format:
       Effective Hamiltonian matrix (Symmetric):
                       1               2               3
            1        -0.86226486
            2         0.00000000     -0.79940459
            3         0.00000000      0.00000105     -0.78429411

       (then later)
       Eigenvectors:
            0.99949415      0.00000000     -0.00001574
            ...
    """
    block_match = re.search(
        r"Effective Hamiltonian matrix \(Symmetric\):\s*(.*?)(?=Total MS-CASPT2 energies|::|\Z)",
        text,
        flags=re.DOTALL,
    )
    if not block_match:
        return None
    matrix_lines = []
    for line in block_match.group(1).splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(tuple("0123456789")):
            tokens = s.split()
            try:
                row_idx = int(tokens[0])
                values = [float(x) for x in tokens[1:]]
                matrix_lines.append({"row": row_idx, "values": values})
            except ValueError:
                continue
    eig_block = re.search(
        r"Eigenvectors:\s*(.*?)(?=\n\s*\n|\+\+|\Z)", text, flags=re.DOTALL
    )
    eigenvectors: list[list[float]] | None = None
    if eig_block:
        eigenvectors = []
        for line in eig_block.group(1).splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                eigenvectors.append([float(x) for x in s.split()])
            except ValueError:
                break
    return {
        "matrix_rows": matrix_lines,
        "eigenvectors": eigenvectors,
        "header_offset_note": _check_offset_note(text),
    }


def _check_offset_note(text: str) -> float | None:
    m = re.search(r"Output diagonal energies have been shifted\. Add\s+(" + _FLOAT_RE + r")", text)
    if m:
        return float(m.group(1))
    return None


def _parse_intruder_report(text: str) -> dict[str, Any]:
    """Statistics over the 'Report on small energy denominators' section.

    A true intruder excitation has BOTH a small denominator (the perturbation
    is divergent) AND a large coefficient (it actually contributes). Either
    alone is normal — large coefficients without small denominators reflect
    chemistry, and small denominators with tiny coefficients reflect benign
    near-degeneracies.
    """
    block = re.search(
        r"Report on small energy denominators.*?(?=\+\+|\Z)", text, flags=re.DOTALL
    )
    if not block:
        return {"row_count": 0, "intruders": [], "large_coefficients_only": []}
    rows = []
    intruders = []
    large_coeffs_only = []
    for line in block.group(0).splitlines():
        tokens = line.split()
        if len(tokens) < 8:
            continue
        try:
            denom = float(tokens[-4])
            coeff = float(tokens[-2])
            contrib = float(tokens[-1])
        except ValueError:
            continue
        rows.append({"denominator": denom, "coefficient": coeff, "contribution": contrib})
        record = {
            "case": tokens[0],
            "denominator": denom,
            "coefficient": coeff,
            "contribution": contrib,
        }
        # True intruder: small denominator AND non-trivial coefficient
        if abs(denom) < 0.3 and abs(coeff) >= 0.05:
            intruders.append(record)
        elif abs(coeff) >= 0.05:
            large_coeffs_only.append(record)
    return {
        "row_count": len(rows),
        "intruders": sorted(intruders, key=lambda r: -abs(r["coefficient"]))[:10],
        "large_coefficients_only": sorted(large_coeffs_only, key=lambda r: -abs(r["coefficient"]))[:5],
    }


def _classify_warnings(
    text: str,
    intruder_report: dict[str, Any],
    per_group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate CASPT2-specific concerns into agent-actionable warnings."""
    warnings: list[dict[str, Any]] = []

    if "WARNING: User-modified 0th-order Hamiltonian" in text:
        warnings.append(
            {
                "code": "user_modified_h0",
                "severity": "info",
                "message": "User explicitly set IPEA shift (overriding default).",
            }
        )

    if "WARNING: User changed nr of frozen orbitals" in text:
        warnings.append(
            {
                "code": "user_changed_frozen",
                "severity": "info",
                "message": "User explicitly set frozen orbital count.",
            }
        )

    # Real intruder: small denominator AND large coefficient.
    intruders = intruder_report.get("intruders") or []
    if intruders:
        max_coeff = max(abs(r["coefficient"]) for r in intruders)
        min_denom = min(abs(r["denominator"]) for r in intruders)
        # Don't suggest IMAGINARY SHIFT if it's already on
        warnings.append(
            {
                "code": "intruder_state",
                "severity": "high" if max_coeff >= 0.10 or min_denom < 0.1 else "medium",
                "message": (
                    f"CASPT2 has {len(intruders)} intruder excitation(s): denominator "
                    f"< 0.3 hartree AND |coefficient| >= 0.05 (max |coeff| {max_coeff:.3f}, "
                    f"min |denom| {min_denom:.3f}). If shift is not already active, "
                    "consider IMAGINARY SHIFT 0.1-0.2 or SIG2 regularization, "
                    "or expand the active space."
                ),
                "max_coefficient": max_coeff,
                "min_denominator": min_denom,
                "row_count": len(intruders),
            }
        )

    # Low reference weight → flag explicitly.
    low_weight = [g for g in per_group if (g.get("reference_weight") or 1.0) < 0.70]
    if low_weight:
        warnings.append(
            {
                "code": "low_reference_weight",
                "severity": "high",
                "message": (
                    f"{len(low_weight)} CASPT2 group(s) have reference weight < 0.70 — "
                    "active space is likely insufficient."
                ),
                "groups": [g["group"] for g in low_weight],
            }
        )

    return warnings


def assess_reference_weights(per_group_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Standalone helper: rate each group's reference weight on a quality scale.

    Empirical thresholds:
      >= 0.85 — healthy
      0.70 - 0.85 — caution (active-space might be missing important configs)
      < 0.70  — likely unreliable; revisit active space
    """
    out: list[dict[str, Any]] = []
    for group in per_group_results:
        rw = group.get("reference_weight")
        if rw is None:
            continue
        if rw >= 0.85:
            quality = "healthy"
        elif rw >= 0.70:
            quality = "caution"
        else:
            quality = "unreliable"
        out.append(
            {
                "group": group["group"],
                "reference_weight": rw,
                "quality": quality,
                "advice": {
                    "healthy": "OK; reference is dominant.",
                    "caution": "Reference weight is low — consider adding π* / Rydberg orbitals to the active space.",
                    "unreliable": "Reference weight too low; revisit active-space selection before trusting CASPT2.",
                }[quality],
            }
        )
    return out
