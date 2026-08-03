"""Validate versioned Orbitron chemistry-analysis payloads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from chemtools.core.units import HARTREE_TO_EV


_GEOMETRY_ROLES = frozenset(
    {"input", "single_point", "converged_final", "last_attempted"}
)

_ANY_SPIN = object()


class OrbitronPayloadError(ValueError):
    """An Orbitron payload contradicts its operation contract."""


def validate_orbital_analysis(
    payload: dict[str, Any],
    source_path: Path,
) -> None:
    contract = "Orbitron orbital analysis"
    _validate_source(payload, source_path, contract)

    total = _nonnegative_int(payload, "total_orbitals", contract)
    occupied = _nonnegative_int(payload, "occupied_count", contract)
    virtual = _nonnegative_int(payload, "virtual_count", contract)
    if total == 0:
        raise OrbitronPayloadError(
            f"{contract} must contain at least one orbital"
        )
    if occupied + virtual != total:
        raise OrbitronPayloadError(
            f"{contract} occupied and virtual counts do not equal total_orbitals"
        )

    threshold = _finite_number(payload, "occupied_threshold", contract)
    if threshold < 0:
        raise OrbitronPayloadError(
            f"{contract} occupied_threshold must be non-negative"
        )
    unrestricted = payload.get("unrestricted")
    if not isinstance(unrestricted, bool):
        raise OrbitronPayloadError(f"{contract} unrestricted must be boolean")

    channels = _validate_spin_channels(payload, threshold, unrestricted)
    if sum(channel["orbital_count"] for channel in channels) != total:
        raise OrbitronPayloadError(
            f"{contract} spin-channel orbital counts do not equal total_orbitals"
        )
    if sum(channel["occupied_count"] for channel in channels) != occupied:
        raise OrbitronPayloadError(
            f"{contract} spin-channel occupied counts do not equal occupied_count"
        )
    if sum(channel["virtual_count"] for channel in channels) != virtual:
        raise OrbitronPayloadError(
            f"{contract} spin-channel virtual counts do not equal virtual_count"
        )

    expected_top_frontier = [
        orbital
        for channel in channels
        for orbital in channel["frontier"]
    ]
    if unrestricted:
        for field_name in ("homo", "lumo", "gap_hartree", "gap_ev"):
            if payload.get(field_name) is not None:
                raise OrbitronPayloadError(
                    f"{contract} unrestricted top-level {field_name} must be null"
                )
        top_frontier = _validate_frontier_list(
            payload.get("frontier"),
            f"{contract} frontier",
            max_entries=12,
        )
    else:
        restricted = channels[0]
        for field_name in ("homo", "lumo", "gap_hartree", "gap_ev"):
            if payload.get(field_name) != restricted[field_name]:
                raise OrbitronPayloadError(
                    f"{contract} top-level {field_name} does not match the "
                    "restricted channel"
                )
        top_frontier = _validate_frontier_list(
            payload.get("frontier"),
            f"{contract} frontier",
            max_entries=6,
            expected_spin=None,
        )
    if top_frontier != expected_top_frontier:
        raise OrbitronPayloadError(
            f"{contract} frontier does not match the spin-channel frontiers"
        )


def validate_population_analysis(
    payload: dict[str, Any],
    source_path: Path,
) -> None:
    contract = "Orbitron population analysis"
    _validate_source(payload, source_path, contract)
    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise OrbitronPayloadError(
            f"{contract} methods must be a non-empty list"
        )

    names = []
    for index, method in enumerate(methods):
        if not isinstance(method, dict):
            raise OrbitronPayloadError(
                f"{contract} method {index} must be an object"
            )
        names.append(_validate_population_method(method, index))
    if len(names) != len(set(names)):
        raise OrbitronPayloadError(
            f"{contract} method names must be unique"
        )


def _validate_spin_channels(
    payload: dict[str, Any],
    occupied_threshold: float,
    unrestricted: bool,
) -> list[dict[str, Any]]:
    contract = "Orbitron orbital analysis"
    channels = payload.get("spin_channels")
    if not isinstance(channels, list):
        raise OrbitronPayloadError(f"{contract} spin_channels must be a list")
    expected_spins = ["alpha", "beta"] if unrestricted else ["restricted"]
    observed_spins = [
        channel.get("spin") if isinstance(channel, dict) else None
        for channel in channels
    ]
    if observed_spins != expected_spins:
        raise OrbitronPayloadError(
            f"{contract} spin channels must be {expected_spins}"
        )

    validated = []
    for channel in channels:
        spin = channel["spin"]
        channel_contract = f"{contract} {spin} channel"
        orbital_count = _nonnegative_int(channel, "orbital_count", channel_contract)
        occupied_count = _nonnegative_int(
            channel, "occupied_count", channel_contract
        )
        virtual_count = _nonnegative_int(channel, "virtual_count", channel_contract)
        if occupied_count + virtual_count != orbital_count:
            raise OrbitronPayloadError(
                f"{channel_contract} occupied and virtual counts do not equal "
                "orbital_count"
            )
        expected_spin = None if spin == "restricted" else spin
        homo, lumo, gap_hartree, gap_ev, frontier = _validate_orbital_partition(
            channel,
            channel_contract,
            expected_spin,
            occupied_threshold,
        )
        validated.append(
            {
                **channel,
                "homo": homo,
                "lumo": lumo,
                "gap_hartree": gap_hartree,
                "gap_ev": gap_ev,
                "frontier": frontier,
            }
        )
    return validated


def _validate_orbital_partition(
    mapping: dict[str, Any],
    contract: str,
    expected_spin: str | None,
    occupied_threshold: float,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    float | None,
    float | None,
    list[dict[str, Any]],
]:
    homo = _validate_frontier_orbital(
        mapping.get("homo"),
        f"{contract} homo",
        expected_spin=expected_spin,
    )
    lumo = _validate_frontier_orbital(
        mapping.get("lumo"),
        f"{contract} lumo",
        expected_spin=expected_spin,
    )
    if mapping["occupied_count"] and homo is None:
        raise OrbitronPayloadError(f"{contract} has occupied orbitals but no HOMO")
    if mapping["virtual_count"] and lumo is None:
        raise OrbitronPayloadError(f"{contract} has virtual orbitals but no LUMO")
    if homo is not None and homo["occupancy"] <= occupied_threshold:
        raise OrbitronPayloadError(
            f"{contract} HOMO occupancy does not exceed occupied_threshold"
        )
    if lumo is not None and lumo["occupancy"] > occupied_threshold:
        raise OrbitronPayloadError(
            f"{contract} LUMO occupancy exceeds occupied_threshold"
        )

    gap_hartree = _optional_finite_number(mapping, "gap_hartree", contract)
    gap_ev = _optional_finite_number(mapping, "gap_ev", contract)
    if (gap_hartree is None) != (gap_ev is None):
        raise OrbitronPayloadError(
            f"{contract} gap_hartree and gap_ev must appear together"
        )
    if gap_hartree is not None and gap_ev is not None:
        if gap_hartree < 0 or gap_ev < 0:
            raise OrbitronPayloadError(
                f"{contract} HOMO-LUMO gap must be non-negative"
            )
        _require_energy_conversion(gap_hartree, gap_ev, f"{contract} gap")
    if homo is not None and lumo is not None:
        expected_gap = lumo["energy_hartree"] - homo["energy_hartree"]
        if gap_hartree is None or not math.isclose(
            gap_hartree,
            expected_gap,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise OrbitronPayloadError(
                f"{contract} gap does not equal LUMO minus HOMO energy"
            )

    frontier = _validate_frontier_list(
        mapping.get("frontier"),
        f"{contract} frontier",
        max_entries=6,
        expected_spin=expected_spin,
    )
    if homo is not None and homo not in frontier:
        raise OrbitronPayloadError(
            f"{contract} frontier does not contain the reported HOMO"
        )
    if lumo is not None and lumo not in frontier:
        raise OrbitronPayloadError(
            f"{contract} frontier does not contain the reported LUMO"
        )
    return homo, lumo, gap_hartree, gap_ev, frontier


def _validate_frontier_list(
    value: object,
    contract: str,
    *,
    max_entries: int,
    expected_spin: object = _ANY_SPIN,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise OrbitronPayloadError(f"{contract} must be a list")
    if len(value) > max_entries:
        raise OrbitronPayloadError(
            f"{contract} exceeds the fixed {max_entries}-orbital limit"
        )
    orbitals = [
        _validate_frontier_orbital(
            orbital,
            f"{contract}[{index}]",
            optional=False,
            expected_spin=expected_spin,
        )
        for index, orbital in enumerate(value)
    ]
    labels = [orbital["label"] for orbital in orbitals]
    if len(labels) != len(set(labels)):
        raise OrbitronPayloadError(f"{contract} labels must be unique")
    return orbitals


def validate_vibration_analysis(
    payload: dict[str, Any],
    source_path: Path,
) -> None:
    contract = "Orbitron vibration analysis"
    _validate_source(payload, source_path, contract)
    validate_geometry_provenance(payload, contract)
    if payload.get("mode_set") != "raw":
        raise OrbitronPayloadError(
            f"{contract} mode_set must be raw for the fixed Chemtools operation"
        )
    if payload.get("frequency_unit") != "cm^-1":
        raise OrbitronPayloadError(f"{contract} frequency_unit must be cm^-1")
    scale_factor = _finite_number(payload, "frequency_scale_factor", contract)
    if scale_factor != 1.0:
        raise OrbitronPayloadError(
            f"{contract} frequency_scale_factor must be 1.0"
        )
    has_displacements = payload.get("has_displacements")
    if not isinstance(has_displacements, bool):
        raise OrbitronPayloadError(
            f"{contract} has_displacements must be boolean"
        )
    mode_count = _nonnegative_int(payload, "mode_count", contract)
    displacement_mode_count = _nonnegative_int(
        payload, "displacement_mode_count", contract
    )
    if displacement_mode_count > mode_count:
        raise OrbitronPayloadError(
            f"{contract} displacement_mode_count exceeds mode_count"
        )
    expected_has_displacements = (
        mode_count > 0 and displacement_mode_count == mode_count
    )
    if has_displacements != expected_has_displacements:
        raise OrbitronPayloadError(
            f"{contract} has_displacements disagrees with displacement_mode_count"
        )
    imaginary_count = _nonnegative_int(payload, "imaginary_count", contract)
    if imaginary_count > mode_count:
        raise OrbitronPayloadError(
            f"{contract} imaginary_count exceeds mode_count"
        )

    statistics = [
        _optional_finite_number(payload, field_name, contract)
        for field_name in (
            "lowest_frequency",
            "highest_frequency",
            "mean_frequency",
        )
    ]
    modes = payload.get("modes")
    if not isinstance(modes, list):
        raise OrbitronPayloadError(f"{contract} modes must be a list")
    expected_sample_count = min(10, mode_count)
    if len(modes) != expected_sample_count:
        raise OrbitronPayloadError(
            f"{contract} modes length does not match the fixed top-ten window"
        )

    if mode_count == 0:
        if any(statistic is not None for statistic in statistics):
            raise OrbitronPayloadError(
                f"{contract} empty mode set must have null statistics"
            )
    elif any(statistic is None for statistic in statistics):
        raise OrbitronPayloadError(
            f"{contract} non-empty mode set must have complete statistics"
        )

    mode_entries = [
        _validate_vibration_mode(mode, index)
        for index, mode in enumerate(modes)
    ]
    indices = [mode["index"] for mode in mode_entries]
    if len(indices) != len(set(indices)):
        raise OrbitronPayloadError(f"{contract} mode indices must be unique")
    frequencies = [mode["frequency"] for mode in mode_entries]
    if any(left > right for left, right in zip(frequencies, frequencies[1:])):
        raise OrbitronPayloadError(
            f"{contract} sampled modes must be sorted by frequency"
        )
    sampled_imaginary = sum(frequency < 0 for frequency in frequencies)
    if sampled_imaginary != min(imaginary_count, expected_sample_count):
        raise OrbitronPayloadError(
            f"{contract} sampled imaginary modes disagree with imaginary_count"
        )
    sampled_displacements = sum(
        mode["has_displacement"] for mode in mode_entries
    )
    if sampled_displacements > displacement_mode_count:
        raise OrbitronPayloadError(
            f"{contract} sampled displacement count exceeds displacement_mode_count"
        )
    if mode_count <= 10 and sampled_displacements != displacement_mode_count:
        raise OrbitronPayloadError(
            f"{contract} sampled displacement count disagrees with "
            "displacement_mode_count"
        )
    if has_displacements and sampled_displacements != len(mode_entries):
        raise OrbitronPayloadError(
            f"{contract} sampled modes contradict has_displacements"
        )

    if mode_count:
        lowest, highest, mean = statistics
        _require_close(frequencies[0], lowest, f"{contract} lowest_frequency")
        if mode_count <= 10:
            _require_close(
                frequencies[-1],
                highest,
                f"{contract} highest_frequency",
            )
            _require_close(
                sum(frequencies) / mode_count,
                mean,
                f"{contract} mean_frequency",
            )
        if lowest > mean or mean > highest:
            raise OrbitronPayloadError(
                f"{contract} frequency statistics are inconsistent"
            )

    _validate_thermochemistry(payload.get("thermochemistry"))


def validate_geometry_provenance(
    payload: dict[str, Any],
    contract: str,
) -> None:
    geometry_role = payload.get("geometry_role")
    if geometry_role not in _GEOMETRY_ROLES:
        raise OrbitronPayloadError(
            f"{contract} geometry_role must be one of "
            f"{sorted(_GEOMETRY_ROLES)}"
        )
    geometry_source = payload.get("geometry_source")
    if not isinstance(geometry_source, str) or not geometry_source.strip():
        raise OrbitronPayloadError(
            f"{contract} geometry_source must be a non-empty string"
        )


def _validate_vibration_mode(
    value: object,
    mode_index: int,
) -> dict[str, Any]:
    contract = f"Orbitron vibration analysis mode {mode_index}"
    if not isinstance(value, dict):
        raise OrbitronPayloadError(f"{contract} must be an object")
    index = value.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
        raise OrbitronPayloadError(f"{contract} index must be positive")
    frequency = _finite_number(value, "frequency", contract)
    magnitude = _finite_number(value, "magnitude", contract)
    if magnitude < 0:
        raise OrbitronPayloadError(f"{contract} magnitude must be non-negative")
    _require_close(magnitude, abs(frequency), f"{contract} magnitude")
    label = value.get("label")
    if label is not None and not isinstance(label, str):
        raise OrbitronPayloadError(f"{contract} label must be a string or null")
    if not isinstance(value.get("has_displacement"), bool):
        raise OrbitronPayloadError(
            f"{contract} has_displacement must be boolean"
        )
    return value


def _validate_thermochemistry(value: object) -> None:
    if value is None:
        return
    contract = "Orbitron vibration analysis thermochemistry"
    if not isinstance(value, dict):
        raise OrbitronPayloadError(f"{contract} must be an object or null")
    temperature = _finite_number(value, "temperature_kelvin", contract)
    if temperature <= 0:
        raise OrbitronPayloadError(
            f"{contract} temperature_kelvin must be positive"
        )
    pressure = _optional_finite_number(value, "pressure_atm", contract)
    if pressure is not None and pressure <= 0:
        raise OrbitronPayloadError(f"{contract} pressure_atm must be positive")
    for field_name in (
        "zero_point_correction_kcal_mol",
        "thermal_correction_energy_kcal_mol",
        "thermal_correction_enthalpy_kcal_mol",
        "thermal_correction_gibbs_kcal_mol",
        "total_entropy_cal_mol_k",
        "cv_total_cal_mol_k",
        "molecular_weight_amu",
    ):
        field_value = _optional_finite_number(value, field_name, contract)
        if field_name == "molecular_weight_amu" and field_value is not None:
            if field_value <= 0:
                raise OrbitronPayloadError(
                    f"{contract} molecular_weight_amu must be positive"
                )
    symmetry_number = value.get("symmetry_number")
    if symmetry_number is not None and (
        isinstance(symmetry_number, bool)
        or not isinstance(symmetry_number, int)
        or symmetry_number <= 0
    ):
        raise OrbitronPayloadError(
            f"{contract} symmetry_number must be a positive integer or null"
        )


def _validate_source(
    payload: dict[str, Any],
    source_path: Path,
    contract: str,
) -> None:
    if payload.get("path") != str(source_path):
        raise OrbitronPayloadError(
            f"{contract} path does not match the source"
        )
    output_format = payload.get("format")
    if output_format is not None and not isinstance(output_format, str):
        raise OrbitronPayloadError(
            f"{contract} format must be a string or null"
        )


def _validate_population_method(
    method: dict[str, Any],
    method_index: int,
) -> str:
    contract = f"Orbitron population analysis method {method_index}"
    name = method.get("method")
    if not isinstance(name, str) or not name:
        raise OrbitronPayloadError(f"{contract} name must be non-empty")
    atom_count = _nonnegative_int(method, "atom_count", contract)
    if atom_count == 0:
        raise OrbitronPayloadError(f"{contract} atom_count must be positive")

    charges = method.get("charges")
    if not isinstance(charges, list) or len(charges) != atom_count:
        raise OrbitronPayloadError(
            f"{contract} charges length does not equal atom_count"
        )
    charge_entries = [
        _validate_atom_charge(entry, contract) for entry in charges
    ]
    atom_indices = [entry["atom_index"] for entry in charge_entries]
    if len(atom_indices) != len(set(atom_indices)):
        raise OrbitronPayloadError(
            f"{contract} atom indices must be unique"
        )
    absolute_charges = [abs(entry["charge"]) for entry in charge_entries]
    if any(
        left < right
        for left, right in zip(absolute_charges, absolute_charges[1:])
    ):
        raise OrbitronPayloadError(
            f"{contract} charges must be sorted by descending magnitude"
        )

    charge_values = [entry["charge"] for entry in charge_entries]
    total_charge = _finite_number(method, "total_charge", contract)
    expected_total_charge = _optional_finite_number(
        method, "expected_total_charge", contract
    )
    expected_charge_source = method.get("expected_charge_source")
    charge_residual = _optional_finite_number(method, "charge_residual", contract)
    if expected_total_charge is None:
        if expected_charge_source is not None or charge_residual is not None:
            raise OrbitronPayloadError(
                f"{contract} expected charge source and residual must be null "
                "when expected_total_charge is null"
            )
    else:
        if expected_charge_source not in {"declared", "formal_charges"}:
            raise OrbitronPayloadError(
                f"{contract} expected_charge_source must identify declared or "
                "formal charges"
            )
        if charge_residual is None:
            raise OrbitronPayloadError(
                f"{contract} charge_residual is required with expected_total_charge"
            )
        _require_close(
            charge_residual,
            total_charge - expected_total_charge,
            f"{contract} charge_residual",
        )
    min_charge = _finite_number(method, "min_charge", contract)
    max_charge = _finite_number(method, "max_charge", contract)
    mean_abs_charge = _finite_number(method, "mean_abs_charge", contract)
    _require_close(total_charge, sum(charge_values), f"{contract} total_charge")
    _require_close(min_charge, min(charge_values), f"{contract} min_charge")
    _require_close(max_charge, max(charge_values), f"{contract} max_charge")
    _require_close(
        mean_abs_charge,
        sum(absolute_charges) / atom_count,
        f"{contract} mean_abs_charge",
    )

    charges_by_atom = method.get("charges_by_atom")
    expected_by_atom = {
        str(entry["atom_index"]): entry for entry in charge_entries
    }
    if charges_by_atom != expected_by_atom:
        raise OrbitronPayloadError(
            f"{contract} charges_by_atom does not match charges"
        )

    top_charges = method.get("top_charges")
    expected_top = charge_entries[:min(8, atom_count)]
    if top_charges != expected_top:
        raise OrbitronPayloadError(
            f"{contract} top_charges does not match the fixed top-eight window"
        )
    warnings = method.get("warnings")
    if (
        not isinstance(warnings, list)
        or any(not isinstance(warning, str) for warning in warnings)
    ):
        raise OrbitronPayloadError(
            f"{contract} warnings must be a list of strings"
        )
    return name


def _validate_atom_charge(
    value: object,
    contract: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OrbitronPayloadError(
            f"{contract} charge entry must be an object"
        )
    atom_index = value.get("atom_index")
    if (
        isinstance(atom_index, bool)
        or not isinstance(atom_index, int)
        or atom_index < 0
    ):
        raise OrbitronPayloadError(
            f"{contract} atom_index must be a non-negative integer"
        )
    element = value.get("element")
    if not isinstance(element, str) or not element:
        raise OrbitronPayloadError(
            f"{contract} element must be non-empty"
        )
    _finite_number(value, "charge", contract)
    return value


def _validate_frontier_orbital(
    value: object,
    field_name: str,
    *,
    optional: bool = True,
    expected_spin: object = _ANY_SPIN,
) -> dict[str, Any] | None:
    contract = field_name
    if value is None and optional:
        return None
    if not isinstance(value, dict):
        raise OrbitronPayloadError(f"{contract} must be an object")
    label = value.get("label")
    if not isinstance(label, str) or not label:
        raise OrbitronPayloadError(f"{contract} label must be non-empty")
    vector = value.get("vector")
    if isinstance(vector, bool) or not isinstance(vector, int) or vector <= 0:
        raise OrbitronPayloadError(f"{contract} vector must be positive")
    energy_hartree = _finite_number(value, "energy_hartree", contract)
    energy_ev = _finite_number(value, "energy_ev", contract)
    _require_energy_conversion(energy_hartree, energy_ev, field_name)
    occupancy = _finite_number(value, "occupancy", contract)
    if occupancy < 0:
        raise OrbitronPayloadError(
            f"{contract} occupancy must be non-negative"
        )
    symmetry = value.get("symmetry")
    if symmetry is not None and not isinstance(symmetry, str):
        raise OrbitronPayloadError(
            f"{contract} symmetry must be a string or null"
        )
    spin = value.get("spin")
    if spin is not None and spin not in {"alpha", "beta"}:
        raise OrbitronPayloadError(
            f"{contract} spin must be alpha, beta, or null"
        )
    if expected_spin is not _ANY_SPIN and spin != expected_spin:
        raise OrbitronPayloadError(
            f"{contract} spin does not match its channel"
        )
    return value


def _nonnegative_int(
    mapping: dict[str, Any],
    field_name: str,
    contract: str,
) -> int:
    value = mapping.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OrbitronPayloadError(
            f"{contract} {field_name} must be a non-negative integer"
        )
    return value


def _optional_finite_number(
    mapping: dict[str, Any],
    field_name: str,
    contract: str,
) -> float | None:
    if mapping.get(field_name) is None:
        return None
    return _finite_number(mapping, field_name, contract)


def _finite_number(
    mapping: dict[str, Any],
    field_name: str,
    contract: str,
) -> float:
    value = mapping.get(field_name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise OrbitronPayloadError(
            f"{contract} {field_name} must be finite"
        )
    return float(value)


def _require_close(observed: float, expected: float, field_name: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise OrbitronPayloadError(f"{field_name} does not match its derived value")


def _require_energy_conversion(
    energy_hartree: float,
    energy_ev: float,
    field_name: str,
) -> None:
    if not math.isclose(
        energy_ev,
        energy_hartree * HARTREE_TO_EV,
        rel_tol=1e-10,
        abs_tol=1e-9,
    ):
        raise OrbitronPayloadError(
            f"Orbitron orbital analysis {field_name} Hartree/eV values disagree"
        )


__all__ = [
    "OrbitronPayloadError",
    "validate_orbital_analysis",
    "validate_population_analysis",
    "validate_vibration_analysis",
]
