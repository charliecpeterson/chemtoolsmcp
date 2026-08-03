"""Bounded inspection of recognized QMCPACK HDF5 artifact layouts.

The inspector reads only small structural metadata.  It deliberately does not
decode orbital coefficients, walker coordinates, estimator values, or arbitrary
datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


MAX_QMCPACK_HDF5_BYTES = 2 * 1024 * 1024 * 1024
MAX_REPORTED_STATE_GROUPS = 64
MAX_REPORTED_ESTIMATORS = 64
MAX_REPORTED_SPECIES = 64
MAX_SMALL_DATASET_ELEMENTS = 128
MAX_SMALL_DATASET_BYTES = 16 * 1024
MAX_SPECIES_ID_ELEMENTS = 100_000


def inspect_qmcpack_hdf5(path: str | Path) -> dict[str, Any]:
    """Classify one recognized QMCPACK HDF5 layout from bounded metadata."""
    source = Path(path).expanduser().resolve()
    try:
        metadata = source.stat()
    except OSError as error:
        return _error("missing", f"could not stat QMCPACK HDF5 file: {error}", source)
    if not source.is_file():
        return _error("invalid_source", "QMCPACK HDF5 path is not a regular file", source)
    if metadata.st_size == 0:
        return _error("invalid_hdf5", "QMCPACK HDF5 file is empty", source)
    if metadata.st_size > MAX_QMCPACK_HDF5_BYTES:
        return _error(
            "source_too_large",
            "QMCPACK HDF5 file exceeds the 2 GiB inspection limit",
            source,
            size_bytes=metadata.st_size,
        )

    try:
        import h5py
    except ImportError:
        return _error(
            "runtime_error",
            "QMCPACK HDF5 inspection requires h5py in the companion science runtime",
            source,
            size_bytes=metadata.st_size,
        )

    try:
        with h5py.File(source, "r") as document:
            inspection = _recognized_layout(document)
    except (OSError, RuntimeError, ValueError) as error:
        return _error(
            "invalid_hdf5",
            f"could not open QMCPACK HDF5 file: {type(error).__name__}: {error}",
            source,
            size_bytes=metadata.st_size,
        )

    return {
        "schema_version": "chemtools.qmcpack-hdf5-inspection/1",
        "status": "recognized" if inspection is not None else "unrecognized_layout",
        "source": {
            "path": str(source),
            "size_bytes": metadata.st_size,
            "modified_ns": metadata.st_mtime_ns,
        },
        **(
            inspection
            if inspection is not None
            else {
                "message": (
                    "The file is readable HDF5, but its layout is not one of the "
                    "fixed QMCPACK wavefunction, variational-parameter, walker "
                    "configuration, or statistics layouts."
                ),
            }
        ),
        "scope_limit": (
            "This reads fixed-layout metadata only. It does not decode orbital "
            "coefficients, density grids, walker coordinates, estimator values, "
            "or arbitrary HDF5 datasets."
        ),
    }


def _recognized_layout(document: Any) -> dict[str, Any] | None:
    if _has_paths(
        document,
        (
            "application/code",
            "atoms/number_of_atoms",
            "electrons/number_of_electrons",
            "electrons/number_of_spins",
            "electrons/number_of_kpoints",
            "supercell/primitive_vectors",
        ),
    ):
        return _wavefunction_layout(document)
    if _has_paths(
        document,
        ("name_value_lists/parameter_names", "name_value_lists/parameter_values"),
    ):
        return _variational_parameter_layout(document)
    states = _state_groups(document)
    if states:
        return _walker_configuration_layout(document, states)
    if "LocalEnergy/value" in document:
        return _statistics_layout(document)
    return None


def _wavefunction_layout(document: Any) -> dict[str, Any]:
    atom_count = _small_value(document, "atoms/number_of_atoms")
    species_count = _small_value(document, "atoms/number_of_species")
    electrons = _small_value(document, "electrons/number_of_electrons")
    spin_count = _small_value(document, "electrons/number_of_spins")
    kpoint_count = _small_value(document, "electrons/number_of_kpoints")
    species_atom_counts = _species_atom_counts(document, species_count)
    species = []
    for index in range(_bounded_count(species_count, MAX_REPORTED_SPECIES)):
        prefix = f"atoms/species_{index}"
        if prefix not in document:
            continue
        species.append({
            "index": index,
            "name": _small_value(document, f"{prefix}/name"),
            "atomic_number": _small_value(document, f"{prefix}/atomic_number"),
            "valence_charge": _small_value(document, f"{prefix}/valence_charge"),
            "atom_count": species_atom_counts.get(index) if species_atom_counts else None,
        })
    density_present = "electrons/density" in document
    return {
        "artifact_kind": "pwscf_wavefunction",
        "message": (
            "Recognized the QMCPACK electronic-structure HDF5 layout written by "
            "a converter such as pw2qmcpack."
        ),
        "wavefunction": {
            "format": _small_value(document, "format"),
            "version": _small_value(document, "version"),
            "application": {
                "code": _small_value(document, "application/code"),
                "version": _small_value(document, "application/version"),
            },
            "atoms": {
                "count": atom_count,
                "species_count": species_count,
                "positions_shape": _dataset_shape(document, "atoms/positions"),
                "species": species,
            },
            "electrons": {
                "spin_populations": electrons,
                "spin_count": spin_count,
                "kpoint_count": kpoint_count,
                "density_metadata_present": density_present,
            },
            "supercell": {
                "primitive_vectors_shape": _dataset_shape(
                    document, "supercell/primitive_vectors"
                ),
            },
        },
    }


def _variational_parameter_layout(document: Any) -> dict[str, Any]:
    names_shape = _dataset_shape(document, "name_value_lists/parameter_names")
    values_shape = _dataset_shape(document, "name_value_lists/parameter_values")
    name_count = _shape_size(names_shape)
    value_count = _shape_size(values_shape)
    return {
        "artifact_kind": "variational_parameters",
        "message": "Recognized the QMCPACK variational-parameter sidecar layout.",
        "variational_parameters": {
            "version": _small_value(document, "version"),
            "timestamp": _small_value(document, "timestamp"),
            "parameter_name_count": name_count,
            "parameter_value_count": value_count,
            "name_value_counts_match": (
                name_count == value_count
                if name_count is not None and value_count is not None
                else None
            ),
        },
    }


def _walker_configuration_layout(document: Any, states: list[str]) -> dict[str, Any]:
    reported = []
    for name in states[:MAX_REPORTED_STATE_GROUPS]:
        prefix = name
        reported.append({
            "name": name,
            "block": _small_value(document, f"{prefix}/block"),
            "number_of_walkers": _small_value(document, f"{prefix}/number_of_walkers"),
            "walkers_shape": _dataset_shape(document, f"{prefix}/walkers"),
            "walker_weights_shape": _dataset_shape(
                document, f"{prefix}/walker_weights"
            ),
        })
    return {
        "artifact_kind": "walker_configuration",
        "message": "Recognized the QMCPACK walker-configuration HDF5 layout.",
        "walker_configuration": {
            "version": _small_value(document, "version"),
            "state_count": len(states),
            "states": reported,
            "states_truncated": len(states) > len(reported),
        },
    }


def _statistics_layout(document: Any) -> dict[str, Any]:
    estimators = sorted(
        name
        for name in document.keys()
        if f"{name}/value" in document
    )
    reported = estimators[:MAX_REPORTED_ESTIMATORS]
    return {
        "artifact_kind": "statistics",
        "message": "Recognized the QMCPACK statistics HDF5 layout.",
        "statistics": {
            "estimator_count": len(estimators),
            "estimators": [
                {"name": name, "value_shape": _dataset_shape(document, f"{name}/value")}
                for name in reported
            ],
            "estimators_truncated": len(estimators) > len(reported),
        },
    }


def _has_paths(document: Any, paths: tuple[str, ...]) -> bool:
    return all(path in document for path in paths)


def _state_groups(document: Any) -> list[str]:
    return sorted(
        name
        for name in document.keys()
        if name.startswith("state_") and f"{name}/walkers" in document
    )


def _dataset_shape(document: Any, path: str) -> list[int] | None:
    dataset = document.get(path)
    if dataset is None or not hasattr(dataset, "shape"):
        return None
    return [int(value) for value in dataset.shape]


def _small_value(document: Any, path: str) -> Any:
    dataset = document.get(path)
    if dataset is None or not hasattr(dataset, "shape"):
        return None
    shape = _dataset_shape(document, path)
    if shape is None or _shape_size(shape) is None:
        return None
    if _shape_size(shape) > MAX_SMALL_DATASET_ELEMENTS:
        return None
    if getattr(dataset, "nbytes", MAX_SMALL_DATASET_BYTES + 1) > MAX_SMALL_DATASET_BYTES:
        return None
    try:
        return _json_value(dataset[()])
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def _species_atom_counts(document: Any, species_count: Any) -> dict[int, int] | None:
    count = _bounded_count(species_count, MAX_REPORTED_SPECIES)
    dataset = document.get("atoms/species_ids")
    if count == 0 or dataset is None or not hasattr(dataset, "shape"):
        return None
    shape = _dataset_shape(document, "atoms/species_ids")
    size = _shape_size(shape)
    if size is None or size > MAX_SPECIES_ID_ELEMENTS:
        return None
    if getattr(dataset, "nbytes", MAX_SPECIES_ID_ELEMENTS * 8 + 1) > MAX_SPECIES_ID_ELEMENTS * 8:
        return None
    try:
        values = dataset[()]
    except (OSError, RuntimeError, ValueError, TypeError):
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    if size == 1 and isinstance(values, int):
        values = [values]
    if not isinstance(values, list) or len(values) != size:
        return None
    if any(not isinstance(value, int) or value < 0 or value >= count for value in values):
        return None
    return {index: values.count(index) for index in range(count)}


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, tuple):
        normalized = [_json_value(item) for item in value]
        return normalized[0] if len(normalized) == 1 else normalized
    if isinstance(value, list):
        normalized = [_json_value(item) for item in value]
        return normalized[0] if len(normalized) == 1 else normalized
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _bounded_count(value: Any, maximum: int) -> int:
    if isinstance(value, int) and 0 <= value <= maximum:
        return value
    return 0


def _shape_size(shape: list[int] | None) -> int | None:
    if shape is None:
        return None
    product = 1
    for value in shape:
        product *= value
    return product


def _error(
    status: str,
    message: str,
    source: Path,
    *,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "schema_version": "chemtools.qmcpack-hdf5-inspection/1",
        "status": status,
        "message": message,
        "source": {"path": str(source)},
    }
    if size_bytes is not None:
        response["source"]["size_bytes"] = size_bytes
    return response


__all__ = ["MAX_QMCPACK_HDF5_BYTES", "inspect_qmcpack_hdf5"]
