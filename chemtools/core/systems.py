"""Immutable molecular and periodic scientific-system specifications.

Calculation choices such as method, cutoff, smearing, and executable settings
stay outside these models. The system boundary owns identity, geometry,
periodicity, k-point sampling, pseudopotentials, charge, and spin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Union

from chemtools.core.common import ELEMENT_TO_Z


CoordinateUnits = Literal["angstrom", "bohr"]
CoordinateMode = Literal["cartesian", "fractional"]
KPointMode = Literal["gamma", "mesh", "explicit"]
KPointCoordinateSystem = Literal["crystal", "cartesian"]
PseudopotentialFormat = Literal[
    "upf",
    "qmcpack_xml",
    "casino",
    "unknown",
]
PeriodicSpinMode = Literal[
    "unpolarized",
    "collinear",
    "noncollinear",
    "spin_orbit",
]

SCIENTIFIC_SYSTEM_SCHEMA = "chemtools.scientific-system/1"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _optional_text(value: str | None, field_name: str) -> None:
    if value is not None:
        _require_text(value, field_name)


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _vector3(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    return (
        _finite_float(value[0], f"{field_name}[0]"),
        _finite_float(value[1], f"{field_name}[1]"),
        _finite_float(value[2], f"{field_name}[2]"),
    )


def _canonical_element(value: str) -> str:
    _require_text(value, "element")
    canonical = value[0].upper() + value[1:].lower()
    if canonical not in ELEMENT_TO_Z:
        raise ValueError(f"unknown element symbol: {value!r}")
    return canonical


def _string_metadata(
    value: Mapping[str, str],
) -> Mapping[str, str]:
    copied: dict[str, str] = {}
    for key, item in value.items():
        _require_text(key, "metadata key")
        if not isinstance(item, str):
            raise TypeError("metadata values must be strings")
        copied[key] = item
    return MappingProxyType(copied)


def _float_mapping(
    value: Mapping[str, float],
    field_name: str,
) -> Mapping[str, float]:
    copied: dict[str, float] = {}
    for key, item in value.items():
        _require_text(key, f"{field_name} key")
        copied[key] = _finite_float(item, f"{field_name}.{key}")
    return MappingProxyType(copied)


def _effective_species(atom: AtomSpec) -> str:
    return atom.species or atom.element


def _validate_atom_labels(atoms: tuple[AtomSpec, ...]) -> None:
    labels = [atom.label for atom in atoms if atom.label is not None]
    if len(labels) != len(set(labels)):
        raise ValueError("atom labels must be unique when provided")


@dataclass(frozen=True)
class AtomSpec:
    element: str
    position: tuple[float, float, float]
    label: str | None = None
    species: str | None = None
    ghost: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "element", _canonical_element(self.element))
        object.__setattr__(
            self,
            "position",
            _vector3(self.position, "position"),
        )
        _optional_text(self.label, "label")
        _optional_text(self.species, "species")

    def to_dict(self) -> dict[str, Any]:
        return {
            "element": self.element,
            "position": list(self.position),
            "label": self.label,
            "species": self.species,
            "ghost": self.ghost,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AtomSpec:
        return cls(
            element=value["element"],
            position=value["position"],
            label=value.get("label"),
            species=value.get("species"),
            ghost=value.get("ghost", False),
        )


@dataclass(frozen=True)
class LatticeSpec:
    vectors: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    units: CoordinateUnits
    periodic: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        if not isinstance(self.vectors, (list, tuple)) or len(self.vectors) != 3:
            raise ValueError("lattice vectors must contain exactly three vectors")
        vectors = (
            _vector3(self.vectors[0], "vectors[0]"),
            _vector3(self.vectors[1], "vectors[1]"),
            _vector3(self.vectors[2], "vectors[2]"),
        )
        object.__setattr__(self, "vectors", vectors)
        if self.units not in ("angstrom", "bohr"):
            raise ValueError("lattice units must be 'angstrom' or 'bohr'")
        if (
            not isinstance(self.periodic, (list, tuple))
            or len(self.periodic) != 3
            or any(not isinstance(value, bool) for value in self.periodic)
        ):
            raise ValueError("periodic must contain exactly three booleans")
        object.__setattr__(self, "periodic", tuple(self.periodic))

        a, b, c = vectors
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if abs(determinant) < 1e-12:
            raise ValueError("lattice vectors must define a nonzero cell")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vectors": [list(vector) for vector in self.vectors],
            "units": self.units,
            "periodic": list(self.periodic),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LatticeSpec:
        return cls(
            vectors=value["vectors"],
            units=value["units"],
            periodic=value.get("periodic", (True, True, True)),
        )


@dataclass(frozen=True)
class KPoint:
    coordinates: tuple[float, float, float]
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinates",
            _vector3(self.coordinates, "coordinates"),
        )
        weight = _finite_float(self.weight, "weight")
        if weight < 0:
            raise ValueError("k-point weight must be non-negative")
        object.__setattr__(self, "weight", weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": list(self.coordinates),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KPoint:
        return cls(
            coordinates=value["coordinates"],
            weight=value["weight"],
        )


@dataclass(frozen=True)
class KPointSampling:
    mode: KPointMode
    mesh: tuple[int, int, int] | None = None
    shift: tuple[float, float, float] | None = None
    points: tuple[KPoint, ...] = ()
    coordinate_system: KPointCoordinateSystem = "crystal"

    def __post_init__(self) -> None:
        if self.mode not in ("gamma", "mesh", "explicit"):
            raise ValueError("invalid k-point mode")
        if self.coordinate_system not in ("crystal", "cartesian"):
            raise ValueError("invalid k-point coordinate system")
        object.__setattr__(self, "points", tuple(self.points))

        if self.mode == "gamma":
            if self.mesh is not None or self.shift is not None or self.points:
                raise ValueError(
                    "gamma sampling cannot include mesh, shift, or points"
                )
            return

        if self.mode == "mesh":
            if (
                not isinstance(self.mesh, (list, tuple))
                or len(self.mesh) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in self.mesh
                )
            ):
                raise ValueError("k-point mesh must contain three positive integers")
            if self.shift is None:
                raise ValueError("mesh sampling requires an explicit shift")
            shift = _vector3(self.shift, "shift")
            if any(value < 0 or value > 1 for value in shift):
                raise ValueError("mesh shift values must be between 0 and 1")
            if self.points:
                raise ValueError("mesh sampling cannot include explicit points")
            object.__setattr__(self, "mesh", tuple(self.mesh))
            object.__setattr__(self, "shift", shift)
            return

        if self.mesh is not None or self.shift is not None:
            raise ValueError("explicit sampling cannot include mesh or shift")
        if not self.points:
            raise ValueError("explicit sampling requires at least one k-point")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mesh": list(self.mesh) if self.mesh is not None else None,
            "shift": list(self.shift) if self.shift is not None else None,
            "points": [point.to_dict() for point in self.points],
            "coordinate_system": self.coordinate_system,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KPointSampling:
        return cls(
            mode=value["mode"],
            mesh=value.get("mesh"),
            shift=value.get("shift"),
            points=tuple(
                KPoint.from_dict(point)
                for point in value.get("points") or ()
            ),
            coordinate_system=value.get("coordinate_system", "crystal"),
        )


@dataclass(frozen=True)
class PseudopotentialAssignment:
    species: str
    element: str
    format: PseudopotentialFormat
    artifact_id: str | None = None
    path_hint: Path | None = None
    family: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.species, "species")
        object.__setattr__(self, "element", _canonical_element(self.element))
        if self.format not in ("upf", "qmcpack_xml", "casino", "unknown"):
            raise ValueError("invalid pseudopotential format")
        _optional_text(self.artifact_id, "artifact_id")
        _optional_text(self.family, "family")
        if self.path_hint is not None:
            object.__setattr__(self, "path_hint", Path(self.path_hint))
        if self.artifact_id is None and self.path_hint is None:
            raise ValueError(
                "pseudopotential assignment requires artifact_id or path_hint"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "element": self.element,
            "format": self.format,
            "artifact_id": self.artifact_id,
            "path_hint": (
                str(self.path_hint) if self.path_hint is not None else None
            ),
            "family": self.family,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> PseudopotentialAssignment:
        path_hint = value.get("path_hint")
        return cls(
            species=value["species"],
            element=value["element"],
            format=value["format"],
            artifact_id=value.get("artifact_id"),
            path_hint=Path(path_hint) if path_hint is not None else None,
            family=value.get("family"),
        )


@dataclass(frozen=True)
class PeriodicSpinSpec:
    mode: PeriodicSpinMode
    net_spin: float | None = None
    starting_magnetization_by_species: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.mode not in (
            "unpolarized",
            "collinear",
            "noncollinear",
            "spin_orbit",
        ):
            raise ValueError("invalid periodic spin mode")
        if self.net_spin is not None:
            object.__setattr__(
                self,
                "net_spin",
                _finite_float(self.net_spin, "net_spin"),
            )
        magnetization = _float_mapping(
            self.starting_magnetization_by_species,
            "starting_magnetization_by_species",
        )
        object.__setattr__(
            self,
            "starting_magnetization_by_species",
            magnetization,
        )
        if self.mode == "unpolarized":
            if self.net_spin not in (None, 0.0) or magnetization:
                raise ValueError(
                    "unpolarized spin cannot include nonzero spin settings"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "net_spin": self.net_spin,
            "starting_magnetization_by_species": dict(
                self.starting_magnetization_by_species
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PeriodicSpinSpec:
        return cls(
            mode=value["mode"],
            net_spin=value.get("net_spin"),
            starting_magnetization_by_species=value.get(
                "starting_magnetization_by_species"
            )
            or {},
        )


@dataclass(frozen=True)
class MolecularSystemSpec:
    atoms: tuple[AtomSpec, ...]
    charge: int
    multiplicity: int
    coordinate_units: CoordinateUnits = "angstrom"
    name: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        if not self.atoms:
            raise ValueError("molecular systems require at least one atom")
        _validate_atom_labels(self.atoms)
        if isinstance(self.charge, bool) or not isinstance(self.charge, int):
            raise TypeError("molecular charge must be an integer")
        if (
            isinstance(self.multiplicity, bool)
            or not isinstance(self.multiplicity, int)
            or self.multiplicity < 1
        ):
            raise ValueError("multiplicity must be a positive integer")
        if self.coordinate_units not in ("angstrom", "bohr"):
            raise ValueError(
                "coordinate_units must be 'angstrom' or 'bohr'"
            )
        _optional_text(self.name, "name")
        object.__setattr__(
            self,
            "metadata",
            _string_metadata(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCIENTIFIC_SYSTEM_SCHEMA,
            "system_type": "molecular",
            "name": self.name,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "coordinate_units": self.coordinate_units,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MolecularSystemSpec:
        _require_system_envelope(value, "molecular")
        return cls(
            atoms=tuple(AtomSpec.from_dict(atom) for atom in value["atoms"]),
            charge=value["charge"],
            multiplicity=value["multiplicity"],
            coordinate_units=value["coordinate_units"],
            name=value.get("name"),
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True)
class PeriodicSystemSpec:
    atoms: tuple[AtomSpec, ...]
    lattice: LatticeSpec
    coordinate_mode: CoordinateMode
    coordinate_units: CoordinateUnits | None
    k_points: KPointSampling
    pseudopotentials: tuple[PseudopotentialAssignment, ...]
    spin: PeriodicSpinSpec
    net_charge: float = 0.0
    name: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(
            self,
            "pseudopotentials",
            tuple(self.pseudopotentials),
        )
        if not self.atoms:
            raise ValueError("periodic systems require at least one atom")
        _validate_atom_labels(self.atoms)
        if self.coordinate_mode not in ("cartesian", "fractional"):
            raise ValueError("invalid coordinate_mode")
        if self.coordinate_mode == "fractional":
            if self.coordinate_units is not None:
                raise ValueError(
                    "fractional coordinates cannot declare coordinate units"
                )
        elif self.coordinate_units not in ("angstrom", "bohr"):
            raise ValueError(
                "cartesian coordinates require angstrom or bohr units"
            )
        object.__setattr__(
            self,
            "net_charge",
            _finite_float(self.net_charge, "net_charge"),
        )
        _optional_text(self.name, "name")
        object.__setattr__(
            self,
            "metadata",
            _string_metadata(self.metadata),
        )

        species_elements: dict[str, str] = {}
        for atom in self.atoms:
            if atom.ghost:
                continue
            species = _effective_species(atom)
            previous = species_elements.setdefault(species, atom.element)
            if previous != atom.element:
                raise ValueError(
                    f"species {species!r} maps to multiple elements"
                )

        pseudo_by_species: dict[str, PseudopotentialAssignment] = {}
        for assignment in self.pseudopotentials:
            if assignment.species in pseudo_by_species:
                raise ValueError(
                    f"duplicate pseudopotential for species "
                    f"{assignment.species!r}"
                )
            pseudo_by_species[assignment.species] = assignment
            expected_element = species_elements.get(assignment.species)
            if expected_element is None:
                raise ValueError(
                    f"pseudopotential species {assignment.species!r} "
                    "is not present in the system"
                )
            if assignment.element != expected_element:
                raise ValueError(
                    f"pseudopotential species {assignment.species!r} "
                    f"expects element {expected_element!r}, not "
                    f"{assignment.element!r}"
                )
        if pseudo_by_species and set(pseudo_by_species) != set(species_elements):
            missing = sorted(set(species_elements) - set(pseudo_by_species))
            raise ValueError(
                f"missing pseudopotentials for species: {missing}"
            )

        unknown_spin_species = sorted(
            set(self.spin.starting_magnetization_by_species)
            - set(species_elements)
        )
        if unknown_spin_species:
            raise ValueError(
                f"spin settings reference unknown species: "
                f"{unknown_spin_species}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCIENTIFIC_SYSTEM_SCHEMA,
            "system_type": "periodic",
            "name": self.name,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "lattice": self.lattice.to_dict(),
            "coordinate_mode": self.coordinate_mode,
            "coordinate_units": self.coordinate_units,
            "k_points": self.k_points.to_dict(),
            "pseudopotentials": [
                assignment.to_dict()
                for assignment in self.pseudopotentials
            ],
            "spin": self.spin.to_dict(),
            "net_charge": self.net_charge,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PeriodicSystemSpec:
        _require_system_envelope(value, "periodic")
        return cls(
            atoms=tuple(AtomSpec.from_dict(atom) for atom in value["atoms"]),
            lattice=LatticeSpec.from_dict(value["lattice"]),
            coordinate_mode=value["coordinate_mode"],
            coordinate_units=value.get("coordinate_units"),
            k_points=KPointSampling.from_dict(value["k_points"]),
            pseudopotentials=tuple(
                PseudopotentialAssignment.from_dict(assignment)
                for assignment in value.get("pseudopotentials") or ()
            ),
            spin=PeriodicSpinSpec.from_dict(value["spin"]),
            net_charge=value.get("net_charge", 0.0),
            name=value.get("name"),
            metadata=value.get("metadata") or {},
        )


ScientificSystemSpec = Union[MolecularSystemSpec, PeriodicSystemSpec]


def _require_system_envelope(
    value: Mapping[str, Any],
    expected_type: str,
) -> None:
    if value.get("schema") != SCIENTIFIC_SYSTEM_SCHEMA:
        raise ValueError(
            f"unsupported scientific-system schema: "
            f"{value.get('schema')!r}"
        )
    if value.get("system_type") != expected_type:
        raise ValueError(
            f"expected {expected_type!r} system, got "
            f"{value.get('system_type')!r}"
        )


def scientific_system_from_dict(
    value: Mapping[str, Any],
) -> ScientificSystemSpec:
    system_type = value.get("system_type")
    if system_type == "molecular":
        return MolecularSystemSpec.from_dict(value)
    if system_type == "periodic":
        return PeriodicSystemSpec.from_dict(value)
    raise ValueError(f"unknown scientific system type: {system_type!r}")


def molecular_system_from_input_spec(
    value: Mapping[str, Any],
) -> MolecularSystemSpec:
    atoms = value.get("atoms")
    if not isinstance(atoms, (list, tuple)) or not atoms:
        raise ValueError("InputSpec atoms are required")
    return MolecularSystemSpec(
        atoms=tuple(
            AtomSpec(
                element=atom["element"],
                position=(atom["x"], atom["y"], atom["z"]),
                label=atom.get("label"),
                ghost=atom.get("ghost", False),
            )
            for atom in atoms
        ),
        charge=value.get("charge", 0),
        multiplicity=value.get("multiplicity", 1),
        coordinate_units=value.get("geometry_units", "angstrom"),
        name=value.get("title"),
        metadata={"source": "InputSpec"},
    )


def molecular_system_to_input_fields(
    system: MolecularSystemSpec,
) -> dict[str, Any]:
    return {
        "atoms": [
            {
                "element": atom.element,
                "x": atom.position[0],
                "y": atom.position[1],
                "z": atom.position[2],
                **({"label": atom.label} if atom.label is not None else {}),
                **({"ghost": True} if atom.ghost else {}),
            }
            for atom in system.atoms
        ],
        "charge": system.charge,
        "multiplicity": system.multiplicity,
        "geometry_units": system.coordinate_units,
        **({"title": system.name} if system.name is not None else {}),
    }


__all__ = [
    "SCIENTIFIC_SYSTEM_SCHEMA",
    "AtomSpec",
    "LatticeSpec",
    "KPoint",
    "KPointSampling",
    "PseudopotentialAssignment",
    "PeriodicSpinSpec",
    "MolecularSystemSpec",
    "PeriodicSystemSpec",
    "ScientificSystemSpec",
    "scientific_system_from_dict",
    "molecular_system_from_input_spec",
    "molecular_system_to_input_fields",
]
