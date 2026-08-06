"""Enumerate LS terms and jj-coupled level counts for atomic configurations.

The calculations provide symmetry and state-count checks. They do not model
radial integrals, level energies, or spin-orbit mixing.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re
from typing import Iterable, Mapping, Sequence


ATOMIC_MULTIPLET_SCHEMA = "chemtools.atomic-multiplets/1"
ORBITAL_LETTERS = "spdfghiklmnoq"
TERM_LETTERS = "SPDFGHIKLMNOQRTUVWXYZ"
MAX_MULTIPLET_MICROSTATES = 10_000_000

_L_FROM_LETTER = {
    letter: angular_momentum
    for angular_momentum, letter in enumerate(ORBITAL_LETTERS)
}
_TOKEN_RE = re.compile(
    rf"(?P<n>\d*)(?P<orbital>[{ORBITAL_LETTERS}])(?:\^)?(?P<electrons>\d+)",
    re.IGNORECASE,
)

Weight = tuple[int, int]
WeightDistribution = dict[Weight, int]


class AtomicConfigurationError(ValueError):
    """The supplied shell occupations cannot describe an atomic configuration."""


@dataclass(frozen=True)
class AtomicSubshell:
    principal: int | None
    angular_momentum: int
    electrons: int

    @property
    def capacity(self) -> int:
        return 2 * (2 * self.angular_momentum + 1)

    @property
    def effective_electrons(self) -> int:
        return min(self.electrons, self.capacity - self.electrons)

    @property
    def is_open(self) -> bool:
        return 0 < self.electrons < self.capacity

    @property
    def name(self) -> str:
        principal = "" if self.principal is None else str(self.principal)
        return f"{principal}{ORBITAL_LETTERS[self.angular_momentum]}"

    @property
    def label(self) -> str:
        return f"{self.name}{self.electrons}"


@dataclass(frozen=True)
class AtomicConfiguration:
    subshells: tuple[AtomicSubshell, ...]

    @property
    def electron_count(self) -> int:
        return sum(shell.electrons for shell in self.subshells)

    @property
    def parity(self) -> str:
        exponent = sum(
            shell.angular_momentum * shell.electrons
            for shell in self.subshells
        )
        return "-" if exponent % 2 else "+"

    @property
    def label(self) -> str:
        return " ".join(shell.label for shell in self.subshells)


@dataclass(frozen=True)
class AtomicTerm:
    angular_momentum: int
    two_s: int
    occurrences: int

    @property
    def multiplicity(self) -> int:
        return self.two_s + 1

    @property
    def label(self) -> str:
        if self.angular_momentum < len(TERM_LETTERS):
            orbital = TERM_LETTERS[self.angular_momentum]
        else:
            orbital = f"L({self.angular_momentum})"
        return f"{self.multiplicity}{orbital}"

    @property
    def degeneracy(self) -> int:
        return (2 * self.angular_momentum + 1) * self.multiplicity


@dataclass(frozen=True)
class HundEstimate:
    term: AtomicTerm
    two_j: int
    filling: str


@dataclass(frozen=True)
class RelativisticOccupation:
    label: str
    two_j: int
    electrons: int


@dataclass(frozen=True)
class RelativisticConfiguration:
    label: str
    occupations: tuple[RelativisticOccupation, ...]
    levels: tuple[tuple[int, int], ...]
    microstates: int


@dataclass(frozen=True)
class JjCensus:
    configurations: tuple[RelativisticConfiguration, ...]
    jj_levels: tuple[tuple[int, int], ...]
    ls_levels: tuple[tuple[int, int], ...]

    @property
    def consistent(self) -> bool:
        return self.jj_levels == self.ls_levels


def parse_atomic_configuration(value: str) -> AtomicConfiguration:
    """Parse compact configurations such as ``4f7 6s2`` or ``p2``."""
    if not isinstance(value, str) or not value.strip():
        raise AtomicConfigurationError("configuration must be a non-empty string")
    if len(value) > 256:
        raise AtomicConfigurationError("configuration must not exceed 256 characters")

    parsed: list[tuple[int | None, int, int]] = []
    cursor = 0
    for match in _TOKEN_RE.finditer(value):
        gap = value[cursor:match.start()]
        if gap.strip(" ,"):
            raise AtomicConfigurationError(
                f"cannot parse configuration near {gap!r}"
            )
        principal_text = match.group("n")
        principal = int(principal_text) if principal_text else None
        orbital = match.group("orbital").lower()
        angular_momentum = _L_FROM_LETTER[orbital]
        electrons = int(match.group("electrons"))
        if electrons == 0:
            raise AtomicConfigurationError(
                "subshell occupancies must be positive"
            )
        if principal is not None and angular_momentum >= principal:
            raise AtomicConfigurationError(
                f"{principal}{orbital} is invalid: an n={principal} shell "
                f"requires l < {principal}"
            )
        parsed.append((principal, angular_momentum, electrons))
        cursor = match.end()

    trailing = value[cursor:]
    if trailing.strip(" ,"):
        raise AtomicConfigurationError(
            f"cannot parse configuration near {trailing!r}"
        )
    if not parsed:
        raise AtomicConfigurationError(f"cannot parse configuration {value!r}")

    occupancies: dict[tuple[int | None, int], int] = {}
    order: list[tuple[int | None, int]] = []
    for principal, angular_momentum, electrons in parsed:
        key = (principal, angular_momentum)
        if key not in occupancies:
            occupancies[key] = 0
            order.append(key)
        occupancies[key] += electrons

    shells = tuple(
        AtomicSubshell(
            principal=principal,
            angular_momentum=angular_momentum,
            electrons=occupancies[(principal, angular_momentum)],
        )
        for principal, angular_momentum in order
    )
    return atomic_configuration(shells)


def atomic_configuration(
    shells: Iterable[AtomicSubshell],
) -> AtomicConfiguration:
    """Validate already parsed shell occupations."""
    subshells = tuple(shells)
    if not subshells:
        raise AtomicConfigurationError("configuration must contain a subshell")
    keys: set[tuple[int | None, int]] = set()
    for shell in subshells:
        if not 0 <= shell.angular_momentum < len(ORBITAL_LETTERS):
            raise AtomicConfigurationError(
                f"angular momentum must be between 0 and "
                f"{len(ORBITAL_LETTERS) - 1}"
            )
        if shell.principal is not None:
            if shell.principal < 1 or shell.angular_momentum >= shell.principal:
                raise AtomicConfigurationError(
                    f"invalid principal/angular quantum numbers for {shell.label}"
                )
        if not 1 <= shell.electrons <= shell.capacity:
            raise AtomicConfigurationError(
                f"{shell.label} exceeds the subshell capacity of {shell.capacity}"
            )
        key = (shell.principal, shell.angular_momentum)
        if key in keys:
            raise AtomicConfigurationError(f"duplicate subshell {shell.name}")
        keys.add(key)
    configuration = AtomicConfiguration(subshells)
    _require_feasible(configuration)
    return configuration


def _require_feasible(configuration: AtomicConfiguration) -> None:
    microstates = math.prod(
        math.comb(shell.capacity, shell.electrons)
        for shell in configuration.subshells
    )
    if microstates > MAX_MULTIPLET_MICROSTATES:
        raise AtomicConfigurationError(
            f"configuration has {microstates} determinant microstates; "
            f"the analysis limit is {MAX_MULTIPLET_MICROSTATES}"
        )


def shell_microstate_distribution(
    angular_momentum: int,
    electrons: int,
) -> WeightDistribution:
    """Count determinant ML/MS weights without materializing determinants."""
    capacity = 2 * (2 * angular_momentum + 1)
    if not 0 <= electrons <= capacity:
        raise ValueError(f"occupancy must be between 0 and {capacity}")

    effective_electrons = min(electrons, capacity - electrons)
    states: dict[tuple[int, int, int], int] = {(0, 0, 0): 1}
    for m_l in range(-angular_momentum, angular_momentum + 1):
        for two_m_s in (-1, 1):
            updated = dict(states)
            for (chosen, total_m_l, total_two_m_s), count in states.items():
                if chosen == effective_electrons:
                    continue
                key = (
                    chosen + 1,
                    total_m_l + m_l,
                    total_two_m_s + two_m_s,
                )
                updated[key] = updated.get(key, 0) + count
            states = updated
    return {
        (total_m_l, total_two_m_s): count
        for (chosen, total_m_l, total_two_m_s), count in states.items()
        if chosen == effective_electrons
    }


def combine_weight_distributions(
    distributions: Iterable[Mapping[Weight, int]],
) -> WeightDistribution:
    combined: WeightDistribution = {(0, 0): 1}
    for distribution in distributions:
        updated: WeightDistribution = {}
        for (left_l, left_s), left_count in combined.items():
            for (right_l, right_s), right_count in distribution.items():
                key = (left_l + right_l, left_s + right_s)
                updated[key] = updated.get(key, 0) + left_count * right_count
        combined = updated
    return combined


def configuration_microstates(
    configuration: AtomicConfiguration,
) -> WeightDistribution:
    return combine_weight_distributions(
        shell_microstate_distribution(
            shell.angular_momentum,
            shell.electrons,
        )
        for shell in configuration.subshells
    )


def decompose_ls(
    distribution: Mapping[Weight, int],
) -> tuple[AtomicTerm, ...]:
    """Decompose an ML/MS distribution using highest-weight multiplicities."""
    if not distribution:
        raise ValueError("microstate distribution is empty")
    maximum_l = max(abs(m_l) for m_l, _ in distribution)
    maximum_two_s = max(abs(two_m_s) for _, two_m_s in distribution)
    spin_parity = next(iter(distribution))[1] % 2
    terms: list[AtomicTerm] = []

    for two_s in range(spin_parity, maximum_two_s + 1, 2):
        for angular_momentum in range(maximum_l + 1):
            occurrences = (
                distribution.get((angular_momentum, two_s), 0)
                - distribution.get((angular_momentum + 1, two_s), 0)
                - distribution.get((angular_momentum, two_s + 2), 0)
                + distribution.get((angular_momentum + 1, two_s + 2), 0)
            )
            if occurrences < 0:
                raise RuntimeError(
                    "invalid LS decomposition at "
                    f"L={angular_momentum}, 2S={two_s}: {occurrences}"
                )
            if occurrences:
                terms.append(AtomicTerm(
                    angular_momentum=angular_momentum,
                    two_s=two_s,
                    occurrences=occurrences,
                ))

    terms.sort(
        key=lambda term: (term.two_s, term.angular_momentum),
        reverse=True,
    )
    microstates = sum(distribution.values())
    term_states = sum(
        term.occurrences * term.degeneracy
        for term in terms
    )
    if term_states != microstates:
        raise RuntimeError(
            f"LS decomposition accounts for {term_states} of "
            f"{microstates} microstates"
        )
    return tuple(terms)


def terms_for_configuration(
    configuration: AtomicConfiguration,
) -> tuple[AtomicTerm, ...]:
    return decompose_ls(configuration_microstates(configuration))


def term_j_values(term: AtomicTerm) -> tuple[int, ...]:
    lower = abs(2 * term.angular_momentum - term.two_s)
    upper = 2 * term.angular_momentum + term.two_s
    return tuple(range(lower, upper + 1, 2))


def lande_g(term: AtomicTerm, two_j: int) -> float | None:
    if two_j == 0:
        return None
    j_value = two_j / 2
    spin = term.two_s / 2
    return 1 + (
        j_value * (j_value + 1)
        + spin * (spin + 1)
        - term.angular_momentum * (term.angular_momentum + 1)
    ) / (2 * j_value * (j_value + 1))


def j_level_counts(terms: Sequence[AtomicTerm]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for term in terms:
        for two_j in term_j_values(term):
            counts[two_j] = counts.get(two_j, 0) + term.occurrences
    return dict(sorted(counts.items()))


def hund_ground_estimate(
    configuration: AtomicConfiguration,
    terms: Sequence[AtomicTerm],
) -> tuple[HundEstimate | None, str | None]:
    open_shells = [shell for shell in configuration.subshells if shell.is_open]
    if not open_shells:
        singlet_s = next(
            (
                term
                for term in terms
                if term.angular_momentum == 0 and term.two_s == 0
            ),
            None,
        )
        if singlet_s is None:
            return None, "closed-shell configuration did not produce a 1S term"
        return HundEstimate(singlet_s, 0, "closed"), None
    if len(open_shells) != 1:
        return None, "Hund guidance is limited to one open subshell"

    shell = open_shells[0]
    maximum_spin = max(term.two_s for term in terms)
    maximum_l = max(
        term.angular_momentum
        for term in terms
        if term.two_s == maximum_spin
    )
    ground_term = next(
        term
        for term in terms
        if term.two_s == maximum_spin
        and term.angular_momentum == maximum_l
    )
    half = shell.capacity // 2
    if shell.electrons < half:
        filling = "less_than_half"
        two_j = abs(2 * ground_term.angular_momentum - ground_term.two_s)
    elif shell.electrons > half:
        filling = "more_than_half"
        two_j = 2 * ground_term.angular_momentum + ground_term.two_s
    else:
        filling = "half_filled"
        two_j = ground_term.two_s
    return HundEstimate(ground_term, two_j, filling), None


def j_shell_m_distribution(two_j: int, electrons: int) -> dict[int, int]:
    """Return 2M_J weights for equivalent electrons in one j subshell."""
    capacity = two_j + 1
    if two_j < 1 or two_j % 2 != 1:
        raise ValueError("electron 2j must be a positive odd integer")
    if not 0 <= electrons <= capacity:
        raise ValueError(f"occupancy must be between 0 and {capacity}")

    effective_electrons = min(electrons, capacity - electrons)
    states: dict[tuple[int, int], int] = {(0, 0): 1}
    for two_m_j in range(-two_j, two_j + 1, 2):
        updated = dict(states)
        for (chosen, total_two_m_j), count in states.items():
            if chosen == effective_electrons:
                continue
            key = (chosen + 1, total_two_m_j + two_m_j)
            updated[key] = updated.get(key, 0) + count
        states = updated
    return {
        total_two_m_j: count
        for (chosen, total_two_m_j), count in states.items()
        if chosen == effective_electrons
    }


def combine_j_distributions(
    distributions: Iterable[Mapping[int, int]],
) -> dict[int, int]:
    combined = {0: 1}
    for distribution in distributions:
        updated: dict[int, int] = {}
        for left_m, left_count in combined.items():
            for right_m, right_count in distribution.items():
                total_m = left_m + right_m
                updated[total_m] = (
                    updated.get(total_m, 0) + left_count * right_count
                )
        combined = updated
    return combined


def extract_j_levels(distribution: Mapping[int, int]) -> dict[int, int]:
    """Obtain J multiplicities from a symmetric 2M_J distribution."""
    if not distribution:
        raise ValueError("M_J distribution is empty")
    maximum_two_j = max(abs(two_m_j) for two_m_j in distribution)
    parity = next(iter(distribution)) % 2
    levels: dict[int, int] = {}
    for two_j in range(parity, maximum_two_j + 1, 2):
        count = distribution.get(two_j, 0) - distribution.get(two_j + 2, 0)
        if count < 0:
            raise RuntimeError(
                f"invalid jj decomposition at 2J={two_j}: {count}"
            )
        if count:
            levels[two_j] = count
    return levels


def jj_census(
    configuration: AtomicConfiguration,
    terms: Sequence[AtomicTerm],
) -> JjCensus:
    """Enumerate relativistic occupations and their J-coupled CSF counts."""
    open_shells = [shell for shell in configuration.subshells if shell.is_open]
    if not open_shells:
        census = ((0, 1),)
        return JjCensus(
            configurations=(RelativisticConfiguration(
                label="closed shells",
                occupations=(),
                levels=census,
                microstates=1,
            ),),
            jj_levels=census,
            ls_levels=census,
        )

    rows: list[RelativisticConfiguration] = []
    jj_levels: dict[int, int] = {}
    option_sets = [_relativistic_options(shell) for shell in open_shells]
    for selected_options in itertools.product(*option_sets):
        occupations = tuple(
            occupation
            for shell_options in selected_options
            for occupation in shell_options
        )
        levels = extract_j_levels(combine_j_distributions(
            j_shell_m_distribution(
                occupation.two_j,
                occupation.electrons,
            )
            for occupation in occupations
        ))
        expected_states = math.prod(
            math.comb(occupation.two_j + 1, occupation.electrons)
            for occupation in occupations
        )
        level_states = sum(
            count * (two_j + 1)
            for two_j, count in levels.items()
        )
        if level_states != expected_states:
            raise RuntimeError(
                f"jj row accounts for {level_states} of {expected_states} states"
            )
        for two_j, count in levels.items():
            jj_levels[two_j] = jj_levels.get(two_j, 0) + count
        rows.append(RelativisticConfiguration(
            label=" ".join(occupation.label for occupation in occupations),
            occupations=occupations,
            levels=tuple(sorted(levels.items())),
            microstates=expected_states,
        ))

    ls_levels = j_level_counts(terms)
    census = JjCensus(
        configurations=tuple(rows),
        jj_levels=tuple(sorted(jj_levels.items())),
        ls_levels=tuple(ls_levels.items()),
    )
    if not census.consistent:
        raise RuntimeError("jj and LS level censuses disagree")
    return census


def _relativistic_options(
    shell: AtomicSubshell,
) -> tuple[tuple[RelativisticOccupation, ...], ...]:
    lower_two_j = 2 * shell.angular_momentum - 1
    upper_two_j = 2 * shell.angular_momentum + 1
    lower_capacity = max(0, 2 * shell.angular_momentum)
    upper_capacity = 2 * shell.angular_momentum + 2
    minimum_lower = max(0, shell.electrons - upper_capacity)
    maximum_lower = min(shell.electrons, lower_capacity)

    options = []
    for lower_electrons in range(maximum_lower, minimum_lower - 1, -1):
        upper_electrons = shell.electrons - lower_electrons
        occupations = []
        if lower_electrons:
            occupations.append(RelativisticOccupation(
                label=(
                    f"{shell.name}_{format_half(lower_two_j)}^{lower_electrons}"
                ),
                two_j=lower_two_j,
                electrons=lower_electrons,
            ))
        if upper_electrons:
            occupations.append(RelativisticOccupation(
                label=(
                    f"{shell.name}_{format_half(upper_two_j)}^{upper_electrons}"
                ),
                two_j=upper_two_j,
                electrons=upper_electrons,
            ))
        options.append(tuple(occupations))
    return tuple(options)


def analyze_atomic_multiplets(configuration: str) -> dict[str, object]:
    """Return a serializable LS and jj symmetry census."""
    parsed = parse_atomic_configuration(configuration)
    return analyze_parsed_configuration(parsed)


def analyze_parsed_configuration(
    configuration: AtomicConfiguration,
) -> dict[str, object]:
    distribution = configuration_microstates(configuration)
    terms = terms_for_configuration(configuration)
    blocks = j_level_counts(terms)
    relativistic = jj_census(configuration, terms)
    hund, hund_note = hund_ground_estimate(configuration, terms)
    determinant_count = sum(distribution.values())
    combinatorial_count = math.prod(
        math.comb(shell.capacity, shell.electrons)
        for shell in configuration.subshells
    )
    ls_count = sum(
        term.occurrences * term.degeneracy
        for term in terms
    )
    j_count = sum(
        count * (two_j + 1)
        for two_j, count in blocks.items()
    )
    if len({determinant_count, combinatorial_count, ls_count, j_count}) != 1:
        raise RuntimeError("atomic state-count checks disagree")

    return {
        "schema_version": ATOMIC_MULTIPLET_SCHEMA,
        "configuration": configuration.label,
        "electron_count": configuration.electron_count,
        "parity": configuration.parity,
        "microstate_counts": {
            "determinant_weights": determinant_count,
            "binomial_subshell_product": combinatorial_count,
            "ls_terms": ls_count,
            "j_levels": j_count,
            "consistent": True,
        },
        "terms": [
            {
                "term": term.label,
                "L": term.angular_momentum,
                "two_s": term.two_s,
                "multiplicity": term.multiplicity,
                "occurrences": term.occurrences,
                "degeneracy_per_occurrence": term.degeneracy,
                "levels": [
                    {
                        "two_j": two_j,
                        "j": format_half(two_j),
                        "degeneracy": two_j + 1,
                        "lande_g_ls": lande_g(term, two_j),
                    }
                    for two_j in term_j_values(term)
                ],
            }
            for term in terms
        ],
        "j_parity_blocks": [
            {
                "two_j": two_j,
                "j": format_half(two_j),
                "parity": configuration.parity,
                "levels": count,
                "magnetic_sublevels": count * (two_j + 1),
            }
            for two_j, count in blocks.items()
        ],
        "hund_ground": (
            {
                "term": hund.term.label,
                "two_j": hund.two_j,
                "j": format_half(hund.two_j),
                "filling": hund.filling,
            }
            if hund is not None
            else None
        ),
        "hund_note": hund_note,
        "jj_coupling": {
            "configurations": [
                {
                    "configuration": row.label,
                    "microstates": row.microstates,
                    "j_levels": [
                        {
                            "two_j": two_j,
                            "j": format_half(two_j),
                            "csfs": count,
                        }
                        for two_j, count in row.levels
                    ],
                }
                for row in relativistic.configurations
            ],
            "jj_census": [
                {
                    "two_j": two_j,
                    "j": format_half(two_j),
                    "levels": count,
                }
                for two_j, count in relativistic.jj_levels
            ],
            "ls_census": [
                {
                    "two_j": two_j,
                    "j": format_half(two_j),
                    "levels": count,
                }
                for two_j, count in relativistic.ls_levels
            ],
            "consistent": relativistic.consistent,
        },
        "scope": {
            "provides": [
                "LS term and occurrence census",
                "allowed J and parity level counts",
                "jj-coupled relativistic occupation and CSF counts",
                "pure-LS Lande factors",
                "single-open-subshell Hund guidance",
            ],
            "does_not_provide": [
                "radial integrals or level energies",
                "spin-orbit splittings or term mixing",
                "unique LS labels for relativistic ASFs",
                "configuration-interaction coefficients",
            ],
        },
    }


def format_half(two_value: int) -> str:
    if two_value % 2 == 0:
        return str(two_value // 2)
    return f"{two_value}/2"


__all__ = [
    "ATOMIC_MULTIPLET_SCHEMA",
    "AtomicConfiguration",
    "AtomicConfigurationError",
    "AtomicSubshell",
    "AtomicTerm",
    "JjCensus",
    "MAX_MULTIPLET_MICROSTATES",
    "RelativisticConfiguration",
    "RelativisticOccupation",
    "analyze_atomic_multiplets",
    "analyze_parsed_configuration",
    "atomic_configuration",
    "combine_j_distributions",
    "decompose_ls",
    "extract_j_levels",
    "format_half",
    "j_level_counts",
    "j_shell_m_distribution",
    "jj_census",
    "lande_g",
    "parse_atomic_configuration",
    "shell_microstate_distribution",
    "term_j_values",
    "terms_for_configuration",
]
