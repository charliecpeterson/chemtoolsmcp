"""Typed access to committed and external scientific reference material."""

from chemtools.reference.atomic_multiplets import (
    ATOMIC_MULTIPLET_SCHEMA,
    AtomicConfiguration,
    AtomicConfigurationError,
    AtomicSubshell,
    analyze_atomic_multiplets,
    parse_atomic_configuration,
)
from chemtools.reference.fblock import (
    FBLOCK_DATASET_SCHEMA,
    FBlockCatalog,
    FBlockCatalogLoadError,
    FBlockDatasetMetadata,
    FBlockElement,
    FBlockState,
    bundled_fblock_directory,
    load_fblock_catalog,
)
from chemtools.reference.fblock_lookup import (
    FBLOCK_LOOKUP_SCHEMA,
    FBlockLookupResult,
    lookup_grasp_fblock_state,
)
from chemtools.reference.fblock_grasp import (
    FBLOCK_GRASP_VALIDATION_SCHEMA,
    validate_grasp_fblock_artifacts,
)
from chemtools.reference.fblock_donors import (
    FBLOCK_DONOR_ALIAS_SCHEMA,
    FBlockDonorAliasManifest,
    FBlockDonorAliasRecord,
    load_fblock_donor_alias_manifest,
)
from chemtools.reference.fblock_plan import (
    ATSPElementRecipes,
    ATSPStateRecipe,
    FBLOCK_PLAN_SCHEMA,
    FBlockAtomicPlan,
    load_atsp_element_recipes,
    plan_fblock_atomic_state,
)
from chemtools.reference.fblock_semantics import (
    FBLOCK_STATE_SEMANTICS_SCHEMA,
    FBlockStateSemanticsManifest,
    load_fblock_state_semantics,
)
from chemtools.reference.grasp_angular_census import (
    GRASP_ANGULAR_CENSUS_SCHEMA,
    validate_grasp_csf_angular_census,
)
from chemtools.reference.external_corpus import (
    DEFAULT_CASE_BYTE_LIMIT,
    REFERENCE_CORPUS_SCHEMA,
    ReferenceManifestError,
    load_reference_manifest,
    verify_reference_case,
)

__all__ = [
    "ATOMIC_MULTIPLET_SCHEMA",
    "AtomicConfiguration",
    "AtomicConfigurationError",
    "AtomicSubshell",
    "FBLOCK_DATASET_SCHEMA",
    "FBlockCatalog",
    "FBlockCatalogLoadError",
    "FBlockDatasetMetadata",
    "FBlockElement",
    "FBlockState",
    "FBLOCK_LOOKUP_SCHEMA",
    "FBLOCK_GRASP_VALIDATION_SCHEMA",
    "FBLOCK_DONOR_ALIAS_SCHEMA",
    "FBlockDonorAliasManifest",
    "FBlockDonorAliasRecord",
    "FBlockLookupResult",
    "FBLOCK_STATE_SEMANTICS_SCHEMA",
    "FBlockStateSemanticsManifest",
    "GRASP_ANGULAR_CENSUS_SCHEMA",
    "DEFAULT_CASE_BYTE_LIMIT",
    "REFERENCE_CORPUS_SCHEMA",
    "ReferenceManifestError",
    "ATSPElementRecipes",
    "ATSPStateRecipe",
    "FBLOCK_PLAN_SCHEMA",
    "FBlockAtomicPlan",
    "analyze_atomic_multiplets",
    "bundled_fblock_directory",
    "load_fblock_catalog",
    "load_atsp_element_recipes",
    "load_fblock_donor_alias_manifest",
    "load_fblock_state_semantics",
    "load_reference_manifest",
    "lookup_grasp_fblock_state",
    "plan_fblock_atomic_state",
    "parse_atomic_configuration",
    "validate_grasp_csf_angular_census",
    "validate_grasp_fblock_artifacts",
    "verify_reference_case",
]
