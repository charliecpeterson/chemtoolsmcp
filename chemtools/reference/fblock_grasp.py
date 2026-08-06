"""Validate GRASP CSF and mixing artifacts against one catalog state."""

from __future__ import annotations

from pathlib import Path

from chemtools.programs.grasp.binary.mixing import inspect_grasp_mixing
from chemtools.programs.grasp.parse.csf import CsfDocument, load_grasp_csf_list
from chemtools.reference.fblock import load_fblock_catalog
from chemtools.reference.fblock_configuration import parse_shell_configuration
from chemtools.reference.fblock_models import FBlockElement, FBlockState


FBLOCK_GRASP_VALIDATION_SCHEMA = "chemtools.fblock-grasp-validation/1"
_ANGULAR_MOMENTUM = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


def validate_grasp_fblock_artifacts(
    element: str,
    state: str,
    csf_path: str | Path,
    *,
    mixing_path: str | Path | None = None,
    level_limit: int = 64,
    component_limit: int = 3,
) -> dict[str, object]:
    """Check generated CSFs and optional ASF mixing against the catalog."""
    catalog = load_fblock_catalog()
    element_record = catalog.element(element)
    state_record = element_record.state(state)
    if not state_record.confline:
        raise ValueError(
            f"{element_record.symbol}.{state_record.slug} has no complete "
            "GRASP configuration to validate"
        )
    csf_document = load_grasp_csf_list(csf_path)
    csf_validation = _validate_csf(
        element_record,
        state_record,
        csf_document,
    )
    mixing = None
    if mixing_path is not None:
        mixing = inspect_grasp_mixing(
            mixing_path,
            level_limit=level_limit,
            component_limit=component_limit,
            csf_path=csf_path,
        )
        _validate_complete_asf_manifold(state_record, mixing)
    return {
        "schema_version": FBLOCK_GRASP_VALIDATION_SCHEMA,
        "query": {
            "element": element_record.symbol,
            "state": state_record.slug,
        },
        "catalog": {
            "dataset_id": catalog.metadata.dataset_id,
            "dataset_version": catalog.metadata.dataset_version,
            "sha256": catalog.metadata.catalog_sha256,
        },
        "csf": csf_validation,
        "mixing": mixing,
        "valid": True,
    }


def _validate_complete_asf_manifold(
    state: FBlockState,
    mixing: dict[str, object],
) -> None:
    blocks = mixing.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(state.ncsf):
        raise ValueError(
            f"GRASP mixing block count does not match {state.slug}"
        )
    for index, (block, expected_count) in enumerate(
        zip(blocks, state.ncsf),
        start=1,
    ):
        if not isinstance(block, dict):
            raise ValueError(f"GRASP mixing block {index} is malformed")
        selected_count = block.get("eigenstate_count")
        if selected_count != expected_count:
            raise ValueError(
                f"GRASP mixing block {index} selected {selected_count} ASFs; "
                f"the catalog configuration average requires all "
                f"{expected_count}"
            )
    checks = mixing.get("checks")
    if isinstance(checks, dict):
        checks["catalog_all_asfs_selected"] = True
        checks["catalog_asf_counts_match"] = True


def catalog_parity(state: FBlockState) -> str:
    configuration = state.confline or state.config
    if configuration == "closed":
        return "+"
    exponent = sum(
        _ANGULAR_MOMENTUM[shell.orbital] * shell.electrons
        for shell in parse_shell_configuration(configuration)
    )
    return "+" if exponent % 2 == 0 else "-"


def _validate_csf(
    element: FBlockElement,
    state: FBlockState,
    document: CsfDocument,
) -> dict[str, object]:
    expected_electrons = element.atomic_number - state.ion
    if document.electron_count != expected_electrons:
        raise ValueError(
            f"GRASP CSF electron count {document.electron_count} does not "
            f"match {element.symbol}.{state.slug}: {expected_electrons}"
        )
    parity = catalog_parity(state)
    expected_blocks = [
        {"j": j, "parity": parity, "ncsf": count}
        for j, count in zip(state.j_blocks, state.ncsf)
    ]
    actual_blocks = [
        {
            "j": block.j_label,
            "parity": block.parity,
            "ncsf": len(block.entries),
        }
        for block in document.blocks
    ]
    if actual_blocks != expected_blocks:
        raise ValueError(
            f"GRASP CSF blocks do not match {element.symbol}.{state.slug}: "
            f"expected {expected_blocks}, found {actual_blocks}"
        )
    return {
        "path": str(document.source),
        "size_bytes": document.size_bytes,
        "sha256": document.sha256,
        "electron_count": document.electron_count,
        "csf_count": document.csf_count,
        "blocks": actual_blocks,
        "checks": {
            "electron_count_matches_ion_charge": True,
            "j_parity_blocks_match_catalog": True,
            "csf_counts_match_catalog": True,
        },
    }


__all__ = [
    "FBLOCK_GRASP_VALIDATION_SCHEMA",
    "catalog_parity",
    "validate_grasp_fblock_artifacts",
]
