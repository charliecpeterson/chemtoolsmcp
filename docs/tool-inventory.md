# MCP tool inventory

Generated from the live decorator registry and tool definitions.
Do not edit counts by hand. Regenerate with:

```bash
.venv/bin/python scripts/generate_tool_inventory.py --write-docs
```

The JSON companion contains every tool description, input schema, and advertised output schema.

## Summary

- Default protocol version: `2025-11-25`
- Supported protocol versions: `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25`, `2026-07-28`
- Canonical tool definitions: 313
- Advertised legacy tool definitions: 9
- Hidden MCP aliases: 15
- Total callable MCP names: 337

### Programs

| Program | Tools |
| --- | ---: |
| `generic` | 66 |
| `nwchem` | 101 |
| `molcas` | 41 |
| `dirac` | 35 |
| `grasp` | 49 |
| `qe` | 18 |
| `qmcpack` | 12 |
| `orca` | 0 |

### Capabilities

| Capability | Tools |
| --- | ---: |
| `none` | 266 |
| `registry` | 18 |
| `runner_profile` | 2 |
| `executable_or_scheduler` | 5 |
| `executable` | 28 |
| `scheduler` | 3 |

### Modes

| Mode | All programs |
| --- | ---: |
| `analysis` | 284 |
| `local` | 319 |
| `hpc` | 322 |

### Program filters

Counts include the 66 generic tools where the active mode permits them.

| Program filter | Analysis | Local | HPC |
| --- | ---: | ---: | ---: |
| `nwchem` | 149 | 164 | 167 |
| `molcas` | 101 | 106 | 107 |
| `dirac` | 96 | 100 | 101 |
| `grasp` | 91 | 114 | 115 |
| `qe` | 79 | 83 | 84 |
| `qmcpack` | 73 | 77 | 78 |
| `orca` | 61 | 65 | 66 |

## Compatibility aliases

Aliases remain callable but are omitted from `tools/list`.

| Alias | Canonical tool | Program | Capability | Contract | Deprecated since | Remove after |
| --- | --- | --- | --- | --- | --- | --- |
| `check_nwchem_run_status` | `get_nwchem_run_status` | `nwchem` | `executable` | `unverified` | `0.1.0` |  |
| `diagnose_nwchem_output` | `analyze_nwchem_case` | `nwchem` | `none` | `unverified` | `0.1.0` |  |
| `prepare_nwchem_run` | `launch_nwchem_run` | `nwchem` | `executable` | `unverified` | `0.1.0` |  |
| `render_nwchem_basis_from_input` | `render_nwchem_basis_block` | `nwchem` | `none` | `unverified` | `0.1.0` |  |
| `render_nwchem_ecp_from_elements` | `render_nwchem_ecp_block` | `nwchem` | `none` | `unverified` | `0.1.0` |  |
| `render_with_orbitron` | `visualize` | `generic` | `none` | `unverified` | `0.1.0` |  |
| `resolve_nwchem_basis_setup` | `render_nwchem_basis_setup` | `nwchem` | `none` | `unverified` | `0.1.0` |  |
| `resolve_nwchem_ecp` | `render_nwchem_ecp_block` | `nwchem` | `none` | `unverified` | `0.1.0` |  |
| `review_nwchem_case` | `analyze_nwchem_case` | `nwchem` | `none` | `unverified` | `0.1.0` |  |
| `review_nwchem_followup_outcome` | `compare_nwchem_runs` | `nwchem` | `none` | `unverified` | `0.1.0` |  |
| `search_knowledge_cards` | `search_knowledge` | `generic` | `none` | `unverified` | `0.1.0` |  |
| `suggest_nwchem_scf_fix_strategy` | `suggest_nwchem_recovery` | `nwchem` | `none` | `unverified` | `0.1.0` |  |
| `suggest_nwchem_state_recovery_strategy` | `suggest_nwchem_recovery` | `nwchem` | `none` | `unverified` | `0.1.0` |  |
| `summarize_cube_file` | `parse_cube_file` | `generic` | `none` | `unverified` | `0.1.0` |  |
| `summarize_nwchem_case` | `analyze_nwchem_case` | `nwchem` | `none` | `unverified` | `0.1.0` |  |

## Advertised legacy tools

| Legacy tool | Canonical replacement | Deprecated since | Remove after |
| --- | --- | --- | --- |
| `register_nwchem_run` | `register_run` | `0.1.0` |  |
| `update_nwchem_run_status` | `update_run_status` | `0.1.0` |  |
| `list_nwchem_runs` | `list_runs` | `0.1.0` |  |
| `get_nwchem_run_summary` | `get_run_summary` | `0.1.0` |  |
| `create_nwchem_campaign` | `create_campaign` | `0.1.0` |  |
| `get_nwchem_campaign_status` | `get_campaign_status` | `0.1.0` |  |
| `get_nwchem_campaign_energies` | `get_campaign_energies` | `0.1.0` |  |
| `create_nwchem_workflow` | `create_workflow` | `0.1.0` |  |
| `advance_nwchem_workflow` | `advance_workflow` | `0.1.0` |  |

## Entrypoint aliases

| Entrypoint | Replacement | State | Contract | Deprecated since | Remove after |
| --- | --- | --- | --- | --- | --- |
| `chemtools-nwchem` | `chemtools` | `callable_deprecated` | `verified_equivalent` | `0.1.0` |  |
| `chemtools-nwchem-docs` | `chemtools` | `legacy_distinct_surface` | `not_equivalent` | `0.1.0` |  |

## Python import shims

| Import | Replacement | State | Deprecated since | Remove after |
| --- | --- | --- | --- | --- |
| `chemtools` | focused chemtools application, execution, integration, persistence, program, and reference modules | `compatibility_deprecated` | `0.1.0` |  |
| `chemtools.api` | focused chemtools.core and chemtools.programs modules | `compatibility_deprecated` | `0.1.0` |  |
| `chemtools.api_input` | chemtools.programs.nwchem.input and strategy.workflow_planner | `compatibility_deprecated` | `0.1.0` |  |
| `chemtools.api_strategy` | chemtools.programs.nwchem.strategy | `compatibility_deprecated` | `0.1.0` |  |
| `chemtools.mcp.nwchem` | chemtools.mcp.cli and chemtools.mcp.dispatch | `compatibility_deprecated` | `0.1.0` |  |
| `chemtools.execution.executors` | chemtools.execution | `compatibility_deprecated` | `0.1.0` |  |

## Tools

| Tool | Program | Capability | Visible modes |
| --- | --- | --- | --- |
| `advance_nwchem_workflow` | `nwchem` | `registry` | `analysis`, `local`, `hpc` |
| `advance_workflow` | `generic` | `registry` | `analysis`, `local`, `hpc` |
| `analyze_atomic_multiplets` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `analyze_case` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `analyze_dirac_open_shell` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `analyze_dirac_open_shell_quality` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `analyze_geometry_with_orbitron` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `analyze_grasp_case` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `analyze_molcas_active_space` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `analyze_molcas_case` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `analyze_nwchem_case` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `analyze_nwchem_frontier_orbitals` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `analyze_nwchem_imaginary_modes` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `analyze_orbitals_with_orbitron` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `analyze_populations_with_orbitron` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `analyze_qmcpack_dmc_input_series` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `analyze_qmcpack_dmc_series` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `analyze_vibrations_with_orbitron` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `append_grasp_session_note` | `grasp` | `executable` | `local`, `hpc` |
| `append_session_log` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `apply_dirac_reorder_to_input` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `apply_molcas_recovery` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `apply_recovery` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `basis_library_summary` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `check_molcas_active_space_consistency` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `check_nwchem_freq_plausibility` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `check_nwchem_geometry_plausibility` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `check_nwchem_memory_fit` | `nwchem` | `executable_or_scheduler` | `local`, `hpc` |
| `check_nwchem_spin_charge_state` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `check_qe_qmcpack_conversion_ready` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `check_qmcpack_vmc_energy_gate` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `compare_cube_densities` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `compare_cube_orbital_subspaces` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `compare_cube_orbitals` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `compare_grasp_levels` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `compare_nwchem_runs` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `compare_pyscf_reference_calculation` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `compare_qmcpack_tmove_locality_shift` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `compare_qmcpack_tmove_locality_shift_from_input` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `compare_runs` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `compute_dirac_core_ip` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `compute_molcas_active_space_partition` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `compute_molcas_reaction_energy` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `compute_nwchem_harmonic_frequencies` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `compute_reaction_energy` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `convert_molecule_with_openbabel` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `create_campaign` | `generic` | `registry` | `analysis`, `local`, `hpc` |
| `create_nwchem_campaign` | `nwchem` | `registry` | `analysis`, `local`, `hpc` |
| `create_nwchem_dft_input_from_request` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `create_nwchem_dft_workflow_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `create_nwchem_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `create_nwchem_input_variant` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `create_nwchem_workflow` | `nwchem` | `registry` | `analysis`, `local`, `hpc` |
| `create_workflow` | `generic` | `registry` | `analysis`, `local`, `hpc` |
| `detect_nwchem_hpc_accounts` | `nwchem` | `scheduler` | `hpc` |
| `displace_nwchem_geometry_along_mode` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_dirac_input` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `draft_dirac_mol` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `draft_dirac_reorder_block` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `draft_initial_geometry` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `draft_input` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `draft_molcas_input` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_atom_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_cube_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_frontier_cube_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_imaginary_mode_inputs` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_mcscf_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_mcscf_retry_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_optimization_followup_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_property_check_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_pyscf_reference` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_scf_stabilization_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_tce_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_tce_restart_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_nwchem_vectors_swap_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `draft_ph_x_input` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `draft_pw2qmcpack_input` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `estimate_nwchem_freq_walltime` | `nwchem` | `executable_or_scheduler` | `local`, `hpc` |
| `evaluate_nwchem_case` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `evaluate_nwchem_cases` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `extract_geometry` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `extract_molcas_geometry` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `extract_nwchem_geometry` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `fetch_nist_atomic_reference` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `find_nwchem_examples` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `find_nwchem_restart_assets` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `find_reference_case` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `generate_nwchem_input_batch` | `nwchem` | `executable_or_scheduler` | `local`, `hpc` |
| `get_campaign_energies` | `generic` | `registry` | `analysis`, `local`, `hpc` |
| `get_campaign_status` | `generic` | `registry` | `analysis`, `local`, `hpc` |
| `get_dirac_topic_guide` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `get_grasp_container` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `get_grasp_topic_guide` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `get_molcas_orbitals` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `get_molcas_topic_guide` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `get_nwchem_campaign_energies` | `nwchem` | `registry` | `analysis`, `local`, `hpc` |
| `get_nwchem_campaign_status` | `nwchem` | `registry` | `analysis`, `local`, `hpc` |
| `get_nwchem_run_status` | `nwchem` | `executable` | `local`, `hpc` |
| `get_nwchem_run_summary` | `nwchem` | `registry` | `analysis`, `local`, `hpc` |
| `get_nwchem_topic_guide` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `get_nwchem_workflow_state` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `get_run_summary` | `generic` | `registry` | `analysis`, `local`, `hpc` |
| `get_server_mode` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `init_session_log` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `inspect_geometry` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `inspect_grasp_mixing` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `inspect_grasp_radial_wfn` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `inspect_molcas_geometry` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `inspect_nbo_with_orbitron` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `inspect_nwchem_geometry` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `inspect_nwchem_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `inspect_nwchem_runner_profiles` | `nwchem` | `runner_profile` | `local`, `hpc` |
| `inspect_periodic_electronic_structure_with_orbitron` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_artifacts` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_atoms` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_charge` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_deck` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_electrons` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_execution` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_geometry` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_ion_species` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_projectors` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_pseudopotentials` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_species` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_spin` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qe_qmcpack_conversion_valence` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qmcpack_determinant_vmc_offsets` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qmcpack_dmc_population` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qmcpack_dmc_population_from_input` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qmcpack_hdf5` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qmcpack_pseudopotential` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qmcpack_referenced_pseudopotentials` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `inspect_qmcpack_scalar` | `qmcpack` | `none` | `analysis`, `local`, `hpc` |
| `inspect_run` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `inspect_science_runtime` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `inspect_structure_identity_with_orbitron` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `inspect_with_orbitron` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `launch_nwchem_run` | `nwchem` | `executable` | `local`, `hpc` |
| `launch_run` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `lint_molcas_input` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `lint_nwchem_input` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `list_dirac_basis_sets` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `list_dirac_docs` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `list_grasp_docs` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `list_molcas_basis_sets` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `list_molcas_docs` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `list_nwchem_docs` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `list_nwchem_protocols` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `list_nwchem_runs` | `nwchem` | `registry` | `analysis`, `local`, `hpc` |
| `list_runs` | `generic` | `registry` | `analysis`, `local`, `hpc` |
| `lookup_dirac_section` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `lookup_grasp_fblock_state` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `lookup_grasp_section` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `lookup_molcas_module` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `lookup_nwchem_block_syntax` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `merge_grasp_radial_wfns` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `monitor_run` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `next_versioned_path` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `parse_cube_file` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `parse_dirac_cosci_energies` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `parse_dirac_input` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `parse_dirac_mol` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `parse_dirac_output` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `parse_dirac_reorder_block` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `parse_dirac_scf_iterations` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `parse_dirac_spinor_spectrum` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `parse_dirac_symmetry` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `parse_dirac_vecpop` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `parse_frequencies` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `parse_grasp_hfs` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `parse_grasp_levels` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `parse_grasp_lsjlbl` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `parse_grasp_ris` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `parse_grasp_rmcdhf_log` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `parse_grasp_sum` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `parse_grasp_transition` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `parse_molcas_frequencies` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `parse_molcas_inporb` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `parse_molcas_output` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `parse_molcas_rassi` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `parse_molcas_tasks` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `parse_molcas_thermochem` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `parse_molcas_trajectory` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `parse_molcas_xmldump` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_freq_progress` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_hessian` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_mcscf_output` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_mos` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_movecs` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_output` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_population_analysis` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_scf` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_tasks` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_tce_amplitudes` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_tce_output` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_thermochem` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_nwchem_trajectory` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `parse_output` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `parse_thermochem` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `parse_trajectory` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `plan_calculation` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `plan_fblock_atomic_state` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `plan_grasp_dhf_workflow` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `plan_grasp_hf_bootstrap_workflow` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `plan_grasp_nonrel_limit_workflow` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `plan_grasp_restart_from_workflow` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `plan_nwchem_calculation` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `plan_nwchem_workflow` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `plan_qe_qmcpack_conversion` | `qe` | `none` | `analysis`, `local`, `hpc` |
| `plan_recovery` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `preflight_check` | `generic` | `runner_profile` | `local`, `hpc` |
| `preflight_molecule_with_rdkit` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `prepare_dirac_atomic_start` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `prepare_dirac_cm_class_workflow` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `prepare_dirac_core_ionization` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `prepare_dirac_launch` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `prepare_dirac_x2c_bootstrap` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `prepare_molcas_atomization` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `prepare_molcas_caspt2_chain` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `prepare_molcas_casscf_setup` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `prepare_molcas_excited_states` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `prepare_molcas_irc_workflow` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `prepare_molcas_launch` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `prepare_molcas_opt_freq_workflow` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `prepare_molcas_scan_workflow` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `prepare_nwchem_freq_restart` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `prepare_nwchem_mcscf_setup` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `prepare_nwchem_next_step` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `prepare_nwchem_tce_setup` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `read_dirac_doc_excerpt` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `read_dirac_h5_geometry` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `read_dirac_h5_metadata` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `read_dirac_mo_coefficients` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `read_dirac_orbitals` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `read_grasp_doc_excerpt` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `read_grasp_session_log` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `read_molcas_doc_excerpt` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `read_nwchem_doc_excerpt` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `refine_molcas_active_space` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `register_nwchem_run` | `nwchem` | `registry` | `analysis`, `local`, `hpc` |
| `register_run` | `generic` | `registry` | `analysis`, `local`, `hpc` |
| `render_basis_set_with_bse` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `render_job_script` | `generic` | `scheduler` | `hpc` |
| `render_nwchem_basis_block` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `render_nwchem_basis_setup` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `render_nwchem_ecp_block` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `review_input` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `review_nwchem_input_request` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `review_nwchem_mcscf_case` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `review_nwchem_mcscf_followup_outcome` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `review_nwchem_progress` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `run_grasp_exe` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_hf` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_jj2lsj` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rangular` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rbiotransform` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rci` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rcsfgenerate` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rhfs` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rhfs_lsj` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_ris4` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rlevels` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rmcdhf` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rnucleus` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rsave` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rtransition` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rwfnestimate` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_rwfnmchfmcdf` | `grasp` | `executable` | `local`, `hpc` |
| `run_grasp_workflow` | `grasp` | `executable` | `local`, `hpc` |
| `run_nwchem_pyscf_matched_reference` | `nwchem` | `executable` | `local`, `hpc` |
| `run_pyscf_single_point` | `generic` | `executable` | `local`, `hpc` |
| `search_dirac_docs` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `search_grasp_docs` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `search_knowledge` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `search_molcas_docs` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `search_nwchem_docs` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `search_nwchem_forum` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `suggest_basis_set` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `suggest_dirac_basis` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `suggest_dirac_orbital_swaps` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `suggest_grasp_recovery` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `suggest_memory` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `suggest_molcas_orbital_swaps` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `suggest_molcas_recovery` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `suggest_nwchem_mcscf_active_space` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `suggest_nwchem_multiplicity_scan` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `suggest_nwchem_partition` | `nwchem` | `scheduler` | `hpc` |
| `suggest_nwchem_recovery` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `suggest_nwchem_resources` | `nwchem` | `executable_or_scheduler` | `local`, `hpc` |
| `suggest_nwchem_tce_freeze` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `suggest_nwchem_vectors_swaps` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `suggest_recovery` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `suggest_relativistic_correction` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `suggest_resources` | `generic` | `executable_or_scheduler` | `local`, `hpc` |
| `suggest_spin_state` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `summarize_dirac_outputs` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `summarize_dirac_run` | `dirac` | `none` | `analysis`, `local`, `hpc` |
| `summarize_grasp_runs` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `summarize_grasp_terms` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `summarize_molcas_output` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `summarize_molcas_outputs` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `summarize_nwchem_electronic_structure` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `summarize_nwchem_output` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `summarize_nwchem_outputs` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `summarize_output` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `summarize_run` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `swap_molcas_inporb_orbitals` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `swap_nwchem_movecs` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `tail_nwchem_output` | `nwchem` | `executable` | `local`, `hpc` |
| `terminate_nwchem_run` | `nwchem` | `executable` | `local`, `hpc` |
| `track_nwchem_spin_state` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `try_molcas_run_with_recovery` | `molcas` | `executable` | `local`, `hpc` |
| `update_nwchem_run_status` | `nwchem` | `registry` | `analysis`, `local`, `hpc` |
| `update_run_status` | `generic` | `registry` | `analysis`, `local`, `hpc` |
| `validate_grasp_csf_angular_census` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `validate_grasp_fblock_artifacts` | `grasp` | `none` | `analysis`, `local`, `hpc` |
| `validate_molcas_caspt2_setup` | `molcas` | `none` | `analysis`, `local`, `hpc` |
| `validate_nwchem_tce_setup` | `nwchem` | `none` | `analysis`, `local`, `hpc` |
| `visualize` | `generic` | `none` | `analysis`, `local`, `hpc` |
| `watch_multiple_runs` | `generic` | `executable` | `local`, `hpc` |
| `watch_nwchem_run` | `nwchem` | `executable` | `local`, `hpc` |
