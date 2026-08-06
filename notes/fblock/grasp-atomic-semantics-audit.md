# GRASP atomic-state semantics audit

The generic GRASP path now preserves multireference generation policies,
binds RMCDHF and RCI selections to labeled CSF blocks, and distinguishes
spectroscopic reference orbitals from correlation layers. Five live
GRASP2018 cases reproduce the manual's CSF and ASF counts exactly.

## Scope

This pass covered the failure modes that the earlier catalog sweep did not:

- multiple configurations in one generation list;
- multiple generation lists with different excitation ranks or 2J ranges;
- S, SD, and SDT substitution semantics;
- RMCDHF's conditional level-weight prompt;
- ordered `(2J, parity, NCSF)` block labels and selected ASF counts;
- reference-orbital and correlation-orbital node policy;
- staged orbital birth; and
- blockwise summary-energy parsing.

The following work was already reviewed and was excluded from this scan:

- the 15 corrected charge/slug records in Gd, Lu, Pa, Cm, and Lr;
- all 616 complete catalog rows reproducing their single-configuration
  electron counts, J/parity blocks, and NCSF values with `rcsfgenerate`;
- the 17 explicitly incomplete Y rows;
- the Ce IV `4f1` RMCDHF and low-frequency Breit RCI reference energies; and
- the refusal to claim catalog-state transfer through unconstrained NWChem or
  PySCF atomic SCF.

## Source contracts

The bundled GRASP2018 manual supplies the controlling semantics:

- A multireference is a list of configurations sharing an active set and
  excitation policy. In the section 6.1 example, excitation rank 2 includes
  both singles and doubles and produces one `J=0+` block with 361 CSFs
  ([manual, lines 11-13](../../chemtools/data/grasp/docs/part_ii_generating_lists_of_csfs/02_Running_the_CSFs_generation_programs.md)).
- A second generation list starts after answering `Generate more lists? y`.
  Section 6.6 uses rank 2 for the first list and `-2` for a second list whose
  correlation orbitals must remain doubly occupied
  ([manual, lines 40-45](../../chemtools/data/grasp/docs/part_ii_generating_lists_of_csfs/02_Running_the_CSFs_generation_programs.md)).
- The mixed-parity example uses different 2J ranges for its two lists and
  produces `1/2+`, `1/2-`, and `3/2-` blocks in that order
  ([manual, lines 51-57](../../chemtools/data/grasp/docs/part_ii_generating_lists_of_csfs/02_Running_the_CSFs_generation_programs.md)).
- Reference orbitals are spectroscopic and must have hydrogenic node counts.
  Correlation orbitals have no node-count restriction, and the manual calls
  for adding and optimizing them one layer at a time
  ([manual, lines 23-46](../../chemtools/data/grasp/docs/part_i_overview_of_grasp2018/04_Important_concepts_and_aspects_of_processing.md)).
- In the Li correlation example, GRASP varies `3*` and leaves the
  spectroscopic-orbital answer blank. The manual says that none of the new
  orbitals are spectroscopic
  ([manual, lines 43-49](../../chemtools/data/grasp/docs/part_iii_sample_runs/01_Running_the_application_programs.md)).

These are prompt-sequence contracts. A shifted answer can fail visibly, but
the orbital-role and block-order mistakes can also converge to a plausible
wrong state.

## Confirmed defects and fixes

### 1. Independent generation lists were inexpressible

The old builder accepted one flat configuration array, one active set, one 2J
range, and one excitation rank. Its documentation suggested setting
`generate_more=True`, which ended the supplied input while GRASP was waiting
for the next list.

`rcsfgenerate_input` now accepts `additional_lists`, each with its own
configurations, active set, 2J limits, and excitation rank. Bare
`generate_more=True` is rejected
([builder, lines 64-132](../../chemtools/programs/grasp/input/heredoc.py)). The
MCP runner and workflow planners expose the same structure.

### 2. Correlation orbitals defaulted to spectroscopic

The generic planners previously emitted `*` for both the varied and
spectroscopic masks even when `excitations` created correlation orbitals.
That applies node constraints to orbitals for which the radial node count is
not part of the physical definition.

Reference-only workflows still default to varying and marking all orbitals as
spectroscopic. Any correlation workflow must now provide both masks
explicitly. The recommended layer policy is `orbitals_to_optimize="n*"` and
`spectroscopic_orbitals=""` for a newly added layer
([workflow policy, lines 40-85](../../chemtools/programs/grasp/strategy/workflows.py)).

### 3. Single-ASF RMCDHF input was shifted

GRASP omits the level-weight prompt when exactly one ASF is selected. The old
builder always sent `5`, so GRASP read it as the orbital mask and shifted every
remaining answer.

The builder now parses the selections, rejects malformed or empty totals, and
emits a weight only when more than one ASF is selected
([builder, lines 304-370](../../chemtools/programs/grasp/input/heredoc.py)).

### 4. Positional ASF selections were not bound to J and parity

The generic workflow prepared RMCDHF selections before `rcsfgenerate` had
produced its block table. A caller could supply valid selection ranges in the
wrong block order and GRASP would accept them.

Generic plans and direct RMCDHF/RCI calls now require an ordered expected block
table containing `two_j`, `parity`, and `ncsf`. Execution compares that table
with the generated `.c` file before reading positional selections
([runner, lines 84-154](../../chemtools/programs/grasp/strategy/runner.py)).

### 5. Catalog validation accepted partial ASF manifolds

The binary inspector correctly permits subset calculations, but the f-block
catalog's configuration-average contract selects every ASF in every block.
The catalog validator previously inherited the generic subset semantics and
still returned `valid: true`.

It now requires each mixing block's `eigenstate_count` to equal the catalog
NCSF count
([catalog validator, lines 43-93](../../chemtools/reference/fblock_grasp.py)).

### 6. The summary parser treated the first block as the ground state

GRASP writes eigenenergies by block, not in global energy order. The parser
used the first row as `ground_energy_au`, which is wrong whenever a later J
block lies lower. It now takes the minimum across all retained ASF energies
([summary parser, lines 120-135](../../chemtools/programs/grasp/parse/sum_file.py)).

### 7. One staged-birth recipe contradicted the recorded failure mode

`Th ion3_7s1` is the only staged catalog state with one CSF. The project notes
record that staged variation of one orbital in a one-CSF state crashes RMCDHF
at input. The planner nevertheless emitted a `7s` stage.

The planner now suppresses that stage and instructs the caller to use the
recorded converged donor directly
([planner, lines 475-526 and 588-605](../../chemtools/reference/fblock_plan.py)).

### 8. The documented radial-file source sequence was malformed

`rwfnestimate_input(sources=["1", "rwfn.inp", "2"])` inserted a subshell
wildcard after every array item, treating the filename as a new source. The
helper now consumes the item after source `1` as the filename. The existing
`file:rwfn.inp` shorthand remains valid
([builder, lines 225-301](../../chemtools/programs/grasp/input/heredoc.py)).

## Live GRASP2018 evidence

Run:

```text
apptainer exec /home/charlie/mycontainers/grasp2018.sif \
  .venv/bin/python scripts/check_grasp_atomic_semantics.py \
  --scratch /home/charlie/scratch/chemtoolsmcp-grasp-atomic-semantics-20260805-final
```

The retained `evidence.json` reports all five cases as passed:

| case | observed result |
|---|---|
| Be multireference SD, n=4 | `J=0+`, 361 CSFs |
| Li reference, even and odd lists | `1/2+`, `1/2-`, `3/2-`, one CSF each |
| Li merged T(n=5) plus SD(n=7) lists | `1/2-`: 2408; `3/2-`: 4174 CSFs |
| Li reference RMCDHF | three matching blocks, one selected ASF each |
| Li 2s n=3 correlation layer | `J=1/2+`, 79 CSFs, one ASF; varied `3*`; blank spectroscopic mask |

Both RMCDHF cases reached the explicit `RMCDHF: Execution complete` marker.
The correlation case also proves the single-ASF prompt sequence: its complete
tail is `1 / 3* / blank / 100`, with no weight answer.

The earlier Ce IV `4f1` RCI result supplies a heavier physics spot check. It
produces the expected odd `J=5/2` and `J=7/2` components, with `5/2` lower, and
a splitting of 2087.50 cm-1. An independent spectroscopy reference reports
the NIST free-ion levels at 0 and 2253 cm-1
([Reid, *Electronic Structure Calculations*, section 3.1](https://spcs.canterbury.ac.nz/~mfr24/electronicstructure/00electronic.pdf)), so
the single-configuration calculation has the right term ordering and is 7.3%
low in the fine-structure interval. That is a method limitation, not a block
labeling error.

## Independent LS and jj census

`analyze_atomic_multiplets` derives allowed LS terms, term recurrence counts,
J/parity levels, and relativistic occupation counts from determinant weights.
The four state counts (binomial subshell product, determinant weights, LS
terms, and J levels) must agree. Its predicted J/parity and CSF counts match
all 616 complete catalog configurations exactly.

The retained differential harness compares the MCP implementation with the
standalone source over every `s` through `f` occupancy plus five multishell
cases:

```bash
.venv/bin/python scripts/check_atomic_multiplet_port.py \
  --reference-root /path/to/multiplet_generator
```

The 2026-08-06 run reports 37 agreements, zero disagreements, and zero target
refusals. Its JSON evidence is in
`~/scratch/chemtoolsmcp-atomic-multiplet-differential-20260806.json`.

`validate_grasp_csf_angular_census` applies the jj count to the relativistic
occupations in a generated `.c` file. A live pass over 621 retained GRASP files
found zero occupation/J multiplicity failures. The 616 catalog files and two
unrestricted reference cases contain their full J manifolds. Three correlation
or multireference cases correctly report restricted manifolds because their
generation lists requested only selected J blocks.

This check is independent of the catalog and of GRASP's block totals. It can
catch a missing or duplicated coupling path for a represented configuration.
It cannot prove that the generation recipe produced every intended excited
configuration.

## Remaining limits

- The live correlation tests use the manual's Li and Be cases. They validate
  GRASP semantics and prompt accounting, but they do not constitute a
  correlated f-block benchmark campaign.
- The multiplet preflight enumerates allowed LS terms, but the catalog does not
  carry computed ASF term labels. J/parity, CSF counts, ASF counts, dominant
  configurations, and energy ordering are checked; assigning a specific LS
  term to a mixed ASF still requires `jj2lsj` output or an external
  spectroscopy source.
- The 17 Y extension rows still lack enough GRASP input fields to generate
  CSFs.
- NWChem was not used for this check. Its generic atomic path cannot preserve
  the catalog occupation, so agreement there would not establish that the
  same atomic state was computed.
- Target-specific SO3 occupation constraints remain separate work. The MCP
  correctly refuses to claim cross-program state transfer until that mapping
  and a post-SCF population check exist.
