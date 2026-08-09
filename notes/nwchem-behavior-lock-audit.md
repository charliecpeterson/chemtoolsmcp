# NWChem behavior-lock audit

Audit date: 2026-08-06

This note selects the first real NWChem workflows for the Chemtools
simplification behavior lock. The source files remain read-only under the
external corpus root `/home/charlie/input_examples`; no corpus bytes were
copied into this repository.

## Custody and review status

The selected `nwchem/hard_cases/` tree is currently untracked in the external
`input_examples` Git repository. Its notes identify it as a curated personal
collection, but that does not establish redistribution terms. Treat every
artifact as:

- storage tier: `external_reference`
- source and attribution: Charles Peterson's local research/example corpus
- redistribution: `review_required`
- scientific status: case-specific review required before promotion

The cases are small enough for bounded external checks. Inputs, outputs, and
notes for cases 01 through 05 total about 12 MB. Binary wavefunctions remain
outside the initial parser and diagnosis lock; workflows that require restart
validation must add those companions explicitly later.

## Selected cases

| Case | Expert decision to preserve | Current guided status |
| --- | --- | --- |
| `01_fecn6_lowspin_fragment` | The atomic guess reaches a high-spin state; the forced low-spin fragment workflow reaches a lower t2g5 state | Both runs remain successful when inspected alone. `compare_runs` reports the 6-to-2 multiplicity change and conditional 0.50538692213 Ha lowering, while the MO parser recovers one 87% visible Fe d, t2g SOMO for the solution. `plan_recovery` requires a fresh target-state input rather than a misleading same-multiplicity swap |
| `02_hexaaquairon_swap_chain` | Inspect SOMOs, try fragment MOs, locate buried Fe d orbitals, apply explicit swaps, reconverge, then inspect again | The notes and current `hexaaquairon.out` energy disagree; current diagnosis accepts the documented bad fragment state and rejects the documented corrected swap state |
| `03_feo_scf_convergence` | Two individually clean calculations require a multiplicity comparison; the quintet lies 0.10209484695 Ha below the triplet | `compare_runs` reports the conditional energy ordering and 3-to-5 multiplicity change without assigning the ground state |
| `04_ferrocene_basis_stepping` | A run that eventually converges can still show dangerous DIIS instability; projected small-basis orbitals plus damping give a controlled path to the same state | `inspect_run` retains the transient energy jump and DIIS error while leaving the task successful. `plan_recovery` prepares optional damping from the converged checkpoint and records smaller-basis projection as a reviewed fallback |
| `05_crco6_freq_restart` | A converged optimization with one large imaginary torsional mode is a saddle; displacement and reoptimization reach the lower minimum | The verdict transition and significant mode are pinned, and `plan_recovery` returns plus/minus displacement inputs without writing files. The directory name and parent hard-case summary still describe an interrupted-frequency restart that is absent from the current bytes |

Case 02 is blocked from a scientific regression contract until the provenance
of `hexaaquairon.out` is resolved. The note assigns the wrong state energy
`-2093.022067822851` Ha to that filename, while the current file contains
`-2093.070628221697` Ha. The former energy appears in
`hexaaquairon_frag.out`, and the swap output contains
`-2093.070929682629` Ha.

Case 04 also contains a documentation mismatch. The standalone
`small_basis.out` run has its own +17.2 Ha iteration-10 excursion with DIIS
error 133. The controlled route is the projected small-basis orbitals plus
damping in `solution.out`, whose SCF path has no excursion above 5 Ha.

Case 01 exposed a parser defect rather than a failed-state diagnostic. NWChem
alpha and beta canonical orbital numbers are not stable cross-spin identities,
so pairing occupied orbitals by approximate character left a ligand orbital as
the apparent SOMO. The bounded unrestricted rule now labels the highest-energy
majority-spin occupied excess. This yields five SOMOs for the multiplicity-six
run and one Fe-centered t2g SOMO for the multiplicity-two run. A single
successful output still cannot establish that its state is chemically desired;
that conclusion requires the comparison and the expected-state context.

## Pinned text artifacts

These hashes identify the bytes inspected during this audit. A future external
manifest should reject a case when either size or SHA-256 changes.

| Relative path under `input_examples` | Bytes | SHA-256 |
| --- | ---: | --- |
| `nwchem/hard_cases/01_fecn6_lowspin_fragment/NOTES.md` | 4,282 | `9fbbaf870b7dc0b3239f7f7f0af8e86527bc7081854c2ada615ac5ed637a0129` |
| `nwchem/hard_cases/01_fecn6_lowspin_fragment/failed.nw` | 1,161 | `b50ffa17b70c4244ab92c55a2b39b422fb9658e0261f50e40a9261935cc00ac9` |
| `nwchem/hard_cases/01_fecn6_lowspin_fragment/failed.out` | 315,253 | `bd21f6ff2d5daf9df5d448c77c11481e8b169132d669214ef474856f9e54a1cc` |
| `nwchem/hard_cases/01_fecn6_lowspin_fragment/solution.nw` | 2,565 | `e46ee2ed07d5f825f21b9b62d248a70f6480a427c5c3db70de18c7767772eb8c` |
| `nwchem/hard_cases/01_fecn6_lowspin_fragment/solution.out` | 639,830 | `96ec817c6d489e2a1129b6baaa0282212c5a145652501011dd5f019e8a1d4a23` |
| `nwchem/hard_cases/02_hexaaquairon_swap_chain/PROBLEM_AND_SOLUTION.md` | 8,490 | `580ffc61520749e1d33d2fb9e2ea01f6ce5fa7823a57da6bac974b64517ca45e` |
| `nwchem/hard_cases/02_hexaaquairon_swap_chain/hexaaquairon.nw` | 1,367 | `73a01ebe578c6552a504b5fc338471dbf89641d84dd392693f288bf92b2a36e0` |
| `nwchem/hard_cases/02_hexaaquairon_swap_chain/hexaaquairon.out` | 545,209 | `1482356797af04f969e0a063811fe6a661e105ce908e0e3f6bf2d94023d083b5` |
| `nwchem/hard_cases/02_hexaaquairon_swap_chain/hexaaquairon_frag.nw` | 3,848 | `5ce833406c29bec559430c200c4c233e8ccea764915b5b5b660b09d1a9ab9272` |
| `nwchem/hard_cases/02_hexaaquairon_swap_chain/hexaaquairon_frag.out` | 1,085,969 | `f738643ea6880762ee5a1cf00363b5ef9227b6e594a867b5aa92f26a6c51c6fc` |
| `nwchem/hard_cases/02_hexaaquairon_swap_chain/hexaaquairon_swap.nw` | 1,672 | `789d16787790f03022a71cf5ffc95fc1100ab85aa1c480367d9b7097d85a9c44` |
| `nwchem/hard_cases/02_hexaaquairon_swap_chain/hexaaquairon_swap.out` | 545,023 | `0f9d0e282b4ed1b9b76a3d11090e61848be7e83a02686b67f961b696f6aed56f` |
| `nwchem/hard_cases/03_feo_scf_convergence/NOTES.md` | 3,455 | `6b46f61af5673269f2463d04b5a71528ea4040a1f32c4a42db0392f40e1ef1e0` |
| `nwchem/hard_cases/03_feo_scf_convergence/failed.nw` | 830 | `9c57bb8f28b01d4f982c1289613d08c3808b730ab0c687f5fe08b036c0e2d2da` |
| `nwchem/hard_cases/03_feo_scf_convergence/failed.out` | 120,618 | `2f45b8efe571726a5d068dcd594c13826b0f9fc84d52d95a92b8de499fd078a6` |
| `nwchem/hard_cases/03_feo_scf_convergence/solution.nw` | 839 | `fe0df25f5f3235d4a74de8f5910bedb9bdd9c1b983d315dcf4c803658a2f9c77` |
| `nwchem/hard_cases/03_feo_scf_convergence/solution.out` | 117,308 | `d39c3dc57e95baf688d0360d889c49fd164c3b6cf037ffc2ae90d34ec31a260e` |
| `nwchem/hard_cases/04_ferrocene_basis_stepping/NOTES.md` | 6,010 | `708065328b25f0cd4e4bc747920108da1b6656ecdc07c27ba1d395392bacdea2` |
| `nwchem/hard_cases/04_ferrocene_basis_stepping/failed.nw` | 1,664 | `65241ac3a22b1c390f7951832fa704c98e22d5056c8b41ed7842a2e48083e2e3` |
| `nwchem/hard_cases/04_ferrocene_basis_stepping/failed.out` | 356,218 | `e5387a8cd71783913f663088c6568563d222db2ef328e6e38b6c4cfc60b16b87` |
| `nwchem/hard_cases/04_ferrocene_basis_stepping/small_basis.nw` | 1,395 | `56a3fb862058df887ba9ba4106747d051b52c243a093ba446db43a046a30a410` |
| `nwchem/hard_cases/04_ferrocene_basis_stepping/small_basis.out` | 220,631 | `0ec7fc3938aa6f1362650fa51b00cc1e27dcc04398ba6e7b2f835ea4536af6f0` |
| `nwchem/hard_cases/04_ferrocene_basis_stepping/solution.nw` | 2,160 | `9b7a6a7549060fa9fde58cf7bc19aaf0bd4cfe036e91f238333d5644b98ac31b` |
| `nwchem/hard_cases/04_ferrocene_basis_stepping/solution.out` | 358,127 | `a3c38a4f1aaf09ac6b6986f4502e247ed6c60c1e2de32a2365e90b1237b36af7` |
| `nwchem/hard_cases/05_crco6_freq_restart/NOTES.md` | 9,030 | `a695339762c9464aba1c9c1f11c7f08aea5b41152dd7f3f529b7567f77c5e8a3` |
| `nwchem/hard_cases/05_crco6_freq_restart/failed.nw` | 2,068 | `214b157c2119ddb42cb9d96213e07964d08c4156937f9d7b68a40f7aba600295` |
| `nwchem/hard_cases/05_crco6_freq_restart/failed.out` | 2,355,250 | `e239bac494eccee7baaad6aca3094dba3ca38f54d8756e6a54064753bde78c2a` |
| `nwchem/hard_cases/05_crco6_freq_restart/solution.nw` | 3,277 | `6e74744eb819bb54c6a039e8724f6e5dfef19a2159ffb480fd49266f3966d82a` |
| `nwchem/hard_cases/05_crco6_freq_restart/solution.out` | 5,575,764 | `dd0024c274ea485f9a34633af475ce421726e19bbfc527de33d0703ea7a4489e` |

## Behavior-lock work

- [ ] Resolve case 02's filename and provenance mismatch before asserting a
      positive or negative scientific verdict.
- [ ] Confirm redistribution terms before copying any reduced fixture into Git.
- [x] Add an opt-in external-corpus check that pins sizes and hashes before
      calling application services.
- [x] Pin case 01 failed and solution verdicts, including the energy difference
      and expected SOMO interpretation.
- [x] Add guided run comparison for case 03 rather than teaching
      `inspect_run` to infer a ground state from one calculation.
- [x] Preserve the unstable SCF trajectory evidence in case 04 even though the
      final task succeeds.
- [x] Pin optional stability hardening for case 04 without claiming that the
      unstable small-basis seed is intrinsically reliable.
- [x] Pin the imaginary-mode count, leading projected frequency, energies, and
      verdict transition for case 05.
- [ ] Decide whether restart artifacts from cases 02 and 04 remain external or
      receive small committed metadata fixtures.

The current discrepancies are regression targets, not expected values to bless.
