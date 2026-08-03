# Output detector collision audit, 2026-07-31

Chemtools now rejects conflicting program overrides, uses output-shaped
detector signatures, and routes GRASP RCI and property files through their
existing parsers. Authoritative dispatch also distinguishes detector crashes
and source-read errors from ordinary no-match results.

## Scope

The audit covered the NWChem, OpenMolcas, DIRAC, and GRASP detectors; registry
resolution; generic and legacy dispatch; recovery dispatch; and output-like
files under `/home/charlie/input_examples`.

The bounded corpus pass read at most 32 KiB from 408 files with `.out`, `.log`,
`.sum`, `.nwo`, or `.alog` suffixes. Three separate code passes checked false
positives and collisions, false negatives, and dispatch behavior. Every
reported trigger was reproduced before changing code.

Previously reviewed behavior was excluded: standalone NBO rejection, sparse
or truncated output with an explicit program, the short `OpenMolcas` header,
and automatic ambiguity rejection.

## Findings

### 1. High, fixed: recovery could patch an input from another program

`apply_recovery` selected the Molcas patcher from the output or explicit
program without checking the input. A NWChem input paired with
`program="molcas"` and `memory_exceeded` received a Molcas `MOLCAS_MEM` line.

The generic dispatcher now compares the detected input and selected program
before dispatch. The Molcas recovery function also refuses any input without a
recognized Molcas module block. No recovered input file is written on mismatch.

Code: `chemtools/mcp/tools/generic.py`,
`chemtools/programs/molcas/strategy/recovery.py`.

### 2. High, fixed: explicit generic dispatch trusted conflicting content

The generic parsers accepted an explicit program without checking positive
detector evidence. For example, parsing an NWChem output as Molcas returned a
Molcas result with no tasks; the inverse returned a fabricated NWChem task.

`registry.resolve()` now raises `ProgramContentMismatch` when detected content
excludes the selected program. Generic, guided, and legacy dispatch translate
that exception to `program_content_mismatch`. Explicit selection still works
for detector-negative fragments and resolves ambiguous content when the
selected program is one of the candidates.

Code: `chemtools/core/registry.py`, `chemtools/mcp/tools/generic.py`,
`chemtools/mcp/tools/guided.py`, `chemtools/mcp/tools/_nwchem_base.py`.

### 3. High, fixed: GRASP property files were parseable but not reachable

The dedicated GRASP parsers already handled hyperfine, isotope-shift, and
LSJ radiative-transition files, but automatic detection returned no program
and the shared parser adapter had no routes for them. Confirmed examples were
`Li2p.h`, `Li2p.hlsj`, `6d2.i`, and `Li2s.Li2p.t.lsj`.

The detector now requires paired format markers for each family. The shared
adapter returns canonical property tasks and compact evidence, including the
number of parsed records and the maximum transition gauge disagreement.

RCI `.csum` files had a related error: they were parsed as RMCDHF summaries,
labeled `MCDHF`, and lost the parsed Breit/QED correction flags. They now use
the distinct `grasp.rci_summary` artifact kind, an `RCI` task, and
`grasp:rci_corrections` evidence. Plain `.sum` files now classify once as
`grasp.rmcdhf_summary` instead of matching a second generic GRASP output kind.

Code: `chemtools/programs/grasp/__init__.py`,
`chemtools/programs/grasp/_plugin_parser.py`.

### 4. High, fixed: loose program-name substrings caused wrong dispatch

The NWChem detector accepted any occurrence of `NWChem`. An unsupported output
with a title such as `compare against NWChem` was therefore labeled NWChem.
Loose OpenMolcas, GRASP-format, and DIRAC release phrases could also turn an
echoed NWChem title into a second candidate.

The NWChem detector now accepts its full product banner or a line-shaped
version banner. OpenMolcas and DIRAC fallback signatures are line anchored,
and GRASP no longer treats the prose phrase `GRASP format` as a product
signature.

Code: `chemtools/programs/nwchem/__init__.py`,
`chemtools/programs/molcas/__init__.py`,
`chemtools/programs/dirac/parse/output.py`,
`chemtools/programs/grasp/__init__.py`.

### 5. Medium, fixed: detector exceptions were indistinguishable from misses

`detect_candidates_from_text()` caught every detector exception and continued;
`detect_candidates_from_file()` converted every file-read error to an empty
candidate set. A broken detector could therefore disappear while a weaker
false positive won, and an unreadable file looked like unsupported content.

The registry now returns a `ProgramDetectionProbe` containing successful
candidates, detector failures, and an optional source failure. `resolve()`
uses the full probe and raises `ProgramDetectorError` or
`ProgramDetectionSourceError`. Automatic resolution fails closed if any
detector crashes. Explicit resolution reports a crash in the selected
detector but does not let an unrelated broken plugin block a healthy explicit
candidate.

Generic, guided, legacy, and recovery dispatch return structured
`program_detector_error` or `program_source_error` payloads. The low-level
`detect_from_*` and `detect_candidates_from_*` helpers retain their previous
lossy behavior for compatibility.

Code: `chemtools/core/registry.py`, `chemtools/mcp/tools/generic.py`,
`chemtools/mcp/tools/guided.py`, `chemtools/mcp/tools/_nwchem_base.py`.

## Corpus result after the fixes

The 408-file pass produced no multiple matches:

| Candidate set | Files |
|---|---:|
| NWChem only | 147 |
| OpenMolcas only | 24 |
| DIRAC only | 18 |
| GRASP only | 23 |
| No match | 196 |

Within the four supported top-level directories, all 24 Molcas and 18 DIRAC
outputs matched. NWChem had 142 direct matches and four known standalone NBO
files. GRASP had 23 primary output matches and 43 misses: 36 saved stdin/input
logs, six binary wavefunction files, and one HF helper log for which the shared
GRASP parser has no route.

Five files stored under the NBO7 examples matched NWChem. Inspection confirmed
that they are parent NWChem run outputs copied into those directories, so they
are true positives.

The specialized GRASP check separately confirmed automatic detection and the
expected adapter route for two HFS files, one isotope-shift file, one
transition file, and one RCI summary.

A post-fix lossless probe of all 408 corpus files found no detector or
source-read failures.

## Tests

Focused detector, artifact, dispatch, recovery, parser, and guided-inspection
tests cover every fixed trigger. The full suite passes: 374 tests in 23.68
seconds.
