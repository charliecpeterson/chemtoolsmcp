# MCP scientific ownership audit

MCP handlers should resolve a tool call, translate its arguments, and format
transport errors. Parsers, numerical cutoffs, chemistry verdicts, and task
selection belong in program packages or application services.

## Moved in this pass

| Previous MCP responsibility | Current owner | Compatibility status |
| --- | --- | --- |
| NWChem next-action decision table | `programs/nwchem/strategy/legacy_next_actions.py` | The old MCP import is an exact alias. |
| DIRAC input/checkpoint occupation comparison | `programs/dirac/strategy/open_shell.py` | MCP passes the input and HDF5 paths. |
| DIRAC text/HDF5 run summary and 1e-6 energy check | `programs/dirac/strategy/triage.py` | Response keys and verdicts are unchanged. |
| DIRAC occupied-spinor cutoff and energy-window filtering | `programs/dirac/parse/output.py` | MCP still reads the requested output and returns the public envelope. |
| QE-to-QMCPACK readiness precedence | `programs/qe/qmcpack.py` | All MCP conversion views call one public reducer. |
| Molcas orbital-task selection | `programs/molcas/parse/output.py` | The backend parser and MCP handler now share it. |
| Molcas RASSCF/CASPT2 task selection and orbital-swap setup | `programs/molcas/strategy/active_space.py` | MCP handlers only translate arguments. |
| Molcas geometry-block selection and RASSI module bounds | `programs/molcas/parse/{geometry,rassi}.py` | MCP retains legacy error wording; the backend parser shares geometry selection. |
| Cross-program geometry normalization and inspection | `application/run_inspection.py` | The MCP handler resolves the backend and forwards user limits. |
| Generic recovery input detection and source agreement | `application/recovery_planning.py` | MCP retains strict output resolution and low-level handler dispatch. |
| NWChem SCF/state recovery aggregation | `programs/nwchem/strategy/recovery.py` | MCP forwards the selected legacy mode. |
| NWChem multiplicity input inference and scan follow-up | `programs/nwchem/strategy/input_advisors.py` | MCP forwards explicit or file-based inputs. |
| NWChem SCF directive rendering | `programs/nwchem/input/general.py` | MCP no longer formats NWChem keywords. |

## Residual handler disposition

| Handler branch | Disposition |
| --- | --- |
| Generic recovery output resolution and low-level dispatch | Keep in MCP as compatibility translation. Input detection and source agreement now live in the application layer. |
| Molcas missing-block and index error envelopes | Keep in MCP while these low-level tools remain. Selection and parsing live in the Molcas package. |
| NWChem compact/detail selection and MCP payload truncation | Keep in MCP. These control response size, not chemistry. |
| QE-to-QMCPACK narrow inspection views | Keep until Phase 4 decides which low-level tools survive. Their checks and readiness order already live in the QE and QMCPACK packages. |
| Execution status, ownership, and cancellation branches | Keep in MCP or application execution adapters according to the existing execution boundary. |

## Accepted MCP behavior

The scan also found checks that belong at the transport boundary: required
argument combinations, list lengths defined by MCP schemas, execution
ownership, and compatibility error translation. Moving these would hide the
boundary rather than simplify it.

The scientific-ownership task is complete. Later Phase 4 removals may delete
some of these adapters, but no chemistry rule needs to move with them.
