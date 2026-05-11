"""Per-program plugins.

Each subpackage `chemtools.programs.<name>` exports a Program plugin instance
and registers it with `chemtools.core.registry` on import. CLI entry points
(`chemtools-<name>`) import only the program(s) they expose, so a session
loaded for one program does not pull in the others.
"""
