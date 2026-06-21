"""NWChem MCP handlers — docs.

Split from mcp/tools/nwchem.py by category. Shared imports/helpers live in
_nwchem_base (pulled in below); nwchem.py imports this module so its @_tool
handlers register.
"""
from __future__ import annotations

from chemtools.mcp.tools._nwchem_base import *  # noqa: F401,F403
from chemtools.mcp.tools._nwchem_base import _tool, _build_next_actions  # noqa: F401


@_tool("evaluate_nwchem_case")
def _handle_evaluate_case(arguments: dict[str, Any]) -> dict[str, Any]:
    return evaluate_case(arguments["case_path"])


@_tool("evaluate_nwchem_cases")
def _handle_evaluate_cases(arguments: dict[str, Any]) -> dict[str, Any]:
    return evaluate_cases(arguments["path"])


@_tool("list_nwchem_docs")
def _handle_list_nwchem_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"files": docs_list_docs()}


@_tool("search_nwchem_docs")
def _handle_search_nwchem_docs(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_search_docs(
        arguments["query"],
        max_results=int(arguments.get("max_results", 8)),
        context_lines=int(arguments.get("context_lines", 2)),
    )


@_tool("lookup_nwchem_block_syntax")
def _handle_lookup_nwchem_block_syntax(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_lookup_block_syntax(
        arguments["block_name"],
        max_results=int(arguments.get("max_results", 6)),
    )


@_tool("find_nwchem_examples")
def _handle_find_nwchem_examples(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_find_examples(
        arguments["topic"],
        max_results=int(arguments.get("max_results", 6)),
    )


@_tool("read_nwchem_doc_excerpt")
def _handle_read_nwchem_doc_excerpt(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_read_doc_excerpt(
        arguments["doc_name"],
        start_line=arguments.get("start_line"),
        end_line=arguments.get("end_line"),
        query=arguments.get("query"),
        context_lines=int(arguments.get("context_lines", 8)),
    )


@_tool("get_nwchem_topic_guide")
def _handle_get_nwchem_topic_guide(arguments: dict[str, Any]) -> dict[str, Any]:
    return docs_get_topic_guide(arguments["topic"])


# ---------------------------------------------------------------------------
# Handlers — NWChem community forum search
# ---------------------------------------------------------------------------


@_tool("search_nwchem_forum")
def _handle_search_nwchem_forum(arguments: dict[str, Any]) -> dict[str, Any]:
    return forum_search(
        arguments["query"],
        max_results=int(arguments.get("max_results", 5)),
        fetch_content=arguments.get("fetch_content", True),
        subforums=arguments.get("subforums"),
    )
