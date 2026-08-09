"""MCP contracts for the independently imported NWChem docs family."""

from __future__ import annotations

from chemtools.mcp.dispatch import dispatch_tool


def test_list_nwchem_docs_returns_the_complete_inventory():
    payload = dispatch_tool("list_nwchem_docs", {})

    assert len(payload["files"]) == 29
    assert payload["files"][0]["name"] == "01_Intro.pdf.txt"
    assert payload["files"][-1]["name"] == "30-Containers.pdf.txt"


def test_read_nwchem_doc_excerpt_preserves_line_coordinates():
    payload = dispatch_tool(
        "read_nwchem_doc_excerpt",
        {
            "doc_name": "01_Intro.pdf.txt",
            "start_line": 1,
            "end_line": 3,
        },
    )

    assert payload["file_name"] == "01_Intro.pdf.txt"
    assert payload["start_line"] == 1
    assert payload["end_line"] == 3
    assert [line["line_number"] for line in payload["excerpt"]] == [1, 2, 3]
