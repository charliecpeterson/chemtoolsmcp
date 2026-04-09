from __future__ import annotations

import unittest
from pathlib import Path

from chemtools.nwchem_docs import (
    DEFAULT_DOCS_ROOT,
    find_examples,
    get_topic_guide,
    list_docs,
    lookup_block_syntax,
    read_doc_excerpt,
    search_docs,
)


class NWChemDocsTests(unittest.TestCase):
    def test_list_docs_returns_files(self) -> None:
        payload = list_docs()
        self.assertTrue(payload)
        self.assertTrue(any(item["name"].endswith(".txt") for item in payload))

    def test_search_docs_finds_mcscf_state(self) -> None:
        payload = search_docs("mcscf state multiplicity", max_results=4)
        self.assertGreater(payload["result_count"], 0)
        self.assertTrue(
            any("QuantumMethods" in result["file_name"] or "QuantumMechanicalMethods" in result["file_name"] for result in payload["results"])
        )

    def test_lookup_block_syntax_prefers_mcscf_docs(self) -> None:
        payload = lookup_block_syntax("mcscf", max_results=3)
        self.assertGreater(payload["result_count"], 0)
        self.assertEqual(payload["block_name"], "mcscf")
        self.assertTrue(any("mcscf" in result["line_text"].casefold() for result in payload["results"]))

    def test_find_examples_can_find_fragment(self) -> None:
        payload = find_examples("fragment guess", max_results=4)
        self.assertGreater(payload["result_count"], 0)
        self.assertTrue(any("fragment" in result["line_text"].casefold() for result in payload["results"]))

    def test_read_doc_excerpt_by_query(self) -> None:
        file_path = Path(DEFAULT_DOCS_ROOT) / "11_QuantumMethods-2.pdf.txt"
        payload = read_doc_excerpt(str(file_path), query="STATE", context_lines=2)
        self.assertEqual(payload["file_name"], "11_QuantumMethods-2.pdf.txt")
        self.assertIsNotNone(payload["matched_line"])
        self.assertTrue(any(item["matched"] for item in payload["excerpt"]))

    def test_get_topic_guide_mcscf(self) -> None:
        payload = get_topic_guide("mcscf")
        self.assertEqual(payload["topic"], "mcscf")
        self.assertTrue(payload["results"])
        self.assertTrue(any("MCSCF" in result["line_text"] or "STATE" in result["line_text"] for result in payload["results"]))

    def test_get_topic_guide_fragment_guess(self) -> None:
        payload = get_topic_guide("fragment_guess")
        self.assertEqual(payload["topic"], "fragment_guess")
        self.assertTrue(payload["results"])
        self.assertTrue(any("fragment" in result["line_text"].casefold() for result in payload["results"]))


if __name__ == "__main__":
    unittest.main()
