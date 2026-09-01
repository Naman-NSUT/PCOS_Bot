"""
tests/test_grade_node.py
Regression tests for the CRAG grading node's payload handling.

The grader model controls both the length and the element types of the JSON it
returns, and none of it is trustworthy. The invariant these tests protect:
grade_node ALWAYS returns exactly one graded entry per retrieved chunk.
Violating it corrupts CRAGState — duplicate chunks reach the generator and the
inflated AMBIGUOUS count forces a needless (paid) second retrieval pass.

Zero LLM API calls — the client is mocked.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.nodes import grade as grade_mod
from src.nodes.grade import grade_node, _strip_code_fence, _extract_grades


def _chunks(n):
    return [
        {"page_content": f"chunk {i} about PCOS", "metadata": {"source": "g.pdf", "page": i}}
        for i in range(n)
    ]


def _state(n):
    return {"query": "pcos", "active_query": "pcos", "chunks": _chunks(n)}


def _run(payload, n):
    """Run grade_node with the client mocked to return `payload`."""
    client = MagicMock()
    resp = MagicMock()
    resp.content = payload
    client.invoke.return_value = resp
    with patch.object(grade_mod, "_get_client", return_value=client):
        return grade_node(_state(n))


class TestGradeCountInvariant:
    """graded_chunks must be 1:1 with chunks, whatever the model returns."""

    def test_happy_path(self):
        out = _run('{"grades": [{"grade": "RELEVANT", "reason": "r0"},'
                   ' {"grade": "IRRELEVANT", "reason": "r1"}]}', 2)
        assert len(out["graded_chunks"]) == 2
        assert [g["grade"] for g in out["graded_chunks"]] == ["RELEVANT", "IRRELEVANT"]

    def test_too_few_grades_pads(self):
        out = _run('{"grades": [{"grade": "RELEVANT", "reason": "r0"}]}', 3)
        assert len(out["graded_chunks"]) == 3

    def test_too_many_grades_truncates(self):
        payload = ('{"grades": [' + ",".join(
            '{"grade": "RELEVANT", "reason": "r"}' for _ in range(6)) + ']}')
        out = _run(payload, 2)
        assert len(out["graded_chunks"]) == 2

    def test_bare_string_element_does_not_inflate(self):
        """
        A non-dict element used to raise mid-loop, and the except handler then
        appended a fallback for EVERY chunk on top of the partial list —
        yielding more graded entries than chunks.
        """
        out = _run('{"grades": [{"grade": "RELEVANT", "reason": "r0"}, "RELEVANT"]}', 2)
        assert len(out["graded_chunks"]) == 2

    def test_null_element_does_not_inflate(self):
        out = _run('{"grades": [{"grade": "RELEVANT", "reason": "r0"}, null, '
                   '{"grade": "RELEVANT", "reason": "r2"}]}', 3)
        assert len(out["graded_chunks"]) == 3

    def test_total_garbage_falls_back_one_per_chunk(self):
        out = _run("not json at all", 4)
        assert len(out["graded_chunks"]) == 4
        assert all(g["grade"] == "AMBIGUOUS" for g in out["graded_chunks"])

    def test_summary_totals_match_chunk_count(self):
        out = _run('{"grades": [{"grade": "RELEVANT", "reason": "r"}, "RELEVANT"]}', 2)
        assert sum(out["grade_summary"].values()) == 2

    def test_no_chunks_short_circuits(self):
        assert grade_node({"query": "q", "active_query": "q", "chunks": []}) == {
            "graded_chunks": [], "grade_summary": {}
        }

    def test_unknown_label_becomes_irrelevant(self):
        out = _run('{"grades": [{"grade": "SORT_OF", "reason": "r"}]}', 1)
        assert out["graded_chunks"][0]["grade"] == "IRRELEVANT"


class TestPayloadShapes:
    def test_wrapped_object(self):
        assert _extract_grades('{"grades": [{"grade": "RELEVANT"}]}') == [{"grade": "RELEVANT"}]

    def test_bare_array(self):
        assert _extract_grades('[{"grade": "RELEVANT"}]') == [{"grade": "RELEVANT"}]

    def test_renamed_wrapper_key(self):
        assert _extract_grades('{"results": [{"grade": "RELEVANT"}]}') == [{"grade": "RELEVANT"}]

    def test_single_ungrouped_object(self):
        assert _extract_grades('{"grade": "RELEVANT", "reason": "r"}') == [
            {"grade": "RELEVANT", "reason": "r"}
        ]


class TestStripCodeFence:
    def test_fenced_json(self):
        assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_bare_fence(self):
        assert _strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_unfenced_untouched(self):
        assert _strip_code_fence('{"a": 1}') == '{"a": 1}'

    def test_does_not_eat_leading_payload_chars(self):
        """
        str.lstrip("```json") would strip a leading run of ` j s o n —
        mangling any payload whose first characters are among them.
        """
        assert _strip_code_fence('json_value') == 'json_value'
        assert _strip_code_fence('{"n": 1}') == '{"n": 1}'


class TestIndexAlignment:
    """
    Grades are matched by their declared "index". Without this, a grader that
    skips one chunk shifts every later grade onto the wrong chunk — silently
    mislabelling content that then reaches the generator.
    """

    def test_skipped_middle_chunk_does_not_shift_later_grades(self):
        # Grader omits chunk 1 entirely; chunks 0 and 2 must keep their labels.
        out = _run('{"grades": ['
                   '{"index": 0, "grade": "RELEVANT", "reason": "a"},'
                   '{"index": 2, "grade": "IRRELEVANT", "reason": "c"}]}', 3)
        grades = [g["grade"] for g in out["graded_chunks"]]
        assert grades == ["RELEVANT", "AMBIGUOUS", "IRRELEVANT"]

    def test_out_of_order_indices_are_reordered(self):
        out = _run('{"grades": ['
                   '{"index": 2, "grade": "IRRELEVANT", "reason": "c"},'
                   '{"index": 0, "grade": "RELEVANT", "reason": "a"},'
                   '{"index": 1, "grade": "AMBIGUOUS", "reason": "b"}]}', 3)
        assert [g["grade"] for g in out["graded_chunks"]] == [
            "RELEVANT", "AMBIGUOUS", "IRRELEVANT"]
        assert [g["reason"] for g in out["graded_chunks"]] == ["a", "b", "c"]

    def test_positional_still_works_without_indices(self):
        out = _run('{"grades": ['
                   '{"grade": "RELEVANT", "reason": "a"},'
                   '{"grade": "IRRELEVANT", "reason": "b"}]}', 2)
        assert [g["grade"] for g in out["graded_chunks"]] == ["RELEVANT", "IRRELEVANT"]

    def test_out_of_range_index_is_ignored_not_crashed(self):
        out = _run('{"grades": [{"index": 99, "grade": "RELEVANT", "reason": "x"}]}', 2)
        assert len(out["graded_chunks"]) == 2

    def test_duplicate_index_does_not_inflate(self):
        out = _run('{"grades": ['
                   '{"index": 0, "grade": "RELEVANT", "reason": "a"},'
                   '{"index": 0, "grade": "IRRELEVANT", "reason": "dup"}]}', 2)
        assert len(out["graded_chunks"]) == 2
        assert out["graded_chunks"][0]["grade"] == "RELEVANT"
