import pytest

from bankscope.chat import CitationSourceResolver, StaleCitationError


def text_record(target: str, document: str, accession: str = "filing-1") -> dict:
    return {
        "record_id": f"record-{target}",
        "target_chunk_id": target,
        "record_type": "text",
        "document": document,
        "metadata": {
            "ticker": "JPM",
            "accession_number": accession,
            "source_url": "https://example.com/filing",
        },
    }


def test_source_context_returns_only_same_filing_neighbors() -> None:
    resolver = CitationSourceResolver(
        [
            text_record("previous", "Previous"),
            text_record("anchor", "Anchor"),
            text_record("other", "Other filing", accession="filing-2"),
        ],
        [],
        corpus_hash="hash-1",
    )
    context = resolver.context("anchor", expected_corpus_hash="hash-1", radius=2)
    assert [chunk["role"] for chunk in context["chunks"]] == ["previous", "anchor"]
    assert context["source_url"] == "https://example.com/filing"


def test_source_context_hydrates_complete_table() -> None:
    chunk = {
        "record_id": "record-table",
        "target_chunk_id": "table-1",
        "record_type": "table",
        "document": "Retrieval description",
        "metadata": {"ticker": "JPM", "accession_number": "filing-1"},
    }
    table = {"table_id": "table-1", "document": "| Complete | Table |"}
    resolver = CitationSourceResolver([chunk], [table], corpus_hash="hash-1")
    context = resolver.context("table-1", expected_corpus_hash="hash-1")
    assert context["chunks"][0]["document"] == "| Complete | Table |"


def test_source_context_fails_closed_for_stale_corpus() -> None:
    resolver = CitationSourceResolver([text_record("anchor", "Anchor")], [], corpus_hash="current")
    with pytest.raises(StaleCitationError):
        resolver.context("anchor", expected_corpus_hash="historical")
