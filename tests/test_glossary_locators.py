from bankscope.retrieval.glossary_locators import (
    GLOSSARY_LOCATOR_VERSION,
    build_glossary_locators,
    is_glossary_table,
    validate_glossary_locators,
)


def parent_record(table_id: str = "table-1") -> dict:
    return {
        "record_id": f"structure_aware::table::{table_id}",
        "target_chunk_id": table_id,
        "record_type": "table",
        "embedding_text": (
            "Bank: BAC\nEntity: Bank of America Corporation\nReport: 2025 10-K\n\n"
            "Compact table description"
        ),
        "document": "Compact table description",
        "metadata": {"ticker": "BAC", "table_id": table_id},
    }


def acronym_table(title: str = "A CRONYMS") -> dict:
    return {
        "table_id": "table-1",
        "table_type": "data_table",
        "document": f"**{title}**\n\n| BANA | Bank of America, National Association |",
        "cell_matrices": [
            [
                ["BANA", "Bank of America, National Association"],
                ["CET1", "Common equity tier 1"],
            ]
        ],
    }


def test_acronym_title_variants_are_glossary_tables() -> None:
    assert is_glossary_table(acronym_table("Acronyms")) is True
    assert is_glossary_table(acronym_table("A CRONYMS")) is True
    assert is_glossary_table(acronym_table("Glossary")) is True


def test_glossary_locators_are_deterministic_and_point_to_parent_table() -> None:
    records = [parent_record()]
    tables = [acronym_table()]

    first = build_glossary_locators(records, tables)
    second = build_glossary_locators(records, tables)

    assert first == second
    assert len(first) == 2
    bana = first[0]
    assert bana["target_chunk_id"] == "table-1"
    assert bana["record_type"] == "table"
    assert "BANA stands for Bank of America, National Association" in bana["embedding_text"]
    assert bana["metadata"]["locator_version"] == GLOSSARY_LOCATOR_VERSION
    validate_glossary_locators(first, records, tables)


def test_numeric_table_pairs_do_not_create_glossary_locators() -> None:
    table = acronym_table("Capital ratios")
    table["document"] = "| CET1 | 11.4% |"
    table["cell_matrices"] = [[["CET1", "11.4%"]]]

    assert build_glossary_locators([parent_record()], [table]) == []
