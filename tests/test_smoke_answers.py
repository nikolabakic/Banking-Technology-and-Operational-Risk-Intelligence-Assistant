from scripts.smoke_answers import assess_smoke_answer, select_smoke_queries


def test_smoke_query_contract_covers_ten_unique_banks() -> None:
    queries = [
        {
            "query_id": query_id,
            "query": query_id,
            "ticker": f"B{index}",
            "status": "answerable",
        }
        for index, query_id in enumerate(
            (
                "dev_ally_operational_risk_definition_2025",
                "dev_bac_cyber_incident_impacts_2025",
                "dev_c_operational_risk_definition_2025",
                "dev_cof_cybersecurity_technology_risk_management_2025",
                "dev_gs_cybersecurity_risk_definition_2025",
                "dev_jpm_cybersecurity_risk_definition_2025",
                "dev_lob_bank_cet1_ratio_2024_split_table",
                "dev_pnc_operational_risk_definition_2025",
                "dev_stt_information_technology_risk_definition_2025",
                "dev_tfc_cyber_incident_response_team_2025",
            )
        )
    ]

    assert len(select_smoke_queries(queries)) == 10


def test_smoke_answer_requires_supported_owned_citations() -> None:
    query = {"ticker": "COF"}
    valid = {
        "ticker": "COF",
        "status": "supported",
        "answer": "Grounded answer [E1]",
        "citations": [{"ticker": "COF", "target_chunk_id": "target"}],
    }
    assert assess_smoke_answer(query, valid)["passed"] is True

    invalid = {**valid, "citations": [{"ticker": "STT", "target_chunk_id": "target"}]}
    assessment = assess_smoke_answer(query, invalid)
    assert assessment["passed"] is False
    assert assessment["checks"]["citation_ownership"] is False
