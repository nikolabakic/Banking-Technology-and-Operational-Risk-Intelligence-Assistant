import argparse
import json
from pathlib import Path
from typing import Any

SOURCE_PATH = Path("data/evaluation/retrieval_queries_dev.jsonl")
RECORDS_PATH = Path("data/processed/sec2md_structure_v3/embedding_records.jsonl")
OUTPUT_PATH = Path("data/evaluation/retrieval_queries_sec2md_v3.jsonl")

QRELS: dict[str, list[str]] = {
    "dev_ally_operational_risk_definition_2025": [
        "3036bb6433b0efa7c29c84a13ff4236f19d89c426d2721857a74028a4586caac",
        "69116c2bcdf9d2a1d1d0d36e4ebcb070a744019c758fc473937c47ca7f010125",
    ],
    "dev_bac_cyber_incident_impacts_2025": [
        "83e9f6b14434466b4b1e9e245e97a0e07be0a067d362d0472100239aa413e9ca"
    ],
    "dev_c_operational_risk_definition_2025": [
        "ae457ccfdfbc35bdd0c9e6736b99d6df334c4a614ae6cb8554107f5d2505cd27"
    ],
    "dev_gs_cybersecurity_risk_definition_2025": [
        "e2efdbe0464ab8e9bd6bc4ef7946c8cc0ebca56d1f07298d47c64712b35c9d4d"
    ],
    "dev_jpm_cybersecurity_risk_definition_2025": [
        "73aeaf45c332d109c296862441dd6ad50b89f492be5560b20f8615b1f3c2ed0e"
    ],
    "dev_pnc_operational_risk_definition_2025": [
        "9ab11d2a15af0c90b9a949dd5389695b90510239d5fd55dfd5638e14af08dedc"
    ],
    "dev_tfc_cyber_incident_response_team_2025": [
        "00f6a912ba9b6e655f4d26e89ca25a979d71d60538c30b80b2a2ba42af5bc5e8"
    ],
    "dev_jpm_standardized_cet1_ratio_2025": [
        "48d7295b548146e5905223d4deacab93b937cc6d9d8ae938be280a662c46642e",
        "54996a7b217d42d439a1bf539de97f7416a6f5281bec44be70b0aea2fbd51c63",
    ],
    "dev_bac_standardized_cet1_ratio_2025": [
        "749ec1c9e8c1d512071920d5f7e7897fc1fb982f30064c1141982ae4b77a133b",
        "f7cd1dd4628c3a420a42aed9833cede84143440305f2ffd02fedf42e7e5010ca",
        "df3fc57e8301b1c994b4d55435e5f70d9763e5168b517fd0b4f036f79e20bd83",
        "c2d27aacd6cd57ef16b5043b630394a7412f33b8fb5324cae54c23cfb7ac04b1",
    ],
    "dev_c_standardized_cet1_ratio_2025": [
        "4a02524f874dd53e7c29e0c55482fcf48cc973e788f6be066807d0942e4158ac"
    ],
    "dev_gs_standardized_cet1_ratio_2025": [
        "d1e0d3640a364411e7f788d3cac85e927bfa37de03468abf3e70640b9f2350d0",
        "c57d80bd601e35408363d1a91ff6ebd684642c8cd3f3b1ac13e3890fd1b8d1b0",
    ],
    "dev_pnc_total_deposits_2025": [
        "67017594d6ce03719d95c093a8d015cdbe3c091e9a32befae24bea34fd739218",
        "bd59d32fce8d6c32ceb9dc93ed121b9d986ac5bc5581c33f184c1d07b7436451",
        "a9f14c05468cd9e0a27e70cac14f8055bca96d338f60696e706dc152f3899b67",
        "600b419a038db04d118c3fcfbbcedd59feb398e0037537a8d5920f65ac12cb92",
    ],
    "dev_tfc_cet1_ratio_corporation_2025": [
        "9784115b7c8289aafc65822053dcb38bc3fd6c3f4a1d226fea60b05d9930d350",
        "4f2a640c749f50b29dea0da480c1a87d86bd9e71ef40bb61113be0fdb20427b8",
    ],
    "dev_ally_net_income_continuing_operations_2025": [
        "a8f85459d7a08d83463d07b6692b72d0614ca3b2b3acd024b836e4b619198ef4",
        "0669668907d4cd68a04d07bfa8f74518544e82139baf0025e4bc694c15e13185",
        "5798fb8164ce83bcbf727b480aa67b2e1d93c4ca94e1a458d74afb796182abc1",
        "a252bcd61381b79e6b16a35a5d4db0eb4113a619a8db9f578fdc714f72805f67",
    ],
    "dev_jpm_standardized_cet1_requirement_2025": [
        "8b6b7ebb5838c34e5778b2a9c3596a3d75d7285cbfe8786eb3e2e90e8e5929ee",
        "bd9ed602dc6d0e19ff8d628674e743223cb4afb060cfa6686e1bc48d83034410",
        "48d7295b548146e5905223d4deacab93b937cc6d9d8ae938be280a662c46642e",
    ],
    "dev_bac_advanced_cet1_ratio_corporation_2025": [
        "df3fc57e8301b1c994b4d55435e5f70d9763e5168b517fd0b4f036f79e20bd83",
        "c2d27aacd6cd57ef16b5043b630394a7412f33b8fb5324cae54c23cfb7ac04b1",
    ],
    "dev_c_advanced_cet1_ratio_citigroup_2025": [
        "4a02524f874dd53e7c29e0c55482fcf48cc973e788f6be066807d0942e4158ac"
    ],
    "dev_tfc_cet1_ratio_bank_2025": [
        "80a954092b065daaaa7431113e158f720be9811ab2940d17dc0f0b7bb09e1f88"
    ],
    "dev_bac_bana_expansion_2025": [
        "557a145dd22382c0caf4cdc871e11eae4539afdf6d083ac6f079a084beead9fd"
    ],
    "dev_pnc_gsib_expansion_2025": [
        "4f6df683d68566080795ed8a58938a65086c1374afbd1a413f9f662c9301dd2b"
    ],
    "dev_tfc_brc_expansion_2025": [
        "1c7c017106267aad925b8aa3aaaaf663377b2f1bb46f3fb827be6ef0aa523da8"
    ],
    "dev_lob_bank_cet1_ratio_2024_split_table": [
        "f4f15ac624f1ebed3d6291e8887288eae73130f45e5c776e9e317fea8cb62f2f"
    ],
    "dev_ally_comprehensive_income_2025_split_table": [
        "366d8d278bc03a6924a0882f96b16605e04d2d9a457d93ed06ebd779f3acccbc"
    ],
    "dev_jpm_bank_advanced_cet1_ratio_2025_metadata": [
        "54996a7b217d42d439a1bf539de97f7416a6f5281bec44be70b0aea2fbd51c63"
    ],
    "dev_bac_bana_standardized_cet1_2025_metadata": [
        "c2d27aacd6cd57ef16b5043b630394a7412f33b8fb5324cae54c23cfb7ac04b1"
    ],
}

CROSS_BANK_GROUPS: dict[str, list[tuple[str, str]]] = {
    "dev_cross_bac_c_standardized_cet1_2025": [
        ("Bank of America Corporation", "dev_bac_standardized_cet1_ratio_2025"),
        ("Citigroup Inc.", "dev_c_standardized_cet1_ratio_2025"),
    ],
    "dev_cross_c_jpm_operational_risk_definitions_2025": [
        ("Citigroup Inc.", "dev_c_operational_risk_definition_2025"),
        (
            "JPMorgan Chase & Co.",
            "dev_jpm_operational_risk_definition_for_cross_bank",
        ),
    ],
    "dev_cross_pnc_tfc_cet1_2025": [
        ("The PNC Financial Services Group, Inc.", "dev_pnc_cet1_for_cross_bank"),
        ("Truist Financial Corporation", "dev_tfc_cet1_ratio_corporation_2025"),
    ],
}

QRELS["dev_jpm_operational_risk_definition_for_cross_bank"] = [
    "27967d3a66bde274872224dfbfe23d2e134b6763d3c9a14b7175c89e8a616fb9"
]
QRELS["dev_pnc_cet1_for_cross_bank"] = [
    "d5fab3a12f4de961e1c11f0ae5f234bea71211f91358140185fb2ad76303e08c"
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sec2md v3 retrieval qrels.")
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--records", type=Path, default=RECORDS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Use --overwrite.")

    queries = load_jsonl(args.source)
    records = load_jsonl(args.records)
    records_by_target_id = {str(record["target_chunk_id"]): record for record in records}
    parent_link_count = 0

    for query in queries:
        query_id = str(query["query_id"])

        if query_id in CROSS_BANK_GROUPS:
            groups = [
                {"entity": entity, "target_chunk_ids": QRELS[source_query_id]}
                for entity, source_query_id in CROSS_BANK_GROUPS[query_id]
            ]
            relevant_ids = [
                target_id for group in groups for target_id in group["target_chunk_ids"]
            ]
            query["required_evidence_groups"] = groups
        elif query["status"] == "answerable":
            relevant_ids = QRELS[query_id]
            query.pop("required_evidence_groups", None)
        else:
            relevant_ids = []
            query.pop("required_evidence_groups", None)

        missing_ids = set(relevant_ids) - records_by_target_id.keys()

        if missing_ids:
            raise ValueError(f"Missing qrels for {query_id}: {sorted(missing_ids)}")

        for target_id in relevant_ids:
            record = records_by_target_id[target_id]
            metadata = record.get("metadata", {})

            if record["record_type"] == "table_locator":
                has_coordinates = bool(metadata.get("cell_coordinates"))
                is_schema_locator = metadata.get("locator_scope") == "table_schema"

                if not metadata.get("parent_id") or not (
                    has_coordinates or is_schema_locator
                ):
                    raise ValueError(f"Incomplete table evidence reference: {target_id}")

                parent_link_count += 1

        query["relevant_target_chunk_ids"] = relevant_ids
        query["primary_target_chunk_id"] = relevant_ids[0] if relevant_ids else None
        query["annotation_notes"] = (
            "sec2md v3 qrels were manually verified against direct narrative evidence or "
            "locator coordinates and their original parent table. "
            + str(query.get("annotation_notes") or "")
        ).strip()

    answerable_ids = {
        str(query["query_id"]) for query in queries if query["status"] == "answerable"
    }
    mapped_ids = set(QRELS) - {
        "dev_jpm_operational_risk_definition_for_cross_bank",
        "dev_pnc_cet1_for_cross_bank",
    }
    mapped_ids.update(CROSS_BANK_GROUPS)

    if answerable_ids != mapped_ids:
        raise ValueError(
            f"Answerable query mapping mismatch: missing={answerable_ids - mapped_ids}, "
            f"extra={mapped_ids - answerable_ids}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        for query in queries:
            file.write(json.dumps(query, ensure_ascii=False) + "\n")

    print(f"Queries: {len(queries)}")
    print(f"Answerable: {len(answerable_ids)}")
    print(f"Validated table-parent evidence links: {parent_link_count}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
