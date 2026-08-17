# BankScope Colab GPU workflows

`BankScope_GPU_Evaluation_Colab.ipynb` is the supported CUDA path for rebuilding the complete
BankScope embedding archive and running the baseline retrieval evaluation. All acquisition,
parsing, Qdrant, generation, comparison, and application checks run locally on CPU/API.

1. Build `data/processed/chunks.jsonl` locally.
2. Run `python scripts/build_colab_bundle.py --overwrite` locally.
3. Open the notebook in Google Colab and select a GPU runtime (T4 or better).
4. Run every cell and upload `artifacts/bankscope_colab_gpu_bundle.zip` when prompted.
5. Download `bankscope_gpu_results.zip` and copy its `embeddings.npz` to
   `data/processed/embeddings.npz`.
6. Run `python scripts/build_qdrant.py --recreate`; this validates record order, source hash,
   model identity, revision, dimension, dtype, and vector normalization before indexing.

The notebook intentionally fails when CUDA is unavailable, verifies every bundled input by
SHA-256, pins the same model revision as `scripts/embed.py`, runs a smoke batch before the full
6,550-record corpus, and returns embeddings plus BM25, dense, and hybrid evaluation results.

The notebook does not require OpenAI credentials. Do not include secrets, local chat state, or Qdrant
storage in uploads. Generated archives and notebook outputs remain ignored local artifacts.

[Repository guide](../README.md) · [Evaluation data](../data/evaluation/README.md)
