# Qwen3.5-0.8B MTP Q4_K_M Assets

Every greppy build expects these Git-LFS assets in this directory:

- `Qwen3.5-0.8B-MTP-Q4_K_M.gguf`
- `Qwen3.5-0.8B-MTP-Q4_K_M.gguf.sha256`
- `tokenizer.json`
- `tokenizer.json.sha256`

The GGUF is Greppy's function-purpose finetune of the pinned
`Qwen/Qwen3.5-0.8B` base model (FP32 from-base checkpoint of 2026-07-12,
shipped via release asset tag `model-assets-v2`). Greppy changed the model
through full-parameter supervised finetuning and trained an MTP draft layer.
The merged checkpoint was converted and quantized to Q4_K_M with llama.cpp;
the checked-in GGUF contains both target and MTP weights.

Release readiness for this checkpoint is recorded in
`licenses/MODEL-REDISTRIBUTION.lock.json` and the repository-level
provenance records.

The tokenizer JSON is from the pinned `Qwen/Qwen3.5-0.8B` revision and is
unchanged by Greppy.

Both sources identify Qwen3.5 as Apache-2.0. The complete license shipped with
Greppy is `licenses/QWEN3.5-APACHE-2.0.txt`. Exact base, data, training, export,
quantization, and modification records are in the repository-level
`licenses/QWEN3.5-*.json` and `licenses/QWEN3.5-MODIFICATIONS.txt` files.

Verified asset digests:

- GGUF: `d09d5028e28ea9df501d83a9a60b80ed73f878b4b98424b09a39505364a1053f`
- tokenizer: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
