# Qwen3.5-0.8B MTP Q4_K_M Assets

Every greppy build expects these Git-LFS assets in this directory:

- `Qwen3.5-0.8B-MTP-Q4_K_M.gguf`
- `Qwen3.5-0.8B-MTP-Q4_K_M.gguf.sha256`
- `tokenizer.json`
- `tokenizer.json.sha256`

The GGUF is the Q4_K_M file from `unsloth/Qwen3.5-0.8B-MTP-GGUF` and contains
the target model plus its MTP draft layer.
The tokenizer JSON is from `Qwen/Qwen3.5-0.8B`.

Both sources identify Qwen3.5 as Apache-2.0. The complete license shipped with
Greppy is `licenses/QWEN3.5-APACHE-2.0.txt`.

Verified asset digests:

- GGUF: `d45e08ad7bb8787ae9b6f56b6915e8b44ac6e13c6b740fdc7bd591249209a72c`
- tokenizer: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
