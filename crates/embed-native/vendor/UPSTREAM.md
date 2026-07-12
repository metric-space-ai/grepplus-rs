# ggml Kernel Snapshot

The vendored kernel subset is derived from `ggml-org/llama.cpp` and remains
under ggml's MIT license. This manifest records reproducible upstream blob
lineage. Greppy wrappers and fused Qwen kernels are original MIT-licensed code
and are identified separately in `README.md`.

Upstream repository: https://github.com/ggml-org/llama.cpp

## Exact upstream blobs

Files in each row are byte-identical to the named upstream commit's version.
Several files share one commit because that snapshot introduced the same blob.

| Greppy paths | Upstream commit |
|---|---|
| `cuda/ggml-cuda/common.cuh`, `mma.cuh` | `4eac5b45095a4e8a1ff1cce4f6d030e0872fb4ad` |
| `cuda/ggml-cuda/mmq.cuh` | `9725a313be0528214c4a02fed906ddaf7b3f712e` |
| `cuda/ggml-cuda/mmvq.cuh` | `ec16a072f06c9c44d33513405a83068b15ae1b2c` |
| `cuda/ggml-cuda/quantize.cu` | `92f7da00b49ad814b95832dd6610a825bbdd3033` |
| `cuda/ggml-cuda/quantize.cuh` | `972f323e73bf0b28358ccaa3b9aa02779421f260` |
| `cuda/ggml-cuda/unary.cuh` | `86db42e97f6f20330b1a54653eeff6814162c39b` |
| `cuda/ggml-cuda/vecdotq.cuh` | `7e72b38bc186deeee41b4d518a42bb50e1d0ba36` |
| `cuda/ggml-cuda/vendors/cuda.h` | `d6f3030047f85a98b009189e76f441fe818ea44d` |
| `cuda/ggml-include/ggml-common.h`, `metal/shaders/ggml/ggml-common.h` | `2e1f0a889e19a3922db57452268f4574c35c36e5` |
| `cuda/ggml-include/ggml-impl.h` | `3f7c29d318e317b63f54c558bc69803963d7d88c` |
| `metal/shaders/ggml/ggml-metal-impl.h` | `d1649047a33d436142c9d496e190742992c08942` |
| `cuda/ggml-include/ggml-cuda.h` | `d6f3030047f85a98b009189e76f441fe818ea44d` |
| `cuda/ggml-include/ggml-backend.h` | `014dca49d6c1c735d58f8bcf4e101f8cc80fbfc5` |
| `cuda/ggml-include/ggml-openvino.h` | `9789c4ecdc01d571331c14e5197514b53839de4b` |
| `cuda/ggml-include/ggml-opt.h` | `92f7da00b49ad814b95832dd6610a825bbdd3033` |
| `cuda/ggml-include/ggml-rpc.h` | `adb541a6ad077d037edcdca346c6c9624b2aac66` |
| `cuda/ggml-include/ggml.h` | `80d8770804eb712f0464c3705b65acf896c1f49c` |
| `cuda/ggml-include/gguf.h` | `36dafba5c476297261692bfb24c49ec657030c62` |
| Remaining `cuda/ggml-include/*.h` files | `972f323e73bf0b28358ccaa3b9aa02779421f260` |

## Reduced or modified derivatives

- `cuda/ggml-cuda/mmvq.cu` differs by two changed lines from upstream commit
  `fc2b0053ffe878ff5a26934bdb555681f15bc699`; Greppy retains only the dispatch
  surface used by its embedded Q4_K pipeline.
- `metal/shaders/ggml/ggml-metal.metal` is based on commit
  `ae2d34899e2a9a172c7f2090ed4dd366bbf25d0d` and contains Greppy-specific
  compile-time reductions and fused kernels. Its current SHA256 is
  `12c17b77c4944a1f164432100b9b3922969c1fd98740871e6fdf67f9524e8662`.

The source snapshot used for the current llama.cpp performance baseline is
recorded separately in `bench/qwen35_llama_cpp_baseline.md`; it is not implied
to be the origin of older vendored blobs.
