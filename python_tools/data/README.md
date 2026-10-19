# neural_tilde package data

## `aoti_cuda_shims.lib`

The MSVC import library for `aoti_cuda_shims.dll`, the ExecuTorch CUDA runtime's shim DLL.
Exporting a model with `delegate="cuda"` on Windows links each method's AOTInductor
wrapper DLL against it. The PyPI `executorch` wheel doesn't ship it, so it's bundled here and
users don't need an ExecuTorch C++ build just to export. `NN_CUDA_SHIM_DIR` overrides it.

- **Built from:** ExecuTorch `v1.5.1`, MSVC 14.44, CUDA 13.0 (`-DEXECUTORCH_BUILD_CUDA=ON`; see
  `Build.md` → Windows (CUDA)). Copied from `<executorch>/cmake-out/lib/aoti_cuda_shims.lib`.
- **SHA-256:** `c6a0643e2adfc2d32c1d12022dafb6ac12a3508abd562e6fe3dde76499c41871`
- **Pairs with** the `aoti_cuda_shims.dll` shipped in the Windows Max package. Regenerate both
  together whenever ExecuTorch is bumped. A wrapper that calls a shim this library doesn't
  list fails to link at export. One linked against a newer library than the shipped DLL fails
  to load in Max.
