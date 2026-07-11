
## Build the externals from source

Build the externals from source if you want to modify it.

**Apple Silicon (arm64), macOS 15+ only** for now.

**Prerequisites:** 
 - **Xcode Command Line Tools** (to install: `xcode-select --install`), 
 - **CMake** (to install: `brew install cmake`), 
 - **ExecuTorch 1.3.1** runtime (see below).
 
<!-- If you want the MLX GPU delegate: install the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`). -->

### 1. Build the ExecuTorch runtime (once)

`neural.live~` links statically to an installed **ExecuTorch 1.3.1** C++ runtime (build it once and reuse it). The commands below will: 
 - Clone ExecuTorch at tag `v1.3.1` into a directory named **exactly `executorch`**
(it refuses other names),
 - Build and install ExecuTorch installs into `executorch/cmake-out` (**This path will be passed as `EXECUTORCH_ROOT` below**). See the [ExecuTorch build guide](https://docs.pytorch.org/executorch/) for the
full guide.

```bash
git clone -b v1.3.1 https://github.com/pytorch/executorch.git --recursive
cd executorch
cmake -S . -B cmake-out -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_COREML=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_MPS=ON       # optional: Apple-GPU MPSGraph delegate (see below)
cmake --build cmake-out --target install -j8
```
<!-- 
> **Metal (experimental):** `-DEXECUTORCH_BUILD_METAL=ON` builds the AOTInductor `metal_backend`
> + `aoti_common` runtime and requires PyTorch at configure time (it links torch's `libomp`).
> `neural_tilde`'s `nn_executorch.cmake` auto-links `metal_backend` when the runtime exports the
> target, so no extra flag is needed on the external build. **Runtime libomp:** the Metal runtime
> and the compiled `.so` embedded in each `.pte` depend on `libomp` at load; the external build
> ships a copy of it inside every ExecuTorch `.mxo` and repoints the dependency to `@rpath` (see
> `cmake/nn_metal_libomp.sh`). Pass `-DNN_LIBOMP_PATH=/path/to/libomp.dylib` — for an exact ABI
> match, use the **same** `libomp` the ET runtime (torch) was built with (e.g. `<torch>/lib/
> libomp.dylib`); it falls back to a Homebrew `libomp` if unset. **Exporting** a Metal `.pte` also
> needs `libomp` on `LIBRARY_PATH` (the exporter auto-adds the active torch's lib dir; set it
> manually otherwise) and tolerates a missing torchao-MPS build. See "Choosing a delegate" below. -->


### 2. Build the `neural.live~` externals

Point `EXECUTORCH_ROOT` at the install dir from step 1 and build (run from this package folder;
clone with `--recursive` if you're fetching it fresh, for the `min-api` submodule):

```bash
cmake -G "Unix Makefiles" -S src -B build_et \
  -DEXECUTORCH_ROOT=/path/to/executorch/cmake-out \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_et -j8
```

This produces the three externals in **`src/externals/`** — `neural.live~.mxo`, `mc.neural.live~.mxo`,
`mcs.neural.live~.mxo`. 

### 3. Install them

Copy the freshly built bundles into the package's top-level `externals/` (the folder Max
actually loads), ad-hoc code-sign them, and restart Max:

```bash
rsync -a --delete src/externals/ externals/
codesign --force --deep -s - externals/*.mxo
```

