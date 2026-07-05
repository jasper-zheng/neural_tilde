
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
  -DEXECUTORCH_BUILD_MPS=ON        # optional: Apple-GPU MPSGraph delegate (see below)
cmake --build cmake-out --target install -j8
```


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

To sanity-check a build without Max, `et_smoke_test` (also built into `build_et/`) loads a `.pte`
through the backend directly:

```bash
build_et/et_smoke_test path/to/model.pte forward
```

### Choosing a delegate for live/streaming models

The ExecuTorch delegate is baked into the `.pte` **at export time** — the externals just load
and run whatever the program was lowered to. The build links all of them (`nn_executorch.cmake`),
so the choice is a *model-export* decision, and it matters for streaming:

- **XNNPACK** (CPU) — correctly persists `cached_conv` streaming state across `execute()`
  everywhere. Safe default for `kind: "live"` models.
- **CoreML** (Apple CPU/GPU/ANE) — also streams correctly: the mutable state buffers are taken
  over as native Core ML state (`take_over_mutable_buffer=True`). Needs **macOS 15+** at runtime.
  Compiles for compute unit `ALL` by default (Core ML picks CPU/GPU/ANE per op); pick a preset
  via the `coreml_compute_units` export kwarg (or `NN_COREML_CU`). Core ML's GPU path miscomputes
  RAVE (CPU/ANE are bit-correct), so `neural_tilde.migrate` pins RAVE to `CPU_AND_NE` — validate
  any model exported with the GPU on target.
- **MLX** (Apple-Silicon GPU) — fast, and **does persist `cached_conv` streaming state** across
  `execute()`: verified bit-identical to XNNPACK on conv models, including multi-rate/strided
  encoders (max abs diff ~1e-7, exactly continuous at block boundaries). So it streams correctly.
  It's still experimental (op coverage is incomplete — e.g. complex/FFT ops are unsupported, and
  some conv `gen` models need `decompose_conv=True`), so validate unusual models. *(Earlier docs
  claimed MLX clicks on streaming; that was an untested assumption — MLX, like XNNPACK, has no
  explicit `take_over_mutable_buffer`, but the mutable caches stay in ExecuTorch-managed memory and persist. )*
- **MPS** (Apple-Silicon GPU via MPSGraph) — **does** persist `cached_conv` streaming state
  (verified bit-identical to XNNPACK), so it streams correctly. The catch is **op/shape coverage**:
  it's experimental (and deprecated upstream in ExecuTorch 1.4), and some ops abort at runtime in
  MPSGraph verification — notably **transposed convolution** (`nn.ConvTranspose1d`/`2d` is silently
  compiled as a forward conv and `SIGABRT`s), which Core ML handles fine. Replace transposed convs with
  a forward-conv upsampler (the bundled `examples/export_gen_example.py` uses sub-pixel/pixel-shuffle
  upsampling for exactly this reason, so it runs on all delegates). Enable MPS with
  `-DEXECUTORCH_BUILD_MPS=ON` (step 1) and **validate each exported model** with
  `et_smoke_test`/`gen_smoke_test` before relying on it.

So: **XNNPACK** and **CoreML** are the safe streaming defaults; **MLX** and **MPS** also persist
streaming state correctly (both verified against XNNPACK), but are experimental — validate a new
model with a smoke test, since their op coverage is narrower (MLX: complex/FFT; MPS: transposed
conv). See `PROTOCOL.md` §2.4 (internal state) for the contract.

To reproduce the MPS validation: `python examples/export_gen_example.py m mps` /
`python examples/export_live_example.py m mps` (the 2nd arg picks the delegate), then run the
matching smoke test on the `.pte`.
