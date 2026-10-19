
## Build the externals from source

Build the externals from source if you want to modify it. Both platforms use **ExecuTorch 1.5.1**.

- **macOS (Apple Silicon, arm64), macOS 15+:** see [macOS](#macos).
- **Windows x64 + NVIDIA CUDA:** see [Windows (CUDA)](#windows-cuda).

## macOS

**Prerequisites:**
 - **Xcode** (the full app, not only the Command Line Tools: the MLX delegate needs the Metal compiler; check with `xcrun -sdk macosx --find metal`, or install the Metal Toolchain with `xcodebuild -downloadComponent MetalToolchain`).
 - **CMake** ≥ 3.24 (`brew install cmake`), **git**, and Python 3.12 (e.g. [uv](https://docs.astral.sh/uv/)).
 - **Rust**, for `neural.tokenizer` (`brew install rust`).
 - The `min-api` submodule (Max SDK): clone with `--recursive`, or run `git submodule update --init --recursive`.

### 1. Python export environment

The PyPI `executorch` wheel for macOS arm64 includes the MLX, Core ML, Metal and XNNPACK partitioners, so no source build is needed for exporting:

```bash
uv venv /path/to/nt-env --python 3.12 --python-preference only-managed
uv pip install --python /path/to/nt-env/bin/python -r requirements.txt
uv pip install --python /path/to/nt-env/bin/python -e . --no-deps
```

 - The `executorch` version in this environment must match the runtime the externals link (step 2).
 - Don't base the environment on a conda Python (which is why the `uv venv` above has `--python-preference only-managed`). For `delegate="metal"`, inductor links the kernel library against the Python's `lib/`. Conda's `libc++.1.dylib` has an `@rpath` install name, so the exported `.pte` then fails to load: `Library not loaded: @rpath/libc++.1.dylib`.

### 2. Build the ExecuTorch 1.5.1 runtime (once)

Clone into a directory named exactly `executorch`, apply neural_tilde's Metal patch, then configure with the Python from step 1. The runtime installs into `executorch/cmake-out`, which is the `EXECUTORCH_ROOT` for step 3.

```bash
git clone -b v1.5.1 --recursive https://github.com/pytorch/executorch.git
git -C executorch apply "$PWD/src/cmake/patches/executorch-1.5.1-metal-conv-transpose.patch"
cd executorch
cmake --preset mlx-release \
  -DPYTHON_EXECUTABLE=/path/to/nt-env/bin/python \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=Error \
  -DEXECUTORCH_BUILD_COREML=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_METAL=ON
cmake --build cmake-out --target install -j8
```

 - The patch (run from the neural_tilde checkout) fixes transposed convolutions in the Metal backend: upstream reads the output channel count and the weight layout as if they were a forward conv, so any `ConvTranspose` model fails or computes the wrong result on `delegate="metal"`.
 - `--preset mlx-release` turns on the MLX delegate and builds MLX and `mlx.metallib` from the submodule.
 - `-DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=Error`: the runtime's own error messages reach the Max console through neural_tilde's log sink.
 - `-DEXECUTORCH_BUILD_METAL=ON` builds the experimental Metal (AOTInductor) runtime and needs torch from step 1 at configure time. Leave it out to build without Metal.
 - ExecuTorch 1.4 removed the MPS delegate, so there is no `-DEXECUTORCH_BUILD_MPS`. Re-export old `delegate="mps"` models with `mlx` or `coreml`.

<!--
> **Metal runtime libomp:** the Metal runtime and the compiled `.so` embedded in each `.pte`
> depend on `libomp` at load; the external build ships a copy of it inside every ExecuTorch `.mxo`
> and repoints the dependency to `@rpath` (see `cmake/nn_metal_libomp.sh`). Pass
> `-DNN_LIBOMP_PATH=/path/to/libomp.dylib` — for an exact ABI match, use the **same** `libomp` the
> ET runtime (torch) was built with (e.g. `<torch>/lib/libomp.dylib`); it falls back to a Homebrew
> `libomp` if unset. **Exporting** a Metal `.pte` also needs `libomp` on `LIBRARY_PATH` (the
> exporter auto-adds the active torch's lib dir; set it manually otherwise) and tolerates a missing
> torchao-MPS build. -->

### 3. Build the Max externals

From this package folder:

```bash
export EXECUTORCH_ROOT=/path/to/executorch/cmake-out
cmake --preset macos-release -S src \
  -DNN_LIBOMP_PATH=/path/to/nt-env/lib/python3.12/site-packages/torch/lib/libomp.dylib
cmake --build build_et -j8
```

This produces:
 - the seven externals in **`src/externals/`**: `neural.live~`, `mc.neural.live~`, `mcs.neural.live~`, `neural.gen~`, `neural.tokenizer`, `neural.info`, `neural.gaussianize` (`.mxo`);
 - the standalone harnesses in `build_et/` (`et_smoke_test`, `gen_smoke_test`), plus `build_et/tokenizer/tok_test` and `build_et/generative/gen_cli`.

`-DNN_LIBOMP_PATH` is only needed with the Metal runtime (see the note in step 2). Without Rust, add `-DNN_ENABLE_TOKENIZER=OFF`. The package version comes from `pyproject.toml`.

### 4. Install them

Copy the freshly built bundles into the package's top-level `externals/` (the folder Max
actually loads), ad-hoc code-sign them, and restart Max:

```bash
rsync -a --delete src/externals/ externals/
codesign --force --deep -s - externals/*.mxo
```

### 5. Runtime smoke tests (no Max needed)

```bash
build_et/et_smoke_test model.pte forward --io in.f32 out.f32 --ref ref.f32 --reset-check
build_et/gen_smoke_test model.pte <method>
```


## Windows (CUDA)

**Windows 10/11 x64** with an NVIDIA GPU. This section covers the Python environment for exporting CUDA models, the ExecuTorch runtime, and the Max externals that link against it.

**Prerequisites:**
 - **Visual Studio 2022** with the "Desktop development with C++" workload (MSVC; ClangCL is not needed). Run the export and runtime commands below in a **Developer PowerShell for VS 2022**, so `cl.exe` and `ninja` are on `PATH`.
 - **CUDA Toolkit 13.0** and an NVIDIA driver ≥ 580.
 - **CMake** ≥ 3.24, **git**, and Python 3.12 (e.g. miniconda).
 - **Rust** (MSVC toolchain), for `neural.tokenizer`: `winget install Rustlang.Rustup`, then `rustup default stable-x86_64-pc-windows-msvc` (open a new shell afterwards so `cargo` is on `PATH`).
 - The `min-api` submodule (Max SDK): clone with `--recursive`, or run `git submodule update --init --recursive`.
 - Windows long paths enabled (admin PowerShell: `Set-ItemProperty HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled 1`) and `git config --global core.longpaths true`.

### 1. Python export environment

```powershell
conda create -n nt-cuda python=3.12 -y
conda activate nt-cuda
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # first: CUDA build of torch
pip install -r requirements-cuda.txt
pip install -e . --no-deps
```

Exporting a CUDA model also needs a CUDA GPU on the export machine. Inductor compiles and autotunes the Triton kernels on it.

### 2. Build the ExecuTorch 1.5 runtime (once)

Clone into a short path (the directory must be named exactly `executorch`), apply neural_tilde's patch, then configure with the active `nt-cuda` Python:

```powershell
git clone -b v1.5.1 --recursive https://github.com/pytorch/executorch.git C:\dev\executorch
git -C C:\dev\executorch apply "$PWD\src\cmake\patches\executorch-1.5.1-cuda-windows-paths.patch"
cmake -S C:\dev\executorch -B C:\dev\executorch\cmake-out -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=C:/dev/executorch/cmake-out `
  -DCMAKE_CXX_STANDARD=20 `
  -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=Error `
  -DFLATC_EXECUTABLE="$env:CONDA_PREFIX\Lib\site-packages\executorch\data\bin\flatc.exe" `
  -DPYTHON_EXECUTABLE="$env:CONDA_PREFIX\python.exe" `
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON `
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON `
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON `
  -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120"
cmake --build C:\dev\executorch\cmake-out --target install -j 10
```

 - The patch (run from the neural_tilde checkout) fixes how the CUDA runtime handles the kernel library it extracts to `%TEMP%`:
   - It loads the library through a wide path and no longer aborts on a `%TEMP%` outside the ANSI code page. Without this, CUDA models fail to load (or crash Max) when `%TEMP%` contains non-ASCII characters (e.g. an accented user name).
   - It gives each external its own file names. Every external links its own copy of the runtime, so without this, a second external (e.g. `neural.gen~` after `neural.live~`) would try to overwrite the first one's loaded, locked library, and its CUDA model fails to load (`Failed to write to file`).
 - `-DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=Error`: Release builds compile ExecuTorch's log messages out by default. With error-level logging on, the runtime's own explanations of a failure (e.g. a kernel library that won't load) reach the Max console through neural_tilde's log sink.
 - `-DCMAKE_CXX_STANDARD=20`: the optimized CPU kernels include torch 2.14 headers that need C++20 under MSVC.
 - `-DFLATC_EXECUTABLE`: with Ninja, ExecuTorch's own flatc sub-build can't be found (`flatc.exe ... missing and no known rule to make it`). The `flatc.exe` shipped in the executorch wheel matches the release.
 - The CUDA backend is installed as static libraries plus two DLLs, `lib/aoti_cuda_shims.dll` and `bin/extension_cuda.dll`. Those two ship beside the externals. No CUDA runtime DLL is needed at run time; the NVIDIA driver provides the rest.

### 3. Build the Max externals

From this package folder (any PowerShell; the preset uses the Visual Studio 2022 generator, which finds MSVC itself):

```powershell
cmake --preset windows-cuda-release -S src     # add -DEXECUTORCH_ROOT=... if not C:/dev/executorch/cmake-out
cmake --build build --config Release
```

This produces:
 - the seven externals in **`src\externals\`**: `neural.live~`, `mc.neural.live~`, `mcs.neural.live~`, `neural.gen~`, `neural.tokenizer`, `neural.info`, `neural.gaussianize` (`.mxe64`);
 - the CUDA runtime DLLs in **`build\support\`**: `aoti_cuda_shims.dll`, `extension_cuda.dll`;
 - the standalone harnesses in `build\Release\` (`et_smoke_test`, `gen_smoke_test`), plus `build\tokenizer\Release\tok_test.exe` and `build\generative\Release\gen_cli.exe`.

Without Rust, add `-DNN_ENABLE_TOKENIZER=OFF` to the configure step: everything except `neural.tokenizer`, `tok_test` and `gen_cli` still builds.

The externals use the dynamic MSVC runtime (`/MD`), like ExecuTorch, the CUDA kernels and Max itself. The machine running them needs a Visual C++ Redistributable (x64) at least as new as the toolset that built them (14.44 for VS 2022 17.14). That is newer than what Max needs (Max 9.2 is built with 14.29), so the redistributable a Max install brings may be too old:
 - With a runtime older than about 14.38, `VCRUNTIME140_THREADS.dll` is missing and Max can't load `neural.live~`, `mc.neural.live~`, `mcs.neural.live~` and `neural.gen~` at all.
 - With 14.38–14.43, they load, and each external posts one console error naming the version needed and the download link (https://aka.ms/vs/17/release/vc_redist.x64.exe).
 - `neural.info`, which doesn't need the newer DLLs, posts the same check and shows the loaded runtime as `host.vc_runtime`.
 - `NN_FAKE_VC_RUNTIME=14.30` in Max's environment fakes an old runtime, to test the message.

The package version comes from `pyproject.toml`.

### 4. Install them

Copy the externals and the DLLs into the package's top-level folders (the ones Max loads), then restart Max:

```powershell
New-Item -ItemType Directory -Force externals, support | Out-Null
Copy-Item src\externals\*.mxe64 externals\
Copy-Item build\support\*.dll support\
```

### 5. Runtime smoke tests (no Max needed)

```powershell
build\Release\et_smoke_test.exe model.pte forward --io in.f32 out.f32 --ref ref.f32 --reset-check
build\Release\gen_smoke_test.exe model.pte <method>
```

A CUDA model's `.ptd` is found through the `data_file` in its JSON (or `--data`). If the model can't run on this machine (no NVIDIA GPU, a driver too old for CUDA 13.0, a GPU older than the model's `cuda_min_arch`, a macOS-only delegate), the harness prints the reason and exits with status 3. `CUDA_VISIBLE_DEVICES=-1` hides the GPU, to test that path (in Max too, when Max is started from that shell).

### 6. Make the Windows zip

```powershell
python src\package\third_party_notices.py   # after a dependency change; run in the nt-cuda environment (it reads torch's license)
python src\package\make_windows_zip.py
```

This writes `releases\neural_tilde_<version>_windows_x64.zip` and a `.sha256` file next to it.
 - **Contents:** one `neural_tilde\` folder holding `externals\` (the 7 `.mxe64`), `support\` (the 2 CUDA runtime DLLs), `docs\` (the `.maxref.xml` pages), and `README.md`, `PROTOCOL.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md` and `INSTALL-windows.txt`. There are no models or help patches; those ship separately.
 - **Import check:** before packing, the script checks every binary's DLL imports with `dumpbin`. Each import must be a Windows DLL, a Max DLL, the Visual C++ runtime, or a DLL in `support\`; anything else stops it (e.g. a debug build, or an import of `nvcuda.dll`, which would stop the externals loading on machines without an NVIDIA driver).
 - **Reproducible:** the same inputs give the same zip.

`third_party_notices.py` regenerates the committed `THIRD_PARTY_NOTICES.md`. It reads the licenses from the ExecuTorch checkout, the tokenizers-cpp sources in the build tree, `min-api`, torch's wheel, and the Rust crates through `cargo metadata --offline`.
