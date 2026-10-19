---
name: export-neural-model
description: Export a PyTorch audio model to a neural.live~ / neural.gen~ ExecuTorch .pte + .json using the neural_tilde Python exporter. Use this when the user wants to export, convert, or lower a PyTorch audio model (RAVE, codec, vocoder, latent-diffusion / text-to-audio, autoencoder) to run in Max/MSP with neural.gen~ or neural.live~, mentions neural_tilde, LiveModule/GenModule, export_to_pte, ExecuTorch/CoreML/MLX/XNNPACK/CUDA lowering, or asks to make a model streamable and produce the .pte/.json metadata pair.
---

# Export a neural audio model for Max/MSP

Guide the user through exporting a PyTorch audio model to the `.pte` + `.json` pair that
`neural.live~` / `neural.gen~` load in Max/MSP, then validating the result. The export uses
the `neural_tilde` Python package (`LiveModule` / `GenModule`), which wraps
`torch.export` → ExecuTorch lowering and writes the metadata JSON described by this repo's
`PROTOCOL.md`.

**First, pick the model kind** — it decides which exporter and workflow you use:

- **`live`** → `LiveModule` (`neural.live~`). Real-time, audio-rate, block-streaming models:
  neural vocoders, codecs (encode/decode), RAVE, timbre transfer. The model processes
  contiguous audio blocks and streams output.
- **`gen`** → `GenModule` (`neural.gen~`). Offline, one-shot generators: latent-diffusion
  text-to-audio, audio-to-audio, inpainting. One method call produces a whole buffer.

If the user has a RAVE TorchScript (`.ts`) model already exported for `nn~`, prefer the
migration tool instead of a manual port: `python -m neural_tilde.migrate model.ts --out model --delegate coreml` (see `README.md` → *Migrate from `nn~`*). `--delegate cuda` on a migrated model is **experimental**: migrated RAVE output currently diverges from the original on CUDA, so for an NVIDIA target re-export from the checkpoint with `LiveModule.export_to_pte(delegate="cuda")` instead.

## Read these first

Before writing any export code, consult these sources and prefer their guidance over
memory — the ExecuTorch API and this repo's protocol evolve.

**ExecuTorch documentation**
- Model Export and Lowering — https://docs.pytorch.org/executorch/stable/using-executorch-export.html
- Core ML Backend — https://github.com/pytorch/executorch/blob/main/docs/source/backends/coreml/coreml-overview.md
- MLX Backend — https://github.com/pytorch/executorch/tree/main/backends/mlx
- CUDA Backend — https://docs.pytorch.org/executorch/stable/backends/cuda/cuda-overview.html

**This repository**
- `README.md` — package overview, install, and complete worked export examples.
- `Build.md` — the export environments (macOS §1, Windows/CUDA §1) and the runtime smoke tests.
- `PROTOCOL.md` — the metadata contract you must satisfy; especially [§2.4 Input Roles](../../../PROTOCOL.md#24-input-roles) (each role's JSON fields + the matching `register_*` call).
- `examples/` — runnable references: `export_live_example.py`, `export_gen_example.py`, `export_tokenizer_example.py`. Read the one matching the user's kind and adapt it.
- `python_tools/` — the exporter API: `live_module.py`, `gen_module.py`, `_exporter_base.py` (shared `register_*`), `_tokenizer.py`, `_lowering.py` (delegate partitioners), `_cuda_lowering.py` (CUDA graph rewrites + preflight).

## Prerequisites

`pip install neural-tilde` only pulls `torch`; **`executorch` is deliberately not a
dependency** because its version must match the runtime the externals link. Install the
pinned export environment from the repo's requirements file (see `Build.md` §1 for each
platform):

```bash
# macOS (Apple Silicon): mlx / coreml / xnnpack / portable
uv venv nt-env --python 3.12 --python-preference only-managed
uv pip install --python nt-env/bin/python -r requirements.txt   # torch 2.14.0, executorch 1.5.1, coremltools 9.0, cached_conv
uv pip install --python nt-env/bin/python -e . --no-deps         # or: pip install neural-tilde --no-deps
```

```powershell
# Windows x64 + NVIDIA: cuda / xnnpack / portable — run in a "Developer PowerShell for VS 2022"
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # CUDA torch FIRST
pip install -r requirements-cuda.txt                                            # executorch 1.5.1, triton-windows, cached_conv
pip install -e . --no-deps
```

- **Version 0.1.0**, pinned to **ExecuTorch 1.5.1** on both platforms (torch 2.14.0). The
  public API is `neural_tilde.LiveModule`, `neural_tilde.GenModule`, `neural_tilde.Tokenizer`.
- Platforms: **macOS on Apple Silicon** (Core ML runtime needs **macOS 15+** — the exporter
  targets `iOS18` / native Core ML state) and **Windows x64** (CUDA models need an NVIDIA GPU
  with driver ≥ 580; XNNPACK models run on any machine).
- Export where the model will run: Apple delegates (`coreml`, `mlx`) are exported on a Mac,
  `cuda` on a Windows machine with a CUDA GPU (Triton kernels are autotuned on the export GPU).
  A model whose delegate the host platform lacks is refused at load with a reason.
- `mps` and `metal` are **no longer supported** as of 0.1.0. `mps` was removed in ExecuTorch
  1.4 (the exporter raises `ValueError`, and legacy `mps` `.pte` files no longer load); use
  `mlx` or `coreml` for the Apple GPU. Re-export any `mps`/`metal` model from 0.0.x.

## Workflow

1. **Pick the kind** (`live` vs `gen`, above).
2. **Shape the model.** All audio tensors are **channel-major `[batch, channels, time]`** — this is the only accepted layout for inputs and the output.
   - **live:** subclass `LiveModule`; `forward(self, signal, *extras)` takes the audio tensor first, then one tensor per extra input in registration order. For click-free streaming across blocks, build the net with `cached_conv` (`cc.use_cached_conv(True)`, `cc.Conv1d`/`cc.CachedSequential`) so per-block state persists.
   - **gen:** build a plain `nn.Module` with one method per generation path (e.g. `prompt2audio`, `audio2audio`) and wrap it by composition: `gm = GenModule(my_model)`.
3. **Register inputs by role** (see the quick reference below and `PROTOCOL.md` §2.4). Register each extra input once by name before registering the method that consumes it.
4. **Register the method(s)** with `register_method(...)`. List the extra inputs in `forward` order via `inputs=[...]`. Pass `test_method=True` to run one forward pass at registration and validate the output shape/dtype against the declaration.
5. **Tokenizer / buffer if needed.** Text-prompt models: `register_tokenizer(tokenizer_file, max_length, ...)` (both exporters) — or build the bundle standalone with `neural_tilde.Tokenizer` (see `export_tokenizer_example.py`). Audio-to-audio gen models: `register_buffer_input(...)`.
6. **Choose a delegate** (see the chooser below).
7. **Export:** `export_to_pte(path, delegate=...)`. This writes `<path>.pte` (program + weights) and the sibling `<path>.json` (metadata). With `delegate="cuda"` the weights go to a third file, `<path>.ptd`, named by the JSON's `data_file` — ship all three together. Place them in a folder on Max's file path (Options → File Preferences → Add Path).

## Input roles → `register_*` quick reference

Mirrors `PROTOCOL.md` [§2.4](../../../PROTOCOL.md#24-input-roles). The host fills inputs **by role, in the listed order**.

| role | kind | Python call | JSON fields (metadata) |
|---|---|---|---|
| `attribute` | live + gen | `register_attribute(name, default, minimum=None, maximum=None, description="", dtype=None)` — `LiveModule` infers dtype and takes no `dtype` arg | `name`, `role`, `shape:[1]`, `dtype` (`float32`/`int64`), `default`; optional `min`/`max`/`description` |
| `condition` | live + gen | `register_condition(name, shape, dtype="float32", description="")` | `name`, `role`, `shape` (any rank), `dtype` (`int64`/`int32`/`float32`); optional `description` |
| `noise` | live + gen | `register_noise(name, shape, dtype="float32")` | `name`, `role`, `shape`, `dtype` (`float32`) |
| `signal` | live only | *not registered* — declared via `register_method(...)` `in_channels`/`in_ratio`/`input_labels` | `name:"signal"`, `role`, `channels`, `ratio`; optional `labels` |
| `buffer` | gen only | `GenModule.register_buffer_input(name, channels, length, sample_rate=44100, description="", dtype="float32", resample=True, pad_value=0.0)` — `resample=False` for control curves (e.g. a 50 Hz F0) copied 1:1 from the `buffer~`; a method may take several (`init <name> <buffer>` in Max) | `name`, `role`, `shape:[1,channels,length]`, `dtype`, `channels`, `length`, `sample_rate`; optional `description`, `resample` (only if false), `pad_value` (only if ≠ 0) |

Token conditions are `int64` (ids) and `int32` (attention masks). Noise is host-sampled
`N(0,1)`, reproducible from the top-level `seed`, so keep RNG **out** of the model graph.

```python
# live (subclass)
model.register_attribute("gain", 1.0, minimum=0.0, maximum=2.0, description="output gain")
model.register_condition("bias", [1, 1], dtype="float32")
model.register_method("forward", in_channels=2, in_ratio=1, out_channels=2, out_ratio=1,
                      inputs=["gain", "bias"])

# gen (compose)
gm.register_condition("input_ids", [1, 256], "int64")
gm.register_noise("init_noise", [1, 256, 32])
gm.register_method("prompt2audio", inputs=["input_ids", "init_noise"],
                   out_channels=2, out_length=131072, out_sample_rate=44100, test_method=True)
```

## `export_to_pte` reference

**LiveModule** (`python_tools/live_module.py`):
```python
export_to_pte(path, delegate="xnnpack", buffer_size=4096, batch=1, strict=False, warmup=True)
```
- `path` — output stem (`.pte` appended if missing); the `.json` is written beside it.
- `delegate` — `"xnnpack"` | `"coreml"` | `"mlx"` | `"cuda"` | `"portable"`.
- `buffer_size` — audio-rate block size baked into the program; **must be a multiple of every `in_ratio`/`out_ratio`**.
- `batch` — fixed batch dim.
- `strict` — `torch.export` tracing mode.
- `warmup` — run one no-grad forward per method before tracing so lazily-allocated `cached_conv` state becomes persistent mutable buffers (leave `True` for streaming models).

**GenModule** (`python_tools/gen_module.py`):
```python
export_to_pte(path, delegate="mlx", graph_transforms=(), decompose_conv=False,
              coreml_compute_units=None)
```
- `path` — output stem (`.pte` appended if missing); `.json` written beside it.
- `delegate` — `"mlx"` | `"coreml"` | `"xnnpack"` | `"cuda"` | `"portable"`.
- `graph_transforms` — callables applied to each method's ExportedProgram before lowering.
- `decompose_conv` — decompose `aten.convolution` → `conv1d`; **set `True` for conv models on the MLX backend**.
- `coreml_compute_units` — CoreML only: `"ALL"` (default) | `"CPU_ONLY"` | `"CPU_AND_GPU"` | `"CPU_AND_NE"`. Leave `None` to use the `NN_COREML_CU` env var if set, else `"ALL"`. (`LiveModule.export_to_pte` takes the same kwarg.)

**Delegate chooser**
- **coreml** — production target on macOS 15+ (native Core ML state via `take_over_mutable_buffer`). Defaults to compute unit `ALL` (Core ML picks CPU/GPU/ANE per op); pick a preset with the `coreml_compute_units` kwarg (or the `NN_COREML_CU` env var). Note Core ML's GPU path miscomputes RAVE (CPU/ANE are bit-correct), so `migrate.py` pins RAVE to `CPU_AND_NE` — validate any model exported with GPU on target.
- **mlx** — Apple-Silicon GPU; good for `gen` / one-shot buffer processing (experimental; convs need `decompose_conv=True`).
- **cuda** — NVIDIA GPU on **Windows x64** via AOTInductor. Each method is compiled whole at export time into Triton kernels + a C++ wrapper DLL embedded in the `.pte`; weights go to the external `<name>.ptd`. **Persists streaming state** (`cached_conv` `LiveModule` exports are correct on CUDA). Exporting needs a CUDA torch (2.14 cu130), an NVIDIA GPU on the export machine, the CUDA Toolkit (`CUDA_PATH`), `triton-windows`, and MSVC (`cl.exe` on `PATH` → run from a Developer PowerShell for VS 2022); a preflight raises with the missing piece. Kernels target **sm_75 (Turing) and newer** by default — `NN_CUDA_ARCH` (e.g. `"86"`) raises the floor, `TORCH_CUDA_ARCH_LIST` sets the embedded SASS; the JSON records `cuda_min_arch` / `cuda_archs` and the host refuses older GPUs. The exporter auto-rewrites convs for the CUDA shims (`conv_transpose1d` → zero-stuffed `conv1d`, `conv1d` → undilated `conv2d`) and pins the device of tensor-factory ops (e.g. `arange`, `hann_window`). No FFT kernels: drop FFT-based stages (e.g. RAVE's noise synth via `migrate --skip-noise-synth`) or reformulate them. Expect a slow (autotuning) export.
- **xnnpack** — optimized CPU kernels; portable default for `live`, and the CPU option on both macOS and Windows.
- **portable** — plain C++ kernels; maximum compatibility, no delegation.

`mps` and `metal` were dropped in 0.1.0 — don't target them (see Prerequisites).

An unknown delegate string raises `ValueError`.

## Validate the export

1. **Check the `.json` against `PROTOCOL.md`.** Confirm: `kind` matches; a `live` method has **exactly one** `signal` input, a `gen` method has **none**; token ids are `int64` and masks `int32`; and for `live`, `buffer_size` is divisible by every `ratio`. Compare the emitted file to the templates in `PROTOCOL.md` §3.1 / §4.1.
2. **Inspect in Max with `neural.info`.** Load the model and use `neural.info` to list the object's inlets/outlets, attributes, and conditions (see the object list in `README.md`). Verify the object instantiates and its I/O matches the intended signature.
3. **Correctness gate.** Verify the host output against the **Python `.pte` runtime** on identical inputs — **not** a raw PyTorch CPU run (a diffusion decoder amplifies tiny FP32 CPU-vs-GPU differences, so a large CPU-vs-GPU gap on MLX/CUDA is expected and not a bug). For `gen`, target `max|Δ| ≤ 1e-4`. For `live`, feed contiguous blocks and confirm state persists across blocks with no boundary clicks.
4. **Runtime smoke test without Max** (optional, needs a source build — `Build.md` §5): `et_smoke_test model.pte forward --io in.f32 out.f32 --ref ref.f32 --reset-check` (live) / `gen_smoke_test model.pte <method>` (gen). A CUDA model's `.ptd` is found through the JSON's `data_file` (or `--data`); exit status 3 means the model can't run on this machine (no NVIDIA GPU, driver too old, GPU below `cuda_min_arch`, or a macOS-only delegate on Windows).

## Common pitfalls

- **Layout:** every audio tensor must be channel-major `[batch, channels, time]`.
- **buffer_size** must be an integer multiple of every `in_ratio` and `out_ratio`.
- **Streaming state:** keep `warmup=True` and build with `cached_conv`, or a stateful model reintroduces block-boundary clicks.
- **RNG externalized:** don't sample noise inside the graph — declare `noise` inputs; the host fills reproducible `N(0,1)` from `seed`.
- **MLX + convs:** pass `decompose_conv=True`.
- **Core ML runtime** needs macOS 15+.
- **CUDA:** export on Windows from a Developer PowerShell (MSVC on `PATH`) with a CUDA build of torch installed *before* `requirements-cuda.txt`; ship the `.ptd` with the `.pte`/`.json`; no FFT ops; the target GPU must be ≥ `cuda_min_arch` (sm_75 default) with driver ≥ 580.
- **Delegate ↔ platform:** `coreml`/`mlx` models run only on macOS, `cuda` only on Windows + NVIDIA; `xnnpack`/`portable` run on both.
- **No `mps`/`metal`:** dropped in 0.1.0 — re-export old models with `mlx`/`coreml` (Apple GPU) or `cuda`.
- **dtypes:** token ids `int64`, attention masks `int32`; unknown delegate strings raise.

## Resources

- ExecuTorch — Model Export and Lowering: https://docs.pytorch.org/executorch/stable/using-executorch-export.html
- ExecuTorch — Core ML Backend: https://github.com/pytorch/executorch/blob/main/docs/source/backends/coreml/coreml-overview.md
- ExecuTorch — MLX Backend: https://github.com/pytorch/executorch/tree/main/backends/mlx
- ExecuTorch — CUDA Backend: https://docs.pytorch.org/executorch/stable/backends/cuda/cuda-overview.html
- `Build.md` — export environments (macOS / Windows CUDA) and smoke-test harnesses
- `PROTOCOL.md` — metadata contract ([§2.4 Input Roles](../../../PROTOCOL.md#24-input-roles))
- `README.md` — overview, install, worked examples
- `examples/export_live_example.py`, `examples/export_gen_example.py`, `examples/export_tokenizer_example.py`
- `python_tools/` — `live_module.py`, `gen_module.py`, `_exporter_base.py`, `_tokenizer.py`, `_lowering.py`, `_cuda_lowering.py`
