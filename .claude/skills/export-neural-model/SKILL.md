---
name: export-neural-model
description: Export a PyTorch audio model to a neural.live~ / neural.gen~ ExecuTorch .pte + .json using the neural_tilde Python exporter. Use this when the user wants to export, convert, or lower a PyTorch audio model (RAVE, codec, vocoder, latent-diffusion / text-to-audio, autoencoder) to run in Max/MSP with neural.gen~ or neural.live~, mentions neural_tilde, LiveModule/GenModule, export_to_pte, ExecuTorch/CoreML/MLX/XNNPACK lowering, or asks to make a model streamable and produce the .pte/.json metadata pair.
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
migration tool instead of a manual port: `python -m neural_tilde.migrate model.ts --out model --delegate coreml` (see `README.md` → *Migrate from `nn~`*).

## Read these first

Before writing any export code, consult these sources and prefer their guidance over
memory — the ExecuTorch API and this repo's protocol evolve.

**ExecuTorch documentation**
- Model Export and Lowering — https://docs.pytorch.org/executorch/1.3/using-executorch-export.html
- Core ML Backend — https://github.com/pytorch/executorch/blob/main/docs/source/backends/coreml/coreml-overview.md
- MLX Backend — https://github.com/pytorch/executorch/tree/main/backends/mlx

**This repository**
- `README.md` — package overview, install, and complete worked export examples.
- `PROTOCOL.md` — the metadata contract you must satisfy; especially [§2.4 Input Roles](../../../PROTOCOL.md#24-input-roles) (each role's JSON fields + the matching `register_*` call).
- `examples/` — runnable references: `export_live_example.py`, `export_gen_example.py`, `export_tokenizer_example.py`. Read the one matching the user's kind and adapt it.
- `python_tools/` — the exporter API: `live_module.py`, `gen_module.py`, `_exporter_base.py` (shared `register_*`), `_tokenizer.py`.

## Prerequisites

```bash
pip install neural-tilde        # installs neural_tilde + cached_conv + executorch + coremltools
```

- Currently Apple Silicon / macOS. The **Core ML** runtime needs **macOS 15+** (the exporter
  targets `iOS18` / native Core ML state).
- The exporter targets **ExecuTorch 1.3.x**. The public API is `neural_tilde.LiveModule`,
  `neural_tilde.GenModule`, `neural_tilde.Tokenizer`.

## Workflow

1. **Pick the kind** (`live` vs `gen`, above).
2. **Shape the model.** All audio tensors are **channel-major `[batch, channels, time]`** — this is the only accepted layout for inputs and the output.
   - **live:** subclass `LiveModule`; `forward(self, signal, *extras)` takes the audio tensor first, then one tensor per extra input in registration order. For click-free streaming across blocks, build the net with `cached_conv` (`cc.use_cached_conv(True)`, `cc.Conv1d`/`cc.CachedSequential`) so per-block state persists.
   - **gen:** build a plain `nn.Module` with one method per generation path (e.g. `prompt2audio`, `audio2audio`) and wrap it by composition: `gm = GenModule(my_model)`.
3. **Register inputs by role** (see the quick reference below and `PROTOCOL.md` §2.4). Register each extra input once by name before registering the method that consumes it.
4. **Register the method(s)** with `register_method(...)`. List the extra inputs in `forward` order via `inputs=[...]`. Pass `test_method=True` to run one forward pass at registration and validate the output shape/dtype against the declaration.
5. **Tokenizer / buffer if needed.** Text-prompt models: `register_tokenizer(tokenizer_file, max_length, ...)` (both exporters) — or build the bundle standalone with `neural_tilde.Tokenizer` (see `export_tokenizer_example.py`). Audio-to-audio gen models: `register_buffer_input(...)`.
6. **Choose a delegate** (see the chooser below).
7. **Export:** `export_to_pte(path, delegate=...)`. This writes `<path>.pte` (program + weights) and the sibling `<path>.json` (metadata). Place both in a folder on Max's file path (Options → File Preferences → Add Path).

## Input roles → `register_*` quick reference

Mirrors `PROTOCOL.md` [§2.4](../../../PROTOCOL.md#24-input-roles). The host fills inputs **by role, in the listed order**.

| role | kind | Python call | JSON fields (metadata) |
|---|---|---|---|
| `attribute` | live + gen | `register_attribute(name, default, minimum=None, maximum=None, description="", dtype=None)` — `LiveModule` infers dtype and takes no `dtype` arg | `name`, `role`, `shape:[1]`, `dtype` (`float32`/`int64`), `default`; optional `min`/`max`/`description` |
| `condition` | live + gen | `register_condition(name, shape, dtype="float32", description="")` | `name`, `role`, `shape` (any rank), `dtype` (`int64`/`int32`/`float32`); optional `description` |
| `noise` | live + gen | `register_noise(name, shape, dtype="float32")` | `name`, `role`, `shape`, `dtype` (`float32`) |
| `signal` | live only | *not registered* — declared via `register_method(...)` `in_channels`/`in_ratio`/`input_labels` | `name:"signal"`, `role`, `channels`, `ratio`; optional `labels` |
| `buffer` | gen only | `GenModule.register_buffer_input(name, channels, length, sample_rate=44100, description="", dtype="float32")` | `name`, `role`, `shape:[1,channels,length]`, `dtype`, `channels`, `length`, `sample_rate`; optional `description` |

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
- `delegate` — `"xnnpack"` | `"coreml"` | `"mlx"` | `"mps"` | `"metal"` | `"portable"` (metal = experimental).
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
- `delegate` — `"mlx"` | `"coreml"` | `"xnnpack"` | `"mps"` | `"metal"` | `"portable"` (metal = experimental).
- `graph_transforms` — callables applied to each method's ExportedProgram before lowering.
- `decompose_conv` — decompose `aten.convolution` → `conv1d`; **set `True` for conv models on the MLX backend**.
- `coreml_compute_units` — CoreML only: `"ALL"` (default) | `"CPU_ONLY"` | `"CPU_AND_GPU"` | `"CPU_AND_NE"`. Leave `None` to use the `NN_COREML_CU` env var if set, else `"ALL"`. (`LiveModule.export_to_pte` takes the same kwarg.)

**Delegate chooser**
- **coreml** — production target on macOS 15+ (native Core ML state via `take_over_mutable_buffer`). Defaults to compute unit `ALL` (Core ML picks CPU/GPU/ANE per op); pick a preset with the `coreml_compute_units` kwarg (or the `NN_COREML_CU` env var). Note Core ML's GPU path miscomputes RAVE (CPU/ANE are bit-correct), so `migrate.py` pins RAVE to `CPU_AND_NE` — validate any model exported with GPU on target.
- **mlx** — Apple-Silicon GPU; good for `gen` / one-shot buffer processing (experimental; convs need `decompose_conv=True`).
- **mps** — Apple-Silicon GPU via MPSGraph; **persists streaming state** (verified bit-identical to xnnpack, so it streams correctly), fp32 by default (`NN_MPS_FP16=1` to opt into fp16). Experimental and **deprecated upstream in ExecuTorch 1.4**; op coverage is incomplete, so some ops abort at runtime — notably **transposed convolution** (`nn.ConvTranspose1d`/`2d`), which Core ML handles: replace it with a forward-conv upsampler (sub-pixel/pixel-shuffle or nearest-neighbor + conv). Validate each model with a smoke test. Needs an MPS-enabled runtime (`-DEXECUTORCH_BUILD_MPS=ON`).
- **metal** — Apple-Silicon GPU via AOTInductor (**experimental**). Whole-graph `torch._inductor` compile to a Metal `.so` embedded in the `.pte` at export time → fast load, no Core ML–style compile stall (trade: multi-minute export compile, larger `.pte`). Broad op coverage (no per-op handler wall). Needs a runtime built with `-DEXECUTORCH_BUILD_METAL=ON`; the exporter auto-handles the AOTInductor specifics (const-fold, `split`/`chunk` decomposition, `libomp` link path, missing torchao-MPS). Streaming (`cached_conv`) persistence is **unverified** — prefer it for `gen` / one-shot; smoke-test `live`.
- **xnnpack** — optimized CPU kernels; portable default for `live`.
- **portable** — plain C++ kernels; maximum compatibility, no delegation.

An unknown delegate string raises `ValueError`.

## Validate the export

1. **Check the `.json` against `PROTOCOL.md`.** Confirm: `kind` matches; a `live` method has **exactly one** `signal` input, a `gen` method has **none**; token ids are `int64` and masks `int32`; and for `live`, `buffer_size` is divisible by every `ratio`. Compare the emitted file to the templates in `PROTOCOL.md` §3.1 / §4.1.
2. **Inspect in Max with `neural.info`.** Load the model and use `neural.info` to list the object's inlets/outlets, attributes, and conditions (see the object list in `README.md`). Verify the object instantiates and its I/O matches the intended signature.
3. **Correctness gate.** Verify the host output against the **Python `.pte` runtime** on identical inputs — **not** a raw PyTorch CPU run (a diffusion decoder amplifies tiny FP32 CPU-vs-Metal differences, so a large CPU-vs-Metal gap is expected and not a bug). For `gen`, target `max|Δ| ≤ 1e-4`. For `live`, feed contiguous blocks and confirm state persists across blocks with no boundary clicks.

## Common pitfalls

- **Layout:** every audio tensor must be channel-major `[batch, channels, time]`.
- **buffer_size** must be an integer multiple of every `in_ratio` and `out_ratio`.
- **Streaming state:** keep `warmup=True` and build with `cached_conv`, or a stateful model reintroduces block-boundary clicks.
- **RNG externalized:** don't sample noise inside the graph — declare `noise` inputs; the host fills reproducible `N(0,1)` from `seed`.
- **MLX + convs:** pass `decompose_conv=True`.
- **Core ML runtime** needs macOS 15+.
- **dtypes:** token ids `int64`, attention masks `int32`; unknown delegate strings raise.

## Resources

- ExecuTorch — Model Export and Lowering: https://docs.pytorch.org/executorch/1.3/using-executorch-export.html
- ExecuTorch — Core ML Backend: https://github.com/pytorch/executorch/blob/main/docs/source/backends/coreml/coreml-overview.md
- ExecuTorch — MLX Backend: https://github.com/pytorch/executorch/tree/main/backends/mlx
- `PROTOCOL.md` — metadata contract ([§2.4 Input Roles](../../../PROTOCOL.md#24-input-roles))
- `README.md` — overview, install, worked examples
- `examples/export_live_example.py`, `examples/export_gen_example.py`, `examples/export_tokenizer_example.py`
- `python_tools/` — `live_module.py`, `gen_module.py`, `_exporter_base.py`, `_tokenizer.py`
