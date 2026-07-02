# `neural.*~` ExecuTorch model protocol

If you are exporting your own models to `neural.gen~` or `neural.live~`, please follow this protocol. It provides details on:

- the **model-export side** (Python: the `pip install neural_tilde` exporter).
- the **MaxMSP runtime side**.

<!-- Both sides build against this single document. Both sides are pinned to **ExecuTorch 1.3.1**. -->


## Contents

- [1. Model kind: `live` vs `gen`](#1-model-kind-live-vs-gen)
- [2. File structure](#2-file-structure)
  - [2.1 Files](#21-files)
  - [2.2 The `.pte` program](#22-the-pte-program)
  - [2.3 The `.json` metadata](#23-the-json-metadata)
  - [2.4 Input Roles](#24-input-roles)
    - [2.4.1 Attribute](#241-attribute-live--gen)
    - [2.4.2 Condition](#242-condition--live--gen)
    - [2.4.3 Noise](#243-noise--live--gen)
    - [2.4.4 Signal](#244-signal--live-only)
    - [2.4.5 Buffer](#245-buffer--gen-only)
  - [2.5 Output Roles](#25-output-roles)
- [3. `kind: "live"` — real-time audio-rate models](#3-kind-live-for-real-time-audio-rate-models)
  - [3.1 Example Metadata JSON (live)](#31-example-metadata-json-live)
  - [3.3 Stateful models](#33-stateful-models)
- [4. `kind: "gen"` — offline one-shot generators](#4-kind-gen-for-offline-one-shot-generators)
  - [4.1 Example Metadata JSON (gen)](#41-example-metadata-json-gen)
- [5. Tokenizer](#5-tokenizer)


## 1. Model kind: `live` vs `gen`

The `neural.*~` package has two Max objects with different runtime behaviors:

- **`kind: "live"`**: For real-time, audio-rate models, use the `neural.live~` / `mc.neural.live~` / `mcs.neural.live~` family, which processes signal inputs and streams outputs.
- **`kind: "gen"`**: For offline, one-shot generation (e.g. a latent-diffusion) models, uses `neural.gen~`, which generates audio into a `buffer~`.

Both kinds describe a model with: one or more named **methods**, each a positional signature of an ordered **inputs** list. Each input is tagged with a **role** that tells the host where its data comes from (e.g. a signal inlet, a vector condition, a Max attribute, etc). Each method returns **one audio output**.


## 2. File structure

Everything here applies to both `live` and `gen`. 

### 2.1 Files

A model is a file pair sharing a basename:

```
mymodel.pte     # ExecuTorch program (the runnable methods + weights)
mymodel.json    # metadata (this protocol)
```

The host is given the `.pte` path and loads the `.json` beside it. A model that takes a text prompt adds a **tokenizer bundle** beside these two files (see §5).

### 2.2 The `.pte` program

- Built with `torch.export.export` → `to_edge_transform_and_lower(..., partitioner=[...])` → `.to_executorch()`. **One ExecuTorch method per registered method**; method names are discovered at runtime via the JSON.
- A method has **an ordered list of inputs**, each tagged with a role (§2.4), returning a single audio tensor. The metadata lists them in call order.
- All audio tensors are **channel-major** (i.e., shape `[batch, channels, time]`). When exporting the model, this is the only accepted layout for inputs and the output.
- Shapes are **fixed at export** by default. A method may be exported with a dynamic batch and/or time dimension (experimental), if so it must be flagged in the JSON (`"batch": null` and/or `"dynamic_time": true`). 

### 2.3 The `.json` metadata

Both kinds share this template:

```jsonc
{
  "kind": "live",                 // REQUIRED: "live" or "gen"
  "executorch_version": "1.3.1",  // Optional
  "delegate": "coreml",           // REQUIRED: "coreml, xnnpack, mlx, or portable"
  "seed": 0,                      // Optional
  "buffer_size": 4096,            // live only

  "methods": {
    "<name>": {
      "inputs": [ /* ordered; each entry has a "role" (§2.4) */ 
        { "name": ..., "role": "signal/buffer | attribute | noise | condition", ...}
        { "name": ..., "role": "signal/buffer | attribute | noise | condition", ...}
        ...
      ],
      "output": { /* the single audio output (§2.4) */ 
        "name": ..., "role": "signal/audio", ...
      },
      "batch": 1,                 // fixed batch dim, or null if dynamic
      "dynamic_time": false       // true if the time dim is a dynamic Dim
    }
  }
}
```

### 2.4 Input Roles

In the `inputs` field: Each input carries a **`role`** to tell the host where its data comes from. The host fills inputs in the listed order. 

Use the `register_*` methods in the Python tool to declare each input's role. 

#### 2.4.1 Attribute (live + gen)

A scalar number exposed as a **Max attribute** (by `name`), clamped to `[min, max]`, defaulting to `default`.

In the JSON metadata:
- **Required:** `name`, `role: "attribute"`, `shape: [1]`, `dtype` (`"float32"` | `"int64"`), `default`.
- **Optional:** `min`, `max`, `description`.

```jsonc
{ "name": "cfg_scale", "role": "attribute", "shape": [1], "dtype": "float32",
  "default": 1.0, "min": 0.0, "max": 15.0,
  "description": "Classifier-free guidance scale." }
```

**Python:** `register_attribute(name, default, minimum=None, maximum=None, description="", dtype=None)` on both `GenModule` / `LiveModule` (`LiveModule` takes no `dtype` — it infers `int64`/`float32` from `default`).

```python
m.register_attribute("cfg_scale", 1.0, 0.0, 15.0, "Classifier-free guidance scale.")
```

#### 2.4.2 Condition  (live + gen)

A control vector supplied from the patch by inlets as a `list` or a `dictionary` (by position or key name). Unsupplied conditions are zero-filled.

A condition's shape can be any rank, **not** restricted to `[1, N]`. The values are supplied as a flatten list in row-major order, and the host reshapes them into the desired shape. For instance:
 - `[1, 4, 256]`: Supply a 1024-number list (`row0[0…255], row1[0…255], …`), the host reshapes it to `[1, 4, 256]`.
 - `[1, 32]`: Supply a 32-number list, the host reshapes it to `[1, 32]`.  

In the JSON metadata:
- **Required:** `name`, `role: "condition"`, `shape`, `dtype` (`"int64"` | `"int32"` | `"float32"`).
- **Optional:** `description`.

```jsonc
{ "name": "input_ids", "role": "condition", "shape": [1, 256], "dtype": "int64" }
```

**Python:** `register_condition(name, shape, dtype="float32", description="")` (both exporters).

```python
m.register_condition("input_ids", [1, 256], "int64")
```

#### 2.4.3 Noise  (live + gen)

The host fills a `noise` input with standard Gaussian noise `N(0,1)` of the given `shape`, drawn from the `seed` attribute. 

**Noise from Jitter matrices:** A `noise` input whose shape folds to `[planes × H × W]` (planes ≤ 32) can optionally be fed from a Jitter matrix. The host gives each such input its own extra Jitter matrix inlet plus a boolean attribute `<name>_inlet` (e.g. `init_noise_inlet`). All tensor dims except the last two fold into the planecount, with `width(X) = shape[-1], height(Y) = shape[-2]`, for instance:
 - `[1, 256, 32]` -> `[jit.matrix 1 float32 32 256]`
 - `[7, 1, 256, 32]` -> `[jit.matrix 7 float32 32 256]`

**Note:** the inlet expects standard-normal `N(0,1)`, but `jit.noise` emits uniform `[0,1]` — put a `[neural.gaussianize]` in between.

- **Required:** `name`, `role: "noise"`, `shape`, `dtype` (`"float32"`).
- **Optional:** none.

```jsonc
{ "name": "init_noise", "role": "noise", "shape": [1, 256, 32], "dtype": "float32" }
```

**Python:** `register_noise(name, shape, dtype="float32")` (both exporters).

```python
m.register_noise("init_noise", [1, 256, 32])
```

#### 2.4.4 Signal  (live only)

Multi-channel signal inlets, downsampled by `ratio` into `[batch, channels, T_in]`. For instance, for a codec with `ratio: 4096`, a 44.1 kHz audio input is downsampled to ~10.76 Hz latent signal for the model. The host upsamples the output by ratio to recover the audio rate.

- **Required:** `name: "signal"`, `role: "signal"`, `channels`, `ratio` (positive int).
- **Optional:** `labels` (per-channel inlet names, length must equal `channels`).

```jsonc
{ "name": "signal", "role": "signal", "channels": 1, "ratio": 1,
  "labels": ["(signal) model input 0"] }
```

**Python:** the `signal` input has no dedicated register call; declare its geometry in `LiveModule.register_method(...)` via `in_channels`/`in_ratio`/`input_labels` (and the output via `out_channels`/`out_ratio`/`output_labels`). `inputs=[...]` lists the extra inputs consumed after the signal.

```python
m.register_method("forward", in_channels=1, in_ratio=1, out_channels=1, out_ratio=1,
                  input_labels=["(signal) model input 0"], inputs=["temperature"])
```

#### 2.4.5 Buffer  (gen only)

An initial waveform read from a Max `buffer~` for audio-to-audio, the shape is `[1, channels, length]`, the host resamples / channel-maps / crops the source clip to the declared shape.

- **Required:** `name`, `role: "buffer"`, `shape: [1, channels, length]`, `dtype` (`"float32"`), `channels`, `length`, `sample_rate`.
- **Optional:** `description`. (`channels`/`length` are duplicated in `shape`)

```jsonc
{ "name": "init_audio", "role": "buffer", "shape": [1, 2, 131072], "dtype": "float32",
  "channels": 2, "length": 131072, "sample_rate": 44100,
  "description": "Init audio to vary from." }
```

**Python:** `GenModule.register_buffer_input(name, channels, length, sample_rate=44100, description="", dtype="float32")`.

```python
m.register_buffer_input("init_audio", channels=2, length=131072, sample_rate=44100,
                        description="Init audio to vary from.")
```

### 2.5 Output Roles

A method emits exactly one audio output:
 - **live** → `role: "signal"` with `channels`/`ratio`/`labels`: the host upsamples the model output by `ratio` (repeat-interleave) to the signal outlets (§3). 
```jsonc
"output": { "name": "audio", "role": "signal", "channels": 1, "ratio": 1,
            "labels": ["(signal) audio out"] }
```
 - **gen** → `role: "audio"` with `dtype`/`shape`/`channels`/`length`/`sample_rate`: the host writes `channels × length` channel-major PCM into a `buffer~` at `sample_rate`.

 ```jsonc
 "output": {
        "name": "audio", "role": "audio", "dtype": "float32",
        "shape": [1, 2, 131072],
        "channels": 2, "length": 131072, "sample_rate": 44100
      }
  ```

## 3. `kind: "live"` for real-time audio-rate models

Use `neural_tilde.LiveModule` (`python_tools/live_module.py`) in the Python tool for live models.


### 3.1 Example Metadata JSON (live)

```jsonc
{
  "kind": "live",
  "executorch_version": "1.3.1",
  "delegate": "coreml",           
  "buffer_size": 4096,            
  "seed": 0,                      
  "methods": {
    "forward": {
      "inputs": [
        { "name": "signal", "role": "signal", 
          "channels": 1, "ratio": 1,
          "labels": ["(signal) model input 0"] },
        { "name": "temperature", "role": "attribute", 
          "shape": [1], "dtype": "float32",
          "default": 1.0, "min": 0.0, "max": 2.0, 
          "description": "Sampling temperature." }
      ],
      "output": { 
          "name": "audio", "role": "signal", 
          "channels": 1, "ratio": 1,
          "labels": ["(signal) model output 0"] },
      "batch": 1,
      "dynamic_time": false
    },
    "encode": {
      "inputs": [ 
        { "name": "signal", "role": "signal", 
          "channels": 1, "ratio": 1,
          "labels": ["(signal) audio in"] } 
      ],
      "output": { 
        "name": "latent", "role": "signal", 
        "channels": 16, "ratio": 2048,
        "labels": ["(signal) latent 0", "..."] }
    },
    "decode": {
      "inputs": [ 
        { "name": "signal", "role": "signal", 
          "channels": 16, "ratio": 2048,
          "labels": ["(signal) latent 0", "..."] } 
      ],
      "output": { 
        "name": "audio", "role": "signal", 
        "channels": 1, "ratio": 1,
        "labels": ["(signal) audio out"] }
    }
  }
}
```

### 3.3 Stateful models

A method MAY keep **persistent internal state** across `execute()` calls — e.g. a real-time cached conv net that retains each block's tail for streaming continuity. In ExecuTorch the state lives in **mutable buffers that persist across `execute()`** (the same mechanism as LLM KV-caches); on the Core ML delegate they are taken over as native Core ML **state** (`take_over_mutable_buffer=True`, iOS18+/macOS15+). The state is internal — never passed in or out — so the method signature is unchanged.


## 4. `kind: "gen"` for offline one-shot generators

Use `neural_tilde.GenModule` (`python_tools/gen_module.py`) for offline generative models.


### 4.1 Example Metadata JSON (gen)

```jsonc
{
  "kind": "gen",
  "executorch_version": "1.3.1",
  "delegate": "mlx",
  "seed": 0,
  "methods": {
    "forward": {
      "inputs": [
        { "name": "input_ids",   "role": "condition", 
          "shape": [1, 256],     "dtype": "int64" },
        { "name": "attention_mask", "role": "condition", 
          "shape": [1, 256],        "dtype": "int32" },
        { "name": "cfg_scale",   "role": "attribute", 
          "shape": [1],          "dtype": "float32",
          "min": 0.0, "max": 15.0, "default": 1.0, 
          "description": "Classifier-free guidance scale." },
        { "name": "init_noise",    "role": "noise",     
          "shape": [1, 256, 32],   "dtype": "float32" },
        { "name": "init_audio",    "role": "buffer",    
          "shape": [1, 2, 131072], "dtype": "float32",
          "channels": 2, "length": 131072, "sample_rate": 44100, 
          "description": "Init audio to vary from." }
      ],
      "output": {
        "name": "audio", "role": "audio", "dtype": "float32",
        "shape": [1, 2, 131072],
        "channels": 2, "length": 131072, "sample_rate": 44100
      }
    }
  }
}
```


## 5. Tokenizer

A model that takes a **text prompt** feeds it as token `condition` inputs (ids + attention mask). The text-to-tokens step is done by the Max `neural.tokenizer` object, not by the model `.pte`. The tokenizer file convention:

```
mymodel.tokenizer.config.json   # tokenizer settings (this section)
mymodel.tokenizer.json          # HuggingFace (HF) tokenizer.json referenced by the config
```

The config (`mymodel.tokenizer.config.json`):

```jsonc
{
  "type": "tokenizers-cpp",         // fixed
  "tokenizer_file": "mymodel.tokenizer.json", // HF tokenizer.json
  "max_length": 256,                // pad/truncate prompt to this many tokens
  "padding": "max_length",
  "pad_token": "<pad>",             // token used to fill padding positions
  "padding_side": "right",          // "right" | "left"
  "ids_key": "input_ids",           // the model's token-ids key name
  "mask_key": "attention_mask"      // = the model's attention-mask key name
}
```

**Tokenization.** The tokenizer encodes the prompt, then truncates/pads to `max_length` on `padding_side` with the id of `pad_token` (`attention_mask` = 1 for content, 0 for pad). A model may take **multiple** token conditions: e.g. negative prompts add `neg_input_ids` + `neg_attention_mask` as extra `condition` inputs, supplied by a second `neural.tokenizer`.

**Python tool.** `register_tokenizer(...)` is available on the `neural_tilde.Tokenizer` class; on export it copies the HF `tokenizer.json` to `<name>.tokenizer.json` and writes `<name>.tokenizer.config.json`.

