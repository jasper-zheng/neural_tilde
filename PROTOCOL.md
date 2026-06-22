# neural ExecuTorch model protocol

## 1. Introduction

If you are exporting your own models to `neural.gen~` or `neural.live~`, please follow this protocol. It provides details on:

- the **model-export side** (Python: the generic `python_tools` exporter)
- the **`neural` runtime side** (C++: `neural.live~`, `neural.gen~`, and the `mc.`/`mcs.` variants).

Both sides build against this single document. Both sides are pinned to **ExecuTorch 1.3.1**.

A model is always a pair of files sharing a basename: a `.pte` program and a `.json` sidecar. The sidecar's top-level **`kind`** field selects one of two profiles:

- **`kind: "live"`**: For **real-time, audio-rate** models, use the `neural.live~` / `mc.neural.live~` / `mcs.neural.live~` family, which processes signal inputs and streams outputs (§3).
- **`kind: "gen"`**: For **offline, one-shot generation** (e.g. a text→audio latent-diffusion) models, uses `neural.gen~`, which generates audio into a `buffer~` (§4).

Both kinds describe a model the **same way** (§2): one or more named **methods**, each a positional signature of an ordered **`inputs`** list — every entry tagged with a **`role`** — plus a single audio **`output`**. The roles are shared; only `signal` (live-only) and `buffer` (gen-only) differ. Settable numeric controls are `attribute`-role inputs in both kinds.

**Read §2 (common structure), then your kind's section: §3 (live) or §4 (gen).** A model that takes a text prompt also ships a separate, kind-agnostic **tokenizer bundle** (§5).

---

## 2. Common model structure

Everything in this section applies to **both** kinds. The kind sections (§3, §4) add only what is specific to that profile.

### 2.1 Files

A model is a file pair sharing a basename:

```
mymodel.pte     # ExecuTorch program (the runnable methods + weights)
mymodel.json    # sidecar metadata (this protocol)
```

The host is given the `.pte` path and loads the `.json` beside it. If the `.json` is missing, the host refuses to load and reports an error. A model that takes a text prompt adds a **tokenizer bundle** beside these two files (§5).

### 2.2 The `.pte` program

- Built with `torch.export.export` → `to_edge_transform_and_lower(..., partitioner=[...])` → `.to_executorch()`. **One ExecuTorch method per registered method**; method names are discovered at runtime via the JSON.
- A method has a **positional signature**: an ordered list of inputs, each tagged with a role (§2.4), returning a single audio tensor. Input **order and dtype are fixed at export** — the sidecar lists them in call order.
- Inputs are **typed** — not all float32 (e.g. token ids are int64, an attention mask is int32).
- **All audio tensors are channel-major** — shape `[batch, channels, time]`. This is the only accepted layout for the `signal`/`buffer` inputs and the `signal`/`audio` output.
- Shapes are **fixed at export** by default. A method MAY be exported with a dynamic batch and/or time `Dim`; if so it must be flagged in the JSON (`"batch": null` and/or `"dynamic_time": true`) and the bounds must cover the sizes the host will use. Other static lengths require a re-export, not a runtime change.

### 2.3 Sidecar JSON template

Both kinds share this template; only the marked fields differ:

```jsonc
{
  "kind": "live",                 // REQUIRED — "live" (§3) | "gen" (§4)
  "executorch_version": "1.3.1",  
  "delegate": "coreml",           // "coreml" | "xnnpack" | "mlx" | "portable"

  "seed": 0,                      // optional, both kinds — default RNG seed for "noise" inputs (§2.4)
  "buffer_size": 4096,            // live only — audio-rate block size the .pte was exported at (§3)

  "methods": {
    "<name>": {
      "inputs": [ /* ordered; each entry has a "role" (§2.4) */ ],
      "output": { /* the single audio output (§2.4) */ },
      "batch": 1,                 // fixed batch dim, or null if exported dynamic
      "dynamic_time": false       // true if the time dim is a dynamic Dim
    }
  },
  "attributes": []                // top-level: reserved (controls are attribute-role inputs)
}
```

The per-kind **input/output descriptor fields** (e.g. the `signal`'s `channels`/`ratio`/`labels`, or a gen input's `shape`/`dtype`) and full worked examples are in §3.2 / §4.1.

### 2.4 Roles

Each input/output carries a **`role`** telling the host where its data comes from / goes to. The host fills inputs **by role, in the listed order**. The roles are shared between the two kinds; `signal` is live-only and `buffer` is gen-only.

| role | dtype | kinds | the host provides |
|---|---|---|---|
| `attribute` | float32 / int64 | live + gen | A scalar `[1]` control exposed as a settable Max attribute, clamped to `[min, max]`, defaulting to `default`; its current value is fed every run (live: every block). |
| `noise` | float32 | live + gen | `N(0,1)` of the given `shape`, **reproducible** from the top-level `seed`: each noise input draws from its own stream keyed by `(seed, name)` (independent of input order / siblings). **gen** re-derives the stream each `generate()` (§4.2); **live** seeds one persistent stream per input at load/reset and **advances it per block**, so the noise evolves yet replays for a given seed. A host MAY instead drive a noise input from a Jitter matrix (§2.4.2). |
| `condition` | int64 / int32 / float32 | live + gen | An externally-supplied control vector matched **by `name`**, supplied as a `list` (by position) or a `dictionary` (by name); **held** across runs/blocks until updated, zero-filled until first set. Any rank — see §2.4.1. |
| `signal` | float32 | **live only** | The per-block multi-channel audio from the signal inlets, decimated by `ratio` into `[batch, channels, T_in]`. **Exactly one** `signal` input per method — it carries the audio-in geometry `channels`/`ratio` (§3). |
| `buffer` | float32 | **gen only** | An init waveform read from a Max `buffer~` for audio-to-audio, channel-major `[1, channels, length]`; the host resamples / channel-maps / crops to the declared geometry (§4). |

Output roles (a method emits exactly one audio output):

| role | kinds | the host does |
|---|---|---|
| `signal` | live | Upsample the model's audio output by `ratio` (repeat-interleave) and write it to the signal outlets (§3). |
| `audio` | gen | Write `channels × length` PCM (channel-major: all of ch0, then ch1, …) into a `buffer~` at `sample_rate`, then signal completion (§4). |

**Inlets.** The host creates one inlet/control per input by role: `signal` → signal inlets (live); `condition` → one control inlet each, in declared order; `attribute` → a Max attribute (not an inlet); `noise` → no inlet, host-filled (but a `noise` MAY gain a matrix inlet, §2.4.2). The output geometry gives the outlets.

**Batch / mc.** `noise` and `condition` tensors are `[n_batches] + shape[1:]` — the sidecar's leading dim is a batch placeholder of `1`, replicated to the host's batch count (the `mc.`/`mcs.` variants); a `condition`'s `list` therefore carries `shape[1:]` values.

#### 2.4.1 Condition shapes

A `condition`'s `shape` can be **any rank** — it is **not** restricted to 2-D `[1, N]`. The inlet/dictionary supplies the values as a **flat list in row-major order**, and the host reshapes that flat list into the declared `shape`. For instance:
 - A condition declared `[1, 4, 256]` expects **1 × 4 × 256 = 1024** values, sent as a flat list of 1024 floats (`row0[0…255], row1[0…255], …`), reshaped to `[1, 4, 256]`.
 - A condition declared `[1, 32]` expects **1 × 32 = 32** values, sent as a flat list of 32 floats, reshaped to `[1, 32]`.

A list of the wrong length is rejected.

#### 2.4.2 Matrix noise

A `noise` input whose `shape` folds to `[planes × H × W]` (planes ≤ 32) can optionally be fed from a Jitter `float32` or `float64` matrix instead of the seed (a `float64` matrix is down-cast to the `float32` noise tensor). The consumer (`neural.gen~` and `neural.live~`; **not** the `mc.`/`mcs.` variants) gives each such input its own extra inlet (accepting a `jit_matrix` message) plus a boolean attribute `<name>_inlet` (e.g. `init_noise_inlet`). The mapping folds all tensor dims except the last two into the planecount, with **width(X) = shape[-1]**, **height(Y) = shape[-2]**. For instance:
 - `[1, 256, 32]`: expect noise from `[jit.matrix 1 float32 32 256]`
 - `[7, 1, 256, 32]`: expect noise from `[jit.matrix 7 float32 32 256]`.

With the toggle **off** (default) or no valid matrix, the input stays **seeded** (gen) / drawn from its seeded stream (live, §2.4). `neural.gen~` reads the matrix once per `generate()`; `neural.live~` feeds the latest matrix every block while the toggle is on.

> This is purely consumer-side: nothing is emitted in the sidecar. A noise input with a compatible shape is automatically picked up according to its `shape`/`dtype`. Max has a maximum of 32 planes for `jit.matrix`, so a noise input that folds to **>32 planes** cannot be fed from an inlet and stays seeded.

**Note:** the matrix inlet expects **standard-normal `N(0,1)`** values, but `jit.noise` emits **uniform `[0,1]`**. Put a **`[neural.gaussianize]`** object to convert uniform `[0,1]` to standard-normal `N(0,1)` before feeding it to the noise inlet.

### 2.5 Field rules (common)

- `kind` (string): **required** — `"live"` (§3) or `"gen"` (§4).
- `methods.<name>.inputs` (array, **ordered**): the positional call signature. Each element carries:
  - `name` (string): **significant** for `attribute` (the consumer exposes one settable control under exactly this name, e.g. `seconds_total`) and for `condition`/`buffer` (matched by name); informational otherwise.
  - `role` (§2.4), and — for every role except `signal` — `shape` (int array) + `dtype` (`"int64"` | `"int32"` | `"float32"` → ExecuTorch `Long` / `Int` / `Float`).
  - `attribute` entries also carry `shape` (`[1]`), `default`, optional `min`/`max`, and an optional `description` (the consumer shows it as the control's inspector label).
  - `noise`/`condition` carry `shape`/`dtype`; a `condition` is matched by `name`.
- `methods.<name>.output`: the single audio output (role `signal` for live, `audio` for gen).
- `methods.<name>.batch` / `dynamic_time` (optional): `batch: null` ⇒ dynamic batch, `dynamic_time: true` ⇒ dynamic time `Dim`.
- `seed` (int, top-level, optional; default `0`): the default RNG seed for **all** `noise` inputs, used by both kinds (§2.4). The consumer exposes it as a settable control.
- `attributes` (top-level): reserved; MUST be `[]`. (Controls live as `attribute`-role inputs, above.)

The kind-specific descriptor fields (`signal`/output `channels`/`ratio`/`labels`; `buffer`/`audio` `channels`/`length`/`sample_rate`; the top-level `buffer_size` / `seed`) are defined in §3 / §4.

---

## 3. `kind: "live"` — real-time audio-rate models

A **live** model is a real-time, audio-rate process: `neural.live~` feeds it contiguous audio blocks and streams the output. On top of the common structure (§2) it adds the `signal` role and host-side rate conversion.

### 3.1 Signal geometry & rate conversion

A live method's signature is **exactly one `signal` input** (the per-block multi-channel audio, a 3-D float32 tensor) optionally followed by the extra inputs it consumes (`attribute` / `noise` / `condition`, §2.4), returning one 3-D float32 audio tensor:

```
signal in :  [batch, in_channels,  T_in ]      T_in  = buffer_size / in_ratio
output    :  [batch, out_channels, T_out]      T_out = buffer_size / out_ratio
```

**Host-side rate conversion.** `neural.live~` converts the sampling rate of the `signal` I/O:
- **Input:** downsample each channel by `in_ratio` to form `T_in = buffer_size / in_ratio`.
- **Output:** upsample the model output by `out_ratio` (repeat-interleave) to recover the `buffer_size` sample rate.
- E.g. an encoder in a codec with `in_ratio` 4096 reduces 44.1 kHz audio to ~10.76 Hz; the decoder expands ~10.76 Hz back to 44.1 kHz audio.

`buffer_size` is the audio-rate block size and MUST be an integer multiple of every `in_ratio` and `out_ratio` in the model. Internal state never changes this dimensionality.

### 3.2 Sidecar JSON (live)

The `signal` input and the `output` carry `channels` + `ratio` (a positive integer) + optional per-channel `labels` (whose length, where given, MUST equal `channels`). The top-level `buffer_size` is the block size the program was exported at; `neural.live~` adopts it to size its internal buffers (a `dynamic_time` method may pick its own multiple instead).

```jsonc
{
  "kind": "live",
  "executorch_version": "1.3.1",
  "delegate": "coreml",           // "coreml" | "xnnpack" | "mlx" | "portable"
  "buffer_size": 4096,            // audio-rate block size the .pte was exported at
  "seed": 0,                      // default RNG seed for noise inputs (optional; §2.4)
  "methods": {
    "forward": {
      // Inputs in CALL ORDER. Exactly one "signal" (the audio); the rest are
      // extra inputs the forward consumes after it (§2.4).
      "inputs": [
        { "name": "signal", "role": "signal", "channels": 1, "ratio": 1,
          "labels": ["(signal) model input 0"] },
        { "name": "temperature", "role": "attribute", "shape": [1], "dtype": "float32",
          "default": 1.0, "min": 0.0, "max": 2.0, "description": "Sampling temperature." }
      ],
      "output": { "name": "audio", "role": "signal", "channels": 1, "ratio": 1,
                  "labels": ["(signal) model output 0"] },
      "batch": 1,                 // fixed batch dim, or null if exported dynamic
      "dynamic_time": false       // true if the time dim is a dynamic Dim
    },
    "encode": {
      "inputs": [ { "name": "signal", "role": "signal", "channels": 1, "ratio": 1,
                    "labels": ["(signal) audio in"] } ],
      "output": { "name": "latent", "role": "signal", "channels": 16, "ratio": 2048,
                  "labels": ["(signal) latent 0", "..."] }
    },
    "decode": {
      "inputs": [ { "name": "signal", "role": "signal", "channels": 16, "ratio": 2048,
                    "labels": ["(signal) latent 0", "..."] } ],
      "output": { "name": "audio", "role": "signal", "channels": 1, "ratio": 1,
                  "labels": ["(signal) audio out"] }
    }
  },
  "attributes": []
}
```

The streaming roles are filled **every block**: `signal` decimated by `ratio`; each `attribute` at its current value; each `noise` a per-block `N(0,1)` draw from its persistent `(seed, name)` stream (or a Jitter matrix when its `<name>_inlet` toggle is on, §2.4.2); each `condition` from its held value (zero until set). The top-level `seed` is exposed as a settable `seed` attribute (changing it restarts the streams deterministically).

**Inlets.** `neural.live~` creates the `channels` signal inlets first, then one control inlet per `condition` input (in declared order), then one `jit_matrix` inlet per matrix-drivable `noise` input (§2.4.2). Attributes (incl. `seed` and each `<name>_inlet` toggle) are Max attributes, not inlets. The `output`'s `channels` give the signal outlets. (The `mc.`/`mcs.` variants expose the `seed` attribute but not the matrix inlets.)

### 3.3 Stateful models

A method MAY keep **persistent internal state** across `execute()` calls — e.g. a real-time cached conv net that retains each block's tail for streaming continuity. In ExecuTorch the state lives in **mutable buffers that persist across `execute()`** (the same mechanism as LLM KV-caches); on the Core ML delegate they are taken over as native Core ML **state** (`take_over_mutable_buffer=True`, iOS18+/macOS15+). The state is internal — never passed in or out — so the method signature is unchanged. The host's only obligations are to load one instance per object and to zero the state on reset (§3.5).

### 3.4 Producer requirements (exporting a live model)

The reference exporter is `neural_tilde.LiveModule` (`python_tools/live_module.py`); a full producer (e.g. RAVE) emits the same protocol from a trained model.

- One `.pte` method per `neural.live~` method: a positional `forward(signal, *extras)` → one audio tensor; a sidecar `.json` of the same basename whose `inputs` list matches that signature, with exactly one `signal` input.
- Deterministic for a given input (and, for a stateful model, its current state). A producer that uses random ops (e.g. RAVE `--stochastic`) needs the corresponding random-op kernels.
- `buffer_size` divisible by every `in_ratio`/`out_ratio`; `signal`/`output` label counts match their `channels`.
- Validated on `portable` and `xnnpack` (execute + equivalence + state persistence); `coreml` persists state via Core ML state take-over; `mlx` does not seem to persist state.

### 3.5 `neural.live~` requirements (consumer / C++)

- Load `.pte` via `executorch::extension::Module`; parse the `.json`; cross-check `method_names()` against JSON `methods`.
- **Adopt the sidecar `buffer_size` as the host block size** (fixed-shape programs): it MUST equal the exported top-level `buffer_size` (the time dim is baked into the `.pte` unless `dynamic_time`). `neural.live~` adopts it, overriding the user's buffer-size argument.
- Build the positional `execute(method, {...})` inputs **by role, in declared order**: the `signal` decimated by `ratio` into `[batch, channels, T_in]`; each `attribute` as a `[1]` scalar at its current value; each `noise` from its persistent `(seed, name)` stream (or a supplied matrix, §2.4.2); each `condition` from its held value (zero until set). Upsample the audio output by `output.ratio`. Validate output shape against `[batch, out_channels, T_out]`.
- Name the signal inlets/outlets from the `signal` labels; add one control inlet per `condition` input; expose each `attribute` input as a Max control.
- **Persist one execution instance per object.** Load the program/method **once per `neural.live~` object** and reuse it for every audio block. **Never** recreate the method per block — that discards any internal state (reintroducing clicks for a stateful model). Feed contiguous blocks; a stateful model handles continuity internally, so the host needs no receptive-field overlap. (`neural.live~` already holds one `Backend`/`Module` per object.)
- **Reset re-initialises state to zero** (§3.3). ExecuTorch does not serialise a guaranteed initial value for the view-`copy_`-mutated state buffers, so the runtime brings them to zero on load and on reset. The current implementation does this by reloading the program (`Backend::reset`); the portable / XNNPACK / Core ML runtimes all reach a defined zero state this way.

### 3.6 Conformance checklist

Producer (Python export):
- [ ] One `.pte` method per `neural.live~` method: positional `forward(signal, *extras)` → one audio tensor; the sidecar `inputs` list matches that order, with exactly one `signal` input.
- [ ] A stateless method is a pure function of its input; a stateful method keeps its state internal (not in the signature).
- [ ] `buffer_size` divisible by every `in_ratio` and `out_ratio`.
- [ ] Sidecar `.json` written with the same basename; method set matches the `.pte`.
- [ ] `signal`/`output` label counts match their `channels`.
- [ ] Each extra input is tagged with its role (`attribute`/`noise`/`condition`) and matches the `forward` argument at that position.

Consumer (`neural.live~` / C++ Backend):
- [ ] Load `.pte` via `executorch::extension::Module`; parse the `.json`; cross-check `method_names()` against JSON `methods`.
- [ ] Adopt the sidecar `buffer_size` as the host block size (fixed-shape programs).
- [ ] Persist one `Module` per object; reuse `execute()` per block (internal-state continuity).
- [ ] Build the positional inputs by role (signal decimated; attribute `[1]`; per-block `(seed, name)` noise or supplied matrix; held condition), `execute(method, {...})`, upsample the audio output by `ratio`.
- [ ] Validate output shape against `[batch, out_channels, T_out]`.
- [ ] Re-initialise a stateful model's state to zero on load/reset (reload).
- [ ] Name signal inlets/outlets from the labels; one control inlet per `condition`; expose each `attribute` as a Max control.

---

## 4. `kind: "gen"` — offline one-shot generators

A **gen** model is an **offline, one-shot generative model**: taking a range of inputs (e.g. token ids, scalar parameters, Gaussian noise, initial audio), it returns one fixed audio buffer in a single method call. It is *not* an audio-rate process. A reference architecture is **Stable Audio 3 (small-music)**: a text-to-audio / audio-to-audio latent-diffusion pipeline (T5Gemma text encoder → 8-step DiT denoiser → SAME-S decoder) exported to a single ExecuTorch/MLX `.pte`.

Because generation can take seconds, `neural.gen~` runs it **on a worker thread, never the audio thread**, and writes the result into a `buffer~`. On top of the common structure (§2) it adds the `buffer` init-audio role, the `seed` field, and (consumer-side) matrix noise.

### 4.1 Sidecar JSON (gen)

All inputs carry `name`/`role`/`shape`/`dtype` (typed, in call order). A `buffer` input and the `audio` output additionally carry `channels`/`length`/`sample_rate` (channel-major, §2.2). The top-level `seed` is the default RNG seed for `noise` inputs (§4.2).

```jsonc
{
  "kind": "gen",
  "executorch_version": "1.3.1",
  "delegate": "mlx",              // "mlx" | "coreml" | "xnnpack" | "portable"
  "seed": 0,                      // default RNG seed for all "noise" inputs
  "methods": {
    "forward": {
      // Inputs in CALL ORDER. Each carries name, role, shape, dtype.
      "inputs": [
        { "name": "input_ids",      "role": "condition", "shape": [1, 256],       "dtype": "int64" },
        { "name": "attention_mask", "role": "condition", "shape": [1, 256],       "dtype": "int32" },
        { "name": "cfg_scale",      "role": "attribute", "shape": [1],            "dtype": "float32",
          "min": 0.0, "max": 15.0, "default": 1.0, "description": "Classifier-free guidance scale." },
        { "name": "init_noise",     "role": "noise",     "shape": [1, 256, 32],   "dtype": "float32" },
        { "name": "init_audio",     "role": "buffer",    "shape": [1, 2, 131072], "dtype": "float32",
          "channels": 2, "length": 131072, "sample_rate": 44100, "description": "Init audio to vary from." }
      ],
      "output": {
        "name": "audio", "role": "audio", "dtype": "float32",
        "shape": [1, 2, 131072],
        "channels": 2, "length": 131072, "sample_rate": 44100
      }
    }
  },
  "attributes": []                // top-level: reserved (generative controls are attribute-role inputs)
}
```

Field rules specific to gen:
- `seed` (int): default seed for **all** `noise` inputs (§4.2); `neural.gen~` exposes it as a settable control.
- `buffer` inputs carry the audio descriptors `channels`/`length`/`sample_rate` — the geometry the host resamples / channel-maps / crops the source clip to (channel-major). `name` is **significant** (the host fills it from a `buffer~` named out-of-band, e.g. `init <name>` in `neural.gen~`); if unset, the host feeds silence.
- `output`: `role: "audio"`, `dtype`, `shape`, and the audio descriptors `channels`/`length`/`sample_rate`.

### 4.2 Seed & reproducible noise

`seed` is the default seed for **all** `noise` inputs (a common top-level field, §2.4). From it the host derives an **independent** RNG sub-stream per noise input, keyed by `(seed, input name)` — so a given seed is reproducible across runs, and toggling/supplying one noise input never perturbs the seeded values of the others (input order is irrelevant). `neural.gen~` exposes this as a settable control and **re-derives each sub-stream every `generate()`** (one-shot). (`neural.live~` shares the keying but seeds one persistent stream per input at load/reset and advances it per block — see §2.4 — so its noise evolves yet still replays for a given seed.)

Matrix noise (driving a `noise` input from a Jitter matrix instead of the seed) is a common, consumer-side feature shared with `neural.live~`; see §2.4.2.

### 4.3 Producer requirements (exporting a gen model)

The reference producer is **`neural_tilde.GenModule`** (`python_tools/gen_module.py`): wrap an `nn.Module`, register each input by name with its role (`register_condition` / `register_attribute` / `register_noise` / `register_buffer_input`), then `register_method(name, inputs=[...], out_channels=..., out_length=..., out_sample_rate=...)` for each generation path — listing, in `forward` order, the inputs that method consumes plus its audio output — and `register_tokenizer` (§5) / `set_seed` (both per-model), then `export_to_pte` → the `.pte` + this sidecar (+ the tokenizer bundle). Multiple `register_method` calls produce multiple entries in `methods{}` (mirroring `LiveModule`). The SA3 `export/` scripts emit the same schema by hand; keep the two in sync.

- One `.pte` with one positional multi-input method per generation path; a generative sidecar `.json` of the same basename whose `methods{}` lists, per method, its `inputs` (call order, typed) and the `output` audio descriptor.
- The tokenizer files referenced by the tokenizer config are bundled next to the `.pte` (§5).
- RNG is **externalised** (the program has no internal RNG) so host-sampled `noise` makes output reproducible from `seed`.
- A golden reference (`*_ref.pt` with matching inputs + the Python-`.pte` output) is provided for the consumer's correctness gate.

> **Correctness gate (important).** Verify the C++ output against the **Python `.pte` runtime** on identical inputs (`max|Δ| ≤ 1e-4`), **not** against a PyTorch CPU run: a diffusion decoder can amplify the tiny FP32 CPU-vs-Metal arithmetic difference by ~10⁴–10⁵×, so a large CPU-vs-Metal divergence is expected and does **not** indicate a host bug. A large C++-vs-Python-`.pte` divergence does (wrong dtype, input order, tensor layout, or tokenizer).

### 4.4 `neural.gen~` requirements (consumer / C++)

1. **Load once, off the audio thread.** Load the `.pte` via `executorch::extension::Module` (lazy; the first `execute()` loads multi-GB weights). **Never** call `execute()` from the audio callback — run generation on a worker thread and notify the patch on completion.
2. **Build typed inputs in order.** For each entry in `methods.<m>.inputs`, construct an `EValue` from a host-owned buffer via `from_blob(data, shape, ScalarType)` with the **exact** dtype named (Long/Int/Float). Pass them to `execute(method, inputs)` in array order. Dtype or order mismatch fails or silently corrupts.
3. **Tokenize per the sidecar** *(unless supplied externally)*. Encode the prompt into the token `condition` inputs — token ids (int64) + attention mask (int32), padded/truncated to `max_length` (§5). C++ token ids MUST match the reference tokenizer **token-for-token**. A consumer MAY instead accept these inputs pre-computed by name — e.g. `neural.gen~` takes them from `neural.tokenizer` via a dictionary and does no tokenization itself.
4. **Sample noise from `seed`.** For each `noise` input, seed an independent RNG with `(seed, input name)` and fill it with `N(0,1)` of its `shape` (§4.2). The per-input keying means a supplied/matrix-driven noise input never shifts the seeded values of the others.
5. **Fill attributes.** Each `attribute` input = its named control's current value (the user's, else `default`), clamped to `[min, max]`.
6. **Deliver the output.** Read the `audio` output tensor; write `channels × length` samples (channel-major) into the target `buffer~` at `sample_rate`; bang a completion outlet (deferred to the main thread).

### 4.5 Conformance checklist

Producer (generative export):
- [ ] `kind: "gen"`; one `.pte` method per generation path with a positional, typed multi-input signature.
- [ ] `inputs` listed in **call order**, each with correct `role`/`shape`/`dtype`.
- [ ] `output` carries `role: "audio"` + `channels`/`length`/`sample_rate` (channel-major).
- [ ] Tokenizer bundle present (if tokens used) and its files bundled beside the `.pte` (§5).
- [ ] Golden `*_ref.pt` (inputs + Python-`.pte` output) provided for the consumer gate.

Consumer (`neural.gen~` / C++ Backend):
- [ ] Load the `.pte` via `Module`; parse the generative `.json`.
- [ ] Build each input `EValue` by role, in order, with the named `ScalarType`.
- [ ] Tokenize per the sidecar; token ids match the reference token-for-token (§5).
- [ ] Seed-driven `N(0,1)` for every `noise` input; params clamped to `[min, max]`.
- [ ] Run `execute()` on a worker thread (never the audio thread).
- [ ] Write the `audio` output into the target `buffer~`; bang completion on the main thread.

---

## 5. Tokenizer bundle (kind-agnostic)

A model that takes a **text prompt** feeds it as token `condition` inputs (ids + attention mask). The text→tokens step is done by the Max `neural.tokenizer` object (or the headless `gen_cli`), **not** by the model `.pte`. The tokenizer settings travel as a **standalone two-file bundle** beside the `.pte`, discovered by filename convention:

```
mymodel.tokenizer.config.json   # tokenizer settings (this section)
mymodel.tokenizer.json          # HuggingFace (HF) tokenizer.json referenced by the config
```

The bundle is **independent of the model sidecar and of `kind`**: `neural.tokenizer` discovers and loads it by filename convention, and the runtime only ever sees the resulting token `condition` inputs (matched by name, §2.4). It is present only if the model has token `condition` inputs. Text prompts are most common with `neural.gen~`, but a `kind: "live"` model that takes a prompt may ship the same bundle.

The config (`<model-stem>.tokenizer.config.json`):

```jsonc
{
  "type": "tokenizers-cpp",                        // loader
  "tokenizer_file": "mymodel.tokenizer.json",      // HF tokenizer.json, relative to THIS file
  "max_length": 256,                               // pad/truncate prompt to this many tokens
  "padding": "max_length",
  "pad_token": "<pad>",                            // token used to fill padding positions
  "padding_side": "right",                         // "right" | "left"
  "ids_key": "input_ids",                          // = the model's token-ids condition input name
  "mask_key": "attention_mask"                     // = the model's attention-mask condition input name
}
```

Field rules:
- `type`: loader (e.g. `"tokenizers-cpp"`). `tokenizer_file`: HF `tokenizer.json`, relative to the config.
- `max_length`; `padding` (`"max_length"`); `pad_token` (default `"<pad>"`); `padding_side` (`"right"` | `"left"`, default `"right"`).
- `ids_key`/`mask_key`: the token-input names. They MUST equal the `condition` input names the method consumes, so `neural.tokenizer`'s output dictionary keys line up with the model inputs.
- (A legacy inline `tokenizer` block inside the model sidecar is still accepted but deprecated; prefer the standalone bundle.)

**Tokenization.** The tokenizer encodes the prompt, then truncates/pads to `max_length` on `padding_side` with the id of `pad_token` (`attention_mask` = 1 for content, 0 for pad). A model may take **multiple** token conditions: e.g. negative prompts add `neg_input_ids` + `neg_attention_mask` as extra `condition` inputs, supplied by a second `neural.tokenizer` configured with `@ids_key neg_input_ids @mask_key neg_attention_mask`.

**Producer.** `register_tokenizer(...)` is available on **both** exporters (backed by the `neural_tilde.Tokenizer` class); on export it copies the HF `tokenizer.json` to `<stem>.tokenizer.json` and writes `<stem>.tokenizer.config.json` beside the `.pte`.

---

## 6. Out of scope (future)

Settable **numeric** controls are now supported: both kinds expose them as `attribute`-role inputs (§2.4). Still out of scope: **non-numeric controls** (string/enum selectors) and **runtime model-state mutation** (e.g. swapping AdaIN timbre-transfer source/target by mutating module state rather than passing a value) — both would need a different control path than a scalar forward input.

Generative (`kind: "gen"`) extensions: dynamic output length (currently the audio `length` is baked at export; arbitrary durations need a re-export), guidance/CFG and inpainting controls (the reference SA3 export bakes `cfg_scale=1.0`, text→audio only), input `audio` / `latent` roles (audio-prompted or latent-prior generation), and a `latent` output role for exposing the diffusion latent before decode.
