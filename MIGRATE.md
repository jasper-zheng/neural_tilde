# Migrating from `nn~` to `neural.live~`

`neural.live~` is a re-implementation of [`nn~`](https://github.com/acids-ircam/nn_tilde)
(Antoine Caillon & Axel Chemla--Romeu-Santos, ACIDS-IRCAM). The audio behaviour is the same —
a real-time, block-based neural audio process — but the runtime moved from **TorchScript /
libtorch** to **ExecuTorch**, and the model metadata that used to be baked *inside* the
TorchScript program now travels in a separate **`.json` sidecar** beside the model.

This document covers what changes for you, whether you **author models** (the Python export) or
**use the objects in Max**. The authoritative schema for the new format is
[`EXECUTORCH_PROTOCOL.md`](./EXECUTORCH_PROTOCOL.md); this guide maps `nn~` concepts onto it.

> If you only have a trained RAVE `.ts` and just want it running, jump to
> [The fastest path: automated migration](#the-fastest-path-automated-migration).

## At a glance

| Concern | `nn~` | `neural.*` |
|---|---|---|
| Model artifact | a single `.ts` (TorchScript) | a **pair**: `<model>.pte` + `<model>.json` |
| Runtime | libtorch / TorchScript | ExecuTorch (pinned 1.3.1) |
| Max objects | `nn~`, `mc.nn~`, `mcs.nn~`, `nn.info` | `neural.live~`, `mc.neural.live~`, `mcs.neural.live~` (+ new `neural.gen~`, `neural.tokenizer`, `neural.gaussianize`) |
| Object args | `nn~ <model.ts> <method> <buffer_size>` | `neural.live~ <model[.pte]> <method> <buffer_size>` |
| Export API | `nn_tilde.Module.export_to_ts()` | `neural_tilde.LiveModule.export_to_pte()` |
| I/O geometry | `<method>_params` tensor in the `.ts` | sidecar `signal` / `output` `channels` + `ratio` |
| Inlet/outlet labels | `<method>_input_labels` / `<method>_output_labels` | sidecar `labels` arrays |
| Attributes | `set <attr> …` / `get <attr>` messages | **native Max attributes** (`attribute`-role inputs) |

The recurring theme below: everything that `nn~` baked into the TorchScript program as a magic
buffer or scripted method is now a **declared, role-tagged field in the sidecar**, and the host
wires the Max object up from it.

## The fastest path: automated migration

A pre-trained RAVE / `nn~` `.ts` can be converted straight to the new format — no retraining,
no checkpoint needed:

```bash
python -m neural_tilde.migrate percussion.ts --out percussion --delegate coreml
# writes percussion.pte + percussion.json
```

It probes the old `<method>_params` / `<method>_input_labels` / `<method>_output_labels` to
rediscover each method (`encode` / `decode` / `forward` / `prior`), converts the TorchScript to
ExecuTorch, preserves the streaming caches as persistent state, and emits a `kind: "live"`
sidecar. Migration is **best-effort per method**: a method is kept only if it converts, matches
the original output shape, and lowers for the chosen delegate. Caveats:

- **CoreML compiles RNG natively** (`--delegate coreml`, the default). On the CPU runtimes
  (`xnnpack` / `portable`) there is no `rand*` kernel, so pass `--rng zero` for a deterministic
  model.
- **FFT noise-synth methods** (RAVE's `decode` / `forward` complex / STFT branch) can't lower to
  CoreML or the portable CPU runtime and are skipped by default. `--skip-noise-synth` excises that
  branch so they lower, at the cost of the synthesized noise texture. `encode` has no FFT and
  migrates cleanly.
- `prior` (an autoregressive loop) is not a per-block streaming method and is expected to be
  skipped.

For **full fidelity with the noise synth intact**, re-export from the original RAVE `.ckpt`
through the eager `LiveModule` path rather than from the `.ts`. See the docstring in
[`python_tools/migrate.py`](./python_tools/migrate.py) for the full set of flags and behaviour.

## Change 1 — the sidecar replaces `<method>_params`

In `nn~`, each exported method carried a buffer `"<method>_params"` holding four ints:

```
forward_params = tensor([in_channels, in_ratio, out_channels, out_ratio])
```

`neural.*` moves this into the sidecar, and **generalises it**. A method is now a positional list
of **role-tagged inputs** plus one audio output; the `signal` input and the `output` carry the
same four numbers (`channels` + `ratio`), but other inputs can declare *other* roles:

```jsonc
{
  "kind": "live",
  "buffer_size": 4096,
  "methods": {
    "forward": {
      "inputs": [
        { "name": "signal", "role": "signal", "channels": 2, "ratio": 1,
          "labels": ["audio in L", "audio in R"] }
        // …optional attribute / noise / condition inputs follow, in call order…
      ],
      "output": { "name": "audio", "role": "signal", "channels": 2, "ratio": 1,
                  "labels": ["audio out L", "audio out R"] }
    }
  }
}
```

The roles are: `signal` (the per-block audio, live-only), `attribute` (a scalar control —
[Change 2](#change-2--dynamic-attributes)), `noise` (host-filled `N(0,1)`), `condition` (a held
control vector the patch supplies), and `buffer` (an init waveform, gen-only). This role system is
what lets the *same* protocol describe both the real-time `neural.live~` and the new offline
`neural.gen~`. See §2.4 of [`EXECUTORCH_PROTOCOL.md`](./EXECUTORCH_PROTOCOL.md) for the full table.

The host reads the sidecar at load time, refuses to load if it's missing, and sizes the object's
inlets/outlets and inputs from it.

## Change 2 — dynamic attributes

`nn~` attributes were driven by **messages**: you registered an attribute in Python
(`register_attribute(name, values)`, which generated `get_<attr>` / `set_<attr>` scripted
methods), then in Max you set it with `set <attr> <value…>` and read it with `get <attr>`,
discovering the list via `get_attributes`. There were no declared bounds, and the value could be
multi-valued.

`neural.*` turns each control into an **`attribute`-role forward input** with a declared
`default`, optional `min` / `max`, and a `description`:

```jsonc
{ "name": "temperature", "role": "attribute", "shape": [1], "dtype": "float32",
  "default": 1.0, "min": 0.0, "max": 2.0, "description": "Sampling temperature." }
```

At load, `neural.live~` creates a **native Max attribute** for it (built dynamically in
`add_dynamic_attributes`, [`neural.live_tilde.cpp`](./src/frontend/maxmsp/neural.live_tilde/neural.live_tilde.cpp)).
That means you now set it the normal Max ways — in the inspector, as an `@temperature 1.5` box
argument, or with a bare `temperature 1.5` message — the value is **clamped to `[min, max]`**, the
`description` shows as its inspector label, and the current value is fed to the model **every
block**. (The old `set`/`get` message style is gone.)

Two control roles that `nn~` had no equivalent for:

- **`condition`** — a held control *vector* (any rank, any dtype) the patch supplies per run.
  `neural.live~` gives each `condition` its own inlet; supply values by position with a `list`, or
  by name with a `dictionary`. Held across blocks until updated (zero until first set).
- **`noise`** — a tensor the host fills with reproducible `N(0,1)` seeded from the sidecar `seed`
  (exposed as a `seed` attribute). A noise input with a compatible shape can optionally be driven
  from a Jitter matrix via an extra inlet + a `<name>_inlet` toggle; convert `jit.noise`'s uniform
  `[0,1]` to standard-normal first with the new `[neural.gaussianize]` object.

## Change 3 — `input_labels` / `output_labels`

`nn~` stored inlet/outlet assist strings as TorchScript attributes `"<method>_input_labels"` /
`"<method>_output_labels"`. `neural.*` moves them to **`labels` arrays** on the `signal` input and
the `output` in the sidecar (read by `Backend::get_input_labels` / `get_output_labels`):

```jsonc
"inputs":  [ { "name": "signal", "role": "signal", "channels": 2, "ratio": 1,
               "labels": ["audio in L", "audio in R"] } ],
"output":  { "name": "audio", "role": "signal", "channels": 2, "ratio": 1,
             "labels": ["audio out L", "audio out R"] }
```

The behaviour is unchanged — labels name the signal inlets/outlets, and a missing label falls back
to `"(signal) model input N"` / `"(signal) model output N"`. A `labels` array, where present, must
have exactly `channels` entries.

On the Python side this is the **least disruptive** change: `register_method` still takes
`input_labels=` / `output_labels=` keyword arguments, exactly as in `nn~`.

## Export API: `Module` → `LiveModule` / `GenModule`

For audio-rate models, subclass `neural_tilde.LiveModule` instead of `nn_tilde.Module`. The core
of `register_method` is **unchanged** (`in_channels`, `in_ratio`, `out_channels`, `out_ratio`,
`input_labels`, `output_labels`, `test_method`), so most of your export script ports directly.
What changes:

| | `nn~` | `neural.live~` |
|---|---|---|
| Base class | `nn_tilde.Module` | `neural_tilde.LiveModule` |
| Attribute | `register_attribute(name, values)` | `register_attribute(name, default, minimum=…, maximum=…, description=…)` — and it's a `[1]` forward input the method consumes, listed in `register_method(..., inputs=[...])` |
| Export | `export_to_ts(path)` → `.ts` | `export_to_pte(path, delegate="coreml"\|"xnnpack"\|"mlx"\|"portable", buffer_size=4096)` → `.pte` + `.json` |

```python
from neural_tilde import LiveModule

class MyModel(LiveModule):
    def forward(self, x, gain):           # signal, then the extra inputs in order
        ...

model = MyModel()
model.register_attribute("gain", 1.0, minimum=0.0, maximum=2.0, description="output gain")
model.register_method("forward", in_channels=2, in_ratio=1, out_channels=2, out_ratio=1,
                      input_labels=["L", "R"], output_labels=["L", "R"], inputs=["gain"])
model.export_to_pte("mymodel", delegate="coreml", buffer_size=4096)   # → mymodel.pte + mymodel.json
```

See [`examples/export_live_example.py`](./examples/export_live_example.py) for a complete stateful
(`cached_conv`) example with an attribute and a condition. For **offline / generative** models
(text→audio, etc.) there is a new `neural_tilde.GenModule` exporter paired with the `neural.gen~`
object — see §4 of [`EXECUTORCH_PROTOCOL.md`](./EXECUTORCH_PROTOCOL.md).

## Behaviour notes worth knowing

- **`buffer_size` is adopted from the sidecar.** A fixed-shape `.pte` bakes its block size at
  export; `neural.live~` reads the sidecar `buffer_size` and uses it, **overriding** the buffer-size
  box argument. It must be an integer multiple of every `ratio` in the model.
- **Streaming state persists as ExecuTorch mutable buffers.** A stateful model (e.g. `cached_conv`)
  keeps its caches across `execute()` calls — the same mechanism as an LLM KV-cache; on the CoreML
  delegate they become native CoreML *state*. This is what keeps streaming click-free, the same
  guarantee `nn~` gave via libtorch. Reset re-initialises the state to zero.
- **Delegates.** Export targets a backend: `coreml` (Apple Neural Engine; compiles RNG),
  `xnnpack` / `portable` (CPU), or `mlx`. Pick per your deployment; CoreML is the default for the
  migration tool.

## New objects beyond `nn~`

The migration also brings objects with no `nn~` equivalent — see
[`EXECUTORCH_PROTOCOL.md`](./EXECUTORCH_PROTOCOL.md) for details:

- **`neural.gen~`** — offline, one-shot generative models (e.g. text→audio latent diffusion); runs
  on a worker thread and writes the result into a `buffer~`.
- **`neural.tokenizer`** — turns a text prompt into the token `condition` inputs (`input_ids` +
  `attention_mask`) a prompt-driven model expects.
- **`neural.gaussianize`** — maps uniform `[0,1]` (e.g. from `jit.noise`) to standard-normal
  `N(0,1)` for feeding a `noise` inlet.
