# Migrating from `nn~` to `neural.*~`

The `neural.*~` package originated from [`nn~`](https://github.com/acids-ircam/nn_tilde) (Antoine Caillon & Axel Chemla--Romeu-Santos, ACIDS-IRCAM). The underlying framework moved from **TorchScript** (deprecated in PyTorch 2.10) to **[ExecuTorch](https://docs.pytorch.org/executorch/stable/index.html)**, with:
 - Hardware acceleration on Apple Silicon (via CoreML, MLX, or XNNPack), 
 - A new offline generation object (`neural.gen~`), 
 - Better support for modern generative models' input types (text, noise, condition),
 - A new JSON model metadata.

See the table below for a comparison of the two packages:


| |`neural.*~`|`nn~`|
|---|---|---|
| Offline generation | ✅ `neural.gen~` | ❌ |
| Real-time streaming | ✅ `neural.live~` | ✅ |
| Input types | ✅ `signal`, `attribute`, `condition`, `noise`, `buffer` | ❌ only `signal` + `attribute` |
| Attributes | ✅ Added as native Max attributes | ❌ use `set` / `get` messages |
| Backends | ✅ CoreML, XNNPack, MLX, MPS, Metal *(experimental)*, Portable | ❌ only TorchScript |
| Dynamic device | ❌ Fixed when exporting | ✅ Can be switched in runtime |
| Dynamic buffer size | ❌ Fixed when exporting | ✅ Specified as an argument |
| Library | ExecuTorch | TorchScript (deprecated in PyTorch 2.10) |
| Python Tools | `pip install neural_tilde` | `pip install nn_tilde` |



## Migration tool for RAVEs

A pre-trained RAVE models exported to `.ts` can be converted to the new `.pte` + `.json` format by:

```bash
python -m neural_tilde.migrate percussion.ts --out percussion --delegate coreml
# writes percussion.pte + percussion.json
```

**Note:** This is experimental, migration is not guaranteed to succeed on all models. It also disables the prior methods if your model has one. It also disable the noise-synth if it has one (e.g., the one used in many percussion models).

## Input types (roles)

The `neural.*~` family supports more input roles than `nn~` (see details in README.md [Input Roles](README.md#input-roles)) :

### Attributes

`nn~` attributes are get/set by messages

`neural.*` turns each attribute into a native Max attribute (can be set in the inspector), with a declared `default`, optional `min` / `max`, and a `description`:

```jsonc
{ "name": "temperature", "role": "attribute", "shape": [1], "dtype": "float32",
  "default": 1.0, "min": 0.0, "max": 2.0, "description": "Sampling temperature." }
```

### Conditions

The **`condition`** role holds a control *vector* (any rank / dtype) that can be supplied from inlets with a `list` / `dictionary`.

### Noise

The **`noise`** role holds a tensor the host fills with `N(0,1)` Gaussian noise. It can be optionally supplied from a Jitter matrix via an extra inlet. Use the new `[neural.gaussianize]` object to convert `jit.noise`'s uniform `[0,1]` to standard-Gaussian.


## Python tools' API


| | `neural_tilde` | `nn_tilde` | 
|---|---|---|
| Base class | `neural_tilde.LiveModule` and `neural_tilde.GenModule` | `nn_tilde.Module` |
| Register inputs | Various `register_*` methods | `register_method` |
| Export | `export_to_pte(path, delegate, buffer_size)` | `export_to_ts(path)` |
