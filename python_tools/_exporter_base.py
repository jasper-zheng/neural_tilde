"""Shared base for the ``neural_tilde`` ExecuTorch exporters.

Both exporters describe a model as a set of named *methods*, each an ordered,
positional ``forward`` signature whose entries are tagged with a role (see
``EXECUTORCH_PROTOCOL.md``). The extra (non-signal) inputs — ``attribute`` /
``noise`` / ``condition`` / ``buffer`` — are registered **once by name** into
shared, name-keyed spec tables, and each method then lists, in order, which of
them its ``forward`` consumes. That common machinery lives here:

- :class:`LiveModule` (``neural.live~``) adds the audio ``signal`` input + I/O
  geometry per method.
- :class:`GenModule` (``neural.gen~``) adds the ``buffer`` input, the per-method
  audio ``output``, the tokenizer bundle and the RNG seed.

This module pulls in no ExecuTorch dependency (the lowering helpers in
``_lowering`` are imported lazily by the subclasses), so the metadata it builds
can be constructed and tested with torch alone.
"""

import json
import logging
import os
from typing import Optional, Sequence, Union

import torch

from ._tokenizer import Tokenizer


class _ExporterBase(torch.nn.Module):
    """Common registration/metadata machinery for the two exporters.

    Subclasses own method registration (``register_method``), the sidecar-metadata
    builder and ``export_to_pte``; everything here is shared and kind-agnostic.
    """

    def __init__(self) -> None:
        super().__init__()
        # name -> sidecar spec dict, one table per kind of extra (non-signal)
        # input. A method lists, in order, which of these its forward consumes
        # after the leading positional input (see _declare_method). ``_buffer_specs``
        # is only ever populated by the generative exporter.
        self._attribute_specs = {}
        self._noise_specs = {}
        self._condition_specs = {}
        self._buffer_specs = {}
        self._methods = []  # registered method names, in registration order
        # Optional tokenizer bundle (shared by every method; either exporter may
        # ship one — tokens arrive as ordinary `condition` inputs matched by name).
        self._tokenizer: Optional[Tokenizer] = None
        # Default RNG seed for "noise" inputs, emitted as a top-level sidecar
        # field for both kinds. gen re-derives reproducible noise per generate();
        # live seeds one persistent stream per noise input from (seed, name) and
        # advances it per block (the host exposes `seed` as a settable control).
        self._seed: int = 0

    # ------------------------------------------------------------ extra inputs

    def register_attribute(self,
                           attribute_name: str,
                           default: Union[bool, int, float],
                           minimum: Optional[float] = None,
                           maximum: Optional[float] = None,
                           description: str = "",
                           dtype: Optional[str] = None):
        """Register a numeric scalar control fed to a method as a ``[1]`` forward input.

        The host owns the value (Max attribute / message / API) and feeds the current
        value at run time, clamped to ``[minimum, maximum]``; ``default`` is used until
        the user sets it. ``dtype`` defaults to being inferred from ``default``'s Python
        type (``bool``/``int`` → ``int64``, ``float`` → ``float32``); pass it explicitly
        to override. The default value is always serialized as a JSON number — the C++
        host reads it as a double regardless of dtype.

        Args:
            attribute_name: name of the attribute (= the Max attribute name)
            default: default value; its Python type selects the dtype when ``dtype`` is None
            minimum, maximum: optional clamp range (the host clamps on set)
            description: shown as the attribute's inspector label
            dtype: ``"float32"`` or ``"int64"``; inferred from ``default`` when omitted
        """
        if dtype is None:
            # bool is a subclass of int; both export as int64.
            if isinstance(default, (bool, int)):
                dtype = "int64"
            elif isinstance(default, float):
                dtype = "float32"
            else:
                raise TypeError(
                    (f"Attribute '{attribute_name}' default must be bool/int/float, "
                     f"got {type(default).__name__}"))

        spec = {"name": attribute_name, "role": "attribute", "shape": [1],
                "dtype": dtype, "default": float(default)}
        if minimum is not None:
            spec["min"] = float(minimum)
        if maximum is not None:
            spec["max"] = float(maximum)
        if description:
            spec["description"] = description
        self._attribute_specs[attribute_name] = spec

    def register_noise(self,
                       name: str,
                       shape: Sequence[int],
                       dtype: str = "float32") -> None:
        """Register a noise input the host fills with ``N(0,1)`` (see each exporter's docs).

        ``shape``'s leading dimension is a batch placeholder of 1; declare e.g.
        ``[1, latent, frames]``.
        """
        self._noise_specs[name] = {"name": name, "role": "noise",
                                   "shape": [int(d) for d in shape],
                                   "dtype": dtype}

    def register_condition(self,
                           name: str,
                           shape: Sequence[int],
                           dtype: str = "float32",
                           description: str = "") -> None:
        """Register an externally-supplied conditioning input, matched by ``name``.

        Held/zero-filled until supplied. Tokens are ``int64``; attention masks are
        ``int32``. ``shape``'s leading dim is a batch placeholder of 1 (e.g. ``[1, 16]``).
        """
        spec = {"name": name, "role": "condition",
                "shape": [int(d) for d in shape], "dtype": dtype}
        if description:
            spec["description"] = description
        self._condition_specs[name] = spec

    # -------------------------------------------------------------- resolution

    def _extra_spec(self, name: str) -> dict:
        """Resolve a registered extra-input name to its sidecar spec dict."""
        for table in (self._attribute_specs, self._noise_specs,
                      self._condition_specs, self._buffer_specs):
            if name in table:
                return table[name]
        raise ValueError(
            (f'Unknown input "{name}" — call register_attribute / register_noise / '
             "register_condition / register_buffer_input before register_method"))

    def _extra_example(self, name: str) -> torch.Tensor:
        """A zero example tensor for a registered extra input (for tracing/testing)."""
        spec = self._extra_spec(name)
        torch_dtype = (torch.int64 if spec["dtype"] == "int64"
                       else torch.int32 if spec["dtype"] == "int32"
                       else torch.float32)
        if spec["role"] == "attribute":
            return torch.tensor([spec["default"]], dtype=torch_dtype)
        return torch.zeros(spec["shape"], dtype=torch_dtype)

    # ------------------------------------------------------------------ method

    def _declare_method(self, method_name: str,
                        input_names: Sequence[str]) -> None:
        """Record a method's ordered extra-input names (validating each exists).

        Stores the list under ``self.<method_name>_inputs``; the subclass appends
        ``method_name`` to ``self._methods`` once its own registration fully succeeds.
        """
        names = list(input_names)
        for a in names:
            self._extra_spec(a)  # raises if unknown
        setattr(self, f"{method_name}_inputs", names)

    def _method_input_names(self, method_name: str) -> list:
        """The ordered extra-input names a method consumes (empty if none)."""
        return list(getattr(self, f"{method_name}_inputs", []))

    def _method_extra_examples(self, method_name: str) -> tuple:
        """Zero example tensors for a method's extra inputs, in declared order."""
        return tuple(self._extra_example(a)
                     for a in self._method_input_names(method_name))

    def _prime_streaming_buffers(self, method_name: str, example: tuple) -> list:
        """Allocate a method's lazily-created streaming buffers before tracing.

        Streaming nets (e.g. ``cached_conv``: ``CachedPadding1d.pad`` /
        ``CachedConvTranspose1d.cache``) register their state buffers lazily, on the
        first ``forward`` — not in ``__init__``. If a method is exported without ever
        running a forward, those buffers do not exist at trace time, so they never
        become persistent ExecuTorch mutable buffers and the per-call state is lost
        (each block runs cold -> clicks at block boundaries).

        Run one warm-up forward (under ``no_grad`` so the buffers are created grad-free
        — otherwise the CoreML partitioner later fails to deepcopy the lowered program),
        then zero whatever buffers it just created so their initial state is a defined
        zero (the warm-up fills them with its own tail). Detecting *newly created*
        buffers keeps this mechanism-agnostic — no dependency on cached_conv's buffer
        names. A no-op for models with no lazy buffers (nothing new is created).

        Returns the fully-qualified names of the buffers it created, so the caller can
        ask ExecuTorch to serialize their zeroed initial state (InitializedMutableBufferPass);
        without that the runtime starts them from undefined memory — a first-block click.
        """
        before = {n for n, _ in self.named_buffers()}
        with torch.no_grad():
            getattr(self, method_name)(*example)
        created = []
        for n, buf in self.named_buffers():
            if n not in before:
                buf.zero_()
                created.append(n)
        return created

    # --------------------------------------------------------------- tokenizer

    def register_tokenizer(self,
                           tokenizer_file: str,
                           max_length: int,
                           pad_token: str = "<pad>",
                           padding_side: str = "right",
                           ids_key: str = "input_ids",
                           mask_key: str = "attention_mask") -> None:
        """Register a HuggingFace fast-tokenizer to bundle next to the ``.pte`` (per-model).

        Available on both exporters: a model that takes a text prompt feeds it as
        token ``condition`` inputs (ids + attention mask), so a ``neural.live~`` model
        may use one as well as ``neural.gen~``. On export the ``tokenizer_file`` (a
        ``tokenizer.json``) is copied to ``<stem>.tokenizer.json`` and a
        ``<stem>.tokenizer.config.json`` is written; the bundle is standalone (the
        model sidecar does not reference it) and shared by every method.

        ``ids_key`` / ``mask_key`` MUST equal the token-condition input names of
        whichever method consumes the tokens, so ``neural.tokenizer``'s output
        dictionary keys line up with the model inputs. See :class:`Tokenizer`.
        """
        self._tokenizer = Tokenizer(tokenizer_file, max_length, pad_token,
                                    padding_side, ids_key, mask_key)

    def _write_tokenizer_files(self, pte_path: str) -> None:
        """Write the registered tokenizer bundle beside the ``.pte`` (no-op if none)."""
        if self._tokenizer is not None:
            self._tokenizer.write_files(pte_path)

    # --------------------------------------------------------------------- seed

    def set_seed(self, seed: int) -> None:
        """Set the default RNG seed for the model's ``noise`` inputs (per-model).

        Emitted as the top-level sidecar ``seed`` for both kinds; the host may
        override it at run time (``neural.gen~`` / ``neural.live~`` expose a
        settable ``seed`` control)."""
        self._seed = int(seed)

    # ------------------------------------------------------------------ output

    def _write_sidecar_json(self, pte_path: str, metadata: dict) -> str:
        json_path = os.path.splitext(pte_path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Wrote sidecar metadata to {json_path}")
        return json_path


class _MethodWrapper(torch.nn.Module):
    """Exposes one registered method of ``parent`` as ``forward`` for export.

    The positional inputs are forwarded verbatim, so this serves both the live
    signature (audio tensor followed by the method's extra inputs) and the gen
    signature (the method's inputs only). The method named ``method_name`` may be
    defined on a wrapped model or on the exporter subclass itself.
    """

    def __init__(self, parent: torch.nn.Module, method_name: str):
        super().__init__()
        self._parent = parent
        self._method_name = method_name

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return getattr(self._parent, self._method_name)(*inputs)
