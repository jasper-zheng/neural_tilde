"""``neural_tilde.LiveModule`` — export a streaming PyTorch model to a ``neural.live~`` ExecuTorch ``.pte``.

The audio-rate exporter; its offline-generator counterpart is :class:`neural_tilde.GenModule`
(``neural.gen~``). Subclass it, ``register_method`` each processing path, and register any extra
inputs by name with their role. Each method's ``forward`` takes one ``signal`` input (the
per-block multi-channel audio) followed by the extra inputs it consumes, each tagged:

    ``attribute`` — a scalar ``[1]`` control the host exposes as a Max attribute;
    ``noise``     — a tensor the host fills with per-block ``N(0,1)``;
    ``condition`` — a held control vector the patch supplies by list / dictionary.

``export_to_pte`` writes the ``.pte`` + a ``kind:"live"`` metadata JSON; the output is a single
``signal`` tensor. A method may carry mutable state that persists across blocks (e.g.
``cached_conv`` caches) so consecutive blocks join click-free. See ``examples/export_live_example.py``.

Usage::

    class MyModel(LiveModule):
        def __init__(self):
            super().__init__()
            self.net = ...                                   # any nn.Module
        def forward(self, x, gain):                          # x: [batch, 2, T] signal
            return self.net(x) * gain.reshape(1, 1, 1)

    model = MyModel()
    model.register_attribute("gain", 1.0, 0.0, 2.0, "output gain")
    model.register_method("forward", in_channels=2, in_ratio=1, out_channels=2, out_ratio=1,
                          inputs=["gain"])
    model.export_to_pte("model.pte", delegate="xnnpack", buffer_size=4096)
"""

import logging
from typing import List, Optional, Sequence, Union

import torch

from ._exporter_base import _ExporterBase, _MethodWrapper
from ._lowering import (_executorch_version, _make_init_mutable_passes,
                        _make_partitioner)


class LiveModule(_ExporterBase):

    def __init__(self) -> None:
        super().__init__()
        # Live exposes its attributes through get_attributes() (TorchScript path);
        # the spec tables + _methods registry come from _ExporterBase.
        self._attributes = ["none"]

    # ------------------------------------------------------------ extra inputs

    def register_attribute(self,
                           attribute_name: str,
                           default: Union[bool, int, float],
                           minimum: Optional[float] = None,
                           maximum: Optional[float] = None,
                           description: str = ""):
        """Register a numeric scalar control fed to a method as a ``[1]`` forward input.

        The host — ``neural.live~`` — owns the value as a Max attribute (inspector /
        ``@box-arg`` / get-set) and feeds the current value every block, clamped to
        ``[minimum, maximum]``. Numeric only: ``float`` exports as ``float32``,
        ``bool``/``int`` as ``int64``.

        Args:
            attribute_name: name of the attribute (= the Max attribute name)
            default: default value; its Python type selects the export dtype
            minimum, maximum: optional clamp range (Max clamps on set)
            description: shown as the attribute's inspector label
        """
        super().register_attribute(attribute_name, default, minimum, maximum,
                                   description)
        self._attributes.append(attribute_name)

    # ------------------------------------------------------------------ method

    def register_method(
        self,
        method_name: str,
        in_channels: int,
        in_ratio: int,
        out_channels: int,
        out_ratio: int,
        input_labels: Optional[Sequence[str]] = None,
        output_labels: Optional[Sequence[str]] = None,
        test_method: bool = False,
        test_buffer_size: int = 8192,
        inputs: Sequence[str] = (),
        attributes: Sequence[str] = (),
    ):
        """Register a class method as usable by neural.live~.

        The method's ``forward`` takes the audio tensor (``[batch, in_channels, T_in]``)
        followed by one tensor per extra input it consumes, in the order listed by
        ``inputs`` — each a previously-registered attribute / noise / condition — and
        returns a single ``[batch, out_channels, T_out]`` tensor.

        Args:
            method_name: name of the method to register
            in_channels/in_ratio/out_channels/out_ratio: audio I/O geometry (the
                ``signal`` role); the audio tensors are [batch, channels, buffer_size/ratio]
            input_labels/output_labels: per-channel inlet/outlet labels
            test_method: run a forward pass during registration to validate shapes
            test_buffer_size: duration of the test buffer
            inputs: ordered names of the extra inputs (attributes/noise/conditions)
                the forward consumes after the audio tensor
            attributes: deprecated shorthand for an attribute-only ``inputs`` list
        """
        logging.info(f'Registering method "{method_name}"')

        # The method's ordered extra (non-signal) inputs (validated + stored by
        # the base; appended to self._methods at the end once registration succeeds).
        extras = list(inputs) if inputs else list(attributes)
        self._declare_method(method_name, extras)
        extra_examples = list(self._method_extra_examples(method_name))
        self.register_buffer(
            f'{method_name}_params',
            torch.tensor([
                in_channels,
                in_ratio,
                out_channels,
                out_ratio,
            ]))

        if input_labels is None:
            input_labels = [
                f"(signal) model input {i}" for i in range(in_channels)
            ]
        if len(input_labels) != in_channels:
            raise ValueError(
                (f"Method {method_name}, expected "
                 f"{in_channels} input labels, got {len(input_labels)}"))
        setattr(self, f"{method_name}_input_labels", list(input_labels))

        if output_labels is None:
            output_labels = [
                f"(signal) model output {i}" for i in range(out_channels)
            ]
        if len(output_labels) != out_channels:
            raise ValueError(
                (f"Method {method_name}, expected "
                 f"{out_channels} output labels, got {len(output_labels)}"))
        setattr(self, f"{method_name}_output_labels", list(output_labels))

        if test_method:
            logging.info(f"Testing method {method_name} with neural.live~ API")
            x = torch.zeros(1, in_channels, test_buffer_size // in_ratio)
            # Run under no_grad so any lazily-allocated mutable/cache buffers are
            # created grad-free. Otherwise the buffers carry an autograd relationship
            # that later makes the CoreML partitioner fail to deepcopy the lowered
            # program ("Only Tensors created explicitly by the user ... deepcopy").
            with torch.no_grad():
                y = getattr(self, method_name)(x, *extra_examples)

            if len(y.shape) != 3:
                raise ValueError(
                    ("Output tensor must have exactly 3 dimensions, "
                     f"got {len(y.shape)}"))
            if y.shape[0] != 1:
                raise ValueError(
                    f"Expecting single batch output, got {y.shape[0]}")
            if y.shape[1] != out_channels:
                raise ValueError((
                    f"Wrong number of output channels for method \"{method_name}\", "
                    f"expected {out_channels} got {y.shape[1]}"))
            if y.shape[2] != test_buffer_size // out_ratio:
                raise ValueError(
                    (f"Wrong output length for method \"{method_name}\", "
                     f"expected {test_buffer_size//out_ratio} "
                     f"got {y.shape[2]}"))
            if y.dtype != torch.float:
                raise ValueError(f"Output tensor must be of type float, got {y.dtype}")

            logging.info(f"Testing method {method_name} with mc.neural.live~ API")
            x = torch.zeros(4, in_channels, test_buffer_size // in_ratio)
            with torch.no_grad():
                y = getattr(self, method_name)(x, *extra_examples)

            if len(y.shape) != 3:
                raise ValueError(
                    ("Output tensor must have exactly 3 dimensions, "
                        f"got {len(y.shape)}"))
            if y.shape[0] != 4:
                raise ValueError(
                    f"Expecting 4 batch output, got {y.shape[0]}")
            if y.shape[1] != out_channels:
                raise ValueError((
                    f"Wrong number of output channels for method \"{method_name}\", "
                    f"expected {out_channels} got {y.shape[1]}"))
            if y.shape[2] != test_buffer_size // out_ratio:
                raise ValueError(
                    (f"Wrong output length for method \"{method_name}\", "
                        f"expected {test_buffer_size//out_ratio} "
                        f"got {y.shape[2]}"))
            logging.info((f"Added method \"{method_name}\" "
                            f"tested with buffer size {test_buffer_size}"))

        else:
            logging.warn(f"Added method \"{method_name}\" without testing it.")

        self._methods.append(method_name)

    @torch.jit.export
    def get_methods(self):
        return self._methods

    @torch.jit.export
    def get_attributes(self) -> List[str]:
        return self._attributes

    def export_to_ts(self, path):
        self.eval()
        scripted = torch.jit.script(self)
        scripted.save(path)

    # ------------------------------------------------------------------ export

    def _method_inputs(self, name: str, in_channels: int, in_ratio: int) -> list:
        """The ordered metadata ``inputs`` list for a method: the signal entry first,
        then one entry per registered extra input the method consumes."""
        signal = {"name": "signal", "role": "signal",
                  "channels": int(in_channels), "ratio": int(in_ratio),
                  "labels": list(getattr(self, f"{name}_input_labels"))}
        extras = [dict(self._extra_spec(a))
                  for a in getattr(self, f"{name}_inputs", [])]
        return [signal] + extras

    def _nn_metadata(self, buffer_size: int, delegate: str,
                     executorch_version: str) -> dict:
        """Build the ``kind:"live"`` metadata JSON dict (see EXECUTORCH_PROTOCOL.md).

        Pure metadata: depends only on the registered methods/specs, so it can be built
        (and tested) without executorch installed.
        """
        methods = {}
        for name in self._methods:
            params = getattr(self, f"{name}_params").tolist()
            in_channels, in_ratio, out_channels, out_ratio = (int(p)
                                                              for p in params)
            for ratio in (in_ratio, out_ratio):
                if buffer_size % ratio != 0:
                    raise ValueError(
                        (f"buffer_size {buffer_size} is not a multiple of "
                         f'ratio {ratio} for method "{name}"'))
            methods[name] = {
                "inputs": self._method_inputs(name, in_channels, in_ratio),
                "output": {
                    "name": "audio", "role": "signal",
                    "channels": out_channels, "ratio": out_ratio,
                    "labels": list(getattr(self, f"{name}_output_labels")),
                },
                "batch": 1,
                "dynamic_time": False,
            }
        return {
            "kind": "live",  # REQUIRED discriminator (neural.live~); generative => "gen"
            "executorch_version": executorch_version,
            "delegate": delegate,
            "buffer_size": int(buffer_size),
            "seed": self._seed,  # default RNG seed for noise inputs (host-overridable)
            "methods": methods,
            "attributes": [],
        }

    def export_to_pte(self,
                      path: str,
                      delegate: str = "xnnpack",
                      buffer_size: int = 4096,
                      batch: int = 1,
                      strict: bool = False,
                      warmup: bool = True) -> str:
        """Export the registered methods to an ExecuTorch ``.pte`` + metadata JSON.

        Each registered method becomes one ExecuTorch method taking a
        ``[batch, in_channels, buffer_size // in_ratio]`` float tensor followed by one
        tensor per extra input it declared (in order) and returning
        ``[batch, out_channels, buffer_size // out_ratio]``. See EXECUTORCH_PROTOCOL.md.

        A method MAY carry internal mutable state that persists across calls (e.g. a
        streaming conv net); the host loads one instance per object and reuses it.

        Args:
            path: output path; ``.pte`` is appended if missing.
            delegate: "xnnpack" (default), "coreml", "mlx", or "portable".
            buffer_size: audio-rate block size to export at; must be a multiple
                of every in_ratio/out_ratio.
            batch: fixed batch dimension baked into the program.
            strict: ``torch.export`` tracing mode (see torch.export).
            warmup: run one no_grad forward per method before tracing so lazily
                allocated streaming buffers (e.g. ``cached_conv`` caches) exist as
                real mutable buffers and persist across calls — without this a
                streaming model is exported stateless and clicks at every block
                boundary. Default on; set ``False`` only if the model has no
                streaming state or has already primed/zeroed its caches manually.
        """
        if not path.endswith(".pte"):
            path = path + ".pte"
        self.eval()

        # Build metadata first so a bad ratio/label fails before the heavy lift.
        metadata = self._nn_metadata(buffer_size, delegate,
                                     _executorch_version())

        try:
            from executorch.exir import to_edge_transform_and_lower
        except ImportError as e:
            raise ImportError(
                "export_to_pte requires the 'executorch' package. Install it "
                "(e.g. `pip install executorch`) or build it from source.") from e

        partitioner = _make_partitioner(delegate)

        method_graphs = {}
        primed_buffers = []  # FQNs of lazily-allocated streaming buffers (caches)
        for name in self._methods:
            in_channels, in_ratio = metadata["methods"][name]["inputs"][0][
                "channels"], metadata["methods"][name]["inputs"][0]["ratio"]
            # Audio tensor, then one example tensor per extra input, in order.
            example = (torch.zeros(batch, in_channels,
                                   buffer_size // in_ratio),
                       ) + self._method_extra_examples(name)
            wrapper = _MethodWrapper(self, name).eval()
            # Allocate (and zero) lazy streaming buffers BEFORE tracing, so they are
            # captured as persistent mutable state — otherwise the model streams cold
            # and clicks at block boundaries (see _prime_streaming_buffers).
            if warmup:
                primed_buffers += self._prime_streaming_buffers(name, example)
            with torch.no_grad():
                method_graphs[name] = torch.export.export(wrapper, example,
                                                          strict=strict)

        # Serialize a defined (zeroed) initial state for the primed streaming buffers
        # so the runtime starts them at zero rather than undefined memory (first-block
        # click); no-op when nothing was primed. CoreML zero-inits its taken-over state
        # regardless, so this matters mainly for the xnnpack/portable runtime.
        transform_passes = (_make_init_mutable_passes(primed_buffers)
                            if warmup else None)
        lowered = to_edge_transform_and_lower(method_graphs,
                                              transform_passes=transform_passes,
                                              partitioner=partitioner)
        executorch_program = lowered.to_executorch()
        with open(path, "wb") as f:
            f.write(executorch_program.buffer)
        logging.info(f"Wrote ExecuTorch program to {path}")

        self._write_metadata_json(path, metadata)
        # If the model takes a text prompt, emit the standalone tokenizer bundle
        # beside the .pte (no-op when no tokenizer was registered).
        self._write_tokenizer_files(path)
        return path
