"""``neural_tilde.GenModule`` — export a PyTorch generator to a ``neural.gen~`` ExecuTorch ``.pte``.

The offline-generator exporter; its audio-rate counterpart is :class:`neural_tilde.LiveModule`
(``neural.live~``). Wrap a ``torch.nn.Module``, register each input **by name** with its role
(``condition`` / ``attribute`` / ``noise`` / ``buffer``), then ``register_method`` one or more
generation paths — each listing, in ``forward`` order, the inputs it consumes and the audio it
produces. ``export_to_pte`` writes the ``.pte`` + a ``kind:"gen"`` metadata JSON (plus, if a
tokenizer is registered, the ``.tokenizer.json`` / ``.tokenizer.config.json`` bundle the
``neural.tokenizer`` object loads). A model may expose several methods (e.g. ``prompt2audio`` +
``audio2audio``); the host picks one by name at load time
(``[neural.gen~ model.pte audio2audio]``). 

Usage::

    gm = GenModule(my_pipeline)                 # method M resolves to my_pipeline.M(*inputs)
    gm.register_condition("input_ids", [1, 256], "int64")
    gm.register_condition("attention_mask", [1, 256], "int32")
    gm.register_attribute("seconds_total", 3.0, 0.0, 384.0, "Length in seconds to condition on.")
    gm.register_noise("x", [1, 256, 32])
    gm.register_noise("noises", [7, 1, 256, 32])
    gm.register_buffer_input("init_audio", channels=2, length=131072, sample_rate=44100)
    gm.register_method("prompt2audio",
                       inputs=["input_ids", "attention_mask", "seconds_total", "x", "noises"],
                       out_channels=2, out_length=131072, out_sample_rate=44100)
    gm.register_method("audio2audio",
                       inputs=["init_audio", "seconds_total", "x", "noises"],
                       out_channels=2, out_length=131072, out_sample_rate=44100)
    gm.register_tokenizer("tokenizer.json", max_length=256)
    gm.set_seed(0)
    gm.export_to_pte("model.pte", delegate="mlx", decompose_conv=True)
"""

import logging
from typing import Callable, Optional, Sequence

import torch
from torch.export import ExportedProgram

from ._exporter_base import _ExporterBase, _MethodWrapper
from ._lowering import _executorch_version, _make_partitioner

# Roles understood by the gen host (see EXECUTORCH_PROTOCOL.md §3 and GenRunner):
#   condition  — externally supplied (tokens/masks), matched by name
#   attribute  — scalar [1] control the host exposes as a Max attribute
#   noise      — host fills with seeded N(0,1)
#   buffer     — host-supplied init waveform read from a Max buffer~ (resampled/cropped to spec)
# Each method's output is always a single audio tensor (role "audio").
# All audio tensors (the "buffer" input + the "audio" output) are channel-major [1, C, L].
# register_condition / register_attribute / register_noise come from _ExporterBase.


class GenModule(_ExporterBase):

    def __init__(self, model: Optional[torch.nn.Module] = None) -> None:
        """Wrap ``model`` (whose methods consume the registered inputs in order).

        Pass ``model`` to export by composition (the common case — an already-built
        pipeline; a registered method ``M`` resolves to ``model.M(*inputs)``). Leave it
        ``None`` and subclass, defining each registered method on the subclass, to export
        by inheritance.

        Args:
            model: the module to export; each registered method is an attribute of it
                taking the registered inputs positionally, in declared order, and
                returning the audio tensor.
        """
        super().__init__()
        self._model = model
        self._outputs = {}              # method name -> output spec dict
        # register_tokenizer / _write_tokenizer_files and set_seed / _seed live on
        # _ExporterBase (the tokenizer bundle and seed are shared by both exporters).

    # ------------------------------------------------------------ extra inputs

    def register_buffer_input(self,
                              name: str,
                              channels: int,
                              length: int,
                              sample_rate: int = 44100,
                              description: str = "",
                              dtype: str = "float32") -> None:
        """Register a host-supplied init waveform read from a Max ``buffer~``
        (channel-major ``[1, channels, length]``).

        The host reads a ``buffer~``, resamples to ``sample_rate``, maps to ``channels`` and
        crops/pads to ``length`` before feeding (audio-to-audio variation).

        (The method is ``register_buffer_input`` rather than ``register_buffer`` to avoid
        shadowing ``torch.nn.Module.register_buffer``.)"""
        spec = {"name": name, "role": "buffer",
                "shape": [1, int(channels), int(length)], "dtype": dtype,
                "channels": int(channels), "length": int(length),
                "sample_rate": int(sample_rate)}
        if description:
            spec["description"] = description
        self._buffer_specs[name] = spec

    # ------------------------------------------------------------------ method

    def register_method(self,
                        method_name: str,
                        inputs: Sequence[str],
                        out_channels: int,
                        out_length: int,
                        out_sample_rate: int = 44100,
                        out_name: str = "audio",
                        out_dtype: str = "float32",
                        test_method: bool = False) -> None:
        """Register one generation path as a ``neural.gen~`` method.

        ``method_name`` must name a callable on the wrapped model (or, in the subclass
        export path, on ``self``) taking the inputs named by ``inputs`` positionally, in
        order — each a previously-registered condition / attribute / noise / buffer — and
        returning a single ``[1, out_channels, out_length]`` audio tensor.

        Args:
            method_name: name of the method (= the metadata ``methods{}`` key the host selects)
            inputs: ordered names of the inputs the method consumes
            out_channels/out_length/out_sample_rate: the audio output geometry
            out_name/out_dtype: output tensor name / dtype (``float32``)
            test_method: run one forward pass during registration to validate the output
                shape/dtype against the declaration
        """
        logging.info(f'Registering method "{method_name}"')
        self._declare_method(method_name, inputs)
        self._outputs[method_name] = {
            "name": out_name, "role": "audio", "dtype": out_dtype,
            "shape": [1, int(out_channels), int(out_length)],
            "channels": int(out_channels), "length": int(out_length),
            "sample_rate": int(out_sample_rate)}

        if test_method:
            logging.info(f"Testing method {method_name} with neural.gen~ API")
            target = self._model if self._model is not None else self
            with torch.no_grad():
                y = getattr(target, method_name)(*self._method_extra_examples(method_name))
            expected = (1, int(out_channels), int(out_length))
            if tuple(y.shape) != expected:
                raise ValueError(
                    (f'Method "{method_name}" output shape {tuple(y.shape)} != '
                     f"declared {expected}"))
            if y.dtype != torch.float:
                raise ValueError(
                    (f'Method "{method_name}" output must be float32, got {y.dtype}'))
            logging.info(f'Added method "{method_name}" (tested)')
        else:
            logging.warning(f'Added method "{method_name}" without testing it.')

        self._methods.append(method_name)

    # ------------------------------------------------------------------ export

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise NotImplementedError(
                "GenModule.forward: pass a model to the constructor, or subclass and "
                "define the registered method(s).")
        return self._model(*inputs)

    def _gen_metadata(self, delegate: str, executorch_version: str) -> dict:
        """Build the ``kind:"gen"`` metadata dict (see EXECUTORCH_PROTOCOL.md §3).

        Pure metadata — depends only on the registered specs, so it can be built (and
        tested) without executorch installed. One entry per registered method, each with
        its own ordered ``inputs`` + audio ``output``; ``seed`` is per-model."""
        if not self._methods:
            raise ValueError("register at least one method before export")
        methods = {}
        for name in self._methods:
            methods[name] = {
                "inputs": [dict(self._extra_spec(a))
                           for a in self._method_input_names(name)],
                "output": dict(self._outputs[name]),
                "batch": 1,
                "dynamic_time": False,
            }
        return {
            "kind": "gen",
            "executorch_version": executorch_version,
            "delegate": delegate,
            "seed": self._seed,
            "methods": methods,
            "attributes": [],
        }

    def export_to_pte(self,
                      path: str,
                      delegate: str = "mlx",
                      graph_transforms: Sequence[Callable[[ExportedProgram],
                                                          ExportedProgram]] = (),
                      decompose_conv: bool = False) -> str:
        """Export every registered method to an ExecuTorch ``.pte`` + metadata (+ tokenizer) JSON.

        Each registered method becomes one ExecuTorch method traced with one zero example
        tensor per declared input (in order), optionally graph-rewritten, then lowered
        through the ``delegate`` partitioner.

        Args:
            path: output path; ``.pte`` is appended if missing.
            delegate: "mlx" (default), "coreml", "xnnpack", or "portable".
            graph_transforms: callables applied to each method's ExportedProgram before
                lowering (e.g. model-specific graph surgery). Applied in order.
            decompose_conv: also apply the built-in ``decompose_convolution_nodes``
                (``aten.convolution`` → ``aten.conv1d``) — needed for conv models on the
                MLX backend.

        Returns:
            the path to the written ``.pte``.
        """
        if not path.endswith(".pte"):
            path = path + ".pte"
        if not self._methods:
            raise ValueError("register at least one method before export_to_pte")
        self.eval()

        # Build metadata first so a bad spec fails before the heavy lift.
        metadata = self._gen_metadata(delegate, _executorch_version())

        try:
            from executorch.exir import (EdgeCompileConfig,
                                         to_edge_transform_and_lower)
        except ImportError as e:
            raise ImportError(
                "export_to_pte requires the 'executorch' package. Install it "
                "(e.g. `pip install executorch`) or build it from source.") from e

        # Methods resolve to attributes of the wrapped model (composition) or self
        # (subclass override path); _MethodWrapper exposes one as forward for export.
        target = self._model if self._model is not None else self
        transforms = list(graph_transforms)
        if decompose_conv:
            transforms.append(decompose_convolution_nodes)

        method_graphs = {}
        for name in self._methods:
            wrapper = _MethodWrapper(target, name).eval()
            with torch.no_grad():
                exported = torch.export.export(
                    wrapper, self._method_extra_examples(name))
            for transform in transforms:
                exported = transform(exported)
            method_graphs[name] = exported

        lowered = to_edge_transform_and_lower(
            method_graphs,
            partitioner=_make_partitioner(delegate),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        executorch_program = lowered.to_executorch()
        with open(path, "wb") as f:
            f.write(executorch_program.buffer)
        logging.info(f"Wrote ExecuTorch program to {path}")

        self._write_metadata_json(path, metadata)
        self._write_tokenizer_files(path)
        return path


def decompose_convolution_nodes(exported: ExportedProgram) -> ExportedProgram:
    """Rewrite aten.convolution.default → aten.conv1d.default in the exported graph
    (safety net before the MLX partitioner, which only accepts aten.conv{1,2,3}d).

    aten.convolution: (input, weight, bias, stride, padding, dilation, transposed,
                       output_padding, groups)
    aten.conv1d:      (input, weight, bias, stride, padding, dilation, groups)
    """
    graph = exported.graph_module.graph
    n_replaced = 0
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target != torch.ops.aten.convolution.default:
            continue
        inp, weight, bias, stride, padding, dilation, transposed, output_padding, groups = node.args
        if transposed:
            continue  # leave transposed convolutions alone
        with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.aten.conv1d.default,
                args=(inp, weight, bias, stride, padding, dilation, groups),
            )
            new_node.meta = dict(node.meta)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        n_replaced += 1
    graph.lint()
    exported.graph_module.recompile()
    logging.info(f"Replaced {n_replaced} aten.convolution node(s) with aten.conv1d")
    return exported
