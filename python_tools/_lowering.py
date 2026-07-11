"""Shared ExecuTorch lowering helpers for the ``neural_tilde`` exporters.

These are delegate/partitioner utilities used by both the audio-rate exporter
(:class:`neural_tilde.LiveModule`, ``neural.live~``) and the generative exporter
(:class:`neural_tilde.GenModule`, ``neural.gen~``)."""

import os
from typing import Optional


def _executorch_version() -> str:
    try:
        import executorch
        return getattr(executorch, "__version__", "unknown")
    except Exception:
        return "unknown"


def _make_init_mutable_passes(buffer_fqns):
    """Transform pass that serializes a *defined* (zeroed) initial state for the given
    lazily-allocated streaming buffers (cached_conv caches), so the runtime starts them
    at zero instead of undefined memory — otherwise the very first block reads garbage
    (a startup click). This is the companion to priming/zeroing the caches before export
    (see ``_ExporterBase._prime_streaming_buffers``).

    ``buffer_fqns`` are module FQNs (e.g. ``net.0.cache.pad``); the pass matches the
    mangled graph placeholder names by substring, so they are sanitized to that form
    (dots -> underscores, e.g. ``net_0_cache_pad`` matches ``b__parent_net_0_cache_pad``).

    Returns ``None`` when there is nothing to initialize or the pass is unavailable (the
    priming alone still restores per-block persistence; only the first-block init is lost).
    """
    if not buffer_fqns:
        return None
    try:
        from executorch.exir.passes.init_mutable_pass import (
            InitializedMutableBufferPass)
    except ImportError:
        import logging
        logging.warning(
            "InitializedMutableBufferPass unavailable; streaming buffers will start "
            "from undefined state (possible first-block click). Update executorch.")
        return None
    patterns = sorted({fqn.replace(".", "_") for fqn in buffer_fqns})
    return [InitializedMutableBufferPass(patterns)]


def _make_partitioner(delegate: str, coreml_disable_weight_sharing: bool = False,
                      coreml_compute_units: Optional[str] = None,
                      method_name: Optional[str] = None):
    """Return the partitioner list for ``to_edge_transform_and_lower``.

    ``method_name`` (Metal only) names the method this partitioner is built for — the Metal
    (AOTInductor) backend compiles one ``.so`` per method and keys the embedded blob on it
    (``<method>_so_blob``), so multi-method exports must build one partitioner per method via
    a ``{name: [...]}`` partitioner dict. Ignored by the other delegates.

    ``coreml_disable_weight_sharing`` (CoreML only) sets the multimethod weight-sharing
    strategy to DISABLED so methods are lowered independently. The default POSITIONAL
    strategy requires every method to produce the *same* number of CoreML partitions and
    errors otherwise - which happens when methods differ structurally (e.g. a migrated RAVE
    where ``decode``'s FFT noise synth partitions differently than ``encode``). Off by default.

    ``coreml_compute_units`` (CoreML only) names the ``coremltools.ComputeUnit`` to compile
    for — one of ``"ALL"`` / ``"CPU_ONLY"`` / ``"CPU_AND_GPU"`` / ``"CPU_AND_NE"`` (you pick a
    preset; coremltools has no arbitrary device union). Resolution precedence: this explicit
    argument, then the ``NN_COREML_CU`` env var, then the default ``"ALL"``. ``migrate.py``
    passes ``"CPU_AND_NE"`` to keep RAVE bit-correct (its GPU path miscomputes); other models
    may benefit from the GPU under ``"ALL"`` — validate on target."""
    if delegate == "portable":
        return []
    if delegate == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner)
        return [XnnpackPartitioner()]
    if delegate == "mlx":
        # It does persist cached_conv streaming state across execute(). Op coverage is narrower
        # (e.g. complex/FFT unsupported; some gen convs need decompose_conv=True).
        from executorch.backends.mlx.partitioner import MLXPartitioner
        return [MLXPartitioner()]
    if delegate == "mps":
        # MPS (Apple GPU via MPSGraph).DEPRECATED in executorch >=1.4 (migrate to coreml/mlx).
        # It does persist cached_conv streaming state across execute().
        # Some ops are unsupported and abort at runtime, need to validate on every exported model.
        # FLOAT32 by default to avoid fp16 audio degradation; set NN_MPS_FP16=1 to opt into fp16.
        from executorch.backends.apple.mps.partition.mps_partitioner import (
            MPSPartitioner)
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        use_fp16 = os.environ.get("NN_MPS_FP16", "0") == "1"
        return [MPSPartitioner(
            compile_specs=[CompileSpec("use_fp16", bytes([use_fp16]))])]
    if delegate == "coreml":
        # CoreML takes torch mutable buffers over as native Core ML state, so a
        # stateful model's internal state persists across execute() calls
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner
        # Compute unit defaults to ALL (CoreML picks CPU/GPU/ANE per op). For RAVE models, CPU_AND_NE is recommended; 
        # FLOAT32 avoids fp16 audio degradation
        cu_name = (coreml_compute_units
                   or os.environ.get("NN_COREML_CU")
                   or "ALL")
        try:
            compute_unit = ct.ComputeUnit[cu_name]
        except KeyError:
            raise ValueError(
                f"unknown CoreML compute unit '{cu_name}'; expected one of "
                f"{[e.name for e in ct.ComputeUnit]}")
        print(f"CoreML compute unit: {compute_unit.name}")
        specs = CoreMLBackend.generate_compile_specs(
            compute_unit=compute_unit,
            minimum_deployment_target=ct.target.iOS18,  # gates Core ML state
            compute_precision=ct.precision.FLOAT32,
        )
        if coreml_disable_weight_sharing:
            from executorch.backends.apple.coreml.compiler.coreml_preprocess import (
                MULTIMETHOD_WEIGHT_SHARING_STRATEGY)
            specs.append(
                CoreMLBackend.generate_multimethod_weight_sharing_strategy_compile_spec(
                    MULTIMETHOD_WEIGHT_SHARING_STRATEGY.DISABLED))
        return [CoreMLPartitioner(compile_specs=specs,
                                  take_over_mutable_buffer=True)]
    if delegate == "metal":
        # NOTE: Metal (AOTInductor) is an EXPERIMENTAL delegate. Unlike the op-handler
        # backends, it delegates the WHOLE graph to torch._inductor's AOT compile (mps
        # device), embedding a compiled, ad-hoc-signed .so in the .pte -> the heavy compile
        # happens at EXPORT (offline), giving a fast (~dlopen) load instead of CoreML's
        # load-time .mlmodelc compile. Needs the ExecuTorch Metal runtime
        # (EXECUTORCH_BUILD_METAL) + macOS. Each method is one AOTI partition, keyed by its
        # method-name compile spec (see ``method_name`` above). See _metal_* helpers below
        # for the AOTI-specific graph passes / env the exporters apply alongside this.
        from executorch.backends.apple.metal.metal_backend import MetalBackend
        from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        _patch_metal_aoti_options()  # tolerate a missing torchao-MPS c-shim build
        name = method_name or "forward"
        return [MetalPartitioner([
            MetalBackend.generate_method_name_compile_spec(name),
            CompileSpec("codesign_identity", b"-")])]  # ad-hoc sign the embedded .so
    raise ValueError(f"unknown delegate '{delegate}' "
                     "(expected 'xnnpack', 'coreml', 'mlx', 'mps', 'metal', or 'portable')")


# --------------------------------------------------------------------- Metal (AOTInductor)
# The Metal delegate needs a few AOTInductor-specific accommodations that the op-handler
# backends don't. They are kept here (not baked into the model) so a Metal export is a pure
# lowering concern: two per-method graph passes (const-fold + split decomposition), a
# tolerant compile-options patch (torchao-MPS may be absent), and a libomp link-env nudge.

_metal_options_patched = False


def _patch_metal_aoti_options() -> None:
    """Make ``MetalBackend.get_aoti_compile_options`` tolerate a missing torchao-MPS build.

    The stock method hard-imports ``torchao.experimental.ops.mps.cshim`` (4-bit quantized
    linear C-shims), which needs ``libtorchao_ops_mps_aten.dylib``
    (``TORCHAO_BUILD_EXPERIMENTAL_MPS=1``). Audio models (RAVE / CoDiCodec) use none of those
    ops. We try the stock method first and only install a fallback when it raises ImportError,
    so a torchao-enabled environment is unaffected. Idempotent."""
    global _metal_options_patched
    if _metal_options_patched:
        return
    from executorch.backends.apple.metal.metal_backend import MetalBackend
    _stock = MetalBackend.get_aoti_compile_options.__func__

    def _tolerant(cls, compile_specs):
        try:
            return _stock(cls, compile_specs)
        except Exception as e:
            # torchao-MPS may be absent in several ways: ImportError (module missing),
            # RuntimeError (libtorchao_ops_mps_aten.dylib not found), etc. The fallback
            # configs don't need torchao, so tolerate any failure of the stock path.
            import importlib
            import logging
            logging.warning(
                "torchao MPS c-shims unavailable (%s: %s); exporting Metal without them "
                "(fine for models with no torchao quantized ops).",
                type(e).__name__, e)
            inductor_configs = {
                "aot_inductor.link_libtorch": False,
                "aot_inductor.package": True,
                "aot_inductor.package_constants_in_so": False,
                "aot_inductor.package_constants_on_disk_format": "binary_blob",
                "max_autotune": True,
                "padding_stride_threshold": float("inf"),
            }
            custom_c_shims = {}
            # keep the pure-Metal custom shims that don't need torchao, if they import
            for mod, attr in (
                ("executorch.backends.apple.metal.ops.gather_qmv", "metal_gather_qmv_c_shim"),
                ("executorch.backends.apple.metal.ops.gated_delta_rule",
                 "metal_gated_delta_rule_c_shim"),
            ):
                try:
                    custom_c_shims.update(getattr(importlib.import_module(mod), attr))
                except Exception:
                    pass
            inductor_configs["aot_inductor.custom_ops_to_c_shims"] = custom_c_shims
            return inductor_configs

    MetalBackend.get_aoti_compile_options = classmethod(_tolerant)
    _metal_options_patched = True


def _metal_constant_fold(exported):
    """Fold constants so a *constant*-base ``aten.pow`` doesn't trip inductor's MPS codegen
    ``StopIteration`` bug (``torch/_inductor/lowering.py`` ``pow()``)."""
    from executorch.exir.passes.constant_prop_pass import constant_prop_pass
    return constant_prop_pass(exported)


def _metal_decompose_splits(exported):
    """Decompose ``split``/``chunk`` to ``as_strided`` views BEFORE ``to_edge`` functionalizes
    them into ``aten::split_copy.Tensor`` — which has no Metal C-shim and otherwise aborts the
    AOTI lowering with "missing fallback kernels"."""
    import torch
    from torch._decomp import get_decompositions
    table = get_decompositions([
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.chunk.default,
    ])
    return exported.run_decompositions(table)


def _metal_graph_transforms():
    """Metal/AOTI per-method ExportedProgram transforms, in order: const-fold (the pow bug),
    then split/chunk decomposition (split_copy). Appended after any user ``graph_transforms``."""
    return [_metal_constant_fold, _metal_decompose_splits]


def _metal_setup_link_env() -> None:
    """Best-effort: put a ``libomp.dylib`` directory on ``LIBRARY_PATH`` so inductor's ``-lomp``
    link step (AOTI codegen) resolves. Metal/AOTI codegen emits ``-lomp`` and conda/venv ET
    environments often lack a bare libomp on the linker search path. Looks in the active torch
    install's ``lib`` dir and ``$CONDA_PREFIX/lib``. No-op (with a warning pointing at Build.md)
    if none is found — no hard-coded paths."""
    import logging
    candidates = []
    try:
        import torch
        candidates.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "lib"))
    for d in candidates:
        if os.path.exists(os.path.join(d, "libomp.dylib")):
            cur = os.environ.get("LIBRARY_PATH", "")
            if d not in cur.split(os.pathsep):
                os.environ["LIBRARY_PATH"] = d + (os.pathsep + cur if cur else "")
            logging.info("Metal export: added %s to LIBRARY_PATH for libomp", d)
            return
    logging.warning(
        "Metal export: no libomp.dylib found on the torch lib dir or $CONDA_PREFIX/lib; the "
        "AOTI link step may fail with \"library 'omp' not found\". Set LIBRARY_PATH manually "
        "(see Build.md).")
