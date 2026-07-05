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
                      coreml_compute_units: Optional[str] = None):
    """Return the partitioner list for ``to_edge_transform_and_lower``.

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
        # NOTE: MLX is an experimental delegate.
        # It does persist cached_conv streaming state across execute(). Op coverage is narrower
        # (e.g. complex/FFT unsupported; some gen convs need decompose_conv=True).
        from executorch.backends.mlx.partitioner import MLXPartitioner
        return [MLXPartitioner()]
    if delegate == "mps":
        # MPS (Apple GPU via MPSGraph). Experimental, and DEPRECATED in executorch >=1.4 (migrate to coreml/mlx there).
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
        # stateful model's internal state persists across execute() calls; needs
        # macOS15+/iOS18+ at runtime.
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner
        # Compute unit defaults to ALL (CoreML picks CPU/GPU/ANE per op). For RAVE models, CPU_AND_NE is recommended; 
        # FLOAT32 avoids fp16 audio degradation. Precedence:
        # explicit arg > NN_COREML_CU env var > "ALL".
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
    raise ValueError(f"unknown delegate '{delegate}' "
                     "(expected 'xnnpack', 'coreml', 'mlx', 'mps', or 'portable')")
