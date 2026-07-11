"""Migrate a RAVE TorchScript (``.ts``) model to a ``neural.live~`` ExecuTorch
``.pte`` + metadata JSON — no retraining, straight from the exported artifact.

    python -m neural_tilde.migrate percussion.ts --out percussion --delegate coreml

### Why this is not just ``LiveModule.export_to_pte``

``LiveModule`` exports an *eager* model via ``torch.export.export``. A ``.ts`` is a compiled
TorchScript program — ``torch.export`` cannot trace the TorchScript interpreter (both the
Dynamo and non-strict paths fail). So this tool takes the only viable bridge,
``torch._export.converter.TS2EPConverter``, and wraps it with the fixups a streaming RAVE
needs, then reuses the rest of the ``neural_tilde`` machinery (delegate partitioners,
mutable-buffer init pass, and the ``kind:"live"`` metadata via ``LiveModule``).

### What a RAVE ``.ts`` looks like

It is a cached_conv *streaming* model. Each ``nn~`` method (``encode`` / ``decode`` /
``forward`` / ``prior``) carries a ``<method>_params`` buffer holding
``[in_channels, in_ratio, out_channels, out_ratio]`` plus ``<method>_input_labels`` /
``<method>_output_labels``; the streaming caches are materialized buffers
(``*.cache.pad`` / ``*.cache`` / ``*.downsampling_delay.pad``).

### The pipeline (per method)

1. ``torch.jit.trace`` a thin wrapper of the scripted method (under ``no_grad`` — the cache
   ``copy_`` ops error with grad on), ``_jit_pass_inline``.
2. **Specialize data-dependent scalar reads** (``_specialize_scalar_reads``): RAVE decode reads
   ``self.latent_size.item()``, which the converter turns into an unbacked symint and chokes on
   (``GuardOnDataDependentSymNode``). The value is a fixed model constant → fold it to a literal.
2b. **Profile-guided ``prim::If`` specialization** (``_specialize_branches``): the converter turns
   *every* ``prim::If`` into ``torch.cond``, including ones it then can't build (RAVE's shape-driven
   length-alignment / bias-presence conditionals lift a non-tensor shape int as a cond operand).
   Since the branch taken is deterministic for a fixed ``buffer_size``, run the model once, bake
   each predicate to its observed constant, and DCE the dead branch → a branch-free graph, no
   ``torch.cond``. This is what lets ``decode`` / ``forward`` convert (not just ``encode``).
3. ``TS2EPConverter(...).convert()`` → an ``ExportedProgram``. A pre-retrace fixup
   (``_patched_conversion``) forces static shapes and rebuilds ``aten.slice.t`` shape-slices
   (``x.shape[:-2]``) as ``getitem`` reads, using fake-tensor ranks (``_populate_ranks``).
4. **(A) RNG** (``_handle_rng``): RAVE is stochastic (VAE reparameterization + the FFT noise
   synth use ``randn_like`` / ``rand_like`` / ``randn`` / ``rand``). ``--rng keep`` (default,
   for **CoreML**, which compiles RNG natively) leaves them; ``--rng zero`` makes the model
   deterministic for kernel-limited CPU runtimes.
5. **(B) Lift cache mutations** (``_lift_cache_mutations``): the converter emits
   ``aten.copy_.default(<cache buffer>, new)`` into lifted ``BUFFER`` inputs but does **not**
   record them as buffer mutations, so the lowered runtime would not persist the caches across
   blocks (→ clicks). Rewrite each into the ``torch.export`` buffer-mutation convention so they
   become persistent ExecuTorch ``MUTABLE_BUFFER`` state.

Then all methods are lowered together to one ``.pte`` and the metadata JSON is written.

### Per-method resilience

Migration is best-effort per method: encode / decode / forward / prior are each attempted
independently and a method is kept only if it converts, matches the original ``.ts`` output shape,
(optionally) survives noise-synth removal with its streaming intact, and lowers for the chosen
delegate. Methods that can't are skipped with a warning and the ``.pte`` ships with those that
succeed. Common skips:

* ``prior`` is an autoregressive ``prim::Loop`` (a generator, not a per-block streaming method)
  that TS2EPConverter can't handle — expected, and it isn't a ``neural.live~`` method anyway.
* A method whose control flow is genuinely *value*-dependent (not shape/config-driven) would be
  mis-specialized by the profile-guided branch folding; the output-shape backstop in
  :func:`migrate` catches that and skips it (re-export from the checkpoint instead).

### Backend op-support caveat

Two RAVE-specific gaps decide whether decode / forward survive:

* **RNG** (``randn_like`` / ``rand_like`` / ``randn`` / ``rand``): not in the core ATen opset,
  so the lowering needs them on its exception list (handled). **CoreML** compiles them; the
  stock xnnpack/portable CPU runtime has no ``rand*`` kernel, so use ``--rng zero`` there.
* **Complex / FFT** (``view_as_complex`` / ``fft_rfft`` / ``fft_irfft`` — RAVE's noise synth,
  in ``decode`` / ``forward``): **not supported by coremltools** (no ``complex64`` dtype) and
  not in the portable CPU kernel set. Methods using them cannot lower to CoreML and won't run
  on the CPU runtime, so they are skipped by default. ``encode`` (no FFT) migrates and runs.

  Pass ``--skip-noise-synth`` to **excise** the noise-synth branch: ``decode`` / ``forward``
  then lower and run on CoreML/CPU, at the cost of the synthesized noise texture (see
  :func:`_skip_noise_synth`). This is clean only when the noise synth is a *terminal additive
  branch* (e.g. percussion). When it rejoins mid-network before skip-connected, cached
  processing (e.g. VCTK), removing it disconnects the streaming caches — :func:`_eager_streams`
  detects this and the method is skipped rather than shipping a clicking model.

For full-fidelity streaming with the noise synth intact, re-export from the original RAVE
checkpoint through the eager ``LiveModule`` path (needs the ``.ckpt``, not just the ``.ts``).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import logging
import operator
from typing import Optional

import torch
from torch._export.converter import TS2EPConverter
from torch.export.graph_signature import (ExportGraphSignature, InputKind,
                                          OutputKind, OutputSpec, TensorArgument)

from ._lowering import (_executorch_version, _make_init_mutable_passes,
                        _make_partitioner)
from .live_module import LiveModule

# Methods a RAVE .ts may expose, in nn~ convention (probed via <method>_params).
_CANDIDATE_METHODS = ("encode", "decode", "forward", "prior")

# RAVE's stochastic ops (VAE reparameterization + FFT noise synth). Not in the core ATen
# opset, so the edge lowering needs them on its exception list; --rng zero replaces them.
_RNG_LIKE = (torch.ops.aten.randn_like.default, torch.ops.aten.rand_like.default)
_RNG_SIZED = (torch.ops.aten.randn.default, torch.ops.aten.rand.default)
_RNG_OPS = _RNG_LIKE + _RNG_SIZED


class _MethodWrapper(torch.nn.Module):
    """Expose one scripted method of the loaded ``.ts`` as a traceable ``forward``."""

    def __init__(self, scripted: torch.jit.ScriptModule, method: str) -> None:
        super().__init__()
        self._scripted = scripted
        self._method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return getattr(self._scripted, self._method)(x)


def _discover_methods(scripted: torch.jit.ScriptModule) -> dict:
    """Read ``[in_channels, in_ratio, out_channels, out_ratio]`` + labels for each method.

    ``get_methods()`` is not reachable on a loaded ``RecursiveScriptModule``, so probe for the
    ``<method>_params`` buffer that ``nn_tilde``/RAVE registers for every exported method."""
    methods = {}
    for name in _CANDIDATE_METHODS:
        if not hasattr(scripted, f"{name}_params"):
            continue
        params = [int(v) for v in getattr(scripted, f"{name}_params").tolist()]
        in_channels, in_ratio, out_channels, out_ratio = params
        in_labels = list(getattr(scripted, f"{name}_input_labels", []) or [])
        out_labels = list(getattr(scripted, f"{name}_output_labels", []) or [])
        methods[name] = {
            "in_channels": in_channels, "in_ratio": in_ratio,
            "out_channels": out_channels, "out_ratio": out_ratio,
            # Keep labels only if they match the declared channel counts; else let
            # LiveModule generate defaults (avoids a register_method length error).
            "input_labels": in_labels if len(in_labels) == in_channels else None,
            "output_labels": out_labels if len(out_labels) == out_channels else None,
        }
    return methods


def _resolve_attr(root: torch.nn.Module, value):
    """Follow a ``prim::GetAttr`` chain from ``value`` back to ``self``; return the attribute
    (or ``None`` if it isn't a clean attribute read)."""
    parts = []
    node = value.node()
    while node.kind() == "prim::GetAttr":
        parts.append(node.s("name"))
        node = node.input().node()
    if node.kind() != "prim::Param":
        return None
    obj = root
    try:
        for part in reversed(parts):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def _fold_constant_attrs(scripted: torch.nn.Module, graph) -> None:
    """Replace ``prim::GetAttr`` of a primitive (bool/int/float) module attribute with a literal.

    RAVE/cached_conv models branch heavily on fixed config flags (``isis`` encode has 251
    ``prim::If``). Those conditions read primitive attributes that constant propagation can't
    fold while they are opaque ``GetAttr`` nodes. Baking them as literals lets
    ``constant_propagation`` + ``dce`` collapse the dead branches (251 -> ~15), which both
    speeds up conversion and avoids many ``prim::If`` the converter can't handle. Tensors are
    left untouched (folding a 1-element tensor to a scalar would break tensor ops)."""
    for node in list(graph.findAllNodes("prim::GetAttr")):
        out = node.output()
        if not list(out.uses()):
            continue
        value = _resolve_attr(scripted, out)
        if isinstance(value, (bool, int, float)) and not torch.is_tensor(value):
            const = graph.insertConstant(value)
            const.node().moveBefore(node)
            out.replaceAllUsesWith(const)


def _specialize_scalar_reads(scripted: torch.nn.Module, graph) -> None:
    """Fold ``aten::item`` / ``aten::Int`` / ``aten::Bool`` / ``aten::Float`` on a constant
    1-element buffer to a Python literal.

    Two reasons: ``self.latent_size.item()`` (RAVE decode) would otherwise become an unbacked
    symint (``GuardOnDataDependentSymNode``); and ``aten::Bool(self.encoder.warmed_up)`` is the
    predicate of ``if warmed_up: z = z.detach()`` — a conditional latent assignment that
    TS2EPConverter cannot express as ``torch.cond`` ("Output z.N not found"). ``warmed_up`` is a
    fixed bool buffer in an exported model, so resolving it to a literal lets constant
    propagation + DCE delete the whole branch before conversion. ``_fold_constant_attrs`` only
    handles non-tensor primitives, so these tensor-scalar casts are handled here."""
    casts = {"aten::item": int, "aten::Int": int, "aten::Bool": bool, "aten::Float": float}
    for kind, cast in casts.items():
        for node in list(graph.findAllNodes(kind)):
            tensor = _resolve_attr(scripted, list(node.inputs())[0])
            if torch.is_tensor(tensor) and tensor.numel() == 1:
                const = graph.insertConstant(cast(tensor.item()))
                const.node().moveBefore(node)
                node.output().replaceAllUsesWith(const)
                node.destroy()


def _specialize_branches(traced: torch.jit.ScriptModule, example: torch.Tensor) -> int:
    """Profile-guided ``prim::If`` specialization — the fix that lets RAVE ``decode`` /
    ``forward`` convert *without* ``torch.cond``.

    TS2EPConverter turns **every** ``prim::If`` into ``torch.cond``, even when the predicate is a
    compile-time constant. RAVE's cached_conv ``decode`` / ``forward`` carry shape-driven
    conditionals (e.g. a length-alignment ``decode_params[1] * z.shape[-1] < y.shape[-1]``, plus
    bias-presence / config checks) whose ``torch.cond`` the converter then cannot build: it lifts a
    non-tensor shape int as a cond operand and aborts (``operands ... only tensor leaves``).

    But those predicates are **deterministic for a fixed ``buffer_size``** — a streaming RAVE takes
    the same branch on every block. So run the model once, read each top-level ``prim::If``'s actual
    boolean, replace the predicate with that constant, and let TorchScript constant-propagation
    inline the live branch and DCE the dead one. The converter then sees a branch-free graph and
    emits no ``torch.cond``. Folding a top-level ``If`` can expose one that was nested in its kept
    branch, so re-profile until none remain (bounded; stops if a round makes no progress).

    The profiling forward warms the streaming caches, so the pre-profile buffer state is saved and
    restored — conversion must capture the same state it would have without profiling.

    Soundness: this bakes the control-flow decisions for the exported ``buffer_size``, exactly as the
    rest of the pipeline bakes static shapes. It holds because cached_conv/RAVE control flow is
    *structural* — a function of tensor shapes and fixed config (padding, caching, channel/bias
    presence), never of sample values — so every block at the baked size follows the profiled path.
    :func:`migrate` shape-checks each converted method against the original ``.ts`` as a backstop.

    Returns the number of ``prim::If`` folded."""
    graph = traced.graph

    def top_level_ifs():
        return [n for n in graph.findAllNodes("prim::If")
                if n.owningBlock() == graph.block()]

    total = 0
    saved = {k: v.detach().clone() for k, v in traced.state_dict().items()}
    try:
        prev_all = None
        for _ in range(32):  # bounded; each round resolves one level of If nesting
            ifs = top_level_ifs()
            if not ifs:
                break
            # Capture each top-level predicate's runtime value: clone the graph, return the
            # predicates as a tuple, and run it on a representative (audio-like) input.
            probe = graph.copy()
            probe_ifs = [n for n in probe.findAllNodes("prim::If")
                         if n.owningBlock() == probe.block()]
            preds = [next(n.inputs()) for n in probe_ifs]
            while len(list(probe.outputs())):
                probe.eraseOutput(0)
            tup = probe.create("prim::TupleConstruct", preds)
            tup.output().setType(torch._C.TupleType([p.type() for p in preds]))
            probe.appendNode(tup)
            probe.registerOutput(tup.output())
            fn = torch._C._create_function_from_graph("nn_branch_profile", probe)
            vals = [bool(v) for v in fn(traced._c, torch.randn_like(example))]
            for node, val in zip(ifs, vals):
                const = graph.insertConstant(val)
                const.node().moveBefore(node)
                node.replaceInput(0, const)
            torch._C._jit_pass_constant_propagation(graph)
            torch._C._jit_pass_dce(graph)
            total += len(ifs)
            # Monotonic guard: each round should inline (remove) the Ifs it folded, so the total
            # If count strictly drops; if it doesn't, constant-prop couldn't inline them — stop.
            cur_all = len(graph.findAllNodes("prim::If"))
            if prev_all is not None and cur_all >= prev_all:
                break
            prev_all = cur_all
    except Exception as exc:
        # Profiling is an enhancement: on failure, keep whatever folded cleanly and let
        # conversion proceed (a still-present prim::If may then fail and the method is skipped).
        logging.debug("branch profiling stopped early: %s: %s", type(exc).__name__, exc)
    finally:
        traced.load_state_dict(saved)
    return total


def _populate_ranks(gm, example: torch.Tensor) -> int:
    """Best-effort fake-tensor propagation to set ``node.meta['val']`` (so shape-slices below can
    resolve tensor ranks) on the converter's pre-retrace graph module.

    The converter emits the TorchScript list ops ``aten.sym_size`` (full-shape variant) and
    ``aten.slice.t``, which fail the boxed dispatcher (``Expected List[t] ... found list``), so the
    stock ``ShapeProp`` / ``FakeTensorProp`` choke on them. Run a tolerant interpreter instead:
    propagate tensor ops under ``FakeTensorMode`` (cheap — params are fakeified by metadata only),
    while evaluating ``sym_size`` / ``slice.t`` / ``getitem`` directly in Python. Each node is
    guarded, so an unhandled list op just leaves that node without a ``val`` rather than aborting.

    Returns the number of nodes given a tensor ``val``."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    env: dict = {}
    done = 0
    with mode:
        fake_example = mode.from_tensor(example)

        def load(a):
            if isinstance(a, torch.fx.Node):
                return env.get(a)
            if isinstance(a, (list, tuple)):
                return type(a)(load(x) for x in a)
            return a

        for node in gm.graph.nodes:
            val = None
            try:
                if node.op == "placeholder":
                    val = fake_example if (placeholders and node is placeholders[0]) else None
                elif node.op == "get_attr":
                    obj = gm
                    for part in node.target.split("."):
                        obj = getattr(obj, part)
                    val = mode.from_tensor(obj) if isinstance(obj, torch.Tensor) else obj
                elif node.op == "call_function":
                    tgt, ts = node.target, str(node.target)
                    args = load(node.args)
                    kwargs = {k: load(v) for k, v in node.kwargs.items()}
                    if "sym_size" in ts:  # full shape (1 arg) or a single dim (2 args)
                        t = args[0]
                        val = list(t.shape) if len(args) == 1 else t.shape[args[1]]
                    elif "slice.t" in ts:
                        lst = args[0]
                        start = args[1] if len(args) > 1 else None
                        end = args[2] if len(args) > 2 else None
                        step = args[3] if len(args) > 3 else 1
                        val = lst[start:end:(step or 1)]
                    elif tgt is operator.getitem:
                        val = args[0][args[1]]
                    else:
                        val = tgt(*args, **kwargs)
                if isinstance(val, torch.Tensor):
                    node.meta["val"] = val
                    done += 1
            except Exception:
                val = None
            env[node] = val
    return done


def _rebuild_shape_slices(gm, in_rank: int) -> int:
    """Replace ``aten.slice.t`` on a tensor-shape list with explicit ``getitem`` element reads.

    RAVE does ``x.shape[:-2]``; the converter emits ``aten.slice.t(aten.sym_size(x), ...)`` —
    a TorchScript *list* op that torch.export cannot retrace ("Expected List[t] ... found
    list"). A slice of a shape is a fixed set of element indices, so rebuild it as
    ``[getitem(sym_size, i) for i in indices]`` (each ``getitem`` on a shape list retraces
    fine). The slice's negative bounds need the shape's rank: take it from the tensor's fake-val
    meta (populated by :func:`_populate_ranks`), the shape-prop ``tensor_meta``, or — for the
    input placeholder — the known input rank. Skip if still unknown."""
    graph = gm.graph
    rebuilt = 0
    for node in list(graph.nodes):
        if not (node.op == "call_function" and "slice.t" in str(node.target)):
            continue
        shape_node = node.args[0]
        if not (getattr(shape_node, "op", None) == "call_function"
                and "sym_size" in str(shape_node.target)):
            continue
        src = shape_node.args[0]
        meta = src.meta if hasattr(src, "meta") else {}
        val = meta.get("val")
        tm = meta.get("tensor_meta")
        rank = (val.dim() if hasattr(val, "dim")
                else len(tm.shape) if tm is not None and hasattr(tm, "shape")
                else in_rank if getattr(src, "op", None) == "placeholder" else None)
        if rank is None:
            continue
        start = node.args[1] if len(node.args) > 1 else None
        end = node.args[2] if len(node.args) > 2 else None
        step = node.args[3] if len(node.args) > 3 else 1
        indices = list(range(*slice(start, end, step).indices(rank)))
        with graph.inserting_before(node):
            items = [graph.call_function(operator.getitem, (shape_node, i)) for i in indices]

        def _sub(a):
            if a is node:
                return items
            if isinstance(a, (list, tuple)):
                return type(a)(_sub(x) for x in a)
            return a

        for user in list(node.users):
            user.args = tuple(_sub(a) for a in user.args)
            user.kwargs = {k: _sub(v) for k, v in user.kwargs.items()}
        rebuilt += 1
    if rebuilt:
        graph.eliminate_dead_code()
        gm.recompile()
    return rebuilt


@contextlib.contextmanager
def _patched_conversion(example: torch.Tensor):
    """Patch TS2EPConverter for the duration of a ``convert()`` to work around two retrace
    limitations: (1) it hard-codes ``[Dim.AUTO]*ndim`` — force **static** shapes instead (we
    bake a fixed ``buffer_size`` anyway), and (2) it leaves ``aten.slice.t`` shape-slices that
    can't retrace — populate tensor ranks (:func:`_populate_ranks`) then rebuild the slices as
    ``getitem`` reads (:func:`_rebuild_shape_slices`)."""
    import torch._export.converter as _conv
    import torch.export._trace as _trace
    orig_export = _trace._export
    orig_retrace = _conv.TS2EPConverter.retrace_as_exported_program

    def _static_export(*args, **kwargs):
        kwargs["dynamic_shapes"] = None
        return orig_export(*args, **kwargs)

    def _retrace(self, gm, name_to_constant):
        _populate_ranks(gm, example)
        _rebuild_shape_slices(gm, example.dim())
        return orig_retrace(self, gm, name_to_constant)

    _trace._export = _static_export
    _conv.TS2EPConverter.retrace_as_exported_program = _retrace
    try:
        yield
    finally:
        _trace._export = orig_export
        _conv.TS2EPConverter.retrace_as_exported_program = orig_retrace


def _convert_method(ts_path: str, name: str, example: torch.Tensor):
    """Trace one scripted method, clean the graph, and convert it to an ``ExportedProgram``."""
    # Fresh load per method so the streaming caches start from the .ts's saved (cold) state.
    scripted = torch.jit.load(ts_path, map_location="cpu").eval()
    wrapper = _MethodWrapper(scripted, name).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example,), check_trace=False)
        torch._C._jit_pass_inline(traced.graph)
        # Resolve GetAttr chains against the traced module (the graph's `self`), whose
        # `_scripted.*` subtree mirrors the original model's buffers.
        _fold_constant_attrs(traced, traced.graph)
        _specialize_scalar_reads(traced, traced.graph)
        # Collapse the now-constant config-driven branches before conversion.
        torch._C._jit_pass_constant_propagation(traced.graph)
        torch._C._jit_pass_dce(traced.graph)
        # Profile-guided specialization of the remaining (shape-driven) prim::If, so the
        # converter sees a branch-free graph and emits no torch.cond.
        _specialize_branches(traced, example)
        with _patched_conversion(example):
            return TS2EPConverter(traced, (example,)).convert()


def _handle_rng(ep, mode: str) -> list:
    """(A) RAVE stochastic ops. ``keep`` leaves them (CoreML compiles RNG); ``zero`` makes the
    model deterministic by replacing each with a zero tensor (for kernel-limited runtimes)."""
    rng_nodes = [n for n in ep.graph_module.graph.nodes
                 if n.op == "call_function" and n.target in _RNG_OPS]
    if mode == "zero":
        for node in rng_nodes:
            # zeros_like / zeros share randn_like / randn's signature, so swap the target.
            node.target = (torch.ops.aten.zeros_like.default
                           if node.target in _RNG_LIKE
                           else torch.ops.aten.zeros.default)
        ep.graph_module.graph.eliminate_dead_code()
        ep.graph_module.recompile()
    return [str(n.target) for n in rng_nodes]


# Ops belonging to RAVE's FFT noise synthesizer (complex / STFT). Unsupported by CoreML
# (no complex64 dtype) and the portable CPU runtime.
_COMPLEX_OPS = ("fft", "view_as_complex", "view_as_real", "complex")


def _skip_noise_synth(ep) -> int:
    """Excise RAVE's FFT noise-synthesizer branch (lossy). Returns the number of complex ops
    removed (0 if the model has none).

    RAVE's optional noise synth is an STFT (overlap-add) branch whose real-valued output is
    *added* to the main waveform near the decoder output. Its complex ops can't lower to CoreML
    or run on the portable CPU runtime, so methods using it are otherwise skipped entirely. This
    removes just the noise contribution: taint forward from the complex/FFT ops, and wherever a
    tainted value merges with a real main-path tensor (an add/mul/cat/copy_ that also takes a
    non-constant untainted input), replace the tainted operand with shape-matched zeros. The now
    dead complex chain is then DCE'd. The migrated model loses the synthesized noise texture but
    becomes fully lowerable and runnable.

    Must run on a **statically-shaped** EP: the replacement uses ``aten.zeros(concrete_shape)``
    rather than ``zeros_like(tensor)`` precisely so it does not reference (and thus keep alive)
    the tensor being cut."""
    graph = ep.graph_module.graph
    const = {s.arg.name for s in ep.graph_signature.input_specs
             if s.kind != InputKind.USER_INPUT}

    def _is_main(n):  # a real main-path operand, not a constant / param / buffer / get_attr
        return (hasattr(n, "op") and n.op != "get_attr"
                and not (n.op == "placeholder" and n.name in const))

    def _tensor_inputs(n):
        flat = []
        for a in n.args:
            flat += a if isinstance(a, (list, tuple)) else [a]
        return [a for a in flat if hasattr(a, "op")]

    seeds = [n for n in graph.nodes if n.op == "call_function"
             and any(k in str(n.target) for k in _COMPLEX_OPS)]
    if not seeds:
        return 0
    taint = set(seeds)
    stack = list(seeds)
    while stack:
        for user in stack.pop().users:
            if user not in taint:
                taint.add(user)
                stack.append(user)
    for node in list(graph.nodes):
        if node.op != "call_function" or node not in taint:
            continue
        inputs = _tensor_inputs(node)
        if any((a not in taint and _is_main(a)) for a in inputs):  # noise merges with main here
            for tainted in [a for a in inputs if a in taint]:
                val = tainted.meta["val"]
                shape = [int(s) for s in val.shape]
                with graph.inserting_before(node):
                    zeros = graph.call_function(torch.ops.aten.zeros.default,
                                                (shape,), {"dtype": val.dtype})
                node.replace_input_with(tainted, zeros)
    graph.eliminate_dead_code()
    ep.graph_module.recompile()
    return len(seeds)


def _has_cache_writes(ep) -> bool:
    """True if the graph still has in-place cache writes (``copy_`` into a buffer / a slice of
    one) — i.e. it is a streaming method that should carry state across calls."""
    buffers = {s.arg.name for s in ep.graph_signature.input_specs
               if s.kind == InputKind.BUFFER}
    for node in ep.graph_module.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            dst = node.args[0]
            root = dst.args[0] if (getattr(dst, "op", None) == "call_function") else dst
            if getattr(root, "name", None) in buffers:
                return True
    return False


def _eager_streams(ep, example: torch.Tensor) -> bool:
    """Warm-vs-cold eager check: run the module on one input, then a second; compare the second
    output to a cold run of the same input. If the caches feed the output, the two differ.

    ``ExportedProgram.module()`` executes the in-place ``copy_`` cache writes eagerly, so this
    reflects whether streaming state actually reaches the output — used to catch a noise-synth
    removal that disconnected the main cached path (then the method must not be shipped)."""
    with torch.no_grad():
        x0, x1 = torch.randn_like(example), torch.randn_like(example)
        warm_mod = ep.module()
        warm_mod(x0)
        warm = warm_mod(x1)
        cold = ep.module()(x1)
    return (warm - cold).abs().max().item() > 1e-6


def _output_shape_matches(ts_path: str, name: str, ep, example: torch.Tensor) -> bool:
    """Backstop for profile-guided If folding (:func:`_specialize_branches`): the converted program
    must produce the same output *shape* as the original ``.ts``. RAVE is stochastic so values can't
    be compared, but a mis-specialized length/padding branch changes the output length — which this
    catches. Returns False (→ skip the method) on any shape divergence or run error."""
    def shp(o):
        if torch.is_tensor(o):
            return tuple(int(d) for d in o.shape)
        if isinstance(o, (list, tuple)):
            return tuple(shp(x) for x in o)
        return None

    try:
        fresh = torch.jit.load(ts_path, map_location="cpu").eval()
        x = torch.randn_like(example)
        with torch.no_grad():
            ref = getattr(fresh, name)(x)
            got = ep.module()(x)
        return shp(ref) == shp(got)
    except Exception as exc:
        logging.warning("  shape check for '%s' could not run (%s: %s)",
                        name, type(exc).__name__, str(exc)[:120])
        return False


def _lift_cache_mutations(ep) -> list:
    """(B) Promote the converter's in-place cache writes to persistent ExecuTorch state.

    TS2EPConverter emits ``aten.copy_.default`` into lifted ``BUFFER`` inputs but records no
    buffer mutation (``buffers_to_mutate`` stays empty), so the lowered runtime would reset the
    cached_conv caches each block (→ clicks). Rewrite each such copy into the ``torch.export``
    buffer-mutation convention — return the new full-buffer value as a ``BUFFER_MUTATION`` output
    targeting the buffer's FQN — and drop the in-place ``copy_``. Two cached_conv write forms:

    * whole-buffer ``copy_(buffer, new)`` → the new value is ``new`` (e.g. percussion);
    * sliced ``copy_(slice(buffer, dim, start, end), new)`` (a ring-buffer update, e.g. VCTK)
      → the new value is ``slice_scatter(buffer, new, dim, start, end)``.

    Returns the lifted buffer FQNs (for the mutable-buffer zero-init pass)."""
    sig = ep.graph_signature
    graph = ep.graph_module.graph
    buffer_inputs = {s.arg.name: s.target for s in sig.input_specs
                     if s.kind == InputKind.BUFFER}
    out_node = next(n for n in graph.nodes if n.op == "output")
    user_outputs = list(out_node.args[0])

    new_value_by_fqn = {}  # buffer_fqn -> latest new-value node (composes repeated writes)
    order = []
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.copy_.default:
            continue
        dst, src = node.args[0], node.args[1]
        if getattr(dst, "op", None) == "placeholder" and dst.name in buffer_inputs:
            fqn = buffer_inputs[dst.name]
            new_value = src
        elif (getattr(dst, "op", None) == "call_function"
              and dst.target == torch.ops.aten.slice.Tensor
              and getattr(dst.args[0], "op", None) == "placeholder"
              and dst.args[0].name in buffer_inputs):
            buffer_node = dst.args[0]
            fqn = buffer_inputs[buffer_node.name]
            sargs = list(dst.args)  # (self, dim=0, start=None, end=None, step=1)
            dim = sargs[1] if len(sargs) > 1 else 0
            start = sargs[2] if len(sargs) > 2 else None
            end = sargs[3] if len(sargs) > 3 else None
            step = sargs[4] if len(sargs) > 4 else 1
            base = new_value_by_fqn.get(fqn, buffer_node)  # compose if already written
            with graph.inserting_before(node):
                new_value = graph.call_function(
                    torch.ops.aten.slice_scatter.default,
                    (base, src, dim, start, end, step))
            # slice_scatter preserves the buffer's shape/dtype; carry the fake-tensor meta
            # so the lowering (and its memory planning) sees a well-typed mutation value.
            if "val" in base.meta:
                new_value.meta["val"] = base.meta["val"]
        else:
            continue  # not an in-place write into a tracked buffer
        if fqn not in new_value_by_fqn:
            order.append(fqn)
        new_value_by_fqn[fqn] = new_value
        node.replace_all_uses_with(src)  # copy_ result (the written region) is rarely read
        graph.erase_node(node)

    mutations = [(fqn, new_value_by_fqn[fqn]) for fqn in order]
    # Buffer mutations must precede user outputs in both the graph output and the spec list.
    out_node.args = ([val for _, val in mutations] + user_outputs,)
    mutation_specs = [OutputSpec(kind=OutputKind.BUFFER_MUTATION,
                                 arg=TensorArgument(name=val.name), target=fqn)
                      for fqn, val in mutations]
    ep._graph_signature = ExportGraphSignature(
        input_specs=sig.input_specs,
        output_specs=mutation_specs + list(sig.output_specs))
    ep.graph_module.recompile()

    fqns = [fqn for fqn, _ in mutations]
    for fqn in fqns:  # zero the caches so the model starts cold (no first-block click)
        if fqn in ep.state_dict:
            ep.state_dict[fqn].zero_()
    return fqns


def _build_metadata(methods: dict, buffer_size: int, delegate: str):
    """Emit the ``kind:"live"`` metadata via ``LiveModule`` — identical to the SAME-S path,
    so the ``neural.live~`` C++ host parses it unchanged. Returns ``(live_module, metadata)``."""
    live = LiveModule()
    for name, info in methods.items():
        live.register_method(
            name, in_channels=info["in_channels"], in_ratio=info["in_ratio"],
            out_channels=info["out_channels"], out_ratio=info["out_ratio"],
            input_labels=info["input_labels"], output_labels=info["output_labels"],
            test_method=False)
    return live, live._nn_metadata(buffer_size, delegate, _executorch_version())


def migrate(ts_path: str, out: str, delegate: str = "coreml",
            buffer_size: Optional[int] = None, rng: str = "keep",
            skip_noise_synth: bool = False,
            compute_units: str = "CPU_AND_NE") -> str:
    """Convert a RAVE ``.ts`` to ``<out>.pte`` + ``<out>.json``. Returns the ``.pte`` path.

    ``compute_units`` (CoreML only) pins the ``coremltools.ComputeUnit``; defaults to
    ``"CPU_AND_NE"`` because CoreML's GPU path miscomputes RAVE (CPU/ANE are bit-correct)."""
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    scripted = torch.jit.load(ts_path, map_location="cpu").eval()
    methods = _discover_methods(scripted)
    if not methods:
        raise ValueError(
            f"No nn~ methods found on {ts_path} (no <method>_params buffers). "
            "Is this a RAVE/nn_tilde-exported TorchScript model?")

    # Smallest legal block = the largest ratio (one latent frame); must divide every ratio.
    max_ratio = max(max(m["in_ratio"], m["out_ratio"]) for m in methods.values())
    if buffer_size is None:
        buffer_size = max_ratio
    for name, info in methods.items():
        for ratio in (info["in_ratio"], info["out_ratio"]):
            if buffer_size % ratio:
                raise ValueError(f"buffer_size {buffer_size} not a multiple of ratio "
                                 f"{ratio} for method '{name}'")

    logging.info("Migrating %s: methods=%s buffer_size=%d delegate=%s rng=%s",
                 ts_path, list(methods), buffer_size, delegate, rng)

    method_graphs = {}
    primed = {}  # method name -> FQNs of its lifted streaming caches (for the zero-init pass)
    for name, info in methods.items():
        example = torch.zeros(1, info["in_channels"], buffer_size // info["in_ratio"])
        try:
            ep = _convert_method(ts_path, name, example)
            # Backstop the profile-guided branch specialization: a mis-specialized branch would
            # change the output length — reject the method if its shape no longer matches the .ts.
            if not _output_shape_matches(ts_path, name, ep, example):
                logging.warning(
                    "  skipping method '%s': converted output shape differs from the .ts "
                    "(branch specialization mismatch); re-export from the RAVE checkpoint.", name)
                continue
            rng_found = _handle_rng(ep, rng)
            # TS2EPConverter leaves the input time dim dynamic; re-export with a concrete
            # example to specialize every shape to this fixed buffer_size (CoreML state buffers
            # require static shapes). The cache copy_ nodes survive re-export so the lift below
            # still finds them.
            with torch.no_grad():
                ep = torch.export.export(ep.module(), (example,), strict=False)
            # Optionally excise the FFT noise synth (after static re-export so zeros can use
            # concrete shapes) — makes decode/forward lowerable at the cost of the noise texture.
            removed = _skip_noise_synth(ep) if skip_noise_synth else 0
            # Safety: the excision is clean only when the noise synth is a terminal additive
            # branch (e.g. percussion). When it rejoins mid-network before skip-connected,
            # cached processing (e.g. VCTK), removing it disconnects the streaming caches. If a
            # streaming method stops streaming after removal, drop it rather than ship a clicking
            # model.
            if removed and _has_cache_writes(ep) and not _eager_streams(ep, example):
                logging.warning(
                    "  skipping method '%s': removing the FFT noise synth disconnected its "
                    "streaming caches (not a terminal additive branch in this model). Re-export "
                    "from the RAVE checkpoint for a full streaming '%s'.", name, name)
                continue
            primed[name] = _lift_cache_mutations(ep)
            method_graphs[name] = ep
            logging.info("  %s: lifted %d cache mutations, rng ops=%s%s",
                         name, len(ep.graph_signature.buffers_to_mutate), set(rng_found),
                         f", removed {removed} FFT noise-synth ops" if removed else "")
        except Exception as conv_exc:
            # Some methods don't convert: a generative `prior` is an autoregressive prim::Loop
            # (belongs to neural.gen~, not a per-block streaming method); other models hit
            # TS2EPConverter limitations (conditional control flow / slice-on-list). Skip the
            # method and keep going so the convertible ones still migrate.
            logging.warning("  skipping method '%s' (conversion failed): %s: %s",
                            name, type(conv_exc).__name__, str(conv_exc)[:160])
    if not method_graphs:
        raise RuntimeError(
            f"No method of {ts_path} could be converted by TS2EPConverter (control flow it "
            "cannot handle). Re-export from the RAVE checkpoint via the eager LiveModule path.")

    cfg = EdgeCompileConfig(_check_ir_validity=False,
                            _core_aten_ops_exception_list=list(_RNG_OPS))

    def _lower(graphs: dict):
        # Migrated RAVE methods can partition differently under CoreML (decode's FFT noise
        # synth vs encode), so disable POSITIONAL multimethod weight sharing (which requires
        # equal partition counts). view_as_complex needs no_grad (autograd-on-complex error).
        # Pin the CoreML compute unit (default CPU_AND_NE): CoreML's GPU path miscomputes RAVE
        # while CPU/ANE are bit-correct — the exporter default is ALL, so RAVE must ask for it.
        # Lower a deep copy: to_edge_transform_and_lower consumes the EP, but we may need to
        # re-lower a method (trial then final), so keep the originals in method_graphs pristine.
        buffers = [fqn for name in graphs for fqn in primed[name]]
        with torch.no_grad():
            return to_edge_transform_and_lower(
                copy.deepcopy(graphs),
                transform_passes=_make_init_mutable_passes(buffers),
                partitioner=_make_partitioner(
                    delegate, coreml_disable_weight_sharing=True,
                    coreml_compute_units=compute_units),
                compile_config=cfg).to_executorch()

    # Try all methods together; if a method uses ops the delegate can't handle (e.g. RAVE's
    # FFT noise synth — view_as_complex/fft_* — which CoreML/coremltools and the portable CPU
    # runtime do not support), drop it and emit a .pte with the methods that do lower.
    try:
        executorch_program = _lower(method_graphs)
        kept = list(method_graphs)
    except Exception as exc:
        logging.warning("Combined lowering failed (%s: %s); lowering methods individually",
                        type(exc).__name__, str(exc)[:160])
        kept_graphs = {}
        for name, ep in method_graphs.items():
            try:
                _lower({name: ep})
                kept_graphs[name] = ep
            except Exception as method_exc:
                logging.warning("  skipping method '%s' for delegate '%s': %s",
                                name, delegate, repr(method_exc)[:140])
        if not kept_graphs:
            raise RuntimeError(
                f"No method of {ts_path} could be lowered for delegate '{delegate}'. RAVE "
                "models with an FFT noise synth use ops unsupported by CoreML / the portable "
                "runtime; try a different delegate or re-export from the RAVE checkpoint.") from exc
        executorch_program = _lower(kept_graphs)
        kept = list(kept_graphs)
    if len(kept) != len(methods):
        logging.warning("Migrated %d/%d methods: kept=%s skipped=%s",
                        len(kept), len(methods), kept,
                        [m for m in methods if m not in kept])

    if not out.endswith(".pte"):
        out += ".pte"
    with open(out, "wb") as f:
        f.write(executorch_program.buffer)
    logging.info("Wrote ExecuTorch program to %s", out)

    live, metadata = _build_metadata({k: methods[k] for k in kept}, buffer_size, delegate)
    live._write_metadata_json(out, metadata)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(
        description="Migrate a RAVE .ts model to a neural.live~ .pte + metadata JSON")
    ap.add_argument("model", help="path to the RAVE TorchScript .ts model")
    ap.add_argument("--out", default=None,
                    help="output path; .pte is appended (default: model name without .ts)")
    ap.add_argument("--delegate", default="coreml",
                    choices=["coreml", "xnnpack", "metal", "portable"],
                    help="ExecuTorch backend (coreml recommended; compiles RNG, but skips "
                         "FFT-synth methods it can't represent — see module docstring). "
                         "metal is EXPERIMENTAL (AOTInductor; unvalidated for RAVE)")
    ap.add_argument("--buffer-size", type=int, default=None,
                    help="audio block size to bake in (default: the model's largest ratio)")
    ap.add_argument("--rng", default="keep", choices=["keep", "zero"],
                    help="'keep' stochastic ops (CoreML) or 'zero' them (deterministic CPU)")
    ap.add_argument("--skip-noise-synth", action="store_true",
                    help="excise RAVE's FFT noise synth so decode/forward lower to CoreML/CPU "
                         "(lossy: drops the synthesized noise texture)")
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
                    help="CoreML compute unit (default CPU_AND_NE; CoreML's GPU miscomputes "
                         "RAVE, so avoid ALL/CPU_AND_GPU unless you validate the output)")
    args = ap.parse_args()

    out = args.out
    if out is None:  # default: model path without the .ts suffix
        out = args.model[:-3] if args.model.endswith(".ts") else args.model
    path = migrate(args.model, out, delegate=args.delegate,
                   buffer_size=args.buffer_size, rng=args.rng,
                   skip_noise_synth=args.skip_noise_synth,
                   compute_units=args.compute_units)
    print(f"migrated {args.model} -> {path} (+ metadata .json), delegate {args.delegate}")


if __name__ == "__main__":
    main()
