"""ExecuTorch CUDA (AOTInductor) lowering for ``delegate="cuda"`` — Windows x64 + NVIDIA.

The CUDA backend compiles each method as one whole graph with AOTInductor: Triton kernels,
autotuned on the export machine's GPU, plus a C++ wrapper linked into a DLL that is embedded
in the ``.pte``; the weights go to an external ``.ptd``. Exporting needs a CUDA GPU, a
CUDA-enabled torch (2.14, cu130) and, on Windows, MSVC (a "Developer PowerShell for VS 2022").

torch 2.14's Windows-target AOTI path assumes a *Linux* host cross-compiling with mingw.
:func:`cuda_lowering` scopes the few patches that let it run natively on a Windows host (and
lifts the small-GPU template gate on any host); everything is restored on exit. Imports are
lazy so ``neural_tilde`` still imports without executorch / a CUDA torch.
"""

import contextlib
import gc
import logging
import os
import re
import shutil
import subprocess
import sys

# Oldest GPU arch exported models target (Turing / RTX 20xx). AOTI autotunes on the export
# GPU, then recompiles the chosen kernel configs for this arch (inductor's cuda.arch), so one
# export runs on this arch and every newer one. Override with NN_CUDA_ARCH (e.g. "86").
DEFAULT_TARGET_ARCH = "75"
# SASS embedded per kernel (archs below the target are dropped); PTX for the target arch is
# always added, so GPUs newer than the list JIT it.
DEFAULT_ARCH_LIST = "7.5;8.0;8.6;8.9;9.0;12.0"

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_SHIM_LIB = "aoti_cuda_shims.lib"


def target_arch() -> str:
    return os.environ.get("NN_CUDA_ARCH", DEFAULT_TARGET_ARCH).replace(".", "")


def _arch_number(arch: str) -> int:
    return int(re.sub(r"\D", "", arch.split("+")[0]))


def _sass_archs() -> list:
    """The SASS archs the fatbins carry: the arch list entries at or above the target."""
    archs = os.environ.get("TORCH_CUDA_ARCH_LIST", DEFAULT_ARCH_LIST)
    floor = _arch_number(target_arch())
    return sorted({n for n in (_arch_number(a) for a in re.split(r"[;\s]+", archs) if a)
                   if n >= floor})


def cuda_metadata() -> dict:
    """Extra metadata JSON fields for a CUDA program (checked by the host at load)."""
    return {"platform": "windows",
            "cuda_min_arch": _arch_number(target_arch()),
            "cuda_archs": _sass_archs()}


def shim_library_dir() -> str:
    """Directory holding ``aoti_cuda_shims.lib``, the import library every Windows CUDA
    wrapper DLL links against. ``NN_CUDA_SHIM_DIR`` overrides the copy bundled with
    neural_tilde (built from the same ExecuTorch release as the shipped runtime DLL)."""
    path = os.environ.get("NN_CUDA_SHIM_DIR", _DATA_DIR)
    if not os.path.exists(os.path.join(path, _SHIM_LIB)):
        raise FileNotFoundError(
            f"{_SHIM_LIB} not found in {path}. Set NN_CUDA_SHIM_DIR to a directory "
            "containing it (e.g. <executorch>/cmake-out/lib of an ExecuTorch CUDA build).")
    return path


def make_cuda_partitioner(method_name: str) -> list:
    """The CUDA partitioner for one method (AOTI keys the embedded DLL on the method name)."""
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    specs = [CudaBackend.generate_method_name_compile_spec(method_name),
             CompileSpec("platform", b"windows")]
    if sys.platform == "win32":
        # On a Linux host upstream's mingw cross-compile finds the lib in its own wheel.
        specs.append(CompileSpec("shim_library_path", shim_library_dir().encode()))
    return [CudaPartitioner(specs)]


def pin_factory_devices(exported):
    """Give every tensor-factory op without a ``device`` kwarg (e.g. ``torch.randn(shape)``,
    ``hann_window``) an explicit one from its traced value. The CUDA backend moves the program
    with ``move_to_device_pass``, which retargets explicit device kwargs only; a factory left
    implicit stays on the CPU and fails against CUDA tensors at compile."""
    import torch
    graph = exported.graph_module.graph
    pinned = 0
    for node in graph.nodes:
        if node.op != "call_function" or "device" in node.kwargs:
            continue
        schema = getattr(node.target, "_schema", None)
        val = node.meta.get("val")
        if (schema is None or not isinstance(val, torch.Tensor)
                or not any(a.name == "device" and a.kwarg_only for a in schema.arguments)):
            continue
        node.kwargs = {**node.kwargs, "device": val.device}
        pinned += 1
    if pinned:
        exported.graph_module.recompile()
        logging.info("CUDA export: pinned the device of %d tensor-factory op(s)", pinned)
    return exported


def decompose_convs(exported):
    """Rewrite convolutions so every one lowers to a Triton conv template: ExecuTorch's CUDA
    shims have no convolution fallback, and inductor's templates take neither transposed nor
    dilated convs. First conv_transpose1d -> zero-stuffing + conv1d
    (:func:`_conv_transpose1d_as_conv1d`), then conv1d -> undilated conv2d
    (:func:`_conv1d_as_conv2d`; conv2d as upstream CUDA exports do)."""
    import torch
    exported = exported.run_decompositions(
        {torch.ops.aten.conv_transpose1d.default: _conv_transpose1d_as_conv1d})
    return exported.run_decompositions({torch.ops.aten.conv1d.default: _conv1d_as_conv2d})


def _conv1d_as_conv2d(input, weight, bias=None, stride=(1,), padding=(0,), dilation=(1,),
                      groups=1):
    """``conv1d`` as an undilated ``conv2d``. A dilation ``d`` becomes a weight with ``d - 1``
    zeros between taps (same result, ``d * (k - 1) + 1`` taps)."""
    import torch
    from torch._inductor.decomposition import conv1d_to_conv2d
    d = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
    if d > 1:
        k = weight.shape[-1]
        weight = (torch.stack([weight] + [torch.zeros_like(weight)] * (d - 1), dim=-1)
                  .flatten(-2)[..., :d * (k - 1) + 1])
    return conv1d_to_conv2d(input, weight, bias, stride, padding, [1], groups)


def _conv_transpose1d_as_conv1d(input, weight, bias=None, stride=(1,), padding=(0,),
                                output_padding=(0,), groups=1, dilation=(1,)):
    """``conv_transpose1d`` as a stride-1 ``conv1d``: insert ``stride - 1`` zeros between input
    samples, pad each side by ``dilation * (k - 1) - padding`` (plus ``output_padding`` on the
    right; negative pads crop), and convolve with the weight transposed per group and flipped
    in time."""
    import torch
    import torch.nn.functional as F
    s, p, op, d = (v[0] if isinstance(v, (list, tuple)) else v
                   for v in (stride, padding, output_padding, dilation))
    cin, cout_g, k = weight.shape
    x = input
    if s > 1:
        # [N, C, L] -> [N, C, L * s]: each sample followed by s - 1 zeros (the trailing s - 1
        # zeros past the last sample are taken off the right pad below).
        x = F.pad(x.unsqueeze(-1), (0, s - 1)).flatten(-2)
    w = (weight.view(groups, cin // groups, cout_g, k).transpose(1, 2)
         .reshape(groups * cout_g, cin // groups, k).flip(-1))
    edge = d * (k - 1) - p
    x = F.pad(x, (edge, edge + op - (s - 1)))
    return torch.ops.aten.conv1d.default(x, w, bias, [1], [0], [d], groups)


def _preflight() -> None:
    import torch
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError(
            "delegate='cuda' needs a CUDA build of torch and an NVIDIA GPU on the export "
            "machine (kernels are autotuned on it), e.g. `pip install torch==2.14.0 "
            "--index-url https://download.pytorch.org/whl/cu130`.")
    try:
        import triton  # noqa: F401
    except ImportError as e:
        raise ImportError("delegate='cuda' needs Triton (`pip install triton-windows` on "
                          "Windows).") from e
    if sys.platform == "win32" and shutil.which("cl") is None:
        raise RuntimeError(
            "delegate='cuda' compiles C++ with MSVC: run the export from a 'Developer "
            "PowerShell for VS 2022' (cl.exe must be on PATH).")
    if sys.platform == "win32" and not os.environ.get("CUDA_PATH"):
        raise RuntimeError("delegate='cuda' needs the CUDA Toolkit (CUDA_PATH is not set).")
    shim_library_dir()


def _swap(stack: contextlib.ExitStack, owner, name: str, value) -> None:
    """Set ``owner.name = value`` until ``stack`` closes. Class attributes are swapped via the
    class ``__dict__`` so classmethods are restored as classmethods."""
    if not hasattr(owner, name):
        raise RuntimeError(
            f"delegate='cuda' patches {getattr(owner, '__name__', owner)}.{name}, which this "
            "torch/executorch does not have (expected torch 2.14 + executorch 1.5).")
    old = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
    setattr(owner, name, value)
    stack.callback(setattr, owner, name, old)


def _setenv(stack: contextlib.ExitStack, name: str, value: str) -> None:
    old = os.environ.get(name)
    os.environ[name] = value
    stack.callback(lambda: os.environ.__setitem__(name, old) if old is not None
                   else os.environ.pop(name, None))


@contextlib.contextmanager
def cuda_lowering():
    """Context for ``to_edge_transform_and_lower`` with CUDA partitioners: preflight checks,
    target-arch recompilation, and (on a Windows host) the native-MSVC AOTI patches."""
    _preflight()
    import torch
    import torch._inductor.utils as inductor_utils

    with contextlib.ExitStack() as stack:
        # Fatbin archs + target arch must be set before CudaBackend reads the device's.
        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            _setenv(stack, "TORCH_CUDA_ARCH_LIST", DEFAULT_ARCH_LIST)
        stack.enter_context(torch._inductor.config.patch({"cuda.arch": target_arch()}))
        # Inductor only uses Triton conv/GEMM templates on GPUs with >= 68 SMs; otherwise
        # convs fall back to aoti_torch_cuda_convolution, which the CUDA shims lack. The SM
        # count only gates autotuning on the export machine, not where the kernels run.
        _swap(stack, inductor_utils, "is_big_gpu", lambda index_or_device=0: True)
        if sys.platform == "win32":
            _patch_windows_host(stack)
        logging.info("CUDA export: target sm_%s, SASS %s (TORCH_CUDA_ARCH_LIST=%s)",
                     target_arch(), _sass_archs(), os.environ["TORCH_CUDA_ARCH_LIST"])
        yield


def _patch_windows_host(stack: contextlib.ExitStack) -> None:
    """Let torch's ``cross_target_platform="windows"`` AOTI path run *on* Windows with MSVC.

    That path assumes a Linux host with mingw. On Windows the compiler is already ``cl`` and
    the toolkit's ``cudart.lib`` is MSVC's, so: skip the mingw probe and its ``libcudart.a``
    generation, point ``WINDOWS_CUDA_HOME`` at ``CUDA_PATH``, drop libtorch's ``sleef`` (a
    Windows host always links it; the libtorch-free wrapper doesn't use it), fix the wrapper
    source for MSVC and embed the kernel binaries (see :func:`_compile_msvc`), and keep
    ExecuTorch's ``trim_host_memory`` from calling glibc's ``malloc_trim``."""
    import torch._inductor.cpp_builder as cpp_builder
    import torch.utils.cpp_extension as cpp_extension
    from executorch.backends.cuda import cuda_backend, cuda_weight_collector
    from torch._inductor.codecache import AotCodeCompiler

    _swap(stack, cpp_builder, "check_mingw_win32_flavor", lambda compiler: "win32")
    _swap(stack, cpp_builder, "_ensure_mingw_cudart_import_lib", lambda libraries_dirs: [])

    cuda_home = os.environ.get("WINDOWS_CUDA_HOME") or os.environ["CUDA_PATH"]
    _setenv(stack, "WINDOWS_CUDA_HOME", cuda_home)
    _swap(stack, cpp_extension, "WINDOWS_CUDA_HOME", cuda_home)

    torch_args = cpp_builder._get_torch_related_args

    def torch_args_without_sleef(*args, **kwargs):
        include_dirs, lib_dirs, libs = torch_args(*args, **kwargs)
        return include_dirs, lib_dirs, [lib for lib in libs if lib != "sleef"]

    _swap(stack, cpp_builder, "_get_torch_related_args", torch_args_without_sleef)

    stock_compile = AotCodeCompiler.__dict__["compile"].__func__

    def compile_msvc(cls, graph, wrapper_code, kernel_code, *args, **kwargs):
        return _compile_msvc(stock_compile, cls, graph, wrapper_code, kernel_code,
                             *args, **kwargs)

    _swap(stack, AotCodeCompiler, "compile", classmethod(compile_msvc))

    _swap(stack, cuda_weight_collector, "trim_host_memory", gc.collect)
    _swap(stack, cuda_backend, "trim_host_memory", gc.collect)


def _compile_msvc(stock_compile, cls, graph, wrapper_code, kernel_code, *args, **kwargs):
    """``AotCodeCompiler.compile`` for a Windows host:

    - the GPU wrapper codegen emits GCC's ``__attribute__((noinline))``; cl needs
      ``__declspec(noinline)``.
    - codecache skips cubin embedding on Windows (``if not _IS_WINDOWS``) but the wrapper
      still references ``__<kernel>_start`` (handed to ``cuModuleLoadData``): define those as
      byte arrays holding each kernel's (multi-arch) binary.
    - the built DLL is named ``<x>.wrapper.pyd``; ExecuTorch's packaging expects
      ``<x>.wrapper.so`` (same file)."""
    def fix(src):
        return src.replace("__attribute__((noinline))", "__declspec(noinline)")

    wrapper_code = fix(wrapper_code) + _embedded_kernel_definitions(wrapper_code)
    paths = stock_compile(cls, graph, wrapper_code, fix(kernel_code), *args, **kwargs)
    if isinstance(paths, list):
        for i, path in enumerate(paths):
            if isinstance(path, str) and path.endswith(".wrapper.pyd"):
                paths[i] = path[:-len(".pyd")] + ".so"
                os.replace(path, paths[i])
    return paths


def _embedded_kernel_definitions(wrapper_code: str) -> str:
    """C++ definitions for every ``__<kernel>_start`` blob the wrapper declares."""
    from torch._inductor.codecache import (CudaKernelParamCache,
                                           get_cpp_wrapper_cubin_path_name)
    names = re.findall(r"extern const unsigned char __(\w+)_start\[\];", wrapper_code)
    if not names:
        return ""
    out = ['\nextern "C" {']
    for name in names:
        entry = CudaKernelParamCache.cache.get(name)
        if entry is None:
            raise RuntimeError(f"no compiled binary cached for kernel {name}")
        bin_path = entry[get_cpp_wrapper_cubin_path_name()]
        if bin_path.endswith(".fatbin"):
            # Multi-arch: the build codecache skips on Windows. Always rebuild — a cached
            # fatbin may hold a different TORCH_CUDA_ARCH_LIST from an earlier export.
            _build_fatbin(entry, bin_path)
        with open(bin_path, "rb") as f:
            blob = f.read()
        rows = (", ".join(str(b) for b in blob[i:i + 64]) for i in range(0, len(blob), 64))
        # explicit `extern`: MSVC gives a namespace-scope const definition internal linkage
        out.append(f"alignas(64) extern const unsigned char __{name}_start[{len(blob)}] = {{\n"
                   + ",\n".join(rows) + "\n};")
    out.append("}\n")
    logging.info("CUDA export: embedded %d kernel binaries", len(names))
    return "\n".join(out)


def _build_fatbin(entry: dict, fatbin_path: str) -> None:
    """Build a kernel's multi-arch fatbin the way codecache does on a Linux host."""
    from torch._inductor import codecache
    nvcc = codecache.cuda_compile_utils._cuda_compiler()
    cmd = codecache._cuda_fatbin_command(entry["asm"], fatbin_path,
                                         entry.get("runtime_bin_path"), nvcc,
                                         shutil.which("fatbinary"), entry.get("cuda_arch"))
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"fatbin build failed: {' '.join(cmd)}\n{e.stderr}") from e
