"""Write THIRD_PARTY_NOTICES.md: the third-party software compiled into the
Windows externals and support DLLs, with each license text.

Reads the licenses from the source trees the build used (no network):
  - the ExecuTorch checkout (--executorch, default C:/dev/executorch) and its
    install's CMake cache (for the CUDA toolkit version);
  - the tokenizers-cpp checkout CMake fetched into the build tree (--build);
  - the min-api submodule;
  - torch's wheel metadata (the active Python's torch, or --torch-license);
  - the Rust crates of tokenizers-cpp via `cargo metadata --offline`.

Fails if an expected license file is missing, so the notices can't silently go
stale when a dependency moves. Run from anywhere:

    python src/package/third_party_notices.py [--build build] [--executorch C:/dev/executorch]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The externals that link ExecuTorch, as named in the table (defined in the intro).
ET_EXTERNALS = "ExecuTorch externals"
TOK_EXTERNALS = "neural.tokenizer"
ALL_EXTERNALS = "all externals"

# SPDX preference when a crate offers a choice (A OR B): the first one listed.
PREFERRED = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "Zlib", "BSL-1.0",
             "Unlicense", "Unicode-3.0"]
# File-name hints for each license, tried before a plain LICENSE/COPYING.
FILE_HINTS = {
    "MIT": ["LICENSE-MIT", "LICENSE.MIT", "LICENSE-MIT.md", "LICENSE-MIT.txt"],
    "Apache-2.0": ["LICENSE-APACHE", "LICENSE.APACHE", "LICENSE-APACHE.md",
                   "LICENSE-APACHE-2.0", "LICENSE-Apache-2.0", "LICENSE-APACHE.txt"],
    "BSD-2-Clause": ["LICENSE-BSD", "LICENSE-BSD-2-Clause"],
    "Zlib": ["LICENSE-ZLIB", "LICENSE-ZLIB.md", "LICENSE-Zlib"],
    "BSL-1.0": ["LICENSE-BOOST", "LICENSE-BSL", "LICENSE_1_0.txt"],
    "Unlicense": ["UNLICENSE"],
    "Unicode-3.0": ["LICENSE-UNICODE", "UNICODE-LICENSE"],
}


def fail(msg):
    sys.exit(f"third_party_notices: {msg}")


def read(path):
    path = Path(path)
    if not path.is_file():
        fail(f"missing license file {path}")
    return path.read_text(encoding="utf-8", errors="replace").strip()


def git_rev(path, describe=False):
    cmd = ["git", "-C", str(path)] + (["describe", "--tags"] if describe
                                      else ["rev-parse", "--short", "HEAD"])
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        fail(f"can't read the git revision of {path}")


def torch_license(override):
    if override:
        return Path(override), "?"
    try:
        from importlib import metadata
        dist = metadata.distribution("torch")
    except Exception:
        fail("torch isn't importable here; activate the export environment "
             "(nt-cuda) or pass --torch-license <torch dist-info LICENSE>")
    for f in dist.files or []:
        if f.name == "LICENSE" and "dist-info" in str(f):
            return Path(dist.locate_file(f)), dist.version
    fail("torch's wheel has no LICENSE in its dist-info")


def cuda_version(et):
    cache = et / "cmake-out" / "CMakeCache.txt"
    m = re.search(r"CMAKE_CUDA_COMPILER:FILEPATH=.*/CUDA/v(\d+\.\d+)/",
                  read(cache))
    if not m:
        fail(f"no CUDA compiler in {cache}")
    return m.group(1)


def cpp_components(args):
    """(name, version, license, used in, [license files]) for the C/C++ code."""
    et = Path(args.executorch)
    xnn = et / "backends" / "xnnpack" / "third-party"
    tok = Path(args.build) / "_deps" / "tokenizers_cpp_fetch-src"
    spm = tok / "sentencepiece"
    minapi = ROOT / "src" / "frontend" / "maxmsp" / "min-api"
    torch_lic, torch_ver = torch_license(args.torch_license)
    json_hpp = read(ROOT / "src" / "backend" / "json.hpp")
    json_ver = re.search(r"version (\d+\.\d+\.\d+)", json_hpp).group(1)
    return [
        ("ExecuTorch", git_rev(et, describe=True), "BSD-3-Clause", ET_EXTERNALS,
         [et / "LICENSE"]),
        ("PyTorch (c10, ATen and AOTInductor headers)", torch_ver, "BSD-3-Clause",
         ET_EXTERNALS, [torch_lic]),
        ("XNNPACK", git_rev(xnn / "XNNPACK"), "BSD-3-Clause", ET_EXTERNALS,
         [xnn / "XNNPACK" / "LICENSE"]),
        ("cpuinfo", git_rev(xnn / "cpuinfo"), "BSD-2-Clause", ET_EXTERNALS,
         [xnn / "cpuinfo" / "LICENSE"]),
        ("pthreadpool", git_rev(xnn / "pthreadpool"), "BSD-2-Clause", ET_EXTERNALS,
         [xnn / "pthreadpool" / "LICENSE"]),
        ("FXdiv", git_rev(xnn / "FXdiv"), "MIT", ET_EXTERNALS,
         [xnn / "FXdiv" / "LICENSE"]),
        ("FP16", git_rev(xnn / "FP16"), "MIT", ET_EXTERNALS,
         [xnn / "FP16" / "LICENSE"]),
        ("Eigen", "bundled with ExecuTorch", "MPL-2.0", ET_EXTERNALS,
         [et / "kernels" / "optimized" / "third-party" / "eigen" / "COPYING.MPL2"]),
        ("pocketfft", "bundled with ExecuTorch", "BSD-3-Clause", ET_EXTERNALS,
         [et / "third-party" / "pocketfft" / "LICENSE.md"]),
        ("FlatBuffers", "bundled with ExecuTorch", "Apache-2.0", ET_EXTERNALS,
         [et / "third-party" / "flatbuffers" / "LICENSE"]),
        ("flatcc", "bundled with ExecuTorch", "Apache-2.0", ET_EXTERNALS,
         [et / "third-party" / "flatcc" / "LICENSE"]),
        ("JSON for Modern C++ (nlohmann/json)", json_ver, "MIT",
         f"{ET_EXTERNALS}, {TOK_EXTERNALS}", [et / "third-party" / "json" / "LICENSE.MIT"]),
        ("tokenizers-cpp", git_rev(tok), "Apache-2.0", TOK_EXTERNALS, [tok / "LICENSE"]),
        ("SentencePiece", git_rev(spm, describe=True), "Apache-2.0", TOK_EXTERNALS,
         [spm / "LICENSE"]),
        ("Abseil", "bundled with SentencePiece", "Apache-2.0", TOK_EXTERNALS,
         [spm / "third_party" / "abseil-cpp" / "LICENSE"]),
        ("Protocol Buffers (lite)", "bundled with SentencePiece", "BSD-3-Clause",
         TOK_EXTERNALS, [spm / "third_party" / "protobuf-lite" / "LICENSE"]),
        ("Darts-clone", "bundled with SentencePiece", "BSD-2-Clause", TOK_EXTERNALS,
         [spm / "third_party" / "darts_clone" / "LICENSE"]),
        ("msgpack-c (C++)", git_rev(tok / "msgpack", describe=True), "BSL-1.0",
         TOK_EXTERNALS, [tok / "msgpack" / "LICENSE_1_0.txt"]),
        ("Min-API", git_rev(minapi), "MIT", ALL_EXTERNALS, [minapi / "License.md"]),
        ("Max SDK (max-sdk-base)", git_rev(minapi / "max-sdk-base"), "MIT",
         ALL_EXTERNALS, [minapi / "max-sdk-base" / "LICENSE.md"]),
    ]


def cargo_exe():
    exe = shutil.which("cargo")
    if exe:
        return exe
    home = Path.home() / ".cargo" / "bin" / "cargo.exe"
    if home.is_file():
        return str(home)
    fail("cargo not found (it's needed to list the tokenizer's Rust crates)")


def choose_licenses(expr):
    """The licenses we take from an SPDX expression: one per OR group."""
    expr = expr.replace("/", " OR ")
    parts = [p.strip(" ()") for p in re.split(r"\bAND\b", expr)]
    chosen = []
    for part in parts:
        options = [o.strip(" ()") for o in re.split(r"\bOR\b", part)]
        ranked = sorted(options, key=lambda o: PREFERRED.index(o)
                        if o in PREFERRED else len(PREFERRED))
        chosen.append(ranked)
    return chosen


def crate_license_files(pkg):
    """The license files of a crate for the licenses chosen from its SPDX."""
    crate_dir = Path(pkg["manifest_path"]).parent
    present = {f.name: f for f in crate_dir.iterdir() if f.is_file()}
    generic = [f for n, f in sorted(present.items())
               if n.upper().startswith(("LICENSE", "COPYING", "UNLICENSE"))]
    files = []
    for ranked in choose_licenses(pkg.get("license") or ""):
        for lic in ranked:
            hit = next((present[h] for h in FILE_HINTS.get(lic, []) if h in present), None)
            if hit:
                files.append(hit)
                break
        else:
            # A single LICENSE file holds whatever the crate's license is.
            files.extend(f for f in generic if f not in files)
    # onig_sys bundles the Oniguruma C library, which has its own license.
    if pkg["name"] == "onig_sys":
        onig = crate_dir / "oniguruma" / "COPYING"
        read(onig)  # fails if it's gone
        files.append(onig)
    if not files:
        fail(f"no license file in {crate_dir}")
    return files


def rust_components(args):
    manifest = (Path(args.build) / "_deps" / "tokenizers_cpp_fetch-src" / "rust" /
                "Cargo.toml")
    out = subprocess.run(
        [cargo_exe(), "metadata", "--offline", "--format-version", "1",
         "--filter-platform", "x86_64-pc-windows-msvc", "--manifest-path", str(manifest)],
        capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        fail(f"cargo metadata failed:\n{out.stderr}")
    meta = json.loads(out.stdout)
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    # Normal (linked) dependencies only: no dev- or build-dependencies.
    seen, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for dep in nodes[pid]["deps"]:
            if any(k["kind"] is None for k in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    seen.discard(root)
    rows = []
    for pid in sorted(seen, key=lambda i: (pkgs[i]["name"], pkgs[i]["version"])):
        p = pkgs[pid]
        lic = p.get("license") or "see license file"
        if p["name"] == "onig_sys":
            lic += " (bundles Oniguruma, BSD-2-Clause)"
        rows.append((p["name"], p["version"], lic, TOK_EXTERNALS, crate_license_files(p)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", default=str(ROOT / "build"), help="the CMake build dir")
    ap.add_argument("--executorch", default="C:/dev/executorch",
                    help="the ExecuTorch checkout the runtime was built from")
    ap.add_argument("--torch-license", help="torch's dist-info LICENSE, if torch "
                    "isn't importable from this Python")
    ap.add_argument("--out", default=str(ROOT / "THIRD_PARTY_NOTICES.md"))
    args = ap.parse_args()

    cpp = cpp_components(args)
    rust = rust_components(args)
    cuda = cuda_version(Path(args.executorch))

    # Identical license texts are printed once, listing every component using it.
    texts, users = {}, {}
    for name, version, _, _, files in cpp + rust:
        for f in files:
            text = read(f)
            key = hashlib.sha256(re.sub(r"\s+", " ", text).encode()).hexdigest()
            texts.setdefault(key, text)
            users.setdefault(key, []).append(f"{name} {version}")

    lines = [
        "# Third-party notices",
        "",
        "neural_tilde itself is licensed under CC BY-NC 4.0 (see `LICENSE`). Its "
        "Windows externals and the DLLs in `support/` also contain the third-party "
        "software below, each under its own license. This file is generated by "
        "`src/package/third_party_notices.py`.",
        "",
        "*ExecuTorch externals* below means `neural.live~`, `mc.neural.live~`, "
        "`mcs.neural.live~`, `neural.gen~` and the `support/` DLLs "
        "(`aoti_cuda_shims.dll`, `extension_cuda.dll`).",
        "",
        "## NVIDIA CUDA runtime",
        "",
        f"The ExecuTorch externals statically link the CUDA {cuda} runtime (`cudart_static`), "
        "redistributed under the NVIDIA CUDA Toolkit End User License Agreement "
        "(Attachment A lists it as redistributable): "
        "https://docs.nvidia.com/cuda/eula/. Copyright NVIDIA Corporation. The NVIDIA "
        "driver itself is not included.",
        "",
        "## C and C++ components",
        "",
        "| Component | Version | License | Used in |",
        "|---|---|---|---|",
    ]
    lines += [f"| {n} | {v} | {l} | {u} |" for n, v, l, u, _ in cpp]
    lines += [
        "",
        "## Rust crates (in `neural.tokenizer`)",
        "",
        "Where a crate offers a choice of licenses, it is used under the one whose "
        "text is included below (MIT when offered).",
        "",
        "| Crate | Version | License |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {v} | {l} |" for n, v, l, _, _ in rust]
    lines += ["", "## License texts", ""]
    for key, text in texts.items():
        lines += ["### " + ", ".join(users[key]), "", "```", text, "```", ""]

    Path(args.out).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(cpp)} C/C++ components, {len(rust)} Rust crates, "
          f"{len(texts)} distinct license texts")


if __name__ == "__main__":
    main()
