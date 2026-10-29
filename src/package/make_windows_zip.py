"""Make the Windows release zip: releases/neural_tilde_<version>_windows_x64.zip.

Run after `cmake --build build --config Release` (and after
third_party_notices.py, whose output it packs):

    python src/package/make_windows_zip.py [--build build]

The zip holds one neural_tilde/ folder, to unzip into Documents/Max 9/Packages:
the 7 externals (.mxe64), the CUDA runtime DLLs in support/, the reference
pages in docs/, and the README, PROTOCOL, LICENSE, third-party notices and
install notes. The help patches and their example models are staged by hand:
with --stage releases/neural_tilde, the files above are copied into that
folder and the whole folder is zipped, help/ included.

Before writing it, every packed binary's DLL imports are checked: each must be
a Windows DLL, a Max DLL, the Microsoft C++ runtime, or a DLL in support/.
Anything else (a debug runtime, nvcuda.dll, a DLL we forgot to ship) fails.
The same inputs give the same zip, byte for byte (sorted entries, fixed
timestamps), and a .sha256 file is written next to it.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXTERNALS = ["neural.live~", "mc.neural.live~", "mcs.neural.live~", "neural.gen~",
             "neural.tokenizer", "neural.info", "neural.gaussianize"]
SUPPORT_DLLS = ["aoti_cuda_shims.dll", "extension_cuda.dll"]
TEXT_FILES = ["README.md", "PROTOCOL.md", "LICENSE", "THIRD_PARTY_NOTICES.md"]

# Imports a packed binary may have, besides the DLLs in support/ (lower case).
MAX_DLLS = {"maxapi.dll", "maxaudio.dll", "jitlib.dll"}
# The Visual C++ Redistributable's DLLs (the user installs it; see INSTALL).
VC_RUNTIME_DLLS = {"msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
                   "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
                   "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll"}
# Windows' own DLLs seen in our imports so far. Kept explicit, rather than
# "anything in System32", because the NVIDIA driver installs nvcuda.dll there
# too, and importing it would stop the externals loading on machines without it.
WINDOWS_DLLS = {"kernel32.dll", "ntdll.dll", "bcrypt.dll", "bcryptprimitives.dll",
                "advapi32.dll", "user32.dll", "shell32.dll", "ole32.dll",
                "oleaut32.dll", "ws2_32.dll", "userenv.dll", "iphlpapi.dll",
                "psapi.dll", "version.dll", "dbghelp.dll", "secur32.dll",
                "crypt32.dll", "shlwapi.dll", "synchronization.dll"}

# Delegates whose models only run on macOS (warned about if packed).
MACOS_ONLY_DELEGATES = {"coreml", "mlx", "mps", "metal"}

# One fixed timestamp for every entry, so the zip only changes with its content.
ZIP_DATE = (2026, 1, 1, 0, 0, 0)

INSTALL_TEXT = """\
neural_tilde {version} for Windows x64
====================================

Install
-------
1. Quit Max.
2. Unzip, and move the neural_tilde folder into your Max Packages folder,
   e.g. Documents\\Max 9\\Packages\\neural_tilde
3. Install (or update) the Microsoft Visual C++ Redistributable (x64), version
   {vc_runtime} or newer: https://aka.ms/vs/17/release/vc_redist.x64.exe
   Without it, Max reports an error loading neural.live~ and the other model
   objects, or the Max console says which version is needed.
4. Start Max.

Running models
--------------
- Models are .pte files (plus a .json, and for CUDA models a .ptd) exported
  with the neural_tilde Python tools. Put them in a folder on Max's search path
  or next to your patch.
- CUDA models (delegate "cuda") need an NVIDIA GPU (Turing / RTX 20 series or
  newer) and an NVIDIA driver 580 or newer. Nothing else from CUDA needs to be
  installed.
- CPU models (delegate "xnnpack") run on any machine, with or without an
  NVIDIA GPU. On a machine without a usable NVIDIA GPU, a CUDA model posts the
  reason in the Max console and stays silent.
- [neural.info] shows a model's metadata and this machine's GPU and runtime
  (the "host" entry).

See README.md for the objects and PROTOCOL.md for the model format.
"""


def fail(msg):
    sys.exit(f"make_windows_zip: {msg}")


def version():
    m = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(),
                  re.MULTILINE)
    if not m:
        fail("no version in pyproject.toml")
    return m.group(1)


def find_dumpbin():
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.is_file():
        fail("vswhere.exe not found; is Visual Studio 2022 installed?")
    vs = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property",
                         "installationPath"], capture_output=True, text=True).stdout.strip()
    hits = sorted(Path(vs).glob("VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"))
    if not hits:
        fail(f"dumpbin.exe not found under {vs}")
    return hits[-1]


def imports(dumpbin, path):
    out = subprocess.run([str(dumpbin), "/nologo", "/dependents", str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        fail(f"dumpbin failed on {path}:\n{out.stdout}{out.stderr}")
    return [l.strip() for l in out.stdout.splitlines() if re.match(r"^\s+\S+\.dll\s*$", l, re.I)]


def vc_runtime_required(dumpbin, path):
    """The toolset a binary was linked with (= the oldest runtime it supports)."""
    out = subprocess.run([str(dumpbin), "/nologo", "/headers", str(path)],
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+)\.(\d+) linker version", out)
    if not m:
        fail(f"no linker version in {path}")
    return int(m.group(1)), int(m.group(2))


def check_imports(dumpbin, binaries):
    shipped = {d.lower() for d in SUPPORT_DLLS}
    problems = []
    for path in binaries:
        for dll in imports(dumpbin, path):
            low = dll.lower()
            if low in shipped or low in MAX_DLLS or low in VC_RUNTIME_DLLS:
                continue
            if low in WINDOWS_DLLS or low.startswith(("api-ms-win-", "ext-ms-win-")):
                continue
            hint = (" (a debug runtime: build with --config Release)"
                    if re.match(r"(msvcp|vcruntime)\d+.*d\.dll$|ucrtbased\.dll$", low)
                    else "")
            problems.append(f"{path.name} imports {dll}{hint}")
    if problems:
        fail("unexpected DLL imports (ship them in support/, or add a Windows DLL to "
             "WINDOWS_DLLS):\n  " + "\n  ".join(problems))


def gather(build):
    """[(path in the zip, source file)], sorted."""
    files = []
    for name in EXTERNALS:
        files.append((f"externals/{name}.mxe64", ROOT / "src" / "externals" / f"{name}.mxe64"))
    for dll in SUPPORT_DLLS:
        files.append((f"support/{dll}", build / "support" / dll))
    for name in EXTERNALS:
        files.append((f"docs/{name}.maxref.xml", ROOT / "docs" / f"{name}.maxref.xml"))
    for name in TEXT_FILES:
        files.append((name, ROOT / name))
    missing = [str(src) for _, src in files if not src.is_file()]
    if missing:
        fail("missing inputs (build first; THIRD_PARTY_NOTICES.md comes from "
             "third_party_notices.py):\n  " + "\n  ".join(missing))
    return sorted(files)


def install_bytes(text):
    return text.replace("\n", "\r\n").encode()


def stage(stage_dir, entries, install):
    """Copy the packed files into stage_dir (the release's neural_tilde folder),
    then return everything in it, so files added there by hand (help/) are
    packed too."""
    for arc, src in entries:
        dst = stage_dir / arc
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(install_bytes(install) if src is None else src.read_bytes())
    return sorted((p.relative_to(stage_dir).as_posix(), p)
                  for p in stage_dir.rglob("*")
                  if p.is_file() and p.name not in (".DS_Store", "Thumbs.db")
                  and not p.name.startswith("._"))


def warn_macos_only_models(entries):
    """Model JSONs in the zip whose delegate can't run on Windows."""
    for arc, src in entries:
        if src is None or not arc.endswith(".json"):
            continue
        try:
            meta = json.loads(src.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        delegate = meta.get("delegate") if isinstance(meta, dict) else None
        if delegate in MACOS_ONLY_DELEGATES:
            print(f"warning: {arc} is a {delegate} model, which only runs on macOS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", default=str(ROOT / "build"), help="the CMake build dir")
    ap.add_argument("--out-dir", default=str(ROOT / "releases"))
    ap.add_argument("--stage", help="a release folder (e.g. releases/neural_tilde): copy "
                    "the build into it and zip all of it, including what's already "
                    "there, such as help/")
    args = ap.parse_args()

    ver = version()
    files = gather(Path(args.build))
    dumpbin = find_dumpbin()
    binaries = [src for _, src in files if src.suffix in (".mxe64", ".dll")]
    check_imports(dumpbin, binaries)
    need = max(vc_runtime_required(dumpbin, b) for b in binaries)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"neural_tilde_{ver}_windows_x64.zip"
    install = INSTALL_TEXT.format(version=ver, vc_runtime=f"{need[0]}.{need[1]}")
    entries = files + [("INSTALL-windows.txt", None)]

    if args.stage:
        entries = stage(Path(args.stage), entries, install)

    names = [arc for arc, _ in entries]
    # Models only as help-patch examples; never macOS bundles.
    bad = [n for n in names if n.endswith(".mxo") or ".mxo/" in n or
           (n.endswith((".pte", ".ptd")) and not n.startswith("help/"))]
    if bad:
        fail(f"these don't belong in the Windows zip: {bad}")
    warn_macos_only_models(entries)

    def add(zf, arc, data):
        info = zipfile.ZipInfo(f"neural_tilde/{arc}", date_time=ZIP_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, data, compresslevel=9)

    with zipfile.ZipFile(zip_path, "w") as zf:
        for arc, src in sorted(entries, key=lambda e: e[0]):
            add(zf, arc, install_bytes(install) if src is None else src.read_bytes())

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (out_dir / (zip_path.name + ".sha256")).write_text(f"{digest}  {zip_path.name}\n")

    print(f"{zip_path}  ({zip_path.stat().st_size / 1048576:.1f} MB)")
    print(f"sha256 {digest}")
    print(f"needs the Visual C++ runtime {need[0]}.{need[1]}+; DLL imports OK")
    for arc, src in sorted(entries, key=lambda e: e[0]):
        size = len(install) if src is None else src.stat().st_size
        print(f"  {size:>12,}  neural_tilde/{arc}")


if __name__ == "__main__":
    main()
