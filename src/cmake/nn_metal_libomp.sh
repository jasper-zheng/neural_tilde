#!/bin/sh
# Bundle libomp.dylib next to a Metal-delegate binary and rewrite the binary's libomp
# dependency to @rpath so it resolves from the bundle at runtime.
#
# The ExecuTorch Metal (AOTInductor) runtime and the compiled .so embedded in each .pte both
# depend on libomp at runtime. torch's libomp carries an ABSOLUTE install id
# (e.g. /opt/llvm-openmp/lib/libomp.dylib) that won't exist on other machines, and rpaths don't
# help an absolute-path load command — so we ship a copy beside the binary and repoint it to
# @rpath (the binary already carries an @loader_path rpath, or we add one here).
#
# usage: nn_metal_libomp.sh <binary> <libomp_src>
set -e
BIN="$1"
SRC="$2"
[ -n "$BIN" ] && [ -e "$BIN" ] || { echo "nn_metal_libomp: missing binary '$BIN'"; exit 0; }
[ -n "$SRC" ] && [ -e "$SRC" ] || { echo "nn_metal_libomp: missing libomp src '$SRC'"; exit 0; }
DIR=$(dirname "$BIN")

cp -f "$SRC" "$DIR/libomp.dylib"
chmod u+w "$DIR/libomp.dylib"
install_name_tool -id @rpath/libomp.dylib "$DIR/libomp.dylib" 2>/dev/null || true

# Repoint the binary's actual libomp load command (whatever absolute path it links) to @rpath.
cur=$(otool -L "$BIN" | grep -oE '[^[:space:]]*libomp[^[:space:]]*\.dylib' | head -1 || true)
if [ -n "$cur" ] && [ "$cur" != "@rpath/libomp.dylib" ]; then
  install_name_tool -change "$cur" @rpath/libomp.dylib "$BIN"
fi

# Ensure an @loader_path rpath exists (frontends already add one via LINK_FLAGS; smoke tests don't).
otool -l "$BIN" | grep -q "path @loader_path/" || install_name_tool -add_rpath @loader_path/ "$BIN"

# Re-sign (ad-hoc); the frontend .mxo also gets a final --deep sign after this.
codesign --force -s - "$DIR/libomp.dylib" 2>/dev/null || true
codesign --force -s - "$BIN" 2>/dev/null || true
echo "nn_metal_libomp: bundled libomp beside $BIN (was: ${cur:-none})"
