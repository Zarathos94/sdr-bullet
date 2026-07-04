#!/usr/bin/env bash
# Builds the WebAssembly module and generates its JavaScript bindings.
#
# The wasm-bindgen CLI version is pinned to match the crate exactly. The two negotiate a
# schema over the generated binary, and a mismatch is a hard error rather than a warning —
# so a routine `cargo update` that bumps the crate will break this step until the CLI is
# bumped to match.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${1:-wasm-release}"
OUT_DIR="web/src/wasm"

echo "building sdr-wasm ($PROFILE)"
cargo build -p sdr-wasm --target wasm32-unknown-unknown --profile "$PROFILE"

WASM="target/wasm32-unknown-unknown/$PROFILE/sdr_wasm.wasm"

echo "generating bindings"
# The default module path is left in place. The loader passes the binary's URL explicitly
# (imported through Vite's ?url so the bundler emits and fingerprints the asset), so the
# glue's own import.meta.url resolution is never relied on — which is what avoids the Vite 8
# trap where import.meta.url becomes undefined in a production worker.
wasm-bindgen "$WASM" --out-dir "$OUT_DIR" --target web

echo "wrote $OUT_DIR"
ls -la "$OUT_DIR"
