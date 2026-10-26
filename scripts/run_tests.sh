#!/usr/bin/env bash
# Build and run every test executable. Exit non-zero if any test fails.
set -uo pipefail
cd "$(dirname "$0")/.."

[ -d /usr/local/cuda-12.6/bin ] && export PATH="/usr/local/cuda-12.6/bin:$PATH"
[ -f build/CMakeCache.txt ] || cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null

names=$(ls tests | sed -E 's/\.(cpp|cu)$//' | sort -u)
cmake -S . -B build >/dev/null
fail=0
for t in $names; do
  if ! cmake --build build --target "$t" -j"$(nproc)" >/tmp/convox_build_$t.log 2>&1; then
    echo "BUILD FAIL: $t (see /tmp/convox_build_$t.log)"; fail=1; continue
  fi
  case "$t" in
    test_image) ./build/$t data/images/img01_4k.png /tmp/convox_roundtrip.png ;;
    *) ./build/$t ;;
  esac || { echo "TEST FAIL: $t"; fail=1; }
done
exit $fail
