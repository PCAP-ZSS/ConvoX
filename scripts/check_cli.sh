#!/usr/bin/env bash
# Smoke-check the convox CLI exit codes (0 ok, 1 validation, 2 bad args, 3 unavailable).
cd "$(dirname "$0")/.."
img=data/images/img01_4k.png
expect() {  # expect <code> <description> <command...>
  want=$1; desc=$2; shift 2
  "$@" >/dev/null 2>&1; got=$?
  if [ "$got" -eq "$want" ]; then echo "ok   $desc (exit $got)"; else echo "FAIL $desc: want $want, got $got"; fail=1; fi
}
fail=0
expect 2 "even K rejected"            ./build/convox run -i $img -k 8 --no-csv
expect 2 "missing image rejected"     ./build/convox run -i nope.png --no-csv
expect 2 "missing --image rejected"   ./build/convox run --no-csv
expect 2 "no subcommand"              ./build/convox
expect 3 "unavailable impl"           ./build/convox run --impl cuda-tiled -i $img --no-csv
expect 0 "serial run passes"          ./build/convox run -i $img -k 3 --reps 1 --no-csv
exit $fail
