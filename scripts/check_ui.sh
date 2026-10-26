#!/usr/bin/env bash
# End-to-end check of the demo interface: start the server, run one job through the same HTTP calls
# the page makes, and verify the results. Exit non-zero on any failure.
cd "$(dirname "$0")/.."
port=${1:-8765}
base=http://127.0.0.1:$port
fail=0
ok()   { echo "ok   $1"; }
bad()  { echo "FAIL $1"; fail=1; }

python3 ui/server.py --port "$port" >/tmp/convox_ui.log 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null' EXIT
for _ in $(seq 1 50); do curl -s "$base/api/info" >/dev/null && break; sleep 0.2; done

curl -s "$base/" | grep -q "ConvoX" && ok "page served" || bad "page served"
curl -s "$base/static/app.js" | grep -q "api/run" && ok "script served" || bad "script served"
info=$(curl -s "$base/api/info")
echo "$info" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['built'] and d['samples']" && ok "info lists samples" || bad "info lists samples"

# bad requests are refused with a message
code=$(curl -s -o /tmp/convox_ui_err.json -w '%{http_code}' -X POST "$base/api/run?k=8&sample=img01_4k")
[ "$code" = 400 ] && ok "even K rejected (400)" || bad "even K rejected (got $code)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$base/api/run?k=7")
[ "$code" = 400 ] && ok "missing image rejected (400)" || bad "missing image rejected (got $code)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST --data-binary 'not an image' "$base/api/run?k=7")
[ "$code" = 400 ] && ok "non-image upload rejected (400)" || bad "non-image upload rejected (got $code)"

# a real job on an uploaded image, with both sweeps
python3 - <<'EOF'
from PIL import Image
Image.open("data/images/img01_4k.png").resize((640, 360)).convert("RGB").save("/tmp/convox_ui_upload.jpg")
EOF
job=$(curl -s -X POST --data-binary @/tmp/convox_ui_upload.jpg "$base/api/run?filter=sobel&k=5&reps=1&omp_sweep=1&block_sweep=1" | python3 -c "import json,sys; print(json.load(sys.stdin)['job'])")
[ -n "$job" ] && ok "job started ($job)" || { bad "job started"; exit 1; }

# a second job while the first runs must be refused
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$base/api/run?k=3&sample=img01_4k&reps=1")
[ "$code" = 409 ] && ok "second job refused while busy (409)" || bad "second job refused (got $code)"

for _ in $(seq 1 240); do
  state=$(curl -s "$base/api/job/$job" | python3 -c "import json,sys; print(json.load(sys.stdin)['state'])")
  [ "$state" != running ] && break
  sleep 0.5
done
[ "$state" = done ] && ok "job finished" || bad "job finished (state=$state)"

curl -s "$base/api/job/$job" > /tmp/convox_ui_job.json
python3 - <<'EOF' && ok "results complete and valid" || bad "results complete and valid"
import json
j = json.load(open("/tmp/convox_ui_job.json"))
steps = j["steps"]
assert all(s["status"] == "done" for s in steps), [s["label"] for s in steps if s["status"] != "done"]
main = [s["result"] for s in steps if s["kind"] == "main"]
assert {r["impl"] for r in main} >= {"serial", "omp"}, "missing implementations"
assert all(r["pass"] for r in main), "a result failed validation"
assert len([s for s in steps if s["kind"] == "omp_sweep"]) == 5
assert "original" in j["images"] and "output" in j["images"]
print("  versions:", ", ".join(r["impl"] for r in main), "| steps:", len(steps))
EOF

for f in original output; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$base/files/$job/$f.jpg")
  [ "$code" = 200 ] && ok "$f image served" || bad "$f image served (got $code)"
done
code=$(curl -s -o /dev/null -w '%{http_code}' "$base/files/$job/../../../etc/passwd")
[ "$code" = 404 ] && ok "path traversal refused" || bad "path traversal (got $code)"
code=$(curl -s -o /dev/null -w '%{http_code}' "$base/plots/omp_scaling.png")
{ [ "$code" = 200 ] || [ "$code" = 404 ]; } && ok "plots route ($code)" || bad "plots route"
exit $fail
