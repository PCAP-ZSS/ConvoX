#!/usr/bin/env python3
"""ConvoX demo interface: a local web page that runs the real benchmark program on an image.

usage: python3 ui/server.py [--port 8000] [--host 127.0.0.1]

Open the printed address in a browser. Pick a sample image or upload your own, choose a filter and
filter size, press Run. The server calls ./build/convox once per implementation (serial, OpenMP and
the CUDA kernels, plus optional thread and block-size sweeps), validates every output against the
serial result, and the page draws the timings, speedups and images as they arrive.

Only the Python standard library and Pillow are used. One job runs at a time because the timings
need the CPU and GPU to themselves.
"""
import argparse
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONVOX = ROOT / "build" / "convox"
STATIC = ROOT / "ui" / "static"
RUNS = ROOT / "ui" / "runs"
PLOTS = ROOT / "results" / "plots"
SAMPLES = ROOT / "data" / "images"

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_PIXELS = 60_000_000
STEP_TIMEOUT_S = 600
FILTERS = ("gaussian", "sobel", "box")
OMP_SWEEP = (1, 2, 4, 8, 16)
BLOCK_SWEEP = ("8x8", "16x16", "32x32")
K_SWEEP = (3, 7, 11, 15)
K_SWEEP_IMPLS = ("serial", "omp", "cuda-naive", "cuda-tiled")
MPI_RANKS = (1, 2, 4, 8, 16)
MPI_BIN = ROOT / "build" / "convox_mpi"
MPI_LINE = re.compile(r"compute\s+([\d.]+) ms\s+total\s+([\d.]+) ms\s+err\s+(\S+)\s+(PASS|FAIL)")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg",
                 ".json": "application/json"}

Image.MAX_IMAGE_PIXELS = 200_000_000


class BadRequest(Exception):
    pass


# ---------------------------------------------------------------- machine info

_info_cache = None


def machine_info():
    """CPU, GPU and which implementations this build offers, from `convox info`."""
    global _info_cache
    if _info_cache is not None:
        return _info_cache
    info = {"built": CONVOX.exists(), "cpu": "", "gpu": "", "cuda": False, "max_threads": 8,
            "mpi": MPI_BIN.exists() and shutil.which("mpirun") is not None}
    if info["built"]:
        out = subprocess.run([str(CONVOX), "info"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("CPU:"):
                info["cpu"] = line.split(":", 1)[1].strip()
            elif line.startswith("GPU:"):
                info["gpu"] = line.split(":", 1)[1].strip()
                info["cuda"] = True
            elif line.startswith("OpenMP:"):
                m = re.search(r"(\d+) max threads", line)
                if m:
                    info["max_threads"] = int(m.group(1))
    _info_cache = info
    return info


def sample_images():
    out = []
    for p in sorted(SAMPLES.glob("*.png")):
        try:
            with Image.open(p) as im:
                out.append({"id": p.stem, "width": im.width, "height": im.height})
        except OSError:
            pass
    return out


def available_plots():
    names = ["omp_scaling", "omp_schedules", "runtime_vs_k", "block_sensitivity", "transfer_breakdown",
             "roofline", "amdahl", "visual_check"]
    return [n for n in names if (PLOTS / f"{n}.png").exists()]


# ------------------------------------------------------------------------ jobs


class Job:
    def __init__(self, params):
        self.id = uuid.uuid4().hex[:10]
        self.params = params
        self.dir = RUNS / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = "running"  # running | done | error
        self.error = ""
        self.steps = []  # {id, label, status, result}
        self.images = {}
        self.started = time.time()
        self.finished = None
        self.lock = threading.Lock()

    def add_step(self, label, kind, spec):
        with self.lock:
            self.steps.append({"id": len(self.steps), "label": label, "kind": kind, "impl": spec["impl"],
                               "threads": spec.get("threads", 0), "block": spec.get("block", ""),
                               "k": spec.get("k", 0), "ranks": spec.get("ranks", 0),
                               "status": "pending", "result": None})
            return self.steps[-1]

    def set_step(self, step, status, result=None):
        with self.lock:
            step["status"] = status
            if result is not None:
                step["result"] = result

    def to_json(self):
        with self.lock:
            return {"id": self.id, "state": self.state, "error": self.error, "params": self.params,
                    "steps": json.loads(json.dumps(self.steps)), "images": dict(self.images),
                    "elapsed": round((self.finished or time.time()) - self.started, 1)}


_jobs = {}
_jobs_lock = threading.Lock()
_current = None


def prepare_image(job, data, sample, maxdim):
    """Write the working image (8-bit grayscale PNG) and the display thumbnail of the original."""
    if sample:
        src = SAMPLES / f"{sample}.png"
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", sample) or not src.exists():
            raise BadRequest(f"unknown sample image '{sample}'")
        im = Image.open(src)
        name = sample
    else:
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception:
            raise BadRequest("that file is not an image the server can read (try PNG or JPG)")
        name = "uploaded image"
    if im.width * im.height > MAX_PIXELS:
        raise BadRequest(f"image is {im.width}x{im.height}; the limit is {MAX_PIXELS // 1_000_000} megapixels")
    im = im.convert("L")
    if maxdim and max(im.size) > maxdim:
        scale = maxdim / max(im.size)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    work = job.dir / "input.png"
    im.save(work)
    thumb = im.copy()
    thumb.thumbnail((1000, 1000))
    thumb.save(job.dir / "original.jpg", quality=90)
    job.images["original"] = f"/files/{job.id}/original.jpg"
    return work, name, im.size


def run_convox(job, args, timeout=STEP_TIMEOUT_S):
    cmd = [str(CONVOX), "run", "--no-csv", "--json"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if not lines:
        raise RuntimeError((proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}")[:300])
    return json.loads(lines[-1])


def run_mpi(work_path, params, ranks, timeout=STEP_TIMEOUT_S):
    """One run of the MPI program with `ranks` processes; its single result line is parsed."""
    cmd = ["mpirun", "--oversubscribe", "--bind-to", "none", "-np", str(ranks), str(MPI_BIN), "-i", str(work_path),
           "-k", str(params["k"]), "--filter", params["filter"], "--reps", str(params["reps"]), "--local", "omp",
           "--no-csv"]
    if hasattr(os, "geteuid") and os.geteuid() == 0:  # containers run as root; Open MPI refuses unless told
        cmd.insert(1, "--allow-run-as-root")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    m = MPI_LINE.search(proc.stdout)
    if not m:
        raise RuntimeError((proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}")[:300])
    compute, total, err, verdict = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)
    return {"impl": "mpi-omp", "ranks": ranks, "kernel_ms": compute, "total_ms": total, "h2d_ms": 0.0, "d2h_ms": 0.0,
            "max_abs_err": err, "pass": verdict == "PASS"}


def worker(job, work_path, params):
    global _current
    info = machine_info()
    p = params
    base = ["-i", str(work_path), "-k", str(p["k"]), "--filter", p["filter"], "--reps", str(p["reps"])]
    memo = {}

    plan = [("serial", "Serial (one core)", {"impl": "serial"}),
            ("omp", f"OpenMP, {p['threads']} threads", {"impl": "omp", "threads": p["threads"]})]
    if info["cuda"]:
        plan += [("cuda-naive", "CUDA naive (global memory)", {"impl": "cuda-naive", "block": "16x16"}),
                 ("cuda-const", "CUDA + constant memory", {"impl": "cuda-const", "block": "16x16"}),
                 ("cuda-tiled", "CUDA + constant + shared tiles", {"impl": "cuda-tiled", "block": "16x16"})]
    sweeps = []
    if p["omp_sweep"]:
        sweeps += [(f"omp-{t}", f"OpenMP scaling: {t} thread{'s' if t > 1 else ''}", {"impl": "omp", "threads": t},
                    "omp_sweep") for t in OMP_SWEEP]
    if p["block_sweep"] and info["cuda"]:
        sweeps += [(f"{impl}-{b}", f"CUDA block {b}: {impl.replace('cuda-', '')}", {"impl": impl, "block": b},
                    "block_sweep") for impl in ("cuda-naive", "cuda-tiled") for b in BLOCK_SWEEP]

    if p["k_sweep"]:
        sweeps += [(f"{impl}-k{k}", f"Filter size {k}×{k}: {impl.replace('cuda-', 'CUDA ')}", {"impl": impl, "k": k,
                    **({"threads": p["threads"]} if impl == "omp" else {}), **({"block": "16x16"} if impl.startswith("cuda") else {})},
                    "k_sweep") for k in K_SWEEP for impl in K_SWEEP_IMPLS if info["cuda"] or not impl.startswith("cuda")]
    if p["mpi_sweep"] and info["mpi"]:
        sweeps += [(f"mpi-{n}", f"MPI, {n} process{'es' if n > 1 else ''}", {"impl": "mpi-omp", "ranks": n}, "mpi_sweep")
                   for n in MPI_RANKS]

    display_impl = "cuda-tiled" if info["cuda"] else "omp"
    steps = [(job.add_step(label, "main", spec), key, spec, False) for key, label, spec in plan]
    steps += [(job.add_step(label, kind, spec), key, spec, True) for key, label, spec, kind in sweeps]
    try:
        for step, key, spec, is_sweep in steps:
            ident = (spec["impl"], spec.get("threads", 0), spec.get("block", ""), spec.get("k", p["k"]), spec.get("ranks", 0))
            if ident in memo:  # same configuration already measured in this job
                job.set_step(step, "done", dict(memo[ident], reused=True))
                continue
            job.set_step(step, "running")
            args = base + ["--impl", spec["impl"]]
            if "k" in spec:
                args[args.index("-k") + 1] = str(spec["k"])
            if "threads" in spec:
                args += ["--threads", str(spec["threads"])]
            if "block" in spec:
                args += ["--block", spec["block"]]
            out_png = None
            if spec["impl"] == display_impl and not is_sweep:
                out_png = job.dir / "output.png"
                args += ["--save-image", str(out_png)]
                if p["filter"] == "sobel":
                    args += ["--save-mode", "abs"]
            try:
                result = run_mpi(work_path, p, spec["ranks"]) if "ranks" in spec else run_convox(job, args)
            except subprocess.TimeoutExpired:
                job.set_step(step, "failed", {"error": "timed out"})
                continue
            except Exception as e:  # noqa: BLE001 - report any failure of one step and keep going
                job.set_step(step, "failed", {"error": str(e)})
                continue
            memo[ident] = result
            job.set_step(step, "done", result)
            if out_png and out_png.exists():
                with Image.open(out_png) as im:
                    im.thumbnail((1000, 1000))
                    im.convert("L").save(job.dir / "output.jpg", quality=90)
                with job.lock:
                    job.images["output"] = f"/files/{job.id}/output.jpg"
                    job.images["output_impl"] = spec["impl"]
        with job.lock:
            job.state = "done"
    except Exception as e:  # noqa: BLE001
        with job.lock:
            job.state = "error"
            job.error = str(e)
    finally:
        job.finished = time.time()
        with _jobs_lock:
            _current = None


def prune_old_jobs(keep=6):
    dirs = sorted([d for d in RUNS.glob("*") if d.is_dir()], key=lambda d: d.stat().st_mtime)
    for d in dirs[:-keep]:
        shutil.rmtree(d, ignore_errors=True)
    with _jobs_lock:
        for jid in [j for j in _jobs if not (RUNS / j).exists()]:
            del _jobs[jid]


def parse_params(query):
    def one(name, default=None):
        v = query.get(name, [default])[0]
        return default if v is None else v

    def integer(name, default, lo, hi):
        try:
            v = int(one(name, default))
        except ValueError:
            raise BadRequest(f"{name} must be a whole number")
        if not lo <= v <= hi:
            raise BadRequest(f"{name} must be between {lo} and {hi}")
        return v

    flt = one("filter", "gaussian")
    if flt not in FILTERS:
        raise BadRequest(f"filter must be one of {', '.join(FILTERS)}")
    k = integer("k", 7, 3, 15)
    if k % 2 == 0:
        raise BadRequest("filter size K must be odd")
    return {"filter": flt, "k": k, "reps": integer("reps", 3, 1, 10),
            "threads": integer("threads", machine_info()["max_threads"], 1, 64),
            "maxdim": integer("maxdim", 0, 0, 20000),
            "omp_sweep": one("omp_sweep", "0") == "1", "block_sweep": one("block_sweep", "0") == "1",
            "k_sweep": one("k_sweep", "0") == "1", "mpi_sweep": one("mpi_sweep", "0") == "1",
            "sample": one("sample", "")}


def start_job(query, body):
    global _current
    if not CONVOX.exists():
        raise BadRequest("the benchmark program is not built; run: bash scripts/build.sh")
    params = parse_params(query)
    if not params["sample"] and not body:
        raise BadRequest("choose a sample image or upload one")
    with _jobs_lock:
        if _current is not None:
            raise RuntimeError("busy")
        RUNS.mkdir(parents=True, exist_ok=True)
        job = Job(params)
        _jobs[job.id] = job
        _current = job.id
    try:
        work, name, size = prepare_image(job, body, params["sample"], params["maxdim"])
    except Exception:
        with _jobs_lock:
            _current = None
            _jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)
        raise
    job.params.update({"image_name": name, "width": size[0], "height": size[1]})
    prune_old_jobs()
    threading.Thread(target=worker, args=(job, work, job.params), daemon=True).start()
    return job


# ---------------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    server_version = "ConvoXUI"

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def send_bytes(self, status, data, ctype, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=60" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, status=200):
        self.send_bytes(status, json.dumps(obj).encode(), "application/json")

    def send_file(self, path, cache=False):
        if not path.is_file():
            return self.send_json({"error": "not found"}, 404)
        self.send_bytes(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"), cache)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        path = url.path
        try:
            if path in ("/", "/index.html"):
                return self.send_file(STATIC / "index.html")
            if path.startswith("/static/"):
                name = pathlib.PurePosixPath(path).name
                return self.send_file(STATIC / name)
            if path == "/api/info":
                info = dict(machine_info())
                info.update(samples=sample_images(), plots=available_plots(), filters=list(FILTERS))
                return self.send_json(info)
            m = re.fullmatch(r"/api/job/([0-9a-f]{10})", path)
            if m:
                job = _jobs.get(m.group(1))
                return self.send_json(job.to_json()) if job else self.send_json({"error": "unknown job"}, 404)
            m = re.fullmatch(r"/files/([0-9a-f]{10})/(original|output)\.jpg", path)
            if m:
                return self.send_file(RUNS / m.group(1) / f"{m.group(2)}.jpg")
            m = re.fullmatch(r"/plots/([a-z_]+)\.png", path)
            if m:
                return self.send_file(PLOTS / f"{m.group(1)}.png", cache=True)
            self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/api/run":
            return self.send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_UPLOAD_BYTES:
                return self.send_json({"error": "file too large (limit 200 MB)"}, 413)
            body = self.rfile.read(length) if length else b""
            job = start_job(parse_qs(url.query), body)
            self.send_json({"job": job.id})
        except BadRequest as e:
            self.send_json({"error": str(e)}, 400)
        except RuntimeError:
            self.send_json({"error": "A run is already in progress. Wait for it to finish."}, 409)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    if not CONVOX.exists():
        print("note: build/convox is missing; run  bash scripts/build.sh  first", file=sys.stderr)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    info = machine_info()
    print(f"ConvoX demo running at http://localhost:{a.port}")
    print(f"  CPU: {info['cpu'] or 'unknown'}   GPU: {info['gpu'] or 'none (CUDA versions will be skipped)'}")
    print("  Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
