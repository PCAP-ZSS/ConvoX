/* ConvoX demo page: talks to ui/server.py, which runs the real ./build/convox program. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  const IMPL = {
    "serial": { name: "Serial (1 core)", color: "#555555" },
    "omp": { name: "OpenMP", color: "#0072B2" },
    "cuda-naive": { name: "CUDA naive", color: "#E69F00" },
    "cuda-const": { name: "CUDA + constant memory", color: "#009E73" },
    "cuda-tiled": { name: "CUDA + shared tiles", color: "#D55E00" },
  };
  const C_IN = "#56B4E9", C_KERNEL = "#D55E00", C_OUT = "#009E73";
  const FILTER_NAME = { gaussian: "Gaussian blur", sobel: "edge detection (Sobel)", box: "box blur" };
  const PLOT_INFO = {
    omp_scaling: ["OpenMP speedup vs number of threads", "Speed grows with threads up to about 4, then flattens: the memory cannot feed more threads."],
    omp_schedules: ["Ways of sharing rows between threads", "Giving out single pixels (collapse with chunk 1) is about 5 times slower than giving out rows."],
    runtime_vs_k: ["Time against filter size", "Every version slows as the filter grows; the GPU stays far below the CPU."],
    block_sensitivity: ["GPU thread-block shape", "16 × 16 blocks are fastest. Tiling pays off more as the filter grows."],
    transfer_breakdown: ["Where the GPU's time goes", "Copying the image to and from the GPU is most of the whole-job time."],
    roofline: ["Memory limit vs compute limit", "Tiling moves the kernel toward the compute limit. The naive points sit above the memory roof because the caches already absorb the reuse."],
    amdahl: ["Amdahl's law check", "Reading and saving the file cannot be sped up, so it caps the whole-job speedup at 1.1 to 2.6 times."],
    visual_check: ["Visual check", "The same crop of the original and of the blurred and edge outputs: the results look right."],
  };

  const state = { info: null, mode: "sample", file: null, timer: null, running: false, sig: "", timeView: "all", job: null };

  // ------------------------------------------------------------ formatting
  const fmtMs = (v) => (v >= 1000 ? (v / 1000).toFixed(2) + " s" : v >= 100 ? v.toFixed(0) + " ms" : v >= 10 ? v.toFixed(1) + " ms" : v.toFixed(2) + " ms");
  const fmtX = (v) => (v >= 100 ? Math.round(v).toLocaleString() : v >= 10 ? v.toFixed(0) : v.toFixed(1)) + "×";
  const sci = (v) => (v === 0 ? "0" : v.toExponential(1));
  const implName = (r) => (r.impl === "omp" ? `OpenMP (${r.threads} threads)` : IMPL[r.impl].name);

  function niceTicks(max, n = 4) {
    const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const ticks = [];
    for (let v = 0; ; v += step) {
      ticks.push(+v.toFixed(6));
      if (v >= max) break;  // the last tick always reaches the largest value
    }
    return ticks;
  }

  // ---------------------------------------------------------------- setup
  async function init() {
    const res = await fetch("/api/info");
    const info = (state.info = await res.json());

    const chips = $("#machine");
    chips.innerHTML = "";
    const chip = (label, value) => {
      const d = document.createElement("span");
      d.className = "chip";
      d.innerHTML = `${label} <b></b>`;
      $("b", d).textContent = value;
      chips.appendChild(d);
    };
    chip("CPU", info.cpu || "unknown");
    chip("GPU", info.gpu || "none found");

    if (!info.built) setMsg("The benchmark program is not built. In Ubuntu run: bash scripts/build.sh", true);

    const sel = $("#sample");
    info.samples.forEach((s) => {
      const o = document.createElement("option");
      o.value = s.id;
      o.textContent = `${s.id}  (${s.width} × ${s.height})`;
      sel.appendChild(o);
    });
    if (!info.samples.length) {
      sel.innerHTML = '<option value="">No sample images (run scripts/gen_images.py)</option>';
    }

    const th = $("#threads");
    const options = [1, 2, 4, 8, 12, 16, 24, 32].filter((t) => t < info.max_threads);
    options.push(info.max_threads);
    options.forEach((t) => {
      const o = document.createElement("option");
      o.value = t;
      o.textContent = t === info.max_threads ? `${t} (all)` : `${t}`;
      th.appendChild(o);
    });
    th.value = info.max_threads;

    if (!info.cuda) $("#block-check").classList.add("hidden");
    if (!info.mpi) $("#mpi-check").classList.add("hidden");

    const plots = $("#plots");
    info.plots.forEach((name) => {
      const [title, caption] = PLOT_INFO[name] || [name, ""];
      const fig = document.createElement("figure");
      fig.className = "card";
      fig.style.margin = "0";
      fig.innerHTML = `<h3></h3><img loading="lazy" alt=""><p class="small muted" style="margin-top:8px"></p>`;
      $("h3", fig).textContent = title;
      $("img", fig).src = `/plots/${name}.png`;
      $("img", fig).alt = title;
      $("p", fig).textContent = caption;
      plots.appendChild(fig);
    });
    if (!info.plots.length) $("#noplots").textContent = "No saved plots yet. Run: python3 analysis/plots.py";
  }

  function setMsg(text, isErr) {
    const m = $("#msg");
    m.textContent = text;
    m.className = "msg " + (isErr ? "err" : "muted");
  }

  // ------------------------------------------------------------ controls
  $$(".tabs button").forEach((b) =>
    b.addEventListener("click", () => {
      $$(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
      $$(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
    })
  );

  $$("#source-seg button").forEach((b) =>
    b.addEventListener("click", () => {
      state.mode = b.dataset.mode;
      $$("#source-seg button").forEach((x) => x.classList.toggle("active", x === b));
      $("#sample-box").classList.toggle("hidden", state.mode !== "sample");
      $("#upload-box").classList.toggle("hidden", state.mode !== "upload");
    })
  );

  const drop = $("#drop"), fileInput = $("#file");
  drop.addEventListener("click", () => fileInput.click());
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) chooseFile(e.dataTransfer.files[0]); });
  fileInput.addEventListener("click", (e) => e.stopPropagation());
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) chooseFile(fileInput.files[0]); });

  function chooseFile(f) {
    state.file = f;
    const img = $("#preview");
    img.src = URL.createObjectURL(f);
    img.classList.remove("hidden");
    $("#fileinfo").textContent = `${f.name} (${(f.size / 1048576).toFixed(1)} MB)`;
  }

  $$("#filters input").forEach((r) =>
    r.addEventListener("change", () => $$("#filters label").forEach((l) => l.classList.toggle("on", $("input", l).checked)))
  );
  $("#k").addEventListener("input", (e) => ($("#kval").textContent = `${e.target.value} × ${e.target.value}`));
  $("#controls").addEventListener("submit", run);

  // ------------------------------------------------------------------ run
  function setRunning(on) {
    state.running = on;
    const b = $("#run");
    b.disabled = on;
    b.textContent = on ? "Running…" : "Run";
  }

  async function run(e) {
    e.preventDefault();
    if (state.running) return;
    if (!state.info.built) return setMsg("Build the program first: bash scripts/build.sh", true);
    const q = new URLSearchParams({
      filter: $("input[name=filter]:checked").value,
      k: $("#k").value,
      reps: $("#reps").value,
      threads: $("#threads").value,
      maxdim: $("#maxdim").value,
      omp_sweep: $("#omp_sweep").checked ? 1 : 0,
      block_sweep: $("#block_sweep").checked && state.info.cuda ? 1 : 0,
      k_sweep: $("#k_sweep").checked ? 1 : 0,
      mpi_sweep: $("#mpi_sweep").checked && state.info.mpi ? 1 : 0,
    });
    let body = null;
    if (state.mode === "sample") {
      if (!$("#sample").value) return setMsg("No sample image available. Upload one instead.", true);
      q.set("sample", $("#sample").value);
    } else {
      if (!state.file) return setMsg("Choose an image to upload first.", true);
      body = state.file;
    }
    setMsg("");
    setRunning(true);
    resetResults();
    try {
      const r = await fetch("/api/run?" + q, { method: "POST", body, headers: { "Content-Type": "application/octet-stream" } });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "the server refused the request");
      poll(data.job);
    } catch (err) {
      setRunning(false);
      setMsg(err.message, true);
      $("#status-title").textContent = "Could not start";
    }
  }

  function resetResults() {
    state.sig = "";
    state.timeView = "all";
    $("#steps").innerHTML = "";
    ["#images-card", "#summary-card", "#charts"].forEach((s) => $(s).classList.add("hidden"));
    $("#img-output").hidden = true;
    $("#frame-output span").hidden = false;
    $("#status-title").textContent = "Starting…";
    $("#status-sub").textContent = "";
  }

  function poll(id) {
    clearInterval(state.timer);
    const tick = async () => {
      try {
        const job = await (await fetch("/api/job/" + id)).json();
        state.job = job;
        render(job);
        if (job.state !== "running") {
          clearInterval(state.timer);
          setRunning(false);
        }
      } catch (err) {
        clearInterval(state.timer);
        setRunning(false);
        setMsg("Lost contact with the server.", true);
      }
    };
    tick();
    state.timer = setInterval(tick, 700);
  }

  // --------------------------------------------------------------- render
  function render(job) {
    const p = job.params;
    const done = job.steps.filter((s) => s.status === "done" || s.status === "failed").length;
    const current = job.steps.find((s) => s.status === "running");
    if (job.state === "running") {
      $("#status-title").textContent = current ? `Running ${done + 1} of ${job.steps.length}: ${current.label}` : "Preparing the image…";
    } else if (job.state === "error") {
      $("#status-title").textContent = "Something went wrong";
      setMsg(job.error, true);
    } else {
      const failed = job.steps.filter((s) => s.status === "failed").length;
      $("#status-title").textContent = `Finished in ${job.elapsed} s` + (failed ? ` (${failed} step${failed > 1 ? "s" : ""} failed)` : "");
    }
    $("#status-sub").textContent = `${p.image_name} · ${p.width} × ${p.height} px · ${FILTER_NAME[p.filter]}, ${p.k} × ${p.k} filter · ${p.reps} repeat${p.reps > 1 ? "s" : ""}`;

    const stepsEl = $("#steps");
    stepsEl.innerHTML = "";
    job.steps.forEach((s) => {
      const d = document.createElement("span");
      d.className = "step " + s.status;
      d.textContent = (s.status === "done" ? "✓ " : s.status === "failed" ? "✗ " : "") + s.label;
      if (s.status === "failed" && s.result) d.title = s.result.error;
      stepsEl.appendChild(d);
    });

    renderImages(job);

    const sig = JSON.stringify(job.steps.map((s) => s.status)) + (job.images.output || "");
    if (sig === state.sig) return;
    state.sig = sig;
    renderResults(job);
  }

  function renderImages(job) {
    if (!job.images.original) return;
    $("#images-card").classList.remove("hidden");
    const o = $("#img-original");
    if (o.dataset.src !== job.images.original) {
      o.src = job.images.original;
      o.dataset.src = job.images.original;
      o.hidden = false;
    }
    const out = $("#img-output");
    if (job.images.output && out.dataset.src !== job.images.output) {
      out.src = job.images.output;
      out.dataset.src = job.images.output;
      out.hidden = false;
      $("#frame-output span").hidden = true;
      const p = job.params;
      $("#cap-output").textContent = `Result: ${FILTER_NAME[p.filter]}, ${p.k} × ${p.k}, computed by ${IMPL[job.images.output_impl].name}`;
    }
  }

  function mainResults(job) {
    return job.steps.filter((s) => s.kind === "main" && s.status === "done" && s.result).map((s) => s.result);
  }

  function renderResults(job) {
    const main = mainResults(job);
    if (!main.length) return;
    const serial = main.find((r) => r.impl === "serial");
    $("#summary-card").classList.remove("hidden");
    $("#charts").classList.remove("hidden");

    // table
    const tbody = $("#table tbody");
    tbody.innerHTML = "";
    main.forEach((r) => {
      const gpu = r.impl.startsWith("cuda");
      const speed = serial ? serial.total_ms / r.total_ms : null;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><span class="dot" style="background:${IMPL[r.impl].color}"></span></td><td></td><td></td><td></td><td></td><td></td>`;
      const c = tr.children;
      c[0].append(implName(r));
      c[1].textContent = fmtMs(r.kernel_ms);
      c[2].textContent = gpu ? fmtMs(r.h2d_ms + r.d2h_ms) : "–";
      c[3].textContent = fmtMs(r.total_ms);
      c[4].textContent = r.impl === "serial" ? "1× (reference)" : speed ? fmtX(speed) : "–";
      c[5].innerHTML = r.pass ? `<span class="pass">✓ matches</span> <span class="muted small">(max diff ${sci(r.max_abs_err)})</span>` : `<span class="fail">✗ differs (${sci(r.max_abs_err)})</span>`;
      tbody.appendChild(tr);
    });

    // headline
    const others = main.filter((r) => r.impl !== "serial");
    const h = $("#headline");
    if (serial && others.length) {
      const best = others.reduce((a, b) => (a.total_ms <= b.total_ms ? a : b));
      let text = `<b></b> did the whole job in <b>${fmtMs(best.total_ms)}</b>, <b>${fmtX(serial.total_ms / best.total_ms)}</b> faster than one core (${fmtMs(serial.total_ms)}).`;
      if (best.impl.startsWith("cuda")) {
        const parts = best.h2d_ms + best.kernel_ms + best.d2h_ms;
        const share = Math.round((100 * (best.h2d_ms + best.d2h_ms)) / parts);
        text += ` Copying data to and from the GPU took ${share}% of that time; the filter itself ran ${fmtX(serial.total_ms / best.kernel_ms)} faster than one core.`;
      }
      const bad = main.filter((r) => !r.pass).length;
      text += bad ? ` <span class="fail">${bad} version(s) did NOT match the serial result.</span>` : ` All ${main.length} outputs matched the serial result.`;
      h.innerHTML = text;
      $("b", h).textContent = implName(best);
    } else {
      h.textContent = "Waiting for the other versions…";
    }

    // charts
    $("#chart-time").innerHTML = timeChart(main);
    $("#chart-speed").innerHTML = serial ? speedChart(main, serial) : "";

    const omp = job.steps.filter((s) => s.kind === "omp_sweep" && s.status === "done" && s.result).map((s) => s.result);
    $("#chart-omp").classList.toggle("hidden", !(serial && omp.length));
    if (serial && omp.length) $("#chart-omp").innerHTML = ompChart(omp, serial);

    const blk = job.steps.filter((s) => s.kind === "block_sweep" && s.status === "done" && s.result).map((s) => s.result);
    $("#chart-block").classList.toggle("hidden", !blk.length);
    if (blk.length) $("#chart-block").innerHTML = blockChart(blk);

    const ks = job.steps.filter((s) => s.kind === "k_sweep" && s.status === "done" && s.result).map((s) => Object.assign({ k: s.k }, s.result));
    $("#chart-k").classList.toggle("hidden", !ks.length);
    if (ks.length) $("#chart-k").innerHTML = kChart(ks);

    const mpi = job.steps.filter((s) => s.kind === "mpi_sweep" && s.status === "done" && s.result).map((s) => s.result);
    $("#chart-mpi").classList.toggle("hidden", !mpi.length);
    if (mpi.length) $("#chart-mpi").innerHTML = mpiChart(mpi);
  }

  // --------------------------------------------------------------- charts
  const svgOpen = (w, h) => `<svg viewBox="0 0 ${w} ${h}" role="img" xmlns="http://www.w3.org/2000/svg">`;

  function timeChart(main) {
    const gpuOnly = state.timeView === "gpu";
    const rows = main.filter((r) => !gpuOnly || r.impl.startsWith("cuda"));
    const head = `<h3>Time per version</h3><p class="small muted">Shorter is better. Each GPU bar splits into copying in, computing and copying out.</p>
      <div class="tools"><button data-view="all" class="${gpuOnly ? "" : "active"}">All versions</button><button data-view="gpu" class="${gpuOnly ? "active" : ""}">GPU only (zoomed in)</button></div>`;
    if (!rows.length) return head + `<p class="small muted">No GPU results.</p>`;
    const W = 640, labelW = 190, right = 70, rowH = 36, top = 8, axisH = 28;
    const H = top + rows.length * rowH + axisH;
    const maxV = Math.max(...rows.map((r) => Math.max(r.total_ms, r.h2d_ms + r.kernel_ms + r.d2h_ms)));
    const ticks = niceTicks(maxV);
    const scale = (W - labelW - right) / ticks[ticks.length - 1];
    let s = svgOpen(W, H);
    ticks.forEach((t) => {
      const x = labelW + t * scale;
      s += `<line class="grid" x1="${x}" x2="${x}" y1="${top}" y2="${H - axisH}"/><text class="muted" x="${x}" y="${H - 10}" font-size="11" text-anchor="middle">${fmtMs(t)}</text>`;
    });
    rows.forEach((r, i) => {
      const y = top + i * rowH, gpu = r.impl.startsWith("cuda");
      s += `<text x="${labelW - 10}" y="${y + rowH / 2 + 4}" font-size="13" text-anchor="end">${implName(r)}</text>`;
      let x = labelW;
      const seg = (v, color) => {
        const w = Math.max(v * scale, v > 0 ? 1 : 0);
        s += `<rect x="${x}" y="${y + 7}" width="${w}" height="${rowH - 14}" fill="${color}" rx="2"/>`;
        x += w;
      };
      if (gpu) { seg(r.h2d_ms, C_IN); seg(r.kernel_ms, C_KERNEL); seg(r.d2h_ms, C_OUT); } else seg(r.kernel_ms, C_KERNEL);
      s += `<text x="${x + 6}" y="${y + rowH / 2 + 4}" font-size="12" font-weight="600">${fmtMs(r.total_ms)}</text>`;
    });
    s += `<line class="axis" x1="${labelW}" x2="${labelW}" y1="${top}" y2="${H - axisH}"/></svg>`;
    return head + s + `<div class="legend"><span><i style="background:${C_IN}"></i>copy image to GPU</span><span><i style="background:${C_KERNEL}"></i>computing</span><span><i style="background:${C_OUT}"></i>copy result back</span></div>`;
  }

  function speedChart(main, serial) {
    const rows = main.filter((r) => r.impl !== "serial");
    const head = `<h3>How many times faster than one core</h3><p class="small muted">Longer is better. Solid bar: the whole job. Outlined bar: the computing alone, without copying.</p>`;
    if (!rows.length) return head + `<p class="small muted">Waiting for results…</p>`;
    const W = 640, labelW = 190, right = 150, rowH = 52, top = 8, axisH = 28;
    const H = top + rows.length * rowH + axisH;
    const maxV = Math.max(10, ...rows.map((r) => serial.total_ms / Math.min(r.kernel_ms, r.total_ms)));
    const decades = Math.ceil(Math.log10(maxV * 1.05));
    const pos = (v) => labelW + (Math.log10(Math.max(v, 1)) / decades) * (W - labelW - right);
    let s = svgOpen(W, H);
    for (let d = 0; d <= decades; d++) {
      const x = pos(Math.pow(10, d));
      s += `<line class="grid" x1="${x}" x2="${x}" y1="${top}" y2="${H - axisH}"/><text class="muted" x="${x}" y="${H - 10}" font-size="11" text-anchor="middle">${Math.pow(10, d).toLocaleString()}×</text>`;
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH, gpu = r.impl.startsWith("cuda"), color = IMPL[r.impl].color;
      const whole = serial.total_ms / r.total_ms, comp = serial.total_ms / r.kernel_ms;
      s += `<text x="${labelW - 10}" y="${y + rowH / 2 + 4}" font-size="13" text-anchor="end">${implName(r)}</text>`;
      const bh = gpu ? 15 : 22, y1 = gpu ? y + 6 : y + 15;
      s += `<rect x="${labelW}" y="${y1}" width="${Math.max(pos(whole) - labelW, 1)}" height="${bh}" fill="${color}" rx="2"/>`;
      s += `<text x="${pos(whole) + 6}" y="${y1 + bh - 3}" font-size="12" font-weight="600">${fmtX(whole)}</text>`;
      if (gpu) {
        s += `<rect x="${labelW}" y="${y + 25}" width="${Math.max(pos(comp) - labelW, 1)}" height="15" fill="${color}" fill-opacity="0.18" stroke="${color}" rx="2"/>`;
        s += `<text class="muted" x="${pos(comp) + 6}" y="${y + 37}" font-size="12">${fmtX(comp)} computing only</text>`;
      }
    });
    s += `<line class="axis" x1="${labelW}" x2="${labelW}" y1="${top}" y2="${H - axisH}"/></svg>`;
    return head + s + `<p class="small muted">Scale is logarithmic: each step to the right is 10 times faster.</p>`;
  }

  function ompChart(omp, serial) {
    const pts = omp.map((r) => ({ t: r.threads, v: serial.total_ms / r.total_ms })).sort((a, b) => a.t - b.t);
    const W = 640, H = 320, L = 52, R = 24, T = 14, B = 44;
    const xmax = Math.max(16, ...pts.map((p) => p.t)), ymax = Math.max(xmax, ...pts.map((p) => p.v)) * 1.05;
    const X = (t) => L + (t / xmax) * (W - L - R), Y = (v) => H - B - (v / ymax) * (H - T - B);
    let s = svgOpen(W, H);
    niceTicks(ymax).forEach((v) => (s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="muted" x="${L - 8}" y="${Y(v) + 4}" font-size="11" text-anchor="end">${v}×</text>`));
    pts.forEach((p) => (s += `<text class="muted" x="${X(p.t)}" y="${H - 22}" font-size="11" text-anchor="middle">${p.t}</text>`));
    s += `<text class="muted" x="${(L + W - R) / 2}" y="${H - 4}" font-size="12" text-anchor="middle">CPU threads</text>`;
    s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(xmax)}" y2="${Y(xmax)}" stroke="#999" stroke-dasharray="5 4"/>`;
    s += `<text class="muted" x="${X(xmax) - 4}" y="${Y(xmax) + 14}" font-size="11" text-anchor="end">perfect scaling</text>`;
    s += `<polyline fill="none" stroke="${IMPL.omp.color}" stroke-width="2.5" points="${pts.map((p) => `${X(p.t)},${Y(p.v)}`).join(" ")}"/>`;
    pts.forEach((p) => (s += `<circle cx="${X(p.t)}" cy="${Y(p.v)}" r="4.5" fill="${IMPL.omp.color}"/><text x="${X(p.t)}" y="${Y(p.v) - 9}" font-size="11" font-weight="600" text-anchor="middle">${fmtX(p.v)}</text>`));
    s += `</svg>`;
    return `<h3>CPU thread scaling</h3><p class="small muted">If every extra thread helped fully, the line would follow the dashed one. It flattens because the memory cannot feed more threads.</p>` + s;
  }

  function blockChart(blk) {
    const blocks = ["8x8", "16x16", "32x32"], impls = ["cuda-naive", "cuda-tiled"];
    const val = (impl, b) => {
      const r = blk.find((x) => x.impl === impl && `${x.block_x}x${x.block_y}` === b);
      return r ? r.kernel_ms : null;
    };
    const all = blocks.flatMap((b) => impls.map((i) => val(i, b))).filter((v) => v !== null);
    if (!all.length) return "";
    const W = 640, H = 300, L = 56, R = 16, T = 14, B = 46;
    const ymax = Math.max(...all) * 1.18, ticks = niceTicks(ymax);
    const top = ticks[ticks.length - 1];
    const Y = (v) => H - B - (v / top) * (H - T - B);
    const gw = (W - L - R) / blocks.length, bw = Math.min(46, gw / 3);
    let s = svgOpen(W, H);
    ticks.forEach((v) => (s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="muted" x="${L - 8}" y="${Y(v) + 4}" font-size="11" text-anchor="end">${fmtMs(v)}</text>`));
    blocks.forEach((b, gi) => {
      const cx = L + gw * gi + gw / 2;
      s += `<text x="${cx}" y="${H - 24}" font-size="13" text-anchor="middle">${b.replace("x", " × ")}</text>`;
      impls.forEach((impl, ii) => {
        const v = val(impl, b);
        if (v === null) return;
        const x = cx + (ii === 0 ? -bw - 3 : 3);
        s += `<rect x="${x}" y="${Y(v)}" width="${bw}" height="${H - B - Y(v)}" fill="${IMPL[impl].color}" rx="2"/><text x="${x + bw / 2}" y="${Y(v) - 5}" font-size="11" font-weight="600" text-anchor="middle">${fmtMs(v)}</text>`;
      });
    });
    s += `</svg>`;
    return `<h3>GPU thread-block shape (computing time only)</h3><p class="small muted">Shorter is better. The shape of each group of GPU workers changes the speed.</p>` + s +
      `<div class="legend"><span><i style="background:${IMPL["cuda-naive"].color}"></i>CUDA naive</span><span><i style="background:${IMPL["cuda-tiled"].color}"></i>CUDA + shared tiles</span></div>`;
  }

  function kChart(rows) {
    const impls = Object.keys(IMPL).filter((i) => rows.some((r) => r.impl === i));
    const ks = [...new Set(rows.map((r) => r.k))].sort((a, b) => a - b);
    const val = (i, k) => { const r = rows.find((x) => x.impl === i && x.k === k); return r ? r.kernel_ms : null; };
    const all = rows.map((r) => r.kernel_ms).filter((v) => v > 0);
    const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...all)))), hi = Math.max(Math.pow(10, Math.ceil(Math.log10(Math.max(...all)))), lo * 10);
    const W = 640, H = 330, L = 64, R = 24, T = 14, B = 44;
    const X = (k) => L + ((k - ks[0]) / Math.max(ks[ks.length - 1] - ks[0], 1)) * (W - L - R);
    const Y = (v) => H - B - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - T - B);
    let s = svgOpen(W, H);
    for (let v = lo; v <= hi * 1.001; v *= 10) s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="muted" x="${L - 8}" y="${Y(v) + 4}" font-size="11" text-anchor="end">${fmtMs(v)}</text>`;
    ks.forEach((k) => (s += `<text class="muted" x="${X(k)}" y="${H - 22}" font-size="11" text-anchor="middle">${k} × ${k}</text>`));
    s += `<text class="muted" x="${(L + W - R) / 2}" y="${H - 4}" font-size="12" text-anchor="middle">filter size</text>`;
    impls.forEach((i) => {
      const pts = ks.map((k) => [k, val(i, k)]).filter((p) => p[1] > 0);
      s += `<polyline fill="none" stroke="${IMPL[i].color}" stroke-width="2.5" points="${pts.map((p) => `${X(p[0])},${Y(p[1])}`).join(" ")}"/>`;
      pts.forEach((p) => (s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="${IMPL[i].color}"/>`));
    });
    s += `</svg>`;
    const legend = impls.map((i) => `<span><i style="background:${IMPL[i].color}"></i>${IMPL[i].name}</span>`).join("");
    return `<h3>Time against filter size (computing only)</h3><p class="small muted">Lower is better. Bigger filters mean more work per pixel. The scale is logarithmic: each gridline is 10 times the one below.</p>` + s + `<div class="legend">${legend}</div>`;
  }

  function mpiChart(rows) {
    const pts = rows.slice().sort((a, b) => a.ranks - b.ranks);
    const W = 640, H = 300, L = 56, R = 16, T = 22, B = 46;
    const ticks = niceTicks(Math.max(...pts.map((p) => p.total_ms)) * 1.1), top = ticks[ticks.length - 1];
    const Y = (v) => H - B - (v / top) * (H - T - B);
    const gw = (W - L - R) / pts.length, bw = Math.min(54, gw * 0.6);
    let s = svgOpen(W, H);
    ticks.forEach((v) => (s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="muted" x="${L - 8}" y="${Y(v) + 4}" font-size="11" text-anchor="end">${fmtMs(v)}</text>`));
    pts.forEach((p, i) => {
      const cx = L + gw * i + gw / 2, base = pts[0].total_ms;
      s += `<rect x="${cx - bw / 2}" y="${Y(p.total_ms)}" width="${bw}" height="${H - B - Y(p.total_ms)}" fill="#7A5195" rx="2"/>`;
      s += `<text x="${cx}" y="${Y(p.total_ms) - 6}" font-size="11" font-weight="600" text-anchor="middle">${fmtMs(p.total_ms)}${i ? " · " + fmtX(base / p.total_ms) : ""}</text>`;
      s += `<text x="${cx}" y="${H - 24}" font-size="13" text-anchor="middle">${p.ranks}</text>`;
    });
    s += `<text class="muted" x="${(L + W - R) / 2}" y="${H - 6}" font-size="12" text-anchor="middle">MPI processes</text></svg>`;
    return `<h3>MPI: image split across processes</h3><p class="small muted">Whole job, including sending the strips out and collecting them back. The multiplier is the gain over 1 process. All processes share this one computer's CPU, so this shows how the method behaves, not what a real cluster would do.</p>` + s;
  }

  $("#chart-time").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-view]");
    if (!b || !state.job) return;
    state.timeView = b.dataset.view;
    $("#chart-time").innerHTML = timeChart(mainResults(state.job));
  });

  init().catch((e) => setMsg("Could not reach the server: " + e.message, true));
})();
