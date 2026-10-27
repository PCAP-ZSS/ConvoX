#!/usr/bin/env python3
"""Build report/ConvoX_Project_Report.pdf in the two-column IEEE layout of the project synopsis.

usage: python report/make_figures.py && python report/build_pdf.py
Needs reportlab and the Times New Roman / Courier New fonts (Windows).
"""
import pathlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "ConvoX_Project_Report.pdf"

FONTS = pathlib.Path("C:/Windows/Fonts")
for name, f in (("Times", "times.ttf"), ("Times-Bold", "timesbd.ttf"), ("Times-Italic", "timesi.ttf"),
                ("Times-BoldItalic", "timesbi.ttf"), ("Mono", "cour.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold", italic="Times-Italic", boldItalic="Times-BoldItalic")

PAGE_W, PAGE_H = letter
MARGIN, GUTTER = 50, 18
TOP, BOTTOM = 54, 58
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
FULL_W = PAGE_W - 2 * MARGIN

BODY = ParagraphStyle("body", fontName="Times", fontSize=9.4, leading=11.2, alignment=TA_JUSTIFY, firstLineIndent=12,
                      spaceAfter=3.5, hyphenationLang="en_GB", allowWidows=0)
BODY0 = ParagraphStyle("body0", parent=BODY, firstLineIndent=0)
H1 = ParagraphStyle("h1", fontName="Times", fontSize=9.4, leading=12, alignment=TA_CENTER, spaceBefore=8, spaceAfter=4, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="Times-Italic", fontSize=9.4, leading=11.5, alignment=TA_LEFT, spaceBefore=4, spaceAfter=2, keepWithNext=1)
ITEM = ParagraphStyle("item", parent=BODY, bulletFontName="Times", bulletFontSize=9.4, firstLineIndent=0, leftIndent=14, bulletIndent=2, spaceAfter=2.5)
BULLET = ParagraphStyle("bullet", parent=ITEM, leftIndent=11, bulletIndent=1)
CAP = ParagraphStyle("cap", fontName="Times", fontSize=7.8, leading=9.2, alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=7)
CAP_C = ParagraphStyle("capc", parent=CAP, alignment=TA_CENTER)
TBL_NO = ParagraphStyle("tblno", fontName="Times", fontSize=7.8, leading=9.2, alignment=TA_CENTER, spaceBefore=3)
TBL_T = ParagraphStyle("tblt", fontName="Times", fontSize=7.8, leading=9.2, alignment=TA_CENTER, spaceAfter=2.5)
CELL = ParagraphStyle("cell", fontName="Times", fontSize=7.3, leading=8.4, alignment=TA_LEFT)
CELL_H = ParagraphStyle("cellh", parent=CELL, fontName="Times-Bold")
EQ = ParagraphStyle("eq", fontName="Times", fontSize=9.4, leading=13, alignment=TA_CENTER, spaceBefore=2, spaceAfter=4)
REF = ParagraphStyle("ref", fontName="Times", fontSize=7.9, leading=9.3, alignment=TA_JUSTIFY, leftIndent=17, firstLineIndent=-17, spaceAfter=1.6)
TITLE = ParagraphStyle("title", fontName="Times-Bold", fontSize=16.5, leading=19, alignment=TA_CENTER, spaceAfter=5)
SUB = ParagraphStyle("sub", fontName="Times-Italic", fontSize=9.4, leading=11.5, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=7)
AUTH = ParagraphStyle("auth", fontName="Times", fontSize=10.5, leading=13, alignment=TA_CENTER, spaceAfter=1)
AFF = ParagraphStyle("aff", fontName="Times-Italic", fontSize=9.4, leading=11.5, alignment=TA_CENTER)
ABS = ParagraphStyle("abs", parent=BODY, firstLineIndent=0)

story = []
state = {"fig": 0, "tbl": 0}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def P(text, style=BODY):
    story.append(Paragraph(text, style))


def H(title):
    state["sec"] = state.get("sec", -1) + 1
    P(f"<b>{ROMAN[state['sec']]}. {title}</b>", H1)


def S(letter_, title):
    P(f"{letter_}. {title}", H2)


def items(rows, numbered=True, style=ITEM):
    for i, t in enumerate(rows, 1):
        story.append(Paragraph(t, style, bulletText=f"{i})" if numbered else "•"))


def fig(name, caption):
    state["fig"] += 1
    ir = ImageReader(str(FIG / f"{name}.png"))
    w, h = ir.getSize()
    img = Image(str(FIG / f"{name}.png"), width=COL_W, height=COL_W * h / w)
    story.append(KeepTogether([img, Paragraph(f"Fig. {state['fig']}. {caption}", CAP)]))


def table(title, header, rows, widths, caption=None, align_right_from=1):
    state["tbl"] += 1
    data = [[Paragraph(str(c), CELL_H) for c in header]]
    data += [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[COL_W * w / sum(widths) for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.2), ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
    ]))
    block = [Paragraph(f"TABLE {ROMAN[state['tbl'] - 1]}", TBL_NO), Paragraph(title, TBL_T), t]
    block.append(Paragraph(caption, CAP) if caption else Spacer(1, 6))
    story.append(KeepTogether(block))


def eq(text, number):
    P(f"{text}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;({number})", EQ)


# ------------------------------------------------------------------------------------- title block
P("A Comparative Study of Shared-Memory, GPU, and Distributed Parallelisation Strategies for 2D Image Convolution", TITLE)
P("Project Report \u2014 Parallel Computing Architecture and Programming (PCAP)", SUB)
P("Saatvik Das, Saket Rama, Zaid Khan", AUTH)
P("Department of Computer Science and Engineering", AFF)
P("Manipal Institute of Technology, Bengaluru", AFF)
P("<br/>Guide: Prof. Anil Kumar", AFF)
P("Semester V", AFF)
story.append(FrameBreak())

P("<b><i>Abstract\u2014</i></b>Two-dimensional convolution is the computational core of classical image processing and of the "
  "convolutional layers of modern computer vision, and a textbook memory-bound stencil. This project implements one "
  "algorithm, 2D convolution with an odd-sized K\u00d7K filter, as a serial C++ baseline, an OpenMP version, a naive CUDA "
  "kernel, a CUDA kernel with the filter in constant memory, a CUDA kernel that adds shared-memory tiling with halo "
  "regions, and an MPI strip decomposition. Every output is validated against the serial reference (largest error "
  "4.8\u00d710\u207b\u2077 against a 10\u207b\u2074 limit across 728 benchmark runs on 4K and 8K images, "
  "K = 3 to 15). On an Intel i7-12650H and an RTX 3050 Laptop GPU, OpenMP reaches 6.7\u00d7 to 8.5\u00d7 over serial at 16 "
  "threads and stops scaling near 4 threads. The tiled kernel is 0.9\u00d7 to 1.9\u00d7 faster than the naive kernel, the "
  "gain growing with K. Nsight Compute explains why: DRAM traffic is identical across the three kernels because the "
  "caches already capture the reuse, while tiling cuts global-load traffic through L1 by up to 86\u00d7. Host\u2013device "
  "transfers take 70% to 93% of the end-to-end GPU time, so a kernel-only speedup of up to 389\u00d7 becomes 104\u00d7 for the "
  "whole job. Measured serial fractions bound end-to-end OpenMP speedup to 1.1\u00d7\u20132.6\u00d7, and the observed "
  "values sit within 2% to 16% of that bound. The deliverable is a reproducible benchmark suite, analysis scripts, a "
  "local demonstration page and this report.", ABS)
P("<b><i>Index Terms\u2014</i></b>2D convolution, OpenMP, CUDA, shared memory, constant memory, MPI, Amdahl\u2019s law, roofline "
  "model, Nsight Compute, GPU acceleration, memory bandwidth, parallel computing.", ABS)

# ---------------------------------------------------------------------------------- I. Introduction
H("Introduction")
P("For roughly four decades application performance improved automatically as transistor density and clock frequency "
  "rose together. That ended in the mid-2000s, when Dennard scaling broke down and clock frequencies settled near "
  "3\u20135 GHz. Vendors responded by scaling outward: multi-core CPUs, wide SIMD units, many-core GPUs and clusters. "
  "Performance is therefore no longer inherited; it must be engineered, and one algorithm can differ by two or three "
  "orders of magnitude in runtime depending only on how it is mapped to hardware.")
P("Image convolution is an unusually good subject for studying this. It is embarrassingly parallel at the output level, "
  "because every output pixel is independent, so it exposes the programming models without algorithmic distractions. "
  "It is not trivially fast, however: its sliding-window access pattern overlaps heavily between neighbouring outputs. "
  "That overlap is both the problem (redundant memory traffic) and the solution (reuse through blocking and tiling). "
  "The scarce resource on contemporary hardware is memory bandwidth, not arithmetic throughput.")
P("This report presents ConvoX, a benchmark suite that runs one convolution algorithm on five execution paths under "
  "identical input, validation and timing conditions, and attributes every performance change to a specific hardware "
  "mechanism. It reports the full measurement campaign (728 validated runs), a profiler-based explanation of the GPU "
  "results, and an honest accounting of data-transfer cost.")

# ------------------------------------------------------------------------------ II. Problem Statement
H("Problem Statement")
P("High-performance convolution implementations are usually delivered as tuned black boxes (cuDNN, OpenCV, oneDNN, "
  "FFT- and Winograd-based methods): the user calls a function and gets a number, with no visibility into which "
  "optimisation produced the speedup. Published comparisons are typically confined to one programming model, use small "
  "images and 3\u00d73 filters that fit in cache, and report only kernel time, omitting host\u2013device transfer and so "
  "overstating the realistic GPU advantage. Amdahl\u2019s law predictions are rarely checked against a measured serial "
  "fraction on the same workload. No compact, reproducible study implements one identical algorithm across serial, "
  "shared-memory, naive-GPU, optimised-GPU and distributed-memory models, sweeps both filter size and launch "
  "configuration, and attributes the differences to levels of the memory hierarchy. ConvoX provides that study.")

# ----------------------------------------------------------------------------- III. Literature Survey
H("Literature Survey")
P("Amdahl [1] bounded the speedup of a program by its non-parallelisable fraction, S(N) = 1/(f + (1\u2212f)/N), the "
  "standard first-order scaling model. It assumes a fixed problem size and ignores parallel overheads and bandwidth "
  "saturation; Gustafson [2] argues the problem size should grow with the machine instead. Williams et al. [3] gave the "
  "roofline model, which bounds attainable performance by the lesser of peak arithmetic rate and memory bandwidth "
  "times arithmetic intensity.")
P("Datta et al. [6] auto-tuned stencil kernels on multicore CPUs and early GPUs and showed they are bandwidth-bound, so "
  "cache and register blocking rather than extra cores drive most of the gain; their 3D finite-difference targets "
  "predate unified L2 caches and they did not isolate constant memory for coefficients. Micikevicius [7] introduced "
  "the shared-memory tiling pattern with explicit halo loading and <font name='Mono'>__syncthreads()</font> phases "
  "for 3D stencils, but compared no CPU or OpenMP baseline. Volkov and Demmel [8] showed that maximum occupancy is not "
  "maximum performance, which motivates the block-dimension sweep here.")
P("cuDNN [9] reformulates convolution as im2col plus GEMM, which is fast but expands the intermediate matrix and is "
  "tuned for small filters and batched tensors, not large single-image kernels up to 15\u00d715. Lavin and Gray [10] "
  "apply Winograd minimal filtering to cut multiplications for small filters; the benefit fades as K grows and hides "
  "memory-hierarchy behaviour behind arithmetic restructuring. None of these works compares serial, OpenMP and several "
  "CUDA stages on one algorithm with transfer cost and profiler evidence, which is the gap ConvoX fills.")

# ------------------------------------------------------------------------- IV. Aim and Objectives
H("Aim and Objectives")
S("A", "Aim")
P("To design, implement and empirically evaluate progressively parallelised implementations of 2D image convolution, "
  "namely serial, OpenMP, naive CUDA and memory-hierarchy-optimised CUDA (with an optional MPI layer), in order to "
  "quantify the contribution of each parallelisation and memory-optimisation technique and to check the observed "
  "scaling against established theoretical models.")
S("B", "Objectives and outcome")
table("Objectives and where each was met",
      ["#", "Objective", "Result"],
      [["1", "Serial reference, K = 3\u201315, zero-pad / clamp border", "Done; oracle for every check (Sec. V)"],
       ["2", "OpenMP rows, collapse(2), schedules, 1\u201316 threads", "Done (Sec. VII-A, B)"],
       ["3", "Naive CUDA and block-dimension study", "Done (Sec. VII-C)"],
       ["4", "Constant memory, shared tiling, profiler evidence", "Done; DRAM traffic measured (Sec. VII-D)"],
       ["5", "4K and 8K sweep, kernel-only vs end-to-end, Amdahl", "Done; 728 runs (Sec. VII-F, G)"],
       ["6", "Optional MPI strip decomposition", "Done; verified for 1, 2, 3, 4, 7 ranks (Sec. VII-H)"]],
      [0.3, 3.0, 2.6])

# --------------------------------------------------------------- V. Methodology and Architecture
H("System Architecture and Methodology")
S("A", "Algorithm")
P("For an input image I of H\u00d7W pixels and a filter F of size K\u00d7K (K odd, radius R = (K\u22121)/2), the output is")
eq("O[y][x] = \u03a3<sub>i=\u2212R</sub><super>R</super> \u03a3<sub>j=\u2212R</sub><super>R</super> I[y+i][x+j]\u00b7F[i+R][j+R]", 1)
P("Pixels outside the image are zero (clamp-to-edge is a compile-time alternative). All arithmetic is single-precision "
  "<font name='Mono'>float</font>. The filter is a normalised Gaussian by default; Sobel and box filters are provided "
  "for visual verification. Image libraries (stb_image) decode and encode files only and never perform the "
  "convolution.", BODY0)
fig("architecture", "ConvoX pipeline. All execution paths share one loader, filter generator, validator and timing harness.")
S("B", "Implementation stages")
P("<b>Serial.</b> Four nested loops (output row, output column, filter row, filter column), compiled at "
  "<font name='Mono'>-O2</font> with auto-vectorisation disabled, so the compiler does not parallelise the reference. "
  "It defines T<sub>serial</sub>, the denominator of every speedup, and the reference output.")
P("<b>OpenMP.</b> <font name='Mono'>parallel for</font> over output rows with serial filter loops inside; variants use "
  "<font name='Mono'>collapse(2)</font>, schedule static, dynamic with chunk 1, 8, 32, and guided, and thread counts "
  "1, 2, 4, 6, 8, 12 and 16 (up to the machine\u2019s 16 hardware threads).")
P("<b>CUDA naive.</b> One thread per output pixel; the grid is ceil(W/B<sub>X</sub>)\u00d7ceil(H/B<sub>Y</sub>) with bounds "
  "guards. Image and filter are both read from global memory.")
P("<b>CUDA constant.</b> The filter moves to <font name='Mono'>__constant__</font> memory so that all threads of a warp "
  "read the same coefficient through the broadcast path.")
P("<b>CUDA tiled.</b> Each block loads a (T+K\u22121)\u00b2 input tile with halo into shared memory, calls "
  "<font name='Mono'>__syncthreads()</font>, then computes from shared memory. The three kernels are kept separate so "
  "each optimisation is attributable on its own.")
P("<b>MPI.</b> Rank 0 scatters horizontal strips of H/N + 2R rows (the 2R rows are the halo) with "
  "<font name='Mono'>MPI_Scatterv</font>; each rank runs a local OpenMP or CUDA kernel; interior rows return through "
  "<font name='Mono'>MPI_Gatherv</font>. It is intentionally simple, not communication-optimal.")
S("C", "Experimental design")
P("Table II lists every parameter that was varied; the sweep is the full cross product of the configurations below.")
table("Parameters swept",
      ["Parameter", "Values"],
      [["Images", "4 synthetic: 2 at 3840\u00d72160, 2 at 7680\u00d74320"],
       ["Filter size K", "3, 5, 7, 9, 11, 13, 15 (Gaussian)"],
       ["OpenMP threads", "1, 2, 4, 6, 8, 12, 16"],
       ["OpenMP schedule", "static, dynamic(1/8/32), guided; rows or collapse(2)"],
       ["CUDA block", "8\u00d78, 16\u00d716, 32\u00d732"],
       ["CUDA variant", "naive, +constant, +constant +shared tiles"],
       ["Total", "26 configurations \u00d7 4 images \u00d7 7 K = 728 runs"]],
      [1.35, 3.1])
P("Convolution cost does not depend on pixel values, so procedural images (smooth regions plus sharp edges, generated "
  "by a seeded script) give exact resolutions with no licensing questions and make blur and edge outputs easy to judge "
  "by eye.", BODY0)
S("D", "Timing and validation")
P("CPU timing uses <font name='Mono'>std::chrono::steady_clock</font>. GPU kernels are timed with CUDA events; "
  "host-to-device copy, kernel and device-to-host copy are recorded separately, with a device synchronise before "
  "any host timer stops. Every configuration runs one discarded warm-up and ten timed iterations and reports the "
  "median, which resists scheduling noise and clock-frequency transients. Each output is compared element-wise with the "
  "serial output; because summation order differs, exact equality is not expected, and the criterion is a maximum "
  "absolute error below 10\u207b\u2074 with the RMS error recorded alongside.")
S("E", "Analytical framework")
P("The naive kernel performs 2K\u00b2 floating-point operations per output pixel while loading 4K\u00b2 bytes, an arithmetic "
  "intensity of about 0.5 FLOP/byte for every K. The tiled kernel loads (T+K\u22121)\u00b2/T\u00b2 elements per output, "
  "so for T = 16 and K = 5 the load count falls from 25 to about 1.56, a reuse factor near 16. The serial fraction f "
  "is measured, not assumed: the parts that never parallelise (image decode, filter build, output allocation and PNG "
  "encode) are timed separately and compared with the serial convolution time c, f = s/(s + c).")
S("F", "Environment")
table("Measurement platform",
      ["Item", "Value"],
      [["CPU", "Intel Core i7-12650H, 6 P + 4 E cores, 16 threads"],
       ["GPU", "NVIDIA RTX 3050 Laptop, 4 GB, cc 8.6; peak about 188 GB/s, 6.1 TFLOP/s"],
       ["Software", "WSL2 Ubuntu 24.04, GCC 13.3 (-O2), CUDA 12.6, OpenMPI 4.1.6, CMake 3.28, Python 3.12"],
       ["Profiling", "Nsight Compute (DRAM, L1, L2, occupancy)"]],
      [0.8, 3.65])

# -------------------------------------------------------------------- VI. Implementation and Testing
H("Implementation and Testing")
P("The code base is organised so that each part can be read and presented on its own: <font name='Mono'>src/common</font> "
  "(image, filter, validator, timing, CSV, runner, sweep driver and command line), <font name='Mono'>src/serial</font>, "
  "<font name='Mono'>src/openmp</font>, <font name='Mono'>src/cuda</font>, <font name='Mono'>src/mpi</font>, "
  "<font name='Mono'>analysis</font> (plots, Amdahl, Nsight driver), <font name='Mono'>tests</font>, "
  "<font name='Mono'>scripts</font> and <font name='Mono'>ui</font>. It comprises about 1,700 lines of C++/CUDA, 500 "
  "lines of tests, 540 lines of analysis code and 1,300 lines for the demonstration page. A single "
  "<font name='Mono'>convox</font> program offers <font name='Mono'>info</font>, <font name='Mono'>run</font>, "
  "<font name='Mono'>sweep</font> (resumable) and <font name='Mono'>profile</font> commands.")
P("Eight test programs cover image I/O, filter generation, the validator, the serial reference, the timing harness, "
  "the sweep driver, all 432 OpenMP variants and all 360 CUDA variants, including odd image sizes and invalid launch "
  "configurations. The MPI program was checked against serial for 1, 2, 3, 4 and 7 ranks with all three local kernels "
  "at K = 3 and 15, including row counts that do not divide evenly. In the final sweep all 728 runs passed validation "
  "with a largest absolute error of 4.8\u00d710\u207b\u2077.")
P("A local demonstration page runs the same <font name='Mono'>convox</font> binary on any uploaded image, shows the "
  "original and filtered pictures, a results table and charts, and offers optional thread, block-shape, filter-size "
  "and MPI experiments. It never fabricates numbers: each value is the median reported by the benchmark program, "
  "validated against serial.")

# ------------------------------------------------------------------ VII. Results and Discussion
H("Results and Discussion")
P("Unless stated, figures are for the 8K image (7680\u00d74320), a Gaussian filter and the median of 10 runs. Speedup "
  "is serial time divided by the version\u2019s time on the same image and K.", BODY0)

S("A", "OpenMP scaling")
fig("omp_scaling", "OpenMP speedup over serial against thread count (static schedule, row parallelism).")
P("Scaling is close to linear to 4 threads (about 3.8\u00d7\u20134.1\u00d7), then flattens: 6.7\u00d7 at K=3, 6.0\u00d7 at K=7 and "
  "8.5\u00d7 at K=15 with 16 threads. Efficiency at 16 threads is 0.38 to 0.53, lower for small filters, which do little "
  "arithmetic per byte and so contend for memory bandwidth sooner. The CPU is hybrid, so threads beyond 6 land on "
  "efficiency cores or on hyper-threads of busy cores; this machine cannot separate bandwidth saturation from "
  "that effect.")
S("B", "Schedules and collapse")
table("OpenMP schedules, 8 threads, K = 7 (ms)",
      ["Schedule", "rows", "collapse(2)"],
      [["static", "271", "288"], ["dynamic, chunk 1", "238", "1209"], ["dynamic, chunk 8", "244", "324"],
       ["dynamic, chunk 32", "247", "266"], ["guided", "246", "255"]],
      [2.0, 1.2, 1.4])
P("The synopsis expected static to win on uniform rows. Here dynamic and guided are 9%\u201312% faster, plausibly because a "
  "hybrid CPU has unequal cores so equal static shares finish unevenly; this was not isolated. The clear result is "
  "the cost of fine-grained distribution: collapse(2) with chunk 1 hands out single pixels and is five times slower.", BODY0)

S("C", "CUDA block dimensions")
fig("block_sensitivity", "CUDA kernel time against block shape for K = 3, 9 and 15.")
P("16\u00d716 is fastest for every kernel and K shown. An 8\u00d78 block holds only two warps and, for the tiled kernel, has a "
  "large halo relative to its tile; 32\u00d732 uses the 1024-thread maximum and a larger shared tile, which limits "
  "resident blocks per multiprocessor. Nsight reports achieved warp occupancy of 71%\u201388% for the naive kernel and "
  "94%\u201397% for the tiled kernel at 16\u00d716.")

S("D", "Constant memory and tiling")
fig("tiling_ratio", "Kernel-time ratio of naive to optimised kernels at 16\u00d716 blocks.")
P("The naive/tiled ratio is 0.89 at K=3 (tiling is slightly slower), 1.39 at K=7 and 1.94 at K=15: the redundancy "
  "removed grows as K\u00b2 while the halo overhead grows with the tile perimeter, so the gain appears only for larger "
  "filters. Constant memory alone helps by 3%\u201312% in the sampled cases.")
table("Nsight Compute, 16\u00d716, 8K image",
      ["K", "Kernel", "DRAM rd MB", "DRAM wr MB", "L1 loads MB", "Warps %", "ms"],
      [["3", "naive", "132.8", "132.1", "1889", "70.9", "3.2"], ["3", "tiled", "132.9", "132.1", "322", "93.6", "3.6"],
       ["7", "naive", "133.1", "132.2", "10905", "81.9", "11.3"], ["7", "tiled", "133.6", "132.1", "408", "96.3", "8.2"],
       ["15", "naive", "137.9", "132.5", "51192", "88.3", "47.7"], ["15", "tiled", "147.6", "132.3", "597", "97.4", "25.0"]],
      [0.4, 0.8, 1.0, 1.0, 1.0, 0.8, 0.6])
P("This corrects an expectation in the synopsis. The 8K image is 133 MB and every kernel reads about that much and "
  "writes the same amount: each pixel crosses DRAM once, because the L1 and L2 caches already serve the K\u00b2 reuse of "
  "the naive kernel. Tiling does not reduce DRAM traffic. It removes traffic between cache and multiprocessor, "
  "51 GB down to 0.6 GB at K=15 (86\u00d7), and the speedup comes from fewer load instructions and higher occupancy. "
  "Profiler kernel times exceed harness times because the profiler serialises launches and fixes clocks.", BODY0)

S("E", "Roofline")
P("Analytical intensity is about 0.5 FLOP/byte for the naive kernel and 2K\u00b2/(4(T+K\u22121)\u00b2/T\u00b2) for the tiled kernel, "
  "3.6 to 32 FLOP/byte at T = 16. Naive points plot above the memory roof because the real DRAM intensity is much "
  "higher than the analytical one, for the cache reason above. Tiled points move right and, at K=15, reach about "
  "930 GFLOP/s, 15% of the 6.1 TFLOP/s peak. They stay well below both roofs, so load and shared-memory "
  "instruction throughput are the likely limiters; this was not measured.")

S("F", "Compute time, transfers and end-to-end speedup")
fig("runtime_vs_k", "Compute-only time against filter size for every version (8K image, best configuration).")
fig("transfer_split", "Where GPU time goes: host-to-device copy, kernel and device-to-host copy (16\u00d716 blocks).")
P("Host\u2013device copies are 93% of the end-to-end time of the tiled kernel at K=3 and 70% at K=15. Reporting kernel time "
  "alone therefore overstates the practical gain.")
table("GPU speedup over serial (x)",
      ["K", "Naive kernel", "Naive end-to-end", "Tiled kernel", "Tiled end-to-end"],
      [["3", "184", "7.8", "165", "9.1"], ["7", "199", "35.0", "277", "36.9"], ["15", "201", "86.3", "389", "103.7"]],
      [0.4, 1.0, 1.2, 1.0, 1.2])
P("Even end-to-end the GPU beats 16 OpenMP threads at every K (9.1\u00d7 against 6.7\u00d7 at K=3, 104\u00d7 against 8.5\u00d7 at "
  "K=15), but the margin is small for small filters. Offload pays when arithmetic per transferred byte is high: large "
  "filters, batches, or pipelines that keep data on the device. Transfers use pageable memory (plain "
  "<font name='Mono'>cudaMemcpy</font>); pinned memory would shorten them.", BODY0)

S("G", "Amdahl\u2019s law")
fig("amdahl", "Amdahl ceiling predicted from the measured serial fraction against the best observed OpenMP speedups.")
P("The serial fraction is dominated by image decode (about 0.8\u20131.4 s for 8K) and PNG encode (2.5\u20133.2 s), giving "
  "f = 0.90 at K=3 and 0.39 at K=15, ceilings of 1.11\u00d7 to 2.58\u00d7, and observed end-to-end speedups 2%\u201316% below the "
  "ceiling. The kernel-only speedup, which the law with f = 0 would put at 16\u00d7 for 16 threads, reaches only 6.6\u00d7\u20138.5\u00d7. "
  "The law does not model that gap (bandwidth contention, hybrid cores, synchronisation) nor PCIe transfer for the GPU.")

S("H", "MPI extension")
fig("mpi_scaling", "MPI strip decomposition, serial local kernel, 8K image, K=7, one node (total time with scatter and gather).")
P("Pure MPI scales almost linearly to 4 ranks, then flattens near 5\u00d7: 1550 ms of compute at 1 rank falls to 306 ms at "
  "16. Scatter and gather add 25\u201335 ms. Splitting OpenMP threads across ranks gives no gain, since the threads "
  "already saturated memory, and with CUDA all ranks share one GPU and serialise on it. These runs demonstrate "
  "decomposition and halo handling, not network scaling: no cluster was available.")

S("I", "Visual check")
fig("visual_check", "Same crop of the original, the Gaussian-blurred and the Sobel edge outputs.")
P("The outputs look as expected, which complements the numerical validation.", BODY0)

# --------------------------------------------------------------------------------- VIII. Findings
H("Key Findings")
items([
    "Memory behaviour, not parallelism alone, sets performance: the CPU stops scaling after 4 threads and the tiled kernel reaches 15% of arithmetic peak.",
    "The caches matter more than the synopsis anticipated: naive CUDA already needs about one DRAM pass, so tiling pays through cache-to-multiprocessor traffic and occupancy, and only once K is large.",
    "Transfers change the conclusion: a kernel-only 389\u00d7 becomes 104\u00d7 end-to-end, and about 8\u00d7 at small K.",
    "Amdahl\u2019s law is a useful first-order bound, here dominated by image I/O; its failure to predict the kernel-only plateau is the instructive part.",
])

# ---------------------------------------------------------------------------------- IX. Limitations
H("Limitations")
items([
    "One machine: a hybrid CPU, a 4 GB laptop GPU that can throttle, and WSL2 on Windows. Results do not transfer directly to other hardware.",
    "Threads above 6 mix efficiency cores and hyper-threads, so the cause of the OpenMP plateau is not isolated.",
    "The schedule and collapse sweep ran at 8 threads only; static being slower than dynamic is a measurement on this CPU, not a rule.",
    "Images are synthetic and grayscale; reported runs use zero-padding (clamp-to-edge builds and tests exist).",
    "A GPU driver reset interrupted the sweep once and one run hung once; the sweep was resumed from the recorded runs.",
    "MPI was measured on a single node, so it carries no network cost.",
    "Profiler data cover one 8K image at K = 3, 7, 15 with 16\u00d716 blocks.",
], numbered=False, style=BULLET)

# ------------------------------------------------------------------------------------ X. Conclusion
H("Conclusion and Future Work")
P("ConvoX delivers a validated, reproducible comparison of one convolution algorithm across serial, OpenMP, three "
  "CUDA stages and MPI. The measurements confirm that convolution is memory-bound on the CPU, show that tiling\u2019s "
  "advantage on a modern GPU comes from cache-to-multiprocessor traffic and occupancy rather than DRAM savings, and "
  "show that transfer cost reduces a 389\u00d7 kernel speedup to about 104\u00d7 for a single image. Future work includes "
  "pinned memory and CUDA streams to overlap transfers, batched and colour images, a vendor-library baseline, "
  "separating the hybrid-core effect with core pinning, and measurements on a multi-node cluster.")
P("<b>Reproduction.</b> <font name='Mono'>bash scripts/build.sh</font>, <font name='Mono'>bash scripts/run_tests.sh</font>, "
  "<font name='Mono'>convox sweep</font>, then the summary, plot and Amdahl scripts. The source is at "
  "github.com/PCAP-ZSS/ConvoX.", BODY0)

# ---------------------------------------------------------------------------------------- References
P("<b>References</b>", H1)
refs = [
    "G. M. Amdahl, \u201cValidity of the single processor approach to achieving large scale computing capabilities,\u201d in <i>Proc. AFIPS Spring Joint Computer Conf.</i>, Atlantic City, NJ, USA, 1967, pp. 483\u2013485.",
    "J. L. Gustafson, \u201cReevaluating Amdahl\u2019s law,\u201d <i>Commun. ACM</i>, vol. 31, no. 5, pp. 532\u2013533, May 1988.",
    "S. Williams, A. Waterman, and D. Patterson, \u201cRoofline: An insightful visual performance model for multicore architectures,\u201d <i>Commun. ACM</i>, vol. 52, no. 4, pp. 65\u201376, Apr. 2009.",
    "D. B. Kirk and W.-m. W. Hwu, <i>Programming Massively Parallel Processors: A Hands-on Approach</i>, 4th ed. Cambridge, MA, USA: Morgan Kaufmann, 2022.",
    "L. Dagum and R. Menon, \u201cOpenMP: An industry standard API for shared-memory programming,\u201d <i>IEEE Comput. Sci. Eng.</i>, vol. 5, no. 1, pp. 46\u201355, Jan.\u2013Mar. 1998.",
    "K. Datta <i>et al.</i>, \u201cStencil computation optimization and auto-tuning on state-of-the-art multicore architectures,\u201d in <i>Proc. ACM/IEEE Conf. Supercomputing (SC \u201908)</i>, Austin, TX, USA, 2008, pp. 1\u201312.",
    "P. Micikevicius, \u201c3D finite difference computation on GPUs using CUDA,\u201d in <i>Proc. 2nd Workshop on General Purpose Processing on Graphics Processing Units (GPGPU-2)</i>, Washington, DC, USA, 2009, pp. 79\u201384.",
    "V. Volkov and J. W. Demmel, \u201cBenchmarking GPUs to tune dense linear algebra,\u201d in <i>Proc. ACM/IEEE Conf. Supercomputing (SC \u201908)</i>, Austin, TX, USA, 2008, pp. 1\u201311.",
    "S. Chetlur <i>et al.</i>, \u201ccuDNN: Efficient primitives for deep learning,\u201d <i>arXiv preprint arXiv:1410.0759</i>, 2014.",
    "A. Lavin and S. Gray, \u201cFast algorithms for convolutional neural networks,\u201d in <i>Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)</i>, Las Vegas, NV, USA, 2016, pp. 4013\u20134021.",
    "NVIDIA Corporation, <i>CUDA C++ Programming Guide</i>, Version 12.x. Santa Clara, CA, USA: NVIDIA, 2024.",
    "NVIDIA Corporation, <i>CUDA C++ Best Practices Guide</i>, Version 12.x. Santa Clara, CA, USA: NVIDIA, 2024.",
    "OpenMP Architecture Review Board, <i>OpenMP Application Programming Interface</i>, Version 5.2, Nov. 2021.",
    "Message Passing Interface Forum, <i>MPI: A Message-Passing Interface Standard</i>, Version 4.0, Jun. 2021.",
    "G. Bradski, \u201cThe OpenCV Library,\u201d <i>Dr. Dobb\u2019s Journal of Software Tools</i>, vol. 25, no. 11, pp. 120\u2013125, 2000.",
    "R. C. Gonzalez and R. E. Woods, <i>Digital Image Processing</i>, 4th ed. New York, NY, USA: Pearson, 2018.",
]
for i, r in enumerate(refs, 1):
    P(f"[{i}]&nbsp;&nbsp;{r}", REF)


# ------------------------------------------------------------------------------------------ document
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times", 8.5)
    canvas.drawCentredString(PAGE_W / 2, 30, str(doc.page))
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=TOP, bottomMargin=BOTTOM,
                      title="A Comparative Study of Shared-Memory, GPU, and Distributed Parallelisation Strategies for 2D Image Convolution",
                      author="Saatvik Das, Saket Rama, Zaid Khan", subject="PCAP project report")
HEAD_H = 150
y0 = BOTTOM
col_h_first = PAGE_H - TOP - BOTTOM - HEAD_H
first = PageTemplate(id="first", frames=[
    Frame(MARGIN, PAGE_H - TOP - HEAD_H, FULL_W, HEAD_H, id="head", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
    Frame(MARGIN, y0, COL_W, col_h_first, id="c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
    Frame(MARGIN + COL_W + GUTTER, y0, COL_W, col_h_first, id="c2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
], onPage=footer, autoNextPageTemplate="later")
later = PageTemplate(id="later", frames=[
    Frame(MARGIN, y0, COL_W, PAGE_H - TOP - BOTTOM, id="l1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
    Frame(MARGIN + COL_W + GUTTER, y0, COL_W, PAGE_H - TOP - BOTTOM, id="l2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
], onPage=footer)
doc.addPageTemplates([first, later])
doc.build(story)
print("wrote", OUT, "pages:", doc.page)
