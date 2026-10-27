#!/usr/bin/env python3
"""Build report/ConvoX_Poster.pdf: an A1 portrait academic poster for the project.

usage: POSTER=1 python report/make_figures.py && python report/build_poster.py
Needs reportlab, qrcode, Pillow and the Times New Roman font (Windows).
"""
import pathlib

import qrcode
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "poster_figures"
OUT = HERE / "ConvoX_Poster.pdf"
REPO = "https://github.com/PCAP-ZSS/ConvoX"

FONTS = pathlib.Path("C:/Windows/Fonts")
for name, f in (("Times", "times.ttf"), ("Times-Bold", "timesbd.ttf"), ("Times-Italic", "timesi.ttf"),
                ("Times-BoldItalic", "timesbi.ttf"), ("Mono", "cour.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold", italic="Times-Italic", boldItalic="Times-BoldItalic")

PAGE_W, PAGE_H = 594 * mm, 841 * mm  # A1 portrait
M, GUT = 46, 34
HEAD_H, FOOT_H = 250, 190
COL_W = (PAGE_W - 2 * M - 2 * GUT) / 3
NAVY, ACCENT, PALE = colors.HexColor("#14325c"), colors.HexColor("#D55E00"), colors.HexColor("#eef3fa")

BODY = ParagraphStyle("body", fontName="Times", fontSize=21, leading=26, alignment=TA_JUSTIFY, spaceAfter=8)
LIST = ParagraphStyle("list", parent=BODY, leftIndent=26, bulletIndent=4, bulletFontName="Times", bulletFontSize=21, spaceAfter=5)
CAP = ParagraphStyle("cap", fontName="Times-Italic", fontSize=16, leading=19, alignment=TA_LEFT, spaceBefore=3, spaceAfter=10,
                     textColor=colors.HexColor("#333333"))
HEAD = ParagraphStyle("head", fontName="Times-Bold", fontSize=28, leading=32, textColor=colors.white)
BIG = ParagraphStyle("big", fontName="Times-Bold", fontSize=38, leading=42, alignment=TA_CENTER, textColor=ACCENT)
TILE = ParagraphStyle("tile", fontName="Times", fontSize=15, leading=18, alignment=TA_CENTER)
EQ = ParagraphStyle("eq", fontName="Times", fontSize=21, leading=30, alignment=TA_CENTER, spaceBefore=2, spaceAfter=8)
CELL = ParagraphStyle("cell", fontName="Times", fontSize=16, leading=19, alignment=TA_CENTER)
CELL_H = ParagraphStyle("cellh", parent=CELL, fontName="Times-Bold")

story = []


def P(t, s=BODY):
    story.append(Paragraph(t, s))


def section(title):
    t = Table([[Paragraph(title, HEAD)]], colWidths=[COL_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 14),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(KeepTogether([Spacer(1, 4), t, Spacer(1, 9)]))


def bullets(rows):
    for r in rows:
        story.append(Paragraph(r, LIST, bulletText="\u2022"))


def fig(name, caption, scale=1.0):
    w, h = ImageReader(str(FIG / f"{name}.png")).getSize()
    im = Image(str(FIG / f"{name}.png"), width=COL_W * scale, height=COL_W * scale * h / w)
    im.hAlign = "CENTER"
    story.append(KeepTogether([im, Paragraph(caption, CAP)]))


def tiles(items):
    cells = [[Paragraph(n, BIG), Paragraph(l, TILE)] for n, l in items]
    rows = [[[c[0], c[1]] for c in cells[i:i + 2]] for i in range(0, len(cells), 2)]
    t = Table(rows, colWidths=[COL_W / 2] * 2)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 1, NAVY), ("INNERGRID", (0, 0), (-1, -1), 1, colors.white),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
    story.append(t)
    story.append(Spacer(1, 10))


# ------------------------------------------------------------------------------------------- column 1
tiles([("728 / 728", "runs match the serial reference (max error 4.8\u00d710\u207b\u2077)"),
       ("8.5\u00d7", "best OpenMP speedup, 16 threads; flat after about 4"),
       ("389\u00d7", "GPU speedup counting only the filter kernel"),
       ("104\u00d7", "the same job including copies to and from the GPU"),
       ("70\u201393%", "of GPU job time spent on data transfer"),
       ("86\u00d7", "less cache traffic with shared-memory tiling (K = 15)")])

section("Why this study")
P("Convolution is the sliding-window filter behind blur, sharpening and edge detection and behind the layers of "
  "convolutional neural networks. Every output pixel is independent, so it parallelises easily, yet it is limited by "
  "memory bandwidth, not arithmetic. Published comparisons usually use one programming model, tiny filters, and "
  "kernel time only, which flatters the GPU.")
P("<b>We ask:</b> how much does each optimisation really buy, why, and what does the GPU cost end to end?")

section("Aim")
bullets(["Run <b>one</b> algorithm on five execution paths under identical input, validation and timing.",
         "Explain each speed change by a specific hardware mechanism, with profiler evidence.",
         "Count host\u2013device transfers, not just kernel time.",
         "Compare measured speedup ceilings with Amdahl\u2019s law [1]."])

section("Algorithm")
P("For a K\u00d7K filter F (K odd, R = (K\u22121)/2) and image I:")
P("O[y][x] = \u03a3<sub>i,j=\u2212R..R</sub> I[y+i][x+j]\u00b7F[i+R][j+R]", EQ)
P("Single-precision float, zero-padded border. The plain four-loop serial code is the oracle: every other output must "
  "match it to within 10\u207b\u2074.")
fig("architecture", "One shared loader, validator and timing harness feed all execution paths and the analysis scripts.", 1.0)

P("<b>CUDA stages:</b> <i>naive</i> reads everything from global memory; <i>constant</i> keeps the filter in constant memory "
  "(warp-wide broadcast); <i>tiled</i> also stages an input tile with halo in shared memory [7]. The serial code is "
  "compiled <font name='Mono'>-O2</font> without vectorisation; OpenMP shares rows between threads.")

story.append(FrameBreak())
# column 2
section("Method")
bullets(["<b>Machine:</b> Intel i7-12650H (6P + 4E cores, 16 threads), RTX 3050 Laptop GPU (4 GB), CUDA 12.6, GCC 13.3, WSL2.",
         "<b>Sweep:</b> 4 images (two 4K, two 8K) \u00d7 K = 3\u201315 \u00d7 26 configurations = 728 runs.",
         "<b>Timing:</b> one warm-up, 10 runs, median; CUDA events for kernel and each copy.",
         "<b>Check:</b> every output compared with serial; Nsight Compute for DRAM, L1 and occupancy."])

section("OpenMP stops scaling near 4 threads")
fig("omp_scaling", "Speedup over serial, 8K image. Ideal is the dashed line.", 0.82)
P("6.7\u00d7\u20138.5\u00d7 at 16 threads (efficiency 0.38\u20130.53). Memory bandwidth and the hybrid P/E-core mix limit it; "
  "small filters saturate sooner. Dynamic and guided schedules beat static by 9\u201312%, and "
  "<font name='Mono'>collapse(2)</font> with chunk 1 is 5\u00d7 slower.")

section("16\u00d716 blocks are fastest")
fig("block_sensitivity", "CUDA kernel time by block shape, 8K image.", 0.9)
P("Small blocks hold too few warps; 1024-thread blocks limit how many fit on a multiprocessor. This echoes the finding "
  "that maximum occupancy is not maximum speed [8].")

section("Tiling pays off only for large filters")
fig("tiling_ratio", "Kernel time of the naive kernel divided by the optimised kernel, 16\u00d716 blocks.", 0.82)
P("The ratio grows from <b>0.89</b> at K = 3 (tiling slightly slower) to <b>1.94</b> at K = 15.")

t = Table([[Paragraph(c, CELL_H) for c in ("K", "Kernel", "DRAM read MB", "L1 loads MB", "Warps active")]] +
          [[Paragraph(c, CELL) for c in r] for r in (("15", "naive", "137.9", "51,192", "88%"), ("15", "tiled", "147.6", "597", "97%"))],
          colWidths=[COL_W * w for w in (0.12, 0.2, 0.26, 0.24, 0.18)])
t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.5, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                       ("LINEBELOW", (0, -1), (-1, -1), 1.5, colors.black), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
story.append(t)
story.append(Spacer(1, 12))
P("<b>Nsight finding:</b> DRAM traffic is the same for all kernels (about one pass over the 133 MB image) because L1 and "
  "L2 already capture the reuse. Tiling removes <b>cache-to-multiprocessor</b> traffic and raises occupancy; it does "
  "not save DRAM bandwidth.")

# column 3 follows on from column 2
section("Transfers dominate the GPU's time")
fig("transfer_split", "Host-to-device copy, kernel and device-to-host copy, 8K image, 16\u00d716 blocks.", 0.88)
P("Copies are 93% of the tiled job at K = 3 and 70% at K = 15. Kernel-only speedup is 165\u00d7 and 389\u00d7; "
  "whole-job speedup is <b>9\u00d7 and 104\u00d7</b>. The GPU still beats 16 threads at every K, but offload pays only "
  "when there is enough work per byte moved.")

section("Amdahl's law against measurement")
fig("amdahl", "Ceiling from the measured serial fraction against the best observed OpenMP speedups.", 0.88)
P("Image decode and PNG encode make the serial fraction f = 0.90 (K = 3) to 0.39 (K = 15), so whole-job speedup is "
  "capped at <b>1.1\u00d7\u20132.6\u00d7</b>; the observed values are within 2\u201316% of that bound. The law misses the "
  "kernel-only gap (6.6\u00d7\u20138.5\u00d7 instead of 16\u00d7): bandwidth contention and hybrid cores.")

section("MPI extension")
fig("mpi_scaling", "Strip decomposition on one node, serial local kernel, 8K image, K = 7.", 0.78)
P("About 5\u00d7 at 16 ranks, then flat. Results match serial for 1, 2, 3, 4 and 7 ranks, including uneven row counts. "
  "Single-node runs show correctness and halo handling, not network scaling.")

section("Conclusions")
bullets(["Memory behaviour, not thread count, sets performance.",
         "Caches already absorb most reuse, so tiling helps through cache traffic and occupancy, and only for large K.",
         "Report end-to-end time: 389\u00d7 became 104\u00d7, and about 8\u00d7 at small K.",
         "Limits: one machine, synthetic grayscale images, single-node MPI."])

story.append(Spacer(1, 6))

# --------------------------------------------------------------------------------------------- page
qr = qrcode.make(REPO)
qr.save(HERE / "poster_figures" / "qr.png")


def decorate(c, doc):
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - HEAD_H, PAGE_W, HEAD_H, stroke=0, fill=1)
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - HEAD_H - 8, PAGE_W, 8, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Times-Bold", 56)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 78, "A Comparative Study of Shared-Memory, GPU, and Distributed")
    c.drawCentredString(PAGE_W / 2, PAGE_H - 142, "Parallelisation Strategies for 2D Image Convolution")
    c.setFont("Times", 30)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 192, "Saatvik Das  \u00b7  Saket Rama  \u00b7  Zaid Khan")
    c.setFont("Times-Italic", 25)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 228, "Department of Computer Science and Engineering, Manipal Institute of Technology, Bengaluru  \u00b7  "
                                                    "Guide: Prof. Anil Kumar  \u00b7  PCAP, Semester V")
    # footer
    c.setFillColor(PALE)
    c.rect(0, 0, PAGE_W, FOOT_H, stroke=0, fill=1)
    c.setFillColor(NAVY)
    c.rect(0, FOOT_H, PAGE_W, 6, stroke=0, fill=1)
    qs = 138
    c.drawImage(ImageReader(str(HERE / "poster_figures" / "qr.png")), PAGE_W - M - qs, (FOOT_H - qs) / 2, qs, qs)
    c.setFillColor(colors.black)
    c.setFont("Times-Bold", 21)
    c.drawString(M, FOOT_H - 38, "References")
    refs = ["[1] G. M. Amdahl, AFIPS Spring Joint Computer Conf., 1967.   [3] S. Williams, A. Waterman, D. Patterson, Roofline, Commun. ACM 52(4), 2009.",
            "[7] P. Micikevicius, 3D finite difference computation on GPUs using CUDA, GPGPU-2, 2009.   [8] V. Volkov, J. Demmel, Benchmarking GPUs to tune dense linear algebra, SC '08.",
            "Source code, results and full report: " + REPO + "  (scan the code)"]
    c.setFont("Times", 19)
    for i, r in enumerate(refs):
        c.drawString(M, FOOT_H - 68 - i * 30, r)
    c.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=(PAGE_W, PAGE_H), title="ConvoX poster", author="Saatvik Das, Saket Rama, Zaid Khan",
                      subject="PCAP project poster")
body_y, body_h = FOOT_H + 18, PAGE_H - HEAD_H - FOOT_H - 18 - 34
frames = [Frame(M + i * (COL_W + GUT), body_y, COL_W, body_h, id=f"c{i}", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
          for i in range(3)]
doc.addPageTemplates([PageTemplate(id="poster", frames=frames, onPage=decorate)])
doc.build(story)
print("wrote", OUT, "pages:", doc.page)
