#!/usr/bin/env python3
"""Build report/ConvoX_Poster.pdf: a sparse A1 portrait poster (big numbers, three charts, one picture).

usage: POSTER=1 python report/make_figures.py && python report/build_poster.py
Needs reportlab, qrcode, Pillow and the Times New Roman font (Windows).
"""
import pathlib

import qrcode
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageTemplate, Paragraph, Spacer, Table, TableStyle)

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "poster_figures"
OUT = HERE / "ConvoX_Poster.pdf"
REPO = "https://github.com/PCAP-ZSS/ConvoX"

FONTS = pathlib.Path("C:/Windows/Fonts")
for name, f in (("Times", "times.ttf"), ("Times-Bold", "timesbd.ttf"), ("Times-Italic", "timesi.ttf"),
                ("Times-BoldItalic", "timesbi.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold", italic="Times-Italic", boldItalic="Times-BoldItalic")

PAGE_W, PAGE_H = 594 * mm, 841 * mm  # A1 portrait
M = 56
HEAD_H, FOOT_H = 300, 190
FULL = PAGE_W - 2 * M
NAVY, ACCENT, PALE = colors.HexColor("#14325c"), colors.HexColor("#D55E00"), colors.HexColor("#eef3fa")
GREY = colors.HexColor("#444444")
CHIP = ["#555555", "#0072B2", "#E69F00", "#009E73", "#D55E00"]

LEAD = ParagraphStyle("lead", fontName="Times", fontSize=40, leading=50, textColor=NAVY, spaceAfter=24)
CHIP_T = ParagraphStyle("chip", fontName="Times-Bold", fontSize=30, leading=34, textColor=colors.white)
CHIP_S = ParagraphStyle("chips", fontName="Times", fontSize=24, leading=28, textColor=colors.white)
BIG = ParagraphStyle("big", fontName="Times-Bold", fontSize=92, leading=96, alignment=TA_CENTER, textColor=ACCENT)
TILE = ParagraphStyle("tile", fontName="Times", fontSize=28, leading=33, alignment=TA_CENTER)
NOTE = ParagraphStyle("note", fontName="Times-Italic", fontSize=26, leading=32, alignment=TA_CENTER, textColor=GREY)
PH = ParagraphStyle("ph", fontName="Times-Bold", fontSize=36, leading=42, textColor=NAVY)
PT = ParagraphStyle("pt", fontName="Times", fontSize=28, leading=35, textColor=colors.black, spaceBefore=12)
BAND = ParagraphStyle("band", fontName="Times-Bold", fontSize=42, leading=48, textColor=colors.white)
TAKE = ParagraphStyle("take", fontName="Times", fontSize=34, leading=44, leftIndent=44, bulletIndent=6, bulletFontName="Times-Bold",
                      bulletFontSize=34, spaceAfter=22)

story = []


def band(title):
    t = Table([[Paragraph(title, BAND)]], colWidths=[FULL])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 22),
                           ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 16)]))
    story.append(t)


# ------------------------------------------------------------------------------- row 1: picture + question
vc = ImageReader(str(FIG / "visual_check.png"))
vw, vh = vc.getSize()
PIC_W = 880
pic = Image(str(FIG / "visual_check.png"), width=PIC_W, height=PIC_W * vh / vw)

chips = []
for c, (t, s) in zip(CHIP, [("Serial", "one CPU core, the baseline"), ("OpenMP", "16 CPU threads"),
                            ("CUDA naive", "GPU, plain memory"), ("CUDA constant", "filter in fast constant memory"),
                            ("CUDA tiled", "image tiles in shared memory")]):
    chips.append((c, t, s))
right_w = FULL - PIC_W - 50
chip_rows = [[[Paragraph(t, CHIP_T), Paragraph(s, CHIP_S)]] for _, t, s in chips]
ct = Table(chip_rows, colWidths=[right_w])
ct.setStyle(TableStyle([("BACKGROUND", (0, i), (0, i), colors.HexColor(c)) for i, (c, _, _) in enumerate(chips)] +
                       [("LEFTPADDING", (0, 0), (-1, -1), 20), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("LINEBELOW", (0, 0), (-1, -2), 5, colors.white)]))
right = [Paragraph("One image filter.<br/>Five ways to run it.", LEAD), ct]
row1 = Table([[pic, right]], colWidths=[PIC_W + 50, right_w])
row1.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
story.append(row1)
story.append(Paragraph("Original, Gaussian blur, edge detection and box blur. Every version produces the same picture.", NOTE))
story.append(Spacer(1, 40))

# ------------------------------------------------------------------------------------------ row 2: numbers
stats = [("8.5\u00d7", "faster with 16 CPU threads"), ("389\u00d7", "faster on the GPU, filter only"),
         ("104\u00d7", "faster on the GPU, whole job"), ("70\u201393%", "of GPU time is copying data")]
tw = FULL / 4
tt = Table([[[Paragraph(n, BIG), Paragraph(l, TILE)] for n, l in stats]], colWidths=[tw] * 4)
tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("INNERGRID", (0, 0), (-1, -1), 4, colors.white),
                        ("TOPPADDING", (0, 0), (-1, -1), 26), ("BOTTOMPADDING", (0, 0), (-1, -1), 30), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 18), ("RIGHTPADDING", (0, 0), (-1, -1), 18)]))
story.append(tt)
story.append(Spacer(1, 14))
story.append(Paragraph("728 runs on 4K and 8K images, filters 3\u00d73 to 15\u00d715. Every output verified against the serial result.", NOTE))
story.append(Spacer(1, 44))

# ---------------------------------------------------------------------------------------- row 3: three charts
GAP = 44
PW = (FULL - 2 * GAP) / 3
panels = [("omp_scaling", "CPU threads stop helping", "Gains flatten after about 4 threads. Memory, not cores, is the limit."),
          ("tiling_ratio", "Tiling helps big filters", "Shared-memory tiles run 1.9\u00d7 faster at 15\u00d715, but not at 3\u00d73."),
          ("transfer_split", "Copying eats the GPU\u2019s lead", "Moving the image to and from the GPU takes most of the time.")]
cells = []
for name, head, text in panels:
    w, h = ImageReader(str(FIG / f"{name}.png")).getSize()
    img = Image(str(FIG / f"{name}.png"), width=PW, height=PW * h / w)
    inner = Table([[Paragraph(head, PH)], [img], [Paragraph(text, PT)]], colWidths=[PW], rowHeights=[56, 370, 105])
    inner.setStyle(TableStyle([("VALIGN", (0, 0), (-1, 0), "TOP"), ("VALIGN", (0, 1), (-1, 1), "MIDDLE"), ("VALIGN", (0, 2), (-1, 2), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    cells.append(inner)
cols = [PW, GAP, PW, GAP, PW]
row3 = Table([[cells[0], "", cells[1], "", cells[2]]], colWidths=cols)
row3.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                          ("LINEBELOW", (0, 0), (0, 0), 0, colors.white)]))
story.append(row3)
story.append(Spacer(1, 40))

# --------------------------------------------------------------------------------------- row 4: take-aways
band("What we learned")
story.append(Spacer(1, 28))
for t in ("<b>Memory, not core count,</b> decides how fast the filter runs.",
          "The GPU\u2019s caches already do most of the reuse, so tiling helps <b>only for large filters</b>.",
          "Always count the copying: <b>389\u00d7 becomes 104\u00d7</b> for the whole job."):
    story.append(Paragraph(t, TAKE, bulletText="\u25a0"))

# ---------------------------------------------------------------------------------------------- page chrome
qr = qrcode.make(REPO)
qr.save(FIG / "qr.png")


def decorate(c, doc):
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - HEAD_H, PAGE_W, HEAD_H, stroke=0, fill=1)
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - HEAD_H - 10, PAGE_W, 10, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Times-Bold", 66)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 96, "Shared-Memory, GPU and Distributed")
    c.drawCentredString(PAGE_W / 2, PAGE_H - 172, "Parallelisation of 2D Image Convolution")
    c.setFont("Times", 34)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 232, "Saatvik Das  \u00b7  Saket Rama  \u00b7  Zaid Khan")
    c.setFont("Times-Italic", 26)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 272, "Manipal Institute of Technology, Bengaluru  \u00b7  Guide: Prof. Anil Kumar  \u00b7  PCAP, Semester V")
    c.setFillColor(PALE)
    c.rect(0, 0, PAGE_W, FOOT_H, stroke=0, fill=1)
    c.setFillColor(NAVY)
    c.rect(0, FOOT_H, PAGE_W, 8, stroke=0, fill=1)
    qs = 140
    c.drawImage(ImageReader(str(FIG / "qr.png")), PAGE_W - M - qs, (FOOT_H - qs) / 2, qs, qs)
    c.setFillColor(NAVY)
    c.setFont("Times-Bold", 34)
    c.drawString(M, FOOT_H / 2 + 14, "Code, results and full report")
    c.setFillColor(GREY)
    c.setFont("Times", 28)
    c.drawString(M, FOOT_H / 2 - 28, "github.com/PCAP-ZSS/ConvoX   \u00b7   i7-12650H, RTX 3050 Laptop GPU, CUDA 12.6")
    c.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=(PAGE_W, PAGE_H), title="ConvoX poster", author="Saatvik Das, Saket Rama, Zaid Khan",
                      subject="PCAP project poster")
body_y = FOOT_H + 40
body_h = PAGE_H - HEAD_H - 10 - 44 - body_y
doc.addPageTemplates([PageTemplate(id="poster", frames=[Frame(M, body_y, FULL, body_h, leftPadding=0, rightPadding=0, topPadding=0,
                                                              bottomPadding=0)], onPage=decorate)])
doc.build(story)
print("wrote", OUT, "pages:", doc.page)
