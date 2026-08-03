"""Typography/formatting pass over the Interim Report docx: consistent font,
size, color, and spacing throughout (headings, body text, captions,
bibliography, tables, code blocks). Content and section structure are left
untouched -- this only changes how the existing text looks.

Usage:
    python scripts/format_interim_report.py
"""
from __future__ import annotations

from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

DOC_PATH = Path(r"C:\Project\MS\DAC\QM640 Data Analytics Capstone Interim Report Sailen (UCI Credit Card).docx")

BODY_FONT = "Times New Roman"
MONO_FONT = "Consolas"

BLACK = RGBColor(0x00, 0x00, 0x00)
NAVY = RGBColor(0x1F, 0x38, 0x64)
GRAY = RGBColor(0x59, 0x59, 0x59)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

doc = docx.Document(str(DOC_PATH))


# ---------------------------------------------------------------------------
# 1. Paragraph styles
# ---------------------------------------------------------------------------
def style_font(style_name, name=BODY_FONT, size=12, color=BLACK, bold=None, italic=None):
    st = doc.styles[style_name]
    f = st.font
    f.name = name
    f.size = Pt(size)
    f.color.rgb = color
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    rpr = st.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)
    return st


def style_spacing(style_name, before=0, after=0, line=1.5, exact_line=None, keep_with_next=None):
    st = doc.styles[style_name]
    pf = st.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if exact_line is not None:
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing = Pt(exact_line)
    else:
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = line
    if keep_with_next is not None:
        pf.keep_with_next = keep_with_next


# Body text
style_font("Normal", size=12, color=BLACK)
style_spacing("Normal", before=0, after=6, line=1.5)

style_font("LO-normal", size=11, color=BLACK)
style_spacing("LO-normal", before=0, after=2, line=1.15)

style_font("List Bullet", size=12, color=BLACK)
style_spacing("List Bullet", before=0, after=6, line=1.5)

style_font("Body Text", size=12, color=BLACK)
style_spacing("Body Text", before=0, after=6, line=1.5)

# Headings
style_font("Heading 1", size=16, color=NAVY, bold=True)
style_spacing("Heading 1", before=22, after=10, line=1.15, keep_with_next=True)
doc.styles["Heading 1"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

style_font("APA Heading 2", size=14, color=NAVY, bold=True)
style_spacing("APA Heading 2", before=16, after=8, line=1.15, keep_with_next=True)
doc.styles["APA Heading 2"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

style_font("Heading 2", size=13, color=NAVY, bold=True)
style_spacing("Heading 2", before=13, after=6, line=1.15, keep_with_next=True)

style_font("Heading 3", size=12, color=GRAY, bold=True, italic=True)
style_spacing("Heading 3", before=10, after=4, line=1.15, keep_with_next=True)

# Code blocks
style_font("Preformatted Text", name=MONO_FONT, size=9, color=BLACK)
style_spacing("Preformatted Text", before=2, after=2, line=1.0)


def shade_style_background(style_name, hex_color):
    st = doc.styles[style_name]
    pPr = st.element.get_or_add_pPr()
    shd = parse_xml(f'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                     f'w:val="clear" w:color="auto" w:fill="{hex_color}"/>')
    pPr.append(shd)


shade_style_background("Preformatted Text", "F2F2F2")

print("Styles normalized.")

# ---------------------------------------------------------------------------
# 2. Force every run's direct formatting to match its (now-correct) style,
#    so leftover direct overrides from earlier edits can't fight the style.
# ---------------------------------------------------------------------------
BODY_STYLES = {"Normal": 12, "List Bullet": 12, "Body Text": 12}
for p in doc.paragraphs:
    if p.style.name in BODY_STYLES:
        size = BODY_STYLES[p.style.name]
        for r in p.runs:
            r.font.name = BODY_FONT
            r.font.size = Pt(size)
            r.font.color.rgb = BLACK
    elif p.style.name in ("Heading 1", "APA Heading 2", "Heading 2", "Heading 3"):
        size = {"Heading 1": 16, "APA Heading 2": 14, "Heading 2": 13, "Heading 3": 12}[p.style.name]
        color = GRAY if p.style.name == "Heading 3" else NAVY
        for r in p.runs:
            r.font.name = BODY_FONT
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.bold = True
            if p.style.name == "Heading 3":
                r.italic = True

print("Run-level overrides normalized for body/heading paragraphs.")

# ---------------------------------------------------------------------------
# 3. LO-normal paragraphs play three different roles at the body level:
#    title-page metadata, figure captions, and bibliography entries.
#    Style each role distinctly with direct paragraph formatting.
# ---------------------------------------------------------------------------
in_bibliography = False
for p in doc.paragraphs:
    if p.style.name == "APA Heading 2":
        in_bibliography = p.text.strip() == "References and Bibliography"
        continue
    if p.style.name != "LO-normal" or not p.text.strip():
        continue
    text = p.text.strip()

    if text.startswith("Figure "):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.space_before = Pt(4)
        pf.space_after = Pt(14)
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
        for r in p.runs:
            r.font.name = BODY_FONT
            r.font.size = Pt(10.5)
            r.font.color.rgb = GRAY
            r.italic = True
    elif in_bibliography:
        pf = p.paragraph_format
        pf.left_indent = Inches(0.5)
        pf.first_line_indent = Inches(-0.5)
        pf.space_after = Pt(10)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.5
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for r in p.runs:
            r.font.name = BODY_FONT
            r.font.size = Pt(12)
            r.font.color.rgb = BLACK
            r.italic = False
    else:
        # Title-page metadata block.
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.space_after = Pt(4)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.15
        for r in p.runs:
            r.font.name = BODY_FONT
            r.font.size = Pt(13)
            r.font.color.rgb = BLACK

print("Caption / bibliography / title-page formatting applied.")

# ---------------------------------------------------------------------------
# 4. Bold inline "Table N -- ..." caption labels (Normal-styled, bold=True)
#    get the same visual treatment as figure captions minus the italics.
# ---------------------------------------------------------------------------
for p in doc.paragraphs:
    if p.style.name == "Normal" and p.text.strip().startswith("Table ") and p.runs and p.runs[0].bold:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf = p.paragraph_format
        pf.space_before = Pt(10)
        pf.space_after = Pt(6)
        for r in p.runs:
            r.font.name = BODY_FONT
            r.font.size = Pt(12)
            r.font.color.rgb = NAVY
            r.bold = True

print("Table caption labels styled.")

# ---------------------------------------------------------------------------
# 5. Tables: consistent font (tiered by column count), navy header row with
#    white bold text, centered vertical alignment, single-spaced cells.
# ---------------------------------------------------------------------------
HEADER_FILL = "1F3864"


def shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                     f'w:val="clear" w:color="auto" w:fill="{hex_color}"/>')
    tcPr.append(shd)


def font_size_for_cols(n_cols: int) -> float:
    if n_cols <= 3:
        return 10.5
    if n_cols <= 4:
        return 10
    if n_cols <= 6:
        return 9
    return 8


for t in doc.tables:
    n_cols = len(t.columns)
    size = font_size_for_cols(n_cols)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for ri, row in enumerate(t.rows):
        is_header = ri == 0
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if is_header:
                shade_cell(cell, HEADER_FILL)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                if not p.runs and p.text == "":
                    continue
                for r in p.runs:
                    r.font.name = BODY_FONT
                    r.font.size = Pt(size)
                    r.font.color.rgb = WHITE if is_header else BLACK
                    r.bold = True if is_header else r.bold

print(f"Formatted {len(doc.tables)} tables.")

# ---------------------------------------------------------------------------
# 6. Preformatted (code) blocks: normalize direct run overrides to match the
#    Preformatted Text style set above.
# ---------------------------------------------------------------------------
for p in doc.paragraphs:
    if p.style.name == "Preformatted Text":
        for r in p.runs:
            r.font.name = MONO_FONT
            r.font.size = Pt(9)
            r.font.color.rgb = BLACK

print("Code blocks normalized.")

doc.save(str(DOC_PATH))
print("Saved ->", DOC_PATH)
