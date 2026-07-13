"""Render docs/DEVELOPER_GUIDE.md to docs/DEVELOPER_GUIDE.pdf.

Usage:
    python scripts/build_developer_guide_pdf.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
MD_PATH = DOCS_DIR / "DEVELOPER_GUIDE.md"
PDF_PATH = DOCS_DIR / "DEVELOPER_GUIDE.pdf"

CSS = """
@page {
    size: letter;
    margin: 2.2cm 1.8cm;
    @frame footer_frame {
        -pdf-frame-content: footer_content;
        bottom: 1cm; margin-left: 1.8cm; margin-right: 1.8cm; height: 1cm;
    }
}
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; color: #14324d; border-bottom: 2pt solid #14324d; padding-bottom: 6pt; margin-top: 0; }
h2 { font-size: 15pt; color: #14324d; margin-top: 20pt; border-bottom: 0.75pt solid #b9c6d0; padding-bottom: 3pt; }
h3 { font-size: 12pt; color: #1c4a70; margin-top: 14pt; }
p { margin: 6pt 0; text-align: left; }
ul, ol { margin: 4pt 0 8pt 0; padding-left: 16pt; }
li { margin: 2pt 0; }
code { font-family: Courier, monospace; font-size: 8.5pt; background-color: #f0f2f5; padding: 1pt 3pt; }
pre { font-family: Courier, monospace; font-size: 8pt; background-color: #f0f2f5; padding: 8pt; border: 0.5pt solid #d0d5db;
      line-height: 1.3; white-space: pre-wrap; }
pre code { background-color: transparent; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; }
th { background-color: #14324d; color: #ffffff; padding: 4pt 6pt; font-size: 8.5pt; text-align: left; }
td { border: 0.5pt solid #c7ccd1; padding: 4pt 6pt; font-size: 8.5pt; vertical-align: top; }
tr:nth-child(even) td { background-color: #f6f8fa; }
blockquote { border-left: 3pt solid #14324d; margin: 6pt 0; padding: 2pt 10pt; color: #333; background-color: #f6f8fa; }
hr { border: none; border-top: 0.5pt solid #c7ccd1; margin: 14pt 0; }
a { color: #14324d; }
#footer_content { font-size: 7.5pt; color: #888; text-align: center; }
"""

FOOTER = '<div id="footer_content">Credit Default Prediction Capstone — Developer Guide — Page <pdf:pagenumber/></div>'

PRE_BLOCK_RE = re.compile(r"(<pre>.*?</pre>)", re.DOTALL)


def _force_line_breaks_in_pre(html: str) -> str:
    """xhtml2pdf does not reliably honor embedded newlines inside <pre>
    blocks (CSS white-space: pre-wrap is not fully supported), so replace
    each newline with an explicit <br/> as well.
    """

    def _replace(match: re.Match) -> str:
        return match.group(1).replace("\n", "<br/>\n")

    return PRE_BLOCK_RE.sub(_replace, html)


def build() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        md_text,
        extensions=["extra", "sane_lists", "toc"],
    )
    body_html = _force_line_breaks_in_pre(body_html)
    full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
{body_html}
{FOOTER}
</body>
</html>"""

    with open(PDF_PATH, "wb") as f:
        result = pisa.CreatePDF(full_html, dest=f)

    if result.err:
        print(f"PDF generation completed with {result.err} error(s)", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
