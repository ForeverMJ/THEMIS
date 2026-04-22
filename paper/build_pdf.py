from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent
PAPER_MD = ROOT / "paper.md"
BIB_FILE = ROOT / "references.bib"
PDF_FILE = ROOT / "paper.pdf"


def parse_bibtex(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    current_key = None
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("@"):
            if current_key and current:
                entries[current_key] = current
            current = {}
            match = re.match(r"@\w+\{([^,]+),", line)
            current_key = match.group(1) if match else None
            continue
        if line == "}":
            if current_key and current:
                entries[current_key] = current
            current_key = None
            current = {}
            continue
        if current_key and "=" in line:
            field, value = line.split("=", 1)
            field = field.strip().lower()
            value = value.strip().rstrip(",").strip()
            if value.startswith("{") and value.endswith("}"):
                value = value[1:-1]
            current[field] = value
    if current_key and current:
        entries[current_key] = current
    return entries


def format_reference(entry: dict[str, str]) -> str:
    author = entry.get("author", "Unknown author")
    year = entry.get("year", "n.d.")
    title = entry.get("title", "Untitled")
    venue = entry.get("booktitle") or entry.get("journal") or entry.get("howpublished", "")
    doi = entry.get("doi", "")
    url = entry.get("url", "")
    parts = [author, f"({year})", f"<i>{title}</i>"]
    if venue:
        parts.append(venue)
    if doi:
        parts.append(f"DOI: {doi}")
    elif url:
        parts.append(url)
    return ". ".join(p for p in parts if p)


def build_story(md_text: str, bib_entries: dict[str, dict[str, str]]):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    ref_style = ParagraphStyle(
        "Ref",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-10,
    )

    cite_order: list[str] = []
    cite_index: dict[str, int] = {}

    def replace_cites(text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            keys = [k.strip() for k in match.group(1).split(",") if k.strip()]
            nums = []
            for key in keys:
                if key not in cite_index:
                    cite_order.append(key)
                    cite_index[key] = len(cite_order)
                nums.append(str(cite_index[key]))
            return f"[{', '.join(nums)}]"

        return re.sub(r"\\cite\{([^}]+)\}", repl, text)

    def md_inline(text: str) -> str:
        text = replace_cites(text)
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
        return text

    story = []
    lines = md_text.splitlines()
    paragraph_buffer: list[str] = []

    def flush_paragraph():
        if paragraph_buffer:
            text = " ".join(s.strip() for s in paragraph_buffer).strip()
            if text:
                story.append(Paragraph(md_inline(text), body_style))
            paragraph_buffer.clear()

    for line in lines:
        if line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(md_inline(line[2:].strip()), title_style))
            story.append(Spacer(1, 4))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(md_inline(line[3:].strip()), h2_style))
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph_buffer.append(line)
    flush_paragraph()

    if cite_order:
        story.append(Paragraph("References", h2_style))
        for key in cite_order:
            if key in bib_entries:
                story.append(Paragraph(f"[{cite_index[key]}] {md_inline(format_reference(bib_entries[key]))}", ref_style))
            else:
                story.append(Paragraph(f"[{cite_index[key]}] Missing bibliography entry for key: {key}", ref_style))
    return story


def main() -> None:
    md_text = PAPER_MD.read_text(encoding="utf-8")
    bib_entries = parse_bibtex(BIB_FILE)
    story = build_story(md_text, bib_entries)
    doc = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="A Controlled Trio Study of Constraint-Guided Semantic Repair under a Bounded Fault Space",
        author="Sisyphus",
    )
    doc.build(story)


if __name__ == "__main__":
    main()
