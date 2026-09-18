"""Tools that produce real files on disk.

The agent's built-in `write_file` tool writes into LangGraph state, which is a
virtual filesystem that never touches the disk. Anything the user should be able
to open afterwards has to be written here instead, into `OUTPUT_DIR`.
"""

import re
from pathlib import Path

from docx import Document
from docx.shared import Pt

OUTPUT_DIR = Path.cwd() / "outputs"


def _safe_output_path(filename: str, suffix: str) -> Path:
    """Resolve a model-supplied filename to a path inside OUTPUT_DIR.

    The model chooses this name, so strip any directory part before using it.
    """
    stem = Path(filename).name.strip() or "report"
    if not stem.lower().endswith(suffix):
        stem = f"{Path(stem).stem}{suffix}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / stem


def _add_runs(paragraph, text: str) -> None:
    """Add text to a paragraph, honouring **bold** and *italic* markers."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*\n]+\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)


def write_docx(filename: str, content: str) -> str:
    """Save a report as a Word (.docx) file that the user can open.

    Args:
        filename: File name for the report, e.g. "quarterly-report.docx".
        content: The report body. Markdown is supported: `#` to `######` for
            headings, `-` or `*` for bullets, `1.` for numbered lists,
            `|` tables, and `**bold**` / `*italic*` inline.

    Returns:
        A confirmation message with the saved path.
    """
    path = _safe_output_path(filename, ".docx")
    document = Document()
    lines = content.split("\n")

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1

        if not line:
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            document.add_heading(heading.group(2), level=min(len(heading.group(1)), 4))
            continue

        # table: a header row followed by a |---|---| separator
        next_line = lines[index].strip() if index < len(lines) else ""
        if "|" in line and "-" in next_line and re.fullmatch(r"[\s|:-]+", next_line):
            cells = lambda row: [c.strip() for c in row.strip().strip("|").split("|")]
            header = cells(line)
            index += 1
            rows = []
            while index < len(lines) and "|" in lines[index]:
                rows.append(cells(lines[index]))
                index += 1
            table = document.add_table(rows=1, cols=len(header))
            table.style = "Light Grid Accent 1"
            for cell, text in zip(table.rows[0].cells, header):
                cell.text = text
            for row in rows:
                target = table.add_row().cells
                for cell, text in zip(target, row):
                    cell.text = text
            document.add_paragraph()
            continue

        bullet = re.match(r"^[-*+]\s+(.*)$", line)
        if bullet:
            _add_runs(document.add_paragraph(style="List Bullet"), bullet.group(1))
            continue

        numbered = re.match(r"^\d+[.)]\s+(.*)$", line)
        if numbered:
            _add_runs(document.add_paragraph(style="List Number"), numbered.group(1))
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", line):
            continue

        _add_runs(document.add_paragraph(), line)

    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            if not run.font.size:
                run.font.size = Pt(11)

    document.save(path)
    return f"Saved to {path}. The user can download it from the Files panel."


def write_text_file(filename: str, content: str) -> str:
    """Save a plain text or markdown file that the user can open.

    Args:
        filename: File name, e.g. "notes.md" or "summary.txt".
        content: The file contents.

    Returns:
        A confirmation message with the saved path.
    """
    suffix = Path(filename).suffix or ".txt"
    path = _safe_output_path(filename, suffix)
    path.write_text(content, encoding="utf-8")
    return f"Saved to {path}. The user can download it from the Files panel."
