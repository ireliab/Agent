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


UPLOAD_DIR = Path.cwd() / "uploads"
MAX_EXTRACTED_CHARS = 20000


def _truncate(text: str) -> str:
    if len(text) <= MAX_EXTRACTED_CHARS:
        return text
    return text[:MAX_EXTRACTED_CHARS] + f"\n\n... (truncated at {MAX_EXTRACTED_CHARS} characters)"


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [f"--- page {i} ---\n{(page.extract_text() or '').strip()}" for i, page in enumerate(reader.pages, 1)]
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    document = Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def _read_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), data_only=True, read_only=True)
    out = []
    for sheet in workbook.worksheets:
        out.append(f"--- sheet: {sheet.title} ---")
        for row in sheet.iter_rows(values_only=True):
            if any(cell is not None for cell in row):
                out.append(" | ".join("" if cell is None else str(cell) for cell in row))
    workbook.close()
    return "\n".join(out)


_READERS = {".pdf": _read_pdf, ".docx": _read_docx, ".xlsx": _read_xlsx, ".xlsm": _read_xlsx}


def read_document(filename: str) -> str:
    """Read a file the user uploaded, and return its text.

    Supports .pdf, .docx, .xlsx, .csv, .txt and .md. Use this whenever the user
    refers to a file they attached.

    Args:
        filename: Name of the uploaded file, e.g. "report.pdf".

    Returns:
        The file's text content, truncated if very long.
    """
    path = UPLOAD_DIR / Path(filename).name
    if not path.is_file():
        available = sorted(p.name for p in UPLOAD_DIR.iterdir()) if UPLOAD_DIR.exists() else []
        return f"No uploaded file named {filename!r}. Available uploads: {available or 'none'}"

    reader = _READERS.get(path.suffix.lower())
    try:
        text = reader(path) if reader else path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read {filename!r}: {type(exc).__name__}: {exc}"

    return _truncate(text) if text.strip() else f"{filename!r} contains no extractable text."


def list_uploads() -> str:
    """List the files the user has uploaded in this session."""
    if not UPLOAD_DIR.exists():
        return "No files uploaded."
    names = sorted(p.name for p in UPLOAD_DIR.iterdir() if p.is_file())
    return "\n".join(names) if names else "No files uploaded."
