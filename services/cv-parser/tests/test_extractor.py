"""Tests for the PDF/DOCX/TXT extraction layer.

The original test suite only exercised the .txt code path (via the shared
sample_resume.txt fixture) — extract_document's PDF and DOCX branches, and
the file-signature validation, had zero coverage. These build minimal but
real PDF/DOCX files at test time so those paths are actually driven.
"""

from __future__ import annotations

import io

import docx
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.parser.extractor import (
    EmptyDocumentError,
    FileSignatureMismatchError,
    UnsupportedFileTypeError,
    extract_document,
)


def _build_pdf_bytes(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 72
    for line in lines:
        c.drawString(72, y, line)
        y -= 16
    c.save()
    return buf.getvalue()


def _build_docx_bytes(paragraphs: list[str]) -> bytes:
    document = docx.Document()
    for para in paragraphs:
        document.add_paragraph(para)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


RESUME_LINES = [
    "Jordan Lee",
    "jordan.lee@example.com",
    "EDUCATION",
    "B.Sc Computer Science, State University 2019 - 2023",
    "SKILLS",
    "Python, Docker, PostgreSQL",
]


def test_pdf_extraction_round_trip():
    data = _build_pdf_bytes(RESUME_LINES)
    doc = extract_document("resume.pdf", data)
    assert doc.file_type == "pdf"
    assert doc.page_count == 1
    assert "Jordan Lee" in doc.text
    assert "PostgreSQL" in doc.text


def test_docx_extraction_round_trip():
    data = _build_docx_bytes(RESUME_LINES)
    doc = extract_document("resume.docx", data)
    assert doc.file_type == "docx"
    assert "Jordan Lee" in doc.text
    assert "PostgreSQL" in doc.text


def test_docx_extracts_table_content():
    document = docx.Document()
    document.add_paragraph("EXPERIENCE")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Role"
    table.rows[0].cells[1].text = "Software Engineer"
    buf = io.BytesIO()
    document.save(buf)

    doc = extract_document("resume.docx", buf.getvalue())
    assert "Software Engineer" in doc.text


def test_txt_extraction_utf8():
    data = "Café Résumé\nSkills: C++".encode("utf-8")
    doc = extract_document("resume.txt", data)
    assert "Café Résumé" in doc.text


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileTypeError):
        extract_document("resume.rtf", b"whatever")


def test_empty_pdf_text_layer_raises_or_warns():
    # A PDF with a valid signature but no actual pages/text should not
    # silently return an empty ParsedResume-worthy document.
    empty_pdf = _build_pdf_bytes([])
    with pytest.raises(EmptyDocumentError):
        extract_document("resume.pdf", empty_pdf)


def test_pdf_signature_mismatch_raises():
    with pytest.raises(FileSignatureMismatchError):
        extract_document("resume.pdf", b"not actually a pdf")


def test_docx_signature_mismatch_raises():
    with pytest.raises(FileSignatureMismatchError):
        extract_document("resume.docx", b"not actually a zip/docx")


def test_txt_has_no_signature_check():
    # .txt has no reliable magic bytes, so arbitrary content is accepted.
    doc = extract_document("resume.txt", b"just plain text, no signature")
    assert "just plain text" in doc.text
