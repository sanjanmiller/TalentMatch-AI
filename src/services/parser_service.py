
"""
File: parser_service.py

Purpose:
Extract readable text from PDF, DOCX, and TXT documents for the
Intelligent Recruitment System.

This service supports PDF and DOCX resumes, and PDF and TXT job
descriptions. It also supports DOCX job descriptions if HR uploads one.

This file only extracts text. It does not extract structured candidate
details, evaluate candidates, create embeddings, or save database records.
"""

from pathlib import Path

import pymupdf
from docx import Document


class DocumentParsingError(Exception):
    """Raise this error when a document cannot be read or parsed."""


def _validate_file(file_path: str | Path) -> Path:
    """Check that the supplied path points to an existing file."""

    path = Path(file_path)

    if not path.exists():
        raise DocumentParsingError(
            f"Document does not exist: {path}"
        )

    if not path.is_file():
        raise DocumentParsingError(
            f"The supplied path is not a file: {path}"
        )

    return path


def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from all pages of a PDF document."""

    try:
        # Use the modern PyMuPDF import: import pymupdf.
        with pymupdf.open(file_path) as pdf_document:
            page_text = []

            for page in pdf_document:
                page_text.append(page.get_text("text"))

        return "\n".join(page_text).strip()

    except Exception as exc:
        raise DocumentParsingError(
            f"Could not read PDF document: {file_path.name}"
        ) from exc


def _extract_docx_text(file_path: Path) -> str:
    """Extract text from paragraphs and tables in a DOCX document."""

    try:
        document = Document(file_path)
        text_parts = []

        # Extract ordinary paragraphs.
        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)

        # Extract table content because resumes may use tables
        # to organize skills, education, and work experience.
        for table in document.tables:
            for row in table.rows:
                cell_text = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]

                if cell_text:
                    text_parts.append(" | ".join(cell_text))

        return "\n".join(text_parts).strip()

    except Exception as exc:
        raise DocumentParsingError(
            f"Could not read DOCX document: {file_path.name}"
        ) from exc


def _extract_txt_text(file_path: Path) -> str:
    """Read a plain-text document using UTF-8 encoding."""

    try:
        # utf-8-sig also handles files that begin with a UTF-8 BOM.
        return file_path.read_text(
            encoding="utf-8-sig"
        ).strip()

    except (UnicodeDecodeError, OSError) as exc:
        raise DocumentParsingError(
            f"Could not read TXT document: {file_path.name}"
        ) from exc


def extract_text_from_file(file_path: str | Path) -> str:
    """
    Extract text from a supported document.

    Args:
        file_path: Path to a PDF, DOCX, or TXT file.

    Returns:
        The extracted text as a string.

    Raises:
        DocumentParsingError: If the file is missing, unsupported,
            unreadable, or contains no extractable text.
    """

    path = _validate_file(file_path)
    extension = path.suffix.lower()

    # Choose the correct extraction method based on the file extension.
    if extension == ".pdf":
        extracted_text = _extract_pdf_text(path)

    elif extension == ".docx":
        extracted_text = _extract_docx_text(path)

    elif extension == ".txt":
        extracted_text = _extract_txt_text(path)

    else:
        raise DocumentParsingError(
            f"Unsupported document format: {extension or 'no extension'}"
        )

    # An empty result may indicate an empty document or a scanned PDF
    # that requires OCR, which this version does not yet implement.
    if not extracted_text:
        raise DocumentParsingError(
            f"No readable text was extracted from: {path.name}"
        )

    return extracted_text
