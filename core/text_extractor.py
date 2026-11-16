import pymupdf
from docx import Document
from pathlib import Path


def extract_pdf(file_path):
    """Extract text from a PDF file, page by page."""

    extracted_pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if text:
                extracted_pages.append({
                    "page_number": page_number,
                    "text": text
                })

    return extracted_pages


def extract_docx(file_path):
    """Extract text from a DOCX document."""

    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [{
        "page_number": None,
        "text": text
    }]


def extract_txt(file_path):
    """Extract text from a TXT file."""

    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    return [{
        "page_number": None,
        "text": text
    }]


def extract_text(file_path):
    """Automatically select the correct extraction method."""

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    elif extension == ".docx":
        return extract_docx(file_path)

    elif extension == ".txt":
        return extract_txt(file_path)

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )