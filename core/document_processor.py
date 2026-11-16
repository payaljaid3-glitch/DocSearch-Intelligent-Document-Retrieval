from pathlib import Path

from core.text_extractor import extract_text
from core.text_chunker import chunk_documents


def process_document(file_path, chunk_size=500, overlap=100):
    """
    Extract, chunk and attach metadata to a document.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    # Document metadata
    document_name = file_path.name
    file_type = file_path.suffix.lower().replace(".", "")

    # Extract text
    extracted_sections = extract_text(file_path)

    # Create chunks
    chunks = chunk_documents(
        extracted_sections,
        chunk_size=chunk_size,
        overlap=overlap
    )

    # Attach metadata to every chunk
    processed_chunks = []

    for chunk in chunks:

        processed_chunks.append({
            "document_name": document_name,
            "file_type": file_type,
            "chunk_number": chunk["chunk_number"],
            "page_number": chunk["page_number"],
            "text": chunk["text"]
        })

    return processed_chunks