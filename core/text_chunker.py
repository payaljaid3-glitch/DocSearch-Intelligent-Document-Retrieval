def chunk_text(text, chunk_size=500, overlap=100):
    """
    Divide text into smaller chunks using words.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_documents(extracted_sections, chunk_size=500, overlap=100):
    """
    Create chunks from extracted document sections.
    Preserve page numbers and chunk numbers.
    """

    all_chunks = []

    chunk_number = 1

    for section in extracted_sections:

        text = section.get("text", "")
        page_number = section.get("page_number")

        if not text or not text.strip():
            continue

        text_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in text_chunks:

            all_chunks.append({
                "chunk_number": chunk_number,
                "page_number": page_number,
                "text": chunk
            })

            chunk_number += 1

    return all_chunks