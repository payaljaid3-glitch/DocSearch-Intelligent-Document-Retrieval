import sqlite3
from pathlib import Path


# Database location
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "documind.db"


def get_connection():
    """Create a connection to the SQLite database."""

    DB_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database():
    """Create database tables if they do not exist."""

    connection = get_connection()

    cursor = connection.cursor()

    # Documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Document chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_number INTEGER NOT NULL,
            page_number INTEGER,
            text TEXT NOT NULL,

            FOREIGN KEY (document_id)
            REFERENCES documents(id)
            ON DELETE CASCADE
        )
    """)

    # Improve document-based retrieval
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id
        ON chunks(document_id)
    """)

    connection.commit()
    connection.close()

def save_processed_document(file_path, chunks):
    """Save document information and its chunks in the database."""

    file_path = Path(file_path)

    connection = get_connection()
    cursor = connection.cursor()

    try:
        # Save document details
        cursor.execute("""
            INSERT INTO documents
            (document_name, file_type, file_size)
            VALUES (?, ?, ?)
        """, (
            file_path.name,
            file_path.suffix.lower().replace(".", ""),
            file_path.stat().st_size
        ))

        document_id = cursor.lastrowid

        # Save all chunks
        for chunk in chunks:

            cursor.execute("""
                INSERT INTO chunks
                (document_id, chunk_number, page_number, text)
                VALUES (?, ?, ?, ?)
            """, (
                document_id,
                chunk["chunk_number"],
                chunk["page_number"],
                chunk["text"]
            ))

        connection.commit()

        return document_id

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_all_documents():
    """Retrieve all stored documents."""

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, document_name, file_type, file_size, uploaded_at
        FROM documents
        ORDER BY id DESC
    """)

    documents = cursor.fetchall()

    connection.close()

    return documents

def get_document_chunks(document_id):
    """Retrieve all chunks belonging to a document."""

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            documents.document_name,
            documents.file_type,
            chunks.chunk_number,
            chunks.page_number,
            chunks.text

        FROM chunks

        JOIN documents
        ON chunks.document_id = documents.id

        WHERE documents.id = ?

        ORDER BY chunks.chunk_number
    """, (document_id,))

    chunks = cursor.fetchall()

    connection.close()

    return chunks

if __name__ == "__main__":

    initialize_database()

    print("DocuMind AI Database Created Successfully!")
    print("Database Location:", DB_PATH)