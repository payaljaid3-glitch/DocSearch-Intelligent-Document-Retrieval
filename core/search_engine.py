import re

import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.database import get_all_documents, get_document_chunks


# ==========================================
# 1. LOAD DOCUMENT CHUNKS
# ==========================================

def load_document_chunks():
    """Load all uploaded document chunks from SQLite."""

    documents = get_all_documents()

    records = []

    for document in documents:

        document_id = document[0]

        chunks = get_document_chunks(document_id)

        for chunk in chunks:

            records.append({
                "document_id": document_id,
                "document_name": chunk[0],
                "file_type": chunk[1],
                "chunk_number": chunk[2],
                "page_number": chunk[3],
                "text": chunk[4]
            })

    return pd.DataFrame(records)


# ==========================================
# 2. TOKENIZATION
# ==========================================

def tokenize(text):
    """Convert text into lowercase tokens."""

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        str(text).lower()
    )


# ==========================================
# 3. BUILD BM25 INDEX
# ==========================================

def build_bm25_index(texts):

    tokenized_corpus = [
        tokenize(text)
        for text in texts
    ]

    return BM25Okapi(tokenized_corpus)


# ==========================================
# 4. LOAD SEMANTIC MODEL
# ==========================================

def load_semantic_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ==========================================
# 5. BUILD SEMANTIC INDEX
# ==========================================

def build_semantic_index(texts, model):

    return model.encode(
        list(texts),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )


# ==========================================
# 6. BM25 SEARCH
# ==========================================

def bm25_search(query, df, bm25, top_k=5):

    query_tokens = tokenize(query)

    if not query_tokens:
        return pd.DataFrame()

    scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(scores)[::-1]

    ranked_indices = [
        idx for idx in ranked_indices
        if scores[idx] > 0
    ]

    results = df.iloc[ranked_indices].copy()

    results["score"] = scores[ranked_indices]

    return results.head(top_k).reset_index(drop=True)


# ==========================================
# 7. SEMANTIC SEARCH
# ==========================================

def semantic_search(query, df, model, embeddings, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(embeddings, query_embedding[0])

    ranked_indices = np.argsort(scores)[::-1]

    results = df.iloc[ranked_indices].copy()

    results["score"] = scores[ranked_indices]

    return results.head(top_k).reset_index(drop=True)


# ==========================================
# 8. HYBRID SEARCH
# ==========================================

def hybrid_search(
    query,
    df,
    bm25,
    model,
    embeddings,
    top_k=5,
    k=60
):

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_candidates = np.flatnonzero(
        bm25_scores > 0
    )

    bm25_ranking = bm25_candidates[
        np.argsort(
            bm25_scores[bm25_candidates]
        )[::-1]
    ]

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    semantic_ranking = np.argsort(
        semantic_scores
    )[::-1]

    # Reciprocal Rank Fusion

    rrf_scores = np.zeros(
        len(df),
        dtype=float
    )

    for rank, idx in enumerate(
        bm25_ranking,
        start=1
    ):
        rrf_scores[idx] += 1 / (k + rank)

    for rank, idx in enumerate(
        semantic_ranking,
        start=1
    ):
        rrf_scores[idx] += 1 / (k + rank)

    final_ranking = np.argsort(
        rrf_scores
    )[::-1]

    results = df.iloc[final_ranking].copy()

    results["score"] = rrf_scores[final_ranking]

    results["bm25_score"] = bm25_scores[final_ranking]

    results["semantic_score"] = semantic_scores[final_ranking]

    return results.head(top_k).reset_index(drop=True)