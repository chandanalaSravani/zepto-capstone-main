"""Ingestion + embedding: docs/*.txt -> chunks -> all-MiniLM-L6-v2 -> ChromaDB.

Run once (the API also calls build_index() on startup if the collection is empty):
    python support_assistant/ingest.py
"""

import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # keep ChromaDB fully offline

import chromadb
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
CHROMA_DIR = HERE / "chroma_db"
COLLECTION = "zepto_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"

TITLES = {
    "doc_01": "Delivery Policy",
    "doc_02": "Returns & Refunds",
    "doc_03": "Membership Tiers",
    "doc_04": "Order Tracking",
    "doc_05": "Order Cancellation Policy",
    "doc_06": "Damaged or Missing Items",
    "doc_07": "Gift Cards",
    "doc_08": "Customer Support Hours",
}


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


def embed(texts):
    # normalized vectors, so cosine similarity = dot product
    return get_embedder().encode(list(texts), normalize_embeddings=True).tolist()


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def load_chunks():
    """One chunk per document: each is ~55-90 words, well inside MiniLM's 256-token window,
    and every document covers exactly one policy topic, so splitting further would only
    separate facts that belong together."""
    chunks = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        doc_id = path.stem
        chunks.append({
            "id": f"{doc_id}_chunk_0",
            "text": path.read_text(encoding="utf-8").strip(),
            "metadata": {"doc_id": doc_id, "title": TITLES.get(doc_id, doc_id), "source": path.name},
        })
    return chunks


def build_index(rebuild=False):
    collection = get_collection()
    chunks = load_chunks()
    expected = {chunk["id"]: chunk for chunk in chunks}

    current = collection.get(include=["documents"])
    current_ids = current["ids"]
    current_docs = dict(zip(current_ids, current.get("documents") or []))

    if rebuild:
        if current_ids:
            collection.delete(ids=current_ids)
        current_docs = {}
    else:
        stale_ids = [doc_id for doc_id in current_ids if doc_id not in expected]
        if stale_ids:
            collection.delete(ids=stale_ids)

    # Add missing chunks and re-embed chunks whose text changed.
    to_upsert = [
        chunk for chunk in chunks
        if current_docs.get(chunk["id"]) != chunk["text"]
    ]

    if to_upsert:
        collection.upsert(
            ids=[chunk["id"] for chunk in to_upsert],
            documents=[chunk["text"] for chunk in to_upsert],
            metadatas=[chunk["metadata"] for chunk in to_upsert],
            embeddings=embed(chunk["text"] for chunk in to_upsert),
        )

    indexed_ids = set(collection.get(include=["documents"])["ids"])
    if indexed_ids != set(expected):
        raise RuntimeError(
            f"Expected {len(expected)} policy chunks, found {len(indexed_ids)}"
        )

    return collection


def retrieve(query, k=3):
    """Top-k chunks by cosine similarity (Chroma returns cosine *distance* = 1 - similarity)."""
    res = get_collection().query(query_embeddings=embed([query]), n_results=k)
    return [
        {"id": cid, "text": doc, "similarity": round(1 - dist, 4), **meta}
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0])
    ]


if __name__ == "__main__":
    col = build_index(rebuild=True)
    print(f"Indexed {col.count()} chunks into ChromaDB collection '{COLLECTION}' at {CHROMA_DIR}")
    for q in ["How much is the delivery fee?", "Can I get a refund on opened shampoo?",
              "What does Zepto Pass+ include?", "Is there phone support?"]:
        top = retrieve(q)
        print(f"\n{q}")
        for r in top:
            print(f"  {r['similarity']:.3f}  {r['id']:15s} {r['title']}")
