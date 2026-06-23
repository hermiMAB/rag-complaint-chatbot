import os
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "cfpb_complaints"
# Point this exactly to your existing folder!
PERSIST_DIR = r"c:\Users\Hermela\Desktop\10academy\Week7\vector_store\chroma"


def stratified_sample(
    df: pd.DataFrame,
    n: int = 12000,
    category_col: str = "product_category",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a stratified sample of n rows, proportional across product categories."""
    fracs = df[category_col].value_counts(normalize=True)
    parts = []
    for cat, frac in fracs.items():
        cat_df = df[df[category_col] == cat]
        k = max(1, round(frac * n))
        k = min(k, len(cat_df))
        parts.append(cat_df.sample(k, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def chunk_dataframe(
    df: pd.DataFrame,
    text_col: str = "processed_feedback",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """Split complaint narratives into overlapping character-level chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    metadata_cols = [
        c for c in df.columns
        if c not in [text_col, "Consumer complaint narrative", "customer_feedback"]
    ]

    records = []
    print("Chunking narratives...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Splitting text"):
        text = row[text_col]
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            meta = {col: str(row[col]) for col in metadata_cols if col in row}
            meta["chunk_index"] = i
            meta["total_chunks"] = len(chunks)
            records.append({"text": chunk, "metadata": meta})
    return records


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: List[Dict],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    print(f"Translating {len(texts):,} text chunks into vector math...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def build_chroma_store(
    chunks: List[Dict],
    embeddings: np.ndarray,
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Persists chunks + embeddings into a ChromaDB collection on disk."""
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 5000
    print("Writing vectors to local ChromaDB database...")
    for start in tqdm(range(0, len(ids), batch_size), desc="Indexing DB"):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
    return collection


def load_chroma_store(
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(collection_name)


def query_store(
    collection: chromadb.Collection,
    question: str,
    model: SentenceTransformer,
    k: int = 5,
    product_filter: Optional[str] = None,
) -> List[Dict]:
    q_embedding = model.encode([question], convert_to_numpy=True).tolist()
    where = {"product_category": product_filter} if product_filter else None
    
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits