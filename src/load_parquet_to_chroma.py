import os
import pandas as pd
import chromadb
from tqdm import tqdm
import ast

def load_prebuilt_store(parquet_file_path: str, chroma_dir: str = "vector_store/chroma"):
    print(f"Loading pre-computed embeddings from: {parquet_file_path}")
    
    # 1. Read the provided parquet file
    df = pd.read_parquet(parquet_file_path)
    print(f"Loaded {len(df)} rows from Parquet file.")
    
    # 2. Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Create or get the collection
    collection = client.get_or_create_collection(
        name="complaints_collection",
        metadata={"hnsw:space": "cosine"} # Use Cosine Similarity
    )
    
    # 3. Prepare data batches (ChromaDB prefers batches of ~5461 or less)
    batch_size = 5000
    
    print(f"Ingesting data into ChromaDB at {chroma_dir}...")
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i : i + batch_size]
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for idx, row in batch_df.iterrows():
            # Generate a unique ID for each chunk
            ids.append(f"chunk_{idx}")
            
            # The actual text paragraph
            # Adjust the column name 'text' or 'chunk' based on what your instructors named it
            documents.append(str(row.get("text", row.get("chunk", ""))))
            
            # The pre-calculated math vector
            # Parquet sometimes saves lists as strings, this ensures it's a list of floats
            emb = row.get("embedding", row.get("embeddings", []))
            if isinstance(emb, str):
                emb = ast.literal_eval(emb)
            embeddings.append(list(emb))
            
            # The sticky notes (metadata)
            meta = {
                "Product": str(row.get("product", "Unknown")),        # Matches your "product" field
                "Issue": str(row.get("issue", "Unknown")),            # Matches your "issue" field
                "Complaint ID": str(row.get("complaint_id", "Unknown")), # Matches your "complaint_id" field
                "Company": str(row.get("company", "Unknown")),        # NEW: Adding missing info
                "State": str(row.get("state", "Unknown")),            # NEW: Adding missing info
                "Category": str(row.get("product_category", "Unknown")) # NEW: Adding missing info
            }
            metadatas.append(meta)
            
        # 4. Add the batch to ChromaDB
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
    print(f"✅ Successfully ingested {len(df)} pre-built vectors into {chroma_dir}!")

if __name__ == "__main__":
    # Dynamically find the project root folder
    # 1. Get the directory where this current script (src) is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up one level to the main project root
    project_root = os.path.dirname(script_dir)
    
    # 3. Construct absolute paths so it never gets lost
    PARQUET_PATH = os.path.join(project_root, "data", "complaint_embeddings.parquet")
    CHROMA_DIR = os.path.join(project_root, "vector_store", "chroma")
    
    load_prebuilt_store(PARQUET_PATH, chroma_dir=CHROMA_DIR)