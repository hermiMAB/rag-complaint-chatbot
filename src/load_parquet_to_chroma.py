import os
import pandas as pd
import chromadb
from tqdm import tqdm
import ast

def load_prebuilt_store(parquet_file_path: str, chroma_dir: str = "vector_store/chroma"):
    print(f"Loading pre-computed embeddings from: {parquet_file_path}")
    
    # 1. Read the provided parquet file
    df = pd.read_parquet(parquet_file_path)
    print("--- DATA DIAGNOSTIC ---")
    print("Available columns in your file:", df.columns.tolist())
    # Print the first row as a dictionary to see exact keys and values
    print("First row data:", df.iloc[0].to_dict()) 
    print("-----------------------")
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
                    
                    # 1. Extract the document text
                    documents.append(str(row.get("document", "")))

                    # 2. Extract embeddings
                    emb = row.get("embedding", [])
                    if isinstance(emb, str):
                        emb = ast.literal_eval(emb)
                    embeddings.append(list(emb))

                    # 3. Extract and parse the nested metadata dictionary
                    raw_meta = row.get("metadata", {})
                    if isinstance(raw_meta, str):
                        raw_meta = ast.literal_eval(raw_meta)

                    # 4. Map the fields (ONLY ONCE)
                    metadatas.append({
                        "Product": str(raw_meta.get("product", "Unknown")),
                        "Issue": str(raw_meta.get("issue", "Unknown")),
                        "Complaint ID": str(raw_meta.get("complaint_id", "Unknown"))
                    })
                    
                # 5. Add the batch to ChromaDB
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