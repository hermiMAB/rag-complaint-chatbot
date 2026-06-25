from __future__ import annotations
from typing import Optional
import os
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints submitted to the CFPB. Use ONLY the retrieved complaint excerpts below to formulate your answer. Cite specific issues mentioned by customers where possible. If the provided context does not contain enough information to answer the question, say so explicitly — do not speculate.

Context:
{context}

Question: {question}

Answer:"""

class RAGPipeline:
    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        k: int = 5,
        max_new_tokens: int = 150,
    ):
        self.k = k
        
        # 1. Dynamically resolve the project root and Chroma directory
        if chroma_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            chroma_dir = os.path.join(project_root, "vector_store", "chroma")
            
        print(f"Connecting to ChromaDB at: {chroma_dir}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        
        # 2. Grab the specific collection name created by the Parquet loader
        self.collection = self.chroma_client.get_collection("complaints_collection")
        
        # 3. Load the Embedding Model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        
        self._llm_model_name = llm_model
        self._max_new_tokens = max_new_tokens
        self._generator = None

    def _get_generator(self):
        if self._generator is None:
            print(f"Loading LLM generator: {self._llm_model_name}...")
            self._generator = hf_pipeline(
                "text-generation",
                model=self._llm_model_name,
                device_map="cpu",
                max_new_tokens=self._max_new_tokens,
                do_sample=False, 
            )
        return self._generator

    def retrieve(self, question: str) -> list[dict]:
        # Convert the question to a vector
        query_embedding = self.embedder.encode(question).tolist()
        
        # Search the Chroma collection natively
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.k
        )
        
        # Format the results into a clean list of dictionaries
        hits = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                hits.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                })
        return hits

    def build_prompt(self, question: str, hits: list[dict], max_chunk_chars: int = 500) -> str:
        context_parts = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            
            # 4. Use the specific metadata keys saved by the Parquet loader
            product = meta.get("Product", "Unknown Product") 
            issue = meta.get("Issue", "General Issue")
            
            header = f"[{i}] Product: {product} | Issue: {issue}"
            text = hit["text"][:max_chunk_chars]
            context_parts.append(f"{header}\n{text}")
            
        context = "\n\n".join(context_parts)
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, prompt: str) -> str:
        gen = self._get_generator()
        output = gen(prompt)[0]["generated_text"]
        if "Answer:" in output:
            return output.split("Answer:")[-1].strip()
        return output[len(prompt):].strip()

    def run(self, question: str) -> dict:
        hits = self.retrieve(question)
        prompt = self.build_prompt(question, hits)
        answer = self.generate(prompt)
        return {"answer": answer, "sources": hits}