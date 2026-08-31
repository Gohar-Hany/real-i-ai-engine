import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import sys

# 1. Load the Models
print("Loading Embedding Model (GTE)...")
embed_model = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)
# Patch for GTE
base_transformer = embed_model[0].auto_model
if hasattr(base_transformer, "embeddings") and hasattr(base_transformer.embeddings, "position_ids"):
    max_positions = base_transformer.config.max_position_embeddings
    base_transformer.embeddings.position_ids = torch.arange(max_positions).expand((1, -1))

print("Loading Reranker Model (BGE)...")
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

# 2. Connect to ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="document_chunks")

# 3. Two-Stage Search function
def perform_reranked_search(query):
    # Stage 1: Vector Retrieval (Top 10)
    print(f"\n[Stage 1] Retrieving top 10 candidates for: '{query}'...")
    instructional_query = f"Instruct: retrieve semantically similar documents.\nQuery: {query}"
    query_emb = embed_model.encode([instructional_query]).tolist()
    
    initial_results = collection.query(
        query_embeddings=query_emb,
        n_results=10
    )
    
    candidates = initial_results['documents'][0]
    metadatas = initial_results['metadatas'][0]
    
    if not candidates:
        print("No candidates found.")
        return

    # Stage 2: Reranking (Top 10 -> Ranked)
    print(f"[Stage 2] Reranking {len(candidates)} candidates with BGE Reranker...")
    
    # Prepare pairs for reranker: [ [query, doc1], [query, doc2], ... ]
    pairs = [[query, doc] for doc in candidates]
    rerank_scores = reranker.predict(pairs)
    
    # Combine results and sort by reranker score (descending)
    reranked_results = []
    for i in range(len(candidates)):
        reranked_results.append({
            "content": candidates[i],
            "metadata": metadatas[i],
            "rerank_score": float(rerank_scores[i])
        })
    
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # 4. Print Results
    print("\n" + "="*50)
    print(f"RERANKED RESULTS FOR: '{query}'")
    print("="*50)
    
    for i, res in enumerate(reranked_results[:5]): # Show top 5 reranked
        print(f"\n[Rank {i+1}] {res['metadata']['section_heading']} (Reranker Score: {res['rerank_score']:.4f})")
        print("-" * 30)
        print(res['content'])
        print("-" * 30)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    search_query = sys.argv[1] if len(sys.argv) > 1 else "python feature"
    perform_reranked_search(search_query)
