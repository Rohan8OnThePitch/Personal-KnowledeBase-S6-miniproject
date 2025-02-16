from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
import preprocess

# Initialize Qdrant client and model
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Consider making this a global constant

def query_documents(collection_name, user_query, top_k=5):
    """Queries Qdrant and retrieves matching documents."""
    try:
        print(f"Original Query: {user_query}")
        user_query = preprocess.preprocess_text(user_query)
        print(f"Preprocessed Query: {user_query}")

        query_vector = model.encode(user_query).tolist()
        
        # Search with no filters
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        if not search_results:
            print("No results found. Try increasing top_k or checking indexing.")
        
        results = [{"id": res.id, "score": res.score, "text": res.payload["text"]}
                   for res in search_results if res.payload]
        
        print(f"Query Results: {results}")  # Debugging
        return results

    except Exception as e:
        print(f"Error during query: {e}")
        return {"error": str(e)}

