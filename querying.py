from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize Qdrant client and model
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

def query_documents(collection_name, user_query, top_k=5):
    try:
        # Generate the query vector from the user query
        query_vector = model.encode(user_query).tolist()  # Ensure it's a list

        # Perform the search query
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Process the search results
        results = []
        for res in search_results:
            results.append({
                "id": res.id,
                "score": res.score,
                "payload": res.payload
            })
        
        return results

    except Exception as e:
        print(f"Error during query: {e}")
        return {"error": str(e)}
