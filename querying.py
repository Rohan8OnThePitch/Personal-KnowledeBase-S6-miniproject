from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

# Initialize Qdrant client and model
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Consider making this a global constant

def query_documents(collection_name, user_query, top_k=20):  # Increased top_k
    """Queries the Qdrant collection for documents similar to the user query."""
    try:
        print(f"Querying collection: {collection_name} for query: {user_query}") #DEBUG
        # Generate the query vector from the user query
        query_vector = model.encode(user_query).tolist()  # Ensure it's a list

        # Perform the search query
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True  # Retrieve the payload (text)
        )
        print(f"Search Results{search_results}")
        # Process the search results
        results = []
        for res in search_results:
            print(f"Payload: {res.payload}") #DEBUG
            results.append({
                "id": res.id,
                "score": res.score,
                "text": res.payload.get("text", "No text available") if res.payload else "No payload available" #Extract text from payload
            })

        print(f"Query results: {results}") #DEBUG
        return results

    except Exception as e:
        print(f"Error during query: {e}")
        return {"error": str(e)}
