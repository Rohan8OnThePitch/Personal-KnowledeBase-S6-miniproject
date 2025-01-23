from qdrant_client import QdrantClient

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Delete a collection
collection_name = "documents"  # Replace with your collection name
qdrant_client.delete_collection(collection_name)

print(f"Collection '{collection_name}' deleted.")