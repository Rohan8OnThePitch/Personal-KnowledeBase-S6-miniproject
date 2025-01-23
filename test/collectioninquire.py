from qdrant_client import QdrantClient

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# List all collections
collections = qdrant_client.get_collections()
print("Collections:", collections)

# Check points in the collection
collection_name = "documents"  # Replace with your collection name
points = qdrant_client.scroll(collection_name=collection_name, limit=10)
print("Points in collection:", points)