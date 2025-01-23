from qdrant_client import QdrantClient

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Get collection info
collection_info = client.get_collection(collection_name="documents")
print(collection_info)