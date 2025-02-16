from qdrant_client import QdrantClient
import os

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# List all collections
collections = client.get_collections()
print("Available Collections:", collections)

# Count indexed documents
collection_name = "documents"  # Change if needed
info = client.get_collection(collection_name)
print("Collection Info:", info)
