from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# Connect to the local Qdrant instance
client = QdrantClient(url="http://localhost:6333")

# Define collection name and vector parameters
collection_name = "documents"
vector_size = 384# Adjust size based on your embeddings
distance_metric = "Cosine"  # Choose from "euclid", "cosine", or "dot"

# Create the collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=distance_metric),
)

print(f"Collection '{collection_name}' created successfully!")
