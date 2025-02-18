from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import os

# Connect to the local Qdrant instance (using environment variables)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Define collection name and vector parameters (using environment variables)
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "documents")
VECTOR_SIZE = int(os.environ.get("QDRANT_VECTOR_SIZE", 384))  # Adjust based on your embeddings

#Map the string to the Distance Enum.
DISTANCE_METRIC_STRING = os.environ.get("QDRANT_DISTANCE_METRIC", "Cosine").lower()
DISTANCE_METRIC = Distance.COSINE
if(DISTANCE_METRIC_STRING == "euclid"):
    DISTANCE_METRIC = Distance.EUCLID
elif(DISTANCE_METRIC_STRING == "dot"):
    DISTANCE_METRIC = Distance.DOT

# Create the collection
try:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC),
    )
    print(f"Collection '{COLLECTION_NAME}' created/recreated successfully!")

except Exception as e:
    print(f"Error creating/recreating collection '{COLLECTION_NAME}': {e}")
