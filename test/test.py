from qdrant_client import QdrantClient
from qdrant_client.http.models import OptimizersConfig

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Update optimizer configuration
client.update_collection(
    collection_name="documents",
    optimizer_config=OptimizersConfig(
        vacuum_min_vector_number=1,  # Optimize more aggressively
    ),
)

print("Optimizer configuration updated.")
