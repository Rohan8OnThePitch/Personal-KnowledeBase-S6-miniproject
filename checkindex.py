from qdrant_client import QdrantClient

# Connect to local Qdrant instance
qdrant_client = QdrantClient(host="localhost", port=6333)

# Collection name
collection_name = "documents"

try:
    # Check if collection exists
    collections = qdrant_client.get_collections()
    existing_collections = [col.name for col in collections.collections]

    if collection_name not in existing_collections:
        print(f"Collection '{collection_name}' does not exist.")
    else:
        # Get the count of indexed points
        count = qdrant_client.count(collection_name)
        print(f"Indexed documents in '{collection_name}': {count.count}")

except Exception as e:
    print(f"Error checking collection '{collection_name}': {e}")
