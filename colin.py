from qdrant_client import QdrantClient

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# List all collections
collections = qdrant_client.get_collections()
print("Collections:", collections)

# Check points in the collection
collection_name = "document_chunks"  # Replace with your collection name
points = qdrant_client.scroll(collection_name=collection_name, limit=10)
print("Points in collection:", points)

retrieved_points = qdrant_client.retrieve(collection_name=collection_name, ids=[1, 2, 3])
for point in retrieved_points:
    if point.vector is None:
        print(f"⚠️ Warning: Vector for ID {point.id} is missing!")
    else:
        print(f"✅ ID: {point.id}, Vector: {point.vector[:5]}... (truncated)")
