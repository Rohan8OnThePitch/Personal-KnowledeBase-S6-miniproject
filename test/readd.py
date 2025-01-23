from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize Qdrant client and model
client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example documents
documents = [
    {"id": 1, "text": "France is a country in Europe. Its capital is Paris."},
    {"id": 2, "text": "Paris is the largest city in France and serves as its capital."},
]

# Add documents to Qdrant
for doc in documents:
    vector = model.encode(doc["text"]).tolist()
    client.upsert(
        collection_name="documents",
        points=[
            {
                "id": doc["id"],
                "vector": vector,
                "payload": {"text": doc["text"]}
            }
        ]
    )

print("Documents added to Qdrant.")