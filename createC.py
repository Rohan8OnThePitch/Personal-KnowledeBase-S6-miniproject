from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Qdrant client setup
qdrant_client = QdrantClient(url="http://localhost:6333")

# Embedding model for chat memory
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Collection names
DOCUMENT_COLLECTION = "document_chunks"
MEMORY_COLLECTION = "chat_memory"

# Create a separate collection for chat memory
qdrant_client.create_collection(
    collection_name=MEMORY_COLLECTION,
    vectors_config=VectorParams(
        size=embedding_model.get_sentence_embedding_dimension(),
        distance=Distance.COSINE
    )
)
