from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance
import uuid
import os
import nltk  # Import nltk for sentence tokenization
from nltk.tokenize import PunktSentenceTokenizer

# Initialize Qdrant client and model
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Consider moving to a global variable


def create_collection_if_not_exists(collection_name):
    """Creates a Qdrant collection if it doesn't already exist."""
    try:
        collections_response = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]

        if collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # Matches embedding dimensions
                    distance=Distance.COSINE  # Cosine distance for similarity
                )
            )
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"Error creating collection: {e}")
        raise  # Re-raise the exception for handling upstream

def index_document(collection_name, document_id, text):
    """Indexes a document by embedding the text and storing it in Qdrant."""
    try:
        print(f"Indexing document: {document_id}") # ADD THIS LINE
        # Ensure the collection exists
        create_collection_if_not_exists(collection_name)

        # Split the text into sentences
        sentences = nltk.sent_tokenize(text)  # Use nltk for sentence tokenization
        print(f"Number of sentences: {len(sentences)}") # ADD THIS LINE
        for sentence in sentences:
            # Generate the embedding for each sentence
            embedding = model.encode(sentence).tolist()

            # Create the payload
            payload = {"text": sentence}

            # Generate a UUID for each sentence
            point_id = str(uuid.uuid4())

            # Perform the upsert operation
            response = qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    {
                        "id": point_id,  # Use a unique ID for each sentence
                        "vector": embedding,  # The embedding vector
                        "payload": payload    # Any metadata
                    }
                ],
                wait=True  # Add wait=True to ensure data is persisted
            )

            print(f"Indexed sentence {point_id} successfully.")

        return {
            "status": "success",
            "document_id": document_id, #use the document ID here
            "message": f"Indexed document {document_id} with {len(sentences)} sentences."
            # you may want to return something about the number of sentences indexed
        }

    except Exception as e:
        print(f"Error indexing document {document_id}: {str(e)}")
        return {"status": "error", "message": str(e)}
