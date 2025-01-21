from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance
import uuid

# Initialize Qdrant client and model
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_collection_if_not_exists(collection_name):
    try:
        # Check if the collection exists
        collections_response = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]

        if collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # Ensure this matches your embedding dimensions
                    distance=Distance.COSINE  # Use cosine distance for similarity
                )
            )
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"Error creating collection: {e}")

def index_document(collection_name, document_id, text):
    try:
        # Ensure the collection exists
        create_collection_if_not_exists(collection_name)

        # Generate the embedding
        embedding = model.encode(text).tolist()  # Convert to list for compatibility

        # Create the payload
        payload = {"text": text}

        # Validate or generate a valid document_id
        if not isinstance(document_id, (int, str)) or not str(document_id).isnumeric():
            document_id = str(uuid.uuid4())  # Generate a UUID if the ID is not valid

        print(f"Using document ID: {document_id}")

        # Perform the upsert operation
        response = qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": document_id,  # Use a valid ID
                    "vector": embedding,  # The embedding vector
                    "payload": payload    # Any metadata
                }
            ]
        )

        print(f"Document {document_id} indexed successfully.")

        # Ensure that the response from Qdrant is serializable
        response_data = {
            "status": "success",
            "document_id": document_id,
            "response": response.to_dict() if hasattr(response, 'to_dict') else str(response)
        }

        return response_data

    except Exception as e:
        print(f"Error indexing document {document_id}: {str(e)}")
        return {"status": "error", "message": str(e)}
