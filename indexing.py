import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
import logging

# Initialize Qdrant client and model
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_collection_if_not_exists(collection_name):
    """
    Creates a Qdrant collection if it doesn't already exist.
    
    Args:
        collection_name (str): Name of the collection.
    """
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
            logging.info(f"Collection '{collection_name}' created.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f"Error creating collection '{collection_name}': {e}")
        raise

def index_document(collection_name, document_id, chunks, batch_size=100):
    """
    Indexes document chunks into Qdrant.
    
    Args:
        collection_name (str): Name of the collection.
        document_id (str): ID of the document.
        chunks (list): List of text chunks to index.
        batch_size (int): Number of chunks to process in a single batch.
    
    Returns:
        dict: Status of the indexing operation.
    """
    try:
        # Validate input
        if not chunks:
            logging.warning("No chunks provided for indexing.")
            return {"status": "error", "message": "No chunks provided"}

        # Ensure the collection exists
        create_collection_if_not_exists(collection_name)

        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]

            # Generate embeddings for the batch
            embeddings = model.encode(batch_chunks).tolist()

            # Prepare points for Qdrant
            points = []
            for idx, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                # Use a UUID for the point ID
                chunk_id = str(uuid.uuid4())

                payload = {
                    "document_id": document_id,
                    "text": chunk,
                    "chunk_index": i + idx,
                    "file_name": document_id  # Include additional metadata
                }
                points.append({
                    "id": chunk_id,  # Use a valid UUID as the point ID
                    "vector": embedding,
                    "payload": payload
                })

            # Upsert the batch into Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logging.info(f"Indexed batch {i // batch_size + 1} with {len(batch_chunks)} chunks.")

        logging.info(f"Successfully indexed {len(chunks)} chunks for document '{document_id}'.")
        return {"status": "success", "chunks": len(chunks)}

    except Exception as e:
        logging.error(f"Error indexing document '{document_id}': {e}")
        return {"status": "error", "message": str(e)}
