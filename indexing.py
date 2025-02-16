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
        logging.error(f" Error creating collection '{collection_name}': {e}")
        raise

def index_document(collection_name, document_id, text, batch_size=100):
    """
    Indexes document text into Qdrant.
    
    Args:
        collection_name (str): Name of the collection.
        document_id (str): ID of the document.
        text (str): Full document text.
        batch_size (int): Number of chunks to process in a single batch.
    
    Returns:
        dict: Status of the indexing operation.
    """
    try:
        # Ensure the collection exists
        create_collection_if_not_exists(collection_name)

        # Split text into sentences (Better Handling)
        if "\n" in text:  
            # If there are line breaks, split by them
            chunks = [s.strip() for s in text.split("\n") if s.strip()]
        else:
            # Otherwise, use sentence tokenizer
            import nltk
            nltk.download("punkt")
            from nltk.tokenize import sent_tokenize
            chunks = sent_tokenize(text)

        # Debugging: Print extracted chunks
        logging.info(f"🔍 Extracted {len(chunks)} chunks from document.")
        if len(chunks) == 1:
            logging.warning(f"Only 1 chunk extracted! Text may not be splitting correctly.")

        # Validate input
        if not chunks:
            logging.warning(" No chunks provided for indexing.")
            return {"status": "error", "message": "No chunks extracted"}

        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]

            # Generate embeddings for the batch
            embeddings = model.encode(batch_chunks).tolist()

            # Prepare points for Qdrant
            points = []
            for idx, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                chunk_id = str(uuid.uuid4())  # Ensure unique ID

                payload = {
                    "document_id": document_id,
                    "text": chunk,
                    "chunk_index": i + idx,
                    "file_name": document_id  # Include additional metadata
                }
                points.append({
                    "id": chunk_id,  # Use UUID for unique IDs
                    "vector": embedding,
                    "payload": payload
                })

            # Upsert the batch into Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )

            logging.info(f"Indexed batch {i // batch_size + 1} ({len(batch_chunks)} chunks).")

            # Debugging: Print first 100 chars of each chunk
            for point in points:
                logging.info(f" Indexed Chunk: {point['payload']['text'][:100]}...")

        logging.info(f"Successfully indexed {len(chunks)} chunks for document '{document_id}'.")
        return {"status": "success", "chunks": len(chunks)}

    except Exception as e:
        logging.error(f"Error indexing document '{document_id}': {e}")
        return {"status": "error", "message": str(e)}

