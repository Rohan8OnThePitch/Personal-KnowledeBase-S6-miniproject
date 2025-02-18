import uuid
import re
import logging
import nltk
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Download tokenizer for sentence splitting
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Initialize Qdrant client and model
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_collection_if_not_exists(collection_name):
    """Creates a Qdrant collection if it doesn't already exist."""
    try:
        collections_response = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]

        if collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # Ensure this matches embedding dimensions
                    distance=Distance.COSINE
                )
            )
            logging.info(f"Collection '{collection_name}' created.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f" Error creating collection '{collection_name}': {e}")
        raise

def split_text_into_chunks(text, max_chunk_size=256):
    """
    Splits text into smaller, manageable chunks for indexing.
    - Uses newline (`\n`) splitting if available.
    - Falls back to `sent_tokenize()` if necessary.
    - Splits large chunks further into smaller ones (max 256 tokens).
    
    Args:
        text (str): Full document text.
        max_chunk_size (int): Maximum token length per chunk.
    
    Returns:
        list: List of properly split chunks.
    """
    # Try splitting by newlines if present
    if "\n" in text:
        chunks = [s.strip() for s in text.split("\n") if s.strip()]
    else:
        # Otherwise, use sentence tokenization
        chunks = sent_tokenize(text)

    # Ensure chunks are not too large (Break long sentences)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:  
            # Further split large chunks at punctuation
            split_sub_chunks = re.split(r'(?<=[.?!])\s+', chunk)  # Split at sentence-ending punctuation
            final_chunks.extend([s.strip() for s in split_sub_chunks if s.strip()])
        else:
            final_chunks.append(chunk)

    logging.info(f" Split document into {len(final_chunks)} chunks.")
    return final_chunks

def index_document(collection_name, document_id, text, batch_size=50):
    """
    Indexes document text into Qdrant with improved chunking.
    
    Args:
        collection_name (str): Name of the collection.
        document_id (str): ID of the document.
        text (str): Full document text.
        batch_size (int): Number of chunks to process in a single batch.
    
    Returns:
        dict: Status of the indexing operation.
    """
    try:
        create_collection_if_not_exists(collection_name)

        # 🔹 Improved chunking logic
        chunks = split_text_into_chunks(text)

        if not chunks:
            logging.warning(" No valid chunks extracted for indexing.")
            return {"status": "error", "message": "No valid chunks extracted"}

        # 🔹 Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = model.encode(batch_chunks).tolist()

            points = []
            for idx, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                chunk_id = str(uuid.uuid4())

                payload = {
                    "document_id": document_id,
                    "text": chunk,
                    "chunk_index": i + idx,
                    "file_name": document_id  
                }
                points.append({
                    "id": chunk_id,
                    "vector": embedding,
                    "payload": payload
                })

            # Upsert the batch into Qdrant
            qdrant_client.upsert(collection_name=collection_name, points=points)
            logging.info(f" Indexed batch {i // batch_size + 1} ({len(batch_chunks)} chunks).")

        logging.info(f" Successfully indexed {len(chunks)} chunks for document '{document_id}'.")
        return {"status": "success", "chunks": len(chunks)}

    except Exception as e:
        logging.error(f"Error indexing document '{document_id}': {e}")
        return {"status": "error", "message": str(e)}
