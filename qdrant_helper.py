import uuid
import logging
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from fuzzywuzzy import fuzz, process

# Initialize Qdrant client and model
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up logging
logging.basicConfig(level=logging.INFO)

COLLECTION_NAME = "document_chunks"

def create_collection_if_not_exists(collection_name):
    """
    Creates a Qdrant collection if it doesn't already exist.
    """
    try:
        collections_response = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]

        if collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # Embedding dimension of all-MiniLM-L6-v2
                    distance=Distance.COSINE  # Cosine distance for similarity search
                )
            )
            logging.info(f"Collection '{collection_name}' created.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f"Error creating collection '{collection_name}': {e}")
        raise

def index_document_with_strategies(
    collection_name: str, 
    document_id: str, 
    chunking_strategies: Dict[str, List], 
    rolling_window_chunks: List = None,
    batch_size: int = 50
) -> Dict:
    """
    Index document chunks from multiple chunking strategies.
    
    Args:
        collection_name: Qdrant collection name
        document_id: Unique identifier for the document
        chunking_strategies: Dictionary of chunking strategies with chunks
        rolling_window_chunks: Optional list of rolling window chunks
        batch_size: Number of chunks to process in each batch
        
    Returns:
        Dictionary with indexing results
    """
    try:
        logging.info(f"Starting multi-strategy indexing for document: {document_id}")
        create_collection_if_not_exists(collection_name)
        
        results = {}
        total_chunks = 0
        
        # Process each chunking strategy
        for strategy_id, chunks in chunking_strategies.items():
            if not chunks:
                logging.warning(f"No chunks provided for strategy '{strategy_id}'")
                results[strategy_id] = {"status": "error", "message": "No chunks found"}
                continue
            
            logging.info(f"Indexing {len(chunks)} chunks for strategy '{strategy_id}'")
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logging.info(f"Processing batch {i//batch_size + 1}, size: {len(batch)}")
                
                # Get embeddings only for text content in chunks
                embeddings = model.encode([chunk.page_content for chunk in batch], convert_to_tensor=False).tolist()
                
                points = []
                for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    chunk_id = str(uuid.uuid4())
                    payload = {
                        "document_id": document_id,
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                        "chunk_index": i + idx,
                        "strategy": strategy_id
                    }
                    points.append(
                        PointStruct(
                            id=chunk_id,
                            vector=embedding,
                            payload=payload
                        )
                    )
                
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
            
            results[strategy_id] = {"status": "success", "chunks": len(chunks)}
            total_chunks += len(chunks)
        
        # Process rolling window chunks if provided
        if rolling_window_chunks:
            logging.info(f"Indexing {len(rolling_window_chunks)} rolling window chunks")
            
            for i in range(0, len(rolling_window_chunks), batch_size):
                batch = rolling_window_chunks[i:i + batch_size]
                
                # Get embeddings only for text content in chunks
                embeddings = model.encode([chunk.page_content for chunk in batch], convert_to_tensor=False).tolist()
                
                points = []
                for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    chunk_id = str(uuid.uuid4())
                    payload = {
                        "document_id": document_id,
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                        "chunk_index": i + idx,
                        "strategy": "rolling_window"
                    }
                    points.append(
                        PointStruct(
                            id=chunk_id,
                            vector=embedding,
                            payload=payload
                        )
                    )
                
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
            
            results["rolling_window"] = {"status": "success", "chunks": len(rolling_window_chunks)}
            total_chunks += len(rolling_window_chunks)
        
        logging.info(f"Total chunks indexed: {total_chunks}")
        
        collection_info = qdrant_client.get_collection(collection_name)
        logging.info(f"Collection now has {collection_info.points_count} points total")
        
        return {"status": "success", "strategies": results, "total_chunks": total_chunks}
    
    except Exception as e:
        logging.error(f"Error indexing document '{document_id}': {e}")
        return {"status": "error", "message": str(e)}

def query_qdrant_multi_strategy(
    collection_name: str, 
    query_text: str, 
    strategies: List[str] = None,
    top_k: int = 5
) -> List[Dict]:
    """
    Query the Qdrant collection across multiple chunking strategies and return top results.
    
    Args:
        collection_name: Qdrant collection name
        query_text: Query text
        strategies: List of strategies to query (None = all strategies)
        top_k: Number of results to retrieve per strategy
        
    Returns:
        Combined and sorted list of results
    """
    try:
        query_vector = model.encode([query_text], convert_to_tensor=False)[0].tolist()
        all_results = []
        
        # If no specific strategies provided, query all strategies
        if not strategies:
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k * 3,  # Get more results to account for multiple strategies
                score_threshold=0.3
            )
            
            for hit in search_results:
                all_results.append({
                    "score": hit.score,
                    "text": hit.payload['text'],
                    "metadata": hit.payload.get("metadata", {}),
                    "strategy": hit.payload.get("strategy", "unknown")
                })
        
        # Otherwise, query each specified strategy
        else:
            for strategy in strategies:
                filter_by_strategy = Filter(
                    must=[
                        FieldCondition(
                            key="strategy",
                            match=MatchValue(value=strategy)
                        )
                    ]
                )
                
                strategy_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=0.3,
                    filter=filter_by_strategy
                )
                
                for hit in strategy_results:
                    all_results.append({
                        "score": hit.score,
                        "text": hit.payload['text'],
                        "metadata": hit.payload.get("metadata", {}),
                        "strategy": hit.payload.get("strategy", strategy)
                    })
        
        # Sort by score and take top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:top_k]
        
        logging.info(f"Multi-strategy query returned {len(top_results)} results from {len(all_results)} candidates")
        return top_results
    
    except Exception as e:
        logging.error(f"Error querying collection '{collection_name}': {e}")
        return []

def fuzzy_search(collection_name: str, query_text: str, min_score: int = 70, top_k: int = 5) -> List[Dict]:
    """
    Perform fuzzy text search on documents using fuzzywuzzy.
    
    Args:
        collection_name: Qdrant collection name
        query_text: Query text
        min_score: Minimum similarity score (0-100)
        top_k: Maximum number of results to return
        
    Returns:
        List of matching results with scores
    """
    try:
        # Get all documents from the collection
        # In a production environment, this would need pagination for large collections
        scroll_results = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,  # Consider implementing proper pagination in production
            with_payload=True
        )
        
        all_documents = scroll_results[0]
        if not all_documents:
            return []
        
        # Extract text content for fuzzy matching
        documents_text = [(doc.id, doc.payload["text"]) for doc in all_documents if "text" in doc.payload]
        
        # Perform fuzzy matching
        results = []
        for doc_id, text in documents_text:
            # Calculate similarity ratio
            similarity = fuzz.token_set_ratio(query_text.lower(), text.lower())
            
            if similarity >= min_score:
                # Find the original document to get the full data
                doc = next((d for d in all_documents if d.id == doc_id), None)
                if doc:
                    results.append({
                        "score": similarity / 100.0,  # Normalize to 0-1 scale to match vector search
                        "text": doc.payload["text"],
                        "metadata": doc.payload.get("metadata", {}),
                        "strategy": doc.payload.get("strategy", "unknown"),
                        "search_type": "fuzzy"
                    })
        
        # Sort by score and take top results
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]
        
        logging.info(f"Fuzzy search returned {len(top_results)} results")
        return top_results
    
    except Exception as e:
        logging.error(f"Error performing fuzzy search: {e}")
        return []

def hybrid_search(
    collection_name: str, 
    query_text: str, 
    strategies: List[str] = None,
    vector_weight: float = 0.7, 
    fuzzy_weight: float = 0.3,
    top_k: int = 5
) -> List[Dict]:
    """
    Perform hybrid search combining vector search and fuzzy text search.
    
    Args:
        collection_name: Qdrant collection name
        query_text: Query text
        strategies: List of strategies to query (None = all strategies)
        vector_weight: Weight for vector search results (0-1)
        fuzzy_weight: Weight for fuzzy search results (0-1)
        top_k: Number of results to return
        
    Returns:
        List of combined and ranked results
    """
    try:
        # Validate weights
        if vector_weight + fuzzy_weight != 1.0:
            logging.warning("Weights don't sum to 1.0, normalizing...")
            total = vector_weight + fuzzy_weight
            vector_weight /= total
            fuzzy_weight /= total
        
        # Get results from both search methods
        vector_results = query_qdrant_multi_strategy(collection_name, query_text, strategies, top_k * 2)
        fuzzy_results = fuzzy_search(collection_name, query_text, min_score=70, top_k=top_k * 2)
        
        # Combine results and assign weighted scores
        combined_results = {}
        
        # Process vector search results
        for result in vector_results:
            doc_id = result.get("id", str(uuid.uuid4()))  # Use a unique ID if not available
            combined_results[doc_id] = {
                "vector_score": result["score"],
                "fuzzy_score": 0,
                "text": result["text"],
                "metadata": result["metadata"],
                "strategy": result["strategy"]
            }
        
        # Process fuzzy search results
        for result in fuzzy_results:
            doc_id = result.get("id", str(uuid.uuid4()))
            if doc_id in combined_results:
                # Document already in results, update fuzzy score
                combined_results[doc_id]["fuzzy_score"] = result["score"]
            else:
                # New document
                combined_results[doc_id] = {
                    "vector_score": 0,
                    "fuzzy_score": result["score"],
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "strategy": result["strategy"]
                }
        
        # Calculate combined scores
        results_list = []
        for doc_id, result in combined_results.items():
            combined_score = (result["vector_score"] * vector_weight) + (result["fuzzy_score"] * fuzzy_weight)
            results_list.append({
                "id": doc_id,
                "score": combined_score,
                "vector_score": result["vector_score"],
                "fuzzy_score": result["fuzzy_score"],
                "text": result["text"],
                "metadata": result["metadata"],
                "strategy": result["strategy"]
            })
        
        # Sort by combined score and take top results
        results_list.sort(key=lambda x: x["score"], reverse=True)
        top_results = results_list[:top_k]
        
        logging.info(f"Hybrid search returned {len(top_results)} results")
        return top_results
    
    except Exception as e:
        logging.error(f"Error performing hybrid search: {e}")
        return []