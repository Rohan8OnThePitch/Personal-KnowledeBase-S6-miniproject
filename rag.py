import os
import logging
import torch
import uuid
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import qdrant_helper as qdrant_helper
from document_loader import load_and_chunk_documents_with_multiple_strategies, create_rolling_window_chunks
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable is not set")

# Load Sentence Transformer for Query Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Specify the model name
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    "question-answering", 
    model=model_name, 
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    return_tensors='pt'
)

def generate_answer(query, context, max_length=256, temperature=1.0):
    """
    Generate an answer for a query based on the provided context using the LLM.
    
    Args:
        query (str): The user's question
        context (str): The context text to use for answering
        
    Returns:
        str: The generated answer
    """
    logger.info("Generating answer using TinyBERT")
    if not context.strip():
        logger.warning("Empty context provided to generate_answer")
        return "No information found in the database."

    try:
        result = question_answerer(
            question=query, 
            context=context,
            max_length=max_length,
            temperature=temperature
        )
        return result["answer"]
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

def process_document(file_path, document_id=None):
    """
    Process a document with multiple chunking strategies and index it in Qdrant.
    
    Args:
        file_path: Path to the document
        document_id: Optional document ID (will generate UUID if not provided)
        
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"Processing document: {file_path}")
        
        if not document_id:
            document_id = str(uuid.uuid4())
        
        # Define chunking strategies
        chunk_strategies = [
            {"id": "small", "chunk_size": 512, "chunk_overlap": 128},
            {"id": "medium", "chunk_size": 1024, "chunk_overlap": 256},
            {"id": "large", "chunk_size": 2048, "chunk_overlap": 512},
        ]
        
        # Load document with different chunking strategies
        chunking_results = load_and_chunk_documents_with_multiple_strategies(
            file_path=file_path,
            chunk_strategies=chunk_strategies
        )
        
        # Get original document to create rolling window chunks
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        
        # Create rolling window chunks
        rolling_window_chunks = create_rolling_window_chunks(
            documents=documents,
            window_size=1000,
            step_size=200
        )
        
        # Index document with all strategies
        indexing_result = qdrant_helper.index_document_with_strategies(
            collection_name=qdrant_helper.COLLECTION_NAME,
            document_id=document_id,
            chunking_strategies=chunking_results,
            rolling_window_chunks=rolling_window_chunks
        )
        
        return {
            "document_id": document_id,
            "strategies": list(chunking_results.keys()) + ["rolling_window"],
            "chunks_count": {
                **{k: len(v) for k, v in chunking_results.items()},
                "rolling_window": len(rolling_window_chunks)
            },
            "indexing_result": indexing_result
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return {"status": "error", "message": str(e)}

def answer_query_enhanced(user_query, search_type="hybrid", top_k=5):
    """
    Enhanced function to answer a user query using RAG with multiple strategies.
    
    Args:
        user_query (str): User's question
        search_type (str): Type of search - "vector", "fuzzy", or "hybrid"
        top_k (int): Maximum number of chunks to retrieve
        
    Returns:
        dict: Dictionary containing the answer and relevant chunks
    """
    try:
        logger.info(f"🔍 Processing query: {user_query}")
        
        # Get relevant contexts based on search type
        if search_type == "vector":
            context_items = qdrant_helper.query_qdrant_multi_strategy(
                collection_name=qdrant_helper.COLLECTION_NAME,
                query_text=user_query,
                top_k=top_k
            )
        elif search_type == "fuzzy":
            context_items = qdrant_helper.fuzzy_search(
                collection_name=qdrant_helper.COLLECTION_NAME,
                query_text=user_query,
                top_k=top_k
            )
        else:  # hybrid
            context_items = qdrant_helper.hybrid_search(
                collection_name=qdrant_helper.COLLECTION_NAME,
                query_text=user_query,
                vector_weight=0.7,
                fuzzy_weight=0.3,
                top_k=top_k
            )
        
        logger.debug(f"Retrieved {len(context_items)} contexts using {search_type} search")

        if not context_items:
            logger.warning("No contexts found")
            return {
                "answer": "No relevant information found in the database.",
                "chunks": []
            }

        # Extract text from the retrieved dictionaries
        contexts = [item["text"] for item in context_items]
        
        # Combine text from all results
        combined_context = " ".join(contexts)
        if not combined_context.strip():
            logger.warning("No valid context after combining results")
            return {"answer": "No relevant information found.", "chunks": []}

        # Generate answer from combined context
        generated_answer = generate_answer(user_query, combined_context)
        logger.info("Answer generated successfully")

        return {
            "answer": generated_answer, 
            "chunks": context_items,
            "search_type": search_type,
            "context_length": len(combined_context)
        }

    except Exception as e:
        logger.error(f"Error during query: {str(e)}", exc_info=True)
        return {"answer": f"An error occurred: {str(e)}", "chunks": []}

def compare_search_strategies(user_query, top_k=5):
    """
    Compare different search strategies for the same query.
    
    Args:
        user_query (str): User's question
        top_k (int): Maximum number of chunks to retrieve per strategy
        
    Returns:
        dict: Dictionary containing answers from different strategies
    """
    strategies = ["vector", "fuzzy", "hybrid"]
    results = {}
    
    for strategy in strategies:
        logger.info(f"Querying with {strategy} strategy")
        result = answer_query_enhanced(user_query, search_type=strategy, top_k=top_k)
        results[strategy] = result
    
    # Determine best strategy based on context relevance
    # This is a simple heuristic - in production you might want more sophisticated metrics
    strategy_scores = {}
    for strategy, result in results.items():
        # Skip strategies with errors or no chunks
        if "chunks" not in result or not result["chunks"]:
            strategy_scores[strategy] = 0
            continue
        
        # Calculate average score of chunks
        scores = [chunk.get("score", 0) for chunk in result["chunks"]]
        avg_score = sum(scores) / len(scores) if scores else 0
        strategy_scores[strategy] = avg_score
    
    # Determine best strategy
    best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    return {
        "best_strategy": best_strategy,
        "best_answer": results[best_strategy]["answer"],
        "strategy_results": results,
        "strategy_scores": strategy_scores
    }


if __name__ == "__main__":
    # Example processing a document with multiple chunking strategies
    """file_path = "example.pdf"
    if os.path.exists(file_path):
        result = process_document(file_path)
        print(f"Document processing result: {result}")"""
    
    # Example query with strategy comparison
    query = "What is the Introduction?"
    comparison = compare_search_strategies(query)
    print(f"Best strategy: {comparison['best_strategy']}")
    print(f"Answer: {comparison['best_answer']}")