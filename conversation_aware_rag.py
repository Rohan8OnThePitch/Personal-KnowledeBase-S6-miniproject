import logging
from typing import Dict, List, Any
from langchain_core.messages import SystemMessage
from memory_manager import (
    retrieve_context_relevant_messages, 
    format_context_messages,
    store_message
)
from qdrant_helper import hybrid_search
from rag import generate_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DOCUMENT_COLLECTION = "document_chunks"

def answer_query_with_conversation_context(
    session_id: str, 
    query: str,
    use_conversation_memory: bool = True,
    conversation_weight: float = 0.3,
    document_weight: float = 0.7,
    top_k_docs: int = 5,
    top_k_conversations: int = 3
):
    """
    Answer a query using both document RAG and conversation history as context.
    
    Args:
        session_id: User's session ID
        query: User query
        use_conversation_memory: Whether to include conversation context
        conversation_weight: Weight to give conversation context (0-1)
        document_weight: Weight to give document context (0-1)
        top_k_docs: Number of document chunks to retrieve
        top_k_conversations: Number of conversation segments to retrieve
        
    Returns:
        Dict containing the answer and source information
    """
    # Store the current query
    store_message(session_id, query, "user")
    
    # Initialize contexts
    document_context = ""
    conversation_context = ""
    all_sources = []
    
    # 1. Get document context from RAG
    logger.info(f"Retrieving document context for query: {query}")
    document_chunks = hybrid_search(
        collection_name=DOCUMENT_COLLECTION,
        query_text=query,
        top_k=top_k_docs
    )
    
    if document_chunks:
        # Remove duplicates while preserving order
        unique_texts = {}
        for chunk in document_chunks:
            unique_texts[chunk["text"]] = chunk
        unique_chunks = list(unique_texts.values())
        
        document_context = " ".join([chunk["text"] for chunk in unique_chunks])
        all_sources.extend([{
            "type": "document",
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "score": chunk.get("score", 0),
            "strategy": chunk.get("strategy", "unknown")
        } for chunk in unique_chunks])
        
        logger.info(f"Retrieved {len(unique_chunks)} unique document chunks")
    
    # 2. Get conversation context if enabled
    if use_conversation_memory:
        logger.info(f"Retrieving conversation context for query: {query}")
        relevant_messages = retrieve_context_relevant_messages(
            session_id=session_id,
            query=query,
            context_window=2,
            top_k=top_k_conversations
        )
        
        if relevant_messages:
            conversation_context = format_context_messages(relevant_messages)
            all_sources.extend([{
                "type": "conversation",
                "text": msg.content,
                "role": msg.type,
                "score": 1.0  # Default score for conversation context
            } for msg in relevant_messages])
            
            logger.info(f"Retrieved {len(relevant_messages)} relevant conversation messages")
    
    # 3. Combine contexts with weighting
    if not document_context and not conversation_context:
        logger.warning("No context found from either documents or conversation")
        answer = "I don't have enough information to answer that question. Please try a different question or upload relevant documents."
    else:
        # Normalize weights if both contexts are available
        if document_context and conversation_context:
            total_weight = document_weight + conversation_weight
            document_weight = document_weight / total_weight
            conversation_weight = conversation_weight / total_weight
        
        # Prepare final context with appropriate weighting
        sections = []
        if document_context:
            sections.append(f"Document Context ({document_weight*100:.0f}% weight):\n{document_context}")
        if conversation_context:
            sections.append(f"Conversation Context ({conversation_weight*100:.0f}% weight):\n{conversation_context}")
        
        combined_context = "\n\n".join(sections)
        
        # 4. Generate answer using combined context
        logger.info("Generating answer using combined context")
        answer = generate_answer(query, combined_context)
    
    # 5. Store the response
    store_message(session_id, answer, "assistant")
    
    # Return the results
    return {
        "answer": answer,
        "sources": all_sources,
        "document_context_used": bool(document_context),
        "conversation_context_used": bool(conversation_context)
    }

def create_context_message(context_sources):
    """
    Create a system message containing context information.
    
    Args:
        context_sources: List of context source dictionaries
        
    Returns:
        SystemMessage with formatted context
    """
    document_sources = [s for s in context_sources if s["type"] == "document"]
    conversation_sources = [s for s in context_sources if s["type"] == "conversation"]
    
    context_parts = []
    
    # Add document context
    if document_sources:
        doc_texts = [f"- {s['text']}" for s in document_sources]
        context_parts.append("Document Context:\n" + "\n".join(doc_texts))
    
    # Add conversation context
    if conversation_sources:
        convo_texts = []
        for s in conversation_sources:
            role = s["role"].capitalize()
            convo_texts.append(f"- {role}: {s['text']}")
        
        context_parts.append("Conversation Context:\n" + "\n".join(convo_texts))
    return SystemMessage(content="Use the following context to answer the question:\n\n" + "\n\n".join(context_parts))
    
import argparse

def main():
    """
    Main function to handle user queries and generate responses using RAG and conversation memory.
    """
    parser = argparse.ArgumentParser(description="Query the RAG system with optional conversation memory.")
    parser.add_argument("session_id", type=str, help="User session ID")
    parser.add_argument("query", type=str, help="User query")
    parser.add_argument("--no_memory", action="store_true", help="Disable conversation memory")
    parser.add_argument("--conv_weight", type=float, default=0.3, help="Weight for conversation context (0-1)")
    parser.add_argument("--doc_weight", type=float, default=0.7, help="Weight for document context (0-1)")
    parser.add_argument("--top_k_docs", type=int, default=5, help="Number of document chunks to retrieve")
    parser.add_argument("--top_k_conversations", type=int, default=3, help="Number of conversation messages to retrieve")
    
    args = parser.parse_args()
    
    response = answer_query_with_conversation_context(
        session_id=args.session_id,
        query=args.query,
        use_conversation_memory=not args.no_memory,
        conversation_weight=args.conv_weight,
        document_weight=args.doc_weight,
        top_k_docs=args.top_k_docs,
        top_k_conversations=args.top_k_conversations
    )
    
    print("\nAnswer:")
    print(response["answer"])
    
    print("\nSources:")
    for source in response["sources"]:
        print(f"- [{source['type'].capitalize()}] {source['text']}")

if __name__ == "__main__":
    main()

