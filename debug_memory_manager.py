import logging
from memory_manager import store_message, get_all_session_messages, retrieve_context_relevant_messages, format_context_messages
from rag import process_document, answer_query_enhanced
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    session_id = "debug_session"
    
    # Sample messages for testing
    messages = [
        ("Hello, how can I assist you today?", "assistant"),
        ("Can you tell me about machine learning?", "user"),
        ("Machine learning is a subset of AI that enables systems to learn from data.", "assistant"),
        ("What are some common algorithms?", "user"),
        ("Common ML algorithms include decision trees, SVMs, and neural networks.", "assistant")
    ]
    
    # Store messages
    logger.info("Storing messages in memory")
    for content, role in messages:
        store_message(session_id, content, role)
    
    # Retrieve all messages
    logger.info("Retrieving all stored messages")
    all_messages = get_all_session_messages(session_id)
    print("\n====================== ALL MESSAGES ======================")
    print(format_context_messages(all_messages))
    
    # User-provided file and query
    file_path = "uploads/grop.docx"  # Replace with actual file path
    user_query = "explain the abstract?"  # Replace with actual query
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' not found. Please check the file path.")
        return
    
    # Process document
    logger.info("Processing document")
    processing_result = process_document(file_path)
    strategies_count = len(processing_result.get('strategies', []))
    chunks_count = sum(processing_result.get('chunks_count', {}).values())
    
    logger.info(f"Document processed with {strategies_count} strategies")
    logger.info(f"Total chunks created: {chunks_count}")
    
    # Display contents of each chunk
    if 'chunks' in processing_result:
        print("\n================== DOCUMENT CHUNKS ==================")
        for i, chunk in enumerate(processing_result['chunks'], 1):
            print(f"\nChunk {i}:\n{chunk}")
    
    # Retrieve relevant messages for a query
    logger.info("Retrieving relevant messages based on query")
    relevant_messages = retrieve_context_relevant_messages(session_id, user_query, top_k=3)
    print("\n====================== RELEVANT MESSAGES ======================")
    print(format_context_messages(relevant_messages))
    
    # Answer query from document
    logger.info("Generating answer from document")
    answer_result = answer_query_enhanced(user_query)
    print("\n====================== QUERY RESULT ======================")
    print(f"Query: {user_query}\n")
    print(f"Generated Answer: {answer_result['answer']}\n")
    print(f"Retrieved {len(answer_result.get('chunks', []))} chunks\n")
    
    if answer_result.get("chunks"):
        print("================== RETRIEVED CHUNKS FROM DOCUMENT ==================")
        for i, chunk in enumerate(answer_result["chunks"], 1):
            print(f"\nChunk {i}:\n{chunk['text']}\n")

if __name__ == "__main__":
    main()

