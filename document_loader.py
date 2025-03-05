from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Any, Tuple
import uuid
import logging
from fuzzywuzzy import fuzz, process

def load_and_chunk_documents_with_multiple_strategies(
    file_path: str, 
    chunk_strategies: List[Dict[str, int]] = None
) -> Dict[str, List]:
    """
    Load a document and chunk it using multiple strategies.
    
    Args:
        file_path: Path to the document
        chunk_strategies: List of dictionaries containing chunk_size and chunk_overlap
            
    Returns:
        Dictionary with strategy_id as key and list of chunks as value
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found. Please check the file path.")
    
    # Default strategies if none provided
    if not chunk_strategies:
        chunk_strategies = [
            {"id": "small", "chunk_size": 500, "chunk_overlap": 50},
            {"id": "medium", "chunk_size": 1000, "chunk_overlap": 100},
            {"id": "large", "chunk_size": 2000, "chunk_overlap": 200},
        ]
    
    # Determine loader based on file extension
    if file_path.endswith('.pdf'):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path)
    
    # Load the document
    documents = loader.load()
    
    # Apply each chunking strategy
    all_chunks = {}
    for strategy in chunk_strategies:
        strategy_id = strategy["id"]
        chunk_size = strategy["chunk_size"]
        chunk_overlap = strategy["chunk_overlap"]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        all_chunks[strategy_id] = chunks
        
        logging.info(f"Strategy '{strategy_id}' generated {len(chunks)} chunks")
    
    return all_chunks

def create_rolling_window_chunks(
    documents, 
    window_size: int = 1000, 
    step_size: int = 200
) -> List:
    """
    Create chunks using a rolling window approach for better context preservation.
    
    Args:
        documents: Loaded documents
        window_size: Size of the rolling window
        step_size: How much to move the window for each new chunk
        
    Returns:
        List of chunks
    """
    rolling_chunks = []
    
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata
        
        # For very short documents, just use the document as is
        if len(text) <= window_size:
            rolling_chunks.append(doc)
            continue
        
        # Create rolling window chunks
        for i in range(0, len(text) - window_size + 1, step_size):
            chunk_text = text[i:i + window_size]
            
            # Create new metadata with position info
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_start": i,
                "chunk_end": i + window_size,
                "chunk_type": "rolling_window"
            })
            
            from langchain.schema import Document
            chunk = Document(page_content=chunk_text, metadata=chunk_metadata)
            rolling_chunks.append(chunk)
    
    logging.info(f"Rolling window approach generated {len(rolling_chunks)} chunks")
    return rolling_chunks