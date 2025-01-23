from docx import Document
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_docx(file_path, chunk_size=500, chunk_overlap=50):
    """
    Processes a .docx file and returns chunks of text.
    
    Args:
        file_path (str): Path to the .docx file.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    
    Returns:
        list: A list of text chunks.
    """
    try:
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text.strip())
        return chunks
    except Exception as e:
        print(f"Error processing .docx file: {e}")
        return []

def process_pdf(file_path, chunk_size=500, chunk_overlap=50):
    """
    Processes a .pdf file and returns chunks of text.
    
    Args:
        file_path (str): Path to the .pdf file.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    
    Returns:
        list: A list of text chunks.
    """
    try:
        text = ""
        pdf = fitz.open(file_path)
        for page in pdf:
            text += page.get_text()
        pdf.close()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text.strip())
        return chunks
    except Exception as e:
        print(f"Error processing .pdf file: {e}")
        return []

def process_txt(file_path, chunk_size=500, chunk_overlap=50):
    """
    Processes a .txt file and returns chunks of text.
    
    Args:
        file_path (str): Path to the .txt file.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    
    Returns:
        list: A list of text chunks.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text.strip())
        return chunks
    except Exception as e:
        print(f"Error processing .txt file: {e}")
        return []