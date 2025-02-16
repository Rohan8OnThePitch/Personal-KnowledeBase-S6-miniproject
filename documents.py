import docx
import fitz  # PyMuPDF

def process_docx(file_path):
    """Extracts text from a .docx file."""
    try:
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        text = '\n'.join(full_text)
        
        print(f"Extracted {len(full_text)} paragraphs from DOCX")  # Debugging
        print(f"Extracted Text: {text[:500]}...")  # Print first 500 chars
        
        return {'text': text.strip()}
    except Exception as e:
        return {'error': str(e)}


def process_pdf(file_path):
    """Extracts text from a .pdf file."""
    try:
        pdf = fitz.open(file_path)
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
        return {'text': text.strip()}  # Return as a dictionary
    except Exception as e:
        return {'error': str(e)}


def process_txt(file_path):
    """Extracts text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return {'text': text.strip()}  # Return as a dictionary
    except Exception as e:
        return {'error': str(e)}
