import docx
import fitz  # PyMuPDF

def process_docx(file_path):
    """Extracts text from a .docx file."""
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        text = '\n'.join(full_text)
        print(f"Extracted text from docx: {text}")  # Add this line
        return {'text': text.strip()}  # Return as a dictionary
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
