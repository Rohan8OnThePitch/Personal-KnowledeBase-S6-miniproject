from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from docx import Document
from PyPDF2 import PdfReader  # Example for PDF files
import sys
app = Flask(__name__)

# Set the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx', 'pdf','txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dictionary mapping file types to processing functions
FILE_PROCESSORS = {
    'docx': 'process_docx_file',
    'pdf': 'process_pdf_file',
    'txt':'process_txt_file'
}

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_page():
    return render_template('index.html')  # Serves the HTML form

# Route to upload and process a file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Identify the processing function from the dictionary
        processor_function_name = FILE_PROCESSORS.get(file_extension)
        if processor_function_name:
            processor_function = globals().get(processor_function_name)
            if callable(processor_function):
                try:
                    extracted_text = processor_function(file_path)
                    return jsonify({"text": extracted_text})
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            else:
                return jsonify({"error": f"No processor function found for {file_extension}"}), 500

    return jsonify({"error": "Invalid file type. Only .docx and .pdf allowed."}), 400

# Function to process DOCX files
def process_docx_file(file_path):
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text.strip()

# Function to process PDF files
def process_pdf_file(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()
def process_txt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = "\n".join([line.strip() for line in file])
        return text.strip()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error: {e}" 

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
