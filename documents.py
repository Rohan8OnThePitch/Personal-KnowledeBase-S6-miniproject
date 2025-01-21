from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from docx import Document
from PyPDF2 import PdfReader  # Example for PDF files
import fitz  # PyMuPDF
import io
import base64
from PIL import Image
import preprocess

app = Flask(__name__)

# Set the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx', 'pdf','txt'}
ALLOWED_EXTENSIONS = {'docx', 'pdf', 'txt'}
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
                    preprocessed_text = preprocess.preprocess_text(extracted_text)
                    return jsonify({"text": preprocessed_text})
                    extracted_data = processor_function(file_path)
                    preprocessed_text = preprocess.preprocess_text(extracted_data['text'])
                    return jsonify({"text": preprocessed_text, "images": extracted_data['images']})
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            else:
                return jsonify({"error": f"No processor function found for {file_extension}"}), 500

    return jsonify({"error": "Invalid file type. Only .docx, .pdf, and .txt allowed."}), 400

# Function to process DOCX files
def process_docx_file(file_path):
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return {"text": text.strip(), "images": []}  # No images for docx files

# Function to process PDF files
def process_pdf_file(file_path):
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()
    images = []  # List to store image data

    # Open the PDF file using PyMuPDF
    pdf_document = fitz.open(file_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Extract text
        text += page.get_text() + "\n"

        # Extract images
        for image_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]  # Reference to the image object
            base_image = pdf_document.extract_image(xref)
            if base_image:
                try:
                    # Decode the image data
                    image_bytes = base_image["image"]
                    image_stream = io.BytesIO(image_bytes)
                    image = Image.open(image_stream)

                    # Save the image to a base64 string
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    # Append image details
                    images.append({
                        "page": page_num + 1,
                        "image_index": image_index + 1,
                        "image_data": img_str
                    })
                except Exception as e:
                    print(f"Error processing image {image_index + 1} on page {page_num + 1}: {e}")

    pdf_document.close()

    return {"text": text.strip(), "images": images}

# Function to process TXT files
def process_txt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = "\n".join([line.strip() for line in file])
        return text.strip()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error: {e}" 
        return {"text": text.strip(), "images": []}  # No images for txt files
    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
