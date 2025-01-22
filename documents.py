from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from docx import Document
import fitz  # PyMuPDF
import io
import base64
from PIL import Image
import preprocess
import json
import os
from openpyxl import load_workbook
import xlrd
import yaml
import csv

app = Flask(__name__)

# Set the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx', 'pdf', 'txt','csv','xlsx','json','py','java','js','html','css','cpp','cs','yaml','ts'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dictionary mapping file types to processing functions
FILE_PROCESSORS = {
    'docx': 'process_docx_file',
    'pdf': 'process_pdf_file',
    'txt': 'process_txt_file',
    'csv': 'process_csv_file',
    'xlsx':'process_excel_file',
    'json':'process_json_file',
    'py':'process_python_file',
    'java':'process_java_file',
    'js':'process_javascript_file',
    'html':'process_html_file',
    'css':'process_css_file',
    'cpp':'process_cpp_file',
    'cs':'process_cs_file',
    'yaml':'process_yaml_file',
    'ts':'process_ts_file'

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
        return {"text": text.strip(), "images": []}  # No images for txt files
    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}
        
        
#.csv files
def process_csv_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            text = "\n".join([", ".join(row) for row in reader])
        return {"text": text.strip(), "images": []}  # No images for CSV files
    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}

#.xlsx files
def process_excel_file(file_path):
    try:
        text = ""
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".xlsx":
            # Process .xlsx files using openpyxl
            workbook = load_workbook(file_path, data_only=True)
            for sheet in workbook.worksheets:
                # Extract text from the sheet
                for row in sheet.iter_rows(values_only=True):
                    row_text = ", ".join([str(cell) if cell is not None else "" for cell in row])
                    text += row_text + "\n"


        else:
            return {"text": "Error: Unsupported file format.", "images": []}

        return {"text": text.strip(), "images": []}

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}

#.json files
def process_json_file(file_path):
    try:
        # Read and parse the JSON file
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        # Convert JSON data to a pretty-printed string
        text = json.dumps(data, indent=4, ensure_ascii=False)
        
        return {"text": text.strip(), "images": []}  # No images for JSON files

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except json.JSONDecodeError:
        return {"text": "Error: Invalid JSON format.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}
        
#.python files
def process_python_file(file_path):
    try:
        # Read the Python file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return {"text": text.strip(), "images": []}  # No images for Python files

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}
        
#.java files
def process_java_file(file_path):
    try:
        # Read the Java file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return {"text": text.strip(), "images": []}  # No images for Java files

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}
        
#.js files
def process_javascript_file(file_path):
    try:
        # Read the JavaScript file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return {"text": text.strip(), "images": []}  # No images for JavaScript files

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}
        
#.html files
def process_html_file(file_path):
    try:
        # Read the HTML file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return {"text": text.strip(), "images": []}  # No images for HTML files (only references)

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}
        
#.css files
def process_css_file(file_path):
    try:
        # Read the CSS file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return {"text": text.strip(), "images": []}  # No images for CSS files (only references)

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}
        
#.cpp files
def process_cpp_file(file_path):
    try:
        # Read the C++ file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return {"text": text.strip(), "images": []}  # No images for C++ files

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}

#.cs files
def process_cs_file(file_path):
    try:
        # Read the C# file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return {"text": text.strip(), "images": []}  # No images for C# files

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}

#.ts files
def process_ts_file(file_path):
    try:
        # Read the TypeScript file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return {"text": text.strip(), "images": []}  # No images for TypeScript files

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}
#.yaml files
def process_yaml_file(file_path):
    try:
        # Read and load the YAML file
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        # Convert YAML data to a pretty-printed string
        text = yaml.dump(data, default_flow_style=False, allow_unicode=True)

        return {"text": text.strip(), "images": []}  # No images for YAML files

    except FileNotFoundError:
        return {"text": "Error: File not found.", "images": []}
    except yaml.YAMLError:
        return {"text": "Error: Invalid YAML format.", "images": []}
    except Exception as e:
        return {"text": f"Error: {e}", "images": []}


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
