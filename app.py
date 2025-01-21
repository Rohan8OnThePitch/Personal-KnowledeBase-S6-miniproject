from flask import Flask, request, jsonify, render_template
import os
from documents import process_docx, process_pdf, process_txt
from indexing import index_document
from querying import query_documents
import preprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to render index.html (upload page)
@app.route('/')
def index():
    return render_template('index.html')  # Serves the HTML form

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process file
    if file.filename.endswith('.docx'):
        text = process_docx(file_path)
    elif file.filename.endswith('.pdf'):
        text = process_pdf(file_path)
    elif file.filename.endswith('.txt'):
        text = process_txt(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400
        preprocessed_text = preprocess.preprocess_text(text['text'])
    # Index the document and get the result
    result = index_document("documents", file.filename,preprocessed_text)

    # Ensure the response is JSON serializable and return it
    return jsonify(result)

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query')
    if not query_text:
        return jsonify({"error": "Query cannot be empty"}), 400

    results = query_documents("documents", query_text)
    #print(results)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
