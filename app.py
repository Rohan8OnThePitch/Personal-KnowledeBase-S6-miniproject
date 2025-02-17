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

    # Process file into chunks
    if file.filename.endswith('.docx'):
        chunks = process_docx(file_path)
    elif file.filename.endswith('.pdf'):
        chunks = process_pdf(file_path)
    elif file.filename.endswith('.txt'):
        chunks = process_txt(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # Preprocess each chunk (if needed)
    preprocessed_chunks = [preprocess.preprocess_text(chunk) for chunk in chunks]

    # Index the document chunks
    result = index_document("documents", file.filename, preprocessed_chunks,batch_size=50)

    # Ensure the response is JSON serializable and return it
    return jsonify(result)

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query')
    if not query_text:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Optional: Get score_threshold from the request (default to 0.5)
    score_threshold = request.json.get('score_threshold', 0.5)
    #preproquery=preprocess.preprocess_text(query_text)
    preproquery=query_text

    results = query_documents("documents", preproquery, score_threshold=score_threshold)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
