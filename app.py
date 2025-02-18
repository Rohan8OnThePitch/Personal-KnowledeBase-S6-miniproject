import os
import nltk
import gradio as gr
from documents import process_docx, process_pdf, process_txt
from indexing import index_document
from querying import query_documents
import preprocess

# Download required NLTK data (do this *once* when the app starts)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def process_and_query(file, query_text):
    """
    Processes a document, indexes it, and performs a query.  This is the
    main function called by the Gradio interface.
    """
    if not file:
        return "No file uploaded", []

    file_path = file.name  # Gradio passes a NamedTemporaryFile

    # Process file
    if file.name.endswith('.docx'):
        text = process_docx(file_path)
    elif file.name.endswith('.pdf'):
        text = process_pdf(file_path)
    elif file.name.endswith('.txt'):
        text = process_txt(file_path)
    else:
        return "Unsupported file type", []
    preprocessed_text = preprocess.preprocess_text(text['text'])
    #print (preprocessed_text) #ADD THIS
    #preprocessed_text =(text['text'])
    # Index the document
    index_result = index_document("documents", file.name, preprocessed_text)

    # Perform the query
    query_results = query_documents("documents", query_text)

    return f"Indexing result: {index_result}", query_results


# Gradio Interface
iface = gr.Interface(
    fn=process_and_query,
    inputs=[
        gr.File(label="Upload Document"),
        gr.Textbox(label="Enter Query")
    ],
    outputs=[
        gr.Textbox(label="Indexing Result"),
        gr.JSON(label="Query Results") # Display query results as JSON
    ],
    title="Document Processing and Query",
    description="Upload a document (docx, pdf, or txt), enter a query, and get the results."
)


if __name__ == '__main__':
    iface.launch()
