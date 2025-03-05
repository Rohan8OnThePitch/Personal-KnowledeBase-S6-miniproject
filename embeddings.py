from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def generate_embeddings(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_model.encode(texts)
    return embeddings

# Example usage
if __name__ == '__main__':
    from langchain.schema import Document
    sample_chunks = [Document(page_content='Sample text for embedding generation.')]
    embeddings = generate_embeddings(sample_chunks)
    print(embeddings)
