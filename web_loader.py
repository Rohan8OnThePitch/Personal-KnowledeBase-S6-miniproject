import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_web_content(url, chunk_size=1000, chunk_overlap=200):
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

# Example usage
if __name__ == '__main__':
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    chunks = load_and_chunk_web_content(url)
    print(chunks[:5])
