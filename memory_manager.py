from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter
from sentence_transformers import SentenceTransformer
import uuid
COLLECTION_NAME = "chat_memory"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
qdrant_client = QdrantClient(url="http://localhost:6333")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def store_message(session_id: str, message_content: str, role: str):
    """Store a chat message in Qdrant."""
    embedding = embedding_model.encode(message_content).tolist()
    
    # Generate a valid UUID for the point ID
    point_id = str(uuid.uuid4())

    point = PointStruct(
        id=point_id,  # Use the valid UUID
        vector=embedding,
        payload={"session_id": session_id, "content": message_content, "role": role}
    )
    qdrant_client.upsert(COLLECTION_NAME, [point])

def retrieve_messages(session_id: str, top_k: int = 5):
    """Retrieve relevant chat messages from the MEMORY_COLLECTION."""
    filter_query = Filter(must=[{"key": "session_id", "match": {"value": session_id}}])
    search_result = qdrant_client.search(
        collection_name="chat_memory",
        query_vector=[0.0] * embedding_model.get_sentence_embedding_dimension(),  # Placeholder vector
        query_filter=filter_query,  # Use `query_filter` instead of `filter`
        limit=top_k,
    )
    messages = []
    for result in search_result:
        payload = result.payload
        if payload["role"] == "user":
            messages.append(HumanMessage(content=payload["content"]))
        elif payload["role"] == "assistant":
            messages.append(AIMessage(content=payload["content"]))
        elif payload["role"] == "system":
            messages.append(SystemMessage(content=payload["content"]))
    return messages
def main():
    # Define a session ID for testing
    session_id = "test_session"

    # Sample messages to store
    messages = [
        ("Hello, how can I help you today?", "assistant"),
        ("I need some information on data science.", "user"),
        ("Data science involves statistics, programming, and domain expertise.", "assistant"),
        ("What tools are commonly used?", "user"),
        ("Popular tools include Python, R, SQL, and various libraries like pandas and scikit-learn.", "assistant")
    ]

    # Store the messages in Qdrant
    for content, role in messages:
        store_message(session_id, content, role)

    # Retrieve the stored messages
    retrieved_messages = retrieve_messages(session_id, top_k=5)

    # Display the retrieved messages
    print("Retrieved Messages:")
    for message in retrieved_messages:
        print(f"{message.type.capitalize()}: {message.content}")

if __name__ == "__main__":
    main()