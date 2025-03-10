from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer
import uuid
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "chat_memory"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize Qdrant and Embedding Model
qdrant_client = QdrantClient(url="http://localhost:6333")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def ensure_memory_collection_exists():
    """Ensure the chat memory collection exists in Qdrant."""
    collections = qdrant_client.get_collections()
    collection_names = [collection.name for collection in collections.collections]

    if COLLECTION_NAME not in collection_names:
        from qdrant_client.models import VectorParams, Distance
        logger.info(f"Creating {COLLECTION_NAME} collection")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            ),
        )


def store_message(session_id: str, message_content: str, role: str, timestamp=None, message_id=None):
    """Store a chat message in Qdrant with metadata."""
    ensure_memory_collection_exists()

    # Generate embedding
    embedding = embedding_model.encode(message_content).tolist()

    # Generate message ID if not provided
    message_id = message_id or str(uuid.uuid4())

    # Get sequence number (number of messages in this session + 1)
    filter_query = Filter(must=[
        FieldCondition(key="session_id", match=MatchValue(value=session_id))
    ])

    existing_messages = qdrant_client.scroll(
    collection_name=COLLECTION_NAME,
    scroll_filter=filter_query,
    limit=1000
    )[0]
    sequence_num = len(existing_messages) + 1

    # Store the message
    point = PointStruct(
        id=message_id,
        vector=embedding,
        payload={
            "session_id": session_id,
            "content": message_content,
            "role": role,
            "sequence_num": sequence_num,
            "timestamp": timestamp or import_time_module().time()
        }
    )

    qdrant_client.upsert(COLLECTION_NAME, [point])
    logger.info(f"Stored message (role={role}, seq={sequence_num}) in session {session_id}")


def import_time_module():
    """Helper function to import time module."""
    import time
    return time


def retrieve_messages_by_sequence(session_id: str, start_seq: int = None, end_seq: int = None) -> List:
    """Retrieve chat messages by sequence range."""
    filter_query = [FieldCondition(key="session_id", match=MatchValue(value=session_id))]

    if start_seq is not None or end_seq is not None:
        range_condition = {}
        if start_seq is not None:
            range_condition["gte"] = start_seq
        if end_seq is not None:
            range_condition["lte"] = end_seq

        filter_query.append(FieldCondition(key="sequence_num", range=Range(**range_condition)))

    # Execute query
    search_result = qdrant_client.scroll(
    collection_name=COLLECTION_NAME,
    scroll_filter=Filter(must=filter_query), 
    limit=1000,
    with_payload=True,
    )[0]

    # Create a mapping of content to sequence number for easier lookup
    content_to_seq = {}
    for result in search_result:
        if "sequence_num" in result.payload:
            content_to_seq[result.payload["content"]] = result.payload["sequence_num"]

    messages = []
    for result in search_result:
        payload = result.payload
        if payload["role"] == "user":
            messages.append(HumanMessage(content=payload["content"]))
        elif payload["role"] == "assistant":
            messages.append(AIMessage(content=payload["content"]))
        elif payload["role"] == "system":
            messages.append(SystemMessage(content=payload["content"]))

    # Sort by sequence number if available, otherwise leave in order returned by DB
    if content_to_seq:
        try:
            messages.sort(key=lambda x: content_to_seq.get(x.content, 0))
        except Exception as e:
            logger.warning(f"Error sorting messages: {e}")
            # Fall back to not sorting if there's an error

    return messages


def retrieve_context_relevant_messages(session_id: str, query: str, context_window: int = 2, top_k: int = 5) -> List:
    """Retrieve relevant messages based on semantic similarity with context."""
    query_vector = embedding_model.encode(query).tolist()

    filter_query = Filter(must=[
        FieldCondition(key="session_id", match=MatchValue(value=session_id))
    ])

    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=filter_query,
        limit=top_k,
        score_threshold=0.6
    )

    if not search_results:
        logger.info(f"No relevant messages found for query in session {session_id}")
        return []

    # Collect relevant sequences
    matched_sequences = []
    for result in search_results:
        if "sequence_num" in result.payload:
            matched_sequences.append(result.payload["sequence_num"])
        else:
            logger.warning(f"Message without sequence_num found: {result.payload}")
    
    if not matched_sequences:
        logger.warning("No messages with sequence numbers found")
        return []

    # Determine context ranges
    sequence_ranges = [(max(1, seq - context_window), seq + context_window) for seq in matched_sequences]
    sequence_ranges.sort()

    # Merge overlapping ranges
    merged_ranges = []
    current_start, current_end = sequence_ranges[0]

    for start, end in sequence_ranges[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end

    merged_ranges.append((current_start, current_end))

    # Retrieve messages within these ranges
    all_messages = []
    for start_seq, end_seq in merged_ranges:
        messages = retrieve_messages_by_sequence(session_id, start_seq, end_seq)
        all_messages.extend(messages)

    # Remove duplicates
    seen = set()
    unique_messages = []
    for msg in all_messages:
        if msg.content not in seen:
            seen.add(msg.content)
            unique_messages.append(msg)

    logger.info(f"Retrieved {len(unique_messages)} relevant messages from {len(merged_ranges)} segments")
    return unique_messages


def get_all_session_messages(session_id: str) -> List:
    """Retrieve all messages from a session."""
    try:
        return retrieve_messages_by_sequence(session_id)
    except Exception as e:
        logger.error(f"Error retrieving session messages: {e}")
        return []  # Return empty list on error to prevent app crash


def format_context_messages(messages: List) -> str:
    """Format messages into a structured context string."""
    formatted_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_parts.append(f"Assistant: {msg.content}")
        elif isinstance(msg, SystemMessage):
            formatted_parts.append(f"System: {msg.content}")

    return "\n\n".join(formatted_parts)


def main():
    """Main function for testing."""
    session_id = "test_session"

    # Sample messages
    messages = [
        ("Hello, how can I help?", "assistant"),
        ("Tell me about Python.", "user"),
        ("Python is a programming language.", "assistant"),
        ("What is Python used for?", "user"),
        ("Python is used for web development, data science, and more.", "assistant")
    ]

    # Store messages
    for content, role in messages:
        store_message(session_id, content, role)

    # Retrieve all messages
    all_messages = get_all_session_messages(session_id)
    print("All Messages:")
    print(format_context_messages(all_messages))

    # Retrieve relevant messages based on a query
    relevant_messages = retrieve_context_relevant_messages(session_id, query="Tell me about Python", top_k=3)
    print("\nRelevant Messages:")
    print(format_context_messages(relevant_messages))


if __name__ == "__main__":
    main()
