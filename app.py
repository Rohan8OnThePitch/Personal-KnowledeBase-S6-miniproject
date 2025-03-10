import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from document_loader import load_and_chunk_documents_with_multiple_strategies
from qdrant_helper import index_document_with_strategies

from memory_manager import (
    retrieve_context_relevant_messages,
    get_all_session_messages,
    store_message,
    format_context_messages
)

from conversation_aware_rag import (
    answer_query_with_conversation_context,
    create_context_message
)

# Constants
SESSION_ID = "user_session"
DOCUMENT_COLLECTION = "document_chunks"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Welcome! I'm your RAG assistant. Upload documents or ask questions.")
    ]
    store_message(SESSION_ID, st.session_state.messages[0].content, "system")

if "last_retrieved_sources" not in st.session_state:
    st.session_state.last_retrieved_sources = []

if "use_conversation_memory" not in st.session_state:
    st.session_state.use_conversation_memory = True

# Default weights (hidden from UI)
document_weight = 0.7
conversation_weight = 0.3

# App title
st.title("📚 Conversation-Aware RAG System")

# Settings sidebar
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Choose documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        os.makedirs("uploads", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = f"uploads/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Processing {uploaded_file.name}..."):
                chunks = load_and_chunk_documents_with_multiple_strategies(file_path)
                response = index_document_with_strategies(DOCUMENT_COLLECTION, uploaded_file.name, chunks)
                if response["status"] == "success":
                    st.success(f"✅ Indexed {sum(len(v) for v in chunks.values())} chunks")
                else:
                    st.error(f"❌ Failed: {response['message']}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.write(message.content)

# Chat input
query = st.chat_input("Ask a question about your documents...")
if query:
    with st.chat_message("user"):
        st.write(query)
    
    st.session_state.messages.append(HumanMessage(content=query))
    store_message(SESSION_ID, query, "user")
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_query_with_conversation_context(
                session_id=SESSION_ID,
                query=query,
                use_conversation_memory=st.session_state.use_conversation_memory,
                conversation_weight=conversation_weight,
                document_weight=document_weight
            )
            
            st.session_state.last_retrieved_sources = response.get("sources", [])
            if response.get("sources"):
                st.session_state.messages.append(create_context_message(response["sources"]))
            
            st.write(response["answer"])
            st.session_state.messages.append(AIMessage(content=response["answer"]))
            store_message(SESSION_ID, response["answer"], "assistant")

# Clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = [SystemMessage(content="Welcome! I'm your RAG assistant. Upload documents or ask questions.")]
    st.session_state.last_retrieved_sources = []
    st.experimental_rerun()
