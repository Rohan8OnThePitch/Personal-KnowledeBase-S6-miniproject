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

# App layout with columns
st.title("📚 Conversation-Aware RAG System")

# Create a two-column layout
col1, col2 = st.columns([2, 1])

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
    
    # Add memory toggle in sidebar
    st.header("⚙️ Settings")
    st.session_state.use_conversation_memory = st.toggle(
        "Use conversation memory", 
        value=st.session_state.use_conversation_memory,
        help="When enabled, the assistant will use previous conversation context"
    )

# Main chat area (left column)
with col1:
    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, SystemMessage) and "Use the following context" in message.content:
            # Skip displaying context messages
            continue
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
                    conversation_weight=0.3,  # Default weights
                    document_weight=0.7
                )
                
                st.session_state.last_retrieved_sources = response.get("sources", [])
                if response.get("sources"):
                    st.session_state.messages.append(create_context_message(response["sources"]))
                
                st.write(response["answer"])
                st.session_state.messages.append(AIMessage(content=response["answer"]))
                store_message(SESSION_ID, response["answer"], "assistant")

    # Clear chat history button (in main column)
    if st.button("Clear Chat History"):
        st.session_state.messages = [SystemMessage(content="Welcome! I'm your RAG assistant. Upload documents or ask questions.")]
        st.session_state.last_retrieved_sources = []
        st.rerun()

# Source panel (right column)
with col2:
    st.header("📑 Referenced Sources")
    
    if st.session_state.last_retrieved_sources:
        # Group sources by type
        document_sources = [s for s in st.session_state.last_retrieved_sources if s["type"] == "document"]
        conversation_sources = [s for s in st.session_state.last_retrieved_sources if s["type"] == "conversation"]
        
        # Display document sources
        if document_sources:
            st.subheader("Document Chunks")
            for i, source in enumerate(document_sources):
                with st.expander(f"Chunk {i+1} (Score: {source['score']:.2f})"):
                    st.markdown(f"**Strategy:** {source.get('strategy', 'Unknown')}")
                    
                    # Display metadata if available
                    if source.get("metadata"):
                        meta = source["metadata"]
                        meta_text = ""
                        if meta.get("page"):
                            meta_text += f"**Page:** {meta['page']}"
                        if meta.get("source"):
                            meta_text += f" | **Source:** {meta['source']}"
                        if meta_text:
                            st.markdown(meta_text)
                    
                    # Display chunk text
                    st.markdown("---")
                    st.markdown(source["text"])
        
        # Display conversation sources
        if conversation_sources:
            st.subheader("Conversation Context")
            for i, source in enumerate(conversation_sources):
                with st.expander(f"{source.get('role', 'Message').capitalize()} {i+1}"):
                    st.markdown(source["text"])
                    
        
        # Display source statistics
        st.info(f"Used {len(document_sources)} document chunks and {len(conversation_sources)} conversation references")
    else:
        st.info("Ask a question to see referenced sources here.")
