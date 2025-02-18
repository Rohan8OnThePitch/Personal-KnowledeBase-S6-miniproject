import os
import torch
import logging
import re
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Load Qdrant Configuration from Environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Initialize Qdrant Client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Load Sentence Transformer for Query Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load GPT-2 from Hugging Face
GPT2_MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
gpt2_model = AutoModelForCausalLM.from_pretrained(
    GPT2_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def validate_context_relevance(context, query, threshold=0.1):  # Lowered threshold
    """
    Validates if the context is relevant to the query using semantic similarity.
    Returns True if context is relevant, False otherwise.
    """
    if not context.strip():
        return False
        
    # Split context into smaller chunks for better comparison
    context_chunks = [context[i:i+512] for i in range(0, len(context), 512)]
    
    # Get embeddings for query and context chunks
    query_embedding = embedding_model.encode(query)
    max_similarity = 0
    
    # Find maximum similarity across chunks
    for chunk in context_chunks:
        if chunk.strip():
            chunk_embedding = embedding_model.encode(chunk)
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(chunk_embedding).unsqueeze(0)
            ).item()
            max_similarity = max(max_similarity, similarity)
    
    return max_similarity >= threshold

def extract_answer_from_response(response):
    """
    Extracts and validates the answer from the model's response.
    Returns None if the answer doesn't meet quality criteria.
    """
    # Remove any prefixes/context from the response
    answer_parts = response.split("Answer:")
    if len(answer_parts) > 1:
        answer = answer_parts[-1].strip()
    else:
        answer = response.strip()
    
    # Basic length validation
    if len(answer) < 2 or len(answer) > 500:  # Increased maximum length
        return None
        
    # Less strict hallucination check
    hallucination_patterns = [
        r"I don't have enough information",
        r"I cannot provide",
        r"no information available",
        r"insufficient data"
    ]
    
    if any(re.search(pattern, answer, re.IGNORECASE) for pattern in hallucination_patterns):
        return None
        
    return answer

def generate_answer(query, context):
    """Generates a concise, focused response using GPT-2 based on the retrieved context."""
    if not context.strip():
        return "No information found in the database."

    # Simplified prompt structure
    prompt = f"""
    Context: {context}
    Question: {query}
    Answer: Let me answer based on the context provided."""

    inputs = tokenizer(prompt, return_tensors="pt").to(gpt2_model.device)
    
    outputs = gpt2_model.generate(
        **inputs,
        max_new_tokens=100,  # Increased token limit
        temperature=0.3,     # Increased temperature for more flexibility
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True      # Enable sampling for more natural responses
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    answer = extract_answer_from_response(response)
    return answer if answer else response.split("Answer:")[-1].strip()

def query_documents(collection_name, user_query, top_k=5, score_threshold=0.3):  # Lowered threshold
    """Queries Qdrant, retrieves matching documents, and generates an answer using GPT-2."""
    try:
        logging.info(f"🔍 Processing query: {user_query}")

        # Generate Query Embedding
        query_vector = embedding_model.encode(user_query).tolist()

        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        if not search_results:
            return {
                "answer": "No relevant information found in the database.",
                "chunks": []
            }

        # Filter and sort results
        filtered_results = [
            {
                "id": res.id,
                "score": res.score,
                "text": res.payload.get("text", ""),
            }
            for res in search_results
            if "text" in res.payload
        ]

        # Sort by relevance score
        filtered_results.sort(key=lambda x: x["score"], reverse=True)

        # Combine contexts
        context = " ".join(result["text"] for result in filtered_results)
        
        if not context.strip():
            return {
                "answer": "No relevant information found in the database.",
                "chunks": []
            }

        answer = generate_answer(user_query, context)
        return {"answer": answer, "chunks": filtered_results}

    except Exception as e:
        logging.error(f"Error during query: {e}")
        return {"error": f"An error occurred: {str(e)}"}

if _name_ == "main":
    import argparse

    parser = argparse.ArgumentParser(description="Query documents with GPT-2")
    parser.add_argument("--collection", type=str, default="documents", help="Qdrant collection name")
    parser.add_argument("--query", type=str, required=True, help="Your search query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()

    result = query_documents(args.collection, args.query, args.top_k)

    if "error" in result:
        logging.error(f"Error: {result['error']}")
    else:
        print("\n=== Generated Answer ===")
        print(result["answer"])
        
        if result["chunks"]:
            print("\n=== Supporting Evidence ===")
            for i, chunk in enumerate(result["chunks"], 1):
                print(f"\nChunk {i} (Score: {chunk['score']:.3f}):")
                print(chunk["text"])
