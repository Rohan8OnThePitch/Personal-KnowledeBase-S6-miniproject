from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Qdrant client and model
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Use Mistral-7B instead of Llama 2
mistral_model_name = "mistralai/Mistral-7B-v0.1"  # Mistral-7B model
tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_name,
    torch_dtype=torch.float16,  # Use float16 for lower memory usage
    device_map="auto"  # Automatically load on GPU if available
)

def generate_answer(query, context):
    """
    Generates a natural language answer using Mistral-7B.
    
    Args:
        query (str): The user's query.
        context (str): Retrieved chunks from Qdrant.
    
    Returns:
        str: Generated answer.
    """
    try:
        # Create a prompt for Mistral-7B
        prompt = f"""
        [INST] <<SYS>>
        Answer the question using only the provided context. If unsure, say "I don't know".
        <</SYS>>
        Context: {context}
        Question: {query}
        [/INST]
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(mistral_model.device)
        outputs = mistral_model.generate(
            **inputs,
            max_new_tokens=200,  # Adjust as needed
            temperature=0.7,  # Controls randomness
            top_p=0.3,  # Nucleus sampling
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I couldn't generate an answer. Please try again"

def preprocess_query(query):
    """
    Preprocesses the user query by cleaning and normalizing it.
    """
    query = query.lower()  # Lowercase
    query = re.sub(r"[^\w\s]", "", query)  # Remove special characters
    query = re.sub(r"\s+", " ", query)  # Remove extra whitespace
    return query.strip()

def query_documents(collection_name, user_query, top_k=5, score_threshold=0.5):
    try:
        if not user_query or not user_query.strip():
            return {"error": "Query cannot be empty"}

        # Preprocess the query
        processed_query = preprocess_query(user_query)
        # Generate the query vector from the user query
        query_vector = model.encode(processed_query).tolist()  # Ensure it's a list
        print("Query vector:", query_vector)
        # Perform the search query
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Process the search results
        filtered_results = [
            {
                "id": res.id,
                "score": res.score,
                "text": res.payload.get("text", ""),
                "document_id": res.payload.get("document_id", ""),
                "chunk_index": res.payload.get("chunk_index", "")
            }
            for res in search_results
            if res.score >= score_threshold
        ]
        context = " ".join([res["text"] for res in filtered_results])  # Fix: Extract text from filtered_results
        answer = generate_answer(user_query, context)
        return {"answer": answer, "chunks": filtered_results}  # Fix: Return filtered_results instead of filtered_chunks
       
    except Exception as e:
        print(f"Error during query: {e}")
        return {"error": str(e)}
    
    # Add this to the end of your querying.py file
if __name__ == "__main__":
    import argparse
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Query documents with natural language")
    parser.add_argument("--collection", type=str, default="documents", help="Qdrant collection name")
    parser.add_argument("--query", type=str, required=True, help="Your search query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()

    # Execute the query
    print(f"\nQuerying for: '{args.query}'")
    result = query_documents(
        collection_name=args.collection,
        user_query=args.query,
        top_k=args.top_k
    )

    # Print results
    if "error" in result:
        print(f"\nError: {result['error']}")
    else:
        print("\n=== Generated Answer ===")
        print(result["answer"])
        
        print("\n=== Relevant Chunks ===")
        for i, chunk in enumerate(result["chunks"]):
            print(f"\nChunk {i+1} (Score: {chunk['score']:.3f}):")
            print(chunk["text"])