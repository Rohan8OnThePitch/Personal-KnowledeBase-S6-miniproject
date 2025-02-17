from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

# Initialize Qdrant client and model
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Use GPT-2 Small instead of Mistral-7B
gpt2_model_name = "gpt2"  # GPT-2 Small model
tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(
    gpt2_model_name,
    torch_dtype=torch.float16,  # Use float16 for lower memory usage
    device_map="auto"  # Automatically load on GPU if available
)
import re

def generate_answer(query, context):
    """
    Generates a natural language answer using GPT-2.
    
    Args:
        query (str): The user's query.
        context (str): Retrieved chunks from Qdrant.
    
    Returns:
        str: Generated answer.
    """
    try:
        # Clean context to remove any irrelevant code or content
        print(f"Original Context: {context}")  # Debug: Print original context
        context = re.sub(r"[\w\s]*\(\s*[^)]+\s*\)", "", context)  # Basic cleanup for code fragments
        print(f"Cleaned Context: {context}")  # Debug: Print cleaned context

        # Create a prompt for GPT-2
        prompt = f"""
        Context: {context}

        Question: {query}

        Answer:
        """
        print(f"Prompt: {prompt}")  # Debug: Print the full prompt to check how it's formed

        inputs = tokenizer(prompt, return_tensors="pt").to(gpt2_model.device)
        outputs = gpt2_model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=100,
            temperature=0.9,
            top_p=0.98,
        )
        print(f"Raw Output Tokens: {outputs}")  # Debug: Print the raw output tokens

        # Decode and clean the answer
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Decoded Answer: {answer}")  # Debug: Print the decoded answer before cleaning
        
        # Post-process the answer to remove repetitive content
        answer = re.sub(r"[\w\s]*\(\s*[^)]+\s*\)", "", answer)
        print(f"Final Answer: {answer}")  # Debug: Print the final cleaned answer

        return answer.strip() if answer else "No answer generated"
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I couldn't generate an answer. Please try again"


def preprocess_query(query):
    """
    Preprocesses the user query by cleaning and normalizing it.
    """
    query = query.lower()  # Lowercase
    #query = re.sub(r"[^\w\s]", "", query)  # Remove special characters
    #query = re.sub(r"\s+", " ", query)  # Remove extra whitespace
    return query

def query_documents(collection_name, user_query, top_k=5, score_threshold=0.5):
    try:
        if not user_query or not user_query.strip():
            return {"error": "Query cannot be empty"}

        # Preprocess the query
        processed_query = preprocess_query(user_query)
        # Generate the query vector from the user query
        query_vector = model.encode(processed_query).tolist()  # Ensure it's a list
       # print("Query vector:", query_vector)

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
                "chunk_index": res.payload.get("chunk_index", ""),
                "payload": res.payload  # Include the entire payload
            }
            for res in search_results
            if res.score >= score_threshold  # Only keep results with a high score
        ]

        # Print the relevant payload for each result
        if filtered_results:
            '''for i, result in enumerate(filtered_results):
                print(f"\nResult {i + 1} (Score: {result['score']:.3f}):")
                print(f"ID: {result['id']}")
                print(f"Text: {result['text']}")
                print(f"Document ID: {result['document_id']}")
                print(f"Chunk Index: {result['chunk_index']}")
                print(f"Full Payload: {result['payload']}")  # Print the entire payload'''
        else:
            print("No relevant results found.")

        # Combine text from filtered results for context
        context = " ".join([res["text"] for res in filtered_results])  
        answer = generate_answer(user_query, context)
        return {"answer": answer, "chunks": filtered_results}

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
        
    '''print("\n=== Relevant Chunks ===")
        for i, chunk in enumerate(result["chunks"]):
            print(f"\nChunk {i + 1} (Score: {chunk['score']:.3f}):")
            print(f"ID: {chunk['id']}")
            print(f"Text: {chunk['text']}")
            print(f"Document ID: {chunk['document_id']}")
            print(f"Chunk Index: {chunk['chunk_index']}")
            print(f"Full Payload: {chunk['payload']}")  # Print the entire payload'''