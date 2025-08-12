import requests
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import os

# Global embedding model - load once at startup
embedding_model = None

def initialize_embedding_model():
    """Initialize the sentence transformer model"""
    global embedding_model
    if embedding_model is None:
        print("Loading sentence transformer model...")
        # This model is small (~80MB) and works well for general text
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully!")
    return embedding_model

def embed_text(text):
    """Convert text to vector using local model"""
    if embedding_model is None:
        initialize_embedding_model()
    
    # Clean text before embedding
    text = text.strip()
    if not text:
        return np.zeros(384)  # all-MiniLM-L6-v2 has 384 dimensions
    
    try:
        # encode returns numpy array directly
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        print(f"Error embedding text: {e}")
        return np.zeros(384)  # Return zero vector on error

def load_and_chunk_pdf(file_path):
    """Load PDF and split into chunks"""
    try:
        doc = fitz.open(file_path)
        text = ""
        
        print(f"Processing {len(doc)} pages...")
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        doc.close()
        
        if not text.strip():
            raise ValueError("No text content found in PDF")
        
        print(f"Extracted {len(text)} characters from PDF")
        
        # Use better chunking parameters for technical documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Slightly smaller chunks for better precision
            chunk_overlap=200,  # More overlap to maintain context
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        
        # Filter out very short chunks
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
        
        print(f"Created {len(chunks)} text chunks")
        return chunks
        
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise

def create_faiss_index(chunks):
    """Create FAISS index from text chunks using local embeddings"""
    try:
        # Initialize embedding model if not already done
        initialize_embedding_model()
        
        print("Generating embeddings for chunks...")
        vectors = []
        successful_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i % 20 == 0:  # Progress indicator every 20 chunks
                print(f"Processing chunk {i+1}/{len(chunks)}")
            
            try:
                embedding = embed_text(chunk)
                if embedding is not None and len(embedding) > 0:
                    vectors.append(embedding)
                    successful_chunks.append(chunk)
                else:
                    print(f"Skipping chunk {i} - empty embedding")
            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                continue
        
        if not vectors:
            raise ValueError("No embeddings were created successfully")
        
        print(f"Successfully created {len(vectors)} embeddings")
        
        # Create FAISS index
        dim = len(vectors[0])
        print(f"Vector dimension: {dim}")
        
        # Use IndexFlatIP for cosine similarity (better for sentence transformers)
        index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity
        
        # Normalize vectors for cosine similarity
        vector_array = np.array(vectors).astype('float32')
        faiss.normalize_L2(vector_array)  # Normalize for cosine similarity
        
        index.add(vector_array)
        
        print(f"FAISS index created with {index.ntotal} vectors of dimension {dim}")
        
        return index, successful_chunks
        
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise

def retrieve_similar_chunks(query, index, chunks, top_k=5):
    """Retrieve most similar chunks to the query"""
    try:
        if embedding_model is None:
            initialize_embedding_model()
        
        # Get query embedding
        query_embedding = embed_text(query)
        if query_embedding is None or len(query_embedding) == 0:
            return []
        
        # Normalize query vector for cosine similarity
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search for similar chunks
        scores, indices = index.search(query_vector, top_k)
        
        # Filter out invalid indices and return chunks with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(chunks):
                results.append({
                    'chunk': chunks[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        # Sort by score (higher is better for cosine similarity)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
        
    except Exception as e:
        print(f"Error retrieving similar chunks: {e}")
        return []

# Groq API configuration (unchanged)
API_KEY = "Your Key"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def chat_with_groq(messages):
    """Send messages to Groq API and get response"""
    try:
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500  # Allow longer responses
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Groq API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return "Sorry, I encountered an error while processing your request."
            
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Sorry, something went wrong while generating the response."

def main():
    """Terminal chatbot for testing"""
    print("🤖 Groq LLaMA3 Terminal Chatbot with Local RAG")
    print("Type 'exit' to end the conversation.\n")

    # Initialize embedding model
    initialize_embedding_model()

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        reply = chat_with_groq(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"Bot: {reply}\n")

if __name__ == "__main__":
    main()