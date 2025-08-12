from flask import Flask, request, jsonify, send_from_directory
from ChatbotV1 import chat_with_groq, load_and_chunk_pdf, create_faiss_index, retrieve_similar_chunks, initialize_embedding_model
from flask_cors import CORS
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# Global variables to store the index and chunks
index = None
chunks = None

def retrieve_context(query, index, chunks, top_k=5):
    """Retrieve relevant context from the document using local embeddings"""
    try:
        # Use the local embedding function
        results = retrieve_similar_chunks(query, index, chunks, top_k)
        
        if not results:
            return ""
        
        # Format the context with relevance scores
        context_parts = []
        for i, result in enumerate(results):
            score_percentage = result['score'] * 100
            context_parts.append(
                f"[Relevance: {score_percentage:.1f}%] {result['chunk']}"
            )
        
        context = "\n\n".join(context_parts)
        print(f"Retrieved {len(results)} relevant chunks for query: {query[:50]}...")
        return context
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

def chat_with_docs(query, index, chunks, messages):
    """Chat with document context using local RAG"""
    context = retrieve_context(query, index, chunks, top_k=5)
    
    if context.strip():
        # Create a new messages list to avoid modifying the original
        enhanced_messages = messages.copy()
        
        # Add system message with context
        system_message = {
            "role": "system", 
            "content": f"""You are a CUDA programming expert assistant. You have access to the CUDA programming guide and documentation.

Use the following document context to answer the user's question. The context includes relevance scores to help you identify the most pertinent information.

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Prioritize information from higher relevance scores when answering
- Be specific and provide detailed explanations with examples when possible
- If you reference information from the context, be clear about it
- If the context doesn't fully answer the question, combine it with your general knowledge
- For code examples, provide complete, working code when relevant
- Focus on practical, actionable advice for CUDA programming

Remember: You are helping with CUDA programming, GPU computing, parallel programming concepts, and related topics."""
        }
        
        enhanced_messages.insert(0, system_message)
        enhanced_messages.append({"role": "user", "content": query})
        
        print(f"Using RAG context for query: {query[:100]}...")
        return chat_with_groq(enhanced_messages)
    else:
        # Fallback to regular chat if no context
        print("No relevant context found, using general knowledge...")
        fallback_messages = messages.copy()
        fallback_messages.insert(0, {
            "role": "system", 
            "content": "You are a helpful CUDA programming expert. Provide detailed explanations about CUDA, GPU programming, and parallel computing concepts."
        })
        fallback_messages.append({"role": "user", "content": query})
        return chat_with_groq(fallback_messages)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        messages = data.get('messages', [])
        
        if not user_message.strip():
            return jsonify({"error": "Empty message"}), 400
        
        print(f"User query: {user_message}")
        
        # Use document context if available
        if index is not None and chunks is not None:
            print("Using local RAG for response generation")
            reply = chat_with_docs(user_message, index, chunks, messages)
        else:
            print("No document index available, using general chat")
            temp_messages = messages.copy()
            temp_messages.insert(0, {
                "role": "system", 
                "content": "You are a helpful CUDA programming expert. Provide detailed explanations about CUDA, GPU programming, and parallel computing concepts."
            })
            temp_messages.append({"role": "user", "content": user_message})
            reply = chat_with_groq(temp_messages)
        
        # Add messages to history
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
        
        return jsonify({"reply": reply, "messages": messages})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/status')
def status():
    """Endpoint to check if document indexing is working"""
    return jsonify({
        "document_loaded": index is not None and chunks is not None,
        "num_chunks": len(chunks) if chunks else 0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "index_type": "FAISS with cosine similarity"
    })

@app.route('/search', methods=['POST'])
def search_documents():
    """Endpoint to test document search functionality"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query.strip():
            return jsonify({"error": "Empty query"}), 400
        
        if index is None or chunks is None:
            return jsonify({"error": "Document not loaded"}), 400
        
        results = retrieve_similar_chunks(query, index, chunks, top_k=5)
        
        return jsonify({
            "query": query,
            "results": results,
            "num_results": len(results)
        })
        
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        return jsonify({"error": "Search failed"}), 500

if __name__ == '__main__':
    print("🚀 Starting CUDA RAG Chatbot with Local Embeddings")
    print("=" * 50)
    
    # Initialize embedding model first
    try:
        print("Initializing local embedding model...")
        initialize_embedding_model()
        print("✅ Local embedding model ready!")
    except Exception as e:
        print(f"❌ Failed to initialize embedding model: {e}")
        print("The application will start but RAG functionality will be disabled.")
    
    # Load and process PDF
    pdf_path = "data/cudabook.pdf"
    if os.path.exists(pdf_path):
        try:
            print(f"Loading PDF: {pdf_path}")
            chunks = load_and_chunk_pdf(pdf_path)
            print(f"✅ Created {len(chunks)} chunks from PDF")
            
            print("Building FAISS index with local embeddings...")
            index, chunks = create_faiss_index(chunks)
            print("✅ FAISS index built successfully!")
            print(f"📊 Index contains {index.ntotal} vectors")
            
        except FileNotFoundError:
            print(f"❌ Error: Could not find '{pdf_path}'")
            print("Please ensure the CUDA book PDF exists in the data/ folder.")
            index = None
            chunks = None
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            index = None
            chunks = None
    else:
        print(f"❌ PDF file not found: {pdf_path}")
        print("The application will start but document-based responses will be unavailable.")
        index = None
        chunks = None
    
    print("=" * 50)
    if index is not None:
        print("🟢 RAG system is ready! Document-aware responses enabled.")
    else:
        print("🟡 Running in basic mode. Upload a PDF to enable RAG functionality.")
    
    print("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)