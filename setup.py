#!/usr/bin/env python3
"""
Setup script for Local CUDA RAG Chatbot
This script helps you set up the environment and test the system
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_pdf_exists():
    """Check if the CUDA PDF exists"""
    pdf_path = "data/cudabook.pdf"
    
    if not os.path.exists("data"):
        os.makedirs("data")
        print("📁 Created data/ directory")
    
    if os.path.exists(pdf_path):
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        print(f"✅ Found CUDA PDF: {pdf_path} ({file_size:.1f} MB)")
        return True
    else:
        print(f"❌ CUDA PDF not found: {pdf_path}")
        print("Please place your CUDA programming guide PDF in the data/ folder")
        print("You can download it from: https://docs.nvidia.com/cuda/cuda-c-programming-guide/")
        return False

def test_embedding_model():
    """Test the local embedding model"""
    print("🧠 Testing local embedding model...")
    try:
        from ChatbotV1 import initialize_embedding_model, embed_text
        
        # Initialize model
        initialize_embedding_model()
        
        # Test embedding
        test_text = "CUDA is a parallel computing platform"
        embedding = embed_text(test_text)
        
        if embedding is not None and len(embedding) > 0:
            print(f"✅ Embedding model working! Vector dimension: {len(embedding)}")
            return True
        else:
            print("❌ Embedding model returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Error testing embedding model: {e}")
        return False

def test_groq_connection():
    """Test Groq API connection"""
    print("🌐 Testing Groq API connection...")
    try:
        from ChatbotV1 import chat_with_groq
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."}
        ]
        
        response = chat_with_groq(messages)
        
        if response and "sorry" not in response.lower():
            print("✅ Groq API connection working!")
            print(f"Test response: {response[:100]}...")
            return True
        else:
            print("❌ Groq API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Groq API: {e}")
        return False

def create_static_folder():
    """Create static folder and move index.html if needed"""
    if not os.path.exists("static"):
        os.makedirs("static")
        print("📁 Created static/ directory")
    
    # If index.html exists in current directory, move it to static/
    if os.path.exists("index.html") and not os.path.exists("static/index.html"):
        import shutil
        shutil.move("index.html", "static/index.html")
        print("📄 Moved index.html to static/ folder")

def main():
    """Main setup function"""
    print("🚀 Local CUDA RAG Chatbot Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create necessary directories
    create_static_folder()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at package installation")
        return
    
    print("\n🔍 Running system tests...")
    print("-" * 30)
    
    # Test embedding model
    embedding_ok = test_embedding_model()
    
    # Test Groq API
    groq_ok = test_groq_connection()
    
    # Check PDF
    pdf_ok = check_pdf_exists()
    
    print("\n📊 Setup Summary:")
    print("-" * 20)
    print(f"Local Embeddings: {'✅' if embedding_ok else '❌'}")
    print(f"Groq API: {'✅' if groq_ok else '❌'}")
    print(f"CUDA PDF: {'✅' if pdf_ok else '❌'}")
    
    if embedding_ok and groq_ok:
        print("\n🎉 Setup completed successfully!")
        if pdf_ok:
            print("🟢 Full RAG functionality available")
        else:
            print("🟡 Basic functionality available (add PDF for full RAG)")
        
        print("\nTo start the application:")
        print("python app.py")
        
    else:
        print("\n❌ Setup incomplete. Please resolve the issues above.")

if __name__ == "__main__":
    main()