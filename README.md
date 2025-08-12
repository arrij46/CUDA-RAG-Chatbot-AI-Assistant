# CUDA RAG Chatbot - AI Assistant

A modern, intelligent chatbot powered by **Local Embeddings** and **Document Search** capabilities, specifically designed for CUDA programming, GPU computing, and parallel programming assistance.

## Features

### **AI Capabilities**
- **Local AI Processing** - No external API dependencies for embeddings
- **Document-Aware Responses** - Searches through CUDA documentation
- **Contextual Understanding** - Provides relevant, accurate answers
- **Real-time Processing** - Instant responses with typing indicators

### **Document Search & RAG**
- **PDF Document Processing** - Extracts and chunks large documents
- **FAISS Vector Search** - High-performance similarity search
- **Local Embeddings** - Uses sentence-transformers for privacy
- **Relevance Scoring** - Shows confidence levels for responses

### **Modern Frontend**
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Beautiful UI** - Modern gradient design with glassmorphism effects
- **Interactive Elements** - Example queries, status indicators, animations
- **Real-time Status** - Shows RAG system status and document chunks

### **Technical Features**
- **Flask Backend** - Lightweight and fast
- **CORS Support** - Cross-origin requests enabled
- **Error Handling** - Graceful fallbacks and user-friendly messages
- **Status Endpoints** - System health monitoring

## Setup Instructions

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Prepare Your Documents**
Place your PDF documents in the `data/` folder:
```
data/
├── cudabook.pdf          # CUDA Programming Guide
└── other-documents.pdf   # Additional documents
```

### 3. **Run the Application**
```bash
python app.py
```

The application will start on `http://localhost:5000`

## Project Structure

```
chat/
├── app.py                 # Main Flask application
├── ChatbotV1.py          # Chat functionality and RAG system
├── embedding.py           # PDF processing and vector search
├── voicebot.py           # Voice interaction capabilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── static/
│   └── index.html       # Modern web interface
└── data/
    └── cudabook.pdf     # CUDA documentation
```

## How It Works

### **Document Processing Pipeline**
1. **PDF Loading** - Extracts text from PDF documents
2. **Text Chunking** - Splits into manageable chunks (1000 chars with 100 char overlap)
3. **Embedding Generation** - Creates vector representations using local AI
4. **FAISS Indexing** - Builds searchable vector database
5. **Query Processing** - Searches for relevant chunks based on user questions

### **Response Generation**
1. **Query Analysis** - Processes user questions
2. **Context Retrieval** - Finds most relevant document sections
3. **Response Synthesis** - Combines context with AI knowledge
4. **Relevance Scoring** - Shows confidence in retrieved information

## Example Queries

The chatbot is specialized in CUDA programming and can help with:

- **Virtual Memory Management** - Memory allocation and management
- **Kernel Optimization** - Performance tuning and best practices
- **Memory Hierarchy** - Understanding GPU memory structure
- **Streams & Events** - Asynchronous processing
- **Error Handling** - Debugging and troubleshooting
- **Performance Analysis** - Profiling and optimization

## Configuration

### **Environment Variables**
Create a `.env` file for optional configurations:
```env
# Optional: For additional AI services
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### **Customization Options**
- **Chunk Size**: Modify in `ChatbotV1.py` for different document processing
- **Embedding Model**: Change the sentence transformer model
- **Search Results**: Adjust the number of retrieved chunks
- **UI Theme**: Customize colors and styling in `static/index.html`

## Performance

### **Current Capabilities**
- **Document Size**: Handles large PDFs (596+ pages)
- **Chunk Processing**: 2180+ text chunks from CUDA guide
- **Search Speed**: Sub-second response times
- **Memory Usage**: Efficient local processing
- **Concurrent Users**: Flask handles multiple connections

### **System Requirements**
- **Python**: 3.8+
- **RAM**: 4GB+ recommended for large documents
- **Storage**: 2GB+ for models and documents
- **GPU**: Optional, CPU processing works well

## Troubleshooting

### **Common Issues**

#### **Document Loading Errors**
```bash
 Error: Could not find 'data/cudabook.pdf'
```
**Solution**: Ensure PDF files exist in the `data/` directory

#### **Import Errors**
```bash
ModuleNotFoundError: No module named 'sentence_transformers'
```
**Solution**: Run `pip install -r requirements.txt`

#### **Memory Issues**
```bash
OutOfMemoryError: CUDA out of memory
```
**Solution**: Reduce chunk size or use CPU-only processing

#### **Frontend Not Loading**
- Clear browser cache (Ctrl+F5)
- Check if Flask server is running
- Verify port 5000 is not blocked

### **Performance Optimization**
- **Reduce chunk size** for faster processing
- **Use smaller embedding model** for memory efficiency
- **Enable GPU acceleration** if available
- **Optimize FAISS index** for your use case

## Future Enhancements

### **Planned Features**
- [ ] **Multi-document Support** - Load multiple PDFs
- [ ] **Voice Interface** - Speech-to-text and text-to-speech
- [ ] **Code Highlighting** - Syntax highlighting for code examples
- [ ] **Export Functionality** - Save conversations and responses
- [ ] **User Authentication** - Multi-user support
- [ ] **Advanced Search** - Filters and advanced queries

### **Technical Improvements**
- [ ] **Caching System** - Redis for faster responses
- [ ] **Database Integration** - PostgreSQL for conversation history
- [ ] **API Documentation** - Swagger/OpenAPI specs
- [ ] **Docker Support** - Containerized deployment
- [ ] **Monitoring** - Prometheus metrics and logging

## Contributing

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **CUDA Programming Guide** - NVIDIA Corporation
- **Sentence Transformers** - Hugging Face
- **FAISS** - Facebook Research
- **Flask** - Pallets Project
- **Font Awesome** - Icons and UI elements

---

**For the CUDA programming community**

*For questions, issues, or contributions, please open an issue on GitHub.*
