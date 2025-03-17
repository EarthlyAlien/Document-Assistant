# Document Assistant

A powerful document analysis tool that uses Retrieval Augmented Generation (RAG) to provide intelligent answers to questions about your documents.

## Author

**Chaitanya Vankadaru**  
AI/ML Engineer | Python Developer | Data Scientist  
[LinkedIn Profile](https://www.linkedin.com/in/chaitanyavankadaru)

## Features

- 📄 **PDF Document Processing**: Advanced PDF parsing and text extraction
- 🔍 **Smart Text Chunking**: Intelligent document segmentation with customizable settings
- 🧠 **Vector Embeddings**: State-of-the-art embeddings using Sentence Transformers
- 💾 **FAISS Vector Store**: Fast and efficient similarity search
- 🤖 **RAG Architecture**: Enhanced question answering using document context
- 🎨 **Modern UI**: Clean, responsive interface with Streamlit
- 📊 **System Statistics**: Real-time performance metrics
- 🔄 **Conversation History**: Track and review Q&A interactions
- ⚙️ **Customizable Settings**: Adjust chunk size and overlap

## Technology Stack

- Python 3.12
- Streamlit (>=1.37.0)
- LangChain (>=0.2.5)
- FAISS-CPU (>=1.7.4)
- Sentence Transformers (>=2.2.2)
- OpenAI GPT (>=1.6.1)
- PyPDF (>=3.17.0)

## Prerequisites

- Python 3.12 or higher
- OpenAI API key
- Git (for version control)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EarthlyAlien/Document-Assistant.git
cd Document-Assistant
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
# For production
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Document Upload**
   - Use the sidebar to upload PDF documents
   - View uploaded document list
   - Clear documents when needed

2. **Configuration**
   - Adjust chunk size (default: 1000)
   - Set chunk overlap (default: 200)
   - Configure these based on document length and complexity

3. **Processing**
   - Click "Process Document" to extract text and generate embeddings
   - Monitor processing status in real-time

4. **Question Answering**
   - Enter questions about your documents
   - View AI-generated responses with source context
   - Track conversation history

## Architecture

The Document Assistant uses a sophisticated RAG (Retrieval Augmented Generation) architecture:

1. **Document Processing**
   - PDF parsing and text extraction
   - Intelligent text chunking with overlap
   - Clean text preprocessing

2. **Vector Store**
   - Chunk embedding generation using Sentence Transformers
   - FAISS vector index for efficient similarity search
   - Persistent storage of embeddings

3. **Question Answering**
   - Query embedding and semantic search
   - Context retrieval from vector store
   - LLM-powered answer generation with context

## Development

For development work:

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Development tools available:
   - pytest (>=7.4.4): Testing framework
   - pytest-cov (>=4.1.0): Code coverage
   - flake8 (>=7.0.0): Code linting
   - mypy (>=1.8.0): Static type checking
   - black (>=24.2.0): Code formatting

3. Run tests:
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=.

# Run with verbose output
pytest -v
```

4. Code formatting:
```bash
# Format code
black .

# Check code style
flake8 .

# Type checking
mypy .
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

- Regular dependency updates
- Security vulnerability monitoring
- Safe API key handling
- Input validation and sanitization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Chaitanya Vankadaru
- **LinkedIn**: [Profile](https://www.linkedin.com/in/chaitanyavankadaru)
- **GitHub**: [EarthlyAlien](https://github.com/EarthlyAlien)

## Acknowledgments

- OpenAI for GPT API
- Streamlit for the UI framework
- FAISS for vector similarity search
- Sentence Transformers for embeddings 
