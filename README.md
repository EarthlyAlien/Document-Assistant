# Document Assistant

A powerful document analysis tool that uses Retrieval Augmented Generation (RAG) to provide intelligent answers to questions about your documents.

## Author

**Chaitanya Vankadaru**  
AI/ML Engineer | Python Developer | Data Scientist  
[LinkedIn Profile](https://www.linkedin.com/in/chaitanyavankadaru)

## Features

- 📄 PDF Document Processing
- 🔍 Advanced Text Chunking
- 🧠 Vector Embeddings with FAISS
- 💡 Intelligent Question Answering
- 🤖 RAG Architecture
- 🎨 Modern, Interactive UI

## Technology Stack

- Python 3.12
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- OpenAI GPT
- PyPDF

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EarthlyAlien/Document-Assistant/
cd Document-Assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload a PDF document using the sidebar
2. Adjust chunking settings if needed
3. Process the document
4. Ask questions about the document content
5. View conversation history and system statistics

## Architecture

The application uses a Retrieval Augmented Generation (RAG) architecture:

1. **Document Processing**: Documents are chunked into smaller segments
2. **Vector Embeddings**: Chunks are converted to vector embeddings
3. **FAISS Index**: Embeddings are stored in a FAISS vector index
4. **Semantic Search**: User queries retrieve the most relevant chunks
5. **Generation**: Retrieved context is sent to an LLM to generate answers

## Development

For development, install the development dependencies:
```
pip install -r requirements-dev.txt
```

### Running Tests

Run the test suite with:
```
pytest
```

For test coverage report:
```
pytest --cov=.
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or opportunities, please reach out on [LinkedIn](https://www.linkedin.com/in/chaitanyavankadaru). 
