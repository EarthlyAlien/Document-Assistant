# Document Assistant

A document processing system that extracts information from unstructured documents using a RAG (Retrieval Augmented Generation) architecture with chunking and FAISS vector search.

## Features

- Document upload and processing
- Text chunking for efficient processing
- Vector embeddings using sentence transformers
- Semantic search with FAISS
- RAG-based question answering on documents

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Architecture

This application implements a RAG (Retrieval Augmented Generation) architecture:
- Documents are processed and chunked into manageable segments
- Chunks are converted to vector embeddings and stored in a FAISS index
- User queries are converted to embeddings and used to retrieve relevant document chunks
- Retrieved context is sent to an LLM to generate accurate responses

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

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

When using this code in your projects, attribution is required. Please include the copyright notice and permission notice in your project. 