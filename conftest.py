import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag import RAG

class MockDocument:
    """Mock LangChain document for testing."""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


@pytest.fixture
def sample_documents() -> List[MockDocument]:
    """Create sample documents for testing."""
    return [
        MockDocument(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test1.pdf", "page": 1}
        ),
        MockDocument(
            page_content="FAISS is a library for efficient similarity search.",
            metadata={"source": "test1.pdf", "page": 2}
        ),
        MockDocument(
            page_content="RAG combines retrieval with generative models.",
            metadata={"source": "test2.pdf", "page": 1}
        ),
    ]


@pytest.fixture
def mock_pdf_file():
    """Create a temporary mock PDF file for testing."""
    # Create a simple PDF-like content
    content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    yield temp_file_path
    
    # Clean up the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


@pytest.fixture
def document_processor():
    """Create a DocumentProcessor instance for testing."""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing."""
    with patch('sentence_transformers.SentenceTransformer', autospec=True) as mock:
        # Configure the mock
        instance = mock.return_value
        instance.get_sentence_embedding_dimension.return_value = 384
        instance.encode.return_value = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384,
        ]
        yield mock


@pytest.fixture
def vector_store(mock_sentence_transformer):
    """Create a VectorStore instance with mocked embeddings for testing."""
    return VectorStore()


@pytest.fixture
def mock_openai():
    """Create a mock OpenAI client for testing."""
    with patch('langchain.chat_models.ChatOpenAI', autospec=True) as mock:
        # Configure the mock response
        instance = mock.return_value
        
        response = MagicMock()
        response.content = "This is a mock response from the language model."
        
        instance.invoke.return_value = response
        yield mock


@pytest.fixture
def rag_instance(vector_store, mock_openai):
    """Create a RAG instance with a mocked vector store and language model."""
    return RAG(vector_store) 