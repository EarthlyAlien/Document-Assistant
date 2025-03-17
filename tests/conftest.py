import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
from vector_store import VectorStore
from rag import RAG

class MockDocument:
    def __init__(self, page_content="Test content", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        MockDocument("Document 1"),
        MockDocument("Document 2"),
        MockDocument("Document 3")
    ]

@pytest.fixture
def mock_pdf_file():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(b"%PDF-1.3\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/Resources <<\n/Font <<\n/F1 4 0 R\n>>\n>>\n/MediaBox [0 0 612 792]\n/Contents 5 0 R\n>>\nendobj\n4 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\nendobj\n5 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n72 720 Td\n(Test PDF) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000015 00000 n\n0000000061 00000 n\n0000000117 00000 n\n0000000217 00000 n\n0000000285 00000 n\ntrailer\n<<\n/Size 6\n/Root 1 0 R\n>>\nstartxref\n384\n%%EOF\n")
        return f.name

@pytest.fixture
def document_processor():
    """Create a DocumentProcessor instance for testing."""
    from document_processor import DocumentProcessor
    return DocumentProcessor()

@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing."""
    with patch('vector_store.SentenceTransformer', autospec=True) as mock:
        instance = MagicMock()
        mock.return_value = instance
        instance.encode = MagicMock(return_value=np.random.rand(3, 384).astype(np.float32))
        instance.get_sentence_embedding_dimension = MagicMock(return_value=384)
        yield mock

@pytest.fixture
def vector_store(mock_sentence_transformer):
    """Create a VectorStore instance for testing."""
    return VectorStore()

@pytest.fixture
def mock_openai():
    """Create a mock ChatOpenAI for testing."""
    with patch('langchain_openai.ChatOpenAI', autospec=True) as mock:
        instance = mock.return_value
        mock_response = MagicMock()
        mock_response.content = "This is a mock response from the language model."
        instance.invoke = MagicMock(return_value=mock_response)
        yield mock

@pytest.fixture
def rag_instance(vector_store, mock_openai):
    """Create a RAG instance for testing."""
    return RAG(vector_store) 