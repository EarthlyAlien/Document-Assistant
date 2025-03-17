import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from vector_store import VectorStore


class TestVectorStore:
    
    def test_init(self, mock_sentence_transformer):
        """Test VectorStore initialization."""
        # Create a new VectorStore instance
        store = VectorStore()
        
        # Verify SentenceTransformer was initialized correctly
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        assert store.dimension == 384  # From our mocked transformer
        assert len(store.documents) == 0
    
    def test_add_documents_empty(self, vector_store):
        """Test adding empty documents list."""
        initial_doc_count = len(vector_store.documents)
        vector_store.add_documents([])
        assert len(vector_store.documents) == initial_doc_count  # Should remain unchanged
    
    def test_add_documents(self, vector_store, sample_documents):
        """Test adding documents to the vector store."""
        # Get initial document count
        initial_doc_count = len(vector_store.documents)
        
        # Mock the encode method
        vector_store.model.encode = MagicMock(return_value=np.random.rand(3, 384).astype(np.float32))
        
        # Add documents
        vector_store.add_documents(sample_documents)
        
        # Verify documents were added
        assert len(vector_store.documents) == initial_doc_count + len(sample_documents)
        
        # Verify the model was used to encode the documents
        vector_store.model.encode.assert_called_once_with([doc.page_content for doc in sample_documents])
    
    def test_similarity_search_empty(self, vector_store):
        """Test similarity search with no documents."""
        results = vector_store.similarity_search("test query")
        assert len(results) == 0
    
    @patch('faiss.IndexFlatL2.search')
    def test_similarity_search(self, mock_search, vector_store, sample_documents):
        """Test similarity search with documents."""
        # Add documents to the store
        vector_store.documents = sample_documents
        
        # Mock FAISS search results
        mock_search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),  # Distances
            np.array([[0, 2, 1]])          # Indices
        )
        
        # Mock the encode method
        vector_store.model.encode = MagicMock(return_value=np.random.rand(1, 384).astype(np.float32))
        
        # Perform search
        query = "test query"
        results = vector_store.similarity_search(query, k=3)
        
        # Verify query was encoded
        vector_store.model.encode.assert_called_once_with([query])
        
        # Verify search results
        assert len(results) == 3
        assert results[0] == sample_documents[0]  # First result
        assert results[1] == sample_documents[2]  # Second result
        assert results[2] == sample_documents[1]  # Third result
    
    @patch('faiss.write_index')
    def test_save_index(self, mock_write, vector_store):
        """Test saving the FAISS index."""
        path = "test_index.faiss"
        vector_store.save_index(path)
        mock_write.assert_called_once_with(vector_store.index, path)
    
    @patch('os.path.exists')
    @patch('faiss.read_index')
    def test_load_index(self, mock_read, mock_exists, vector_store):
        """Test loading the FAISS index."""
        path = "test_index.faiss"
        mock_exists.return_value = True
        
        # Create a mock index
        mock_index = MagicMock()
        mock_read.return_value = mock_index
        
        # Load the index
        vector_store.load_index(path)
        
        # Verify the index was loaded
        mock_exists.assert_called_once_with(path)
        mock_read.assert_called_once_with(path)
        assert vector_store.index == mock_index
    
    @patch('os.path.exists')
    @patch('faiss.read_index')
    def test_load_index_not_exists(self, mock_read, mock_exists, vector_store):
        """Test loading a non-existent FAISS index."""
        path = "non_existent_index.faiss"
        mock_exists.return_value = False
        
        # Original index
        original_index = vector_store.index
        
        # Load the index
        vector_store.load_index(path)
        
        # Verify functions were called correctly
        mock_exists.assert_called_once_with(path)
        mock_read.assert_not_called()
        
        # Index should remain unchanged
        assert vector_store.index == original_index 