import pytest
from unittest.mock import patch, MagicMock

from rag import RAG
from langchain.schema import HumanMessage, SystemMessage


class TestRAG:
    
    @patch('rag.ChatOpenAI')
    def test_init(self, mock_openai, vector_store):
        """Test RAG initialization."""
        # Create a new RAG instance
        rag = RAG(vector_store)
        
        # Verify vector store was set
        assert rag.vector_store == vector_store
        
        # Verify ChatOpenAI was initialized correctly
        mock_openai.assert_called_once_with(model_name="gpt-3.5-turbo")
    
    def test_generate_answer_no_docs(self, rag_instance):
        """Test generate_answer when no documents are found."""
        # Mock the vector store to return no documents
        rag_instance.vector_store.similarity_search = MagicMock(return_value=[])
        
        # Generate answer
        answer = rag_instance.generate_answer("What is artificial intelligence?")
        
        # Verify similarity search was called
        rag_instance.vector_store.similarity_search.assert_called_once_with(
            "What is artificial intelligence?", k=4
        )
        
        # Verify the response when no documents are found
        assert "No relevant information found" in answer
    
    def test_generate_answer(self, rag_instance, sample_documents):
        """Test generate_answer with documents."""
        # Mock the vector store to return documents
        rag_instance.vector_store.similarity_search = MagicMock(return_value=sample_documents)
        
        # Mock the model's response
        mock_response = MagicMock()
        mock_response.content = "Test response"
        rag_instance.model = MagicMock()
        rag_instance.model.invoke.return_value = mock_response
        
        # Generate answer
        query = "What is artificial intelligence?"
        answer = rag_instance.generate_answer(query, k=2)
        
        # Verify similarity search was called
        rag_instance.vector_store.similarity_search.assert_called_once_with(query, k=2)
        
        # Verify LLM invocation
        assert rag_instance.model.invoke.call_count == 1
        
        # Verify response was returned
        assert answer == "Test response"
    
    @patch('dotenv.load_dotenv')
    def test_dotenv_loaded(self, mock_load_dotenv):
        """Test that environment variables are loaded."""
        # Import the module again to trigger load_dotenv
        import importlib
        import rag as rag_module
        importlib.reload(rag_module)
        
        # Verify load_dotenv was called
        mock_load_dotenv.assert_called_once() 