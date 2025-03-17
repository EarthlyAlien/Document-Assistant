import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import tempfile
import os

# Import the app module
import app


@pytest.fixture
def mock_streamlit():
    """Mock streamlit components."""
    with patch('streamlit.title') as mock_title, \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.file_uploader') as mock_uploader, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.slider') as mock_slider, \
         patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.spinner') as mock_spinner, \
         patch('streamlit.session_state') as mock_state:
        
        # Configure mock columns
        col1 = MagicMock()
        col2 = MagicMock()
        mock_columns.return_value = [col1, col2]
        
        # Configure session state
        mock_state.vector_store = MagicMock()
        mock_state.uploaded_files = []
        mock_state.processor = MagicMock()
        mock_state.rag = MagicMock()
        mock_state.chat_history = []
        
        yield {
            'title': mock_title,
            'write': mock_write,
            'sidebar': mock_sidebar,
            'columns': mock_columns,
            'file_uploader': mock_uploader,
            'button': mock_button,
            'slider': mock_slider,
            'text_input': mock_text_input,
            'warning': mock_warning,
            'spinner': mock_spinner,
            'session_state': mock_state,
            'col1': col1,
            'col2': col2
        }


class TestApp:
    
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    def test_process_document(self, mock_unlink, mock_temp_file, mock_streamlit):
        """Test the process_document function."""
        # Setup mock uploaded file
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "test.pdf"
        mock_uploaded_file.size = 1024
        
        # Setup mock temp file
        mock_file = MagicMock()
        mock_temp_file.return_value.__enter__.return_value = mock_file
        mock_file.name = "/tmp/test.pdf"
        
        # Setup mock processor and vector store
        mock_chunks = [MagicMock(), MagicMock()]
        mock_streamlit['session_state'].processor.process_document.return_value = mock_chunks
        
        # Call the function
        app.process_document(mock_uploaded_file)
        
        # Verify temporary file was created and written to
        mock_temp_file.assert_called_once()
        mock_file.write.assert_called_once_with(mock_uploaded_file.getvalue())
        
        # Verify document was processed
        mock_streamlit['session_state'].processor.process_document.assert_called_once_with(mock_file.name)
        
        # Verify chunks were added to vector store
        mock_streamlit['session_state'].vector_store.add_documents.assert_called_once_with(mock_chunks)
        
        # Verify file was added to uploaded files list
        assert len(mock_streamlit['session_state'].uploaded_files) == 1
        assert mock_streamlit['session_state'].uploaded_files[0]["name"] == "test.pdf"
        assert mock_streamlit['session_state'].uploaded_files[0]["size"] == 1024
        assert mock_streamlit['session_state'].uploaded_files[0]["chunks"] == 2
        
        # Verify temp file was cleaned up
        mock_unlink.assert_called_once_with(mock_file.name)
    
    @patch('app.RAG')
    def test_clear_documents(self, mock_rag_class, mock_streamlit):
        """Test the clear_documents function."""
        # Setup initial state
        mock_streamlit['session_state'].uploaded_files = [{"name": "test.pdf"}]
        mock_vector_store = MagicMock()
        mock_rag = MagicMock()
        mock_streamlit['session_state'].vector_store = mock_vector_store
        mock_streamlit['session_state'].rag = mock_rag
        
        # Setup mock RAG class
        mock_rag_instance = MagicMock()
        mock_rag_class.return_value = mock_rag_instance
        
        # Clear documents
        app.clear_documents()
        
        # Verify state was reset
        assert mock_streamlit['session_state'].uploaded_files == []
        
        # Verify new instances were created
        assert mock_streamlit['session_state'].vector_store != mock_vector_store
        assert mock_streamlit['session_state'].rag != mock_rag
        
        # Verify RAG was initialized with the vector store
        mock_rag_class.assert_called_once_with(mock_streamlit['session_state'].vector_store) 