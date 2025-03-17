import os
import pytest
from unittest.mock import patch, MagicMock

from document_processor import DocumentProcessor


class TestDocumentProcessor:
    
    def test_init(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
        assert processor.text_splitter is not None
    
    @patch('document_processor.PyPDFLoader')
    def test_load_pdf(self, mock_loader, mock_pdf_file):
        """Test loading a PDF file."""
        # Setup mock
        mock_instance = MagicMock()
        mock_loader.return_value = mock_instance
        
        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {}
        mock_instance.load.return_value = [mock_doc1, mock_doc2]
        
        # Test loading PDF
        processor = DocumentProcessor()
        docs = processor.load_pdf(mock_pdf_file)
        
        # Verify loader was created and used
        mock_loader.assert_called_once_with(mock_pdf_file)
        mock_instance.load.assert_called_once()
        
        # Verify metadata was updated
        assert len(docs) == 2
        for doc in docs:
            assert "source" in doc.metadata
            assert doc.metadata["source"] == os.path.basename(mock_pdf_file)
    
    def test_load_pdf_file_not_found(self):
        """Test loading a non-existent PDF file."""
        processor = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            processor.load_pdf("non_existent_file.pdf")
    
    def test_chunk_documents(self, sample_documents):
        """Test chunking documents."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        # Use a mock for the text splitter to control the output
        processor.text_splitter = MagicMock()
        processor.text_splitter.split_documents.return_value = [
            MagicMock(page_content="Chunk 1", metadata={"source": "test.pdf", "page": 1}),
            MagicMock(page_content="Chunk 2", metadata={"source": "test.pdf", "page": 1}),
        ]
        
        chunks = processor.chunk_documents(sample_documents)
        
        # Assertions
        processor.text_splitter.split_documents.assert_called_once_with(sample_documents)
        assert len(chunks) == 2
    
    @patch.object(DocumentProcessor, 'load_pdf')
    @patch.object(DocumentProcessor, 'chunk_documents')
    def test_process_document(self, mock_chunk, mock_load, mock_pdf_file):
        """Test the full document processing pipeline."""
        # Setup mocks
        mock_docs = [MagicMock(), MagicMock()]
        mock_chunks = [MagicMock(), MagicMock(), MagicMock()]
        
        mock_load.return_value = mock_docs
        mock_chunk.return_value = mock_chunks
        
        # Process document
        processor = DocumentProcessor()
        result = processor.process_document(mock_pdf_file)
        
        # Assertions
        mock_load.assert_called_once_with(mock_pdf_file)
        mock_chunk.assert_called_once_with(mock_docs)
        assert result == mock_chunks 