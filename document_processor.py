import os
from typing import List, cast
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF document and split it into chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        loader = PyPDFLoader(file_path)
        try:
            documents = loader.load()
            
            # Add filename to metadata
            filename = os.path.basename(file_path)
            for doc in documents:
                doc.metadata["source"] = filename
                
            return documents
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        return self.text_splitter.split_documents(documents)
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a document: load and chunk it.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of processed document chunks
        """
        documents = self.load_pdf(file_path)
        return self.chunk_documents(documents) 