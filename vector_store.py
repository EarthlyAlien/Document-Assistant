import os
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, cast
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

class VectorStore:
    """Handles document embeddings and vector search using FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents: List[Document] = []
        except Exception as e:
            raise Exception(f"Error initializing vector store: {str(e)}")
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
            
        try:
            texts = [doc.page_content for doc in documents if isinstance(doc, Document)]
            embeddings = self.model.encode(texts)
            
            # Add documents to the store
            self.documents.extend(documents)
            
            # Add embeddings to the index
            if len(embeddings) > 0:
                self.index.add(np.array(embeddings).astype('float32'))
        except Exception as e:
            raise Exception(f"Error adding documents: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search for the query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.documents:
            return []
            
        try:
            # Get query embedding
            query_embedding = self.model.encode([query])
            
            # Search the index
            D, I = self.index.search(np.array(query_embedding).astype('float32'), k)
            
            # Return the top k documents
            results: List[Document] = []
            for idx in I[0]:
                if idx < len(self.documents):
                    results.append(self.documents[idx])
            return results
        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")
    
    def save_index(self, path: str) -> None:
        """
        Save the FAISS index to disk.
        
        Args:
            path: Path to save the index
        """
        faiss.write_index(self.index, path)
        
    def load_index(self, path: str) -> None:
        """
        Load a FAISS index from disk.
        
        Args:
            path: Path to the saved index
        """
        if os.path.exists(path):
            self.index = faiss.read_index(path) 