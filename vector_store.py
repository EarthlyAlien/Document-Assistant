import os
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

class VectorStore:
    """Handles document embeddings and vector search using FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
            
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts)
        
        # Add documents to the store
        self.documents.extend(documents)
        
        # Add embeddings to the index
        if len(embeddings) > 0:
            self.index.add(np.array(embeddings).astype('float32'))
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Perform a similarity search for the query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of document chunks most similar to the query
        """
        if not self.documents:
            return []
            
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Perform search
        k = min(k, len(self.documents))  # Ensure k is not larger than the number of documents
        if k == 0:
            return []
            
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Get the documents corresponding to the indices
        results = [self.documents[idx] for idx in indices[0]]
        
        return results
    
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