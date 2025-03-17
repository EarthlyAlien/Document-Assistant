import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

class RAG:
    """Retrieval Augmented Generation for question answering."""
    
    def __init__(self, vector_store, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the RAG system.
        
        Args:
            vector_store: Vector store instance
            model_name: OpenAI model name
        """
        self.vector_store = vector_store
        self.model = ChatOpenAI(model_name=model_name)
        
    def generate_answer(self, query: str, k: int = 4) -> str:
        """
        Generate an answer to a query using RAG.
        
        Args:
            query: User query
            k: Number of top documents to retrieve
            
        Returns:
            Generated answer
        """
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=k)
        
        if not docs:
            return "No relevant information found. Please try a different question or upload documents."
        
        # Format context from documents
        context = "\n\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}\n{doc.page_content}" for doc in docs])
        
        # Create prompt for the model
        messages = [
            SystemMessage(content=f"You are a helpful assistant that answers questions based on the provided documents. Use only the information in the documents to answer the question. If the answer cannot be found in the documents, say so directly.\n\nDocuments:\n{context}"),
            HumanMessage(content=query)
        ]
        
        # Generate response
        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}" 