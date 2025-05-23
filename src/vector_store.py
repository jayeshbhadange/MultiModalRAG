import os
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import pinecone
from tqdm import tqdm

from .config import Config

class VectorStore:
    def __init__(self):
        self.config = Config
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Initialize Gemini
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=Config.GEMINI_API_KEY)
        try:
            self.gemini_embedding_model = genai.GenerativeModel(Config.GEMINI_EMBEDDING_MODEL)
        except Exception as e:
            print(f"Warning: Could not initialize Gemini embedding model: {str(e)}")
            self.gemini_embedding_model = None
        
        # Initialize Pinecone
        self.pinecone_client = self._init_pinecone()
        self.index = self._get_or_create_index()

    def _init_pinecone(self) -> pinecone.Pinecone:
        """Initialize Pinecone client."""
        return pinecone.Pinecone(api_key=self.config.PINECONE_API_KEY)

    def _get_or_create_index(self) -> pinecone.Index:
        """Get existing index or create a new one if it doesn't exist."""
        index_name = self.config.INDEX_NAME
        
        try:
            # List all indexes
            existing_indexes = self.pinecone_client.list_indexes()
            index_names = [index.name for index in existing_indexes] if existing_indexes else []
            
            if index_name in index_names:
                # Check if existing index has the correct dimension
                index_info = self.pinecone_client.describe_index(index_name)
                if hasattr(index_info, 'dimension') and index_info.dimension != self.config.GEMINI_EMBEDDING_DIM:
                    print(f"Warning: Existing index dimension {index_info.dimension} does not match required dimension {self.config.GEMINI_EMBEDDING_DIM}")
                    print("Deleting and recreating index...")
                    self.pinecone_client.delete_index(index_name)
                    existing_indexes = [idx for idx in existing_indexes if idx.name != index_name]
                    index_names.remove(index_name)
            
            if index_name not in index_names:
                # Create a new index with the correct dimension
                print(f"Creating new index '{index_name}' with dimension {self.config.GEMINI_EMBEDDING_DIM}")
                self.pinecone_client.create_index(
                    name=index_name,
                    dimension=self.config.GEMINI_EMBEDDING_DIM,
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                while not self.pinecone_client.describe_index(index_name).status.ready:
                    time.sleep(1)
                print("Index is ready!")
            
            # Connect to the index
            return self.pinecone_client.Index(index_name)
            
        except Exception as e:
            print(f"Error managing Pinecone index: {str(e)}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using Google's Gemini embedding model."""
        try:
            # Truncate text if it's too long (Gemini has token limits)
            if len(text) > 10000:  # Approximate character limit
                text = text[:10000]
            
            # Generate embedding using the genai client directly
            result = genai.embed_content(
                model=self.config.GEMINI_EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            
            # Extract the embedding values
            if 'embedding' in result:
                return result['embedding']
            else:
                raise ValueError("No embedding found in the response")
                
        except Exception as e:
            print(f"Error generating Gemini embedding: {str(e)}")
            # Fallback to random embeddings in case of error
            import random
            return [random.random() for _ in range(self.config.GEMINI_EMBEDDING_DIM)]

    def upsert_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Upsert documents to the vector store."""
        vectors = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            # Generate embedding for the document content
            embedding = self.get_embedding(doc['content'])
            
            # Create a vector with metadata
            vector = {
                'id': str(doc['chunk_id']),
                'values': embedding,
                'metadata': {
                    'page_number': doc['page_number'],
                    'has_images': doc.get('has_images', False),
                    'content': doc['content'][:500]  # Store first 500 chars as metadata
                }
            }
            vectors.append(vector)
            
            # Upsert in batches
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                vectors = []
        
        # Upsert any remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors)

    def search(self, query: str, top_k: int = 5, filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents."""
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        try:
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_conditions,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata or {},
                    'content': (match.metadata or {}).get('content', '')
                })
                
            return formatted_results
            
        except Exception as e:
            print(f"Error searching index: {str(e)}")
            return []

    def delete_all_vectors(self) -> None:
        """Delete all vectors from the index."""
        try:
            # Get index stats
            index_stats = self.index.describe_index_stats()
            total_vectors = index_stats.total_vector_count
            
            if total_vectors > 0:
                # If there are vectors, delete them
                self.index.delete(delete_all=True)
                print(f"Deleted all vectors from index '{self.config.INDEX_NAME}'")
            else:
                print("No vectors to delete")
        except Exception as e:
            print(f"Error deleting vectors: {str(e)}")
