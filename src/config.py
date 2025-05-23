import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIM = 3072  # Dimension for text-embedding-3-large
    VISION_MODEL = "gpt-4o"
    
    # Google Gemini 1.5 Flash (single model for both text and vision)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-1.5-flash"  # Single model for both text and vision
    
    # Embedding configuration
    GEMINI_EMBEDDING_MODEL = "models/embedding-001"  # Default embedding model
    GEMINI_EMBEDDING_DIM = 768  # Dimension for Gemini embedding-001 model
    
    # Pinecone index settings - must match the embedding dimension
    PINECONE_INDEX_DIMENSION = GEMINI_EMBEDDING_DIM  # This ensures consistency
    
    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
    INDEX_NAME = "multimodal-rag-index"
    
    # File paths
    DATA_DIR = "data"
    DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    
    # Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(cls.IMAGES_DIR, exist_ok=True)
