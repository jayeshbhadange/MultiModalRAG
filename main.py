import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.query_processor import QueryProcessor
from src.config import Config

def process_document(file_path: str):
    """Process a document and add it to the vector store."""
    print(f"Processing document: {file_path}")
    
    # Initialize components
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Process the PDF
    print("Extracting text and visual content...")
    processed_pages = doc_processor.process_pdf(file_path)
    
    # Chunk the document
    print("Chunking document...")
    chunks = doc_processor.chunk_document(processed_pages)
    
    # Add to vector store
    print(f"Adding {len(chunks)} chunks to vector store...")
    vector_store.upsert_documents(chunks)
    print("Document processing complete!")

def query_system(query: str, use_vision: bool = False):
    """Query the RAG system."""
    print(f"\nProcessing query: {query}")
    
    # Initialize components
    vector_store = VectorStore()
    query_processor = QueryProcessor(vector_store)
    
    # Get response
    response = query_processor.generate_response(query, use_vision=use_vision)
    
    # Format and display response
    print("\n" + "="*80)
    print(query_processor.format_response(response))
    print("="*80 + "\n")

def main():
    # Load environment variables
    load_dotenv()
    
    # Create necessary directories
    Config.create_directories()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Multimodal RAG System with Pinecone and OpenAI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a document')
    process_parser.add_argument('file_path', type=str, help='Path to the PDF file to process')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('query', type=str, help='Your question')
    query_parser.add_argument('--vision', action='store_true', help='Include visual context in the response')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the vector store')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        if not os.path.exists(args.file_path):
            print(f"Error: File not found: {args.file_path}")
            return
        process_document(args.file_path)
    
    elif args.command == 'query':
        query_system(args.query, use_vision=args.vision)
    
    elif args.command == 'clear':
        vector_store = VectorStore()
        vector_store.delete_all_vectors()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
