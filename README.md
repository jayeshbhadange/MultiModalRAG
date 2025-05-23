# Multimodal RAG with Pinecone and OpenAI Vision

A powerful Retrieval-Augmented Generation (RAG) system that combines text and visual understanding using OpenAI's GPT-4 Vision and Pinecone vector database.

## Features

- **Multimodal Processing**: Extract and understand both text and visual content from documents
- **Efficient Retrieval**: Fast semantic search using Pinecone vector database
- **Flexible Querying**: Support for both text-based and visual-based queries
- **Scalable Architecture**: Designed to handle large document collections
- **Easy Integration**: Simple CLI for processing documents and querying the system

## Prerequisites

- Python 3.8+
- OpenAI API key with access to GPT-4 Vision
- Pinecone API key
- Poppler (for PDF processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multimodal-rag.git
   cd multimodal-rag
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Poppler (required for PDF processing):
   - On macOS: `brew install poppler`
   - On Ubuntu/Debian: `sudo apt-get install poppler-utils`
   - On Windows: Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

4. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

## Usage

### Processing Documents

To process a PDF document:
```bash
python main.py process path/to/your/document.pdf
```

### Querying the System

Ask a question about your documents:
```bash
python main.py query "Your question here"
```

To include visual context in the response (for documents with images):
```bash
python main.py query "Your question about images here" --vision
```

### Clearing the Vector Store

To clear all stored vectors:
```bash
python main.py clear
```

## Project Structure

- `src/`
  - `config.py`: Configuration settings and environment variables
  - `document_processor.py`: Handles PDF processing and text/image extraction
  - `vector_store.py`: Manages interactions with Pinecone
  - `query_processor.py`: Handles query processing and response generation
- `data/`
  - `documents/`: Store your PDFs here
  - `images/`: Temporary storage for extracted images
- `main.py`: Command-line interface

## Example Workflow

1. Add a PDF document to the `data/documents/` directory
2. Process the document:
   ```bash
   python main.py process data/documents/your_document.pdf
   ```
3. Query the system:
   ```bash
   python main.py query "What are the key points from this document?"
   ```
4. For documents with images, ask about visual content:
   ```bash
   python main.py query "Describe the charts in this document" --vision
   ```

## Customization

You can customize various parameters in `src/config.py`:
- Chunk size and overlap for document processing
- Embedding model configuration
- Directory paths
- API settings

## Troubleshooting

- **Missing Poppler**: Ensure Poppler is installed and in your system PATH
- **API Errors**: Verify your API keys are correctly set in the `.env` file
- **Memory Issues**: For large documents, you may need to adjust chunk sizes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for their powerful language and vision models
- Pinecone for the vector database
- The open-source community for various Python libraries used in this project