import os
import io
import base64
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import google.generativeai as genai
from pathlib import Path

from .config import Config

class DocumentProcessor:
    def __init__(self):
        self.config = Config
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Initialize Gemini 1.5 Flash
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=Config.GEMINI_API_KEY)
        try:
            self.gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)
            print(f"Initialized Gemini model: {Config.GEMINI_MODEL}")
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            raise
        
        Config.create_directories()

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a PDF file and extract text and visual information."""
        pages = self._extract_pages(pdf_path)
        processed_pages = []
        
        for page_num, page in enumerate(tqdm(pages, desc="Processing PDF pages")):
            # Extract text
            text = page.get_text()
            
            # Extract images and get descriptions
            image_descriptions = self._extract_and_describe_images(page, page_num)
            
            # Combine text and image descriptions
            page_content = {
                'page_number': page_num + 1,
                'text': text,
                'image_descriptions': image_descriptions,
                'full_content': f"Page {page_num + 1}\n\n{text}"
            }
            
            if image_descriptions:
                page_content['full_content'] += "\n\nVisual Content:\n" + "\n".join(image_descriptions)
            
            processed_pages.append(page_content)
        
        return processed_pages

    def _extract_pages(self, pdf_path: str) -> List[Any]:
        """Extract pages from a PDF file."""
        doc = fitz.open(pdf_path)
        return [doc.load_page(i) for i in range(len(doc))]

    def _extract_and_describe_images(self, page: Any, page_num: int) -> List[str]:
        """Extract images from a page and get descriptions using GPT-4 Vision."""
        image_list = page.get_images(full=True)
        if not image_list:
            return []
            
        descriptions = []
        for img_index, img in enumerate(image_list):
            try:
                # Extract image
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image temporarily
                image_path = os.path.join(
                    self.config.IMAGES_DIR,
                    f"page_{page_num + 1}_img_{img_index + 1}.png"
                )
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Get description from GPT-4 Vision
                description = self._get_image_description(image_path)
                descriptions.append(description)
                
                # Clean up
                os.remove(image_path)
                
            except Exception as e:
                print(f"Error processing image {img_index} on page {page_num + 1}: {str(e)}")
                continue
                
        return descriptions

    def _get_image_description(self, image_path: str) -> str:
        """Get description of an image using Gemini 1.5 Flash."""
        if not hasattr(self, 'gemini_model') or not self.gemini_model:
            return "[Image processing not available]"
            
        try:
            # Read and resize image to reduce processing requirements
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if image is too large
            max_size = (2048, 2048)  # Higher resolution for better OCR
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=90)
            img_byte_arr = img_byte_arr.getvalue()
            
            # Prepare the prompt
            prompt = """
            Please analyze this image in detail and provide a comprehensive description that includes:
            1. Any visible text (transcribe it exactly as shown)
            2. A description of any charts, graphs, or diagrams
            3. The overall purpose and content of the image
            4. Any important details that would be relevant for search and retrieval
            
            Be thorough and precise in your description.
            """
            
            # Create the message parts
            message = [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(img_byte_arr).decode('utf-8')
                    }
                }
            ]
            
            # Generate the response
            response = self.gemini_model.generate_content(
                contents=message,
                generation_config={
                    'max_output_tokens': 2048,
                    'temperature': 0.2,
                }
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini model")
                
            return response.text.strip()
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return f"[Image description error: {str(e)}]"

    def chunk_document(self, processed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split document into smaller chunks for processing."""
        chunks = []
        chunk_id = 0
        
        for page in processed_pages:
            text = page['full_content']
            # Simple chunking by characters, can be improved with better text splitting
            for i in range(0, len(text), self.config.CHUNK_SIZE - self.config.CHUNK_OVERLAP):
                chunk = text[i:i + self.config.CHUNK_SIZE]
                chunks.append({
                    'chunk_id': chunk_id,
                    'page_number': page['page_number'],
                    'content': chunk,
                    'has_images': len(page.get('image_descriptions', [])) > 0
                })
                chunk_id += 1
                
        return chunks
