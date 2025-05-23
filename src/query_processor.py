from typing import List, Dict, Any
from openai import OpenAI
import google.generativeai as genai
from tqdm import tqdm

from .config import Config

class QueryProcessor:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Initialize Gemini 1.5 Flash
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=Config.GEMINI_API_KEY)
        try:
            self.gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)
            print(f"Initialized Gemini model for query processing: {Config.GEMINI_MODEL}")
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            raise
        
        self.config = Config

    def generate_response(self, query: str, use_vision: bool = False) -> Dict[str, Any]:
        """
        Generate a response to a user query using RAG.
        
        Args:
            query: The user's query
            use_vision: Whether to include visual context in the response
            
        Returns:
            Dict containing the response and metadata
        """
        # 1. Retrieve relevant documents
        search_results = self.vector_store.search(query, top_k=3)
        
        if not search_results:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'context': []
            }
        
        # 2. Format context for the LLM
        context_parts = []
        sources = []
        
        for i, result in enumerate(search_results):
            source_info = f"[Source {i+1}, Page {result['metadata'].get('page_number', 'N/A')}]"
            context_parts.append(f"{source_info}\n{result['content']}")
            sources.append({
                'page': result['metadata'].get('page_number', 'N/A'),
                'content': result['content'][:500] + '...',
                'has_images': result['metadata'].get('has_images', False)
            })
        
        context = "\n\n".join(context_parts)
        
        # 3. Generate response using GPT-4
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based on the provided context. 
                If the context doesn't contain the answer, say you don't know. 
                Be concise and accurate in your responses."""
            },
            {
                "role": "user",
                "content": f"""Context:
                {context}
                
                Question: {query}
                
                Answer the question based on the context above. If the context doesn't contain the answer, say you don't know."""
            }
        ]
        
        # 4. If visual context is needed, enhance the prompt
        if use_vision and any(src['has_images'] for src in sources):
            messages[0]['content'] += " When describing visual content, be sure to include details from the image descriptions provided in the context."
        
        # 5. Get response from the model using Gemini 1.5 Flash
        if not hasattr(self, 'gemini_model') or not self.gemini_model:
            return {
                'answer': "Error: Gemini model is not available. Please check your API key and model configuration.",
                'sources': sources
            }
            
        try:
            # Format messages for Gemini 1.5 Flash
            # Gemini 1.5 Flash supports both system and user messages in a conversation
            formatted_messages = []
            
            # Add system message if present
            system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
            if system_message:
                formatted_messages.append({
                    'role': 'user',
                    'parts': [{'text': system_message['content']}]
                })
                formatted_messages.append({
                    'role': 'model',
                    'parts': [{'text': 'I understand and will assist with your request.'}]
                })
            
            # Add user message
            user_message = next((msg for msg in messages if msg['role'] == 'user'), None)
            if user_message:
                formatted_messages.append({
                    'role': 'user',
                    'parts': [{'text': user_message['content']}]
                })
            
            # Generate response with Gemini 1.5 Flash
            response = self.gemini_model.generate_content(
                contents={
                    'role': 'user',
                    'parts': [{'text': user_message['content']}]
                },
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 2048,  # Increased for more detailed responses
                    'top_p': 0.95,
                    'top_k': 40
                }
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini model")
                
            answer = response.text.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"Error generating response with Gemini: {error_msg}")
            
            if "quota" in error_msg:
                answer = "I've reached my API quota limit. Please check your Google AI Studio account for usage limits."
            elif "access" in error_msg or "permission" in error_msg:
                answer = "I'm having trouble accessing the AI model. Please verify your API key and permissions."
            else:
                answer = "I'm sorry, I encountered an error while generating a response. The error was: " + str(e)
        
        return {
            'answer': answer,
            'sources': sources,
            'context': context_parts
        }
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """Format the response for display."""
        formatted = f"{response['answer']}\n\n"
        
        if response['sources']:
            formatted += "\nSources:\n"
            for i, source in enumerate(response['sources'], 1):
                formatted += f"\n[{i}] Page {source['page']}\n"
                formatted += f"{source['content']}\n"
                if source['has_images']:
                    formatted += "(Contains visual content)\n"
        
        return formatted
