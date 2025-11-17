"""Text Chunking Utilities using LangChain Text Splitters."""
from typing import List, Dict, Any
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from src.utils.logger import setup_logger


class TextChunker:
    """Handle text chunking for large documents using LangChain."""
    
    def __init__(self):
        """Initialize the text chunker."""
        self.logger = setup_logger(__name__)
        
        # Recursive Character Text Splitter (Best for most use cases)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Character Text Splitter (Simple splitting)
        self.character_splitter = CharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separator="\n"
        )
        
        # Token Text Splitter (For token-based limits)
        self.token_splitter = TokenTextSplitter(
            chunk_size=1500,  # Safe limit for embeddings (8192 token limit)
            chunk_overlap=150
        )
        
        self.logger.info("Text chunker initialized with LangChain splitters")
    
    def chunk_text_recursive(self, text: str) -> List[str]:
        """Split text using recursive character splitter (recommended).
        
        This method tries to split on paragraphs first, then sentences, then words.
        Best for maintaining semantic coherence.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        try:
            chunks = self.recursive_splitter.split_text(text)
            self.logger.info(f"Split text into {len(chunks)} chunks using recursive splitter")
            return chunks
        except Exception as e:
            self.logger.error(f"Error in recursive chunking: {e}")
            return [text]
    
    def chunk_text_by_character(self, text: str) -> List[str]:
        """Split text using character splitter.
        
        Simple splitting by character count with specified separator.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        try:
            chunks = self.character_splitter.split_text(text)
            self.logger.info(f"Split text into {len(chunks)} chunks using character splitter")
            return chunks
        except Exception as e:
            self.logger.error(f"Error in character chunking: {e}")
            return [text]
    
    def chunk_text_by_tokens(self, text: str) -> List[str]:
        """Split text using token splitter.
        
        Splits based on token count to ensure chunks fit within model limits.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        try:
            chunks = self.token_splitter.split_text(text)
            self.logger.info(f"Split text into {len(chunks)} chunks using token splitter")
            return chunks
        except Exception as e:
            self.logger.error(f"Error in token chunking: {e}")
            return [text]
    
    def chunk_documents(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text and return with metadata.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        try:
            chunks = self.chunk_text_recursive(text)
            
            result = []
            for idx, chunk in enumerate(chunks):
                chunk_data = {
                    'text': chunk,
                    'chunk_index': idx,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                }
                
                if metadata:
                    chunk_data['metadata'] = metadata
                
                result.append(chunk_data)
            
            self.logger.info(f"Created {len(result)} document chunks with metadata")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in document chunking: {e}")
            return [{'text': text, 'chunk_index': 0, 'total_chunks': 1, 'chunk_size': len(text)}]
    
    def get_chunk_stats(self, chunks: List[str]) -> Dict[str, Any]:
        """Get statistics about chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'avg_chunk_size': sum(chunk_sizes) // len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }

