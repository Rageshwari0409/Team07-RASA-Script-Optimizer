"""Milvus vector store for storing and retrieving transcripts."""
import json
import litellm
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.utils.text_chunker import TextChunker
from src.utils.retry_handler import retry_llm_call


class MilvusVectorStore:
    """Manage transcript storage and retrieval using Milvus with chunking support."""

    def __init__(self):
        """Initialize Milvus vector store."""
        self.config = get_config()
        self.logger = setup_logger(__name__)

        # Configure LiteLLM for Azure OpenAI embeddings
        self.api_key = self.config.get('azure_openai.api_key')
        self.api_base = self.config.get('azure_openai.endpoint')
        self.api_version = self.config.get('azure_openai.api_version')
        self.embedding_deployment = self.config.get('embeddings.deployment_name', 'text-embedding-ada-002')

        # Set LiteLLM configuration
        litellm.api_key = self.api_key
        litellm.api_base = self.api_base
        litellm.api_version = self.api_version

        self.collection_name = self.config.get('milvus.collection_name', 'test')
        self.dimension = self.config.get('milvus.dimension', 1536)

        # Initialize text chunker
        self.chunker = TextChunker()

        # Connect to Milvus
        self._connect()

        # Create or load collection
        self._setup_collection()
    
    def _connect(self):
        """Connect to Milvus server."""
        try:
            host = self.config.get('milvus.host', 'localhost')
            port = self.config.get('milvus.port', 19530)
            user = self.config.get('milvus.user', '')
            password = self.config.get('milvus.password', '')
            secure = self.config.get('milvus.secure', False)

            # Build connection parameters
            conn_params = {
                "alias": "default",
                "host": host,
                "port": port
            }

            # Add authentication if provided (for Zilliz Cloud)
            if user and password:
                conn_params["user"] = user
                conn_params["password"] = password

            # Add secure connection if needed (for Zilliz Cloud)
            if secure:
                conn_params["secure"] = True

            connections.connect(**conn_params)
            self.logger.info(f"Connected to Milvus at {host}:{port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_collection(self):
        """Create or load the collection."""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            else:
                # Create new collection
                self._create_collection()
                self.logger.info(f"Created new collection: {self.collection_name}")
            
            # Load collection to memory
            self.collection.load()
            
        except Exception as e:
            self.logger.error(f"Failed to setup collection: {e}")
            raise
    
    def _create_collection(self):
        """Create a new Milvus collection for transcripts."""
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="transcript_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="transcript_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="analysis_result", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="timestamp", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Sales conversation transcripts with embeddings"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create index
        index_params = {
            "metric_type": self.config.get('milvus.metric_type', 'L2'),
            "index_type": self.config.get('milvus.index_type', 'IVF_FLAT'),
            "params": {"nlist": self.config.get('milvus.nlist', 128)}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
    
    @retry_llm_call
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using LiteLLM with Azure OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Truncate text if too long (8192 tokens ≈ 6000 words ≈ 30000 chars)
        # Use a safe limit of 20000 characters to avoid token limit
        max_chars = 20000
        if len(text) > max_chars:
            self.logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} chars for embedding")
            text = text[:max_chars]

        response = litellm.embedding(
            model=f"azure/{self.embedding_deployment}",
            input=[text],
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=self.api_version
        )

        return response.data[0]['embedding']

    def chunk_and_display(self, text: str) -> Dict[str, Any]:
        """Chunk text and display statistics (for demonstration).

        This method demonstrates the chunking functionality using LangChain text splitters.

        Args:
            text: Text to chunk

        Returns:
            Dictionary with chunks and statistics
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("CHUNKING DEMONSTRATION")
            self.logger.info("=" * 80)

            # Method 1: Recursive Character Splitter (Recommended)
            self.logger.info("\n📄 Method 1: Recursive Character Text Splitter")
            self.logger.info("-" * 80)
            recursive_chunks = self.chunker.chunk_text_recursive(text)
            recursive_stats = self.chunker.get_chunk_stats(recursive_chunks)

            self.logger.info(f"✓ Total Chunks: {recursive_stats['total_chunks']}")
            self.logger.info(f"✓ Avg Chunk Size: {recursive_stats['avg_chunk_size']} chars")
            self.logger.info(f"✓ Min/Max Size: {recursive_stats['min_chunk_size']}/{recursive_stats['max_chunk_size']} chars")

            for idx, chunk in enumerate(recursive_chunks[:2], 1):  # Show first 2 chunks
                self.logger.info(f"\nChunk {idx} Preview (first 200 chars):")
                self.logger.info(f"  {chunk[:200]}...")

            # Method 2: Token-based Splitter
            self.logger.info("\n\n📄 Method 2: Token-based Text Splitter")
            self.logger.info("-" * 80)
            token_chunks = self.chunker.chunk_text_by_tokens(text)
            token_stats = self.chunker.get_chunk_stats(token_chunks)

            self.logger.info(f"✓ Total Chunks: {token_stats['total_chunks']}")
            self.logger.info(f"✓ Avg Chunk Size: {token_stats['avg_chunk_size']} chars")
            self.logger.info(f"✓ Min/Max Size: {token_stats['min_chunk_size']}/{token_stats['max_chunk_size']} chars")

            # Method 3: Document Chunks with Metadata
            self.logger.info("\n\n📄 Method 3: Document Chunks with Metadata")
            self.logger.info("-" * 80)
            doc_chunks = self.chunker.chunk_documents(
                text,
                metadata={'source': 'demo', 'type': 'transcript'}
            )

            self.logger.info(f"✓ Total Document Chunks: {len(doc_chunks)}")
            if doc_chunks:
                self.logger.info(f"\nFirst Document Chunk Info:")
                self.logger.info(f"  Chunk Index: {doc_chunks[0]['chunk_index']}")
                self.logger.info(f"  Total Chunks: {doc_chunks[0]['total_chunks']}")
                self.logger.info(f"  Chunk Size: {doc_chunks[0]['chunk_size']} chars")
                self.logger.info(f"  Metadata: {doc_chunks[0].get('metadata', {})}")

            self.logger.info("\n" + "=" * 80)

            return {
                'recursive_chunks': recursive_chunks,
                'recursive_stats': recursive_stats,
                'token_chunks': token_chunks,
                'token_stats': token_stats,
                'document_chunks': doc_chunks
            }

        except Exception as e:
            self.logger.error(f"Error in chunking demonstration: {e}")
            return {}
    
    def store_transcript(
        self,
        transcript_id: str,
        transcript_text: str,
        analysis_result: Dict[str, Any],
        source_type: str = "text"
    ) -> bool:
        """Store transcript and its analysis in Milvus.
        
        Args:
            transcript_id: Unique identifier for the transcript
            transcript_text: The transcript text
            analysis_result: Analysis results dictionary
            source_type: Source type (text or audio)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import time
            
            # Generate embedding
            embedding = self._get_embedding(transcript_text)
            
            # Prepare data
            data = [
                [transcript_id],
                [embedding],
                [transcript_text],
                [json.dumps(analysis_result)],
                [source_type],
                [int(time.time())]
            ]
            
            # Insert into collection
            self.collection.insert(data)
            self.collection.flush()
            
            self.logger.info(f"Stored transcript: {transcript_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store transcript: {e}")
            return False
    
    def search_similar_transcripts(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar transcripts.
        
        Args:
            query_text: Query text to search for
            top_k: Number of results to return
            
        Returns:
            List of similar transcripts with their analysis
        """
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query_text)
            
            # Search parameters
            search_params = {
                "metric_type": self.config.get('milvus.metric_type', 'L2'),
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["transcript_id", "transcript_text", "analysis_result", "source_type", "timestamp"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "transcript_id": hit.entity.get("transcript_id"),
                        "transcript_text": hit.entity.get("transcript_text"),
                        "analysis_result": json.loads(hit.entity.get("analysis_result")),
                        "source_type": hit.entity.get("source_type"),
                        "timestamp": hit.entity.get("timestamp"),
                        "distance": hit.distance
                    })
            
            self.logger.info(f"Found {len(formatted_results)} similar transcripts")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Failed to search transcripts: {e}")
            return []
    
    def get_transcript_by_id(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve transcript by ID.
        
        Args:
            transcript_id: Transcript identifier
            
        Returns:
            Transcript data or None if not found
        """
        try:
            results = self.collection.query(
                expr=f'transcript_id == "{transcript_id}"',
                output_fields=["transcript_id", "transcript_text", "analysis_result", "source_type", "timestamp"]
            )
            
            if results:
                result = results[0]
                return {
                    "transcript_id": result.get("transcript_id"),
                    "transcript_text": result.get("transcript_text"),
                    "analysis_result": json.loads(result.get("analysis_result")),
                    "source_type": result.get("source_type"),
                    "timestamp": result.get("timestamp")
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve transcript: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from Milvus."""
        try:
            connections.disconnect("default")
            self.logger.info("Disconnected from Milvus")
        except Exception as e:
            self.logger.error(f"Error disconnecting from Milvus: {e}")

