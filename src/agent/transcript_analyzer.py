"""Sales transcript analyzer using LiteLLM with LangChain text chunking."""
import json
import litellm
from typing import Dict, Any, Optional
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.utils.text_chunker import TextChunker
from src.utils.retry_handler import retry_llm_call


class TranscriptAnalyzer:
    """Analyze sales conversation transcripts using LiteLLM with LangChain chunking."""

    def __init__(self):
        """Initialize the transcript analyzer."""
        self.config = get_config()
        self.logger = setup_logger(__name__)

        # Configure LiteLLM for Azure OpenAI
        self.api_key = self.config.get('azure_openai.api_key')
        self.api_base = self.config.get('azure_openai.endpoint')
        self.api_version = self.config.get('azure_openai.api_version')
        self.deployment_name = self.config.get('azure_openai.deployment_name')

        # Set LiteLLM configuration
        litellm.api_key = self.api_key
        litellm.api_base = self.api_base
        litellm.api_version = self.api_version

        # Initialize text chunker (LangChain)
        self.chunker = TextChunker()

        self.logger.info(f"LiteLLM configured with Azure OpenAI deployment: {self.deployment_name}")
        self.logger.info(f"LangChain text chunker initialized")

    @retry_llm_call
    def _call_llm_completion(self, messages, temperature, max_tokens, response_format=None):
        """Internal method to call LLM with retry logic.

        Args:
            messages: List of message dictionaries
            temperature: Temperature setting
            max_tokens: Maximum tokens
            response_format: Optional response format

        Returns:
            LLM response
        """
        kwargs = {
            "model": f"azure/{self.deployment_name}",
            "messages": messages,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if response_format:
            kwargs["response_format"] = response_format

        return litellm.completion(**kwargs)

    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """Analyze a sales conversation transcript using LiteLLM with LangChain chunking.

        Args:
            transcript: The conversation transcript text

        Returns:
            Dictionary containing analysis results with requirements, recommendations, and summary
        """
        self.logger.info("Starting transcript analysis with LiteLLM + LangChain chunking")

        try:
            # Demonstrate chunking if text is long
            if len(transcript) > 5000:
                self.logger.info(f"Transcript is long ({len(transcript)} chars), demonstrating chunking...")
                chunks = self.chunker.chunk_text_recursive(transcript)
                stats = self.chunker.get_chunk_stats(chunks)
                self.logger.info(f"📊 Chunking Stats: {stats['total_chunks']} chunks, "
                               f"avg size: {stats['avg_chunk_size']} chars")

            # Get prompts
            system_prompt = self.config.get_prompt('system_prompt')
            analysis_prompt = self.config.get_prompt('analysis_prompt')

            # Format the analysis prompt with transcript
            user_prompt = analysis_prompt.format(transcript=transcript)

            self.logger.info(f"Calling LiteLLM with Azure deployment: {self.deployment_name}")

            # Call LiteLLM with Azure OpenAI (supports JSON mode) with retry logic
            response = self._call_llm_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.get('azure_openai.temperature', 0.7),
                max_tokens=self.config.get('azure_openai.max_tokens', 2000),
                response_format={"type": "json_object"}
            )

            # Extract and parse response
            content = response.choices[0].message.content
            self.logger.info(f"LiteLLM Response: {content[:200]}...")

            # Clean response - remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()

            analysis_result = json.loads(content)
            self.logger.info(f"Parsed result keys: {list(analysis_result.keys())}")

            self.logger.info("Transcript analysis completed successfully")
            return analysis_result

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse response as JSON: {e}")
            return self._get_error_response("Failed to parse analysis results")
        except Exception as e:
            self.logger.error(f"Error during transcript analysis: {e}")
            return self._get_error_response(str(e))
    
    def extract_requirements(self, transcript: str) -> Dict[str, Any]:
        """Extract client requirements from transcript.
        
        Args:
            transcript: The conversation transcript text
            
        Returns:
            Dictionary containing extracted requirements
        """
        self.logger.info("Extracting requirements from transcript")
        
        try:
            system_prompt = self.config.get_prompt('system_prompt')
            requirements_prompt = self.config.get_prompt('requirements_extraction_prompt')
            
            user_prompt = requirements_prompt.format(transcript=transcript)
            
            # Call with retry logic
            response = self._call_llm_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            self.logger.info("Requirements extraction completed")
            return {"requirements": content}
            
        except Exception as e:
            self.logger.error(f"Error extracting requirements: {e}")
            return {"error": str(e)}
    
    def generate_recommendations(self, transcript: str) -> Dict[str, Any]:
        """Generate product recommendations based on transcript.
        
        Args:
            transcript: The conversation transcript text
            
        Returns:
            Dictionary containing recommendations
        """
        self.logger.info("Generating recommendations from transcript")
        
        try:
            system_prompt = self.config.get_prompt('system_prompt')
            recommendations_prompt = self.config.get_prompt('recommendations_prompt')
            
            user_prompt = recommendations_prompt.format(transcript=transcript)

            # Call with retry logic
            response = self._call_llm_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            self.logger.info("Recommendations generation completed")
            return {"recommendations": content}
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return {"error": str(e)}
    
    def generate_summary(self, transcript: str) -> Dict[str, Any]:
        """Generate summary of the conversation.
        
        Args:
            transcript: The conversation transcript text
            
        Returns:
            Dictionary containing summary
        """
        self.logger.info("Generating summary from transcript")
        
        try:
            system_prompt = self.config.get_prompt('system_prompt')
            summary_prompt = self.config.get_prompt('summary_prompt')
            
            user_prompt = summary_prompt.format(transcript=transcript)

            # Call with retry logic
            response = self._call_llm_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            self.logger.info("Summary generation completed")
            return {"summary": content}
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}
    
    def _get_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response structure.
        
        Args:
            error_message: Error message to include
            
        Returns:
            Error response dictionary
        """
        return {
            "error": error_message,
            "requirements": [],
            "recommendations": [],
            "summary": {
                "overview": "Analysis failed",
                "error": error_message
            },
            "key_points": [],
            "action_items": []
        }

