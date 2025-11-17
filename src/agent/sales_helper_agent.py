"""Sales Helper Agent with Agentic Approach using LiteLLM."""
import json
import litellm
from typing import Dict, Any, List, Optional
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.agent.vector_store import MilvusVectorStore
from src.utils.retry_handler import retry_llm_call


class SalesHelperAgent:
    """Agentic sales helper that captures requirements and searches database using LiteLLM."""

    def __init__(self):
        """Initialize the sales helper agent."""
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
        
        # Initialize vector store for database search
        try:
            self.vector_store = MilvusVectorStore()
            self.db_enabled = True
            self.logger.info("Vector store initialized for sales helper agent")
        except Exception as e:
            self.vector_store = None
            self.db_enabled = False
            self.logger.warning(f"Vector store not available: {e}")
        
        # Agent state
        self.conversation_history = []

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

    def process_salesperson_input(self, user_input: str) -> Dict[str, Any]:
        """Process salesperson input with agentic approach.
        
        Args:
            user_input: Input from salesperson describing client needs
            
        Returns:
            Dictionary with extracted requirements and search results
        """
        self.logger.info("Processing salesperson input with agentic approach")
        
        try:
            # Step 1: Extract requirements from salesperson input
            requirements = self._extract_requirements(user_input)
            
            # Step 2: Search database for similar cases
            search_results = []
            if self.db_enabled and requirements:
                search_results = self._search_similar_cases(requirements)
            
            # Step 3: Generate recommendations based on requirements and search results
            recommendations = self._generate_recommendations(
                user_input, 
                requirements, 
                search_results
            )
            
            # Step 4: Update conversation history
            self.conversation_history.append({
                "input": user_input,
                "requirements": requirements,
                "recommendations": recommendations
            })
            
            return {
                "success": True,
                "requirements": requirements,
                "search_results": search_results,
                "recommendations": recommendations,
                "conversation_id": len(self.conversation_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing salesperson input: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_requirements(self, user_input: str) -> List[Dict[str, Any]]:
        """Extract structured requirements from salesperson input.
        
        Args:
            user_input: Salesperson's description of client needs
            
        Returns:
            List of extracted requirements
        """
        self.logger.info("Extracting requirements from input")
        
        try:
            system_prompt = self.config.get_prompt('sales_helper_system_prompt')
            extraction_prompt = self.config.get_prompt('requirement_extraction_prompt')
            
            user_prompt = extraction_prompt.format(input=user_input)

            # Use LiteLLM with JSON mode and retry logic
            response = self._call_llm_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            # Clean response - remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)

            self.logger.info(f"Extracted {len(result.get('requirements', []))} requirements")
            return result.get('requirements', [])
            
        except Exception as e:
            self.logger.error(f"Error extracting requirements: {e}")
            return []
    
    def _search_similar_cases(self, requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search database for similar past cases.

        Args:
            requirements: Extracted requirements

        Returns:
            List of similar cases from database
        """
        if not self.db_enabled or not requirements:
            self.logger.warning("Database search disabled or no requirements provided")
            return []

        self.logger.info("🔍 Searching Milvus database for similar cases")

        try:
            # Create search query from requirements
            search_query = " ".join([req.get('requirement', '') for req in requirements])
            self.logger.info(f"Search query: {search_query[:100]}...")

            # Search vector store
            results = self.vector_store.search_similar_transcripts(
                query_text=search_query,
                top_k=3
            )

            self.logger.info(f"✅ Found {len(results)} similar cases from database")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching database: {e}")
            return []

    def _generate_recommendations(
        self,
        user_input: str,
        requirements: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on requirements and similar cases.

        Args:
            user_input: Original salesperson input
            requirements: Extracted requirements
            search_results: Similar cases from database

        Returns:
            List of recommendations
        """
        self.logger.info("Generating recommendations")

        try:
            system_prompt = self.config.get_prompt('sales_helper_system_prompt')
            recommendation_prompt = self.config.get_prompt('sales_recommendation_prompt')

            # Format search results for context
            context = ""
            if search_results:
                context = "Similar past cases:\n"
                for idx, result in enumerate(search_results[:3], 1):
                    context += f"\nCase {idx}:\n"
                    context += f"Transcript: {result.get('transcript_text', '')[:200]}...\n"
                    analysis = result.get('analysis_result', {})
                    if isinstance(analysis, str):
                        analysis = json.loads(analysis)
                    context += f"Recommendations: {json.dumps(analysis.get('recommendations', []))}\n"

            user_prompt = recommendation_prompt.format(
                input=user_input,
                requirements=json.dumps(requirements, indent=2),
                context=context
            )

            # Use LiteLLM with JSON mode and retry logic
            response = self._call_llm_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            # Clean response - remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)

            self.logger.info(f"Generated {len(result.get('recommendations', []))} recommendations")
            return result.get('recommendations', [])

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.logger.info("Conversation history reset")

