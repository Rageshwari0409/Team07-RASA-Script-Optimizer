"""Chat Agent with Agentic AI using LangChain ReAct Agent."""
import json
import os
from typing import Dict, Any, List, Optional
from langchain_community.chat_models import ChatLiteLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.agent.vector_store import MilvusVectorStore


class ChatAgent:
    """Agentic AI Chat Agent using LangChain ReAct Agent with Tools."""

    def __init__(self):
        """Initialize the agentic chat agent."""
        self.config = get_config()
        self.logger = setup_logger(__name__)

        # Configure environment for LiteLLM
        os.environ["AZURE_API_KEY"] = self.config.get('azure_openai.api_key')
        os.environ["AZURE_API_BASE"] = self.config.get('azure_openai.endpoint')
        os.environ["AZURE_API_VERSION"] = self.config.get('azure_openai.api_version')

        deployment_name = self.config.get('azure_openai.deployment_name')

        # Initialize LangChain LLM with LiteLLM
        self.llm = ChatLiteLLM(
            model=f"azure/{deployment_name}",
            temperature=0.7,
            max_tokens=1000
        )

        # Initialize vector store
        try:
            self.vector_store = MilvusVectorStore()
            self.db_enabled = True
            self.logger.info("Vector store initialized for chat agent")
        except Exception as e:
            self.vector_store = None
            self.db_enabled = False
            self.logger.warning(f"Vector store not available: {e}")

        # Initialize conversation memory for the agent
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        # Create tools for the agent
        self.tools = self._create_tools()

        # Create the ReAct agent
        self.agent = self._create_agent()

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

        self.logger.info("✅ Agentic AI Chat Agent initialized with LangChain ReAct framework")

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent to use."""
        tools = []

        # Tool 1: Search Vector Database
        def search_database(query: str) -> str:
            """Search the vector database for relevant sales transcripts and information.
            Use this tool when the user asks questions about past conversations, requirements, or sales data.

            Args:
                query: The search query to find relevant documents

            Returns:
                Formatted context from relevant documents
            """
            if not self.db_enabled:
                return "Database is not available."

            try:
                search_results = self.vector_store.search_similar_transcripts(
                    query_text=query,
                    top_k=3
                )

                if not search_results:
                    return "No relevant documents found in the database."

                context_parts = []
                for idx, result in enumerate(search_results, 1):
                    transcript_text = result.get('transcript_text', '')
                    analysis = result.get('analysis_result', {})

                    if isinstance(analysis, str):
                        try:
                            analysis = json.loads(analysis)
                        except:
                            analysis = {}

                    context_parts.append(f"Document {idx}:")
                    max_context_chars = 2000
                    if len(transcript_text) > max_context_chars:
                        context_parts.append(f"Transcript: {transcript_text[:max_context_chars]}...")
                    else:
                        context_parts.append(f"Transcript: {transcript_text}")

                    if analysis:
                        if 'summary' in analysis:
                            summary = analysis['summary']
                            if isinstance(summary, dict):
                                overview = summary.get('overview', '')
                                sentiment = summary.get('sentiment', '')
                                context_parts.append(f"Summary: {overview}")
                                if sentiment:
                                    context_parts.append(f"Sentiment: {sentiment}")
                        if 'requirements' in analysis:
                            reqs = analysis['requirements'][:3]
                            context_parts.append(f"Requirements: {json.dumps(reqs)}")
                        if 'key_points' in analysis:
                            key_points = analysis['key_points'][:5]
                            context_parts.append(f"Key Points: {json.dumps(key_points)}")
                        if 'action_items' in analysis:
                            actions = analysis['action_items'][:3]
                            context_parts.append(f"Action Items: {json.dumps(actions)}")
                        if 'recommendations' in analysis:
                            recs = analysis['recommendations'][:2]
                            context_parts.append(f"Recommendations: {json.dumps(recs)}")

                    context_parts.append("")

                return "\n".join(context_parts)
            except Exception as e:
                self.logger.error(f"Error searching database: {e}")
                return f"Error searching database: {str(e)}"

        search_tool = Tool(
            name="search_database",
            func=search_database,
            description="Search the vector database for relevant sales transcripts, requirements, and past conversations. Use this when the user asks about specific information from uploaded documents."
        )
        tools.append(search_tool)

        return tools

    def _create_agent(self):
        """Create the ReAct agent with custom prompt."""
        # ReAct prompt template
        react_prompt = PromptTemplate.from_template(
            """You are RASA, an AI assistant for sales transcript analysis. You have access to tools to help answer questions.

You must provide crisp, clear, and exact answers (1-2 sentences maximum). You can understand semantic meaning and draw conclusions from context.

TOOLS:
{tools}

TOOL NAMES: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (keep it crisp and short, 1-2 sentences max)

Begin!

Chat History:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""
        )

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )

        return agent

    def chat(self, user_message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process user message using the agentic AI framework.

        Args:
            user_message: User's message/question
            session_id: Optional session ID for tracking conversations

        Returns:
            Dictionary with response and metadata
        """
        self.logger.info(f"🤖 Agent processing message: {user_message[:100]}...")

        try:
            # Use the agent executor to process the message
            # The agent will decide whether to use tools or answer directly
            result = self.agent_executor.invoke({
                "input": user_message
            })

            # Extract the answer from agent result
            answer = result.get("output", "I apologize, but I couldn't generate a response.")

            self.logger.info("✅ Agent response generated successfully")

            return {
                "success": True,
                "answer": answer,
                "relevant_documents": 0,  # Agent handles this internally
                "session_id": session_id
            }

        except Exception as e:
            self.logger.error(f"❌ Error in agent execution: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "answer": "I apologize, but I encountered an error processing your message."
            }
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        self.logger.info("Agent conversation memory cleared")

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get formatted chat history from agent memory.

        Returns:
            List of message dictionaries
        """
        messages = self.memory.chat_memory.messages
        history = []
        for msg in messages:
            if hasattr(msg, 'type'):
                history.append({
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                })
        return history

