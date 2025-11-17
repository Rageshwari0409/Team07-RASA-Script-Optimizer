"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class InputType(str, Enum):
    """Input type enumeration."""
    TEXT = "text"
    AUDIO = "audio"


class Priority(str, Enum):
    """Priority level enumeration."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Sentiment(str, Enum):
    """Sentiment enumeration."""
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class EngagementLevel(str, Enum):
    """Engagement level enumeration."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Requirement(BaseModel):
    """Client requirement model."""
    requirement: str
    priority: Priority
    mentioned_by: str
    context: str


class Recommendation(BaseModel):
    """Product recommendation model."""
    recommendation: str
    rationale: str
    product_fit: str
    priority: Priority


class Summary(BaseModel):
    """Conversation summary model."""
    overview: str
    client_needs: str
    pain_points: str
    opportunities: str
    next_steps: str
    sentiment: Sentiment
    engagement_level: EngagementLevel


class ActionItem(BaseModel):
    """Action item model."""
    action: str
    owner: str
    priority: Priority


class AnalysisResult(BaseModel):
    """Complete analysis result model."""
    requirements: List[Requirement]
    recommendations: List[Recommendation]
    summary: Summary
    key_points: List[str]
    action_items: List[ActionItem]


class TextAnalysisRequest(BaseModel):
    """Request model for text transcript analysis."""
    transcript: str = Field(..., description="The conversation transcript text")
    transcript_id: Optional[str] = Field(None, description="Optional unique identifier for the transcript")
    store_in_db: bool = Field(True, description="Whether to store the analysis in the database")


class AudioAnalysisRequest(BaseModel):
    """Request model for audio file analysis."""
    transcript_id: Optional[str] = Field(None, description="Optional unique identifier for the transcript")
    store_in_db: bool = Field(True, description="Whether to store the analysis in the database")


class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    success: bool
    transcript_id: Optional[str] = None
    transcript: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    source_type: str


class SearchRequest(BaseModel):
    """Request model for searching similar transcripts."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)


class SearchResult(BaseModel):
    """Search result model."""
    transcript_id: str
    transcript_text: str
    analysis_result: Dict[str, Any]
    source_type: str
    timestamp: int
    distance: float


class SearchResponse(BaseModel):
    """Response model for search."""
    success: bool
    results: List[SearchResult]
    count: int
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: Dict[str, str]


# Sales Helper Agent Models
class SalesHelperRequest(BaseModel):
    """Request model for sales helper agent."""
    salesperson_input: str = Field(..., description="Salesperson's description of client needs")


class SalesHelperResponse(BaseModel):
    """Response model for sales helper agent."""
    success: bool
    requirements: Optional[List[Dict[str, Any]]] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    conversation_id: Optional[int] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chat agent."""
    message: str = Field(..., description="User's chat message")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation tracking")


class ChatResponse(BaseModel):
    """Response model for chat agent."""
    success: bool
    answer: str
    relevant_documents: Optional[int] = None
    session_id: Optional[str] = None
    error: Optional[str] = None

