"""FastAPI application for sales transcript analysis."""
import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.models import (
    TextAnalysisRequest,
    AnalysisResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    HealthResponse,
    InputType,
    SalesHelperRequest,
    SalesHelperResponse,
    ChatRequest,
    ChatResponse
)
from src.agent.transcript_analyzer import TranscriptAnalyzer
from src.agent.vector_store import MilvusVectorStore
from src.agent.sales_helper_agent import SalesHelperAgent
from src.agent.chat_agent import ChatAgent
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.utils.document_processor import DocumentProcessor

# Simple session storage
SESSIONS = {}
USERS = {"admin": "admin123", "demo": "demo123"}
USER_DATA = {
    "admin": {"email": "admin@rasa.com", "full_name": "Administrator"},
    "demo": {"email": "demo@rasa.com", "full_name": "Demo User"}
}
CHAT_HISTORY = {}  # {username: [{id, title, messages: [{role, content}]}]}

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str
    full_name: str



# Initialize configuration and logger
config = get_config()
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config.get('fastapi.title', 'Sales Transcript Analysis API'),
    description=config.get('fastapi.description', 'API for analyzing sales conversations'),
    version=config.get('fastapi.version', '1.0.0')
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# Initialize components
transcript_analyzer = TranscriptAnalyzer()
sales_helper_agent = SalesHelperAgent()
chat_agent = ChatAgent()

# Try to initialize Milvus, but continue without it if it fails
try:
    vector_store = MilvusVectorStore()
    MILVUS_ENABLED = True
    logger.info("Milvus vector store initialized successfully")
except Exception as e:
    vector_store = None
    MILVUS_ENABLED = False
    logger.warning(f"Milvus not available: {e}. Search functionality will be disabled.")

# Create temp directory for audio uploads
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    from pathlib import Path
    login_file = Path("src/api/templates/login.html")
    return HTMLResponse(content=login_file.read_text(encoding="utf-8"))

@app.post("/register")
async def register(request: RegisterRequest):
    if request.username in USERS:
        return JSONResponse({"success": False, "message": "Username already exists"}, status_code=400)
    USERS[request.username] = request.password
    USER_DATA[request.username] = {"email": request.email, "full_name": request.full_name}
    return {"success": True, "message": "Registration successful"}

@app.post("/login")
async def login(request: LoginRequest, response: Response):
    import secrets
    if request.username in USERS and USERS[request.username] == request.password:
        token = secrets.token_urlsafe(32)
        SESSIONS[token] = request.username
        response.set_cookie(key="session", value=token, httponly=True)
        return {"success": True}
    return JSONResponse({"success": False, "message": "Invalid credentials"}, status_code=401)

@app.post("/logout")
async def logout(response: Response, session: Optional[str] = Cookie(None)):
    if session in SESSIONS:
        del SESSIONS[session]
    response.delete_cookie("session")
    return {"success": True}

@app.get("/", response_class=HTMLResponse)
async def root(session: Optional[str] = Cookie(None)):
    """Root endpoint - Web UI for file upload."""
    if not session or session not in SESSIONS:
        return RedirectResponse(url="/login", status_code=302)
    from pathlib import Path
    dashboard_file = Path("src/api/templates/dashboard.html")
    return HTMLResponse(content=dashboard_file.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=config.get('fastapi.version', '1.0.0'),
        services={
            "api": "running",
            "llm": "configured",
            "milvus": "connected"
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=config.get('fastapi.version', '1.0.0'),
        services={
            "api": "running",
            "llm": "configured",
            "milvus": "connected"
        }
    )


@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text_transcript(request: TextAnalysisRequest):
    """Analyze a text transcript.

    Args:
        request: Text analysis request containing the transcript

    Returns:
        Analysis results including requirements, recommendations, and summary
    """
    try:
        logger.info("Received text transcript analysis request")

        # Generate transcript ID if not provided
        transcript_id = request.transcript_id or str(uuid.uuid4())

        # Analyze transcript
        analysis_result = transcript_analyzer.analyze_transcript(request.transcript)

        # Check for errors in analysis
        if "error" in analysis_result:
            return AnalysisResponse(
                success=False,
                transcript_id=transcript_id,
                transcript=request.transcript,
                error=analysis_result["error"],
                source_type=InputType.TEXT
            )

        # Store in database if requested
        if request.store_in_db and MILVUS_ENABLED:
            logger.info(f"Storing transcript {transcript_id} in Milvus database")
            vector_store.store_transcript(
                transcript_id=transcript_id,
                transcript_text=request.transcript,
                analysis_result=analysis_result,
                source_type=InputType.TEXT
            )
            logger.info(f"✅ Transcript {transcript_id} stored successfully")

        return AnalysisResponse(
            success=True,
            transcript_id=transcript_id,
            transcript=request.transcript,
            analysis=analysis_result,
            source_type=InputType.TEXT
        )

    except Exception as e:
        logger.error(f"Error analyzing text transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(..., description="Document file (PDF, Word, CSV, Excel, TXT)"),
    transcript_id: Optional[str] = Form(None),
    store_in_db: bool = Form(True)
):
    """Analyze a document file (PDF, Word, CSV, Excel, TXT).

    Args:
        file: Document file upload
        transcript_id: Optional unique identifier
        store_in_db: Whether to store in database

    Returns:
        Analysis results including requirements, recommendations, and summary
    """
    try:
        logger.info(f"Received file analysis request: {file.filename}")

        # Generate transcript ID if not provided
        transcript_id = transcript_id or str(uuid.uuid4())

        # Read file content
        file_content = await file.read()

        # Extract text from file
        try:
            transcript_text = DocumentProcessor.process_file(file.filename, file_content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"Missing dependency: {str(e)}")

        if not transcript_text or not transcript_text.strip():
            return AnalysisResponse(
                success=False,
                transcript_id=transcript_id,
                error="No text could be extracted from the file",
                source_type=InputType.TEXT
            )

        logger.info(f"Extracted {len(transcript_text)} characters from {file.filename}")

        # Analyze the extracted text
        analysis_result = transcript_analyzer.analyze_transcript(transcript_text)

        # Store in vector database if requested
        if store_in_db and MILVUS_ENABLED:
            try:
                logger.info(f"Storing file transcript {transcript_id} in Milvus database")
                vector_store.store_transcript(
                    transcript_id=transcript_id,
                    transcript_text=transcript_text,
                    analysis_result=analysis_result,
                    source_type=f"file_{Path(file.filename).suffix}"
                )
                logger.info(f"✅ File transcript {transcript_id} stored successfully")
            except Exception as e:
                logger.warning(f"Failed to store in vector database: {e}")
                analysis_result["storage_warning"] = "Analysis completed but not stored in database"

        return AnalysisResponse(
            success=True,
            transcript_id=transcript_id,
            transcript=transcript_text,
            analysis=analysis_result,
            source_type=InputType.TEXT
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_transcripts(request: SearchRequest):
    """Search for similar transcripts.

    Args:
        request: Search request with query text

    Returns:
        List of similar transcripts
    """
    try:
        logger.info(f"Searching for similar transcripts: {request.query}")

        results = vector_store.search_similar_transcripts(
            query_text=request.query,
            top_k=request.top_k
        )

        search_results = [
            SearchResult(
                transcript_id=r["transcript_id"],
                transcript_text=r["transcript_text"],
                analysis_result=r["analysis_result"],
                source_type=r["source_type"],
                timestamp=r["timestamp"],
                distance=r["distance"]
            )
            for r in results
        ]

        return SearchResponse(
            success=True,
            results=search_results,
            count=len(search_results)
        )

    except Exception as e:
        logger.error(f"Error searching transcripts: {e}")
        return SearchResponse(
            success=False,
            results=[],
            count=0,
            error=str(e)
        )


@app.get("/transcript/{transcript_id}")
async def get_transcript(transcript_id: str):
    """Retrieve a transcript by ID.

    Args:
        transcript_id: Transcript identifier

    Returns:
        Transcript data and analysis
    """
    try:
        result = vector_store.get_transcript_by_id(transcript_id)

        if result:
            return JSONResponse(content={
                "success": True,
                "data": result
            })
        else:
            raise HTTPException(status_code=404, detail="Transcript not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sales-helper", response_model=SalesHelperResponse)
async def sales_helper(request: SalesHelperRequest):
    """Sales helper agent endpoint.

    Captures requirements from salesperson, searches database, and provides recommendations.

    Args:
        request: Salesperson's description of client needs

    Returns:
        Requirements, search results, and recommendations
    """
    try:
        logger.info(f"Sales helper request received: {request.salesperson_input[:100]}...")

        result = sales_helper_agent.process_salesperson_input(request.salesperson_input)

        return SalesHelperResponse(**result)

    except Exception as e:
        logger.error(f"Error in sales helper: {e}")
        return SalesHelperResponse(
            success=False,
            error=str(e)
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session: Optional[str] = Cookie(None)):
    """Chat with AI agent about stored transcript data.

    Uses LangChain with conversation memory to answer questions based on stored data.

    Args:
        request: User's chat message and optional session ID

    Returns:
        AI response with relevant information from database
    """
    try:
        logger.info(f"Chat request received: {request.message[:100]}...")

        username = SESSIONS.get(session, "guest")

        result = chat_agent.chat(
            user_message=request.message,
            session_id=request.session_id
        )

        # Save to chat history
        if username not in CHAT_HISTORY:
            CHAT_HISTORY[username] = []

        chat_id = request.session_id or str(uuid.uuid4())
        existing_chat = next((c for c in CHAT_HISTORY[username] if c["id"] == chat_id), None)

        if existing_chat:
            existing_chat["messages"].append({"role": "user", "content": request.message})
            existing_chat["messages"].append({"role": "assistant", "content": result.get("answer", "")})
        else:
            title = request.message[:50] + "..." if len(request.message) > 50 else request.message
            CHAT_HISTORY[username].append({
                "id": chat_id,
                "title": title,
                "messages": [
                    {"role": "user", "content": request.message},
                    {"role": "assistant", "content": result.get("answer", "")}
                ]
            })

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return ChatResponse(
            success=False,
            answer="I apologize, but I encountered an error processing your message.",
            error=str(e)
        )


@app.get("/chat/history")
async def get_chat_history(session: Optional[str] = Cookie(None)):
    """Get chat history for current user."""
    username = SESSIONS.get(session, "guest")
    return {"success": True, "history": CHAT_HISTORY.get(username, [])}

@app.post("/chat/clear")
async def clear_chat(session: Optional[str] = Cookie(None)):
    """Clear chat conversation memory."""
    try:
        chat_agent.clear_memory()
        username = SESSIONS.get(session, "guest")
        if username in CHAT_HISTORY:
            CHAT_HISTORY[username] = []
        return {"success": True, "message": "Chat memory cleared"}
    except Exception as e:
        logger.error(f"Error clearing chat memory: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    host = config.get('fastapi.host', '0.0.0.0')
    port = config.get('fastapi.port', 8000)
    reload = config.get('fastapi.reload', True)

    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)

