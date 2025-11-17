"""FastAPI application for sales transcript analysis."""
import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

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


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - Web UI for file upload."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sales Transcript Analysis</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 30px;
            }
            .left-panel, .middle-panel, .right-panel {
                flex: 1;
            }
            .panel-title {
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .input-section {
                margin-bottom: 30px;
            }
            label {
                display: block;
                margin-bottom: 10px;
                color: #333;
                font-weight: 600;
                font-size: 1.1em;
            }
            .chat-container {
                display: flex;
                flex-direction: column;
                height: 600px;
            }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9ff;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .chat-message {
                margin-bottom: 15px;
                padding: 12px 16px;
                border-radius: 10px;
                max-width: 80%;
            }
            .chat-message.user {
                background: #667eea;
                color: white;
                margin-left: auto;
            }
            .chat-message.assistant {
                background: white;
                border: 2px solid #e0e0e0;
            }
            .chat-input-container {
                display: flex;
                gap: 10px;
            }
            .chat-input {
                flex: 1;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
            }
            .chat-button {
                width: auto;
                padding: 12px 30px;
            }
            select, textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            select:focus, textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            textarea {
                min-height: 200px;
                resize: vertical;
                font-family: 'Courier New', monospace;
            }
            .file-upload {
                border: 3px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                background: #f8f9ff;
            }
            .file-upload:hover {
                background: #e8ebff;
                border-color: #764ba2;
            }
            .file-upload input {
                display: none;
            }
            .file-info {
                margin-top: 15px;
                color: #667eea;
                font-weight: 600;
            }
            button {
                width: 100%;
                padding: 18px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .results {
                margin-top: 30px;
                padding: 25px;
                background: #f8f9ff;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }
            .results h3 {
                color: #667eea;
                margin-bottom: 15px;
            }
            .result-section {
                margin-bottom: 20px;
                padding: 15px;
                background: white;
                border-radius: 8px;
            }
            .result-section h4 {
                color: #764ba2;
                margin-bottom: 10px;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #667eea;
                font-size: 1.2em;
            }
            .error {
                background: #ffe0e0;
                border-left-color: #ff4444;
                color: #cc0000;
            }
            .hidden { display: none; }
            .supported-formats {
                text-align: center;
                color: #888;
                font-size: 0.9em;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <h1 style="text-align: center; color: white; margin-bottom: 40px;">📊 Sales Transcript Analysis</h1>

        <div class="container">
            <!-- LEFT PANEL: Transcript Analysis -->
            <div class="left-panel">
                <h2 class="panel-title">📝 Transcript Analysis</h2>
                <div class="input-section">
                    <label for="inputType">Select Input Type:</label>
                    <select id="inputType" onchange="toggleInputType()">
                        <option value="text">Text Transcript</option>
                        <option value="file">Upload File (PDF, Word, CSV, Excel, Audio)</option>
                    </select>
                </div>

            <div id="textInput" class="input-section">
                <label for="transcript">Paste Your Transcript:</label>
                <textarea id="transcript" placeholder="Sales Rep: Hi! I'm calling about our CRM solution.

Client: We need a system for 100 users with mobile access.

Sales Rep: Perfect! Our Enterprise plan supports that at $45 per user monthly.

Client: That fits our $5,000 budget. What's next?"></textarea>
            </div>

            <div id="fileInput" class="input-section hidden">
                <label>Upload File:</label>
                <div class="file-upload" onclick="document.getElementById('fileUpload').click()">
                    <input type="file" id="fileUpload" accept=".pdf,.docx,.doc,.csv,.xlsx,.xls,.txt,.mp3,.wav,.m4a,.ogg" onchange="handleFileSelect(event)">
                    <div>
                        <p style="font-size: 3em; margin-bottom: 10px;">📁</p>
                        <p style="font-size: 1.2em; color: #667eea; font-weight: 600;">Click to upload or drag & drop</p>
                        <p class="supported-formats">Supported: PDF, Word, CSV, Excel, TXT, MP3, WAV, M4A, OGG</p>
                    </div>
                    <div id="fileInfo" class="file-info"></div>
                </div>
            </div>

                <button onclick="analyzeTranscript()" id="analyzeBtn">🚀 Analyze Transcript</button>
            </div>

            <!-- MIDDLE PANEL: Analysis Results -->
            <div class="middle-panel">
                <h2 class="panel-title">📊 Analysis Results</h2>
                <div id="results" class="hidden"></div>
            </div>

            <!-- RIGHT PANEL: Chat Agent -->
            <div class="right-panel">
                <h2 class="panel-title">🤖 Chat with RASA</h2>

                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="chat-message assistant">
                            <strong>RASA:</strong><br>
                            Hello! I can help you find information from stored transcripts. Ask me anything about past conversations, requirements, or recommendations.
                        </div>
                    </div>

                    <div class="chat-input-container">
                        <input type="text" id="chatInput" class="chat-input" placeholder="Ask me about stored transcripts..." onkeypress="if(event.key==='Enter') sendChat()">
                        <button onclick="sendChat()" class="chat-button">Send</button>
                        <button onclick="clearChat()" class="chat-button" style="background: #888;">Clear</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let selectedFile = null;

            function toggleInputType() {
                const inputType = document.getElementById('inputType').value;
                const textInput = document.getElementById('textInput');
                const fileInput = document.getElementById('fileInput');

                if (inputType === 'text') {
                    textInput.classList.remove('hidden');
                    fileInput.classList.add('hidden');
                    selectedFile = null;
                } else {
                    textInput.classList.add('hidden');
                    fileInput.classList.remove('hidden');
                }
            }

            function handleFileSelect(event) {
                selectedFile = event.target.files[0];
                if (selectedFile) {
                    const fileInfo = document.getElementById('fileInfo');
                    const sizeMB = (selectedFile.size / (1024 * 1024)).toFixed(2);
                    fileInfo.innerHTML = `✅ Selected: ${selectedFile.name} (${sizeMB} MB)`;
                }
            }

            async function analyzeTranscript() {
                const inputType = document.getElementById('inputType').value;
                const resultsDiv = document.getElementById('results');
                const analyzeBtn = document.getElementById('analyzeBtn');

                // Show loading
                resultsDiv.className = 'results';
                resultsDiv.innerHTML = '<div class="loading">⏳ Analyzing... Please wait...</div>';
                resultsDiv.classList.remove('hidden');
                analyzeBtn.disabled = true;

                try {
                    let response;

                    if (inputType === 'text') {
                        const transcript = document.getElementById('transcript').value;
                        if (!transcript.trim()) {
                            throw new Error('Please enter a transcript');
                        }

                        response = await fetch('/analyze/text', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                transcript: transcript,
                                store_in_db: true
                            })
                        });
                    } else {
                        if (!selectedFile) {
                            throw new Error('Please select a file');
                        }

                        const formData = new FormData();
                        formData.append('file', selectedFile);
                        formData.append('store_in_db', 'true');

                        // Determine endpoint based on file type
                        const fileExt = selectedFile.name.split('.').pop().toLowerCase();
                        const audioFormats = ['mp3', 'wav', 'm4a', 'ogg'];
                        const endpoint = audioFormats.includes(fileExt) ? '/analyze/audio' : '/analyze/file';

                        response = await fetch(endpoint, {
                            method: 'POST',
                            body: formData
                        });
                    }

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Analysis failed');
                    }

                    const data = await response.json();
                    displayResults(data);

                } catch (error) {
                    resultsDiv.className = 'results error';
                    resultsDiv.innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
                } finally {
                    analyzeBtn.disabled = false;
                }
            }

            function displayResults(data) {
                const resultsDiv = document.getElementById('results');

                // Extract analysis from response
                const analysis = data.analysis || data;

                let html = '<h3>✅ Analysis Complete</h3>';

                // Show storage confirmation (without ID)
                if (data.transcript_id) {
                    html += `<div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 15px;">`;
                    html += `💾 <strong>Stored in Database</strong>`;
                    html += `</div>`;
                }

                // Requirements (without priority)
                if (analysis.requirements && analysis.requirements.length > 0) {
                    html += '<div class="result-section"><h4>📋 Requirements</h4><ul>';
                    analysis.requirements.forEach(req => {
                        html += `<li><strong>${req.requirement}</strong></li>`;
                    });
                    html += '</ul></div>';
                }

                // Recommendations (handle undefined values)
                if (analysis.recommendations && analysis.recommendations.length > 0) {
                    html += '<div class="result-section"><h4>💡 Recommendations</h4><ul>';
                    analysis.recommendations.forEach(rec => {
                        const product = rec.product || rec.product_service || 'Recommendation';
                        const rationale = rec.rationale || '';
                        if (rationale && rationale !== 'undefined') {
                            html += `<li><strong>${product}</strong>: ${rationale}</li>`;
                        }
                    });
                    html += '</ul></div>';
                }

                // Summary
                if (analysis.summary) {
                    html += '<div class="result-section"><h4>📝 Summary</h4>';
                    if (analysis.summary.overview) {
                        html += `<p><strong>Overview:</strong> ${analysis.summary.overview}</p>`;
                    }
                    if (analysis.summary.pain_points) {
                        const painPoints = Array.isArray(analysis.summary.pain_points)
                            ? analysis.summary.pain_points.join(', ')
                            : analysis.summary.pain_points;
                        html += `<p><strong>Pain Points:</strong> ${painPoints}</p>`;
                    }
                    if (analysis.summary.next_steps) {
                        const nextSteps = Array.isArray(analysis.summary.next_steps)
                            ? analysis.summary.next_steps.join(', ')
                            : analysis.summary.next_steps;
                        html += `<p><strong>Next Steps:</strong> ${nextSteps}</p>`;
                    }
                    html += '</div>';
                }

                // Action Items (without priority)
                if (analysis.action_items && analysis.action_items.length > 0) {
                    html += '<div class="result-section"><h4>✅ Action Items</h4><ul>';
                    analysis.action_items.forEach(item => {
                        html += `<li><strong>${item.action}</strong> - ${item.owner}</li>`;
                    });
                    html += '</ul></div>';
                }

                resultsDiv.innerHTML = html;
            }

            async function getSalesHelp() {
                const salesInput = document.getElementById('salesInput').value.trim();
                const helperBtn = document.getElementById('helperBtn');
                const resultsDiv = document.getElementById('helperResults');

                if (!salesInput) {
                    alert('Please describe the client needs');
                    return;
                }

                helperBtn.disabled = true;
                resultsDiv.className = 'results';
                resultsDiv.innerHTML = '<h3>⏳ Processing...</h3>';

                try {
                    const response = await fetch('/sales-helper', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ salesperson_input: salesInput })
                    });

                    if (!response.ok) {
                        throw new Error('Request failed');
                    }

                    const data = await response.json();
                    displaySalesHelperResults(data);

                } catch (error) {
                    resultsDiv.className = 'results error';
                    resultsDiv.innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
                } finally {
                    helperBtn.disabled = false;
                }
            }

            function displaySalesHelperResults(data) {
                const resultsDiv = document.getElementById('helperResults');

                let html = '<h3>✅ Analysis Complete</h3>';

                // Requirements
                if (data.requirements && data.requirements.length > 0) {
                    html += '<div class="result-section"><h4>📋 Extracted Requirements</h4><ul>';
                    data.requirements.forEach(req => {
                        html += `<li><strong>${req.requirement}</strong><br>`;
                        html += `Category: ${req.category} | Priority: ${req.priority}<br>`;
                        if (req.details) html += `<em>${req.details}</em>`;
                        html += `</li>`;
                    });
                    html += '</ul></div>';
                }

                // Search Results
                if (data.search_results && data.search_results.length > 0) {
                    html += '<div class="result-section"><h4>🔍 Similar Past Cases</h4>';
                    data.search_results.forEach((result, idx) => {
                        html += `<div style="margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 5px;">`;
                        html += `<strong>Case ${idx + 1}</strong><br>`;
                        html += `<em>${result.transcript_text.substring(0, 150)}...</em>`;
                        html += `</div>`;
                    });
                    html += '</div>';
                }

                // Recommendations
                if (data.recommendations && data.recommendations.length > 0) {
                    html += '<div class="result-section"><h4>💡 Recommendations</h4>';
                    data.recommendations.forEach(rec => {
                        html += `<div style="margin-bottom: 20px; padding: 15px; background: #e8f5e9; border-radius: 8px;">`;
                        html += `<h5 style="color: #2e7d32; margin-bottom: 10px;">🎯 ${rec.product_service}</h5>`;
                        html += `<p><strong>Rationale:</strong> ${rec.rationale}</p>`;
                        if (rec.key_benefits && rec.key_benefits.length > 0) {
                            html += `<p><strong>Key Benefits:</strong></p><ul>`;
                            rec.key_benefits.forEach(benefit => {
                                html += `<li>${benefit}</li>`;
                            });
                            html += `</ul>`;
                        }
                        html += `<p><strong>Next Steps:</strong> ${rec.next_steps}</p>`;
                        html += `<p><em>Priority: ${rec.priority} | Confidence: ${rec.confidence}</em></p>`;
                        html += `</div>`;
                    });
                    html += '</div>';
                }

                resultsDiv.innerHTML = html;
            }

            // Chat functions
            async function sendChat() {
                const input = document.getElementById('chatInput');
                const message = input.value.trim();

                if (!message) return;

                // Display user message
                addChatMessage('user', message);
                input.value = '';

                // Show loading
                const loadingId = 'loading-' + Date.now();
                addChatMessage('assistant', '<em>Thinking...</em>', loadingId);

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message })
                    });

                    const data = await response.json();

                    // Remove loading message
                    document.getElementById(loadingId)?.remove();

                    if (data.success) {
                        addChatMessage('assistant', data.answer);
                        // Removed: relevant documents message
                    } else {
                        addChatMessage('assistant', '❌ Error: ' + (data.error || 'Unknown error'));
                    }
                } catch (error) {
                    document.getElementById(loadingId)?.remove();
                    addChatMessage('assistant', '❌ Error: ' + error.message);
                }
            }

            function addChatMessage(role, content, id = null, noStrong = false) {
                const messagesDiv = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `chat-message ${role}`;
                if (id) messageDiv.id = id;

                const label = role === 'user' ? 'You' : 'RASA';
                if (noStrong) {
                    messageDiv.innerHTML = content;
                } else {
                    messageDiv.innerHTML = `<strong>${label}:</strong><br>${content}`;
                }

                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            async function clearChat() {
                if (!confirm('Clear chat history?')) return;

                try {
                    await fetch('/chat/clear', { method: 'POST' });
                    const messagesDiv = document.getElementById('chatMessages');
                    messagesDiv.innerHTML = `
                        <div class="chat-message assistant">
                            <strong>RASA:</strong><br>
                            Chat history cleared. How can I help you?
                        </div>
                    `;
                } catch (error) {
                    alert('Error clearing chat: ' + error.message);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
async def chat(request: ChatRequest):
    """Chat with AI agent about stored transcript data.

    Uses LangChain with conversation memory to answer questions based on stored data.

    Args:
        request: User's chat message and optional session ID

    Returns:
        AI response with relevant information from database
    """
    try:
        logger.info(f"Chat request received: {request.message[:100]}...")

        result = chat_agent.chat(
            user_message=request.message,
            session_id=request.session_id
        )

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return ChatResponse(
            success=False,
            answer="I apologize, but I encountered an error processing your message.",
            error=str(e)
        )


@app.post("/chat/clear")
async def clear_chat():
    """Clear chat conversation memory."""
    try:
        chat_agent.clear_memory()
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

