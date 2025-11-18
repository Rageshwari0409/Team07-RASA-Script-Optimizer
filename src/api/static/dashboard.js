let selectedFile = null;
let currentTab = 'chat';

function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    event.target.classList.add('active');
    if (tab === 'transcript') document.getElementById('transcriptTab').classList.add('active');
    if (tab === 'chat') document.getElementById('chatTab').classList.add('active');
}

function newChat() {
    document.getElementById('chatMessages').innerHTML = '';
    fetch('/chat/clear', { method: 'POST' });
    loadChatHistory();
}

async function logout() {
    await fetch('/logout', { method: 'POST' });
    window.location.href = '/login';
}

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

    // Show storage confirmation and switch to chat
    if (data.transcript_id) {
        html += `<div style="background: #10a37f; padding: 10px; border-radius: 5px; margin-bottom: 15px; color: white;">`;
        html += `💾 <strong>Stored in Database - You can now chat about this transcript!</strong>`;
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

// Chat functions
let chatHistory = [];

async function loadChatHistory() {
    try {
        const response = await fetch('/chat/history');
        const data = await response.json();
        if (data.success) {
            chatHistory = data.history || [];
            renderChatHistory();
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

function renderChatHistory() {
    const sidebar = document.getElementById('chatHistorySidebar');
    if (!sidebar) return;

    let html = '';
    if (chatHistory.length === 0) {
        html += '<p style="color: #999; font-size: 12px; padding: 12px;">No chat history yet</p>';
    } else {
        chatHistory.forEach((chat, idx) => {
            html += `<div class="chat-history-item" onclick="loadChat(${idx})">`;
            html += `${chat.title}`;
            html += `</div>`;
        });
    }
    sidebar.innerHTML = html;
}

function loadChat(index) {
    const chat = chatHistory[index];
    const messagesDiv = document.getElementById('chatMessages');
    messagesDiv.innerHTML = '';

    chat.messages.forEach(msg => {
        addChatMessage(msg.role, msg.content);
    });
}

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
            loadChatHistory();
        } else {
            addChatMessage('assistant', '❌ Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        document.getElementById(loadingId)?.remove();
        addChatMessage('assistant', '❌ Error: ' + error.message);
    }
}

function addChatMessage(role, content, id = null) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    if (id) messageDiv.id = id;

    const avatar = document.createElement('div');
    avatar.className = `message-avatar ${role}`;
    avatar.textContent = role === 'user' ? 'U' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);

    const container = document.querySelector('.chat-container');
    container.scrollTop = container.scrollHeight;
}

async function clearChat() {
    try {
        await fetch('/chat/clear', { method: 'POST' });
        const messagesDiv = document.getElementById('chatMessages');
        messagesDiv.innerHTML = '';
        loadChatHistory();
    } catch (error) {
        alert('Error clearing chat: ' + error.message);
    }
}

// Load chat history on page load
window.addEventListener('load', loadChatHistory);