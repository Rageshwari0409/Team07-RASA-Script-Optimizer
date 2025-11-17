# Sales Transcript Analysis Agent

AI-powered agent that analyzes sales conversations and extracts requirements, recommendations, and summaries using Azure OpenAI and LiteLLM.

## 🚀 Quick Start

### 1. Test Your Setup
```bash
python test_setup.py
```

### 2. Analyze a Sample Transcript
```bash
python examples/analyze_text.py
```

### 3. Start the API Server
```bash
python run_api.py
```
Then open: http://localhost:8000/docs

## Features

- 📝 Text & Audio transcript analysis
- 🤖 AI-powered insights via Azure OpenAI + LiteLLM
- 🚀 FastAPI REST API
- 📊 Structured JSON output
- 🔄 Automatic retry with exponential backoff for rate limit handling


## Project Structure

```
Captsone/
├── config/              # Configuration
│   ├── config.yaml     # Settings
│   ├── prompts.yaml    # LLM prompts
│   └── .env            # Your credentials
├── src/                # Source code
│   ├── agent/         # Analysis logic
│   ├── api/           # FastAPI app
│   └── utils/         # Utilities
├── data/              # Sample data
│   └── text/          # Sample transcripts
├── examples/          # Examples
│   ├── analyze_text.py
│   ├── api_client_example.py
│   └── demo.html
├── notebooks/         # Jupyter tutorials
├── test_setup.py      # Test credentials
└── run_api.py         # Start server
```

## Setup

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pydantic python-multipart litellm openai pyyaml python-dotenv requests colorlog
```

### 2. Configure Credentials

Edit `config/.env` with your Azure OpenAI credentials:
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

### 3. Test Setup
```bash
python test_setup.py
```



## Usage

### Start the API Server
```bash
python run_api.py
```
Open: http://localhost:8000/docs

### Analyze a Transcript
```bash
python examples/analyze_text.py
```

### Use the Web Demo
1. Start API: `python run_api.py`
2. Open: `examples/demo.html` in browser

## API Endpoints

- `POST /analyze/text` - Analyze text transcript
- `POST /analyze/audio` - Analyze audio file
- `GET /health` - Health check

## Analysis Output

The agent provides structured analysis including:

```json
{
  "requirements": [
    {
      "requirement": "Mobile access for sales team",
      "priority": "High",
      "mentioned_by": "Client",
      "context": "Our sales team is always on the go"
    }
  ],
  "recommendations": [
    {
      "recommendation": "Implement mobile-first CRM solution",
      "rationale": "Addresses primary need for mobile access",
      "product_fit": "Our platform has native mobile apps",
      "priority": "High"
    }
  ],
  "summary": {
    "overview": "Discussion about CRM replacement",
    "client_needs": "Mobile access, integrations, reporting",
    "pain_points": "Outdated system, slow performance",
    "opportunities": "Enterprise plan with volume discount",
    "next_steps": "Schedule demo for next Tuesday",
    "sentiment": "Positive",
    "engagement_level": "High"
  },
  "key_points": [...],
  "action_items": [...]
}
```

## Sample Data

3 sample sales transcripts included in `data/text/`:
- CRM Solution Sales Call
- Cybersecurity Solution Sales Call
- Marketing Analytics Sales Call

## Rate Limit Handling

The application automatically handles Azure OpenAI rate limits with **exponential backoff retry logic**:

- ✅ Automatic retry on rate limit errors (HTTP 429)
- ✅ Exponential backoff: 2s → 4s → 8s → 16s → 32s
- ✅ Up to 5 retry attempts with max 120s total
- ✅ Detailed logging of retry attempts
- ✅ No code changes needed - works transparently

**Test rate limit handling:**
```bash
python test_rate_limit.py
```

**See full documentation:** [docs/RATE_LIMIT_HANDLING.md](docs/RATE_LIMIT_HANDLING.md)

## Troubleshooting

### Authentication Error
- Check `config/.env` has correct credentials
- Get fresh API key from Azure Portal
- Verify deployment names in Azure OpenAI Studio

### Rate Limit Errors
- The application automatically retries with exponential backoff
- Check logs for retry attempts: `tail -f logs/app.log | grep "Backing off"`
- If persistent, consider upgrading Azure OpenAI quota

### Module Not Found
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install fastapi uvicorn pydantic python-multipart litellm openai pyyaml python-dotenv requests colorlog backoff
```

### Deployment Not Found
- Go to https://oai.azure.com/
- Verify deployment exists
- Update deployment name in `config/.env`

