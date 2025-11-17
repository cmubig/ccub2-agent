# CCUB2 Agent - Backend API

FastAPI backend for the CCUB2 Agent Node-based UI.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

Server will start at: `http://localhost:8000`

## API Endpoints

### Pipeline
- `POST /api/pipeline/start` - Start pipeline execution
- `GET /api/pipeline/status` - Get current status
- `POST /api/pipeline/stop` - Stop pipeline
- `GET /api/pipeline/countries` - Get available countries
- `GET /api/pipeline/models` - Get available models

### Nodes
- `GET /api/nodes/{node_id}` - Get node details
- `GET /api/nodes` - Get all nodes
- `GET /api/nodes/{node_id}/outputs` - Get node outputs

### WebSocket
- `WS /ws/pipeline` - Real-time pipeline updates

## WebSocket Message Format

```json
{
  "type": "node_update",
  "node_id": "vlm_detector",
  "status": "processing",
  "data": {
    "cultural_score": 4.2,
    "prompt_score": 6.1
  },
  "timestamp": 1234567890.123
}
```

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
