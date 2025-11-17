# CCUB2 Agent - Node-based GUI

Complete node-based workflow visualization for CCUB2 Agent with React Flow + FastAPI.

## 🎨 Features

- **Dark Monotone Design** - Professional, gradient-free UI
- **Real-time Updates** - WebSocket-powered live status
- **Interactive Nodes** - Click to view detailed information
- **Visual Pipeline** - See exactly what each agent is doing
- **ComfyUI-style** - Familiar node-based workflow

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Frontend (Next.js + React Flow)        │
│  http://localhost:3000                  │
│  - Node visualization                   │
│  - Real-time updates                    │
│  - Configuration panel                  │
└──────────────┬──────────────────────────┘
               │ REST API + WebSocket
               ▼
┌─────────────────────────────────────────┐
│  Backend (FastAPI)                      │
│  http://localhost:8000                  │
│  - Pipeline orchestration               │
│  - Node state management                │
│  - WebSocket broadcasting               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  CCUB2 Agent Core                       │
│  - VLM Detector                         │
│  - Reference Selector                   │
│  - I2I Editors                          │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Start Backend (Terminal 1)

```bash
cd gui/backend

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python main.py
```

Backend will run at: `http://localhost:8000`

### 2. Start Frontend (Terminal 2)

```bash
cd gui/frontend

# Install dependencies
npm install

# Start Next.js dev server
npm run dev
```

Frontend will run at: `http://localhost:3000`

### 3. Open Browser

Navigate to: **http://localhost:3000**

## 📸 What You'll See

### Main Canvas

```
┌────────────────────────────────────────────────────┐
│  [Connected] [Pipeline Running]  [Show Config]     │
├────────────────────────────────────────────────────┤
│                                                     │
│                  ┌──────────┐                      │
│                  │  Input   │                      │
│                  └────┬─────┘                      │
│                       ▼                            │
│                  ┌──────────┐                      │
│                  │T2I Gen ✅│                      │
│                  └────┬─────┘                      │
│                       ▼                            │
│                  ┌──────────┐                      │
│                  │VLM Det 🔄│                      │
│                  │ 4.2/10   │  ← Click for details │
│                  └─┬────┬───┘                      │
│                    │    │                          │
│         ┌──────────┘    └──────────┐              │
│         ▼                           ▼              │
│    ┌────────┐                 ┌────────┐          │
│    │Text KB│                  │CLIP RAG│          │
│    └───┬────┘                 └───┬────┘          │
│        └──────────┬───────────────┘               │
│                   ▼                               │
│            ┌─────────────┐                        │
│            │Reference   │                        │
│            │Selector    │                        │
│            └──────┬──────┘                        │
│                   ▼                               │
│            ┌─────────────┐                        │
│            │ I2I Editor  │                        │
│            │  Step 15/28 │                        │
│            │ ████░░░░ 54%│                        │
│            └─────────────┘                        │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Configuration Panel (Right Side)

```
┌──────────────────────────────┐
│ Pipeline Configuration        │
├──────────────────────────────┤
│ Prompt:                       │
│ ┌──────────────────────────┐ │
│ │A Korean woman in hanbok  │ │
│ └──────────────────────────┘ │
│                               │
│ Country: [Korea ▼]            │
│ Category: [Traditional ▼]     │
│                               │
│ T2I Model: [SDXL ▼]           │
│ I2I Model: [Qwen ▼]           │
│                               │
│ Max Iterations: [3 ━━━○━ 5]  │
│ Target Score: [8.0 ━━○━━ 10] │
│                               │
│ [🚀 Start Pipeline]           │
└──────────────────────────────┘
```

### Node Detail Panel (Click any node)

```
┌──────────────────────────────┐
│ VLM Detector            [×]  │
├──────────────────────────────┤
│ Status: PROCESSING 🔄         │
│                               │
│ Scores:                       │
│ Cultural: 4.2/10              │
│ Prompt: 6.1/10                │
│                               │
│ Detected Issues:              │
│ • Insufficient jeogori collar │
│ • Missing dongjeong (collar)  │
│ • Inappropriate goreum colors │
│                               │
│ Context Retrieved:            │
│ • Text KB: 5 entries          │
│ • CLIP RAG: 10 images         │
└──────────────────────────────┘
```

## 🎯 Workflow

1. **Configure**: Set prompt, country, models
2. **Start**: Click "Start Pipeline"
3. **Watch**: See nodes light up as they process
4. **Inspect**: Click nodes to see detailed information
5. **Track**: Monitor VLM scores and issues in real-time
6. **Iterate**: Watch the system improve images automatically

## 🔧 Node Types & Status

### Node Types

- 🎤 **Input** - User configuration
- 🎨 **T2I Generator** - Initial image generation
- 🔍 **VLM Detector** - Cultural accuracy analysis
- 📚 **Text KB Query** - Knowledge base retrieval
- 🖼️ **CLIP RAG Search** - Image similarity search
- 🎯 **Reference Selector** - Best reference picking
- ✍️ **Prompt Adapter** - Model-specific optimization
- ✏️ **I2I Editor** - Image editing
- ✅ **Iteration Check** - Score validation
- 📦 **Output** - Final result

### Status Indicators

- ⏸ **Pending** - Not started (Gray)
- 🔄 **Processing** - Running (White, pulsing)
- ✅ **Completed** - Done (White)
- ❌ **Error** - Failed (Red)

## 📡 API Endpoints

### REST API (http://localhost:8000)

- `POST /api/pipeline/start` - Start pipeline
- `GET /api/pipeline/status` - Get status
- `POST /api/pipeline/stop` - Stop pipeline
- `GET /api/pipeline/countries` - List countries
- `GET /api/pipeline/models` - List models
- `GET /api/nodes/{id}` - Node details

### WebSocket (ws://localhost:8000/ws/pipeline)

Real-time updates:
```json
{
  "type": "node_update",
  "node_id": "vlm_detector",
  "status": "processing",
  "data": {
    "cultural_score": 4.2,
    "issues": ["..."]
  }
}
```

## 🎨 Design System

### Dark Monotone Theme

- **No gradients** - Clean, professional look
- **Monochrome** - Black, white, grays only
- **One color** - Red for errors only
- **Sharp borders** - No rounded corners abuse
- **Subtle animations** - Pulse for processing

### Colors

- Background: `#0a0a0a` → `#1e1e1e`
- Text: `#ffffff` → `#737373`
- Borders: `#2a2a2a` → `#404040`
- Error: `#ef4444`

## 🔍 Troubleshooting

### Backend won't start

```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Restart backend
python gui/backend/main.py
```

### Frontend won't start

```bash
# Check if port 3000 is in use
lsof -i :3000

# Kill process if needed
kill -9 <PID>

# Clear cache and restart
rm -rf .next
npm run dev
```

### WebSocket not connecting

1. Ensure backend is running at port 8000
2. Check CORS settings in `backend/main.py`
3. Check browser console for errors
4. Try refreshing the page

### Nodes not updating

1. Check WebSocket connection status (top panel)
2. Open browser DevTools → Network → WS tab
3. Verify messages are being received
4. Check backend logs for errors

## 📁 Project Structure

```
gui/
├── backend/              # FastAPI backend
│   ├── main.py          # Entry point
│   ├── api/             # API routes
│   ├── models/          # Pydantic models
│   ├── services/        # Business logic
│   └── requirements.txt
│
└── frontend/            # Next.js frontend
    ├── src/
    │   ├── app/         # Next.js pages
    │   ├── components/  # React components
    │   ├── hooks/       # Custom hooks
    │   └── lib/         # Utilities
    ├── package.json
    └── tailwind.config.js
```

## 🚧 Current Status

✅ Backend API complete
✅ WebSocket real-time updates
✅ React Flow visualization
✅ Node components (Base, VLM, I2I)
✅ Configuration panel
✅ Node detail panel
✅ Dark monotone design

🚧 TODO: Integrate with actual CCUB2 Agent
🚧 TODO: Image display in nodes
🚧 TODO: History/replay feature

## 📝 Notes

- This is a **visualization layer** on top of CCUB2 Agent
- Backend currently uses **mock data** for demonstration
- To integrate with real CCUB2, modify `services/pipeline_runner.py`
- Design follows **dark monotone** principles (no gradients!)
- WebSocket ensures **real-time** updates during pipeline execution

## 🤝 Contributing

1. Backend changes → `gui/backend/`
2. Frontend changes → `gui/frontend/`
3. Follow dark monotone design guidelines
4. Test both REST API and WebSocket
5. Update documentation

## 📄 License

Same as CCUB2 Agent project.
