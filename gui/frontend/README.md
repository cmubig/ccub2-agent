# CCUB2 Agent - Frontend

Next.js + React Flow + TypeScript + TailwindCSS frontend for CCUB2 Agent.

## Features

- 🎨 **Dark Monotone Design** - Clean, professional UI with no gradients
- 🔄 **Real-time Updates** - WebSocket connection for live pipeline status
- 📊 **Node-based Visualization** - ComfyUI-style workflow display
- 🔍 **Interactive Nodes** - Click nodes for detailed information
- ⚡ **Fast & Responsive** - Optimized performance with React Flow

## Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will start at: `http://localhost:3000`

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/pipeline
```

## Project Structure

```
src/
├── app/                    # Next.js app directory
│   ├── globals.css        # Global styles (dark monotone theme)
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
│
├── components/
│   ├── Pipeline/
│   │   └── PipelineCanvas.tsx    # Main React Flow canvas
│   │
│   ├── Nodes/
│   │   ├── BaseNode.tsx          # Base node component
│   │   ├── VLMDetectorNode.tsx   # VLM detector node
│   │   └── I2IEditorNode.tsx     # I2I editor node
│   │
│   └── Panels/
│       ├── ConfigPanel.tsx       # Pipeline configuration
│       └── NodeDetailPanel.tsx   # Node details display
│
├── hooks/
│   └── useWebSocket.ts     # WebSocket hook
│
└── lib/
    ├── api.ts              # API client
    └── types.ts            # TypeScript types
```

## Design System

### Color Palette (Dark Monotone)

- **Background**: Pure black (#0a0a0a) to dark gray (#1e1e1e)
- **Text**: White (#ffffff) to gray (#737373)
- **Borders**: Dark gray (#2a2a2a) to light gray (#404040)
- **Accent**: White borders only
- **Status Error**: Red (#ef4444) - only colored element

### No Gradients Policy

This project strictly avoids gradients for a clean, professional monotone aesthetic.

## Development

```bash
# Development mode with hot reload
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

## Node Types

1. **Input** - User input configuration
2. **T2I Generator** - Text-to-image generation
3. **VLM Detector** - Cultural accuracy detection
4. **Text KB Query** - Text knowledge base query
5. **CLIP RAG Search** - Image similarity search
6. **Reference Selector** - Best reference selection
7. **Prompt Adapter** - Model-specific prompt optimization
8. **I2I Editor** - Image-to-image editing
9. **Iteration Check** - Score validation
10. **Output** - Final result

## WebSocket Messages

The frontend receives real-time updates via WebSocket:

```typescript
{
  type: "node_update",
  node_id: "vlm_detector",
  status: "processing",
  data: {
    cultural_score: 4.2,
    prompt_score: 6.1,
    issues: ["Issue 1", "Issue 2"]
  },
  timestamp: 1234567890.123
}
```

## Keyboard Shortcuts

- **Scroll**: Pan around canvas
- **Scroll + Ctrl**: Zoom in/out
- **Click Node**: View details
- **Drag Node**: Reposition (optional)
