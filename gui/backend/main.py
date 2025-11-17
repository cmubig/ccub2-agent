"""
CCUB2 Agent GUI - FastAPI Backend

This backend provides REST API and WebSocket endpoints for the
React Flow frontend to interact with the CCUB2 agent pipeline.
"""

import sys
from pathlib import Path

# Add ccub2-agent root to Python path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from api.pipeline import router as pipeline_router
from api.nodes import router as nodes_router
from api.system import router as system_router
from api.history import router as history_router
from api.images import router as images_router
from api.websocket import manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CCUB2 Agent API",
    description="Backend API for CCUB2 Agent Node-based UI",
    version="1.0.0"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pipeline_router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(nodes_router, prefix="/api/nodes", tags=["nodes"])
app.include_router(system_router, prefix="/api/system", tags=["system"])
app.include_router(history_router, prefix="/api/history", tags=["history"])
app.include_router(images_router, prefix="/api/images", tags=["images"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "CCUB2 Agent API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.websocket("/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pipeline updates

    Sends messages in the format:
    {
        "type": "node_update" | "pipeline_status" | "error",
        "node_id": str,
        "status": "pending" | "processing" | "completed" | "error",
        "data": {...}
    }
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            logger.info(f"Received from client: {data}")

            # Echo back (for testing)
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")


if __name__ == "__main__":
    logger.info("Starting CCUB2 Agent API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
