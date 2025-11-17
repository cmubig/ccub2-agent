"""
WebSocket Manager for Real-time Updates
"""

from fastapi import WebSocket
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and store a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[Any, Any], websocket: WebSocket):
        """Send a message to a specific client"""
        await websocket.send_text(json.dumps(message))

    async def broadcast(self, message: Dict[Any, Any]):
        """Send a message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_node_update(
        self,
        node_id: str,
        status: str,
        data: Dict[Any, Any]
    ):
        """
        Broadcast a node status update

        Args:
            node_id: The ID of the node (e.g., "vlm_detector")
            status: Node status ("pending", "processing", "completed", "error")
            data: Additional node data (scores, images, etc.)
        """
        message = {
            "type": "node_update",
            "node_id": node_id,
            "status": status,
            "data": data,
            "timestamp": __import__('time').time()
        }
        await self.broadcast(message)
        logger.info(f"Broadcasted update for node {node_id}: {status}")

    async def broadcast_pipeline_status(
        self,
        status: str,
        current_node: str = None,
        progress: float = 0.0
    ):
        """
        Broadcast overall pipeline status

        Args:
            status: Pipeline status ("idle", "running", "completed", "error")
            current_node: Currently executing node ID
            progress: Overall progress (0.0 to 1.0)
        """
        message = {
            "type": "pipeline_status",
            "status": status,
            "current_node": current_node,
            "progress": progress,
            "timestamp": __import__('time').time()
        }
        await self.broadcast(message)
        logger.info(f"Broadcasted pipeline status: {status}")

    async def broadcast_error(self, error: str, node_id: str = None):
        """Broadcast an error message"""
        message = {
            "type": "error",
            "error": error,
            "node_id": node_id,
            "timestamp": __import__('time').time()
        }
        await self.broadcast(message)
        logger.error(f"Broadcasted error: {error}")

    async def broadcast_gpu_stats(self, stats: Dict[Any, Any]):
        """
        Broadcast GPU statistics

        Args:
            stats: GPU stats dictionary from api.system.get_gpu_stats()
        """
        message = {
            "type": "gpu_stats",
            "stats": stats,
            "timestamp": __import__('time').time()
        }
        await self.broadcast(message)
        logger.debug(f"Broadcasted GPU stats")

    async def broadcast_progress(
        self,
        node_id: str,
        current_step: int,
        total_steps: int,
        eta_seconds: float = None,
        message: str = None
    ):
        """
        Broadcast real-time progress updates for long-running operations

        Args:
            node_id: The node being processed (e.g., "t2i_generator", "i2i_editor")
            current_step: Current step number (e.g., 15)
            total_steps: Total number of steps (e.g., 40)
            eta_seconds: Estimated time remaining in seconds
            message: Optional progress message (e.g., "Generating image...")
        """
        progress_data = {
            "type": "progress",
            "node_id": node_id,
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_percent": (current_step / total_steps * 100) if total_steps > 0 else 0,
            "eta_seconds": eta_seconds,
            "message": message,
            "timestamp": __import__('time').time()
        }
        await self.broadcast(progress_data)
        logger.debug(f"Progress: {node_id} - {current_step}/{total_steps}")


# Global connection manager instance
manager = ConnectionManager()
