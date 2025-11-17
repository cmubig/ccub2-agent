"""
Nodes API Endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging

from models.node import NodeData, NodeType
from services.pipeline_runner import pipeline_runner

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{node_id}", response_model=NodeData)
async def get_node_detail(node_id: str):
    """
    Get detailed information about a specific node

    Args:
        node_id: The node ID (e.g., "vlm_detector", "t2i_generator")
    """
    try:
        node_data = pipeline_runner.get_node_data(node_id)

        if not node_data:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        return node_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_all_nodes():
    """Get information about all nodes in the pipeline"""
    try:
        nodes = pipeline_runner.get_all_nodes()
        return {"nodes": nodes}
    except Exception as e:
        logger.error(f"Error getting all nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{node_id}/outputs")
async def get_node_outputs(node_id: str):
    """
    Get outputs (images, text, etc.) from a specific node

    Args:
        node_id: The node ID
    """
    try:
        outputs = pipeline_runner.get_node_outputs(node_id)

        if outputs is None:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        return {"outputs": outputs}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node outputs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
