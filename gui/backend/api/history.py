"""
History API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from services.history_manager import get_history_manager
from models.history import PipelineHistory

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/list")
async def list_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get list of pipeline execution history.

    Args:
        limit: Maximum number of entries to return

    Returns:
        List of history summaries
    """
    try:
        history_mgr = get_history_manager()
        summaries = history_mgr.list_history(limit=limit)

        return summaries

    except Exception as e:
        logger.error(f"Failed to list history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}")
async def get_history(pipeline_id: str) -> PipelineHistory:
    """
    Get detailed history for a specific pipeline execution.

    Args:
        pipeline_id: Pipeline ID

    Returns:
        Complete pipeline history
    """
    try:
        history_mgr = get_history_manager()
        history = history_mgr.get_history(pipeline_id)

        if history is None:
            raise HTTPException(status_code=404, detail=f"History not found: {pipeline_id}")

        return history

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{pipeline_id}")
async def delete_history(pipeline_id: str) -> Dict[str, Any]:
    """
    Delete a specific pipeline history.

    Args:
        pipeline_id: Pipeline ID

    Returns:
        Success message
    """
    try:
        history_mgr = get_history_manager()
        success = history_mgr.delete_history(pipeline_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"History not found: {pipeline_id}")

        return {"message": "History deleted successfully", "pipeline_id": pipeline_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_old_history(keep_days: int = 30) -> Dict[str, Any]:
    """
    Delete history older than specified days.

    Args:
        keep_days: Number of days to keep history

    Returns:
        Success message
    """
    try:
        history_mgr = get_history_manager()
        history_mgr.cleanup_old_history(keep_days=keep_days)

        return {"message": f"Cleaned up history older than {keep_days} days"}

    except Exception as e:
        logger.error(f"Failed to cleanup history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
