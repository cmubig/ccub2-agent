"""
History Manager - Store and retrieve pipeline execution history
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from models.history import PipelineHistory, NodeHistory
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


class HistoryManager:
    """
    Manages pipeline execution history.

    Stores detailed execution data including:
    - Prompt transformations
    - RAG search results
    - Node execution details
    - Final outputs
    """

    def __init__(self, history_dir: Optional[Path] = None):
        """
        Initialize history manager.

        Args:
            history_dir: Directory to store history files
        """
        self.history_dir = history_dir or (OUTPUT_DIR / ".history")
        self.history_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"History manager initialized: {self.history_dir}")

    def save_history(self, history: PipelineHistory):
        """
        Save pipeline history to disk.

        Args:
            history: PipelineHistory object
        """
        try:
            # Save as JSON
            history_file = self.history_dir / f"{history.pipeline_id}.json"

            with open(history_file, 'w') as f:
                json.dump(history.dict(), f, indent=2)

            logger.info(f"Saved history: {history.pipeline_id}")

        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def get_history(self, pipeline_id: str) -> Optional[PipelineHistory]:
        """
        Load pipeline history from disk.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            PipelineHistory or None if not found
        """
        try:
            history_file = self.history_dir / f"{pipeline_id}.json"

            if not history_file.exists():
                logger.warning(f"History not found: {pipeline_id}")
                return None

            with open(history_file, 'r') as f:
                data = json.load(f)

            return PipelineHistory(**data)

        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return None

    def list_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all pipeline history summaries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of history summaries, sorted by timestamp (newest first)
        """
        try:
            summaries = []

            # Get all history files
            history_files = sorted(
                self.history_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            # Load summaries
            for history_file in history_files[:limit]:
                try:
                    with open(history_file, 'r') as f:
                        data = json.load(f)

                    history = PipelineHistory(**data)
                    summaries.append(history.to_summary())

                except Exception as e:
                    logger.warning(f"Failed to load {history_file}: {e}")
                    continue

            logger.info(f"Loaded {len(summaries)} history entries")
            return summaries

        except Exception as e:
            logger.error(f"Failed to list history: {e}")
            return []

    def delete_history(self, pipeline_id: str) -> bool:
        """
        Delete pipeline history.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            True if deleted, False otherwise
        """
        try:
            history_file = self.history_dir / f"{pipeline_id}.json"

            if history_file.exists():
                history_file.unlink()
                logger.info(f"Deleted history: {pipeline_id}")
                return True
            else:
                logger.warning(f"History not found: {pipeline_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete history: {e}")
            return False

    def cleanup_old_history(self, keep_days: int = 30):
        """
        Delete history older than specified days.

        Args:
            keep_days: Number of days to keep history
        """
        import time

        try:
            current_time = time.time()
            max_age = keep_days * 24 * 3600

            deleted_count = 0
            for history_file in self.history_dir.glob("*.json"):
                file_age = current_time - history_file.stat().st_mtime

                if file_age > max_age:
                    history_file.unlink()
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old history entries")

        except Exception as e:
            logger.error(f"Failed to cleanup old history: {e}")


# Global history manager instance
_history_manager: Optional[HistoryManager] = None


def get_history_manager() -> HistoryManager:
    """Get or create global history manager instance."""
    global _history_manager

    if _history_manager is None:
        _history_manager = HistoryManager()

    return _history_manager
