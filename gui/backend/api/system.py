"""
System monitoring API endpoints

Provides GPU usage, memory, temperature, and other system stats.
"""

from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def get_gpu_stats() -> dict:
    """
    Get current GPU statistics.

    Returns:
        Dictionary with GPU usage, memory, temperature, etc.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "available": False,
                "message": "CUDA not available"
            }

        # Get GPU info
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()

        # Get memory stats
        allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(current_device) / 1024**3  # GB
        total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3  # GB

        stats = {
            "available": True,
            "gpu_count": gpu_count,
            "current_device": current_device,
            "device_name": torch.cuda.get_device_name(current_device),
            "memory": {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "used_percent": round((reserved / total) * 100, 1),
            }
        }

        # Try to get nvidia-smi stats (utilization, temperature)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(",")
                if len(values) >= 2:
                    stats["utilization_percent"] = int(values[0].strip())
                    stats["temperature_c"] = int(values[1].strip())

        except Exception as e:
            logger.debug(f"Could not get nvidia-smi stats: {e}")

        return stats

    except Exception as e:
        logger.error(f"Error getting GPU stats: {e}")
        return {
            "available": False,
            "error": str(e)
        }


@router.get("/gpu")
async def get_gpu_status():
    """
    Get current GPU status.

    Returns:
        GPU statistics including utilization, memory, temperature
    """
    return get_gpu_stats()
