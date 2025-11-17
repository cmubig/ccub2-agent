"""
Image Serving API Endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Base directory for country pack images
DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data"
COUNTRY_PACKS_DIR = DATA_ROOT / "country_packs"

# Base directory for pipeline outputs
GUI_ROOT = Path(__file__).parent.parent.parent  # ccub2-agent/gui
OUTPUTS_DIR = GUI_ROOT / "outputs"


@router.get("/serve/{country}/{category}/{filename}")
async def serve_image(country: str, category: str, filename: str):
    """
    Serve an image file from the country packs directory

    Example: /api/images/serve/korea/architecture/CSx4IS2VM24qjZok4FRA.jpg
    """
    try:
        # Build the file path
        image_path = COUNTRY_PACKS_DIR / country / "images" / category / filename

        # Security check: ensure the resolved path is within COUNTRY_PACKS_DIR
        resolved_path = image_path.resolve()
        if not str(resolved_path).startswith(str(COUNTRY_PACKS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if file exists
        if not resolved_path.exists() or not resolved_path.is_file():
            logger.warning(f"Image not found: {resolved_path}")
            raise HTTPException(status_code=404, detail="Image not found")

        # Return the image file
        return FileResponse(
            resolved_path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/serve-by-path")
async def serve_image_by_path(path: str):
    """
    Serve an image by full path (for flexibility)

    Query param:
        path=<project-root>/data/country_packs/korea/images/...
        or path=<project-root>/gui/outputs/[pipeline-id]/step_1_edited.png
    """
    try:
        image_path = Path(path)

        # Security check: ensure path is within allowed directories
        resolved_path = image_path.resolve()
        data_root_resolved = DATA_ROOT.resolve()
        outputs_root_resolved = OUTPUTS_DIR.resolve()

        # Allow either data directory OR outputs directory
        is_in_data = str(resolved_path).startswith(str(data_root_resolved))
        is_in_outputs = str(resolved_path).startswith(str(outputs_root_resolved))

        if not (is_in_data or is_in_outputs):
            logger.warning(f"Access denied for path: {resolved_path}")
            logger.warning(f"  Not in data: {data_root_resolved}")
            logger.warning(f"  Not in outputs: {outputs_root_resolved}")
            raise HTTPException(
                status_code=403,
                detail="Access denied: path must be in data or outputs directory"
            )

        # Check if file exists
        if not resolved_path.exists() or not resolved_path.is_file():
            logger.warning(f"Image not found: {resolved_path}")
            raise HTTPException(status_code=404, detail="Image not found")

        # Determine media type based on extension
        ext = resolved_path.suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        media_type = media_types.get(ext, 'application/octet-stream')

        # Return the image file
        return FileResponse(
            resolved_path,
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=3600"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image by path: {e}")
        raise HTTPException(status_code=500, detail=str(e))
