"""
Pipeline API Endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging

from models.pipeline import (
    PipelineRequest,
    PipelineResponse,
    PipelineState,
    PipelineStatus
)
from services.pipeline_runner import pipeline_runner

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/start", response_model=PipelineResponse)
async def start_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Start the CCUB2 pipeline execution

    This endpoint starts the pipeline in the background and returns immediately.
    Use WebSocket at /ws/pipeline to receive real-time updates.
    """
    try:
        # Validate configuration
        config = request.config

        if not config.prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        if not config.country:
            raise HTTPException(status_code=400, detail="Country is required")

        # Check if pipeline is already running
        if pipeline_runner.is_running():
            raise HTTPException(
                status_code=409,
                detail="Pipeline is already running. Please wait for it to complete."
            )

        # Start pipeline in background
        pipeline_id = await pipeline_runner.start(config)
        background_tasks.add_task(pipeline_runner.run)

        logger.info(f"Pipeline started with ID: {pipeline_id}")

        return PipelineResponse(
            success=True,
            message="Pipeline started successfully",
            pipeline_id=pipeline_id,
            state=pipeline_runner.get_state()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=PipelineState)
async def get_pipeline_status():
    """Get current pipeline status"""
    return pipeline_runner.get_state()


@router.post("/stop")
async def stop_pipeline():
    """Stop the currently running pipeline"""
    try:
        if not pipeline_runner.is_running():
            raise HTTPException(status_code=400, detail="No pipeline is currently running")

        await pipeline_runner.stop()

        return {"success": True, "message": "Pipeline stopped"}
    except Exception as e:
        logger.error(f"Error stopping pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_pipeline_history():
    """Get pipeline execution history"""
    return {"history": []}


@router.get("/countries")
async def get_available_countries():
    """Get list of available countries"""
    return {
        "countries": [
            {"id": "korea", "name": "Korea", "image_count": 338},
            {"id": "japan", "name": "Japan", "image_count": 201},
            {"id": "china", "name": "China", "image_count": 187},
            {"id": "india", "name": "India", "image_count": 243},
        ]
    }


@router.get("/models")
async def get_available_models():
    """Get list of available T2I and I2I models"""
    return {
        "t2i_models": [
            {"id": "sdxl", "name": "Stable Diffusion XL", "description": "Fast and reliable"},
            {"id": "flux", "name": "FLUX.1-dev", "description": "SOTA quality"},
            {"id": "sd35", "name": "SD 3.5 Medium", "description": "Latest architecture"},
            {"id": "gemini", "name": "Gemini Imagen 3", "description": "Best photorealism"},
        ],
        "i2i_models": [
            {"id": "qwen", "name": "Qwen-Image-Edit", "description": "Best for detail (recommended)"},
            {"id": "flux", "name": "FLUX Kontext", "description": "Style preservation"},
            {"id": "sdxl", "name": "SDXL InstructPix2Pix", "description": "Fast editing"},
            {"id": "sd35", "name": "SD3.5 Inpainting", "description": "Region-specific"},
        ]
    }


@router.post("/jobs/approve")
async def approve_jobs(data: Dict[str, Any]):
    """
    Approve selected job proposals and create jobs in Firebase

    Request body:
    {
        "proposal_ids": ["pipeline_id_gap_0", "pipeline_id_gap_1"],
        "pipeline_id": "abc123"
    }
    """
    try:
        proposal_ids = data.get("proposal_ids", [])
        pipeline_id = data.get("pipeline_id")

        if not proposal_ids:
            raise HTTPException(status_code=400, detail="No proposals selected")

        if not pipeline_id or pipeline_runner.pipeline_id != pipeline_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid pipeline ID or pipeline not running"
            )

        # Get pending proposals
        proposals = pipeline_runner.pending_job_proposals
        if not proposals:
            raise HTTPException(status_code=404, detail="No pending job proposals")

        # Filter approved proposals
        approved_proposals = [p for p in proposals if p["id"] in proposal_ids]

        if not approved_proposals:
            raise HTTPException(status_code=404, detail="None of the selected proposals were found")

        # Create jobs
        from services.component_manager import get_component_manager
        comp_mgr = get_component_manager(approved_proposals[0]["country"])
        job_creator = comp_mgr.get_job_creator()

        created_jobs = []
        for proposal in approved_proposals:
            try:
                job_id = job_creator.create_job(
                    country=proposal["country"],
                    category=proposal["category"],
                    subcategory=proposal.get("subcategory", ""),
                    keywords=proposal.get("keywords", []),
                    description=proposal["reason"],
                    points=proposal.get("points", 50),
                    target_count=proposal.get("target_count", 50),
                )

                if job_id:
                    created_jobs.append({
                        "job_id": job_id,
                        "category": proposal["category"],
                        "subcategory": proposal.get("subcategory"),
                    })

                    logger.info(f"✓ Job approved and created: {job_id}")

                    # Broadcast success
                    from api.websocket import manager
                    await manager.broadcast({
                        "type": "job_creation",
                        "status": "created",
                        "job_id": job_id,
                        "gap": {
                            "category": proposal["category"],
                            "subcategory": proposal.get("subcategory"),
                        },
                        "message": f"Job {job_id} created",
                        "timestamp": __import__('time').time()
                    })

            except Exception as e:
                logger.error(f"Failed to create job: {e}")
                # Continue with other jobs

        # Remove approved proposals from pending list
        pipeline_runner.pending_job_proposals = [
            p for p in proposals if p["id"] not in proposal_ids
        ]

        return {
            "success": True,
            "message": f"Created {len(created_jobs)} jobs",
            "created_jobs": created_jobs
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/reject")
async def reject_jobs(data: Dict[str, Any]):
    """
    Reject selected job proposals

    Request body:
    {
        "proposal_ids": ["pipeline_id_gap_0"],
        "pipeline_id": "abc123"
    }
    """
    try:
        proposal_ids = data.get("proposal_ids", [])
        pipeline_id = data.get("pipeline_id")

        if not proposal_ids:
            raise HTTPException(status_code=400, detail="No proposals selected")

        if not pipeline_id or pipeline_runner.pipeline_id != pipeline_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid pipeline ID or pipeline not running"
            )

        # Remove rejected proposals from pending list
        proposals = pipeline_runner.pending_job_proposals
        pipeline_runner.pending_job_proposals = [
            p for p in proposals if p["id"] not in proposal_ids
        ]

        logger.info(f"Rejected {len(proposal_ids)} job proposals")

        return {
            "success": True,
            "message": f"Rejected {len(proposal_ids)} proposals"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
