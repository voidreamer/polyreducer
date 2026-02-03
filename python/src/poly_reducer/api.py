"""REST API for poly-reducer service."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
    HAS_API = True
except ImportError:
    HAS_API = False

if HAS_API:
    from poly_reducer import PolyReducer, analyze
    from poly_reducer.models import MeshAnalysis
    
    # =========================================================================
    # Models
    # =========================================================================
    
    class JobStatus(str, Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"
    
    class ReductionRequest(BaseModel):
        """Request to reduce a mesh."""
        target_tris: Optional[int] = Field(None, description="Target triangle count")
        ratio: Optional[float] = Field(None, ge=0.01, le=1.0, description="Reduction ratio")
        preserve_uvs: bool = Field(True, description="Preserve UV coordinates")
        preserve_normals: bool = Field(True, description="Preserve custom normals")
        preserve_boundaries: bool = Field(True, description="Preserve mesh boundaries")
        output_format: Optional[str] = Field(None, description="Output format (fbx, obj, gltf)")
        webhook_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    
    class LODChainRequest(BaseModel):
        """Request to generate LOD chain."""
        targets: list[int] = Field(..., description="List of target triangle counts")
        preserve_uvs: bool = True
        preserve_normals: bool = True
        webhook_url: Optional[str] = None
    
    class JobResponse(BaseModel):
        """Response for job creation."""
        job_id: str
        status: JobStatus
        message: str
        created_at: datetime
    
    class JobStatusResponse(BaseModel):
        """Response for job status query."""
        job_id: str
        status: JobStatus
        progress: float = 0.0
        original_tris: Optional[int] = None
        final_tris: Optional[int] = None
        output_url: Optional[str] = None
        error_message: Optional[str] = None
        created_at: datetime
        completed_at: Optional[datetime] = None
    
    class AnalysisResponse(BaseModel):
        """Response for mesh analysis."""
        triangle_count: int
        vertex_count: int
        edge_count: int
        face_count: int
        has_uvs: bool
        uv_island_count: int
        sharp_edge_count: int
        boundary_edge_count: int
        material_count: int
        bounds_min: tuple[float, float, float]
        bounds_max: tuple[float, float, float]
        complexity_score: float
        suggested_lods: list[dict]
    
    # =========================================================================
    # In-memory job store (replace with Redis/DB in production)
    # =========================================================================
    
    class JobStore:
        """Simple in-memory job store."""
        
        def __init__(self):
            self.jobs: dict[str, dict] = {}
        
        def create(self, job_type: str, params: dict) -> str:
            job_id = str(uuid.uuid4())[:8]
            self.jobs[job_id] = {
                "id": job_id,
                "type": job_type,
                "status": JobStatus.PENDING,
                "params": params,
                "progress": 0.0,
                "result": None,
                "error": None,
                "created_at": datetime.now(),
                "completed_at": None,
            }
            return job_id
        
        def get(self, job_id: str) -> Optional[dict]:
            return self.jobs.get(job_id)
        
        def update(self, job_id: str, **kwargs):
            if job_id in self.jobs:
                self.jobs[job_id].update(kwargs)
        
        def set_completed(self, job_id: str, result: dict):
            self.update(
                job_id,
                status=JobStatus.COMPLETED,
                result=result,
                completed_at=datetime.now(),
                progress=1.0,
            )
        
        def set_failed(self, job_id: str, error: str):
            self.update(
                job_id,
                status=JobStatus.FAILED,
                error=error,
                completed_at=datetime.now(),
            )
    
    # =========================================================================
    # App
    # =========================================================================
    
    app = FastAPI(
        title="Poly Reducer API",
        description="Smart headless polygon reduction service",
        version="0.1.0",
    )
    
    # Storage paths (configure via environment)
    import os
    UPLOAD_DIR = Path(os.environ.get("POLYREDUCE_UPLOAD_DIR", "/tmp/polyreduce/uploads"))
    OUTPUT_DIR = Path(os.environ.get("POLYREDUCE_OUTPUT_DIR", "/tmp/polyreduce/outputs"))
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Job store
    job_store = JobStore()
    
    # Reducer instance
    reducer: Optional[PolyReducer] = None
    
    @app.on_event("startup")
    async def startup():
        global reducer
        try:
            reducer = PolyReducer()
        except RuntimeError as e:
            print(f"Warning: {e}")
    
    # =========================================================================
    # Endpoints
    # =========================================================================
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "blender_available": reducer is not None,
        }
    
    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze_mesh(file: UploadFile = File(...)):
        """Analyze a mesh file without reducing it."""
        if not reducer:
            raise HTTPException(503, "Blender not available")
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        try:
            analysis = reducer.analyze(file_path)
            
            return AnalysisResponse(
                triangle_count=analysis.triangle_count,
                vertex_count=analysis.vertex_count,
                edge_count=analysis.edge_count,
                face_count=analysis.face_count,
                has_uvs=analysis.has_uvs,
                uv_island_count=analysis.uv_island_count,
                sharp_edge_count=analysis.sharp_edge_count,
                boundary_edge_count=analysis.boundary_edge_count,
                material_count=analysis.material_count,
                bounds_min=analysis.bounds_min,
                bounds_max=analysis.bounds_max,
                complexity_score=analysis.complexity_score,
                suggested_lods=[
                    {"level": l.level, "target_tris": l.target_tris, "ratio": l.ratio}
                    for l in analysis.suggested_lods
                ],
            )
        finally:
            file_path.unlink(missing_ok=True)
    
    @app.post("/reduce", response_model=JobResponse)
    async def reduce_mesh(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        target_tris: Optional[int] = None,
        ratio: Optional[float] = None,
        preserve_uvs: bool = True,
        preserve_normals: bool = True,
    ):
        """Submit a mesh for reduction. Returns job ID for status polling."""
        if not reducer:
            raise HTTPException(503, "Blender not available")
        
        if target_tris is None and ratio is None:
            raise HTTPException(400, "Either target_tris or ratio must be specified")
        
        # Save uploaded file
        upload_id = str(uuid.uuid4())[:8]
        file_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Create job
        job_id = job_store.create("reduce", {
            "file_path": str(file_path),
            "filename": file.filename,
            "target_tris": target_tris,
            "ratio": ratio,
            "preserve_uvs": preserve_uvs,
            "preserve_normals": preserve_normals,
        })
        
        # Process in background
        background_tasks.add_task(process_reduction_job, job_id)
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Job submitted successfully",
            created_at=job_store.get(job_id)["created_at"],
        )
    
    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(job_id: str):
        """Get status of a reduction job."""
        job = job_store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        
        result = job.get("result", {}) or {}
        
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            original_tris=result.get("original_tris"),
            final_tris=result.get("final_tris"),
            output_url=f"/download/{job_id}" if job["status"] == JobStatus.COMPLETED else None,
            error_message=job.get("error"),
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
        )
    
    @app.get("/download/{job_id}")
    async def download_result(job_id: str):
        """Download the reduced mesh."""
        job = job_store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(400, "Job not completed")
        
        output_path = Path(job["result"]["output_path"])
        if not output_path.exists():
            raise HTTPException(404, "Output file not found")
        
        return FileResponse(
            output_path,
            filename=output_path.name,
            media_type="application/octet-stream",
        )
    
    # =========================================================================
    # Background tasks
    # =========================================================================
    
    async def process_reduction_job(job_id: str):
        """Process a reduction job in the background."""
        job = job_store.get(job_id)
        if not job:
            return
        
        job_store.update(job_id, status=JobStatus.PROCESSING, progress=0.1)
        
        params = job["params"]
        file_path = Path(params["file_path"])
        
        # Generate output path
        output_path = OUTPUT_DIR / f"{job_id}_{params['filename']}"
        
        try:
            # Run reduction (blocking, in thread pool)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: reducer.reduce(
                    file_path,
                    target_tris=params.get("target_tris"),
                    ratio=params.get("ratio"),
                    output=output_path,
                    preserve_uvs=params.get("preserve_uvs", True),
                    preserve_normals=params.get("preserve_normals", True),
                ),
            )
            
            if result.success:
                job_store.set_completed(job_id, {
                    "original_tris": result.original_tris,
                    "final_tris": result.final_tris,
                    "output_path": str(result.output_path),
                    "reduction_ratio": result.reduction_ratio,
                })
            else:
                job_store.set_failed(job_id, result.error_message or "Unknown error")
        
        except Exception as e:
            job_store.set_failed(job_id, str(e))
        
        finally:
            # Clean up upload
            file_path.unlink(missing_ok=True)
    
    # =========================================================================
    # Run
    # =========================================================================
    
    def run_server(host: str = "0.0.0.0", port: int = 8000):
        """Run the API server."""
        import uvicorn
        uvicorn.run(app, host=host, port=port)

else:
    app = None
    
    def run_server(*args, **kwargs):
        print("API dependencies not installed. Run: pip install poly-reducer[api]")
