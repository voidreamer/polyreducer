"""Background worker for processing reduction jobs from a queue."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import boto3
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

from poly_reducer import PolyReducer


class Worker:
    """Background worker that processes jobs from a queue.
    
    Supports:
    - Redis (redis://host:port/db)
    - AWS SQS (sqs://queue-name or full URL)
    - Local file-based queue (file:///path/to/jobs.json)
    
    Example:
        worker = Worker("redis://localhost:6379/0")
        worker.run()
    """
    
    def __init__(
        self,
        queue_url: str,
        upload_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        s3_bucket: Optional[str] = None,
    ) -> None:
        """Initialize worker.
        
        Args:
            queue_url: Queue connection URL
            upload_dir: Local directory for downloaded files
            output_dir: Local directory for output files
            s3_bucket: S3 bucket for input/output (if using S3)
        """
        self.queue_url = queue_url
        self.upload_dir = upload_dir or Path(os.environ.get("POLYREDUCE_UPLOAD_DIR", "/tmp/polyreduce/uploads"))
        self.output_dir = output_dir or Path(os.environ.get("POLYREDUCE_OUTPUT_DIR", "/tmp/polyreduce/outputs"))
        self.s3_bucket = s3_bucket or os.environ.get("POLYREDUCE_S3_BUCKET")
        
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.reducer = PolyReducer()
        self.running = False
        
        # Parse queue URL
        parsed = urlparse(queue_url)
        self.queue_type = parsed.scheme
        
        if self.queue_type == "redis":
            if not HAS_REDIS:
                raise ImportError("redis package required for Redis queue")
            self.redis = redis.from_url(queue_url)
            self.queue_name = os.environ.get("POLYREDUCE_QUEUE_NAME", "polyreduce:jobs")
        elif self.queue_type == "sqs":
            if not HAS_BOTO:
                raise ImportError("boto3 package required for SQS queue")
            self.sqs = boto3.client("sqs")
            self.queue_name = queue_url  # Full SQS URL or queue name
        else:
            raise ValueError(f"Unsupported queue type: {self.queue_type}")
        
        # S3 client for file transfer
        if self.s3_bucket and HAS_BOTO:
            self.s3 = boto3.client("s3")
        else:
            self.s3 = None
    
    def run(self) -> None:
        """Run the worker loop."""
        self.running = True
        
        # Handle shutdown signals
        def shutdown(signum, frame):
            print("\nShutting down worker...")
            self.running = False
        
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        
        print(f"Worker started, listening to {self.queue_url}")
        
        while self.running:
            try:
                job = self._get_job()
                if job:
                    self._process_job(job)
                else:
                    time.sleep(1)  # No job, wait before polling again
            except Exception as e:
                print(f"Error processing job: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _get_job(self) -> Optional[dict]:
        """Get next job from queue."""
        if self.queue_type == "redis":
            # Blocking pop with timeout
            result = self.redis.blpop(self.queue_name, timeout=5)
            if result:
                _, job_data = result
                return json.loads(job_data)
        
        elif self.queue_type == "sqs":
            response = self.sqs.receive_message(
                QueueUrl=self.queue_name,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=5,
            )
            messages = response.get("Messages", [])
            if messages:
                msg = messages[0]
                # Delete from queue (we'll process it)
                self.sqs.delete_message(
                    QueueUrl=self.queue_name,
                    ReceiptHandle=msg["ReceiptHandle"],
                )
                return json.loads(msg["Body"])
        
        return None
    
    def _process_job(self, job: dict) -> None:
        """Process a single job."""
        job_id = job.get("job_id", "unknown")
        print(f"Processing job {job_id}")
        
        try:
            job_type = job.get("type", "reduce")
            
            if job_type == "reduce":
                self._process_reduce_job(job)
            elif job_type == "analyze":
                self._process_analyze_job(job)
            elif job_type == "lod_chain":
                self._process_lod_chain_job(job)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
            
            print(f"Job {job_id} completed successfully")
            
        except Exception as e:
            print(f"Job {job_id} failed: {e}")
            self._report_failure(job, str(e))
    
    def _process_reduce_job(self, job: dict) -> None:
        """Process a reduction job."""
        job_id = job["job_id"]
        params = job.get("params", {})
        
        # Download input file
        input_path = self._download_input(job)
        
        # Generate output path
        output_filename = f"{job_id}_reduced{Path(input_path).suffix}"
        output_path = self.output_dir / output_filename
        
        # Run reduction
        result = self.reducer.reduce(
            input_path,
            target_tris=params.get("target_tris"),
            ratio=params.get("ratio"),
            output=output_path,
            preserve_uvs=params.get("preserve_uvs", True),
            preserve_normals=params.get("preserve_normals", True),
        )
        
        if result.success:
            # Upload output
            output_url = self._upload_output(output_path, job)
            
            # Report success
            self._report_success(job, {
                "original_tris": result.original_tris,
                "final_tris": result.final_tris,
                "reduction_ratio": result.reduction_ratio,
                "output_url": output_url,
            })
        else:
            self._report_failure(job, result.error_message)
        
        # Cleanup
        input_path.unlink(missing_ok=True)
    
    def _process_analyze_job(self, job: dict) -> None:
        """Process an analysis job."""
        input_path = self._download_input(job)
        
        analysis = self.reducer.analyze(input_path)
        
        self._report_success(job, {
            "triangle_count": analysis.triangle_count,
            "vertex_count": analysis.vertex_count,
            "has_uvs": analysis.has_uvs,
            "suggested_lods": [
                {"level": l.level, "target_tris": l.target_tris}
                for l in analysis.suggested_lods
            ],
        })
        
        input_path.unlink(missing_ok=True)
    
    def _process_lod_chain_job(self, job: dict) -> None:
        """Process an LOD chain job."""
        from poly_reducer import LODChain
        
        job_id = job["job_id"]
        params = job.get("params", {})
        
        input_path = self._download_input(job)
        output_dir = self.output_dir / job_id
        output_dir.mkdir(exist_ok=True)
        
        chain = LODChain(
            source=input_path,
            targets=params.get("targets"),
            output_dir=output_dir,
            preserve_uvs=params.get("preserve_uvs", True),
            preserve_normals=params.get("preserve_normals", True),
        )
        
        results = chain.generate()
        
        # Upload all LODs
        lod_urls = []
        for lod in results:
            if lod.success:
                url = self._upload_output(lod.output_path, job, f"lod{lod.level}")
                lod_urls.append({
                    "level": lod.level,
                    "triangle_count": lod.triangle_count,
                    "output_url": url,
                })
        
        self._report_success(job, {"lods": lod_urls})
        
        input_path.unlink(missing_ok=True)
    
    def _download_input(self, job: dict) -> Path:
        """Download input file from S3 or use local path."""
        input_info = job.get("input", {})
        
        if "s3_key" in input_info and self.s3:
            # Download from S3
            local_path = self.upload_dir / Path(input_info["s3_key"]).name
            self.s3.download_file(self.s3_bucket, input_info["s3_key"], str(local_path))
            return local_path
        
        elif "local_path" in input_info:
            return Path(input_info["local_path"])
        
        elif "url" in input_info:
            # Download from URL
            import urllib.request
            local_path = self.upload_dir / Path(urlparse(input_info["url"]).path).name
            urllib.request.urlretrieve(input_info["url"], local_path)
            return local_path
        
        raise ValueError("No input source specified in job")
    
    def _upload_output(self, local_path: Path, job: dict, suffix: str = "") -> str:
        """Upload output file to S3 or return local path."""
        if self.s3 and self.s3_bucket:
            s3_key = f"outputs/{job['job_id']}/{local_path.name}"
            self.s3.upload_file(str(local_path), self.s3_bucket, s3_key)
            return f"s3://{self.s3_bucket}/{s3_key}"
        
        return str(local_path)
    
    def _report_success(self, job: dict, result: dict) -> None:
        """Report job success (webhook, Redis, etc.)."""
        webhook_url = job.get("webhook_url")
        if webhook_url:
            import urllib.request
            data = json.dumps({"job_id": job["job_id"], "status": "completed", "result": result})
            req = urllib.request.Request(
                webhook_url,
                data=data.encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req)
        
        # Also publish to Redis results channel if using Redis
        if self.queue_type == "redis":
            self.redis.publish(
                f"polyreduce:results:{job['job_id']}",
                json.dumps({"status": "completed", "result": result}),
            )
    
    def _report_failure(self, job: dict, error: str) -> None:
        """Report job failure."""
        webhook_url = job.get("webhook_url")
        if webhook_url:
            import urllib.request
            data = json.dumps({"job_id": job["job_id"], "status": "failed", "error": error})
            req = urllib.request.Request(
                webhook_url,
                data=data.encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req)
            except Exception:
                pass
        
        if self.queue_type == "redis":
            self.redis.publish(
                f"polyreduce:results:{job['job_id']}",
                json.dumps({"status": "failed", "error": error}),
            )


def run_worker(queue_url: str) -> None:
    """Run the worker."""
    worker = Worker(queue_url)
    worker.run()
