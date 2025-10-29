#!/usr/bin/env python3
"""
Mock Phantom Server - Test harness for Project Phantom remote QPU

This is a lightweight mock server that simulates a remote quantum execution
service. It accepts job submissions, tracks job status, and returns simulated
measurement results.

Usage:
    python mock_phantom_server.py [--port PORT]

Example:
    python mock_phantom_server.py --port 5000
"""

import argparse
import json
import random
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List
from flask import Flask, request, jsonify

app = Flask(__name__)


class JobStatus(Enum):
    """Job status enumeration matching C++ side"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Job data structure"""
    job_id: str
    status: JobStatus
    kernel_name: str
    shots: int
    submitted_at: str
    completed_at: str = None
    results: Dict[str, int] = None
    error_message: str = None


# In-memory job storage
jobs: Dict[str, Job] = {}
job_counter = 0
jobs_lock = threading.Lock()


def generate_job_id() -> str:
    """Generate a unique job ID"""
    global job_counter
    with jobs_lock:
        job_counter += 1
        return f"phantom-mock-{job_counter}"


def simulate_quantum_execution(job_id: str, shots: int):
    """
    Simulate quantum circuit execution in a background thread.
    
    For testing, we generate random measurement outcomes with
    roughly uniform distribution.
    """
    # Simulate processing time
    time.sleep(0.1 + random.random() * 0.2)
    
    with jobs_lock:
        if job_id not in jobs:
            return
        
        # Mark as running
        jobs[job_id].status = JobStatus.RUNNING
    
    # Simulate more processing
    time.sleep(0.1)
    
    # Generate simulated results
    # For simplicity, generate outcomes for a single qubit (0 or 1)
    results = {"0": 0, "1": 0}
    for _ in range(shots):
        outcome = "0" if random.random() < 0.5 else "1"
        results[outcome] += 1
    
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].status = JobStatus.COMPLETED
            jobs[job_id].results = results
            jobs[job_id].completed_at = datetime.now().isoformat()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "phantom-mock-server",
        "version": "1.0.0",
        "jobs_count": len(jobs)
    })


@app.route('/submit', methods=['POST'])
def submit_job():
    """
    Submit a new quantum job.
    
    Expected JSON payload:
    {
        "kernel_name": "my_kernel",
        "shots": 1000,
        "kernel_ir": "...",  // Optional: MLIR/LLVM IR
        "arguments": []       // Optional: Runtime arguments
    }
    
    Returns:
    {
        "job_id": "phantom-mock-123",
        "status": "queued"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
        
        kernel_name = data.get("kernel_name", "unknown")
        shots = data.get("shots", 1000)
        
        # Validate shots
        if not isinstance(shots, int) or shots < 1:
            return jsonify({"error": "Invalid shots value"}), 400
        
        # Generate job ID and create job
        job_id = generate_job_id()
        
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            kernel_name=kernel_name,
            shots=shots,
            submitted_at=datetime.now().isoformat()
        )
        
        with jobs_lock:
            jobs[job_id] = job
        
        # Start background execution simulation
        thread = threading.Thread(
            target=simulate_quantum_execution,
            args=(job_id, shots),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": job.status.value
        }), 202  # 202 Accepted
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """
    Get the status of a job.
    
    Returns:
    {
        "job_id": "phantom-mock-123",
        "status": "completed",
        "submitted_at": "2025-10-29T12:00:00",
        "completed_at": "2025-10-29T12:00:01"
    }
    """
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": f"Job not found: {job_id}"}), 404
        
        job = jobs[job_id]
        return jsonify({
            "job_id": job.job_id,
            "status": job.status.value,
            "submitted_at": job.submitted_at,
            "completed_at": job.completed_at
        })


@app.route('/results/<job_id>', methods=['GET'])
def get_job_results(job_id: str):
    """
    Get the results of a completed job.
    
    Returns:
    {
        "job_id": "phantom-mock-123",
        "status": "completed",
        "results": {
            "0": 487,
            "1": 513
        }
    }
    """
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": f"Job not found: {job_id}"}), 404
        
        job = jobs[job_id]
        
        if job.status != JobStatus.COMPLETED:
            return jsonify({
                "error": f"Job not completed: {job_id}",
                "status": job.status.value
            }), 400
        
        return jsonify({
            "job_id": job.job_id,
            "status": job.status.value,
            "results": job.results,
            "shots": job.shots
        })


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs (for debugging)"""
    with jobs_lock:
        jobs_list = []
        for job in jobs.values():
            jobs_list.append({
                "job_id": job.job_id,
                "status": job.status.value,
                "kernel_name": job.kernel_name,
                "shots": job.shots
            })
        return jsonify({"jobs": jobs_list, "count": len(jobs_list)})


@app.route('/reset', methods=['POST'])
def reset_server():
    """Reset server state (for testing)"""
    global job_counter
    with jobs_lock:
        jobs.clear()
        job_counter = 0
    return jsonify({"message": "Server reset successfully"})


def main():
    parser = argparse.ArgumentParser(description="Mock Phantom Server")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to run the server on (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting Mock Phantom Server on {args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  - POST /submit      - Submit a new job")
    print(f"  - GET  /status/<id> - Get job status")
    print(f"  - GET  /results/<id> - Get job results")
    print(f"  - GET  /health      - Health check")
    print(f"  - GET  /jobs        - List all jobs")
    print(f"  - POST /reset       - Reset server state")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

