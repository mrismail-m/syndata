import os
import sys
import subprocess
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="HackDataV2 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import List

@app.post("/api/upload")
async def upload_files(project_name: str = Form(...), files: List[UploadFile] = File(...)):
    workspace_dir = f"workspaces/{project_name}"
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(f"{workspace_dir}/uploaded_data", exist_ok=True)
    for file in files:
        with open(f"{workspace_dir}/uploaded_data/{file.filename}", "wb") as f:
            f.write(await file.read())
    
    if len(files) > 0:
        with open(f"{workspace_dir}/uploaded_data.csv", "wb") as f:
            f.write(open(f"{workspace_dir}/uploaded_data/{files[0].filename}", "rb").read())
            
    return {"message": "Files uploaded successfully"}

@app.post("/api/generate")
async def generate_data(config: str = Form(...)):
    config_data = json.loads(config)
    project_name = config_data.get("project_name", "Tabular")
    workspace_dir = f"workspaces/{project_name}"
    os.makedirs(workspace_dir, exist_ok=True)
    with open(f"{workspace_dir}/generation_config.json", "w") as f:
        json.dump(config_data, f, indent=4)
    return {"message": "Config saved, ready to generate."}

@app.get("/api/stream-logs")
async def stream_logs(project_name: str):
    async def log_generator():
        workspace_dir = f"workspaces/{project_name}"
        os.makedirs(workspace_dir, exist_ok=True)
        project_type = "Tabular"
        try:
            with open(f"{workspace_dir}/generation_config.json", "r") as f:
                config = json.load(f)
                project_type = config.get("project_type", "Tabular")
        except Exception:
            pass

        if project_type == "Relational":
            cmd = [sys.executable, "-u", "../../relational_engine.py"]
        elif project_type == "Documents":
            cmd = [sys.executable, "-u", "../../document_engine.py"]
        else:
            cmd = [sys.executable, "-u", "../../tabular_data_analyzer.py", "uploaded_data.csv"]
            
        env = os.environ.copy()
        env["PYTHONPATH"] = "../.."
            
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=workspace_dir,
                env=env
            )
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                yield f"data: {line.decode('utf-8')}\n\n"
                
            await process.wait()
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"
        
    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.get("/api/results")
async def get_results(project_name: str):
    try:
        with open(f"workspaces/{project_name}/evaluation_report.json", "r") as f:
            report = json.load(f)
        return report
    except:
        return {"error": "No report found"}

@app.get("/api/image/{image_name}")
async def get_image(image_name: str, project_name: str):
    valid_images = ["pca_projection.png", "correlation_heatmap.png", "dcr_histogram.png"]
    image_path = f"workspaces/{project_name}/{image_name}"
    if image_name in valid_images and os.path.exists(image_path):
        return FileResponse(image_path)
    return {"error": "Image not found"}

@app.get("/api/download")
async def download_csv(project_name: str):
    file_path = f"workspaces/{project_name}/universal_synthetic_output.csv"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/csv', filename="synthetic_data.csv")
    
    # Check for document download (zip or just invoices)
    doc_path = f"workspaces/{project_name}/document_output/invoices.csv"
    if os.path.exists(doc_path):
        return FileResponse(doc_path, media_type='text/csv', filename="synthetic_invoices.csv")
        
    return {"error": "File not found"}

@app.get("/api/document")
async def get_document(project_name: str):
    file_path = f"workspaces/{project_name}/document_output/sample_invoice.html"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/html')
    return {"error": "Document not found"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
