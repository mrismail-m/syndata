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

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    with open("uploaded_data.csv", "wb") as f:
        f.write(await file.read())
    return {"message": "File uploaded successfully"}

@app.post("/api/generate")
async def generate_data(config: str = Form(...)):
    config_data = json.loads(config)
    with open("generation_config.json", "w") as f:
        json.dump(config_data, f, indent=4)
    return {"message": "Config saved, ready to generate."}

@app.get("/api/stream-logs")
async def stream_logs():
    async def log_generator():
        project_type = "Tabular"
        try:
            with open("generation_config.json", "r") as f:
                config = json.load(f)
                project_type = config.get("project_type", "Tabular")
        except Exception:
            pass

        if project_type == "Relational":
            cmd = [sys.executable, "-u", "relational_engine.py"]
        elif project_type == "Documents":
            cmd = [sys.executable, "-u", "document_engine.py"]
        else:
            cmd = [sys.executable, "-u", "tabular_data_analyzer.py", "uploaded_data.csv"]
            
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            yield f"data: {line.decode('utf-8')}\n\n"
            
        await process.wait()
        yield f"data: [DONE]\n\n"
        
    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.get("/api/results")
async def get_results():
    try:
        with open("evaluation_report.json", "r") as f:
            report = json.load(f)
        return report
    except:
        return {"error": "No report found"}

@app.get("/api/image/{image_name}")
async def get_image(image_name: str):
    valid_images = ["pca_projection.png", "correlation_heatmap.png", "dcr_histogram.png"]
    if image_name in valid_images and os.path.exists(image_name):
        return FileResponse(image_name)
    return {"error": "Image not found"}

@app.get("/api/download")
async def download_csv():
    if os.path.exists("universal_synthetic_output.csv"):
        return FileResponse("universal_synthetic_output.csv", media_type='text/csv', filename="synthetic_data.csv")
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
