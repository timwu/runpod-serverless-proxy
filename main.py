# Importing necessary libraries
import os
from fastapi import FastAPI, APIRouter, Request, HTTPException
from runpod_serverless import ApiConfig, Params
from runpod.endpoint.helpers import FINAL_STATES
from fastapi.responses import StreamingResponse, JSONResponse
import json, time
from uvicorn import Config, Server
import runpod
import asyncio
import aiohttp
from cryptography.fernet import Fernet



# Initializing variables
model_data = {
    "object": "list",
    "data": []
}

configs = []


f = None
if "KEY" in os.environ:
    f = Fernet(os.environ["KEY"])

def run(host: str = "127.0.0.1", port: int = 3000):
    config = Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
    )
    server = Server(config=config)
    server.run()

# Function to get configuration by model name
def get_config_by_model(model_name):
    for config in configs:
        if config.model == model_name:
            return config
        
def get_model_endpoint_id(model_name):
    endpoings = runpod.get_endpoints()
    for endpoint in endpoings:
        if endpoint["name"] == model_name:
            return endpoint["id"]
    return None

# Creating API router
router = APIRouter()

params = Params()

async def stream_job(job: runpod.AsyncioJob):
    while True:
        stream_partial = await job._fetch_job(source="stream")
        if (
            stream_partial["status"] not in FINAL_STATES
            or len(stream_partial.get("stream", [])) > 0
        ):
            for chunk in stream_partial.get("stream", []):
                yield chunk["output"]
        elif stream_partial["status"] in FINAL_STATES:
            break

# API endpoint for completions
@router.post('/v1/chat/completions')
@router.post('/v1/completions')
async def request_prompt(request: Request):
    try:
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"detail ": "Missing model in request."})
        # payload = data.get("prompt")[0]
        endpoint_id = get_model_endpoint_id(model)

        print(f"Request:\n{json.dumps(data, indent=4)}")

        stream = data.get("stream", False)

        # encrypt if needed
        if f:
            data = {"enc": f.encrypt(json.dumps(data).encode()).decode()}

        if stream:
            async def streamed_response():
                async with aiohttp.ClientSession() as session:
                    endpoint = runpod.AsyncioEndpoint(endpoint_id, session)
                    job: runpod.AsyncioJob = await endpoint.run(data)
                    async for chunk in stream_job(job):
                        
                        if "enc" in chunk:
                            chunk = json.loads(f.decrypt(chunk["enc"].encode()).decode())
                        
                        if "batch" in chunk:
                            for c in chunk["batch"]:
                                yield f"data: {json.dumps(c)}\n\n".encode("utf-8")
                        else:
                            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    
                    # double-check the status
                    job_state = await job._fetch_job()
                    status = job_state["status"]
                    print(f"Current job status: {status}")
                    if status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
                        message = job_state.get("error", f"job failed with status: {status}")
                        error = {"error": {"message": message}}
                        yield f"data: {json.dumps(error)}\n\n".encode("utf-8")

            return StreamingResponse(content=streamed_response(), media_type="text/event-stream")
        else:
            async with aiohttp.ClientSession() as session:
                endpoint = runpod.AsyncioEndpoint(endpoint_id, session)

                job: runpod.AsyncioJob = await endpoint.run(data)

                while True:
                    status = await job.status()
                    print(f"Current job status: {status}")
                    if status == "COMPLETED":
                        output = await job.output()
                        output = output[0]

                        # decrypt if needed
                        if "enc" in output:
                            output = json.loads(f.decrypt(output["enc"].encode()).decode())

                        print(f"Response:\n{json.dumps(output, indent=4)}")

                        return output
                    elif status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
                        raise HTTPException(status_code=500, detail=f"Job failed with status: {status}")
                    else:
                        await asyncio.sleep(1)  # Wait for 3 seconds before polling again
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# Create a FastAPI application
app = FastAPI()

# Include the router in the application
app.include_router(router)

# Endpoint to list all models
@app.get("/v1/models")
async def list_models():
    return  {
        "object": "list",
        "data": [
            {"id": endpoint["name"], 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "organization-owner"} for endpoint in runpod.get_endpoints()
        ]
    }

# Endpoint to get a specific model
@app.get("/models/{model_id}")
async def get_model(model_id):
    # Function to find a model by id
    def find_model(models, id):
        return next((model for model in models['data'] if model['id'] == id), None)
    # Return the found model
    return find_model(model_data, model_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", type=str, default=None)
    parser.add_argument("-e", "--endpoint", help="API endpoint", type=str, default=None)
    parser.add_argument("-k", "--api_key", help="API key", type=str, default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("-m", "--model", help="Model", type=str, default=None)
    parser.add_argument("-t", "--timeout", help="Timeout", type=int, default=None)
    parser.add_argument("-o", "--use_openai_format", help="Use OpenAI format", type=bool, default=None)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=None)
    parser.add_argument("--host", help="Host", type=str, default="0.0.0.0")
    parser.add_argument("--port", help="Port", type=int, default=3000)
    args = parser.parse_args()

    runpod.api_key = args.api_key
    endpoints = runpod.get_endpoints()
    configs.extend([ApiConfig(**{
        "endpoint_id": e["id"],
        "api_key": args.api_key,
        "model": e["name"],
        **({"timeout": args.timeout} if args.timeout is not None else {}),
        **({"use_openai_format": args.use_openai_format} if args.use_openai_format is not None else {}),
        **({"batch_size": args.batch_size} if args.batch_size is not None else {}),
    }) for e in endpoints])
    run(host=args.host, port=args.port)

