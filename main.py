# Importing necessary libraries
import os
from fastapi import FastAPI, APIRouter, Request, HTTPException
from runpod_serverless import ApiConfig, RunpodServerlessCompletion, Params, RunpodServerlessEmbedding
from fastapi.responses import StreamingResponse, JSONResponse
import json, time
from uvicorn import Config, Server
from pathlib import Path
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

def run(config_path: str, host: str = "127.0.0.1", port: int = 3000):
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

# Function to format the response data
def format_response(data):
    try:
        text_value = data['output'][0]['choices'][0]['tokens'][0]
    except (KeyError, IndexError, TypeError):
        try:
            text_value = data['output'][0]['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            text_value = ''
    
    usage = data['output'][0].get('usage', {})

    if 'prompt_tokens' in usage and 'completion_tokens' in usage:
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
    elif 'input' in usage and 'output' in usage:
        prompt_tokens = usage.get('input', 0)
        completion_tokens = usage.get('output', 0)
        total_tokens = prompt_tokens + completion_tokens
    else:
        prompt_tokens = completion_tokens = total_tokens = 0

    openai_like_response = {
        'id': data['id'],
        'object': 'text_completion',
        'created': int(time.time()),
        'model': 'gpt-3.5-turbo-instruct',
        'system_fingerprint': "fp_44709d6fcb",
        'choices': [
            {
                'index': 0,
                'text': text_value,
                'logprobs': None,
                'finish_reason': 'stop' if data['status'] == 'COMPLETED' else 'length'
            }
        ],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        }
    }

    return openai_like_response

# Creating API router
router = APIRouter()

params = Params()

# API endpoint for chat completions
@router.post('/chat/completions')
async def request_chat(request: Request):
    try:
        data = await request.json()
        print(data)
        model = data.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"detail": "Missing model in request."})
        
        api = get_config_by_model(model)
        payload = data.get("messages")
        params_dict = params.dict()
        params_dict.update(data)
        new_params = Params(**params_dict)
        runpod: RunpodServerlessCompletion = RunpodServerlessCompletion(api=api, params=new_params)
        
        if(data["stream"]==False):
            response = get_chat_synchronous(runpod, payload)
            return response
        else:
            stream_data = get_chat_asynchronous(runpod, payload)
            response = StreamingResponse(content=stream_data, media_type="text/event-stream")
            response.body_iterator = stream_data.__aiter__()
            return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

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
        async with aiohttp.ClientSession() as session:
            endpoint = runpod.AsyncioEndpoint(endpoint_id, session)
            
            print(f"Request:\n{json.dumps(data, indent=4)}")

            # encrypt if needed
            if f:
                data = {"enc": f.encrypt(json.dumps(data).encode()).decode()}

            job: runpod.AsyncioJob = await endpoint.run(data)

            while True:
                status = await job.status()
                print(f"Current job status: {status}")
                if status == "COMPLETED":
                    output = await job.output()
                    
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

# API endpoint for completions
@router.post('/v1/embeddings')
async def request_embeddings(request: Request):
    try:
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"detail ": "Missing model in request."})
        payload = data.get("input")
        api = get_config_by_model(model)
        runpod: RunpodServerlessEmbedding = RunpodServerlessEmbedding(api=api)
        return get_embedding(runpod, payload)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# Function to get chat synchronously
def get_chat_synchronous(runpod, chat):
    # Generate a response from the runpod
    response = runpod.generate(chat)
    # Check if the response is not cancelled
    if(response ['status'] != "CANCELLED"):
            # Extract data from the response
            data = response["output"][0]
    else:
        # If the request is cancelled, raise an exception
        raise HTTPException(status_code=408, detail="Request timed out.")
    return data

# Function to get chat asynchronously
async def get_chat_asynchronous(runpod, chat):
    # Generate a response from the runpod in an asynchronous manner
    async for chunk in runpod.stream_generate(chat):
        # Check if the chunk is not cancelled
        if(chunk['status'] != "CANCELLED"):
            # Prepare the chat message for SSE
            prepared_message = prepare_chat_message_for_sse(chunk["stream"])
            # Encode the prepared message
            data = f'data: {prepared_message}\n\n'.encode('utf-8')
            yield data
        else:
            # If the request is cancelled, raise an exception
            raise HTTPException(status_code=408, detail="Request timed out.")

# Function to get synchronous response
def get_synchronous(runpod, prompt):
    # Generate a response from the runpod
    response = runpod.generate(prompt)
    # Check if the response is not cancelled
    if(response['status'] != "CANCELLED"):
            # Format the response
            data = format_response(response)
    else:
        # If the request is cancelled, raise an exception
        raise HTTPException(status_code=408, detail="Request timed out.")
    return data

# Function to get synchronous response
def get_embedding(runpod, embedding):
    # Generate a response from the runpod
    response = runpod.generate(embedding)
    # Check if the response is not cancelled
    if(response['status'] != "CANCELLED"):
            # Format the response
            data = response["output"]
    else:
        # If the request is cancelled, raise an exception
        raise HTTPException(status_code=408, detail="Request timed out.")
    return data


# Function to prepare chat message for SSE
def prepare_chat_message_for_sse(message: dict) -> str:
    generated_text = ""
    for stream_chunk in message:
        output = stream_chunk["output"]

        # Loop through all choices, if any
        for choice in output.get('choices', []):
            # Check if 'delta' and 'content' in choice
            if 'delta' in choice and 'content' in choice['delta']:
                # Join the content list into a string
                joined_content = ''.join(choice['delta']['content'])
                # Update the 'content' in 'delta' with the joined string
                generated_text += joined_content

    message[0]["output"]["choices"][0]["delta"]["content"] = generated_text

    # Return the message as a JSON string
    return json.dumps(message[0]["output"])

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
    run(None, host=args.host, port=args.port)

