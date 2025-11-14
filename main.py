## Imports
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from utils import *
from tools import *
from services import *
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

app = FastAPI()

class AssistantRequest(BaseModel):
    user_input: str
class AssistantResponse(BaseModel):
    status: str  # "success" or "error"
    data: Optional[Dict[str, Any]] = None


@app.post("/assist", response_model=AssistantResponse)
async def assist(request: AssistantRequest):
    user_input = request.user_input.strip()
    try:
        result = await process_user_input(user_input)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AssistantResponse(
        status=result["status"],
        data=result.get("data", None)
    )

# Dialogflow webhook endpoint
@app.post("/dialogflow")
async def dialogflow_webhook(request: Request):
    try:
        req = await request.json()
        user_text = req["queryResult"]["queryText"]
        
        # Call your service
        result = await process_user_input(user_text)
        
        # Extract the data
        status = result["status"]
        data = result["data"]
        note_updated = data["note_updated"]
        intent = data["intent"]
        error = data["error"]
        
        # Use note_updated as the spoken response
        if status == "success" and not error:
            response_text = note_updated  # This gets spoken by Google Assistant
        else:
            response_text = error or "Sorry, something went wrong."
        
        return {
            "fulfillmentText": response_text,
            "google": {
                "expectUserResponse": False
            }
        }
        
    except Exception as e:
        print(f"Dialogflow error: {str(e)}")
        return {
            "fulfillmentText": "Sorry, I couldn't process that request."
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

        