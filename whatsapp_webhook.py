from fastapi import APIRouter, Form, BackgroundTasks, Response, Request
from twilio.rest import Client
import os
from typing import Optional
import logging
from dotenv import load_dotenv

from services import process_user_input

router = APIRouter(prefix="/whatsapp", tags=["WhatsApp"])

# Twilio credentials from environment
load_dotenv(dotenv_path="../.env")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
USER_WHATSAPP_NUMBER = os.getenv("USER_WHATSAPP_NUMBER")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_success_response(result: dict) -> str:
    """
    Format the result from process_user_input into a WhatsApp-friendly message
    
    Customize this based on what your process_user_input returns
    """
    data = result.get("data", {})
    
    # Example formatting - customize based on your actual response structure
    if not data:
        return "✅ Done! Your request has been processed."
    
    response_lines = ["✅ Success!"]
    
    for key, value in data.items():
        response_lines.append(f"{key}: {value}")
    
    return "\n".join(response_lines)

async def process_and_reply(user_message: str, user_phone: str = USER_WHATSAPP_NUMBER):
    """
    Background task to process message and send reply
    This allows the webhook to respond immediately to Twilio (prevent timeout)
    """
    try:
        logger.info(f"Processing WhatsApp message from {user_phone}: {user_message}")
        
        # Call your existing process_user_input function
        # This is the SAME logic you use for /assist endpoint
        result = await process_user_input(user_message)
        
        # Format response for WhatsApp
        if result["status"] == "success":
            response_text = format_success_response(result)
        else:
            response_text = "❌ Sorry, something went wrong processing your request."
        
        # Send response back via Twilio
        message = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=response_text,
            to=user_phone
        )
        
        logger.info(f"Response sent to {user_phone}: {message.sid}")
        
    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {str(e)}")
        
        # Send error message to user
        try:
            twilio_client.messages.create(
                from_=TWILIO_WHATSAPP_NUMBER,
                body=f"❌ Sorry, there was an error: {str(e)}",
                to=user_phone
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {str(send_error)}")

@router.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Twilio WhatsApp webhook endpoint
    Receives incoming WhatsApp messages and processes them in the background
    """
    try:
        # Parse form data from Twilio
        form_data = await request.form()
        
        # Extract message details
        incoming_message = form_data.get("Body", "").strip()
        from_number = form_data.get("From", "")
        
        logger.info(f"Webhook received - From: {from_number}, Body: {incoming_message}")
        
        # Ignore empty messages
        if not incoming_message:
            logger.info("Ignoring empty message")
            return Response(content="", status_code=200)
        
        # Process message in background to avoid Twilio timeout
        background_tasks.add_task(process_and_reply, incoming_message, from_number)
        
        # Respond immediately to Twilio (must be within 15 seconds)
        return Response(content="", status_code=200)
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        # Always return 200 to Twilio to avoid retries
        return Response(content="", status_code=200)


@router.get("/health")
async def health_check():
    """
    Health check endpoint for WhatsApp webhook
    """
    return {
        "status": "healthy",
        "service": "WhatsApp Webhook",
        "twilio_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
    }