from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_and_main import ACEAdvisoryLegalChatbot
import os
from dotenv import load_dotenv
import traceback
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="ACE Advisory Legal Chatbot API")

# Initialize the chatbot
mistral_api_key = os.environ.get("MISTRAL_API_KEY")
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable is required")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

try:
    logger.info("Initializing chatbot...")
    chatbot = ACEAdvisoryLegalChatbot(
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key
    )
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    logger.error(traceback.format_exc())
    raise

class Query(BaseModel):
    text: str

@app.post("/query")
async def process_query(query: Query):
    try:
        logger.info(f"Processing query: {query.text[:100]}...")  # Log first 100 chars of query
        response = chatbot.generate_response(query.text)
        logger.info("Query processed successfully")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)