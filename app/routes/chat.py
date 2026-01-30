from fastapi import APIRouter, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
import os
from app.schemas.chat import ChatRequest
from app.services.llm_service import LLMService

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=api_key)
llm_service = LLMService(client=client)


client = OpenAI(api_key=api_key)
router = APIRouter(prefix="/chat",tags =["chat"])

@router.post("")
def chat_endpoint(Req:ChatRequest):
    
    response = llm_service.chat(
        system_prompt=Req.SystemPrompt,
        user_prompt=Req.UserMessage,
        temperature=Req.temp,
        max_tokens=Req.max_tokens,
        enforce_no_guessing=Req.guard_rails
    )
    return response