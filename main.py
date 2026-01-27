from fastapi import FastAPI, HTTPException,UploadFile,File
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os
from openai import OpenAI
import time
import logging
import uuid
import fitz


load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
client = OpenAI(api_key=api_key)


def get_document_path(document_id: str)-> Path:
    file_path = list(UPLOAD_DIR.glob(f"{document_id}_*"))
    if not file_path:
        raise HTTPException(
            status_code=404,
            detail= "Document not found"
        )
    return file_path[0]

app = FastAPI()

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("prompt-playground")

class ChatRequest(BaseModel):
    SystemPrompt: str
    UserMessage : str
    temp : float = 0.7
    guard_rails: bool = True
    max_tokens:int = 256
    
    
     
@app.get('/')
def health():
    return {"status": "ok"}


@app.post('/chat')
def chat_endpoint(Req: ChatRequest):
    guard_rails_prompt = ""
    max_out = max(16,max(Req.max_tokens,800))
    if Req.guard_rails:
        guard_rails_prompt = (
            "IMPORTANT: Ensure that you don't answer the questions if you don't know the answer. \" Say \"I don't know\" when applicable. \n"
            "Do not make up answers. Do not invent the Resources and Facts. If you can't Verify the information"
        )
    start = time.time()
    try:
        logger.info(f"max_out tokens used: {max_out}")
        res = client.chat.completions.create(
            model ="gpt-4o-mini",
            messages=[
                {"role":"system","content":guard_rails_prompt + Req.SystemPrompt},
                {"role":"user","content":Req.UserMessage}
            ],
            temperature=Req.temp,
            max_tokens=max_out
        )
        # using usage extraction
        usage  = getattr(res,'usage',None)
        usage_dict ={}
        if usage:
            usage_dict = {
            "prompt_tokens":getattr(usage,'prompt_tokens',0),
            "completion_tokens":getattr(usage,'completion_tokens',0),
            "total_tokens":getattr(usage,'total_tokens',0)
            }
        reply = res.choices[0].message.content
        latency = int((time.time()- start)*1000)
        logger.info(
        f"Chat Temp :{Req.temp} max_out ={max_out}  guard_rails = {Req.guard_rails} "
        f"Latency_ms :{latency} usage = {usage_dict}"
        )
    
        return {
            "reply":reply,
            "latency_ms" :latency, 
            "usage":usage_dict
        }
    except Exception as e:
        latency = int((time.time()- start)*1000)
        logger.exception(f"LLm failed to give response latency_ms = {latency}")
        raise HTTPException(
            status_code=502,
            detail="Upstream LLm error Failed to get response"
            )
    
@app.post("/document/upload")
async def upload_document(file:UploadFile=File(...)):
    # Validating the Docs
    if file.content_type not in ["application/pdf"]:
        return {"Error":"File type not supported."}
    document_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{document_id}_{file.filename}"
    
    # save to the disk
    content = await file.read()
    save_path.write_bytes(content)
    
    
    return {
        "document_id":document_id,
        "filename":file.filename,
        "message":"File Uploaded Successfully"
    }

@app.post("/document/{doument_id}/extract_text")
def extract_text(document_id:str):
    pdf_path = get_document_path(document_id)
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to open the document."
        )
    parts = []
    for page in doc:
        text = page.get_text()
        parts.append(text)
    full_text = "\n".join(parts).strip()
    if not full_text:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in the document."
        )
    # Limiting the output to first 1000 characters
    trimmed_text  = full_text[:1000]
    logger.info(
        f" The document id is {document_id}\n"
        f" The Retrieved Filename is {pdf_path.name}"
    )
    
    return {
        "document_id":document_id,
        "filename":pdf_path.name,
        "num_characters":len(full_text),
        "returned_characters":len(trimmed_text),
        "text_preview":trimmed_text
    }