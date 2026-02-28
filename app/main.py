import logging
from fastapi import FastAPI
from app.routes.chat import router as chat_router
from app.routes.documents import router as documents_router
from app.routes.rag import router as rag_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

app = FastAPI(
    title="Prompt Playground",
    version="0.1.0",
    description= "LLM Playground with RAG-ready architechture"
)

    
@app.get("/")
def health():
    return {"status": "ok"}

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(rag_router)