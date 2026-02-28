from fastapi import APIRouter,HTTPException
from pathlib import Path

from app.schemas.rag import RAGQueryResponse,RagQueryRequest,SourceChunk
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.rag_prompt_service import build_rag_prompt
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/rag",tags=["RAG"])

embedding_service = EmbeddingService()
vectorstore_service = VectorStoreService(index_dir=Path("indexes"))
llm=OpenAI(api_key=os.getenv("OPEN_API_KEY"))

@router.post("/query",response_model=RAGQueryResponse)
def rag_query(document_id:str,req:RagQueryRequest):
    
    # 0) Clamping the top_k value
    top_k = max(1,min(req.top_k,10))
    max_out = max(32,min(req.max_tokens ,600)) 
    
    
    # 1) Load index + metadata for searching
    index,metadata =vectorstore_service.load(document_id=document_id)
    if not index:
        raise HTTPException(
            status_code=404,
            detail="Doocument was not Found"
        )
    # 2) Create Embeddings for the query
    qvecs = embedding_service.embed_text([req.query])
    if not qvecs:
        raise HTTPException(
            status_code=422,
            detail="Query is Empty or Invalid"
        )
    query_embedding = qvecs[0]
    
    # 3) Search FAISS
    search_k = min(30,top_k*3)
    indices,_ = vectorstore_service.search(index,query_embedding,top_k = search_k)
    
    # 4) Retrieve the relevant chunks using the metadata
    chunks  = metadata.get("chunks",[])
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No Chunks found in the document. Try to reindex the document."
        )
    relevant_chunks = [chunks[i] for i in indices if i < len(chunks)]
    if not relevant_chunks:
        return RAGQueryResponse(
            answer= "I don't Know",
            sources=[]
        )
    #5) Build the Grounded Rag Prompt
    contexts = [c.get("text","") for c in relevant_chunks if c.get("text","").strip()]
    if not contexts:
        raise HTTPException(
            status_code=422,
            detail="Selected chunks do not have text content. Try to reindex the document."
        )
    prompt = build_rag_prompt(question=req.query,contexts=contexts)
    
    # 6) Call LLM (Grounded Response Generation)
    try:
        response = llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"You are a helpful assistant. Follow the user prompt exactly."},
                      {"role":"user","content":prompt}
                      ],
            temperature=req.temperature,
            max_tokens=max_out
        )
        answer = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens":getattr(response.usage,"prompt_tokens",0),
            "completion_tokens":getattr(response.usage,"completion_tokens",0),
            "total":getattr(response.usage,"total_tokens",0)
        }
        
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Upstream model error. Please try again later."
        )
        
    # 7) Return the answer + Sources 
    sources = []
    seen_ids= set()
    for chunk in relevant_chunks:
        cid = chunk.get("chunk_id","unknown")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        sources.append(SourceChunk(chunk_id=cid,preview=(chunk.get("preview","") or "")[:200]))
        
        
    return RAGQueryResponse(
        answer=answer,
        sources=sources,
        usage=usage
    )