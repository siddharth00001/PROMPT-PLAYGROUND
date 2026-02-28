import uuid
import logging
import fitz
from fastapi import APIRouter,HTTPException,UploadFile,File
from pathlib import Path
from app.schemas.documents import UploadResponse
from app.services.document_service import DocumentService
from app.schemas.chunking import ChunkRequest
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.chunking_service import ChunkingService
from app.services.vector_store_service import VectorStoreService

router = APIRouter(prefix="/document",tags=["document"])
logger = logging.getLogger("prompt-playground")


##----------------Importing All relevant Services--------------------##

document_service = DocumentService(upload_dir=Path("uploads"))
embedding_service = EmbeddingService()
vectorstore_service = VectorStoreService(index_dir=Path("indexes"))



##---------------------Upload Document Endpoint-----------------------##

@router.post("/upload",response_model=UploadResponse)
async def upload_document(file:UploadFile=File(...)):
    # Validating the docs
    try:
        meta = document_service.save_pdf(file)
        content = await file.read()
        meta["save_path"].write_bytes(content)
        logger.info(
            f"Document is saved to the disk location : {meta["save_path"]}"
        )
        return {
            "document_id":meta["document_id"],
            "filename":file.filename,
            "message": "File Uploaded Successfully"
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to save the document."
        )
##---------------------Extract Text From the Document-----------------------##


@router.get("/{document_id}/extract_text")
def extract_text(document_id:str,max_chars:int = 2000) -> dict:
    
    return document_service.extract_text(document_id)


##---------------------Chunking Endpoint------------------------------------##

@router.post("/{document_id}/chunks")
def get_chunks(document_id:str,req:ChunkRequest):
    # Reusing the code for extraction
    extracted_text = document_service.extract_text(document_id=document_id,max_chars=req.max_chars)
    text = extracted_text["full_text"]
    
    chunker  = ChunkingService(chunk_size=req.chunk_size,overlap_size=req.overlap)
    chunks  = chunker.chunk_text(document_id=document_id,text=text)
    
    return {
        "document_id":document_id,
        "chunk_size":req.chunk_size,
        "overlap":req.overlap,
        "num_chunks":len(chunks),
        "chunks":chunker.to_dicts(chunks)
    }
    

##---------------------Endpoint for creating the index for Documents-----------------------##


@router.post("/{document_id}/index")
def build_faiss_index(document_id:str,req:ChunkRequest):
    
    # 1) Extracting the full text
    extracted = document_service.extract_text(document_id=document_id)
    text = extracted["full_text"]
    
    # 2) chunking the text
    chunker  = ChunkingService(chunk_size=req.chunk_size,overlap_size=req.overlap)
    chunks = chunker.chunk_text(document_id=document_id,text=text)
    
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No chunks created from the document"
        )
        
    chunks_text = [c.text for c in chunks]
    
    # 3) Creating the embedding for the chunked text
    
    embeddings = embedding_service.embed_text(chunks_text)
    
    if len(embeddings) != len(chunks_text):
        raise HTTPException(
            status_code=500,
            detail="Embeddings Count mismatch"
        )
        
    # 4) Buil FAISS Index
    
    index  = vectorstore_service.build_index(embeddings=embeddings)
    
    #5) Save the index and the metadata
    
    metadata=  {
        "document_id":document_id,
        "filename":extracted["filename"],
        "chunk_size": req.chunk_size,
        "overlap": req.overlap,
        "embedding_model": embedding_service.model,
        "num_chunks": len(chunks),
    
    "chunks":[
        {
            "chunk_id":c.chunk_id, 
            "index": c.index,
            "start_char":c.start_char,
            "end_char":c.end_char,
            "num_full_text":len(c.text),
            "text":c.text,
            "preview":c.text[:200]
        }
        for c in chunks
        ],
    }
    vectorstore_service.save(document_id=document_id,index=index,metadata=metadata)
    return {
        "document_id":document_id,
        "chunk_size": req.chunk_size,
        "embedding_dim": index.d,
        "status":"indexed and saved"
    }