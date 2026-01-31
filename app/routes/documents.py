import uuid
import logging
import fitz
from fastapi import APIRouter,HTTPException,UploadFile,File
from pathlib import Path
from app.schemas.documents import UploadResponse
from app.services.document_service import DocumentService
from app.schemas.chunking import ChunkRequest
from app.services.chunking_service import ChunkingService

router = APIRouter(prefix="/document",tags=["document"])
document_service = DocumentService(upload_dir=Path("uploads"))

logger = logging.getLogger("prompt-playground")


# Upload Document Endpoint

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

# Extract Text From the Document
@router.get("/{document_id}/extract_text")
def extract_text(document_id:str,max_chars:int = 2000) -> dict:
    
    return document_service.extract_text(document_id)

##--------------------- Chunking Endpoint-----------------------##

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