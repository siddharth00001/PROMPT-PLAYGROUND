import uuid
from pathlib import Path
import fitz
from fastapi import HTTPException,UploadFile

class DocumentService:
    
    def __init__(self,upload_dir:Path):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(exist_ok=True)
        
    def save_pdf(self,file:UploadFile)-> dict:
        """ Checks for the PDF File and returns the Document ID and save_path """
        if file.content_type not in ["application/pdf"]:
            raise HTTPException(
                status_code=415,
                detail="only PDF files are supported"
            )
        document_id = str(uuid.uuid4())
        save_path = self.upload_dir / f"{document_id}_{file.filename}"
        return {
            "document_id":document_id,
            "save_path":save_path
            }
    
    def find_document_path(self,document_id:str)->Path:
        file_path = list(self.upload_dir.glob(f"{document_id}_*"))
        if not file_path:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        return file_path[0]
    
    def extract_text(self,document_id:str,max_chars:int=2000)-> dict:
        try:
            document_path = self.find_document_path(document_id)
            doc = fitz.open(document_path)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Failed to open the document."
            )
        pages = [page.get_text("text") for page in doc]
        full_text = "\n".join(pages).strip()
        if not full_text:
            raise HTTPException(
                status_code= 422,
                detail = "No Extractable text found in the document."
            )
        trimmed_text = full_text[:max_chars]
        return {
            "document_id": document_id,
            "filename": document_path.name,
            "extracted_text_preview": trimmed_text,
            "total_extracted_chars": len(full_text),
            "trimmed_text_chars": len(trimmed_text)
        }