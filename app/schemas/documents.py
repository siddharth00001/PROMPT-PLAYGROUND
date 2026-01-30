from pydantic import BaseModel

class UploadResponse(BaseModel):
    document_id :str
    filename :str
    message:str = "File Uploaded Successfully"
    
    