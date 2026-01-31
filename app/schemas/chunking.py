from pydantic import BaseModel

class ChunkRequest(BaseModel):
    chunk_size:int=800
    overlap:int=120
    max_chars:int = 200000