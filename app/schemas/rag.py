from pydantic import BaseModel
from typing import List

class RagQueryRequest(BaseModel):
    query:str
    top_k:int
    temperature:float=0.7
    max_tokens:int = 200
    
class SourceChunk(BaseModel):
    chunk_id:str
    preview:str
    
class RAGQueryResponse(BaseModel):
    answer:str
    sources:List[SourceChunk]
    usage:dict={}
