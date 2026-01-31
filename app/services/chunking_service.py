from dataclasses import dataclass
from typing import List,Dict

@dataclass
class Chunk:
    chunk_id:str
    index:int
    start_char:int
    end_char:int
    text:str

class ChunkingService:
    """
    simple, production-friendly chunking
    - chunk by character length
    - overlap to preserve the context across boundaries
    """
    
    def __init__(self,chunk_size :int = 800,overlap_size:int=120):
        if overlap_size >= chunk_size:
            raise ValueError("Overlap size shoud be smaller than chunk size")
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
    def chunk_text(self,document_id:str,text:str)-> List[Chunk]:
        
        text = (text or "").strip()
        if not text:
            return []
        chunks = []
        step = self.chunk_size - self.overlap_size
        start = 0
        idx = 0
        while start < len(text):
            end = min(start+self.chunk_size,len(text))
            chunk_txt = text[start:end].strip()
            if chunk_txt:
                chunks.append(
                    Chunk(
                        chunk_id = f"{document_id}_{idx}",
                        index=idx,
                        start_char=start,
                        end_char=end,
                        text = chunk_txt,
                    )
                )
            idx+=1
            start += step
        return chunks

    def to_dicts(self,chars:List[Chunk],preview_chars:int=200)->List[Chunk]:
        out_dict = []
        for c in chars:
            out_dict.append({
                "chunk_id":c.chunk_id,
                "index":c.index,
                "start_char":c.start_char,
                "end_char":c.end_char,
                "num_full_text":len(c.text),
                "preview":c.text[:preview_chars]
            
            })
        return out_dict
        