from pathlib import Path
import json
import numpy as np
import faiss
from typing import List

class VectorStoreService:
    """
    Single Responsiblity:
    - build FAISS index
    - save/load FAISS index + metadata
    """
    def __init__(self,index_dir:Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(exist_ok=True)
        
    
    def _index_path(self,document_id:str)->Path:
        return self.index_dir / f"{document_id}.faiss"
    
    def _meta_path(self,document_id:str)-> Path:
        return self.index_dir / f"{document_id}.meta.json"
    
    def build_index(self,embeddings:list[list[float]]):
        if not embeddings:
            raise ValueError("No Embeddings provided")
        
        mat = np.array(embeddings,dtype="float32")
        
        dim = mat.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(mat)
        return index
    
    def save(self,document_id:str,index,metadata:dict):
        faiss.write_index(index,str(self._index_path(document_id=document_id)))
        self._meta_path(document_id=document_id).write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        
    def load(self,document_id:str):
        index_path = self._index_path(document_id=document_id)
        metadata_path = self._meta_path(document_id=document_id)
        if not index_path.exists() or not metadata_path.exists():
            return None,None
        
        index  = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text())
        return index,metadata
    
    def search(self,index,query_embedding:List[float],top_k:int):
        import numpy as np
        q = np.array([query_embedding],dtype="float32")
        distances,indices= index.search(q,top_k)
        return indices[0],distances[0]
    