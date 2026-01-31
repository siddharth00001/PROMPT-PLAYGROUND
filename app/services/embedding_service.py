import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class EmbeddingService:
    """
    Open ai embedder to convert the text into embeddings
    """
    def __init__(self,model:str="text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
        self.model = model
    
    def embed_text(self,texts:list[str])-> list[list[float]]:
        # Basic Input hygiene
        cleaned  = [(t or "").strip() for t in texts]
        cleaned = [t for t in cleaned if t]
        
        if not cleaned:
            return []
        
        res  = self.client.embeddings.create(
            model=self.model,
            input=cleaned
        )
        
        return [item.embedding for item in res.data]
    
if __name__ == "__main__":
    svc = EmbeddingService()
    embeddings = svc.embed_text(["hello world"])
    print(len(embeddings),len(embeddings[0]))