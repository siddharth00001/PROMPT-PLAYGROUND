from pydantic import BaseModel

class ChatRequest(BaseModel):
    SystemPrompt: str
    UserMessage : str
    temp : float = 0.7
    guard_rails: bool = True
    max_tokens:int = 256