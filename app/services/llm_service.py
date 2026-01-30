import time
from openai import OpenAI
import logging
import os
from fastapi import HTTPException

logger = logging.getLogger("prompt-playground")


class LLMService:
    
    def __init__(self,client:OpenAI,model:str="gpt-4o-mini"):
        self.client = client
        self.model = model
        
    def chat(self,system_prompt:str,user_prompt:str,
             temperature:float=0.7,max_tokens:int=500,
             enforce_no_guessing:bool=False
             ):
        start = time.time()
        guardrail = ""
        try:
            if enforce_no_guessing:
                guardrail = (
                    "If the answer to the question is not known based on "
                    "the provided context, respond with "
                    "'I don't know based on the provided context.'"
                )
            max_out = max(0,min(max_tokens,800))
            response = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {"role":"system","content": guardrail + system_prompt},
                {"role":"user","content":user_prompt},
                ],
            temperature=temperature,
            max_tokens=max_out,
            )
        except Exception:
            latency_ms = int((time.time() - start)*1000)
            logger.error(
                f"Upstream LLM Sercvice error after {latency_ms} ms"
            )
            raise HTTPException(
                status_code=502,
                detail = "Upstream LLM service error."
            )
        latency_ms = int((time.time() - start)*1000)
        usage = getattr(response,"usage",None)
        usage_dict=None
        if usage:
            usage_dict = {
                "prompt_tokens":getattr(usage,"prompt_tokens",None),
                "completion_tokens":getattr(usage,"completion_tokens",None),
                "total_tokens":getattr(usage,"total_tokens",None),
            }
        reply = response.choices[0].message.content
        logger.info(
            f"LLM Response in {latency_ms} ms | "
            f"Model: {self.model} | "
            f"Temperature: {temperature} | "
            f"Max Tokens: {max_out} | "
            f"Enforce No Guessing: {enforce_no_guessing} | "
            f"Usage: {usage_dict}"
        )
        return {
            "reply":reply,
            "latency_ms":latency_ms,
            "usage":usage_dict
        }