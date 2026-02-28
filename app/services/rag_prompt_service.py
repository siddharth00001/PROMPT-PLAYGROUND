def build_rag_prompt(question:str,contexts:list[str]):
    
    joined_context = "\n\n--\n\n".join(contexts)
    
    return f""""
You are a helpful AI assistant.
Answer the question ONLY using the provided context.
If the answer is not in the context, say "I don't Know".

Context:
{joined_context}

Question:
{question}

Answer:
""".strip()