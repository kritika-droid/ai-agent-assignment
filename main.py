from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# import your agent function
from agent.agent import ask_agent

app = FastAPI(title="AI Agent with RAG")

class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    source: list[str]

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = ask_agent(
        query=request.query,
        session_id=request.session_id
    )
    return result

@app.get("/")
def health():
    return {"status": "API is running"}
