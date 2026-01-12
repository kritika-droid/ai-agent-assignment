from fastapi import FastAPI
from agent import AIAgent

app = FastAPI()
agent = AIAgent()

@app.get("/")
def health():
    return {"status": "AI Agent running"}

@app.post("/ask")
def ask(question: str):
    response = agent.answer(question)
    return response
