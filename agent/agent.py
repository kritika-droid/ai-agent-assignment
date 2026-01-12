from typing import Dict, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class AIAgent:
    def __init__(self):
        """
        Initialize embeddings, vector store, and session memory
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory="vector_store",
            embedding_function=self.embeddings
        )

        # Simple in-memory session storage
        self.memory = {}

    # --------------------------------------------------
    # Decision making: Should we use RAG or not?
    # --------------------------------------------------
    def needs_rag(self, query: str) -> bool:
        """
        Decide whether the query requires internal documents
        """
        keywords = [
            "policy",
            "leave",
            "company",
            "internal",
            "employee",
            "refund",
            "work from home",
            "salary"
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in keywords)

    # --------------------------------------------------
    # Tool: Vector search (RAG)
    # --------------------------------------------------
    def retrieve_docs(self, query: str):
        """
        Retrieve relevant documents from vector store
        """
        return self.vectorstore.similarity_search(query, k=3)

    # --------------------------------------------------
    # Core reasoning logic
    # --------------------------------------------------
    def answer(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Generate an answer using either RAG or direct LLM logic
        """

        # ---- Load session history (if any) ----
        history = []
        if session_id:
            history = self.memory.get(session_id, [])

        # ---- Case 1: Internal document-based question ----
        if self.needs_rag(query):
            docs = self.retrieve_docs(query)

            if not docs:
                answer = "No relevant internal documents were found."
                sources = []
            else:
                sources = list(
                    set(doc.metadata.get("source", "unknown") for doc in docs)
                )

                context = "\n".join(doc.page_content for doc in docs)

                # Placeholder (LLM will replace later)
                answer = (
                    "Answer based on internal documents:\n"
                    f"{context[:500]}..."
                )

        # ---- Case 2: General question ----
        else:
            answer = "This is a general answer generated directly by the LLM."
            sources = []

        # ---- Save conversation to memory ----
        if session_id:
            history.append({"query": query, "answer": answer})
            self.memory[session_id] = history

        return {
            "answer": answer,
            "source": sources
        }


# --------------------------------------------------
# SINGLETON AGENT INSTANCE
# --------------------------------------------------
agent = AIAgent()


# --------------------------------------------------
# PUBLIC FUNCTION (USED BY FASTAPI)
# --------------------------------------------------
def ask_agent(query: str, session_id: Optional[str] = None) -> Dict:
    """
    Public function used by FastAPI / API layer
    """
    return agent.answer(query=query, session_id=session_id)
