from typing import Dict, Optional, List
import os

from llm import get_llm

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage


class AIAgent:
    def __init__(self):
        """
        Initializes heavy components ONLY when first needed
        (Azure-safe lazy initialization)
        """

        # -------- Embeddings --------
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # -------- Vector Store --------
        self.vectorstore = Chroma(
            persist_directory="vector_store",
            embedding_function=self.embeddings
        )

        # -------- Ingest documents once --------
        if self.vectorstore._collection.count() == 0:
            self._ingest_documents()

        # -------- Lazy LLM --------
        self.llm = get_llm()  # Can be None (SAFE)

        # -------- Session Memory --------
        self.memory: Dict[str, List[Dict]] = {}

    # --------------------------------------------------
    # Document ingestion
    # --------------------------------------------------
    def _ingest_documents(self):
        docs_path = "data/docs"

        if not os.path.exists(docs_path):
            print("⚠️ data/docs folder not found. Skipping ingestion.")
            return

        documents = []

        for file in os.listdir(docs_path):
            if file.endswith(".txt"):
                loader = TextLoader(
                    os.path.join(docs_path, file),
                    encoding="utf-8"
                )
                documents.extend(loader.load())

        if not documents:
            print("⚠️ No documents found for ingestion.")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        split_docs = splitter.split_documents(documents)
        self.vectorstore.add_documents(split_docs)

        print(f"✅ Ingested {len(split_docs)} document chunks.")

    # --------------------------------------------------
    # Decide RAG or not
    # --------------------------------------------------
    def needs_rag(self, query: str) -> bool:
        keywords = [
            "policy", "leave", "company", "internal",
            "employee", "work from home", "salary",
            "security", "conduct"
        ]
        return any(k in query.lower() for k in keywords)

    # --------------------------------------------------
    # Vector Search
    # --------------------------------------------------
    def retrieve_docs(self, query: str):
        return self.vectorstore.similarity_search(query, k=3)

    # --------------------------------------------------
    # Core Logic
    # --------------------------------------------------
    def answer(self, query: str, session_id: Optional[str] = None) -> Dict:

        history = self.memory.get(session_id, []) if session_id else []

        # -------- RAG FLOW --------
        if self.needs_rag(query):
            docs = self.retrieve_docs(query)

            if not docs:
                answer = "No relevant internal documents were found."
                sources = []
            else:
                context = "\n\n".join(doc.page_content for doc in docs)
                sources = list(
                    set(doc.metadata.get("source", "unknown") for doc in docs)
                )

                if self.llm:
                    messages = [
                        SystemMessage(
                            content=(
                                "You are an AI assistant answering strictly "
                                "from internal company documents. "
                                "If the answer is not found, say so."
                            )
                        ),
                        HumanMessage(
                            content=f"Context:\n{context}\n\nQuestion:\n{query}"
                        )
                    ]
                    answer = self.llm.invoke(messages).content
                else:
                    answer = (
                        "LLM not configured. Showing relevant document context:\n\n"
                        f"{context[:800]}"
                    )

        # -------- DIRECT ANSWER FLOW --------
        else:
            if self.llm:
                messages = [
                    SystemMessage(content="You are a helpful AI assistant."),
                    HumanMessage(content=query)
                ]
                answer = self.llm.invoke(messages).content
            else:
                answer = (
                    "LLM not configured. API is running successfully, "
                    "but AI responses are disabled."
                )
            sources = []

        # -------- Save memory --------
        if session_id:
            history.append({"query": query, "answer": answer})
            self.memory[session_id] = history

        return {
            "answer": answer,
            "source": sources
        }


# --------------------------------------------------
# ✅ LAZY SINGLETON (CRITICAL FIX)
# --------------------------------------------------
_agent_instance = None


def get_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AIAgent()
    return _agent_instance


# --------------------------------------------------
# PUBLIC FUNCTION FOR FASTAPI
# --------------------------------------------------
def ask_agent(query: str, session_id: Optional[str] = None) -> Dict:
    agent = get_agent()
    return agent.answer(query=query, session_id=session_id)
