import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_vector_store():
    docs_path = "data/docs"

    documents = []

    if not os.path.exists(docs_path):
        raise ValueError("❌ data/docs folder not found")

    for file in os.listdir(docs_path):
        if file.endswith(".txt"):
            loader = TextLoader(
                os.path.join(docs_path, file),
                encoding="utf-8"
            )
            documents.extend(loader.load())

    if len(documents) == 0:
        raise ValueError("❌ No .txt files found")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_store"
    )

    vectordb.persist()

    print("✅ Vector store created successfully")
    print(f"📄 Documents: {len(documents)}")
    print(f"🧩 Chunks: {len(chunks)}")


if __name__ == "__main__":
    build_vector_store()
