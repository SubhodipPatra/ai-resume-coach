from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import OPENAI_API_KEY
import os

BASE_DIR = "vectordb"

def get_vectorstore(collection_name: str):
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    os.makedirs(BASE_DIR, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=BASE_DIR
    )
