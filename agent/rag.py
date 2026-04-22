from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.md"

_CHUNK_SIZE = 400         
_CHUNK_OVERLAP = 60     
_TOP_K = 3                

_RAG_SYSTEM_PROMPT = """\
You are AutoStream's helpful product assistant.

Answer the user's question using ONLY the context provided below.
Be concise, friendly, and accurate. If the context does not contain enough
information to fully answer the question, say so honestly.

Context:
{context}
"""

def _load_knowledge_base() -> list[Document]:
    if not _KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at '{_KB_PATH}'.")

    text = _KB_PATH.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": "knowledge_base.md"})]


def _build_faiss_index(embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    raw_docs = _load_knowledge_base()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("Knowledge base split into %d chunks.", len(chunks))

    vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built successfully.")
    return vector_store

class RAGPipeline:
    def __init__(self, llm: ChatGoogleGenerativeAI, embeddings: GoogleGenerativeAIEmbeddings) -> None:
        self._llm = llm
        self._vector_store = _build_faiss_index(embeddings)

    def retrieve_and_answer(self, query: str) -> str:
        docs = self._vector_store.similarity_search(query, k=_TOP_K)
        if not docs:
            return (
                "I don't have specific information about that. "
                "Please visit autostream.io or contact our support team."
            )

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        messages = [
            SystemMessage(content=_RAG_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=query),
        ]
        response = self._llm.invoke(messages)
        logger.debug("RAG answer generated for query: %s", query[:60])
        return response.content.strip()
