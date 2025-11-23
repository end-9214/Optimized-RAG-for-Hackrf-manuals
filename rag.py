import os
from dotenv import load_dotenv

load_dotenv()

from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_chroma import Chroma

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)

llm = ChatGroq(model="openai/gpt-oss-20b")
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def load_documents(folder_path: str) -> List[Document]:
    docs = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def create_vector_store(folder_path: str, db_path="./chroma_db", collection="my_rag"):
    docs = load_documents(folder_path)
    chunks = split_documents(docs)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=db_path,
        collection_name=collection,
    )
    return True


def load_vector_store(db_path="./chroma_db", collection="my_rag"):
    store = Chroma(
        persist_directory=db_path,
        collection_name=collection,
        embedding_function=embedding,
    )
    return store


def get_retriever(vector_store, k=3):
    return vector_store.as_retriever(search_kwargs={"k": k})


def get_conversational_rag(retriever):
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a standalone question which can be understood
    without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
    """

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."
             "Keep your answer very concise. "
             "Limit the response to **3-4 sentences OR under 120 words**. "
             "Do NOT add unnecessary details or long explanations."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    question_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_chain)
    return rag_chain


def chat(rag_chain, query: str, chat_history: List = None) -> Dict:
    if chat_history is None:
        chat_history = []

    result = rag_chain.invoke({"input": query, "chat_history": chat_history})

    answer = result["answer"]
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    return {"answer": answer, "chat_history": chat_history}


# if __name__ == "__main__":
#     store = load_vector_store()
#     retriever = get_retriever(store)
#     rag = get_conversational_rag(retriever)

#     history = []

#     q1 = "Hello, can you tell me what HackRF is?"
#     res1 = chat(rag, q1, history)
#     print("Q1:", q1)
#     print("A1:", res1["answer"])

#     q2 = "Where can it be used?"
#     res2 = chat(rag, q2, res1["chat_history"])
#     print("Q2:", q2)
#     print("A2:", res2["answer"])
