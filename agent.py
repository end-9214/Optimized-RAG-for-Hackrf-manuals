import os
import asyncio
from typing import Annotated, List

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import groq, deepgram, cartesia, silero

from rag import (
    load_vector_store,
    get_retriever,
    get_conversational_rag,
    chat as rag_chat,
)

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    vector_store = load_vector_store(db_path="./chroma_db", collection="my_rag")
    retriever = get_retriever(vector_store, k=3)
    rag_chain = get_conversational_rag(retriever)

    rag_history: List = []

    @llm.function_tool
    async def rag_query(
        query: Annotated[
            str,
            (
                "A clear, fully-formed question that should be answered using the "
                "uploaded documents (PDF/DOCX). Extract this from the user's message."
            ),
        ]
    ) -> str:
        """
        Use this tool when you need to answer questions based on the user's
        private documents (PDF/DOCX) indexed in the RAG system.
        Always call this tool for factual / knowledge-type questions
        about that custom content instead of guessing.
        """
        nonlocal rag_history

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rag_chat(rag_chain, query, rag_history),
        )

        rag_history = result["chat_history"]
        answer = result["answer"]
        return answer

    agent = Agent(
        instructions=(
            "You are a helpful voice assistant that can talk naturally with the user. "
            "You have access to a knowledge base built from the user's PDF/DOCX documents. "
            "Whenever the user asks a question that might be answered by those documents, "
            "you MUST call the 'rag_query' tool with a clean version of the question, "
            "then use the tool's result to form your final answer. "
            "If the question is clearly outside the scope of the documents, answer normally. "
            "Keep your responses clear, concise, and voice-friendly."
        ),
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=groq.LLM(model="openai/gpt-oss-20b"),
        tts=cartesia.TTS(
            model="sonic-2",
            voice=os.getenv("CARTESIA_VOICE_ID", "bf0a246a-8642-498a-9950-80c35e9276b5"),
        ),
        tools=[rag_query],
    )

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    await session.say(
        "Hey! I'm your RAG assistant. You can ask me questions based on your uploaded documents.",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))