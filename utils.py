from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def format_history(messages: List[BaseMessage]) -> List[str]:
    readable: List[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            readable.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            readable.append(f"AI: {msg.content}")
        else:
            readable.append(msg.content)
    return readable