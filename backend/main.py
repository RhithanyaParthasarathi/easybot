import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()

# --- 1. SET UP THE MODEL ---

google_api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=google_api_key)

# --- 2. MEMORY HANDLER USING JSON FILES ---

class FileMemory:
    def __init__(self, base_dir="./memory_data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def get_file_path(self, session_id: str) -> str:
        return os.path.join(self.base_dir, f"{session_id}.json")

    def load(self, session_id: str) -> Dict[str, List[Dict]]:
        path = self.get_file_path(session_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"messages": data.get("messages", [])}
        return {"messages": []}

    def save(self, session_id: str, state: Dict[str, List[Dict]]) -> None:
        path = self.get_file_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

file_memory = FileMemory()

# --- 3. SET UP THE FASTAPI APP ---

app = FastAPI(
    title="LangGraph Chatbot Server with File-Based Memory",
    description="A simple API server for a chatbot with memory stored in JSON files.",
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    state = file_memory.load(session_id)

    # Convert message dicts back to BaseMessage objects
    messages = []
    for msg in state["messages"]:
        if msg["type"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
        elif msg["type"] == "system":
            messages.append(SystemMessage(content=msg["content"]))

    # Append the new user message
    human_input = HumanMessage(content=request.message)
    messages.append(human_input)

    # Prepare the prompt
    system_prompt = SystemMessage(
        content="You are a helpful AI assistant named Easy. Your tone is friendly and direct. Always refer to yourself as Easy."
    )
    messages_with_system_prompt = [system_prompt] + messages

    # Get AI response
    response = model.invoke(messages_with_system_prompt)

    # Append AI message to the conversation
    messages.append(response)

    # Save updated conversation
    # Convert messages to dicts for JSON serialization
    state_to_save = {
        "messages": [
            {"type": "human", "content": msg.content} if isinstance(msg, HumanMessage) else
            {"type": "ai", "content": msg.content} if isinstance(msg, AIMessage) else
            {"type": "system", "content": msg.content}
            for msg in messages
        ]
    }
    file_memory.save(session_id, state_to_save)

    return ChatResponse(response=response.content)

@app.get("/")
def read_root():
    return {"message": "LangGraph Chatbot Server with persistent memory is running."}

# --- 4. RUN THE SERVER ---

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
