# rag_agent.py

import os
import uuid
from dotenv import load_dotenv
from typing import List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# RAG Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- SETUP THE ENVIRONMENT ---
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- DEFINE THE RAG TOOL ---
@tool
def get_information_from_website(url: str, query: str) -> str:
    """
    Loads content from a website URL, stores it in a Qdrant collection,
    and retrieves the most relevant information based on a user query.
    """
    print(f"--- Executing RAG Tool for URL: {url} using Qdrant ---")
    collection_name = "rag_" + str(uuid.uuid5(uuid.NAMESPACE_DNS, url))
    print(f"--- Using Qdrant collection: {collection_name} ---")

    loader = WebBaseLoader([url])
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Qdrant.from_documents(
        documents=splits, embedding=embedding_model, url=QDRANT_URL,
        api_key=QDRANT_API_KEY, collection_name=collection_name,
    )
    
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        return "No relevant information was found on the website for that query."
    
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print("--- RAG Tool finished, returning context to the agent ---")
    return context

# --- SET UP THE LANGGRAPH AGENT ---
tools = [get_information_from_website]
model_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: List[BaseMessage]

# --- !!! THIS IS THE CORRECTED FUNCTION !!! ---
def call_model(state: AgentState):
    """Calls the LLM, but now correctly appends the response to the history."""
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    # By returning messages + [response], we accumulate the conversation history.
    return {"messages": messages + [response]}
# --- END CORRECTION ---

tool_node = ToolNode(tools)

def should_continue(state: AgentState):
    if state['messages'][-1].tool_calls:
        return "continue"
    return "end"

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("action", tool_node)
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)
graph_builder.add_edge("action", "agent")

memory = MemorySaver()
rag_app = graph_builder.compile(checkpointer=memory)

# --- INTERACT WITH THE RAG AGENT ---
if __name__ == "__main__":
    print("RAG Agent with Qdrant & Memory is ready.")
    print("You can now have a conversation. Provide the URL first, then ask questions.")

    config = {"configurable": {"thread_id": "my-rag-conversation-1"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        inputs = {"messages": [HumanMessage(content=user_input)]}
        result = rag_app.invoke(inputs, config=config)
        
        final_answer = result['messages'][-1].content
        print(f"Agent: {final_answer}")