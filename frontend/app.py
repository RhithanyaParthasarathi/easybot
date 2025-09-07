# frontend/app.py

import streamlit as st
import requests
import uuid

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Easy Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BACKEND URL ---
BACKEND_URL = "http://127.0.0.1:8000/chat"

# --- MINIMAL CSS FOR ALIGNMENT ONLY ---
# This is the only style we need. It finds the user's message container and aligns it to the right.
# The bot's messages remain on the left by default. All colors are handled by Streamlit's theme.
st.markdown("""
<style>
    [data-testid="stChatMessage"]:has(span[data-testid="stChatAvatarIcon-user"]) {
        justify-content: flex-end;
    }
</style>
""", unsafe_allow_html=True)


# --- API COMMUNICATION ---
def get_ai_response(message, session_id):
    """Sends a message to the backend and gets the AI's response."""
    try:
        response = requests.post(
            BACKEND_URL,
            json={"message": message, "session_id": session_id}
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None

# --- SESSION STATE MANAGEMENT ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("Easy Chat")
    
    if st.button("+ New Chat", use_container_width=True):
        chat_id = str(uuid.uuid4())
        st.session_state.active_chat_id = chat_id
        st.session_state.chat_sessions[chat_id] = {
            "title": "New Conversation",
            "messages": [{"role": "assistant", "content": "Hey there! I'm Easy. How can I assist you today?"}]
        }
        st.rerun()

    st.subheader("Conversations")
    chat_ids = list(st.session_state.chat_sessions.keys())
    for chat_id in reversed(chat_ids):
        chat_title = st.session_state.chat_sessions[chat_id]['title']
        if st.button(chat_title, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.rerun()

# --- MAIN CHAT INTERFACE ---
if not st.session_state.active_chat_id:
    # A simple, clean welcome message that respects the theme.
    st.info("Welcome to Easy Chat! Click 'New Chat' in the sidebar to begin.")
else:
    active_chat = st.session_state.chat_sessions[st.session_state.active_chat_id]
    
    # Display all messages in the active chat
    for msg in active_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # The chat input field
    if prompt := st.chat_input("What can I help you with?"):
        active_chat["messages"].append({"role": "user", "content": prompt})
        
        # Set the title of the conversation based on the first user message
        if len(active_chat["messages"]) == 2:
            active_chat["title"] = " ".join(prompt.split()[:4]) + "..."
            
        # Display the user's message immediately
        with st.chat_message("user"):
            st.write(prompt)
            
        # Get and display the bot's response
        with st.chat_message("assistant"):
            with st.spinner("Easy is thinking..."):
                ai_response = get_ai_response(prompt, st.session_state.active_chat_id)
                if ai_response:
                    st.write(ai_response)
                    active_chat["messages"].append({"role": "assistant", "content": ai_response})
        
        # Rerun to update the sidebar with the new title if necessary
        st.rerun()
