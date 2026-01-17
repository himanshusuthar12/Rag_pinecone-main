import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()


User_Id = os.getenv("User_Id")
# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #8bc34a;
    }
    .metadata-box {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "user_id" not in st.session_state:
    st.session_state.user_id = User_Id if User_Id else "default_user"
if "top_k" not in st.session_state:
    st.session_state.top_k = 3
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False

def check_api_health(api_url: str) -> Optional[dict]:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def get_config(api_url: str) -> Optional[dict]:
    """Get API configuration."""
    try:
        response = requests.get(f"{api_url}/config", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def send_query(api_url: str, query: str, user_id: str, top_k: int) -> Optional[dict]:
    """Send a query to the API."""
    try:
        payload = {
            "query": query,
            "user_id": user_id,
            "top_k": top_k
        }
        response = requests.post(
            f"{api_url}/query",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def load_history(api_url: str, user_id: str) -> list:
    """Load chat history from API."""
    try:
        response = requests.get(f"{api_url}/history/{user_id}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("history", [])
        return []
    except Exception as e:
        return []

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Settings")
    
    # API URL configuration
    api_url = st.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="URL of the FastAPI backend (e.g., http://localhost:8000)"
    )
    st.session_state.api_url = api_url
    
    # User ID configuration
    user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="Unique identifier for this user session"
    )
    st.session_state.user_id = user_id
    
    # Top K configuration
    top_k = st.slider(
        "Top K Results",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        help="Number of search results to return"
    )
    st.session_state.top_k = top_k
    
    st.divider()
    
    # Health check
    st.subheader("🔍 System Status")
    if st.button("Check API Health", use_container_width=True):
        health = check_api_health(api_url)
        if health:
            st.success("✅ API is healthy")
            st.json(health)
        else:
            st.error("❌ API is not responding")
    
    # Load configuration
    if st.button("Load Configuration", use_container_width=True):
        config = get_config(api_url)
        if config:
            st.success("✅ Configuration loaded")
            with st.expander("View Configuration"):
                st.json(config)
        else:
            st.error("❌ Failed to load configuration")
    
    # Load history button
    if st.button("🔄 Refresh History", use_container_width=True):
        history = load_history(api_url, user_id)
        if history:
            st.session_state.messages = []
            for item in history:
                st.session_state.messages.append({
                    "role": "user",
                    "content": item.get("query", ""),
                    "timestamp": item.get("Date", ""),
                    "metadata": {
                        "domain": item.get("namespace", ""),
                        "refined_query": item.get("refined_query", "")
                    }
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": item.get("answer", ""),
                    "timestamp": item.get("Date", ""),
                    "metadata": {
                        "domain": item.get("namespace", ""),
                        "refined_query": item.get("refined_query", "")
                    }
                })
            st.session_state.history_loaded = True
            st.success(f"Loaded {len(history)} conversation(s)")
        else:
            st.warning("No history found")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history_loaded = False
        st.rerun()

# Main content area
st.markdown('<div class="main-header">🤖 RAG Chat Assistant</div>', unsafe_allow_html=True)

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display metadata if available
            if "metadata" in message and message["metadata"]:
                with st.expander("📊 Query Details"):
                    if message["metadata"].get("domain"):
                        st.write(f"**Domain:** {message['metadata']['domain']}")
                    if message["metadata"].get("refined_query"):
                        st.write(f"**Refined Query:** {message['metadata']['refined_query']}")
                    if message.get("timestamp"):
                        st.write(f"**Timestamp:** {message['timestamp']}")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_query(api_url, prompt, user_id, top_k)
            
            if response:
                # Display answer
                st.write(response.get("answer", "No answer provided"))
                
                # Display metadata
                with st.expander("📊 Query Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Domain", response.get("domain", "N/A"))
                    with col2:
                        st.metric("Namespace", response.get("namespace", "N/A"))
                    with col3:
                        st.metric("Results Found", response.get("search_results_count", 0))
                    
                    if response.get("refined_query"):
                        st.write(f"**Refined Query:** {response['refined_query']}")
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.get("answer", ""),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "metadata": {
                        "domain": response.get("domain", ""),
                        "namespace": response.get("namespace", ""),
                        "refined_query": response.get("refined_query", ""),
                        "search_results_count": response.get("search_results_count", 0)
                    }
                })
            else:
                st.error("Failed to get response from API. Please check your connection and try again.")

# Footer with API information
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"API URL: {api_url}")
with col2:
    st.caption(f"User ID: {user_id}")
with col3:
    st.caption(f"Top K: {top_k}")
