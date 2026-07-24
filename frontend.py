import streamlit as st
import requests

st.set_page_config(page_title="AI Knowledge Agent", page_icon="🤖", layout="centered")

st.title("🤖 Enterprise AI Knowledge Agent")
st.markdown("Ask anything from your uploaded PDF documents!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send request to FastAPI backend
    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"question": prompt, "session_id": "streamlit_user_session"}
        )

        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("answer", "No response received.")
        else:
            ai_response = f"Error: Backend returned status code {response.status_code}"
    except Exception as e:
        ai_response = f"Connection Error: Could not connect to FastAPI backend. ({e})"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})