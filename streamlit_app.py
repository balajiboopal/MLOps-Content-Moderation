import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001"

st.title("Content Moderation System")

st.header("Single Text Moderation")
text_input = st.text_area("Enter text to moderate:")
if st.button("Moderate"):
    response = requests.post(f"{API_URL}/moderate", json={"text": text_input})
    if response.status_code == 200:
        result = response.json()
        st.write(f"**Toxicity Score:** {result['toxicity_score']:.2f}")
        st.write(f"**Moderation Result:** {result['moderation_result'].capitalize()}")
    else:
        st.error(f"Error: {response.text}")
