import streamlit as st
from huggingface_hub import InferenceClient
import base64
import os
from dotenv import load_dotenv

st.set_page_config(page_title="GENAI Webpage Generator", layout="wide")
st.title("Webpage Generator")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


if not HF_TOKEN:
    st.error("Hugging Face token not found. Please set HF_TOKEN in a .env file.")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

prompt = st.text_area("Enter your prompt:", placeholder="e.g. Create a portfolio webpage with a navbar and about section")

if st.button("Generate Webpage"):
    if prompt:
        with st.spinner("Generating HTML using Mixtral..."):
            instruction = (
                f"Generate a complete HTML5 webpage for the following request:\n"
                f"{prompt}\n"
                f"Make sure to include full HTML, head, body, and relevant CSS."
            )
            messages = [{"role": "user", "content": instruction}]
            response = client.chat_completion(
                model=MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95,
            )
            html_code = response.choices[0].message.content.strip()
            if not html_code.endswith("</html>"):
                html_code += "\n</html>"

        st.subheader("Generated HTML Code")
        st.code(html_code, language="html")

        st.subheader("Live Webpage Preview")
        st.components.v1.html(html_code, height=500, scrolling=True)

        def download_button(code: str, filename: str):
            b64 = base64.b64encode(code.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download as {filename}</a>'
            st.markdown(href, unsafe_allow_html=True)

        download_button(html_code, "generated_webpage.html")

        st.subheader("Copy Code to Clipboard")
        st.text_area("Click inside to copy", html_code, height=200)
    else:
        st.warning("Please enter a prompt to generate code.")
