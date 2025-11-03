import streamlit as st
from huggingface_hub import InferenceClient
import base64
import os
from dotenv import load_dotenv

st.set_page_config(page_title="GENAI Webpage Generator", layout="wide")
st.title("Webpage Generator")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")

MODELS = {
    "Qwen 2.5 Coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Meta Llama 3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "Google Gemma 2": "google/gemma-2-2b-it",
    "Mixtral 8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

if not HF_TOKEN:
    st.error("Hugging Face token not found.")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

selected_model_name = st.selectbox("Select Model:", list(MODELS.keys()))
MODEL = MODELS[selected_model_name]

st.info(f"Using: {MODEL}")

prompt = st.text_area("Enter your prompt:", placeholder="e.g. Create a portfolio webpage")

if st.button("Generate Webpage"):
    if prompt:
        with st.spinner(f"Generating HTML using {selected_model_name}..."):
            try:
                instruction = f"Generate a complete HTML5 webpage for: {prompt}\n\nInclude full HTML structure with embedded CSS. Return ONLY the HTML code."
                
                messages = [{"role": "user", "content": instruction}]
                
                response = client.chat_completion(
                    model=MODEL,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7,
                )
                
                html_code = response.choices[0].message.content.strip()
                
                if "```
                    parts = html_code.split("```html")
                    if len(parts) > 1:
                        html_code = parts[1].split("```
                elif "```" in html_code:
                    parts = html_code.split("```
                    if len(parts) > 2:
                        html_code = parts.strip()[1]
                
                if not html_code.endswith("</html>"):
                    html_code += "\n</html>"

                st.success("✅ Generated successfully!")
                
                st.subheader("Generated HTML Code")
                st.code(html_code, language="html")

                st.subheader("Live Preview")
                st.components.v1.html(html_code, height=500, scrolling=True)

                b64 = base64.b64encode(html_code.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="webpage.html">Download HTML</a>'
                st.markdown(href, unsafe_allow_html=True)

                st.subheader("Copy Code")
                st.text_area("Click to copy", html_code, height=200)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.warning("Try a different model from the dropdown or wait a few minutes.")
    else:
        st.warning("Please enter a prompt.")
