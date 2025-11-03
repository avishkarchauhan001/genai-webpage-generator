import streamlit as st
from huggingface_hub import InferenceClient
import base64
import os
from dotenv import load_dotenv

st.set_page_config(page_title="GENAI Webpage Generator", layout="wide")
st.title("Webpage Generator")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")

# Try different models that work with free tier
MODELS = {
    "Qwen 2.5 Coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Meta Llama 3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "Google Gemma 2": "google/gemma-2-2b-it",
}

if not HF_TOKEN:
    st.error("Hugging Face token not found.")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

# Model selector
selected_model_name = st.selectbox("Select Model:", list(MODELS.keys()))
MODEL = MODELS[selected_model_name]

prompt = st.text_area("Enter your prompt:", placeholder="e.g. Create a portfolio webpage with a navbar and about section")

if st.button("Generate Webpage"):
    if prompt:
        with st.spinner(f"Generating HTML using {selected_model_name}..."):
            try:
                instruction = (
                    f"Generate a complete HTML5 webpage for the following request:\n"
                    f"{prompt}\n\n"
                    f"Requirements:\n"
                    f"- Include complete HTML structure with <!DOCTYPE html>\n"
                    f"- Add embedded CSS in <style> tags\n"
                    f"- Make it responsive and visually appealing\n"
                    f"- Return ONLY the HTML code, no explanations"
                )
                
                messages = [{"role": "user", "content": instruction}]
                
                response = client.chat_completion(
                    model=MODEL,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7,
                )
                
                html_code = response.choices[0].message.content.strip()
                
                # Clean up markdown formatting
                if "```
                    html_code = html_code.split("```html").split("```
                elif "```" in html_code:
                    html_code = html_code.split("``````")[0].strip()
                
                if not html_code.endswith("</html>"):
                    html_code += "\n</html>"

                st.success("✅ Generated successfully!")
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
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.warning("This model may not be available on the free tier. Try selecting a different model from the dropdown.")
    else:
        st.warning("Please enter a prompt to generate code.")
