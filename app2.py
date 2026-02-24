import streamlit as st
from transformers import pipeline
import os

st.set_page_config(page_title="AI Text Generator", )

st.title("AI Text Generator")

MODEL_PATH = r"D:\mlproject7\model_gpt2"

@st.cache_resource  
def load_sherlock():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model folder not found at {MODEL_PATH}")
        return None
    return pipeline("text-generation", model=MODEL_PATH)

with st.spinner("loading model..."):
    sherlock_pipe = load_sherlock()

prompt = st.text_input("Start a sentence:", "The mystery of the")

col1, col2 = st.columns(2)
with col1:
    max_len = st.slider("Length of response", 20, 200, 50)
with col2:
    temp = st.slider("Creativity (Temperature)", 0.1, 1.0, 0.7)

if st.button("Generate Response"):
    if sherlock_pipe:
        with st.spinner("Loading"):
            result = sherlock_pipe(
                prompt, 
                max_length=max_len, 
                temperature=temp,
                do_sample=True,
                truncation=True
            )
            
            st.subheader("Result:")
            st.write(result[0]['generated_text'])
    else:
        st.warning("Please check your model path!")
