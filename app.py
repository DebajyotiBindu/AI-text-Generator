import streamlit as st
import tensorflow as tf
import numpy as np
from src.preprocessing import tokenizer, SEQ_LENGTH

st.set_page_config(page_title="AI Text Generator")

@st.cache_resource
def load_sherlock_model():
    return tf.keras.models.load_model("models/sherlock_lstm_final.h5")

model = load_sherlock_model()
vocab = tokenizer.get_vocabulary()

# UI
st.title("AI text generator")
st.sidebar.header("Tuning Settings")
num_words = st.sidebar.slider("Words", 5, 50, 20)
top_k = st.sidebar.slider("Top-K Filter (Lower = More logical)", 1, 50, 10)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.5) 

user_input = st.text_input("Seed Sentence:", "Sherlock Holmes looked at")

def smart_sample(preds, temperature, k):
    # 1. Apply Temperature
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    top_indices = preds.argsort()[-k:][::-1]
    top_preds = preds[top_indices]
    top_preds = top_preds / np.sum(top_preds) 
    
    choice = np.random.choice(top_indices, p=top_preds)
    return choice

if st.button("Generate"):
    if user_input:
        generated = user_input
        current_seed = user_input
        
        for _ in range(num_words):
            tokens = tokenizer([current_seed])
            input_tokens = tokens[:, -SEQ_LENGTH:]          
            preds = model.predict(input_tokens, verbose=0)[0]
            next_idx = smart_sample(preds, temperature, top_k)
            next_word = vocab[next_idx]
            if generated.split()[-3:].count(next_word) >= 2:
                next_idx = smart_sample(preds, temperature + 0.5, top_k)
                next_word = vocab[next_idx]

            generated += " " + next_word
            current_seed += " " + next_word
            
        st.write(f"**Result:** {generated}")