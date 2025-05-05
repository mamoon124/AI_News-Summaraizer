import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("📰 News Summarizer using BART")
text = st.text_area("Paste news article text below:", height=300)

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        # Preprocess
        inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        
        # Generate summary
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=130,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("📝 Summary:")
        st.write(summary)
