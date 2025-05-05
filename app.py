import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
from news_fetcher import get_top_headlines

# Load model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def run_streamlit():
    st.title("📰 News Summarizer using BART")
    
    # Fetch top news headlines
    articles = get_top_headlines()

    if articles:
        st.subheader("Top 5 News Headlines")
        for idx, article in enumerate(articles):
            st.write(f"{idx + 1}. **{article['title']}**")
            if st.button(f"Summarize Article {idx + 1}"):
                # Preprocess the text
                text = article["content"] or article["description"]  # Get content or description
                if text:
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

                    # Decode and display the summary
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    st.subheader("📝 Summary:")
                    st.write(summary)
                else:
                    st.warning(f"No content found for the article: {article['title']}")
    else:
        st.warning("No articles fetched.")

if __name__ == "__main__":
    run_streamlit()  # Just run the function directly without threading
