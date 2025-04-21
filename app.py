# Import necessary libraries
import streamlit as st
from transformers import pipeline

# Initialize the summarization pipeline
@st.cache_resource
def load_summarizer():
    # Load a pre-trained model for summarization from Hugging Face
    return pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

summarizer = load_summarizer()

# Streamlit app
def run_app():
    # App Title and Introduction
    st.title("📝 Article Summarizer")
    st.write("This app summarizes lengthy articles into concise summaries using NLP techniques.")

    # Text input area for users to paste the article
    text_input = st.text_area("Enter the article text", height=300)

    # Define the length of the summary
    max_len = st.slider("Max summary length", 50, 500, 150)
    min_len = st.slider("Min summary length", 10, 200, 30)

    # Button to trigger summarization
    if st.button("Summarize"):
        if not text_input.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Summarizing..."):
                # Generate the summary using the transformer model
                summary = summarizer(text_input, max_length=max_len, min_length=min_len, do_sample=False)
                
                # Display the result
                st.success("✅ Summary Ready!")
                st.subheader("Summary:")
                st.write(summary[0]['summary_text'])

if __name__ == "__main__":
    run_app()
