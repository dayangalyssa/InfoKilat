import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import re
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Suppress HF warning about symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Import from your model.py
from script.model import tokenizer, model, device, cleaned
def ringkas_teks_indo(text, max_input_length=512, max_output_length=150, min_output_length=40):
    """Customize the summary generation with adjustable length parameters"""
    input_text = "ringkasan: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)
    
    summary_ids = model.generate(
        inputs, 
        max_length=max_output_length, 
        min_length=min_output_length, 
        length_penalty=2.0,           
        num_beams=4, 
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

st.set_page_config(
    page_title="Indonesian News Summarizer",
    page_icon="📰",
    layout="centered"
)

# Simple custom CSS
st.markdown("""
<style>
    .summary-box {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Extract text from URL
def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title').text if soup.find('title') else "No title found"
        article_text = ""
        
        # Common article content containers
        article_selectors = [
            'article', '.article-content', '.post-content', 
            '.entry-content', '#article-body', '.content-article',
            '.article__content', '.artikel', '.detail-artikel',
            '.detail__body-text', '.read__content', '.itp_bodycontent'
        ]
        
        # Try each selector
        for selector in article_selectors:
            article = soup.select_one(selector)
            if article:
                paragraphs = article.find_all('p')
                if paragraphs:
                    article_text = ' '.join([p.get_text() for p in paragraphs])
                    break
        
        # Fallback to all paragraphs
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs[:20]])
        
        # Clean the text
        article_text = cleaned(article_text)
        
        return {
            'title': title,
            'content': article_text,
            'url': url
        }
    
    except Exception as e:
        st.error(f"Error extracting content: {e}")
        return None

def process_tfidf_for_article(text, num_top_sentences=5):
    """Extract top sentences using TF-IDF, default increased to 5 sentences"""
    try:
        # Split article into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_top_sentences:
            return text
        
        temp_df = pd.DataFrame({
            'document_id': [0] * len(sentences),
            'original_sentence': sentences,
            'processed_sentence': [cleaned(s) for s in sentences]
        })
        
        temp_df = temp_df[temp_df['processed_sentence'].str.len() > 10]
        
        if len(temp_df) <= num_top_sentences:
            return text
        
        # Get processed sentences
        kalimat_list = temp_df['processed_sentence'].tolist()
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(kalimat_list)
        
        tf_df = pd.DataFrame(
            X.toarray().T,  
            index=vectorizer.get_feature_names_out(),
            columns=[f"Kalimat {i+1}" for i in range(len(kalimat_list))]
        )
        
        # Calculate total term frequency
        tf_df["tf"] = tf_df.sum(axis=1)
        tf_df = tf_df.sort_values("tf", ascending=False)
        
        tf_only = tf_df.drop(columns=["tf"]).copy()
        weight_tf = tf_only.applymap(lambda x: 1 + np.log10(x) if x > 0 else 0)
        weight_tf.loc["Ws"] = weight_tf.sum(axis=0)
        weight_tf["W_tf"] = weight_tf.sum(axis=1)
        
        # Get top N sentences
        ws_series = weight_tf.loc["Ws"].drop("W_tf").sort_values(ascending=False)
        top_kalimat_cols = ws_series.head(num_top_sentences).index.tolist()
        
        # Sort sentences in original order
        sorted_kalimat_cols = sorted(top_kalimat_cols, key=lambda x: int(x.split()[-1]))
        
        # Get original sentences
        selected_original_sentences = []
        for kal_col in sorted_kalimat_cols:
            kal_index = int(kal_col.split()[-1]) - 1 
            if kal_index < len(temp_df):
                selected_original_sentences.append(temp_df.iloc[kal_index]['original_sentence'])
        
        # Join the top sentences
        return ' '.join(selected_original_sentences)
    
    except Exception as e:
        st.error(f"Error in TF-IDF processing: {e}")
        # Fallback
        sentences = text.split('. ')
        if len(sentences) <= num_top_sentences:
            return text
            
      
        selected = [
            sentences[0],  
            sentences[len(sentences) // 4],  
            sentences[len(sentences) // 2], 
            sentences[3 * len(sentences) // 4], 
            sentences[-1] 
        ]
        
        return '. '.join(selected[:num_top_sentences])

# Main app
def main():
    st.title("Indonesian News Summarizer")
    
    # URL input
    url = st.text_input("Enter news URL:", placeholder="https://example.com/news/article")
    
    # Summary settings in sidebar
    st.sidebar.title("Summary Settings")
    
    # Number of sentences for TF-IDF
    num_sentences = st.sidebar.slider("Key sentences to extract:", min_value=3, max_value=7, value=5, step=1)
    
    # Min and max summary length
    min_words = st.sidebar.slider("Minimum summary length (words):", min_value=20, max_value=80, value=40, step=5)
    max_words = st.sidebar.slider("Maximum summary length (words):", min_value=80, max_value=200, value=150, step=10)
    
    if url:
        with st.spinner("Fetching article..."):
            # Extract article content
            article_data = extract_text_from_url(url)
            
            if article_data and article_data['content']:
                # Show title
                st.subheader(article_data['title'])
                
                # Generate summary
                with st.spinner("Generating summary..."):
                    # Start timer
                    start_time = time.time()
                    
                    top_sentences = process_tfidf_for_article(article_data['content'], num_top_sentences=num_sentences)
                    
                    min_length = min_words * 0.8  
                    max_length = max_words * 1.2  
                    
                    summary = ringkas_teks_indo(
                        top_sentences, 
                        max_input_length=512,
                        max_output_length=int(max_length),
                        min_output_length=int(min_length)
                    )
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                
                # Display summary
                st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                st.markdown("### Summary")
                st.write(summary)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Original: {len(article_data['content'].split())} words")
                with col2:
                    st.caption(f"Summary: {len(summary.split())} words")
                with col3:
                    st.caption(f"Time: {processing_time:.2f}s")
            else:
                st.error("Failed to extract content from the URL. Please try another URL.")

if __name__ == "__main__":
    main()