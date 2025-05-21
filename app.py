import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import sent_tokenize
import os

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Suppress HF warning about symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Import model functions (adjust path as needed)
from script.model import tokenizer, model, device, clean_text

# Page configuration
st.set_page_config(
    page_title="InfoKilat",
    page_icon="📰",
    layout="centered"
)


# Extract text from URL
def extract_text_from_url(url):
    """Extract article content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.find('title').text if soup.find('title') else "No title found"
        
        # Try to find article content
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
        article_text = clean_text(article_text)
        
        return {
            'title': title,
            'content': article_text,
            'url': url
        }
    
    except Exception as e:
        st.error(f"Error extracting content: {e}")
        return None

# Process TF-IDF for sentence extraction
def extract_top_sentences(text, num_sentences=5):
    """Extract top sentences using TF-IDF"""
    try:
        # Split article into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create temporary DataFrame
        temp_df = pd.DataFrame({
            'original_sentence': sentences,
            'processed_sentence': [clean_text(s) for s in sentences]
        })
        
        # Drop any sentences that are too short
        temp_df = temp_df[temp_df['processed_sentence'].str.len() > 10]
        
        if len(temp_df) <= num_sentences:
            return text
        
        # Get processed sentences
        sentences_list = temp_df['processed_sentence'].tolist()
        
        # Create stopwords list
        stopwords = ['dan', 'yang', 'di', 'dengan', 'untuk', 'pada', 'ke', 'dari', 
                    'dalam', 'adalah', 'ini', 'itu', 'oleh', 'akan', 'tidak', 'telah', 
                    'atau', 'juga', 'bisa', 'ada', 'tersebut', 'sebagai', 'karena',
                    'tempo', 'co', 'info', 'nasional', 'cnbc', 'indonesia']
        
        # Create vectorizer and get TF-IDF matrix
        vectorizer = CountVectorizer(stop_words=stopwords)
        X = vectorizer.fit_transform(sentences_list)
        
        # Get sentence scores
        tf_matrix = X.toarray().T
        sentence_scores = np.sum(tf_matrix, axis=0)
        
        # Get indices of top sentences
        top_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
        
        # Sort indices to maintain original order
        top_indices = sorted(top_indices)
        
        # Get original sentences
        selected_sentences = [temp_df.iloc[i]['original_sentence'] for i in top_indices]
        
        # Join the sentences
        return ' '.join(selected_sentences)
    
    except Exception as e:
        st.error(f"Error in sentence extraction: {e}")
        # Fallback to simple extraction
        sentences = text.split('. ')
        if len(sentences) <= num_sentences:
            return text
        
        # Select evenly distributed sentences
        indices = [0]  # Always include first sentence
        if num_sentences > 1:
            step = len(sentences) / (num_sentences - 1)
            indices.extend([min(len(sentences)-1, int(i*step)) for i in range(1, num_sentences)])
        
        return '. '.join([sentences[i] for i in indices])

# Function to generate summary
def generate_summary(text, min_length=50, max_length=150):
    """Generate summary using the loaded model"""
    try:
        # Extract key sentences first
        key_sentences = extract_top_sentences(text, num_sentences=5)
        
        # Add prefix for Indonesian summarization
        input_text = "ringkasan: " + key_sentences
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,  # Encourage longer outputs
                early_stopping=True
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Clean up extra_id tokens that might appear
        summary = re.sub(r'<extra_id_\d+>', '', summary)
        
        return summary
    
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Failed to generate summary. Please try another article."

# Main app
def main():
    # Header
    st.title("InfoKilat")
    
    # URL input
    url = st.text_input("Enter news URL:", placeholder="https://www.example.com/news/article")
    
    # Generate button
    generate_button = st.button("Generate Summary")
    
    # Process when URL is entered and button is clicked
    if generate_button and url:
        with st.spinner("Processing..."):
            # Extract article content
            article_data = extract_text_from_url(url)
            
            if article_data and article_data['content']:
                # Start timer
                start_time = time.time()
                
                # Generate summary with increased minimum length
                summary = generate_summary(
                    article_data['content'],
                    min_length=50,  # Set minimum length to 40 tokens
                    max_length=150
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Show article title
                st.subheader(article_data['title'])
                
                # Display summary
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.write(summary)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show simple stats
                st.caption(f"Original: {len(article_data['content'].split())} words | Summary: {len(summary.split())} words | Time: {processing_time:.2f}s")
            else:
                st.error("Failed to extract content from the URL. Please try another URL.")
    
    # Footer
    st.markdown("<footer>InfoKilat - Yolanda - Sekar - Dayang</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()