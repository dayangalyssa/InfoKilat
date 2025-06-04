import streamlit as st
import pandas as pd
import torch
import sys
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import nltk
from script.model import load_bert2bert_model, summarize_text
sys.path.append('script')
from script.embedding import calculate_tfidf, generate_summaries

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

@st.cache_resource
def load_model():
    return load_bert2bert_model()

def extract_content_from_url(url):
    """Extract article content from URL"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        article = soup.find('article') or soup.find('div', class_='article-content') or soup
        paragraphs = article.find_all('p')
        
        content = ' '.join([p.text for p in paragraphs])
        
        # Fallback 
        if len(content) < 100 and len(soup.find_all('p')) > 5:
            content = ' '.join([p.text for p in soup.find_all('p')])
            
        return content
    except Exception as e:
        st.error(f"Error extracting content: {e}")
        return ""

def preprocess_text(content):
    """Pre-process the content into sentence dataset format"""
    from nltk.tokenize import sent_tokenize
    import pandas as pd
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from nltk.corpus import stopwords
    
    # Initialize stemmer and stopwords
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))
    additional_stopwords = {"tempo", "co", "info", "nasional", "cnbc", "CNBC", "indonesia"}
    stop_words.update(additional_stopwords)
    
    # Split into sentences
    sentences = sent_tokenize(content)
    
    # Preprocess each sentence
    original_sentences = []
    processed_sentences = []
    
    for sentence in sentences:
        # Keep original sentence
        original_sentences.append(sentence)
        sentence = sentence.lower()
        tokens = nltk.word_tokenize(sentence)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [stemmer.stem(word) for word in tokens]
        processed_sentence = " ".join(tokens)
        
        processed_sentences.append(processed_sentence)
    
    sentences_df = pd.DataFrame({
        'original_sentence': original_sentences,
        'processed_sentence': processed_sentences,
        'document_id': 0  
    })
    
    return sentences_df

def handle_long_text(text, tokenizer, model):
    """Handle long text by chunking it into sentences and summarizing multiple chunks"""
    sentences = sent_tokenize(text)
    
    if len(tokenizer.encode("summarize: " + text)) <= 512:
        return summarize_text(text, tokenizer, model)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence)) + 1
        
        if current_length + sentence_length > 450: 
            if current_chunk:  
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                chunks.append(sentence)
                current_chunk = []
                current_length = 0
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        summaries.append(summarize_text(chunk, tokenizer, model))
    
    # Combine summaries
    if len(summaries) == 1:
        return summaries[0]
    else:
        combined = " ".join(summaries)
        if len(tokenizer.encode("summarize: " + combined)) > 512:
            return combined
        return summarize_text(combined, tokenizer, model)

# App interface
st.title("InfoKilat")
st.write("Enter a URL to an Indonesian news article for instant summarization")

url_input = st.text_input("News article URL:")

if url_input:
    with st.spinner("Processing..."):
        content = extract_content_from_url(url_input)
    
    if content:
        st.subheader("Article Preview")
        st.write(content[:300] + "..." if len(content) > 300 else content)
        
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                # Process with TF-IDF
                sentences_df = preprocess_text(content)
                tfidf_dict = calculate_tfidf(sentences_df)
                tfidf_summary_dict = generate_summaries(sentences_df, tfidf_dict)
                top_sentences = tfidf_summary_dict[0]
                tfidf_content = ' '.join(top_sentences)
                
                # Generate final summary with BERT2BERT
                tokenizer, model = load_model()
                summary = handle_long_text(tfidf_content, tokenizer, model)
            
            st.subheader("Summary")
            st.success(summary)
    else:
        st.error("Couldn't extract content from this URL. Please try a different news article.")
        
st.markdown("---")
st.markdown("© 2025 InfoKilat | All rights reserved")