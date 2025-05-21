from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import pandas as pd
import torch
import os
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("cahya/bert2bert-indonesian-summarization")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Basic summarization function
def summarize(text):
    """Basic summarization with minimal preprocessing"""
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs, 
        max_length=150, 
        min_length=40, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return clean_extra_ids(summary)

# Indonesian-specific summarization function
def ringkas_teks_indo(text, max_input_length=512, max_output_length=150, min_output_length=40):
    """Indonesian-specific summarization with better parameters"""
    # Clean the text first
    text = clean_text(text)
    
    # Add prefix for Indonesian summarization
    input_text = "ringkasan: " + text
    
    # Tokenize
    inputs = tokenizer.encode(
        input_text, 
        return_tensors="pt", 
        max_length=max_input_length, 
        truncation=True
    ).to(device)

    # Generate summary with improved parameters
    summary_ids = model.generate(
        inputs, 
        max_length=max_output_length, 
        min_length=min_output_length,  
        length_penalty=2.0,    # Higher value favors longer summaries
        num_beams=4,           # Beam search for better quality
        early_stopping=True,
        no_repeat_ngram_size=2 # Avoid repetition
    )

    # Decode and clean up the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = clean_extra_ids(summary)
    
    return summary

# Text cleaning function
def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    
    # List of terms to remove
    terms_to_remove = [
        "tempo", "co", "info", "nasional", "cnbc", "CNBC", 
        "indonesia", "tempo.", ".co", "tempo.co", "TEMPO", 
        "CO", "TEMPO.CO"
    ]
    
    # Create a regex pattern to match these terms with word boundaries
    pattern = r'\b(?:' + '|'.join(re.escape(term) for term in terms_to_remove) + r')\b'
    
    # Remove the terms
    cleaned_text = re.sub(pattern, '', text)
    
    # Remove excessive newlines and spaces
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Remove URLs
    cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)
    
    # Remove HTML tags
    cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
    
    # Remove common citation patterns in Indonesian news
    cleaned_text = re.sub(r'\s*-\s*[A-Za-z]+\s*,', ',', cleaned_text)
    
    # Trim whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# Clean extra_id tokens function
def clean_extra_ids(text):
    """Remove extra_id tokens from the generated text"""
    # Pattern to match <extra_id_X> tokens
    pattern = r'<extra_id_\d+>'
    
    # Remove all occurrences of extra_id tokens
    cleaned_text = re.sub(pattern, '', text)
    
    # Clean up any double spaces that might result
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

# Extract key sentences using TF-IDF
def extract_key_sentences(text, num_sentences=5):
    """Extract key sentences from text using TF-IDF ranking"""
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return text
    
    # Create a list of processed sentences
    processed_sentences = [clean_text(s) for s in sentences]
    
    # Create stopwords list
    stopwords = ['dan', 'yang', 'di', 'dengan', 'untuk', 'pada', 'ke', 'dari', 
                'dalam', 'adalah', 'ini', 'itu', 'oleh', 'akan', 'tidak', 'telah', 
                'atau', 'juga', 'bisa', 'ada', 'tersebut', 'sebagai', 'karena']
    
    # Add specific stopwords
    stopwords.extend(['tempo', 'co', 'info', 'nasional', 'cnbc', 'indonesia', 
                    'tempo.', '.co', 'tempo.co', 'TEMPO', 'CO', 'TEMPO.CO'])
    
    # Create TF-IDF vectorizer
    try:
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        
        # Calculate sentence scores
        sentence_scores = [tfidf_matrix[i].sum() for i in range(len(sentences))]
        
        # Get indices of top sentences
        ranked_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
        
        # Sort indices to maintain original order
        ranked_indices = sorted(ranked_indices)
        
        # Extract the top sentences
        key_sentences = [sentences[i] for i in ranked_indices]
        
        return ' '.join(key_sentences)
    
    except Exception as e:
        print(f"Error in TF-IDF extraction: {e}")
        # Fallback to CountVectorizer if TF-IDF fails
        return extract_key_sentences_count(text, num_sentences)

# Alternative using CountVectorizer (as fallback)
def extract_key_sentences_count(text, num_sentences=5):
    """Extract key sentences using Count-based weighting (fallback method)"""
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return text
    
    # Create a DataFrame for processing
    temp_df = pd.DataFrame({
        'original_sentence': sentences,
        'processed_sentence': [clean_text(s) for s in sentences]
    })
    
    # Drop any sentences that are too short
    temp_df = temp_df[temp_df['processed_sentence'].str.len() > 10]
    
    if len(temp_df) <= num_sentences:
        return text
    
    # Get processed sentences
    kalimat_list = temp_df['processed_sentence'].tolist()
    
    # Create stopwords list
    stopwords = ['dan', 'yang', 'di', 'dengan', 'untuk', 'pada', 'ke', 'dari', 
                'dalam', 'adalah', 'ini', 'itu', 'oleh', 'akan', 'tidak', 'telah', 
                'atau', 'juga', 'bisa', 'ada', 'tersebut', 'sebagai', 'karena']
    
    # Add specific stopwords
    stopwords.extend(['tempo', 'co', 'info', 'nasional', 'cnbc', 'indonesia', 
                    'tempo.', '.co', 'tempo.co', 'TEMPO', 'CO', 'TEMPO.CO'])
    
    # Create term frequency matrix
    vectorizer = CountVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform(kalimat_list)
    
    # Calculate weighted sentence scores
    tf_matrix = X.toarray().T  # transpose
    sentence_scores = np.sum(tf_matrix, axis=0)
    
    # Get indices of top sentences
    top_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
    
    # Sort indices to maintain original order
    top_indices = sorted(top_indices)
    
    # Get original sentences
    selected_sentences = [temp_df.iloc[i]['original_sentence'] for i in top_indices]
    
    # Join sentences
    return ' '.join(selected_sentences)

# Enhanced summary generation
def generate_summary(text, max_length=150, min_length=40, use_key_sentences=True):
    """Complete pipeline for summary generation"""
    # First clean the text
    clean_text_input = clean_text(text)
    
    # Extract key sentences if enabled
    if use_key_sentences:
        preprocessed_text = extract_key_sentences(clean_text_input, num_sentences=5)
    else:
        preprocessed_text = clean_text_input
    
    # Generate summary
    summary = ringkas_teks_indo(
        preprocessed_text,
        max_output_length=max_length,
        min_output_length=min_length
    )
    
    # Clean any extra_id tokens
    summary = clean_extra_ids(summary)
    
    # Ensure proper ending punctuation
    if summary and not summary[-1] in ['.', '!', '?']:
        summary += '.'
    
    return summary

# Process documents from TF-IDF results
def process_tfidf_documents(tfidf_file='data/top_sentences_document.csv', output_file='data/indot5_summary.csv'):
    """Process documents from TF-IDF results and generate summaries"""
    # Load the TF-IDF results
    try:
        tfidf_df = pd.read_csv(tfidf_file)
        print(f"Loaded {len(tfidf_df)} documents from TF-IDF results")
    except FileNotFoundError:
        print(f"Error: TF-IDF results not found at {tfidf_file}")
        return None
    
    # Initialize results dictionary
    summary_dict = {}
    
    # Process each document
    if not tfidf_df.empty:
        for _, row in tfidf_df.iterrows():
            doc_id = row['document_id']
            
            # Combine top sentences from TF-IDF
            top_sentences = []
            for i in range(1, 4):
                col_name = f'top_sentence_{i}'
                if col_name in row and not pd.isna(row[col_name]) and row[col_name]:
                    top_sentences.append(row[col_name])
            
            # Combine and clean the top sentences
            if top_sentences:
                doc_text = ' '.join(top_sentences)
                doc_text = clean_text(doc_text)
                
                if len(doc_text) < 50:
                    continue
                    
                try:
                    # Generate summary
                    summary = ringkas_teks_indo(
                        doc_text, 
                        max_output_length=150,
                        min_output_length=40
                    )
                    
                    # Clean extra_id tokens
                    summary = clean_extra_ids(summary)
                    
                    summary_dict[doc_id] = summary
                except Exception as e:
                    print(f"Error summarizing document {doc_id}: {e}")
                    summary_dict[doc_id] = "(Ringkasan gagal)"
        
        # Create DataFrame from the summaries
        summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=['indot5_summary'])
        summary_df.reset_index(inplace=True)
        summary_df.rename(columns={'index': 'document_id'}, inplace=True)
        
        # Save the results
        summary_df.to_csv(output_file, index=False)
        print(f"Saved {len(summary_df)} summaries to {output_file}")
        
        # Create combined dataset
        full_df = pd.merge(tfidf_df, summary_df, on='document_id', how='left')
        combined_file = 'data/complete_summaries.csv'
        full_df.to_csv(combined_file, index=False)
        print(f"Saved complete dataset with {len(full_df)} documents to {combined_file}")
        
        return summary_df
    else:
        print("No documents found in TF-IDF results")
        return None

# Run the processing if executed directly
if __name__ == "__main__":
    process_tfidf_documents()