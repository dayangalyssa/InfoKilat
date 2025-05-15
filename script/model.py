from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import pandas as pd
import torch
import os

# Load tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("cahya/bert2bert-indonesian-summarization")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fungsi untuk merangkum teks
def summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def ringkas_teks_indo(text, max_input_length=512, max_output_length=100):
    # Tambah prefix khusus untuk tugas summarization
    input_text = "ringkasan: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)

    summary_ids = model.generate(inputs, max_length=max_output_length, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Bersihkan teks
def bersihkan(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

# Load the TF-IDF results
try:
    # Load the TF-IDF top sentences
    tfidf_df = pd.read_csv('data/top_sentences_document.csv')
    print(f"Loaded {len(tfidf_df)} documents from TF-IDF results")
except FileNotFoundError:
    print("Error: TF-IDF results not found at data/top_sentences_document.csv")
    tfidf_df = pd.DataFrame()

# Simpan hasil ringkasan
indot5_summary_dict = {}

# Process each document in the TF-IDF results
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
            doc_text = bersihkan(doc_text)
            
            if len(doc_text) < 50:
                continue
                
            try:
                summary = ringkas_teks_indo(doc_text[:1024])  # batas aman
                indot5_summary_dict[doc_id] = summary
            except Exception as e:
                print(f"Error summarizing document {doc_id}: {e}")
                indot5_summary_dict[doc_id] = "(Ringkasan gagal)"
else:
    print("No TF-IDF results found. Cannot generate summaries.")

# Create DataFrame from the summaries
indot5_summary_df = pd.DataFrame.from_dict(indot5_summary_dict, orient='index', columns=['indot5_summary'])
indot5_summary_df.reset_index(inplace=True)
indot5_summary_df.rename(columns={'index': 'document_id'}, inplace=True)

# Display sample
print("\nSample of generated summaries:")
print(indot5_summary_df.head())

# Save the results
indot5_summary_df.to_csv('data/indot5_summary.csv', index=False)
print(f"Saved {len(indot5_summary_df)} summaries to data/indot5_summary.csv")

# Optional: Merge with original TF-IDF data for a complete dataset
if not tfidf_df.empty:
    full_df = pd.merge(tfidf_df, indot5_summary_df, on='document_id', how='left')
    full_df.to_csv('data/complete_summaries.csv', index=False)
    print(f"Saved complete dataset with {len(full_df)} documents to data/complete_summaries.csv")