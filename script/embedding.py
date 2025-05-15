import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import warnings
import os

# Load data from the CSV file
df = pd.read_csv('data/preprocessed_sentences.csv')

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize dictionaries to store results
tf_dict = {}
weight_tf_dict = {}
top3_summary_dict = {}

# Get all unique document IDs
unique_doc_ids = df['document_id'].unique()

# Process each document
for doc_id in unique_doc_ids:
    doc_df = df[df['document_id'] == doc_id].reset_index(drop=True)
    
    if doc_df.empty:
        continue
    
    # Drop NaN in processed_sentence column and ensure all are strings
    kalimat_list = doc_df['processed_sentence'].dropna().astype(str).tolist()
    
    # Skip if no valid sentences
    if len(kalimat_list) == 0:
        continue
    
    # Create term frequency matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(kalimat_list)
    
    # Create DataFrame with terms as rows and sentences as columns
    tf_df = pd.DataFrame(
        X.toarray().T,  # transpose so rows = terms, columns = sentences
        index=vectorizer.get_feature_names_out(),
        columns=[f"Kalimat {i+1}" for i in range(len(kalimat_list))]
    )
    
    # Calculate total term frequency
    tf_df["tf"] = tf_df.sum(axis=1)
    tf_df = tf_df.sort_values("tf", ascending=False)
    tf_dict[doc_id] = tf_df
    
    # Calculate weighted term frequency (TF)
    tf_only = tf_df.drop(columns=["tf"]).copy()
    weight_tf = tf_only.applymap(lambda x: 1 + np.log10(x) if x > 0 else 0)
    weight_tf.loc["Ws"] = weight_tf.sum(axis=0)
    weight_tf["W_tf"] = weight_tf.sum(axis=1)
    weight_tf_dict[doc_id] = weight_tf

# Get top 3 sentences for each document
for doc_id in unique_doc_ids:
    if doc_id not in weight_tf_dict:
        continue
    
    weight_tf = weight_tf_dict[doc_id]
    ws_series = weight_tf.loc["Ws"].drop("W_tf").sort_values(ascending=False)
    top_kalimat_cols = ws_series.head(3).index.tolist()
    
    # Sort sentences in original order
    sorted_kalimat_cols = sorted(top_kalimat_cols, key=lambda x: int(x.split()[-1]))
    
    # Get original sentences
    doc_df = df[df['document_id'] == doc_id].reset_index(drop=True)
    selected_original_sentences = []
    for kal_col in sorted_kalimat_cols:
        kal_index = int(kal_col.split()[-1]) - 1 
        if kal_index < len(doc_df):
            selected_original_sentences.append(doc_df.loc[kal_index, 'original_sentence'])
    
    top3_summary_dict[doc_id] = selected_original_sentences

# Create a DataFrame with top sentences
summary_rows = []
for doc_id, sentences in top3_summary_dict.items():
    # Get the document title if available (assuming there's a title column)
    doc_title = ""
    doc_data = df[df['document_id'] == doc_id]
    if 'title' in doc_data.columns and len(doc_data) > 0:
        doc_title = doc_data['title'].iloc[0]
    
    # Create row with document ID, title, and top sentences
    summary_row = {
        'document_id': doc_id,
        'title': doc_title,
        'top_sentence_1': sentences[0] if len(sentences) > 0 else "",
        'top_sentence_2': sentences[1] if len(sentences) > 1 else "",
        'top_sentence_3': sentences[2] if len(sentences) > 2 else "",
        'summary': " ".join(sentences)
    }
    summary_rows.append(summary_row)

# Create and save the summary DataFrame
summary_df = pd.DataFrame(summary_rows)

# Make sure the directory exists
os.makedirs('data', exist_ok=True)

# Save the results
summary_df.to_csv('data/top_sentences_document.csv', index=False)

print(f"Successfully processed {len(summary_df)} documents")
print("Top sentences saved to data/top_sentences_document.csv")