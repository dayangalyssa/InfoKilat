import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    """Main function to process sentences with TF-IDF weighting."""
    df = pd.read_csv("data/preprocessed_sentences.csv")
    
    num_docs = df['document_id'].nunique()
    print(f"Number of unique documents: {num_docs}")
    
    tfidf_dict = calculate_tfidf(df)
    tfidf_summary_dict = generate_summaries(df, tfidf_dict)
    
    print_example_summary(tfidf_summary_dict, doc_id=1)
    save_results(tfidf_summary_dict)


def calculate_tfidf(df):
    """Calculate TF-IDF scores for each document in the dataset."""
    tfidf_dict = {}
    
    for doc_id in range(51):  # Assuming max doc_id is 50
        doc_df = df[df['document_id'] == doc_id].reset_index(drop=True)
        
        if doc_df.empty:
            continue
        
        kalimat_list = doc_df['processed_sentence'].dropna().astype(str).tolist()
        
        if len(kalimat_list) == 0:
            continue
        
        vectorizer = TfidfVectorizer(use_idf=True, norm=None)
        X = vectorizer.fit_transform(kalimat_list)
        
        tfidf_df = pd.DataFrame(
            X.toarray().T,
            index=vectorizer.get_feature_names_out(),
            columns=[f"Kalimat {i+1}" for i in range(len(kalimat_list))]
        )
        
        tfidf_df["W_tfidf"] = tfidf_df.sum(axis=1)
        tfidf_dict[doc_id] = tfidf_df
    
    print(f"TF-IDF calculated for {len(tfidf_dict)} documents")
    return tfidf_dict


def generate_summaries(df, tfidf_dict):
    """Select top 3 sentences from each document based on TF-IDF weights."""
    tfidf_summary_dict = {}
    
    for doc_id, tfidf_df in tfidf_dict.items():
        ws_series = tfidf_df.drop(columns=["W_tfidf"]).sum(axis=0)
        top_kalimat_cols = ws_series.sort_values(ascending=False).head(3).index.tolist()
        sorted_kalimat_cols = sorted(top_kalimat_cols, key=lambda x: int(x.split()[-1]))
        
        doc_df = df[df['document_id'] == doc_id].reset_index(drop=True)
        selected_sentences = []
        
        for kal_col in sorted_kalimat_cols:
            kal_index = int(kal_col.split()[-1]) - 1
            if kal_index < len(doc_df):
                selected_sentences.append(doc_df.loc[kal_index, 'original_sentence'])
        
        tfidf_summary_dict[doc_id] = selected_sentences
    
    return tfidf_summary_dict


def print_example_summary(tfidf_summary_dict, doc_id=0):
    """Print example summary for visualization purposes."""
    if doc_id not in tfidf_summary_dict:
        print(f"No summary available for document {doc_id}")
        return
        
    print(f"\nTop 3 sentences summary for document_id {doc_id}:\n")
    for i, kalimat in enumerate(tfidf_summary_dict[doc_id], 1):
        print(f"{i}. {kalimat}")


def save_results(tfidf_summary_dict):
    """Save extracted top sentences to CSV file."""
    results = []
    for doc_id, sentences in tfidf_summary_dict.items():
        results.append({
            "document_id": doc_id,
            "top_sentences": " ".join(sentences)
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/top_sentences_document.csv", index=False)
    print(f"\nResults saved to data/top_sentences_document.csv")


if __name__ == "__main__":
    main()