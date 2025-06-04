import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

def main():
    """Main function to generate BERT2BERT summaries from TF-IDF extracted sentences."""

    tfidf_summary_dict = load_preprocessed_sentences()
    
    tokenizer, model = load_bert2bert_model()
    
    tfidf_to_bert2bert_summary_dict = generate_bert2bert_summaries(tfidf_summary_dict, tokenizer, model)
    
    # Save results
    save_summaries(tfidf_to_bert2bert_summary_dict)


def load_preprocessed_sentences():
    """Load preprocessed sentences from the TF-IDF output file."""
    try:
       
        df = pd.read_csv("data/top_sentences_document.csv")
        print(f"Loaded {len(df)} documents from top_sentences_document.csv")
        
        tfidf_summary_dict = {}
        for _, row in df.iterrows():
            doc_id = int(row['document_id'])
 
            sentences = str(row['top_sentences']).split('. ')
            tfidf_summary_dict[doc_id] = sentences
            
        return tfidf_summary_dict
        
    except FileNotFoundError:
        print("Error: File data/top_sentences_document.csv not found.")
        print("Please run embedding.py first to generate TF-IDF extracted sentences.")
        return {}
    except Exception as e:
        print(f"Error loading preprocessed sentences: {e}")
        return {}


def load_bert2bert_model():
    """Load BERT2BERT model and tokenizer for Indonesian summarization."""
    print("Loading BERT2BERT model and tokenizer...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("cahya/bert2bert-indonesian-summarization")
    
    elapsed_time = time.time() - start_time
    print(f"Model loaded in {elapsed_time:.2f} seconds")
    
    return tokenizer, model


def summarize_text(text, tokenizer, model, max_input_length=512, max_output_length=150):
    """Generate summary for a given text using BERT2BERT model."""
    input_text = "summarize: " + text
    
    # Encode and truncate to max_input_length
    inputs = tokenizer.encode(
        input_text, 
        return_tensors="pt", 
        max_length=max_input_length, 
        truncation=True
    )
    
    # Generate summary
    summary_ids = model.generate(
        inputs,
        max_length=max_output_length,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def generate_bert2bert_summaries(tfidf_summary_dict, tokenizer, model):
    """Generate BERT2BERT summaries from TF-IDF extracted sentences."""
    tfidf_to_bert2bert_summary_dict = {}
    total_docs = len(tfidf_summary_dict)
    
    print(f"Generating summaries for {total_docs} documents...")
    start_time = time.time()
    
    for i, (doc_id, kalimat_list) in enumerate(tfidf_summary_dict.items()):
        input_text = ' '.join(kalimat_list).strip()
        
        try:
            summary = summarize_text(input_text, tokenizer, model)
            tfidf_to_bert2bert_summary_dict[doc_id] = summary
            
            if (i + 1) % 5 == 0 or (i + 1) == total_docs:
                elapsed = time.time() - start_time
                docs_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"Processed {i+1}/{total_docs} documents ({docs_per_sec:.2f} docs/sec)")
                
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")
            tfidf_to_bert2bert_summary_dict[doc_id] = "(Summary generation failed)"
    
    elapsed_time = time.time() - start_time
    print(f"All summaries generated in {elapsed_time:.2f} seconds")
    
    return tfidf_to_bert2bert_summary_dict


def save_summaries(tfidf_to_bert2bert_summary_dict):
    """Save BERT2BERT summaries to CSV file."""
  
    tfidf_to_bert2bert_summary_df = pd.DataFrame.from_dict(
        tfidf_to_bert2bert_summary_dict,
        orient='index',
        columns=['tfidf_to_bert2bert_summary']
    ).reset_index().rename(columns={'index': 'document_id'})
    
    tfidf_to_bert2bert_summary_df['document_id'] = tfidf_to_bert2bert_summary_df['document_id'].astype(int)
    
    # Save to CSV
    tfidf_to_bert2bert_summary_df.to_csv("data/ringkasan_tfidf_bert2bert.csv", index=False)
    print(f"Summaries saved to data/ringkasan_tfidf_bert2bert.csv")
    
    return tfidf_to_bert2bert_summary_df


if __name__ == "__main__":
    main()
