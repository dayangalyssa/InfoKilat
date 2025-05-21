import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

manual_df = pd.read_csv('../data/human_evaluation.csv')
manual_df.columns

full_df = pd.read_csv('../data/indot5_summary.csv')

# Pastikan document_id bertipe string
manual_df['document_id'] = manual_df['document_id'].astype(str)
full_df['document_id'] = full_df['document_id'].astype(str)

# Merge berdasarkan document_id
merged_df = pd.merge(full_df, manual_df[['document_id', 'manual_summary']], on='document_id', how='inner')

def get_similarity(row):
    texts = [str(row['indot5_summary']), str(row['manual_summary'])]
    tfidf = TfidfVectorizer().fit_transform(texts)
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

merged_df['cosine_similarity'] = merged_df.apply(get_similarity, axis=1)

merged_df[['document_id', 'indot5_summary', 'manual_summary', 'cosine_similarity']].head(50)

average_cosine = merged_df['cosine_similarity'].mean()
average_cosine = average_cosine*100
print(f"Average Cosine Similarity: {average_cosine:.0f}%")
