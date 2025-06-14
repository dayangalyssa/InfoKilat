# InfoKilat

## Indonesian News Summarization Tool

InfoKilat is a web application that automatically generates concise summaries of Indonesian news articles. Built using advanced Natural Language Processing techniques, it combines TF-IDF feature extraction with a BERT2BERT neural network to produce high-quality summaries that capture the essence of news articles.

## Features

- **URL-based Summarization**: Simply paste any Indonesian news article URL to generate a summary
- **Adaptive Content Extraction**: Automatically extracts article content from various news sites
- **Smart Chunking**: Handles articles of any length by intelligently breaking them into manageable chunks
- **State-of-the-art NLP**: Leverages modern transformer-based language models

## How It Works

### 1. Text Preprocessing

The system first tokenizes the article into sentences and performs preprocessing:
- Case folding (lowercase)
- Tokenization (breaking text into words)
- Stopword removal (removing common words like "dan", "yang", etc.)
- Stemming (reducing words to their root form)

### 2. TF-IDF Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to identify the most important sentences:

- **Term Frequency (TF)**: Measures how frequently a term appears in a document
- **Inverse Document Frequency (IDF)**: Measures how important a term is by scaling down commonly used words
- **Sentence Scoring**: Each sentence receives a score based on the TF-IDF values of its terms

The system selects the top 3 sentences with the highest TF-IDF scores from each article as key content.


### 3. BERT2BERT Summarization

The extracted top sentences are then passed to a BERT2BERT model trained specifically for Indonesian summarization:

- **Encoder-Decoder Architecture**: Uses two BERT models - one to understand the input text and another to generate the summary
- **Fine-tuned for Indonesian**: Adapted to understand Indonesian language nuances
- **Coherent Output**: Produces grammatically correct and contextually appropriate summaries

This two-stage approach combines the precision of statistical methods (TF-IDF) with the fluency and contextual understanding of neural networks (BERT2BERT).

## Performance

Our model has been evaluated using:

1. **Cosine Similarity**: Measuring textual similarity between generated summaries and human-written references
2. **BERTScore**: Evaluating semantic similarity at a deeper level

## Usage

### Requirements

```
pandas
requests
beautifulsoup4
torch
numpy
scikit-learn
nltk
transformers
matplotlib
seaborn
kagglehub
Sastrawi
bert-score
streamlit
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/InfoKilat.git
cd InfoKilat

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Project Structure

```
InfoKilat/
├── app.py                    # Streamlit web application
├── requirements.txt          # Project dependencies
├── data/                     # Data directory
├── script/                   # Python scripts
│   ├── embedding.py          # TF-IDF implementation
│   ├── model.py              # BERT2BERT model implementation
│   └── preprocessing.py      # Text preprocessing
└── notebook/                 # Jupyter notebooks
    ├── preprocessing.ipynb
    └── evaluation.ipynb
```

## Contributors

InfoKilat @2025 by Yolanda - Sekar - Dayang
