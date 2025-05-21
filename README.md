# 📰 InfoKilat - Indonesian News Summarizer

InfoKilat is an advanced text summarization system designed specifically for Indonesian news articles. The application uses a combination of TF-IDF (Term Frequency-Inverse Document Frequency) text extraction and state-of-the-art BERT2BERT language model to generate concise, accurate summaries of news content from any URL.

## Features

- **URL-based News Extraction**: Automatically extracts content from Indonesian news websites
- **TF-IDF Sentence Extraction**: Identifies the most important sentences using statistical methods
- **Advanced Summarization**: Leverages the "cahya/bert2bert-indonesian-summarization" model for high-quality Indonesian summaries
- **Customizable Parameters**: Adjust the number of key sentences and summary length
- **Simple Interface**: Clean, user-friendly Streamlit interface
- **Performance Statistics**: Shows word count and processing time information

## Project Structure

```bash
infokilat/
├── data/
│   └── preprocessed_sentences.csv  # Preprocessed sentences data
│
├── notebook/
│   └── preprocessing.ipynb         # Notebook with data preprocessing steps
│
├── script/
│   ├── embedding.py                # TF-IDF implementation
│   └── model.py                    # BERT2BERT model implementation
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## How It Works

1. **Text Extraction**: When a user inputs a news URL, InfoKilat extracts the article content using BeautifulSoup
2. **Sentence Selection**: The TF-IDF algorithm identifies the most informative sentences
3. **Summary Generation**: The selected sentences are passed to the BERT2BERT model for abstractive summarization
4. **Result Display**: The summary is displayed in a clean, readable format with statistics

## Installation

1. Clone this repository:
```bash
git clone https://github.com/dayangalyssa/infokilat.git
cd infokilat
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Enter a URL of an Indonesian news article in the input field

3. Adjust summary parameters if needed (in the sidebar)

4. View the generated summary

## Technical Implementation

### TF-IDF Sentence Extraction

The TF-IDF implementation follows these steps:
- Tokenize the article into sentences
- Create a document-term matrix
- Calculate TF-IDF scores for each term
- Rank sentences based on their combined term scores
- Select the top N sentences with the highest scores

## Future Improvements

- Add support for PDF document summarization
- Implement keyword extraction
- Create a batch processing mode for multiple articles

