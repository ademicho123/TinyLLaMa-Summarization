import pandas as pd
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple
import os
from sklearn.model_selection import train_test_split

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

def process_text(text: str) -> str:
    """Clean and preprocess text."""
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, '')
    
    # Remove emoticons
    text = re.sub(r'[:;=]-?[()DPp]', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Initialize NLP tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize, remove stop words, and lemmatize
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_data(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Process articles and highlights."""
    data = data.dropna(subset=['article', 'highlights'])
    data['processed_article'] = data['article'].apply(process_text)
    data['processed_highlights'] = data['highlights'].apply(process_text)
    return data['processed_article'].tolist(), data['processed_highlights'].tolist()

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(file_path)

def save_data(articles: List[str], highlights: List[str], output_dir: str, filename: str):
    """Save processed data."""
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame({
        'article': articles,
        'highlights': highlights
    }).to_csv(os.path.join(output_dir, filename), index=False)