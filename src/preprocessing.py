import os
import json
import re
from typing import List, Tuple
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> List[dict]:
    """Load raw JSONL data from the specified file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def clean_text(text: str) -> str:
    """Clean text by removing special characters and extra spaces."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def preprocess_data(data: List[dict]) -> Tuple[List[str], List[str]]:
    """Extract and preprocess articles and summaries."""
    articles = []
    summaries = []
    for item in data:
        if 'content' in item:
            articles.append(clean_text(item['content']))
            summaries.append("")  # You may want to add a default summary here
        elif 'article' in item and 'summary' in item:
            articles.append(clean_text(item['article']))
            summaries.append(clean_text(item['summary']))
        else:
            print(f"Skipping item with missing keys: {item}")
    return articles, summaries

def split_and_save_data(articles: List[str], summaries: List[str], output_dir: str, test_size: float = 0.2):
    """Split data into training and testing sets, then save to processed folder."""
    os.makedirs(output_dir, exist_ok=True)
    train_articles, test_articles, train_summaries, test_summaries = train_test_split(
        articles, summaries, test_size=test_size, random_state=42
    )
    train_data = [{"article": a, "summary": s} for a, s in zip(train_articles, train_summaries)]
    test_data = [{"article": a, "summary": s} for a, s in zip(test_articles, test_summaries)]

    with open(os.path.join(output_dir, "train.json"), "w") as train_file:
        json.dump(train_data, train_file, indent=4)
    with open(os.path.join(output_dir, "test.json"), "w") as test_file:
        json.dump(test_data, test_file, indent=4)

    print(f"Train and test datasets saved in {output_dir}")