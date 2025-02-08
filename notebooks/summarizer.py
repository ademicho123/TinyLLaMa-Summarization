#!/usr/bin/env python
# coding: utf-8

# Text Summarization Model Training
# 
# This notebook demonstrates the process of training a text summarization model using the provided dataset.

# In[1]:


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import sys
from pathlib import Path
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



# Add the project root directory to Python path
project_root = str(Path.cwd().parent) if 'notebooks' in str(Path.cwd()) else str(Path.cwd())
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom modules
from src.preprocessing import load_data, preprocess_data, save_data
from src.model import initialize_model, train_model, save_model, summarize
from src.evaluation import calculate_rouge, calculate_bleu
from src.visualizations import plot_loss


# In[ ]:


# Paths
raw_data_dir = r"C:\Users\ELITEBOOK\OneDrive\Desktop\Projects\TinyLLaMa-Summarization\data\raw"
processed_data_dir = r"C:\Users\ELITEBOOK\OneDrive\Desktop\Projects\TinyLLaMa-Summarization\data\processed"

# Sample size
SAMPLE_SIZE = 0.10

files = ["train.csv", "test.csv", "validation.csv"]
for file in files:
    data = load_data(os.path.join(raw_data_dir, file))
    sampled_data = data.sample(frac=SAMPLE_SIZE, random_state=42)
    articles, highlights = preprocess_data(sampled_data)
    save_data(articles, highlights, processed_data_dir, f"processed_{file}")
    print(f"Processed {len(articles)} samples from {file}")


# In[ ]:


# Paths
processed_data_dir = r"C:\Users\ELITEBOOK\OneDrive\Desktop\Projects\TinyLLaMa-Summarization\data\processed"

# Load processed training data
train_df = pd.read_csv(os.path.join(processed_data_dir, "processed_train.csv"))
train_data = [
    {'article': row['article'], 'highlights': row['highlights']} 
    for _, row in train_df.iterrows()
]

# Initialize model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model, tokenizer = initialize_model(model_name)

# Train model
training_stats = train_model(model, tokenizer, train_data, epochs=5)

# Plot training loss
plot_loss(training_stats['losses'])


# In[ ]:


# Load test data
print("Loading test data...")
with open(os.path.join(processed_data_dir, "processed_test.csv"), "r") as f:
    test_data = json.load(f)

# Evaluate on sample from test set
print("\nEvaluating model on sample test data:")
rouge_scores = []
bleu_scores = []

for idx, item in enumerate(test_data[:5]):
    generated_summary = summarize(model, tokenizer, item['article'])
    rouge = calculate_rouge(item['highlights'], generated_summary)
    bleu = calculate_bleu(item['highlights'], generated_summary)
    
    rouge_scores.append(rouge)
    bleu_scores.append(bleu)

# Results dictionary
results = {
    'average_scores': {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'bleu': avg_bleu
    },
    'example_predictions': [
        {
            'article': item['article'],
            'reference_summary': item['highlights'],
            'generated_summary': summarizer.summarize(item['article']),
        }
        for item in test_data[:5]
    ]
}


# In[ ]:


# Create results dictionary
results = {
    'average_scores': {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'bleu': avg_bleu
    },
    'example_predictions': [
        {
            'article': item['article'],
            'reference_summary': item['summary'],
            'generated_summary': summarizer.summarize(item['article']),
        }
        for item in test_data[:5]
    ]
}

# Save results
os.makedirs('../reports', exist_ok=True)
with open('../reports/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Evaluation results saved to reports/evaluation_results.json")


# In[ ]:


# Save model
print("Saving model...")
save_model(model, tokenizer, "../models/tiny-llama-model")

