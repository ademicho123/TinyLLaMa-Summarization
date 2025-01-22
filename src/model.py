import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def initialize_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Initialize the model and tokenizer.
    
    Args:
    model_name (str): The name of the model to use. Defaults to "TinyLlama/TinyLlama-1.1B-Chat-v1.0".
    
    Returns:
    model (AutoModelForCausalLM): The initialized model.
    tokenizer (AutoTokenizer): The initialized tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def train_model(model, tokenizer, train_data, epochs=3, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    training_stats = {'losses': []}

    # Filter out empty samples
    train_data = [data for data in train_data if data['article'].strip() and data['summary'].strip()]
    print(f"Number of training samples: {len(train_data)}")

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            print(f"Processing batch {i} of size {len(batch)}")
            
            inputs = tokenizer([item['article'] for item in batch],
                             truncation=True,
                             max_length=512,
                             padding=True,
                             return_tensors="pt")
                             
            labels = tokenizer([item['summary'] for item in batch],
                             truncation=True,
                             max_length=128,
                             padding=True,
                             return_tensors="pt")

            if inputs['input_ids'].size(0) == 0 or labels['input_ids'].size(0) == 0:
                print(f"Skipping empty batch {i}")
                continue
               
            outputs = model(input_ids=inputs['input_ids'].to(device),
                          labels=labels['input_ids'].to(device))
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            batch_count += 1
            
            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            training_stats['losses'].append(avg_loss)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
        else:
            print(f"Epoch {epoch + 1}, No batches processed")
        
    return training_stats

        
def save_model(model, tokenizer, save_path: str):
    """
    Save the model and tokenizer.
    
    Args:
    model (AutoModelForSeq2SeqLM): The model to save.
    tokenizer (AutoTokenizer): The tokenizer to save.
    save_path (str): The directory to save the model.
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def summarize(model, tokenizer, text: str, max_length: int = 50) -> str:
    """
    Generate a summary for the given text.
    
    Args:
    model (AutoModelForCausalLM): The model to use.
    tokenizer (AutoTokenizer): The tokenizer to use.
    text (str): The text to summarize.
    max_length (int): The maximum length of the summary. Defaults to 50.
    
    Returns:
    str: The generated summary.
    """
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
