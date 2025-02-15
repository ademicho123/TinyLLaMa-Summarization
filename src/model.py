import os
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path.cwd().parent) if 'notebooks' in str(Path.cwd()) else str(Path.cwd())
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.preprocessing import preprocess_batch  # Changed import statement

def initialize_model(model_name: str = "facebook/opt-350m"):
    """Initialize the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the input text
        input_text = f"Summarize: {item['article']}\nSummary: {item['highlights']}"
        
        # Tokenize with proper padding and truncation
        encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove the batch dimension since DataLoader will add it
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': encodings['input_ids'].squeeze(0)
        }

class SummarizationModel(L.LightningModule):
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", lr=5e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Explicitly set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model to specific device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        # Ensure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def train_model(train_data, batch_size=1, epochs=3):
    # Force CUDA initialization
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    dataset = TextDataset(train_data, AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True  # Add this for faster data transfer to GPU
    )

    model = SummarizationModel().to(device)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        precision=16,
        strategy='ddp',  # Explicitly set the strategy
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        logger=False
    )

    # Add GPU warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    trainer.fit(model, dataloader)
    return model

def save_model(model, tokenizer, save_path):
    """Save the trained model and tokenizer."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
        
def summarize(model, tokenizer, text: str, max_length: int = 50) -> str:
    """Generate a summary for the given text."""
    if not text:
        return ""
    
    input_text = f"Summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    
    if not inputs:
        return ""
    
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    if not outputs.size(0):
        return ""
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)