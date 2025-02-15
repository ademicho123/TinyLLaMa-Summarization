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
    def __init__(self, model_name="facebook/opt-350m", lr=5e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        # PyTorch Lightning will handle device placement, so no need to manually move tensors
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

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
    
    # Initialize dataset and dataloader
    dataset = TextDataset(train_data, AutoTokenizer.from_pretrained("facebook/opt-350m"))
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True  # Add this for faster data transfer to GPU
    )

    # Initialize the model
    model = SummarizationModel()

    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16,
        strategy='ddp_notebook',  # Use a notebook-compatible strategy
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        logger=False
    )

    # Add GPU warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Train the model
    trainer.fit(model, dataloader)
    
    # Return training stats (e.g., losses)
    return {"losses": [loss.item() for loss in trainer.callback_metrics.get("train_loss", [])]}

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