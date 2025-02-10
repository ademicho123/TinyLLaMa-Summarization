import os
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

# Add the project root directory to Python path
project_root = str(Path.cwd().parent) if 'notebooks' in str(Path.cwd()) else str(Path.cwd())
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.preprocessing import preprocess_batch  # Changed import statement

def initialize_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Initialize the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
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
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        self.log("train_loss", outputs.loss)
        return outputs.loss

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

def train_model(train_data, batch_size=4, epochs=3):
    """Train the summarization model."""
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset using custom TextDataset class
    dataset = TextDataset(train_data, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )

    model = SummarizationModel()
    
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4
    )

    trainer.fit(model, dataloader)

    # Save model
    os.makedirs("./models/summarizer_model", exist_ok=True)
    model.model.save_pretrained("./models/summarizer_model")
    tokenizer.save_pretrained("./models/summarizer_model")

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