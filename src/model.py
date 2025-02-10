import os
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from src.preprocessing import preprocess_batch

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

# Lightning Model Class
class SummarizationModel(L.LightningModule):
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", lr=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# Training Function Using Lightning AI
def train_model(train_data, batch_size=4, epochs=3):
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("🔹 Debug: Checking first training sample...")
    print(train_data[0])  # Print first training example

    # Ensure preprocess_batch is recognized
    print("🔹 Debug: Checking preprocess_batch function...")
    print(preprocess_batch)

    dataset = [preprocess_batch([item], tokenizer) for item in train_data]
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SummarizationModel()

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1
    )

    trainer.fit(model, dataloader)

    # Save model
    model.model.save_pretrained("./models/summarizer_model")
    tokenizer.save_pretrained("./models/summarizer_model")

    return model

def save_model(model, tokenizer, save_path):
    """
    Save the trained model and tokenizer.
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
    if not text:
        return ""
    
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True)
    
    if not inputs:
        return ""
    
    outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    
    if not outputs:
        return ""
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

