from fastapi import FastAPI
from pydantic import BaseModel
from models import Summarizer

app = FastAPI()
summarizer = Summarizer()

class TextInput(BaseModel):
    text: str

@app.post("/summarize/")
def summarize_text(input: TextInput):
    summary = summarizer.summarize(input.text)
    return {"summary": summary}
