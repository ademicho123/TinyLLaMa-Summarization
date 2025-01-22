from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

def calculate_rouge(reference: str, generated: str):
    """Calculate ROUGE score between reference and generated text."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

def calculate_bleu(reference: str, generated: str):
    """Calculate BLEU score between reference and generated text."""
    reference = [reference.split()]
    generated = generated.split()
    return sentence_bleu(reference, generated)
