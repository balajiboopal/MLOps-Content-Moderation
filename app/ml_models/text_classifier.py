import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model
MODEL_NAME = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def predict_toxicity(text):
    """Predicts toxicity score for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    toxicity_score = probabilities[0][1].item()
    return toxicity_score
