from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime
import torch
from transformers import AutoTokenizer

# Initialize FastAPI app
app = FastAPI()

# Load ONNX model
onnx_path = "notebooks/toxic_comment.onnx"
session = onnxruntime.InferenceSession(onnx_path)

# Load tokenizer
model_path = "notebooks/distilbert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define request body
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    # Tokenize input text
    inputs = tokenizer(
        request.text, 
        return_tensors="np",  
        padding="max_length",  
        truncation=True,       
        max_length=128
    )

    # Convert inputs to numpy (int64)
    input_ids = inputs["input_ids"].astype("int64")
    attention_mask = inputs["attention_mask"].astype("int64")

    # Run ONNX inference
    outputs = session.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})
    logits = torch.tensor(outputs[0])

    # Get predictions
    prediction = torch.argmax(logits, dim=1).item()
    label = "Toxic" if prediction == 1 else "Non-Toxic"

    return {"text": request.text, "prediction": label}
