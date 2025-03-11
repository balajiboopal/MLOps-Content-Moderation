import onnxruntime
import torch
from transformers import AutoTokenizer

# Load tokenizer
model_path = "./distilbert-finetuned"  # Path to fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load ONNX model
onnx_path = "toxic_comment.onnx"
session = onnxruntime.InferenceSession(onnx_path)

# Sample text for inference
text = "I love this community!"
inputs = tokenizer(
    text, 
    return_tensors="np",  # Ensure NumPy format
    padding="max_length",  # Pad to max length (128)
    truncation=True,       # Truncate if longer than 128
    max_length=128
)

# Convert inputs to numpy and set dtype to int64
input_ids = inputs["input_ids"].astype("int64")
attention_mask = inputs["attention_mask"].astype("int64")

# Run ONNX inference
outputs = session.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})
logits = torch.tensor(outputs[0])

# Get predictions
prediction = torch.argmax(logits, dim=1).item()
label = "Toxic" if prediction == 1 else "Non-Toxic"

print(f"✅ Prediction: {label}")
