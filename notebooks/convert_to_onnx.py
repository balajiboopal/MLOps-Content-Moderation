import torch
import onnx
from transformers import AutoModelForSequenceClassification

# Load fine-tuned model
model_path = "./distilbert-finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Dummy input for ONNX export
dummy_input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.int64)  # Ensure int64
dummy_attention_mask = torch.ones((1, 128), dtype=torch.int64)  # Ensure int64

# Export to ONNX with opset 14
onnx_path = "toxic_comment.onnx"
torch.onnx.export(
    model, 
    (dummy_input_ids, dummy_attention_mask),  # Ensure int64
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=14,
)

print("✅ Model successfully exported to ONNX format!")
