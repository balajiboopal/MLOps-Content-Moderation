import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset import ToxicCommentDataset

# Load fine-tuned model
model_path = "./distilbert-finetuned"  # Adjust path if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load validation dataset (since test set was not created)
val_dataset = ToxicCommentDataset("../data/val_tokenized.csv")

# Make predictions
preds, labels = [], []
model.eval()
for item in val_dataset:
    inputs = {
        "input_ids": torch.tensor(item["input_ids"]).unsqueeze(0).clone().detach(),
        "attention_mask": torch.tensor(item["attention_mask"]).unsqueeze(0).clone().detach(),
    }
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    preds.append(prediction)
    labels.append(item["labels"].argmax().item())  # FIXED LINE

# Compute metrics
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)

print(f"✅ Model Evaluation:")
print(f"🔹 Accuracy: {accuracy:.2f}")
print(f"🔹 Precision: {precision:.2f}")
print(f"🔹 Recall: {recall:.2f}")
print(f"🔹 F1 Score: {f1:.2f}")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(labels, preds)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Toxic", "Toxic"], yticklabels=["Non-Toxic", "Toxic"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
