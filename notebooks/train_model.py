from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import torch
from dataset import ToxicCommentDataset

# Load datasets
train_dataset = ToxicCommentDataset("../data/train_tokenized.csv")
val_dataset = ToxicCommentDataset("../data/val_tokenized.csv")

# Load DistilBERT model
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save fine-tuned model
model.save_pretrained("./distilbert-finetuned")
tokenizer.save_pretrained("./distilbert-finetuned")
