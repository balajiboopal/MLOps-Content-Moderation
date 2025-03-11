import torch
from torch.utils.data import Dataset
import pandas as pd
import ast  # To safely convert string representations of lists to actual lists

class ToxicCommentDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        # Convert string representations of lists back to actual lists
        self.data["input_ids"] = self.data["input_ids"].apply(ast.literal_eval)
        self.data["attention_mask"] = self.data["attention_mask"].apply(ast.literal_eval)

        # Ensure labels are in one-hot format (Fixes Target Size Issue)
        self.data["labels"] = self.data["label"].apply(lambda x: [1, 0] if x == 0 else [0, 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.data.iloc[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(self.data.iloc[idx]["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.data.iloc[idx]["labels"], dtype=torch.float),  # Ensure Float type for BCE Loss
        }
        return item

# Load datasets to check if they are working
if __name__ == "__main__":
    train_dataset = ToxicCommentDataset("../data/train_tokenized.csv")
    val_dataset = ToxicCommentDataset("../data/val_tokenized.csv")

    print(f"✅ Dataset loaded successfully! Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
