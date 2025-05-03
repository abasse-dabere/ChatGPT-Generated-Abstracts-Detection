import torch
import torch.nn as nn
from transformers import AutoModel

class DistilRoBERTaWithFeatures(nn.Module):
    def __init__(self, text_model_name="distilroberta-base", num_features=38, hidden_size=256):
        super().__init__()
        # DistilRoBERTa
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.dropout = nn.Dropout(0.3)

        # Combined classifier for text + numerical features
        self.classifier = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size + num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)  # Binary output
        )

    def forward(self, input_ids, attention_mask, features):
        # Text encoding via DistilRoBERTa
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Embeddings fusion with numerical features
        x = torch.cat((cls_embedding, features), dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
