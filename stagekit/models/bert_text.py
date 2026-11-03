import torch
import torch.nn as nn
from transformers import AutoModel


class BertTextClassifier(nn.Module):
    """
    Text-only BERT classifier.
    Uses [CLS] embedding followed by an MLP head.
    """

    def __init__(
        self,
        bert_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_name)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
            # Permanently in eval mode — weights are never saved or
            # fine-tuned, they come from the HuggingFace checkpoint.
            self.bert.eval()

        bert_dim = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [CLS] token embedding
        cls_emb = bert_out.last_hidden_state[:, 0]

        logits = self.classifier(cls_emb).squeeze(-1)
        return logits