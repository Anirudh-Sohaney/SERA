"""
E7 Stage-2 Span Filter Model.

Binary classifier that determines whether a candidate span from Stage 1
represents persistent project-memory information.

Architecture:
    [CLS] PROMPT [SEP] CANDIDATE [SEP]
        ↓
    BERT-medium encoder
        ↓
    [CLS] representation
        ↓
    Dropout → Linear → Sigmoid
        ↓
    P(KEEP)

The model receives the full prompt plus the candidate span as a
sequence-pair classification task. It learns to distinguish between
true project-memory spans and false-positive extractions.

Key constraints:
- Under 100M parameters
- Must preserve unseen-vocabulary performance
- Must not generate text — only KEEP/REJECT decisions
- Stage-1 offsets are passed through unchanged on KEEP
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class SpanFilter(nn.Module):
    """
    Binary span classifier for E7 Stage 2.

    Takes [CLS] prompt [SEP] candidate [SEP] and outputs P(KEEP).

    Uses the [CLS] token representation from the encoder, projected
    through a classification head to produce a binary logit.
    """

    def __init__(
        self,
        model_name: str = "google/bert_uncased_L-6_H-512_A-8",
        dropout: float = 0.1,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        self.model_name = model_name

        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        # Classification head: [CLS] → dropout → linear → logit
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

        # Freeze embeddings if requested
        if freeze_embeddings:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            logger.info("Frozen encoder embeddings")

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
                Format: [CLS] prompt [SEP] candidate [SEP]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Optional binary labels [batch_size] (1=KEEP, 0=REJECT)

        Returns:
            Dict with keys:
            - logits: Binary logits [batch_size]
            - probs: P(KEEP) [batch_size]
            - loss: BCE loss (only if labels provided)
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get [CLS] representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        cls_output = cls_output.float()

        # Classify
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)  # [batch]
        probs = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "probs": probs,
        }

        # Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            result["loss"] = loss

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict KEEP/REJECT with confidence scores.

        Returns:
            (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
            probs = result["probs"]
            predictions = (probs >= threshold).long()

        return predictions, probs

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable

        embedding_params = sum(
            p.numel() for p in self.encoder.embeddings.parameters()
        )
        encoder_params = sum(
            p.numel()
            for name, p in self.encoder.named_parameters()
            if "embeddings" not in name
        )
        head_params = sum(p.numel() for p in self.classifier.parameters()) + sum(
            p.numel() for p in self.dropout.parameters()
        )

        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
            "embedding": embedding_params,
            "encoder": encoder_params,
            "task_head": head_params,
        }


def create_span_filter(
    model_name: str = "google/bert_uncased_L-6_H-512_A-8",
    dropout: float = 0.1,
    freeze_embeddings: bool = True,
) -> SpanFilter:
    """Create and return a span filter model."""
    model = SpanFilter(
        model_name=model_name,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    )

    # Log parameter counts
    params = model.count_parameters()
    logger.info(f"SpanFilter model: {model_name}")
    logger.info(f"  Total parameters: {params['total']:,}")
    logger.info(f"  Trainable parameters: {params['trainable']:,}")
    logger.info(f"  Non-trainable parameters: {params['non_trainable']:,}")
    logger.info(f"  Embedding parameters: {params['embedding']:,}")
    logger.info(f"  Encoder parameters: {params['encoder']:,}")
    logger.info(f"  Task head parameters: {params['task_head']:,}")

    return model
