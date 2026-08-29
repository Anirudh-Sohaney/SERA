"""
Model architecture for extractive span detection.

Architecture: Pretrained encoder + token classification head.

Selected encoder: microsoft/deberta-v3-small
- 44M parameters (well under 500M limit)
- 512 token context length
- Strong performance on token-level tasks
- Pretrained on diverse text including code-adjacent content
- Apache-2.0 license

Architecture rationale:
1. Token classification (BIO tagging) is preferred over span prediction
   for the prototype because:
   - Simpler implementation
   - O(n) complexity vs O(n²) for span prediction
   - Well-understood training dynamics
   - Sufficient for demonstrating generalization

2. DeBERTa is preferred over BERT because:
   - Disentangled attention captures position-content interactions better
   - Deconvolutional sentiment improves token-level predictions
   - Strong performance on NER and span extraction tasks

3. Embeddings are frozen to:
   - Reduce trainable parameters
   - Prevent catastrophic forgetting of pretrained representations
   - Speed up training on CPU

The model outputs per-token logits over the BIO label set.
At inference, BIO tags are converted to character offsets using
the tokenizer's offset_mapping.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..data.label_schema import NUM_LABELS

logger = logging.getLogger(__name__)


class ExtractionClassifier(nn.Module):
    """
    Token classification model for project-memory span extraction.

    Architecture:
        Input → Tokenizer → Encoder → Dropout → Linear → BIO logits

    The encoder produces contextual representations for each token.
    A dropout layer regularizes the representations.
    A linear layer maps to the BIO label space.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        freeze_embeddings: bool = True,
        freeze_encoder_layers: int = 0,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.model_name = model_name

        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Freeze embeddings if requested
        if freeze_embeddings:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            logger.info("Frozen encoder embeddings")

        # Freeze bottom encoder layers if requested
        if freeze_encoder_layers > 0:
            for i in range(freeze_encoder_layers):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = False
            logger.info(f"Frozen bottom {freeze_encoder_layers} encoder layers")

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
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Optional BIO labels [batch_size, seq_length]

        Returns:
            Dict with keys:
            - logits: Per-token logits [batch_size, seq_length, num_labels]
            - loss: Cross-entropy loss (only if labels provided)
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]

        # Ensure float32 for classification head
        hidden_states = hidden_states.float()

        # Apply dropout and classify
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)  # [batch, seq, num_labels]

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # Reshape for loss computation
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fn(active_logits, active_labels)
            result["loss"] = loss

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict BIO tags with confidence scores.

        Returns:
            (predicted_tags, confidences)
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
            logits = result["logits"]
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidences = probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

        return predictions, confidences

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


def create_model(
    model_name: str = "microsoft/deberta-v3-small",
    num_labels: int = NUM_LABELS,
    dropout: float = 0.1,
    freeze_embeddings: bool = True,
) -> ExtractionClassifier:
    """Create and return an extraction classifier."""
    model = ExtractionClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    )

    # Log parameter counts
    params = model.count_parameters()
    logger.info(f"Model: {model_name}")
    logger.info(f"  Total parameters: {params['total']:,}")
    logger.info(f"  Trainable parameters: {params['trainable']:,}")
    logger.info(f"  Non-trainable parameters: {params['non_trainable']:,}")
    logger.info(f"  Embedding parameters: {params['embedding']:,}")
    logger.info(f"  Encoder parameters: {params['encoder']:,}")
    logger.info(f"  Task head parameters: {params['task_head']:,}")

    return model
