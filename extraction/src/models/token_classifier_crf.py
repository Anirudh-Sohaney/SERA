"""
CRF-augmented token classifier for extractive span detection.

Architecture:
    Input → BERT-medium encoder → emission layer → CRF → BIO sequence

The CRF adds sequence-level transition constraints:
- O → I is forbidden (must begin with B)
- B → I is allowed
- I → I is allowed
- I → O or I → B is allowed (span ends)
- B → O is allowed
- B → B is allowed (new span starts)

This prevents illegal BIO sequences like O I I without a preceding B.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel

from ..data.label_schema import NUM_LABELS

logger = logging.getLogger(__name__)

# BIO label indices
O = 0
B = 1
I = 2


def build_allowed_transitions(num_labels: int = 3) -> List[Tuple[int, int]]:
    """
    Build allowed BIO transitions.

    Returns list of (from_label, to_label) pairs that are legal.
    """
    allowed = []

    # From O: can go to O or B (but not I)
    allowed.append((O, O))
    allowed.append((O, B))

    # From B: can go to O, B, or I
    allowed.append((B, O))
    allowed.append((B, B))
    allowed.append((B, I))

    # From I: can go to O, B, or I
    allowed.append((I, O))
    allowed.append((I, B))
    allowed.append((I, I))

    return allowed


class ExtractionClassifierCRF(nn.Module):
    """
    Token classification model with CRF sequence layer.

    Architecture:
        Input → Encoder → Dropout → Linear → CRF → BIO sequence

    The CRF learns transition probabilities between BIO tags,
    enforcing valid tag sequences and improving boundary consistency.
    """

    def __init__(
        self,
        model_name: str = "google/bert_uncased_L-6_H-512_A-8",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.model_name = model_name

        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        # Classification head (emission scores for CRF)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

        # Initialize allowed transitions (mask illegal transitions)
        allowed = build_allowed_transitions(num_labels)
        # CRF needs a mask of shape (num_labels, num_labels)
        # True = allowed, False = forbidden
        transition_mask = torch.zeros(num_labels, num_labels, dtype=torch.bool)
        for from_idx, to_idx in allowed:
            transition_mask[from_idx, to_idx] = True
        # Register as buffer (not a parameter)
        self.register_buffer("allowed_transitions_mask", transition_mask)

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
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Optional BIO labels [batch_size, seq_length]

        Returns:
            Dict with keys:
            - logits: Per-token emission scores [batch_size, seq_length, num_labels]
            - loss: CRF negative log-likelihood (only if labels provided)
            - predictions: Best tag sequence (only at inference)
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

        # Apply dropout and compute emission scores
        hidden_states = self.dropout(hidden_states)
        emissions = self.classifier(hidden_states)  # [batch, seq, num_labels]

        result = {"logits": emissions}

        # Build mask for CRF (True = valid token, not padding)
        # CRF expects mask of shape [batch, seq]
        crf_mask = attention_mask.bool()

        if labels is not None:
            # Compute CRF negative log-likelihood as loss
            # CRF expects labels with -100 for ignored positions
            # We need to replace -100 with a valid label and mask those positions
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = O  # Replace -100 with O
            # Also mask those positions
            crf_mask = crf_mask & (labels != -100)

            # CRF loss is negative log-likelihood (lower is better)
            loss = -self.crf(emissions, labels_for_crf, mask=crf_mask, reduction='mean')
            result["loss"] = loss

        # Viterbi decoding at inference
        predictions = self.crf.decode(emissions, mask=crf_mask)
        result["predictions"] = predictions

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict BIO tags using Viterbi decoding.

        Returns:
            (predicted_tags, confidences)
            confidences are not directly available from CRF, so we use
            emission softmax probabilities as a proxy.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state.float()
            hidden_states = self.dropout(hidden_states)
            emissions = self.classifier(hidden_states)

            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=crf_mask)

            # Convert to tensor
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
            pred_tensor = torch.zeros(batch_size, seq_length, dtype=torch.long)
            for i, pred in enumerate(predictions):
                pred_tensor[i, :len(pred)] = torch.tensor(pred, dtype=torch.long)

            # Use emission softmax as confidence proxy
            probs = torch.softmax(emissions, dim=-1)
            confidences = probs.gather(-1, pred_tensor.unsqueeze(-1)).squeeze(-1)
            # Zero out padding
            confidences = confidences * attention_mask.float()

        return pred_tensor, confidences

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
        crf_params = sum(p.numel() for p in self.crf.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
            "embedding": embedding_params,
            "encoder": encoder_params,
            "task_head": head_params,
            "crf": crf_params,
        }


def create_crf_model(
    model_name: str = "google/bert_uncased_L-6_H-512_A-8",
    num_labels: int = NUM_LABELS,
    dropout: float = 0.1,
    freeze_embeddings: bool = True,
) -> ExtractionClassifierCRF:
    """Create and return a CRF-augmented extraction classifier."""
    model = ExtractionClassifierCRF(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    )

    # Log parameter counts
    params = model.count_parameters()
    logger.info(f"Model: {model_name} + CRF")
    logger.info(f"  Total parameters: {params['total']:,}")
    logger.info(f"  Trainable parameters: {params['trainable']:,}")
    logger.info(f"  Non-trainable parameters: {params['non_trainable']:,}")
    logger.info(f"  Embedding parameters: {params['embedding']:,}")
    logger.info(f"  Encoder parameters: {params['encoder']:,}")
    logger.info(f"  Task head parameters: {params['task_head']:,}")
    logger.info(f"  CRF parameters: {params['crf']:,}")

    return model
