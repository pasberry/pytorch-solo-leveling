"""
Phase 3 Project 3: Embedding-Based Retrieval with Two-Tower Model
Implement a two-tower architecture for semantic search and recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class TextEncoder(nn.Module):
    """
    Text encoder tower using transformer-style architecture.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            embeddings: (batch, embedding_dim)
        """
        # Embed tokens
        x = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        x = self.pos_encoding(x)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()

        # Transformer encoding
        # Convert mask to transformer format
        key_padding_mask = (attention_mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Pool: mean over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1)
        sum_embeddings = (x * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Project
        embeddings = self.projection(mean_embeddings)

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class ItemEncoder(nn.Module):
    """
    Item encoder tower (e.g., for products, videos, documents).
    Uses features like ID, category, metadata.
    """
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        num_categories: int = 1000,
        hidden_dims: list = [512, 256]
    ):
        super().__init__()

        # Item ID embedding
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Category embedding
        self.category_embedding = nn.Embedding(num_categories, embedding_dim // 2)

        # MLP on concatenated features
        input_dim = embedding_dim + embedding_dim // 2
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.projection = nn.Linear(prev_dim, embedding_dim)

    def forward(self, item_ids: torch.Tensor, category_ids: torch.Tensor):
        """
        Args:
            item_ids: (batch,)
            category_ids: (batch,)
        Returns:
            embeddings: (batch, embedding_dim)
        """
        # Embed item and category
        item_emb = self.item_embedding(item_ids)
        cat_emb = self.category_embedding(category_ids)

        # Concatenate
        x = torch.cat([item_emb, cat_emb], dim=1)

        # MLP
        x = self.mlp(x)

        # Project
        embeddings = self.projection(x)

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class TwoTowerModel(nn.Module):
    """
    Two-tower model for retrieval.

    Training: Maximize similarity between positive pairs
    Inference: Encode queries and items separately, retrieve via similarity
    """
    def __init__(
        self,
        text_encoder: nn.Module,
        item_encoder: nn.Module,
        temperature: float = 0.07
    ):
        super().__init__()

        self.text_encoder = text_encoder
        self.item_encoder = item_encoder
        self.temperature = temperature

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor],
        item_ids: torch.Tensor,
        category_ids: torch.Tensor
    ):
        """
        Forward pass for training with in-batch negatives.

        Args:
            query_input_ids: (batch, seq_len)
            query_attention_mask: (batch, seq_len)
            item_ids: (batch,)
            category_ids: (batch,)
        Returns:
            loss: Contrastive loss
        """
        # Encode queries and items
        query_emb = self.text_encoder(query_input_ids, query_attention_mask)
        item_emb = self.item_encoder(item_ids, category_ids)

        # Compute similarity matrix
        # (batch, batch) where [i, j] = similarity(query_i, item_j)
        logits = torch.matmul(query_emb, item_emb.T) / self.temperature

        # Labels: positive pairs on diagonal
        batch_size = query_emb.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Contrastive loss (both directions)
        loss_q2i = F.cross_entropy(logits, labels)
        loss_i2q = F.cross_entropy(logits.T, labels)

        loss = (loss_q2i + loss_i2q) / 2

        return loss

    def encode_queries(self, input_ids, attention_mask=None):
        """Encode text queries."""
        return self.text_encoder(input_ids, attention_mask)

    def encode_items(self, item_ids, category_ids):
        """Encode items."""
        return self.item_encoder(item_ids, category_ids)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def create_synthetic_data(batch_size: int, vocab_size: int = 10000, num_items: int = 100000):
    """Create synthetic query-item pairs."""
    # Queries
    seq_len = 20
    query_input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    query_attention_mask = torch.ones_like(query_input_ids).float()

    # Items
    item_ids = torch.randint(0, num_items, (batch_size,))
    category_ids = torch.randint(0, 1000, (batch_size,))

    return query_input_ids, query_attention_mask, item_ids, category_ids


def train_two_tower_model():
    """Train two-tower retrieval model."""
    print("=" * 70)
    print("Training Two-Tower Retrieval Model")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create encoders
    text_encoder = TextEncoder(
        vocab_size=10000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_heads=4
    )

    item_encoder = ItemEncoder(
        num_items=100000,
        embedding_dim=128,
        num_categories=1000,
        hidden_dims=[256, 128]
    )

    # Two-tower model
    model = TwoTowerModel(text_encoder, item_encoder, temperature=0.07).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    batch_size = 64

    for epoch in range(num_epochs):
        model.train()

        # Generate batch
        query_ids, query_mask, item_ids, cat_ids = create_synthetic_data(batch_size)
        query_ids = query_ids.to(device)
        query_mask = query_mask.to(device)
        item_ids = item_ids.to(device)
        cat_ids = cat_ids.to(device)

        # Forward pass
        loss = model(query_ids, query_mask, item_ids, cat_ids)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")

    print("\n" + "=" * 70)
    return model


def demo_retrieval(model):
    """Demonstrate retrieval with trained model."""
    print("\n" + "=" * 70)
    print("Retrieval Demonstration")
    print("=" * 70)

    device = next(model.parameters()).device
    model.eval()

    # Create query
    query_ids = torch.randint(1, 10000, (1, 20)).to(device)
    query_mask = torch.ones_like(query_ids).float()

    # Create item catalog (100 items)
    num_catalog_items = 100
    item_ids = torch.arange(num_catalog_items).to(device)
    cat_ids = torch.randint(0, 1000, (num_catalog_items,)).to(device)

    with torch.no_grad():
        # Encode query
        query_emb = model.encode_queries(query_ids, query_mask)  # (1, embedding_dim)

        # Encode all catalog items
        item_embs = model.encode_items(item_ids, cat_ids)  # (100, embedding_dim)

        # Compute similarities
        similarities = torch.matmul(query_emb, item_embs.T).squeeze(0)  # (100,)

        # Get top-k
        top_k = 10
        top_scores, top_indices = torch.topk(similarities, k=top_k)

        print(f"Top {top_k} retrieved items:")
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            print(f"  {i+1}. Item {idx.item()}: Score = {score.item():.4f}")

    print("\n" + "=" * 70)


def demo_batch_retrieval():
    """Demonstrate batch retrieval for efficiency."""
    print("\n" + "=" * 70)
    print("Batch Retrieval for Scale")
    print("=" * 70)

    print("For large-scale retrieval (millions of items):")
    print()
    print("1. Pre-compute and store all item embeddings")
    print("2. Use approximate nearest neighbor search (FAISS, ScaNN)")
    print("3. Index embeddings for fast retrieval")
    print("4. Encode queries on-the-fly")
    print()
    print("Retrieval pipeline:")
    print("  Query → Text Encoder → Query Embedding")
    print("  Query Embedding + Item Index → ANN Search → Top-K Items")
    print()
    print("This enables sub-millisecond retrieval over millions of items")

    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = train_two_tower_model()
    demo_retrieval(model)
    demo_batch_retrieval()

    print("\n✓ Two-tower retrieval demonstration complete!")
