"""
Phase 3 Project 1: Feed Ranking with Wide & Deep Model
Implement a production-ready ranking model for social media feeds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class WideComponent(nn.Module):
    """
    Wide component: Memorization through cross-product feature transformations.
    Learns direct feature interactions.
    """
    def __init__(self, num_sparse_features: int, embedding_dim: int = 8):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(1000, embedding_dim)  # Simplified: assume 1000 values per feature
            for _ in range(num_sparse_features)
        ])

        # Cross-product features (simplified)
        self.cross_product_dim = num_sparse_features * embedding_dim

    def forward(self, sparse_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_features: (batch_size, num_sparse_features)
        Returns:
            wide_output: (batch_size, cross_product_dim)
        """
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            embedded.append(emb_layer(sparse_features[:, i]))

        # Concatenate all embeddings
        wide_out = torch.cat(embedded, dim=1)
        return wide_out


class DeepComponent(nn.Module):
    """
    Deep component: Generalization through deep neural network.
    Learns feature representations.
    """
    def __init__(
        self,
        dense_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = dense_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, dense_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense_features: (batch_size, dense_dim)
        Returns:
            deep_output: (batch_size, output_dim)
        """
        return self.network(dense_features)


class WideAndDeepModel(nn.Module):
    """
    Wide & Deep Model for feed ranking.

    Components:
    - Wide: Memorization (cross-product features)
    - Deep: Generalization (DNN on embeddings + dense features)
    - Combined: Joint training

    Args:
        num_sparse_features: Number of categorical features
        num_dense_features: Number of continuous features
        embedding_dim: Dimension for sparse feature embeddings
        deep_hidden_dims: Hidden layer dimensions for deep component
        dropout: Dropout probability
    """
    def __init__(
        self,
        num_sparse_features: int,
        num_dense_features: int,
        embedding_dim: int = 8,
        deep_hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1
    ):
        super().__init__()

        # Wide component
        self.wide = WideComponent(num_sparse_features, embedding_dim)

        # Deep component
        # Input: sparse embeddings + dense features
        deep_input_dim = num_sparse_features * embedding_dim + num_dense_features
        self.deep = DeepComponent(deep_input_dim, deep_hidden_dims, dropout)

        # Combined output layer
        combined_dim = self.wide.cross_product_dim + self.deep.output_dim
        self.output = nn.Linear(combined_dim, 1)

    def forward(
        self,
        sparse_features: torch.Tensor,
        dense_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sparse_features: (batch_size, num_sparse_features)
            dense_features: (batch_size, num_dense_features)
        Returns:
            scores: (batch_size, 1) ranking scores
        """
        # Wide path
        wide_out = self.wide(sparse_features)

        # Deep path
        # Combine sparse embeddings + dense features
        deep_input = torch.cat([wide_out, dense_features], dim=1)
        deep_out = self.deep(deep_input)

        # Combine wide + deep
        combined = torch.cat([wide_out, deep_out], dim=1)

        # Final output
        scores = self.output(combined)
        return scores


class FeedRankingModel(nn.Module):
    """
    Complete feed ranking model with multi-task learning.

    Predicts:
    - Engagement score (CTR, like, comment, share)
    - Quality score
    - Combined ranking score
    """
    def __init__(
        self,
        num_sparse_features: int,
        num_dense_features: int,
        embedding_dim: int = 8,
        deep_hidden_dims: List[int] = [512, 256, 128]
    ):
        super().__init__()

        # Shared Wide & Deep backbone
        self.backbone = WideAndDeepModel(
            num_sparse_features,
            num_dense_features,
            embedding_dim,
            deep_hidden_dims
        )

        # Task-specific heads
        shared_dim = self.backbone.wide.cross_product_dim + self.backbone.deep.output_dim

        # Engagement prediction (multi-class or multi-label)
        self.engagement_head = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # [like, comment, share, click]
        )

        # Quality prediction
        self.quality_head = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Final ranking score combination
        self.ranking_weights = nn.Parameter(torch.ones(2) / 2)  # Learned weights

    def forward(
        self,
        sparse_features: torch.Tensor,
        dense_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - 'engagement': (batch_size, 4) engagement predictions
                - 'quality': (batch_size, 1) quality score
                - 'ranking_score': (batch_size, 1) final ranking score
        """
        # Get shared representations
        wide_out = self.backbone.wide(sparse_features)
        deep_input = torch.cat([wide_out, dense_features], dim=1)
        deep_out = self.backbone.deep(deep_input)
        combined = torch.cat([wide_out, deep_out], dim=1)

        # Task-specific predictions
        engagement = self.engagement_head(combined)
        quality = self.quality_head(combined)

        # Combined ranking score
        # Normalize weights
        weights = F.softmax(self.ranking_weights, dim=0)

        # Aggregate engagement (e.g., weighted sum)
        engagement_score = torch.sigmoid(engagement).mean(dim=1, keepdim=True)
        quality_score = torch.sigmoid(quality)

        ranking_score = weights[0] * engagement_score + weights[1] * quality_score

        return {
            'engagement': engagement,
            'quality': quality,
            'ranking_score': ranking_score
        }


def create_synthetic_feed_data(
    batch_size: int,
    num_sparse_features: int = 10,
    num_dense_features: int = 20
):
    """Create synthetic feed ranking data."""
    sparse_features = torch.randint(0, 1000, (batch_size, num_sparse_features))
    dense_features = torch.randn(batch_size, num_dense_features)

    # Synthetic labels
    engagement_labels = torch.randint(0, 2, (batch_size, 4)).float()
    quality_labels = torch.rand(batch_size, 1)

    return {
        'sparse_features': sparse_features,
        'dense_features': dense_features,
        'engagement_labels': engagement_labels,
        'quality_labels': quality_labels
    }


def train_feed_ranker():
    """Train feed ranking model."""
    print("=" * 70)
    print("Training Feed Ranking Model")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = FeedRankingModel(
        num_sparse_features=10,
        num_dense_features=20,
        embedding_dim=8,
        deep_hidden_dims=[256, 128, 64]
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer and losses
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    engagement_criterion = nn.BCEWithLogitsLoss()
    quality_criterion = nn.MSELoss()

    # Training loop
    num_epochs = 5
    batch_size = 128

    for epoch in range(num_epochs):
        model.train()

        # Generate batch
        data = create_synthetic_feed_data(batch_size)

        sparse_features = data['sparse_features'].to(device)
        dense_features = data['dense_features'].to(device)
        engagement_labels = data['engagement_labels'].to(device)
        quality_labels = data['quality_labels'].to(device)

        # Forward pass
        outputs = model(sparse_features, dense_features)

        # Multi-task loss
        engagement_loss = engagement_criterion(outputs['engagement'], engagement_labels)
        quality_loss = quality_criterion(outputs['quality'], quality_labels)

        # Combined loss
        total_loss = engagement_loss + 0.5 * quality_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Engagement Loss: {engagement_loss.item():.4f}")
        print(f"  Quality Loss: {quality_loss.item():.4f}")
        print(f"  Total Loss: {total_loss.item():.4f}")
        print(f"  Ranking Weights: {F.softmax(model.ranking_weights, dim=0).tolist()}")
        print()

    print("=" * 70)
    return model


def demo_inference():
    """Demonstrate inference for ranking."""
    print("\n" + "=" * 70)
    print("Feed Ranking Inference")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = FeedRankingModel(
        num_sparse_features=10,
        num_dense_features=20
    ).to(device)
    model.eval()

    # Generate candidate posts
    num_candidates = 100
    data = create_synthetic_feed_data(num_candidates)

    with torch.no_grad():
        sparse_features = data['sparse_features'].to(device)
        dense_features = data['dense_features'].to(device)

        outputs = model(sparse_features, dense_features)

        # Get ranking scores
        scores = outputs['ranking_score'].squeeze()

        # Rank posts
        ranked_indices = torch.argsort(scores, descending=True)

        print(f"Ranked top 10 posts:")
        for i, idx in enumerate(ranked_indices[:10]):
            print(f"  {i+1}. Post {idx.item()}: Score = {scores[idx].item():.4f}")

    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = train_feed_ranker()
    demo_inference()

    print("\n✓ Feed ranking demonstration complete!")
