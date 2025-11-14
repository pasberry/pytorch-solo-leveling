"""
Phase 2, Lab 2: Transformer Encoder from Scratch
Build the complete Transformer Encoder architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (from Lab 1)"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V and split heads
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        output = torch.matmul(attention, V)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        return output, attention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer

    Architecture:
        Input
          ↓
        Multi-Head Attention
          ↓
        Add & Norm (residual + layer norm)
          ↓
        Feed-Forward Network
          ↓
        Add & Norm
          ↓
        Output
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, attention_weights = self.attention(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x, attention_weights


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder

    Architecture:
        Input Embedding
          ↓
        Positional Encoding
          ↓
        N × Encoder Layers
          ↓
        Output
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        num_layers=6,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Scaling for embedding
        self.scale = math.sqrt(embed_dim)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: (batch_size, seq_len) - token indices
            src_mask: (batch_size, 1, seq_len, seq_len) - attention mask

        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: List of attention weights from each layer
        """
        # Embed tokens and scale
        x = self.embedding(src) * self.scale

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through encoder layers
        attention_weights_list = []
        for layer in self.layers:
            x, attention_weights = layer(x, src_mask)
            attention_weights_list.append(attention_weights)

        return x, attention_weights_list


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder for Classification

    Use case: Text classification, sentiment analysis, etc.
    """

    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        num_layers=6,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout, max_len
        )

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, src, src_mask=None):
        # Encode
        encoder_output, attention_weights = self.encoder(src, src_mask)

        # Pool: use [CLS] token (first token) or mean pooling
        # Here we'll use mean pooling
        pooled = encoder_output.mean(dim=1)  # (batch_size, embed_dim)

        # Classify
        logits = self.classifier(pooled)  # (batch_size, num_classes)

        return logits, attention_weights


def create_padding_mask(seq, pad_idx=0):
    """Create padding mask for attention"""
    # seq: (batch_size, seq_len)
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
    return mask.float()


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 60)
    print("Lab 2: Transformer Encoder from Scratch")
    print("=" * 60)

    # Example 1: Positional Encoding
    print("\n1. Positional Encoding")
    print("-" * 60)

    embed_dim = 512
    max_len = 100
    pos_enc = PositionalEncoding(embed_dim, max_len)

    # Visualize positional encoding
    import matplotlib.pyplot as plt
    pe_matrix = pos_enc.pe[0, :max_len, :].detach().numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(pe_matrix.T, aspect='auto', cmap='RdBu')
    plt.xlabel('Position')
    plt.ylabel('Embedding Dimension')
    plt.title('Positional Encoding')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
    print("Positional encoding visualization saved as 'positional_encoding.png'")
    plt.close()

    print(f"Positional encoding shape: {pos_enc.pe.shape}")

    # Example 2: Single Encoder Layer
    print("\n2. Single Transformer Encoder Layer")
    print("-" * 60)

    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048

    x = torch.randn(batch_size, seq_len, embed_dim)
    encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, ff_dim)

    output, attn_weights = encoder_layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Number of parameters: {count_parameters(encoder_layer):,}")

    # Example 3: Complete Transformer Encoder
    print("\n3. Complete Transformer Encoder")
    print("-" * 60)

    vocab_size = 10000
    num_layers = 6

    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers
    )

    # Create sample input (token indices)
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input tokens shape: {src.shape}")

    # Forward pass
    output, attn_weights_list = encoder(src)

    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight tensors: {len(attn_weights_list)}")
    print(f"Total parameters: {count_parameters(encoder):,}")

    # Example 4: Encoder with Padding Mask
    print("\n4. Encoder with Padding Mask")
    print("-" * 60)

    # Create sequence with padding
    pad_idx = 0
    src_with_padding = torch.randint(1, vocab_size, (batch_size, seq_len))
    src_with_padding[0, 5:] = pad_idx  # Pad first sequence
    src_with_padding[1, 7:] = pad_idx  # Pad second sequence

    print(f"Sequence with padding:\n{src_with_padding}")

    # Create padding mask
    padding_mask = create_padding_mask(src_with_padding, pad_idx)
    print(f"Padding mask shape: {padding_mask.shape}")

    # Forward with mask
    output_masked, _ = encoder(src_with_padding, padding_mask)
    print(f"Output with masking: {output_masked.shape}")

    # Example 5: Transformer for Classification
    print("\n5. Transformer Classifier")
    print("-" * 60)

    num_classes = 5  # e.g., sentiment classification (1-5 stars)

    classifier = TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers
    )

    # Sample classification task
    src_text = torch.randint(1, vocab_size, (batch_size, seq_len))
    logits, _ = classifier(src_text)

    print(f"Input: {src_text.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Predicted classes: {logits.argmax(dim=1)}")
    print(f"Total parameters: {count_parameters(classifier):,}")

    # Example 6: Training Loop Skeleton
    print("\n6. Training Loop Skeleton")
    print("-" * 60)

    print("""
    # Training loop for text classification
    model = TransformerClassifier(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in dataloader:
            src, labels = batch

            # Forward
            logits, _ = model(src)

            # Compute loss
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')
    """)

    # Example 7: Model Comparison
    print("\n7. Model Size Comparison")
    print("-" * 60)

    configs = [
        ("Small", 256, 4, 1024, 4),
        ("Base", 512, 8, 2048, 6),
        ("Large", 768, 12, 3072, 12),
    ]

    print(f"{'Config':<10} {'Embed':<8} {'Heads':<8} {'FFN':<8} {'Layers':<8} {'Params':<12}")
    print("-" * 60)

    for name, emb, heads, ffn, layers in configs:
        model = TransformerEncoder(vocab_size, emb, heads, ffn, layers)
        params = count_parameters(model)
        print(f"{name:<10} {emb:<8} {heads:<8} {ffn:<8} {layers:<8} {params:>10,}")

    print("\n" + "=" * 60)
    print("Lab 2 Complete!")
    print("=" * 60)
    print("\nKey Concepts Learned:")
    print("1. Positional Encoding: Add position information to embeddings")
    print("2. Encoder Layer: Multi-head attention + FFN with residual connections")
    print("3. Layer Normalization: Stabilize training")
    print("4. Padding Masks: Handle variable-length sequences")
    print("5. Classification Head: Pool encoder output for downstream tasks")
    print("6. Model Scaling: Trade-offs between size and performance")
    print("\nNext: Build the complete Transformer (Encoder-Decoder)!")


if __name__ == "__main__":
    main()
