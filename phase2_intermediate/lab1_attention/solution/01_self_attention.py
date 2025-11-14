"""
Phase 2, Lab 1: Self-Attention Mechanism from Scratch
Build the foundation of Transformers: self-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism (Scaled Dot-Product Attention)

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(embed_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len, seq_len) - optional

        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        Q = self.query(x)  # (B, L, D)
        K = self.key(x)    # (B, L, D)
        V = self.value(x)  # (B, L, D)

        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, L, L)

        # Scale
        scores = scores / self.scale

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, L, L)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (B, L, D)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    Allows model to jointly attend to information from different representation subspaces
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, 1, seq_len, seq_len) or (batch_size, seq_len, seq_len)

        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)  # (B, L, 3*D)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_h)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: (B, H, L, D_h)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, L, L)
        scores = scores / self.scale

        # Apply mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, L, L)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (B, H, L, D_h)

        # Concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D_h)
        output = output.reshape(batch_size, seq_len, embed_dim)  # (B, L, D)

        # Final linear projection
        output = self.out_proj(output)

        return output, attention_weights


def create_causal_mask(seq_len):
    """
    Create causal (autoregressive) mask for decoder

    Args:
        seq_len: Sequence length

    Returns:
        mask: (seq_len, seq_len) lower triangular matrix
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask

    Args:
        seq: (batch_size, seq_len)
        pad_idx: Padding token index

    Returns:
        mask: (batch_size, 1, 1, seq_len)
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def visualize_attention(attention_weights, src_tokens=None, tgt_tokens=None):
    """
    Visualize attention weights

    Args:
        attention_weights: (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        src_tokens: List of source tokens (optional)
        tgt_tokens: List of target tokens (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if attention_weights.dim() == 3:
        # Multi-head: visualize first head
        attention_weights = attention_weights[0]

    attention_weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention_weights, cmap='viridis')

    if src_tokens and tgt_tokens:
        ax.set_xticks(range(len(src_tokens)))
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha='right')
        ax.set_yticklabels(tgt_tokens)

    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('Attention Weights')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
    print("Attention visualization saved as 'attention_visualization.png'")
    plt.close()


def main():
    print("=" * 60)
    print("Lab 1: Self-Attention Mechanism")
    print("=" * 60)

    # Example 1: Basic self-attention
    print("\n1. Basic Self-Attention")
    print("-" * 60)

    batch_size = 2
    seq_len = 5
    embed_dim = 128

    x = torch.randn(batch_size, seq_len, embed_dim)

    attention = SelfAttention(embed_dim)
    output, weights = attention(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights (first sequence, first 5x5):")
    print(weights[0, :5, :5])

    # Example 2: Multi-head attention
    print("\n2. Multi-Head Attention")
    print("-" * 60)

    num_heads = 8
    mha = MultiHeadAttention(embed_dim, num_heads)

    output, weights = mha(x)

    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {embed_dim // num_heads}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Example 3: Causal masking
    print("\n3. Causal Masking (for Decoder)")
    print("-" * 60)

    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask:\n{causal_mask}")

    # Apply causal mask
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
    output, weights = mha(x, mask=causal_mask)

    print(f"\nAttention with causal mask (first head, first sequence):")
    print(weights[0, 0, :5, :5])

    # Example 4: Padding mask
    print("\n4. Padding Mask")
    print("-" * 60)

    # Create sequence with padding
    vocab_size = 1000
    pad_idx = 0
    seq = torch.randint(1, vocab_size, (batch_size, seq_len))
    seq[0, 3:] = pad_idx  # Add padding to first sequence
    seq[1, 4:] = pad_idx  # Add padding to second sequence

    print(f"Sequence with padding:\n{seq}")

    padding_mask = create_padding_mask(seq, pad_idx)
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Padding mask (first sequence):\n{padding_mask[0, 0, 0, :]}")

    # Example 5: Cross-attention
    print("\n5. Cross-Attention (Query from one sequence, K/V from another)")
    print("-" * 60)

    class CrossAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.mha = MultiHeadAttention(embed_dim, num_heads)

        def forward(self, query, key_value, mask=None):
            """
            Args:
                query: (B, L_q, D)
                key_value: (B, L_kv, D)
            """
            # Modify MHA to accept separate Q and KV
            # For simplicity, we'll just concatenate and use first for Q
            # In practice, you'd modify the MHA implementation
            return self.mha(query, mask)

    query_seq = torch.randn(batch_size, 3, embed_dim)
    kv_seq = torch.randn(batch_size, 5, embed_dim)

    print(f"Query sequence: {query_seq.shape}")
    print(f"Key-Value sequence: {kv_seq.shape}")
    print("(In decoder: query=decoder states, kv=encoder outputs)")

    # Example 6: Attention score interpretation
    print("\n6. Understanding Attention Scores")
    print("-" * 60)

    # Simple example with interpretable tokens
    simple_embed_dim = 8
    simple_seq_len = 4

    simple_x = torch.randn(1, simple_seq_len, simple_embed_dim)
    simple_attn = SelfAttention(simple_embed_dim)

    output, weights = simple_attn(simple_x)

    print("Attention weights (how much each position attends to others):")
    print(weights[0])
    print("\nEach row sums to 1.0 (softmax):")
    print(weights[0].sum(dim=1))

    # Example 7: Visualize attention
    print("\n7. Attention Visualization")
    print("-" * 60)

    tokens = ["The", "cat", "sat", "on", "mat"]
    demo_x = torch.randn(1, len(tokens), embed_dim)
    demo_mha = MultiHeadAttention(embed_dim, num_heads=4)
    _, demo_weights = demo_mha(demo_x)

    visualize_attention(demo_weights[0], tokens, tokens)

    print("\n" + "=" * 60)
    print("Lab 1 Complete!")
    print("=" * 60)
    print("\nKey Concepts Learned:")
    print("1. Self-attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V")
    print("2. Multi-head attention: Parallel attention in different subspaces")
    print("3. Causal masking: Prevent attending to future positions")
    print("4. Padding masking: Ignore padding tokens")
    print("5. Cross-attention: Attend to different sequence (encoder-decoder)")
    print("6. Attention weights sum to 1 for each query position")
    print("\nNext: Build a complete Transformer Encoder!")


if __name__ == "__main__":
    main()
