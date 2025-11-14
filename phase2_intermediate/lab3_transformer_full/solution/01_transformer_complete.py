"""
Phase 2, Lab 3: Complete Transformer (Encoder-Decoder)
Build the full Transformer architecture for sequence-to-sequence tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with support for cross-attention"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Separate projections for Q, K, V (for cross-attention support)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (B, L_q, D)
            key: (B, L_k, D)
            value: (B, L_v, D)  [L_k == L_v]
            mask: (B, 1, L_q, L_k) or (B, L_q, L_k)

        Returns:
            output: (B, L_q, D)
            attention: (B, H, L_q, L_k)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Project and reshape for multi-head
        Q = self.q_proj(query).reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = self.k_proj(key).reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
        V = self.v_proj(value).reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)  # (B, H, L_q, D_h)
        K = K.permute(0, 2, 1, 3)  # (B, H, L_k, D_h)
        V = V.permute(0, 2, 1, 3)  # (B, H, L_v, D_h)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L_q, L_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention, V)  # (B, H, L_q, D_h)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(output)

        return output, attention


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention
        attn_output, attention = self.self_attn(x, x, x, src_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x, attention


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer

    Architecture:
        Input
          ↓
        Masked Self-Attention (causal)
          ↓
        Add & Norm
          ↓
        Cross-Attention (attend to encoder output)
          ↓
        Add & Norm
          ↓
        Feed-Forward
          ↓
        Add & Norm
          ↓
        Output
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Masked self-attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Cross-attention
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Feed-forward
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x: Decoder input (B, L_tgt, D)
            encoder_output: Encoder output (B, L_src, D)
            tgt_mask: Causal mask for decoder self-attention
            src_mask: Padding mask for encoder outputs

        Returns:
            output: (B, L_tgt, D)
            self_attention: Self-attention weights
            cross_attention: Cross-attention weights
        """
        # Masked self-attention
        self_attn_output, self_attention = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(self_attn_output)
        x = self.norm1(x)

        # Cross-attention (query from decoder, key/value from encoder)
        cross_attn_output, cross_attention = self.cross_attn(
            x, encoder_output, encoder_output, src_mask
        )
        x = x + self.dropout2(cross_attn_output)
        x = self.norm2(x)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x, self_attention, cross_attention


class Transformer(nn.Module):
    """
    Complete Transformer Model (Encoder-Decoder)

    Use case: Machine translation, text summarization, etc.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Encoder
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.encoder_pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.decoder_pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(embed_dim, tgt_vocab_size)

        self.scale = math.sqrt(embed_dim)

    def encode(self, src, src_mask=None):
        """
        Encode source sequence

        Args:
            src: (B, L_src) - source token indices
            src_mask: (B, 1, L_src, L_src) - padding mask

        Returns:
            encoder_output: (B, L_src, D)
        """
        # Embed and add positional encoding
        x = self.encoder_embedding(src) * self.scale
        x = self.encoder_pos_encoding(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x, _ = layer(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """
        Decode target sequence

        Args:
            tgt: (B, L_tgt) - target token indices
            encoder_output: (B, L_src, D) - encoder output
            tgt_mask: (B, 1, L_tgt, L_tgt) - causal mask
            src_mask: (B, 1, L_src, L_src) - padding mask for encoder

        Returns:
            output: (B, L_tgt, vocab_size) - logits
        """
        # Embed and add positional encoding
        x = self.decoder_embedding(tgt) * self.scale
        x = self.decoder_pos_encoding(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x, _, _ = layer(x, encoder_output, tgt_mask, src_mask)

        # Project to vocabulary
        output = self.output_proj(x)

        return output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Full forward pass

        Args:
            src: (B, L_src)
            tgt: (B, L_tgt)
            src_mask: Padding mask for source
            tgt_mask: Causal mask for target

        Returns:
            output: (B, L_tgt, vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        return output


def create_causal_mask(size):
    """Create causal (autoregressive) mask"""
    mask = torch.tril(torch.ones(size, size))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def create_padding_mask(seq, pad_idx=0):
    """Create padding mask"""
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.float()


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    """
    Greedy decoding for inference

    Args:
        model: Transformer model
        src: (1, L_src) - source sequence
        src_mask: Source padding mask
        max_len: Maximum generation length
        start_symbol: Start token ID
        end_symbol: End token ID (EOS)

    Returns:
        Generated sequence
    """
    model.eval()

    # Encode source
    encoder_output = model.encode(src, src_mask)

    # Initialize decoder input with start symbol
    tgt = torch.full((1, 1), start_symbol, dtype=torch.long, device=src.device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            # Create causal mask
            tgt_mask = create_causal_mask(tgt.size(1)).to(tgt.device)

            # Decode
            output = model.decode(tgt, encoder_output, tgt_mask, src_mask)

            # Get next token (greedy)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append to target
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if EOS
            if next_token.item() == end_symbol:
                break

    return tgt


def main():
    print("=" * 60)
    print("Lab 3: Complete Transformer (Encoder-Decoder)")
    print("=" * 60)

    # Configuration
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.1

    # Example 1: Decoder Layer
    print("\n1. Transformer Decoder Layer")
    print("-" * 60)

    batch_size = 2
    src_len = 10
    tgt_len = 8

    decoder_input = torch.randn(batch_size, tgt_len, embed_dim)
    encoder_output = torch.randn(batch_size, src_len, embed_dim)

    decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)

    # Create causal mask for decoder
    tgt_mask = create_causal_mask(tgt_len)

    output, self_attn, cross_attn = decoder_layer(
        decoder_input, encoder_output, tgt_mask=tgt_mask
    )

    print(f"Decoder input: {decoder_input.shape}")
    print(f"Encoder output: {encoder_output.shape}")
    print(f"Decoder output: {output.shape}")
    print(f"Self-attention weights: {self_attn.shape}")
    print(f"Cross-attention weights: {cross_attn.shape}")

    # Example 2: Complete Transformer
    print("\n2. Complete Transformer Model")
    print("-" * 60)

    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout
    )

    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total parameters: {total_params:,}")

    # Example 3: Forward Pass
    print("\n3. Forward Pass (Training Mode)")
    print("-" * 60)

    # Sample input
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    print(f"Source: {src.shape}")
    print(f"Target: {tgt.shape}")

    # Create masks
    src_mask = create_padding_mask(src, pad_idx=0)
    tgt_mask = create_causal_mask(tgt_len)

    # Forward
    output = transformer(src, tgt, src_mask, tgt_mask)
    print(f"Output logits: {output.shape}")

    # Example 4: Greedy Decoding (Inference)
    print("\n4. Greedy Decoding (Inference Mode)")
    print("-" * 60)

    src_sentence = torch.randint(1, src_vocab_size, (1, 12))
    src_mask = create_padding_mask(src_sentence)

    start_symbol = 1  # <BOS>
    end_symbol = 2    # <EOS>
    max_len = 20

    generated = greedy_decode(
        transformer, src_sentence, src_mask, max_len, start_symbol, end_symbol
    )

    print(f"Source: {src_sentence.squeeze().tolist()}")
    print(f"Generated: {generated.squeeze().tolist()}")
    print(f"Generated length: {generated.size(1)}")

    # Example 5: Training Loop Skeleton
    print("\n5. Training Loop for Machine Translation")
    print("-" * 60)

    print("""
    # Training setup
    model = Transformer(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            src, tgt = batch  # (B, L_src), (B, L_tgt)

            # Prepare input and target
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]  # Remove first token (shifted)

            # Create masks
            src_mask = create_padding_mask(src, pad_idx)
            tgt_mask = create_causal_mask(tgt_input.size(1))

            # Forward
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Compute loss
            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    """)

    # Example 6: Attention Visualization
    print("\n6. Visualizing Cross-Attention")
    print("-" * 60)

    # Get cross-attention from last decoder layer
    model_eval = Transformer(
        src_vocab_size=100, tgt_vocab_size=100,
        embed_dim=64, num_heads=4, ff_dim=256,
        num_encoder_layers=2, num_decoder_layers=2
    )
    model_eval.eval()

    src_small = torch.randint(1, 100, (1, 6))
    tgt_small = torch.randint(1, 100, (1, 5))
    tgt_mask_small = create_causal_mask(5)

    encoder_out = model_eval.encode(src_small)

    with torch.no_grad():
        x = model_eval.decoder_embedding(tgt_small) * model_eval.scale
        x = model_eval.decoder_pos_encoding(x)

        for layer in model_eval.decoder_layers:
            x, self_attn, cross_attn = layer(x, encoder_out, tgt_mask_small)

    print(f"Cross-attention shape: {cross_attn.shape}")
    print("Cross-attention shows which source words the decoder attends to")
    print("when generating each target word")

    print("\n" + "=" * 60)
    print("Lab 3 Complete!")
    print("=" * 60)
    print("\nKey Concepts Learned:")
    print("1. Decoder Layer: Masked self-attention + cross-attention + FFN")
    print("2. Cross-Attention: Query from decoder, Key/Value from encoder")
    print("3. Causal Masking: Prevent decoder from seeing future tokens")
    print("4. Encoder-Decoder Architecture: Full seq2seq Transformer")
    print("5. Greedy Decoding: Simple inference strategy")
    print("6. Training Setup: Teacher forcing with shifted targets")
    print("\nYou've built a complete Transformer from scratch!")
    print("This is the architecture behind GPT, BERT, and modern LLMs!")


if __name__ == "__main__":
    main()
