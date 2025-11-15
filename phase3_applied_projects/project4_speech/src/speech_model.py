"""
Phase 3 Project 4: Speech Recognition
Implement end-to-end speech recognition with CTC loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Conv1DBlock(nn.Module):
    """1D Convolutional block for audio features."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AudioFeatureExtractor(nn.Module):
    """
    Extract features from raw audio using Conv1D layers.
    """
    def __init__(self, input_dim: int = 80, hidden_dims: list = [256, 256, 256]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Stride 2 for first layer to downsample
            stride = 2 if i == 0 else 1
            layers.append(Conv1DBlock(prev_dim, hidden_dim, kernel_size=3, stride=stride))
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim, time)
        Returns:
            features: (batch, output_dim, time')
        """
        return self.layers(x)


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for sequential modeling.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_dim = hidden_dim * 2  # Bidirectional

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, time, features)
            lengths: (batch,) sequence lengths
        Returns:
            output: (batch, time, hidden_dim * 2)
        """
        if lengths is not None:
            # Pack padded sequence
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, _ = self.lstm(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        else:
            output, _ = self.lstm(x)

        return output


class CTCSpeechRecognizer(nn.Module):
    """
    End-to-end speech recognition model with CTC loss.

    Architecture:
    - Conv feature extraction
    - BiLSTM encoder
    - Linear projection to vocabulary
    - CTC decoding
    """
    def __init__(
        self,
        input_dim: int = 80,  # Mel filterbank features
        vocab_size: int = 29,  # 26 letters + blank + space + apostrophe
        conv_dims: list = [256, 256],
        lstm_hidden: int = 512,
        lstm_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size

        # Feature extraction
        self.feature_extractor = AudioFeatureExtractor(input_dim, conv_dims)

        # BiLSTM encoder
        self.encoder = BiLSTMEncoder(
            self.feature_extractor.output_dim,
            lstm_hidden,
            lstm_layers,
            dropout
        )

        # Output projection
        self.projection = nn.Linear(self.encoder.output_dim, vocab_size)

    def forward(self, features, lengths=None):
        """
        Args:
            features: (batch, input_dim, time) mel features
            lengths: (batch,) sequence lengths
        Returns:
            logits: (batch, time', vocab_size) log probabilities for CTC
            output_lengths: (batch,) output sequence lengths
        """
        # Extract features
        x = self.feature_extractor(features)  # (batch, hidden, time')

        # Transpose for LSTM: (batch, time', hidden)
        x = x.transpose(1, 2)

        # Update lengths after conv downsampling
        if lengths is not None:
            output_lengths = (lengths / 2).long()  # Stride 2 in first conv
        else:
            output_lengths = None

        # BiLSTM encoding
        x = self.encoder(x, output_lengths)

        # Project to vocabulary
        logits = self.projection(x)

        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=2)

        return log_probs, output_lengths

    def greedy_decode(self, log_probs, lengths=None):
        """
        Greedy CTC decoding.

        Args:
            log_probs: (batch, time, vocab)
            lengths: (batch,)
        Returns:
            decoded: List of decoded sequences
        """
        # Get argmax at each timestep
        _, predictions = log_probs.max(dim=2)  # (batch, time)

        decoded = []
        blank_idx = self.vocab_size - 1  # Assuming blank is last token

        for i in range(predictions.size(0)):
            pred = predictions[i]

            if lengths is not None:
                pred = pred[:lengths[i]]

            # Remove consecutive duplicates and blanks
            result = []
            prev = None

            for token in pred:
                token = token.item()
                if token != blank_idx and token != prev:
                    result.append(token)
                prev = token

            decoded.append(result)

        return decoded


class TransformerASR(nn.Module):
    """
    Transformer-based ASR model (simplified).
    """
    def __init__(
        self,
        input_dim: int = 80,
        vocab_size: int = 29,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, features, lengths=None):
        """
        Args:
            features: (batch, input_dim, time)
            lengths: (batch,)
        Returns:
            log_probs: (batch, time, vocab)
        """
        # Transpose: (batch, time, input_dim)
        x = features.transpose(1, 2)

        # Project to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create padding mask
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        else:
            mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Project to vocabulary
        logits = self.output_proj(x)

        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=2)

        return log_probs, lengths


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
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


def create_synthetic_audio_data(batch_size: int, input_dim: int = 80, max_len: int = 400):
    """Create synthetic audio features and transcripts."""
    # Audio features (mel spectrograms)
    lengths = torch.randint(100, max_len, (batch_size,))
    max_input_len = lengths.max().item()

    features = torch.randn(batch_size, input_dim, max_input_len)

    # Transcripts (character sequences)
    max_target_len = 50
    target_lengths = torch.randint(10, max_target_len, (batch_size,))
    targets = torch.randint(0, 28, (batch_size, max_target_len))  # 28 = vocab_size - 1 (no blank)

    return features, lengths, targets, target_lengths


def train_ctc_asr():
    """Train CTC-based ASR model."""
    print("=" * 70)
    print("Training CTC Speech Recognition Model")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = CTCSpeechRecognizer(
        input_dim=80,
        vocab_size=29,
        conv_dims=[128, 128],
        lstm_hidden=256,
        lstm_layers=2
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # CTC loss
    ctc_loss = nn.CTCLoss(blank=28, zero_infinity=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    batch_size = 4

    for epoch in range(num_epochs):
        model.train()

        # Generate batch
        features, input_lengths, targets, target_lengths = create_synthetic_audio_data(batch_size)
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        log_probs, output_lengths = model(features, input_lengths.to(device))

        # Reshape for CTC loss
        # CTC expects (time, batch, vocab)
        log_probs = log_probs.transpose(0, 1)

        # Remove padding from targets
        targets_concat = []
        for i in range(batch_size):
            targets_concat.append(targets[i, :target_lengths[i]])
        targets_concat = torch.cat(targets_concat)

        # Compute CTC loss
        loss = ctc_loss(
            log_probs,
            targets_concat,
            output_lengths,
            target_lengths
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")

    print("\n" + "=" * 70)
    return model


def demo_decoding(model):
    """Demonstrate CTC decoding."""
    print("\n" + "=" * 70)
    print("CTC Decoding Demonstration")
    print("=" * 70)

    device = next(model.parameters()).device
    model.eval()

    # Create sample
    features = torch.randn(1, 80, 200).to(device)
    lengths = torch.tensor([200]).to(device)

    with torch.no_grad():
        log_probs, output_lengths = model(features, lengths)

        # Greedy decode
        decoded = model.greedy_decode(log_probs, output_lengths)

        print("Greedy decoded sequence:")
        print(f"  Token IDs: {decoded[0][:20]}...")  # Show first 20 tokens
        print(f"  Length: {len(decoded[0])}")

    print("\n" + "=" * 70)


def demo_model_comparison():
    """Compare model architectures."""
    print("\n" + "=" * 70)
    print("Model Architecture Comparison")
    print("=" * 70)

    # CTC-LSTM
    ctc_model = CTCSpeechRecognizer(
        input_dim=80,
        vocab_size=29,
        lstm_hidden=256,
        lstm_layers=3
    )

    # Transformer
    transformer_model = TransformerASR(
        input_dim=80,
        vocab_size=29,
        d_model=256,
        num_layers=6
    )

    ctc_params = sum(p.numel() for p in ctc_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())

    print(f"CTC-LSTM Model: {ctc_params:,} parameters")
    print(f"Transformer Model: {transformer_params:,} parameters")
    print()
    print("CTC-LSTM:")
    print("  + Proven architecture for ASR")
    print("  + Handles variable-length sequences well")
    print("  - Sequential processing")
    print()
    print("Transformer:")
    print("  + Parallelizable training")
    print("  + Better long-range dependencies")
    print("  - More parameters, needs more data")

    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = train_ctc_asr()
    demo_decoding(model)
    demo_model_comparison()

    print("\n✓ Speech recognition demonstration complete!")
