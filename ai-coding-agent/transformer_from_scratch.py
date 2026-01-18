"""
Transformer model implementation from scratch using PyTorch.
Based on "Attention Is All You Need" by Vaswani et al. (2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module that injects information about the relative
    or absolute position of the tokens in the sequence.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Final linear layer
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calculate the attention weights
        Q: [batch_size, num_heads, seq_len, d_k]
        K: [batch_size, num_heads, seq_len, d_k]
        V: [batch_size, num_heads, seq_len, d_k]
        """
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for decoder)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k)
        x: [batch_size, seq_len, d_model]
        Returns: [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine heads back to original shape
        x: [batch_size, num_heads, seq_len, d_k]
        Returns: [batch_size, seq_len, d_model]
        """
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for multi-head attention
        Q: [batch_size, seq_len, d_model]
        K: [batch_size, seq_len, d_model]
        V: [batch_size, seq_len, d_model]
        mask: [batch_size, 1, 1, seq_len] (optional)
        """
        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads
        output = self.combine_heads(attn_output)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    """
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        mask: [batch_size, 1, 1, seq_len] (optional)
        """
        # Self attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        x: [batch_size, tgt_seq_len, d_model]
        enc_output: [batch_size, src_seq_len, d_model]
        src_mask: [batch_size, 1, 1, src_seq_len] (optional)
        tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len] (optional)
        """
        # Self attention (masked)
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Cross attention
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x


class Transformer(nn.Module):
    """
    Complete Transformer model
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, max_seq_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder and Decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Final linear layer
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize parameters with Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None):
        """
        Encoder forward pass
        src: [batch_size, src_seq_len]
        src_mask: [batch_size, 1, 1, src_seq_len] (optional)
        """
        # Embedding and positional encoding
        src = self.src_embedding(src)
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        return src
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """
        Decoder forward pass
        tgt: [batch_size, tgt_seq_len]
        enc_output: [batch_size, src_seq_len, d_model]
        src_mask: [batch_size, 1, 1, src_seq_len] (optional)
        tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len] (optional)
        """
        # Embedding and positional encoding
        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)
        
        return tgt
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Full Transformer forward pass
        src: [batch_size, src_seq_len]
        tgt: [batch_size, tgt_seq_len]
        src_mask: [batch_size, 1, 1, src_seq_len] (optional)
        tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len] (optional)
        """
        # Encode source
        enc_output = self.encode(src, src_mask)
        
        # Decode target
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # Final projection
        output = self.fc(dec_output)
        
        return output


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask for sequences
    seq: [batch_size, seq_len]
    Returns: [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder self-attention
    size: sequence length
    Returns: [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def create_masks(src, tgt, pad_idx=0):
    """
    Create all masks needed for Transformer
    src: [batch_size, src_seq_len]
    tgt: [batch_size, tgt_seq_len]
    pad_idx: padding token index
    Returns: src_mask, tgt_mask
    """
    # Source padding mask
    src_mask = create_padding_mask(src, pad_idx)
    
    # Target padding mask
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)
    
    # Target look-ahead mask
    tgt_seq_len = tgt.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_seq_len)
    
    # Combine target masks
    tgt_mask = tgt_pad_mask & tgt_look_ahead_mask
    
    return src_mask, tgt_mask


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    batch_size = 32
    src_seq_len = 50
    tgt_seq_len = 40
    
    # Create model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                       num_layers, d_ff, dropout=dropout)
    model = model.to(device)
    
    # Create dummy data
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)
    
    # Create masks
    src_mask, tgt_mask = create_masks(src, tgt)
    
    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")