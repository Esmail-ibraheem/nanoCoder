"""
Transformer model implementation from scratch using only NumPy.
This implementation follows the original "Attention is All You Need" paper.
"""

import numpy as np
import math

class Transformer:
    """Complete Transformer model implementation"""
    
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        """
        Initialize the Transformer model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            num_layers: Number of encoder/decoder layers
            d_ff: Dimension of feed-forward network
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Initialize components
        self.embedding = self._init_embedding()
        self.positional_encoding = self._init_positional_encoding()
        
        # Create encoder and decoder layers
        self.encoder_layers = [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        
        # Final linear layer
        self.fc_out = self._init_linear(d_model, vocab_size)
        
    def _init_embedding(self):
        """Initialize embedding layer"""
        return np.random.randn(self.vocab_size, self.d_model) * 0.02
    
    def _init_positional_encoding(self):
        """Initialize positional encoding"""
        return PositionalEncoding(self.d_model, self.max_seq_len)
    
    def _init_linear(self, in_features, out_features):
        """Initialize linear layer weights"""
        return np.random.randn(in_features, out_features) * 0.02
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of the Transformer model.
        
        Args:
            src: Source input tensor (batch_size, src_seq_len)
            tgt: Target input tensor (batch_size, tgt_seq_len)
            src_mask: Source mask tensor
            tgt_mask: Target mask tensor
        
        Returns:
            Output tensor (batch_size, tgt_seq_len, vocab_size)
        """
        # Embedding and positional encoding
        src_embedded = self._embed_and_encode(src)
        tgt_embedded = self._embed_and_encode(tgt)
        
        # Encoder
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Final linear layer
        output = self._linear(dec_output, self.fc_out)
        
        return output
    
    def _embed_and_encode(self, x):
        """Embed input and add positional encoding"""
        # Embedding lookup
        embedded = self.embedding[x]  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        pos_encoded = self.positional_encoding(embedded)
        
        return pos_encoded
    
    def _linear(self, x, weight):
        """Linear transformation"""
        return np.dot(x, weight)

class PositionalEncoding:
    """Positional Encoding module"""
    
    def __init__(self, d_model, max_seq_len):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of model embeddings
            max_seq_len: Maximum sequence length
        """
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create positional encoding matrix
        self.pe = self._create_positional_encoding()
    
    def _create_positional_encoding(self):
        """Create positional encoding matrix"""
        pe = np.zeros((self.max_seq_len, self.d_model))
        
        position = np.arange(0, self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def __call__(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor with positional encoding added
        """
        batch_size, seq_len, _ = x.shape
        
        # Add positional encoding (broadcasting)
        return x + self.pe[:seq_len]

class MultiHeadAttention:
    """Multi-Head Attention module"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Ensure d_model is divisible by n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_k = d_model // n_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def __call__(self, q, k, v, mask=None):
        """
        Multi-head attention forward pass.
        
        Args:
            q: Query tensor (batch_size, seq_len, d_model)
            k: Key tensor (batch_size, seq_len, d_model)
            v: Value tensor (batch_size, seq_len, d_model)
            mask: Mask tensor for attention
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size = q.shape[0]
        
        # Linear projections
        Q = self._linear(q, self.W_q)  # (batch_size, seq_len, d_model)
        K = self._linear(k, self.W_k)  # (batch_size, seq_len, d_model)
        V = self._linear(v, self.W_v)  # (batch_size, seq_len, d_model)
        
        # Split into multiple heads
        Q = self._split_heads(Q)  # (batch_size, n_heads, seq_len, d_k)
        K = self._split_heads(K)  # (batch_size, n_heads, seq_len, d_k)
        V = self._split_heads(V)  # (batch_size, n_heads, seq_len, d_k)
        
        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        concat_output = self._concatenate_heads(attention_output)  # (batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self._linear(concat_output, self.W_o)
        
        return output
    
    def _linear(self, x, weight):
        """Linear transformation"""
        return np.dot(x, weight)
    
    def _split_heads(self, x):
        """Split tensor into multiple attention heads"""
        batch_size, seq_len, _ = x.shape
        
        # Reshape to (batch_size, seq_len, n_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose to (batch_size, n_heads, seq_len, d_k)
        return np.transpose(x, (0, 2, 1, 3))
    
    def _concatenate_heads(self, x):
        """Concatenate multiple attention heads"""
        batch_size, _, seq_len, _ = x.shape
        
        # Transpose back to (batch_size, seq_len, n_heads, d_k)
        x = np.transpose(x, (0, 2, 1, 3))
        
        # Reshape to (batch_size, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def _scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention"""
        # Matmul Q and K transpose
        scores = np.matmul(Q, np.transpose(K, (0, 1, 3, 2)))  # (batch_size, n_heads, seq_len, seq_len)
        
        # Scale by sqrt(d_k)
        scores = scores / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply dropout (simplified - in practice would randomly zero some elements)
        if self.dropout > 0:
            attention_weights = attention_weights * (1 - self.dropout)
        
        # Multiply by V
        output = np.matmul(attention_weights, V)
        
        return output
    
    def _softmax(self, x, axis=-1):
        """Softmax function"""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

class FeedForward:
    """Feed-forward network module"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Dimension of model embeddings
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Initialize weight matrices
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def __call__(self, x):
        """
        Feed-forward network forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # First linear layer with ReLU activation
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)
        
        # Apply dropout (simplified)
        if self.dropout > 0:
            h = h * (1 - self.dropout)
        
        # Second linear layer
        output = np.dot(h, self.W2) + self.b2
        
        return output

class LayerNorm:
    """Layer normalization module"""
    
    def __init__(self, d_model, eps=1e-6):
        """
        Initialize layer normalization.
        
        Args:
            d_model: Dimension of model embeddings
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Initialize learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def __call__(self, x):
        """
        Layer normalization forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            Normalized tensor (batch_size, seq_len, d_model)
        """
        # Calculate mean and variance
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        return output

class EncoderLayer:
    """Encoder layer module"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Initialize components
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def __call__(self, x, mask=None):
        """
        Encoder layer forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Mask tensor for attention
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = x + attn_output
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x

class DecoderLayer:
    """Decoder layer module"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Initialize decoder layer.
        
        Args:
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Initialize components
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def __call__(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Decoder layer forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            enc_output: Encoder output tensor (batch_size, seq_len, d_model)
            src_mask: Source mask tensor
            tgt_mask: Target mask tensor
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = x + attn_output
        x = self.norm1(x)
        
        # Cross-attention with residual connection
        attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = x + attn_output
        x = self.norm2(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm3(x)
        
        return x

# Mark first task as completed
todos_action = {"action": "complete", "id": "bafa5226"}
