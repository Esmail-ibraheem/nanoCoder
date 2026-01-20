# Transformer Model from Scratch using NumPy

This project implements a complete Transformer model from scratch using only NumPy, following the original "Attention is All You Need" paper by Vaswani et al.

## Implementation Details

### Core Components Implemented

1. **Multi-Head Attention** (`MultiHeadAttention`)
   - Scaled dot-product attention
   - Query, Key, Value projections
   - Multiple attention heads with concatenation
   - Optional attention masking

2. **Positional Encoding** (`PositionalEncoding`)
   - Sinusoidal positional encodings
   - Different frequencies for different dimensions
   - Added to input embeddings

3. **Feed-Forward Network** (`FeedForward`)
   - Two linear transformations with ReLU activation
   - Position-wise (applied to each position separately)
   - Residual connection support

4. **Layer Normalization** (`LayerNorm`)
   - Learnable gamma and beta parameters
   - Applied across features (last dimension)
   - Numerical stability with epsilon

5. **Encoder Layer** (`EncoderLayer`)
   - Self-attention sub-layer
   - Feed-forward sub-layer
   - Residual connections and layer normalization

6. **Decoder Layer** (`DecoderLayer`)
   - Self-attention sub-layer (with masking)
   - Cross-attention sub-layer
   - Feed-forward sub-layer
   - Residual connections and layer normalization

7. **Complete Transformer** (`Transformer`)
   - Input embedding layer
   - Positional encoding
   - Stack of encoder layers
   - Stack of decoder layers
   - Final linear projection to vocabulary

### Key Features

- **Pure NumPy Implementation**: No deep learning frameworks used
- **Modular Design**: Each component is implemented as a separate class
- **Configurable Architecture**: Adjustable hyperparameters
- **Complete Forward Pass**: Full sequence-to-sequence transformation
- **Attention Mechanisms**: Both self-attention and cross-attention

### Hyperparameters

- `vocab_size`: Size of the vocabulary
- `d_model`: Dimension of model embeddings (typically 512)
- `n_heads`: Number of attention heads (typically 8)
- `num_layers`: Number of encoder/decoder layers (typically 6)
- `d_ff`: Dimension of feed-forward network (typically 2048)
- `max_seq_len`: Maximum sequence length for positional encoding
- `dropout`: Dropout probability (simplified implementation)

## Usage Example

```python
import numpy as np
from transformer_from_scratch import Transformer

# Create transformer model
vocab_size = 1000
d_model = 64
n_heads = 4
num_layers = 2
d_ff = 128
max_seq_len = 50
dropout = 0.1

transformer = Transformer(vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len, dropout)

# Create input sequences (batch_size, seq_len)
batch_size = 2
src_seq_len = 8
tgt_seq_len = 10

src = np.random.randint(0, vocab_size, (batch_size, src_seq_len))
tgt = np.random.randint(0, vocab_size, (batch_size, tgt_seq_len))

# Forward pass
output = transformer.forward(src, tgt)
# output shape: (batch_size, tgt_seq_len, vocab_size)
```

## Files

- `transformer_from_scratch.py`: Main implementation
- `test_transformer.py`: Comprehensive test suite
- `demo_transformer.py`: Demonstration scripts
- `TRANSFORMER_SUMMARY.md`: This documentation

## Testing

Run the test suite to verify the implementation:

```bash
python test_transformer.py
```

Run the demonstration:

```bash
python demo_transformer.py
```

## Limitations

This is a pure NumPy implementation for educational purposes. For production use, consider:

1. **Performance**: NumPy is much slower than optimized deep learning frameworks
2. **Training**: No training loop or optimization implemented
3. **Dropout**: Simplified dropout implementation
4. **Masking**: Basic masking support
5. **Memory**: No memory optimization for long sequences

## Educational Value

This implementation demonstrates:

- How attention mechanisms work at a fundamental level
- The complete transformer architecture without framework abstractions
- Matrix operations that power modern deep learning
- Implementation of key components like layer normalization
- The flow of data through encoder and decoder stacks

## References

- Original Paper: "Attention is All You Need" by Vaswani et al. (2017)
- Implementation follows the architectural details described in the paper
- All components implemented from first principles using NumPy operations

This implementation serves as an excellent educational resource for understanding the inner workings of transformer models that power modern NLP applications.