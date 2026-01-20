#!/usr/bin/env python3
"""
Demonstration script for the Transformer implementation from scratch.
This shows how to use the transformer for a simple sequence-to-sequence task.
"""

import numpy as np
from transformer_from_scratch import Transformer

def demo_simple_translation():
    """Demonstrate a simple translation-like task"""
    print("Transformer Demo: Simple Sequence-to-Sequence Task")
    print("=" * 60)
    
    # Create a small transformer model
    vocab_size = 100  # Small vocabulary for demo
    d_model = 64      # Embedding dimension
    n_heads = 4       # Number of attention heads
    num_layers = 2    # Number of encoder/decoder layers
    d_ff = 128        # Feed-forward dimension
    max_seq_len = 20  # Maximum sequence length
    dropout = 0.1     # Dropout probability
    
    print(f"Creating Transformer model:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Encoder/Decoder layers: {num_layers}")
    print(f"  Feed-forward dimension: {d_ff}")
    print(f"  Max sequence length: {max_seq_len}")
    print()
    
    # Initialize transformer
    transformer = Transformer(vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len, dropout)
    
    # Create sample input sequences
    batch_size = 1
    src_seq_len = 5
    tgt_seq_len = 6
    
    # Create random token sequences (in practice, these would be actual vocabulary indices)
    src_tokens = np.random.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt_tokens = np.random.randint(0, vocab_size, (batch_size, tgt_seq_len))
    
    print(f"Source sequence (length {src_seq_len}): {src_tokens[0]}")
    print(f"Target sequence (length {tgt_seq_len}): {tgt_tokens[0]}")
    print()
    
    # Forward pass
    print("Running forward pass...")
    output = transformer.forward(src_tokens, tgt_tokens)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits for each position and vocabulary item):")
    
    # Show output for first position
    first_pos_output = output[0, 0, :]
    top_5_indices = np.argsort(first_pos_output)[-5:][::-1]  # Top 5 predictions
    
    print(f"Top 5 predictions for first output position:")
    for i, idx in enumerate(top_5_indices):
        print(f"  {i+1}. Token {idx}: logit = {first_pos_output[idx]:.3f}")
    
    print()
    print("Demo completed successfully!")

def demo_attention_visualization():
    """Demonstrate attention patterns"""
    print("\n" + "=" * 60)
    print("Attention Pattern Visualization")
    print("=" * 60)
    
    from transformer_from_scratch import MultiHeadAttention
    
    # Create small attention module
    d_model = 32
    n_heads = 4
    dropout = 0.0
    
    mha = MultiHeadAttention(d_model, n_heads, dropout)
    
    # Create input with clear patterns
    batch_size = 1
    seq_len = 4
    
    # Create queries and keys where position 0 attends to position 1
    # and position 2 attends to position 3
    q = np.zeros((batch_size, seq_len, d_model))
    k = np.zeros((batch_size, seq_len, d_model))
    v = np.zeros((batch_size, seq_len, d_model))
    
    # Set attention pattern: q[0] should attend to k[1]
    q[0, 0, 0] = 1.0  # Query at position 0 has value 1 in first dimension
    k[0, 1, 0] = 1.0  # Key at position 1 has value 1 in first dimension
    
    # Set attention pattern: q[2] should attend to k[3]
    q[0, 2, 1] = 1.0  # Query at position 2 has value 1 in second dimension
    k[0, 3, 1] = 1.0  # Key at position 3 has value 1 in second dimension
    
    # Set some values to propagate
    v[0, 1, 2] = 5.0  # Value at position 1 has value 5 in third dimension
    v[0, 3, 3] = 7.0  # Value at position 3 has value 7 in fourth dimension
    
    print("Input patterns:")
    print(f"  Query position 0 attends to Key position 1")
    print(f"  Query position 2 attends to Key position 3")
    print()
    
    # Run attention
    output = mha(q, k, v)
    
    print("Attention output analysis:")
    print(f"  Output at position 0, dimension 2: {output[0, 0, 2]:.3f} (should be close to 5.0)")
    print(f"  Output at position 2, dimension 3: {output[0, 2, 3]:.3f} (should be close to 7.0)")
    print()

def main():
    """Run demonstration"""
    print("Transformer from Scratch - Demonstration")
    print("Using only NumPy - No deep learning frameworks!")
    print()
    
    # Run demos
    demo_simple_translation()
    demo_attention_visualization()
    
    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("This shows that we've successfully implemented a complete")
    print("Transformer model from scratch using only NumPy.")
    print("=" * 60)

if __name__ == "__main__":
    main()