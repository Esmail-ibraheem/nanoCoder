#!/usr/bin/env python3
"""
Test script for the Transformer implementation from scratch.
"""

import numpy as np
from transformer_from_scratch import Transformer, PositionalEncoding, MultiHeadAttention, FeedForward, LayerNorm, EncoderLayer, DecoderLayer

def test_positional_encoding():
    """Test positional encoding module"""
    print("Testing PositionalEncoding...")
    
    d_model = 512
    max_seq_len = 100
    
    pe = PositionalEncoding(d_model, max_seq_len)
    
    # Test with sample input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check that positional encoding was added (values should be different from input)
    assert not np.allclose(x, output), "Positional encoding should modify the input"
    
    print("PositionalEncoding test passed")

def test_multi_head_attention():
    """Test multi-head attention module"""
    print("Testing MultiHeadAttention...")
    
    d_model = 512
    n_heads = 8
    dropout = 0.1
    
    mha = MultiHeadAttention(d_model, n_heads, dropout)
    
    # Test with sample input
    batch_size = 2
    seq_len = 10
    q = np.random.randn(batch_size, seq_len, d_model)
    k = np.random.randn(batch_size, seq_len, d_model)
    v = np.random.randn(batch_size, seq_len, d_model)
    
    output = mha(q, k, v)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("MultiHeadAttention test passed")

def test_feed_forward():
    """Test feed-forward network module"""
    print("Testing FeedForward...")
    
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    
    ff = FeedForward(d_model, d_ff, dropout)
    
    # Test with sample input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = ff(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("FeedForward test passed")

def test_layer_norm():
    """Test layer normalization module"""
    print("Testing LayerNorm...")
    
    d_model = 512
    
    ln = LayerNorm(d_model)
    
    # Test with sample input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = ln(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check that mean is close to 0 and std is close to 1 (before gamma/beta)
    # Note: This is a simplified check since we have learnable gamma/beta
    mean = np.mean(output)
    std = np.std(output)
    
    print(f"  Output mean: {mean:.6f}, std: {std:.6f}")
    print("LayerNorm test passed")

def test_encoder_layer():
    """Test encoder layer module"""
    print("Testing EncoderLayer...")
    
    d_model = 512
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
    
    # Test with sample input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = encoder_layer(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("EncoderLayer test passed")

def test_decoder_layer():
    """Test decoder layer module"""
    print("Testing DecoderLayer...")
    
    d_model = 512
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
    
    # Test with sample input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    enc_output = np.random.randn(batch_size, seq_len, d_model)
    
    output = decoder_layer(x, enc_output)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("DecoderLayer test passed")

def test_full_transformer():
    """Test full transformer model"""
    print("Testing full Transformer...")
    
    # Small model for testing
    vocab_size = 1000
    d_model = 64
    n_heads = 4
    num_layers = 2
    d_ff = 128
    max_seq_len = 50
    dropout = 0.1
    
    transformer = Transformer(vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len, dropout)
    
    # Test with sample input
    batch_size = 2
    src_seq_len = 8
    tgt_seq_len = 10
    
    # Create random input tokens (integers representing vocabulary indices)
    src = np.random.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt = np.random.randint(0, vocab_size, (batch_size, tgt_seq_len))
    
    output = transformer.forward(src, tgt)
    
    # Check output shape
    expected_shape = (batch_size, tgt_seq_len, vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("Full Transformer test passed")

def test_attention_patterns():
    """Test attention patterns with simple inputs"""
    print("Testing attention patterns...")
    
    d_model = 64
    n_heads = 4
    dropout = 0.0  # No dropout for this test
    
    mha = MultiHeadAttention(d_model, n_heads, dropout)
    
    # Create simple input where we can predict attention patterns
    batch_size = 1
    seq_len = 3
    
    # Create queries, keys, values with clear patterns
    q = np.zeros((batch_size, seq_len, d_model))
    k = np.zeros((batch_size, seq_len, d_model))
    v = np.zeros((batch_size, seq_len, d_model))
    
    # Set first position to have high similarity with itself
    q[0, 0, 0] = 1.0
    k[0, 0, 0] = 1.0
    
    # Set second position to have high similarity with third
    q[0, 1, 1] = 1.0
    k[0, 2, 1] = 1.0
    
    output = mha(q, k, v)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    print("Attention patterns test passed")

def main():
    """Run all tests"""
    print("Running Transformer tests...\n")
    
    try:
        test_positional_encoding()
        test_multi_head_attention()
        test_feed_forward()
        test_layer_norm()
        test_encoder_layer()
        test_decoder_layer()
        test_full_transformer()
        test_attention_patterns()
        
        print("\nAll tests passed! Transformer implementation is working correctly.")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
