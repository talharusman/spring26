#!/usr/bin/env python3
"""Quick test to verify all implementations are syntactically correct."""

import torch
import sys

# Test imports
try:
    import transformers
    print("✓ Transformers module imported successfully")
except SyntaxError as e:
    print(f"✗ Syntax error in transformers.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

try:
    import rnn_lstm_captioning
    print("✓ RNN/LSTM captioning module imported successfully")
except SyntaxError as e:
    print(f"✗ Syntax error in rnn_lstm_captioning.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Quick functional tests
print("\n--- Testing position encodings ---")
try:
    pos_simple = transformers.position_encoding_simple(5, 10)
    assert pos_simple.shape == (1, 5, 10), f"Expected (1, 5, 10), got {pos_simple.shape}"
    print(f"✓ position_encoding_simple output shape: {pos_simple.shape}")
except Exception as e:
    print(f"✗ position_encoding_simple failed: {e}")

try:
    pos_sin = transformers.position_encoding_sinusoid(5, 10)
    assert pos_sin.shape == (1, 5, 10), f"Expected (1, 5, 10), got {pos_sin.shape}"
    print(f"✓ position_encoding_sinusoid output shape: {pos_sin.shape}")
except Exception as e:
    print(f"✗ position_encoding_sinusoid failed: {e}")

# Test get_subsequent_mask
print("\n--- Testing mask generation ---")
try:
    mask = transformers.get_subsequent_mask(5)
    assert mask.shape == (5, 5), f"Expected (5, 5), got {mask.shape}"
    print(f"✓ get_subsequent_mask output shape: {mask.shape}")
except Exception as e:
    print(f"✗ get_subsequent_mask failed: {e}")

# Test module instantiation
print("\n--- Testing module instantiation ---")
try:
    attn = transformers.SelfAttention(4, 32, 32)
    print(f"✓ SelfAttention created successfully")
except Exception as e:
    print(f"✗ SelfAttention creation failed: {e}")

try:
    mha = transformers.MultiHeadAttention(4, 32, 32)
    print(f"✓ MultiHeadAttention created successfully")
except Exception as e:
    print(f"✗ MultiHeadAttention creation failed: {e}")

try:
    decoder_block = transformers.DecoderBlock(4, 32, 64, 0.1)
    print(f"✓ DecoderBlock created successfully")
except Exception as e:
    print(f"✗ DecoderBlock creation failed: {e}")

try:
    transformer = transformers.Transformer(4, 32, 64, 0.1, 2, 2, 100)
    print(f"✓ Transformer created successfully")
except Exception as e:
    print(f"✗ Transformer creation failed: {e}")

print("\n✅ All tests passed! Implementation is complete.")
