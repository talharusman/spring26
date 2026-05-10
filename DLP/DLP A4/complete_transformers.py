#!/usr/bin/env python3
import re

# Read the transformers.py file
with open('transformers.py', 'r') as f:
    lines = f.readlines()

# Find and implement remaining functions
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # DecoderBlock forward pass
    if 'y = None' in line and i < len(lines) - 1 and 'TODO: Using the layers initialized' in lines[i+2]:
        # Skip until we find pass
        new_lines.append(line)
        i += 1
        while i < len(lines) and 'pass' not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        
        # Replace pass with implementation
        decoder_impl = '''        self_attn_out = self.attention_self(dec_inp, dec_inp, dec_inp, mask)
        self_attn_out = self.dropout(self_attn_out)
        out1 = self.norm1(dec_inp + self_attn_out)
        cross_attn_out = self.attention_cross(out1, enc_inp, enc_inp)
        cross_attn_out = self.dropout(cross_attn_out)
        out2 = self.norm2(out1 + cross_attn_out)
        ff_out = self.feed_forward(out2)
        ff_out = self.dropout(ff_out)
        y = self.norm3(out2 + ff_out)
'''
        new_lines.append(decoder_impl)
        i += 1
    
    # Position encoding simple
    elif 'def position_encoding_simple' in line:
        new_lines.append(line)
        i += 1
        while i < len(lines) and 'y = None' not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        new_lines.append(lines[i])  # y = None
        i += 1
        while i < len(lines) and 'pass' not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        pos_impl = '''    pos = torch.arange(K, dtype=torch.float32) / K
    y = pos.unsqueeze(1).repeat(1, M).unsqueeze(0)
'''
        new_lines.append(pos_impl)
        i += 1
    
    # Position encoding sinusoid
    elif 'def position_encoding_sinusoid' in line:
        new_lines.append(line)
        i += 1
        while i < len(lines) and 'y = None' not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        new_lines.append(lines[i])  # y = None
        i += 1
        while i < len(lines) and 'pass' not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        pos_sin_impl = '''    d_model = 10000
    positions = torch.arange(K, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(0, M, 2, dtype=torch.float32)
    angles = positions / (d_model ** (dimensions / M))
    y = torch.zeros(K, M)
    y[:, 0::2] = torch.sin(angles)
    if M % 2 != 0:
        y[:, 1::2] = torch.cos(angles[:, :-1])
    else:
        y[:, 1::2] = torch.cos(angles)
    y = y.unsqueeze(0)
'''
        new_lines.append(pos_sin_impl)
        i += 1
    
    else:
        new_lines.append(line)
        i += 1

# Write the file back
with open('transformers.py', 'w') as f:
    f.writelines(new_lines)

print("Transformers.py updated successfully")
