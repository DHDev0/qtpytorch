# mxtorch

**A comprehensive sub-byte quantization library for PyTorch with Triton GPU kernels**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [MX Data Types](#mx-data-types)
  - [Block-wise Quantization](#block-wise-quantization)
  - [Real Bit Packing](#real-bit-packing)
- [API Reference](#api-reference)
  - [Configuration](#configuration)
  - [Type System](#type-system)
  - [mx_tensor: The Quantized Tensor](#mx_tensor-the-quantized-tensor)
  - [Quantization Methods](#quantization-methods)
  - [Neural Network Modules](#neural-network-modules)
  - [Optimizers](#optimizers)
  - [Fused Operations](#fused-operations)
  - [Model-Level Operations](#model-level-operations)
  - [Analysis Tools](#analysis-tools)
  - [Custom Kernel Development](#custom-kernel-development)
  - [Distributed Training](#distributed-training)
  - [KV Cache Quantization](#kv-cache-quantization)
- [Training with Quantization](#training-with-quantization)
- [Inference Optimization](#inference-optimization)
- [Advanced Topics](#advanced-topics)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Overview

mxtorch implements **MX (Microscaling) quantization** for PyTorch, enabling real bit-packed storage and arithmetic on quantized tensors. Unlike "fake quantization" approaches that simulate low precision while maintaining float32 copies, mxtorch packs values into actual int8/int32 storage, achieving **4-32x memory compression**.

### Key Features

| Feature | Description |
|---------|-------------|
| **2048 Data Types** | int1-int128, float1-float128 with multiple variants |
| **Real Bit Packing** | Values are actually packed into int8/int32 storage |
| **Triton GPU Kernels** | Optimized packed matmul, quantize, and fused operations |
| **Seamless PyTorch Integration** | `mx_tensor` IS a `torch.Tensor` subclass |
| **Training Support** | Straight-through estimator (STE), stochastic rounding |
| **Advanced Techniques** | NF4, GPTQ, AWQ, Hadamard rotation, vector-wise quant |
| **nn.Module Drop-ins** | `mx_linear`, `mx_conv2d`, `mx_lstm`, etc. |
| **Distributed Ready** | DDP hooks, FSDP policies for multi-GPU |

### Compression Ratios

| Data Type | Bits | Compression vs FP32 | Values per int8 |
|-----------|------|---------------------|-----------------|
| int1d | 1 | 32x | 8 |
| int2d | 2 | 16x | 4 |
| int4d | 4 | 8x | 2 |
| int8d | 8 | 4x | 1 |
| float4d | 4 | 8x | 2 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mxtorch.git
cd mxtorch

# Install dependencies
pip install torch triton

# Use the library
python -c "import mxtorch as mxt; print(mxt.get_version_info())"
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Triton (optional, for GPU kernels)
- CUDA-capable GPU (recommended) or ROCm (AMD)

---

## Quick Start

```python
import mxtorch as mxt
import torch
import torch.nn as nn

# ── Basic Tensor Quantization ────────────────────────────────────────────────
x = torch.randn(512, 512)

# Quantize to int4 (8x compression)
q = mxt.mx_tensor.quantize(x, mxt.int4d, block=128)
print(f"Original: {x.numel() * 4 / 1024:.1f} KB")
print(f"Quantized: {q.nbytes_packed / 1024:.1f} KB ({q.compression_ratio:.1f}x)")

# Dequantize back
x_restored = q.dequantize()
print(f"SNR: {mxt.snr(x, mxt.int4d):.1f} dB")

# ── Model Quantization ──────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
)

# Quantize entire model to int4
model = mxt.to_mx(model, "int4d")

# Or use patched .to()
model = model.to("int4d")

# Mixed precision per layer
model = mxt.to_mx(model, {
    "0": "int4d",      # First linear: int4
    "2": "int8d",      # Second linear: int8
})

# ── Training with Quantized Weights ─────────────────────────────────────────
optimizer = mxt.mx_adam_w(model.parameters(), state_dtype="int8d")

for epoch in range(10):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # States stored at int8 precision
```

---

## Core Concepts

### MX Data Types

mxtorch supports **2048 quantization data types** organized as:

```
{kind}{bits}{mode}{variant}

kind    ∈ {int, float}
bits    ∈ {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128}
mode    ∈ {d, u}    (down=saturating, up=zero-padded)
variant ∈ {"", "h", "v", "s", "b"}
```

#### Variants Explained

| Variant | Suffix | Description | Use Case |
|---------|--------|-------------|----------|
| Base | (none) | Standard block-wise absmax quantization | General inference |
| Hadamard | `h` | QuIP# rotation before quantization | Better SNR at int2/int4 |
| Vector-wise | `v` | Per-row/col absmax (bitsandbytes style) | Attention weights |
| Stochastic | `s` | Stochastic rounding (unbiased) | Training |
| Boolean | `b` | Binary clamp to {0, 1} | 1-bit networks |

```python
# Common dtypes
int4d    # 4-bit integer, saturating, base variant
int4dh   # 4-bit with Hadamard rotation
int4dv   # 4-bit vector-wise
int4ds   # 4-bit stochastic rounding
int4db   # 4-bit boolean/binary

float8u  # 8-bit float, zero-padded
float8us # 8-bit float, stochastic rounding
```

### Block-wise Quantization

Quantization scales are computed per block of consecutive elements, capturing local magnitude variations:

```python
# Block size 128: one scale per 128 values
q = mxt.mx_tensor.quantize(weight, mxt.int4d, block=128)

# Smaller blocks = more accurate, larger overhead
q = mxt.mx_tensor.quantize(weight, mxt.int4d, block=32)   # More accurate
q = mxt.mx_tensor.quantize(weight, mxt.int4d, block=256)  # Less overhead
```

### Real Bit Packing

Values are **actually packed** into int8/int32 storage—not simulated:

```python
x = torch.randn(128)  # 512 bytes in float32
q = mxt.mx_tensor.quantize(x, mxt.int4d, block=128)

# int4d packs 2 values per byte
print(f"Packed storage: {q.packed.shape} = {q.packed.numel()} bytes")
# tensor([64]) = 64 bytes (2x packed, 128/2 = 64 bytes)

# Access raw packed data
packed_int8 = q.packed  # Raw int8 storage
scales = q.mx_scales    # Per-block float32 scales
```

---

## API Reference

### Configuration

```python
import mxtorch as mxt

# ── Global Configuration ─────────────────────────────────────────────────────
config = mxt.mx_config.current()
print(config)
# mx_config(block_size=128, strict=False, debug=False, verbose=False, 
#           default_dtype='int4d', cache_kernels=True, max_autotune=True)

# Temporarily override settings
with mxt.mx_config.override(block_size=64, strict=True):
    q = mxt.mx_tensor.quantize(x, mxt.int4d)  # Uses block=64

# Set defaults
mxt.mx_config.set_default("block_size", 64)

# ── Environment Variables ────────────────────────────────────────────────────
# MX_DEBUG=1         → Enable verbose debug logging
# MX_DEBUG_VERBOSE=1 → Include stack traces in debug output
# MX_STRICT=1        → Raise errors on fallback to fp32
```

### Type System

```python
# ── Resolve dtype by name ────────────────────────────────────────────────────
dt = mxt.get_mx_dtype("int4d")
print(dt.name)         # "int4d"
print(dt.bits)         # 4
print(dt.is_float)     # False
print(dt.is_int)       # True
print(dt.pack_ratio)   # 2 (values per byte)
print(dt.compression_vs_fp32)  # 8.0

# ── Dtypes are singletons ────────────────────────────────────────────────────
assert mxt.get_mx_dtype("int4d") is mxt.int4d

# ── mx_dtype_proxy for torch.dtype compatibility ─────────────────────────────
proxy = mxt.mx_dtype_proxy(mxt.int4d)
print(proxy.name)           # "int4d"
print(proxy.is_floating_point)  # False

# Use with torch.dtype()
model.to(torch.dtype("int4d"))  # Returns mx_dtype_proxy
```

### mx_tensor: The Quantized Tensor

`mx_tensor` is a `torch.Tensor` subclass that stores packed quantized data:

```python
# ── Creation ────────────────────────────────────────────────────────────────
x = torch.randn(256, 512)

# From float tensor
q = mxt.mx_tensor.quantize(x, mxt.int4d, block=128)

# ── Properties ──────────────────────────────────────────────────────────────
print(q.shape)           # torch.Size([256, 512]) — logical shape
print(q.dtype)           # mx.int4d
print(q.device)          # cuda:0
print(q.nbytes_packed)   # Actual bytes used
print(q.compression_ratio)  # ~8.0 for int4d

# ── Dequantize ──────────────────────────────────────────────────────────────
x_restored = q.dequantize()  # → float32 tensor

# ── Arithmetic (dequant → op → requant) ─────────────────────────────────────
a = mxt.mx_tensor.quantize(torch.randn(128, 64), mxt.int4d)
b = mxt.mx_tensor.quantize(torch.randn(64, 128), mxt.int4d)

# Matrix multiplication (uses packed Triton kernels for int1/int2/int4)
c = a @ b  # → mx_tensor

# Element-wise ops
d = a + b.T
e = a * 0.5

# ── Shape Operations ─────────────────────────────────────────────────────────
q_reshaped = q.reshape(128, 1024)
q_flat = q.flatten()
q_t = q.t()  # Transpose (requires requantization)

# ── Device Transfer ─────────────────────────────────────────────────────────
q_cpu = q.cpu()
q_gpu = q.cuda()
q_gpu1 = q.cuda(1)  # Multi-GPU

# ── Re-quantization ─────────────────────────────────────────────────────────
q_int8 = q.to("int8d")  # Change precision
q_float = q.to(torch.float32)  # Dequantize to float

# ── Indexing ─────────────────────────────────────────────────────────────────
row = q[0]      # First row (dequant → requant)
block = q[:10]  # First 10 rows

# ── Reductions ──────────────────────────────────────────────────────────────
s = q.sum()
m = q.mean()
mx = q.max()
```

### Quantization Methods

#### Basic Quantization

```python
import mxtorch as mxt

# ── Standard block-wise quantization ─────────────────────────────────────────
q = mxt.mx_quantize(tensor, "int4d", block=128)
q = mxt.mx_tensor.quantize(tensor, mxt.int4d, block=128)

# ─-- Via tensor method ──────────────────────────────────────────────────────
q = tensor.quantize("int4d", block=128)

# ── Via mx_quantizer class ──────────────────────────────────────────────────
q = mxt.mx_quantizer.quantize(tensor, "int4d", block=128)
```

#### Stochastic Rounding (Training-Friendly)

```python
# Unbiased quantization noise for training
q = mxt.stochastic_mx_quantize(tensor, "int8d", block=128)

# Or use the tensor method
q = tensor.stochastic_quantize("int8d")

# Low-level stochastic rounding
rounded = mxt.stochastic_round(tensor, bits=8)
```

#### Hadamard Rotation (QuIP# Style)

Improves SNR by 2-5 dB for int2/int4 by spreading outliers:

```python
# Get rotation + quantized weight
rotation, q = mxt.hadamard_quantize(weight, "int4d", block=128, seed=42)

# At inference: rotate input before multiply
out = mxt.fused_linear_relu(rotation.rotate(x), q.dequantize())

# Measure improvement
snr_plain = mxt.snr(weight, "int4d")
snr_had = mxt.snr(rotation.rotate(weight), "int4d")
print(f"SNR improvement: {snr_had - snr_plain:+.1f} dB")
```

#### Vector-wise Quantization (bitsandbytes Style)

```python
# Per-row scaling
codes, scales = mxt.vector_quantize(weight, "int8d", axis=1)
reconstructed = mxt.vector_dequantize(codes, scales, axis=1)

# Per-column scaling
codes, scales = mxt.vector_quantize(weight, "int8d", axis=0)
```

#### NF4 Quantization (QLoRA/bitsandbytes)

```python
# Non-uniform 4-bit quantization with optimal value distribution
q = mxt.nf4_quantize(tensor, block=64)
restored = mxt.nf4_dequantize(q)

# Via tensor method
q = tensor.nf4_quantize(block=64)
```

#### Double Quantization (GPTQ Style)

Quantize the scales themselves for extra compression:

```python
dq = mxt.double_quantize(tensor, "int4d", block=128)
# dq contains: packed values + packed scales + meta-scales
restored = mxt.double_dequantize(dq)
```

#### Dynamic Quantization (Runtime Activation)

```python
# Per-token quantization (common in LLM inference)
x_q = mxt.dynamic_quantize(x, "int8d", granularity="per_token")

# Per-channel (for convolutions)
x_q = mxt.dynamic_quantize(x, "int8d", granularity="per_channel")

# Per-tensor (fastest, lowest accuracy)
x_q = mxt.dynamic_quantize(x, "int8d", granularity="per_tensor")

# Per-block (same as static)
x_q = mxt.dynamic_quantize(x, "int8d", granularity="per_block", block=64)
```

#### GPTQ and AWQ Quantization

```python
# GPTQ: Post-training quantization with calibration data
q = mxt.gptq_quantize(weight, "int4d", block=128, groupsize=128, perchannel=True)

# AWQ: Activation-aware weight quantization
q = mxt.awq_quantize(weight, "int4d", block=128, alpha=0.5)
```

#### GGML Quantization (llama.cpp Compatible)

```python
# Q4_K: 4-bit with two-level scaling
q = mxt.ggml_quantize(tensor, "Q4_K")

# Q5_K: 5-bit with two-level scaling
q = mxt.ggml_quantize(tensor, "Q5_K")

# Q6_K: 6-bit with two-level scaling
q = mxt.ggml_quantize(tensor, "Q6_K")
```

#### SmoothQuant

```python
# Balance activation and weight scales
q = mxt.smooth_quantize(weight, "int8d", block=128, alpha=0.5)
```

### Neural Network Modules

mxtorch provides drop-in replacements for standard PyTorch modules:

#### Linear Layers

```python
# ── mx_linear ───────────────────────────────────────────────────────────────
import torch.nn as nn
import mxtorch as mxt

# From scratch
linear = mxt.mx_linear(512, 256, mx_dtype=mxt.int4d, block=128)

# From existing nn.Linear
fp_linear = nn.Linear(512, 256)
linear = mxt.mx_linear.from_linear(fp_linear, mxt.int4d, block=128)

# Forward pass (automatic dequant/requant)
out = linear(x)  # x can be float or mx_tensor

# ── mx_dynamic_linear (per-token activation quant) ──────────────────────────
dyn_linear = mxt.mx_dynamic_linear(512, 256, mx_dtype=mxt.int8d)
out = dyn_linear(x)  # Activations quantized dynamically

# ── mx_sparse_linear (pruned weights) ──────────────────────────────────────
sparse_linear = mxt.mx_sparse_linear.from_linear(
    fp_linear, mxt.int4d, sparsity=0.5, structured=False
)
out = sparse_linear(x)

# ── mx_lora_linear (QLoRA style) ────────────────────────────────────────────
qlora = mxt.mx_lora_linear.from_linear(
    fp_linear, rank=16, base_dtype="int4d", lora_dtype=torch.bfloat16
)
out = qlora(x)
# Only qlora.lora_A and qlora.lora_B have gradients

# ── mx_mixed_int8_linear (LLM.int8() style) ─────────────────────────────────
mixed = mxt.mx_mixed_int8_linear.from_linear(
    fp_linear, threshold=6.0
)
# Outlier columns kept in fp16, rest in int8
```

#### Convolution Layers

```python
# ── mx_conv2d ───────────────────────────────────────────────────────────────
conv = mxt.mx_conv2d(3, 64, kernel_size=3, padding=1, mx_dtype=mxt.int4d)

# From existing nn.Conv2d
fp_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
conv = mxt.mx_conv2d.from_conv2d(fp_conv, mxt.int4d, block=128)

# ── mx_conv1d ───────────────────────────────────────────────────────────────
conv1d = mxt.mx_conv1d(64, 128, kernel_size=3, mx_dtype=mxt.int4d)

# ── mx_conv_transpose2d ─────────────────────────────────────────────────────
conv_t = mxt.mx_conv_transpose2d(64, 3, kernel_size=3, mx_dtype=mxt.int4d)
```

#### Normalization Layers

```python
# ── mx_layer_norm ───────────────────────────────────────────────────────────
ln = mxt.mx_layer_norm(512, mx_dtype=mxt.int8d, block=128)
ln = mxt.mx_layer_norm.from_layer_norm(nn.LayerNorm(512), mxt.int8d)

# ── mx_rms_norm (LLaMA style) ───────────────────────────────────────────────
rms = mxt.mx_rms_norm(512, eps=1e-6, mx_dtype=mxt.int8d)

# ── mx_group_norm ───────────────────────────────────────────────────────────
gn = mxt.mx_group_norm(32, 64, mx_dtype=mxt.int8d)

# ── mx_batch_norm2d ─────────────────────────────────────────────────────────
bn = mxt.mx_batch_norm2d.from_batch_norm(nn.BatchNorm2d(64), mxt.int8d)
```

#### Recurrent Layers

```python
# ─-- mx_lstm ────────────────────────────────────────────────────────────────
lstm = mxt.mx_lstm(input_size=64, hidden_size=128, mx_dtype=mxt.int8d)
out, (h_n, c_n) = lstm(x, (h0, c0))

# From existing nn.LSTM
fp_lstm = nn.LSTM(64, 128, batch_first=True)
lstm = mxt.mx_lstm.from_lstm(fp_lstm, mx_dtype="int8d")

# ── mx_gru ──────────────────────────────────────────────────────────────────
gru = mxt.mx_gru(input_size=64, hidden_size=128, mx_dtype=mxt.int8d)
out, h_n = gru(x, h0)
```

#### Attention and Transformer

```python
# ── mx_multihead_attention ──────────────────────────────────────────────────
mha = mxt.mx_multihead_attention(embed_dim=512, num_heads=8, mx_dtype=mxt.int4d)
out, attn_weights = mha(query, key, value)

# ── mx_transformer_encoder_layer ────────────────────────────────────────────
encoder = mxt.mx_transformer_encoder_layer(
    d_model=512, nhead=8, dim_feedforward=2048, mx_dtype=mxt.int4d
)
out = encoder(src)
```

#### Embedding

```python
# ── mx_embedding ────────────────────────────────────────────────────────────
emb = mxt.mx_embedding(num_embeddings=10000, embedding_dim=512, mx_dtype=mxt.int4d)
embedded = emb(indices)

# From existing nn.Embedding
fp_emb = nn.Embedding(10000, 512)
emb = mxt.mx_embedding.from_embedding(fp_emb, mxt.int4d)

# ── mx_embedding_bag ────────────────────────────────────────────────────────
emb_bag = mxt.mx_embedding_bag(10000, 512, mx_dtype=mxt.int4d)
```

### Optimizers

#### mx_adam_w: Quantized-State AdamW

Stores momentum and variance at MX precision, reducing optimizer memory by 4x:

```python
import mxtorch as mxt
import torch.nn as nn

model = nn.Linear(512, 256)
model = mxt.to_mx(model, "int4d")

# Standard AdamW states: fp32 (4 bytes each) × 2 states = 8x model size
# mx_adam_w states: int8 (1 byte each) × 2 states = 2x model size

optimizer = mxt.mx_adam_w(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    state_dtype="int8d",  # Store m, v at int8 precision
    block=128
)

# Training loop (identical to standard AdamW)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Fused Operations

Fused kernels avoid intermediate float32 buffers:

```python
import mxtorch as mxt
import torch

# ── fused_int8_linear ───────────────────────────────────────────────────────
x_q = mxt.dynamic_quantize(x, "int8d", "per_token")
w_q = mxt.mx_tensor.quantize(weight, mxt.int8d)
out = mxt.fused_int8_linear(x_q, w_q, bias)

# ── fused_linear_relu ───────────────────────────────────────────────────────
out = mxt.fused_linear_relu(x, weight_mx, bias)

# ── fused_silu_and_mul (SwiGLU) ─────────────────────────────────────────────
# For LLaMA/Mistral MLP: down_proj(silu(gate) * up)
gate_q = mxt.mx_tensor.quantize(gate_proj(x), mxt.int8d)
up_q = mxt.mx_tensor.quantize(up_proj(x), mxt.int8d)
act = mxt.fused_silu_and_mul(gate_q, up_q)

# ── fused_qkv_projection ────────────────────────────────────────────────────
# Reads input once, projects to Q, K, V simultaneously
Q, K, V = mxt.fused_qkv_projection(x, wq_mx, wk_mx, wv_mx, n_heads=32)

# ── fused_rope_int8 ─────────────────────────────────────────────────────────
# Fused dequantize + rotary position embedding
q_rot, k_rot = mxt.fused_rope_int8(q_mx, k_mx, cos, sin)

# ── fused_sdpa_int8 ─────────────────────────────────────────────────────────
# Fused scaled dot-product attention
out = mxt.fused_sdpa_int8(q_mx, k_mx, v_mx, scale=None)

# ── fused_add_rms_norm ──────────────────────────────────────────────────────
# Common in transformer blocks: out = RMSNorm(x + residual)
out = mxt.fused_add_rms_norm(attn_out, residual, norm_weight, eps=1e-6)
```

### Model-Level Operations

```python
import mxtorch as mxt
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.LayerNorm(128)
)

# ── Quantize Entire Model ───────────────────────────────────────────────────
# Simple: all layers to int4d
model = mxt.to_mx(model, "int4d")

# Via patched .to()
model = model.to("int4d")

# Via torch.dtype proxy
model = model.to(torch.dtype("int4d"))

# ── Mixed Precision ─────────────────────────────────────────────────────────
# Regex-based per-layer precision
model = mxt.to_mx(model, {
    ".*Linear.*": "int4d",
    ".*LayerNorm.*": "int8d",
})

# Named layer precision
model = mxt.to_mx(model, {
    "0": "int4d",      # First linear
    "2": "int8d",      # Second linear
    "3": "float32",    # Skip LayerNorm
})

# ── Activation Quantization ─────────────────────────────────────────────────
# Wrap model to quantize activations on-the-fly
model = mxt.to_mx(model, "int4d")          # Weight quantization
model = mxt.wrap_activations(model, "int8d")  # Activation quantization

# Remove hooks
mxt.unwrap_activations(model)

# ── Save/Load ───────────────────────────────────────────────────────────────
mxt.save_quantized(model, "model_int4.mx")
model = mxt.load_quantized("model_int4.mx", MyModelClass, dtype="int4d")

# ── Inspect Model ───────────────────────────────────────────────────────────
print(mxt.inspect_model(model))
# mx_linear(512→256, int4d, 8.0x compression)
# mx_linear(256→128, int8d, 4.0x compression)
```

### Analysis Tools

```python
import mxtorch as mxt

# ── SNR (Signal-to-Noise Ratio) ─────────────────────────────────────────────
snr_db = mxt.snr(weight, "int4d", block=128)
print(f"int4d SNR: {snr_db:.1f} dB")

# Via tensor method
snr_db = weight.snr("int4d")

# ── Quantization Error ──────────────────────────────────────────────────────
rmse = mxt.quantization_error(weight, "int4d", metric="rmse")
mae = mxt.quantization_error(weight, "int4d", metric="mae")
max_error = mxt.quantization_error(weight, "int4d", metric="max")

# ── Compare Dtypes ──────────────────────────────────────────────────────────
comparison = mxt.compare_dtypes(weight, ["int2d", "int4d", "int8d"])
for name, metrics in comparison.items():
    print(f"{name}: SNR={metrics['snr']:.1f} dB, RMSE={metrics['rmse']:.4f}")

# ── Pack Strategy ───────────────────────────────────────────────────────────
strategy = mxt.pack_strategy(mxt.int4d)
print(strategy)
# PackStrategy(dtype=int4d, bits=4, pack_ratio=2, storage_dtype=int8)

# ── Hardware Info ───────────────────────────────────────────────────────────
print(mxt.mx_info.hw_info())
# Hardware: rx_7900_xtx
#   int1d       : 8x per 8-bit native op  (32x vs fp32)
#   int2d       : 4x per 8-bit native op  (16x vs fp32)
#   int4d       : 2x per 8-bit native op  (8x vs fp32)
#   ...

# ── Dtype Info ──────────────────────────────────────────────────────────────
print(mxt.mx_info.dtype_info("int4d"))
```

### Custom Kernel Development

mxtorch provides utilities for writing Triton kernels that operate on `mx_tensor`:

```python
import mxtorch as mxt
import torch
import triton
import triton.language as tl

# ── Understanding mx_tensor Layout ──────────────────────────────────────────
q = mxt.mx_tensor.quantize(torch.randn(128), mxt.int8d, block=32)

# Internal storage
packed_int8 = q.packed      # Packed int8 data (or q._mx_packed)
scales = q.mx_scales        # Per-block float32 scales (or q._mx_scales)
n = q.numel()               # Original element count (or q._mx_n)
block = q.mx_block          # Block size (or q._mx_block)
dtype = q.mx_dtype          # MX dtype (or q._mx_dtype)

# ── Example: Custom Scale Kernel ────────────────────────────────────────────
@triton.jit
def custom_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    N, BS: tl.constexpr, BLK: tl.constexpr,
):
    """
    Scale each element of packed int8 data by per-block scale.
    """
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    mask = offs < N
    
    x_int = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int8)
    scale = tl.load(scale_ptr + offs // BS, mask=mask, other=1.0)
    
    x_float = x_int.to(tl.float32) * scale
    tl.store(out_ptr + offs, x_float, mask=mask)


def custom_mx_scale(x: mxt.mx_tensor) -> torch.Tensor:
    """Wrapper that calls the custom kernel."""
    if not torch.cuda.is_available():
        return x.dequantize()
    
    N = x._mx_n
    BLK = 256
    n_blocks = (N + BLK - 1) // BLK
    
    out = torch.empty(N, dtype=torch.float32, device=x.device)
    
    custom_scale_kernel[(n_blocks,)](
        x._mx_packed, x._mx_scales, out,
        N, BS=x._mx_block, BLK=BLK,
    )
    
    return out.reshape(x.shape)


# ── Example: ReLU That Stays Quantized ──────────────────────────────────────
@triton.jit
def mx_relu_kernel(
    x_ptr, scale_ptr, out_ptr, out_scale_ptr,
    N, BS: tl.constexpr, BLK: tl.constexpr,
):
    """
    Apply ReLU to packed int8 data, output stays quantized.
    """
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BLK)
    mask = offs < N
    
    x_int = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int8)
    scale = tl.load(scale_ptr + offs // BS, mask=mask, other=1.0)
    
    # Dequantize, apply ReLU
    x_float = x_int.to(tl.float32) * scale
    x_relu = tl.where(x_float > 0, x_float, 0.0)
    
    # Compute new scale for this block
    relu_max = tl.max(x_relu, axis=0)
    new_scale = tl.where(relu_max < 1e-12, 1e-12, relu_max / 127.0)
    tl.store(out_scale_ptr + pid, new_scale)
    
    # Quantize and store
    out_int = tl.minimum(tl.maximum((x_relu / new_scale + 0.5).to(tl.int8), -128), 127)
    tl.store(out_ptr + offs, out_int, mask=mask)


def custom_mx_relu(x: mxt.mx_tensor) -> mxt.mx_tensor:
    """ReLU that outputs a quantized mx_tensor."""
    N = x._mx_n
    BS = x._mx_block
    n_blocks = (N + BS - 1) // BS
    
    out_packed = torch.empty(N, dtype=torch.int8, device=x.device)
    out_scales = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    
    mx_relu_kernel[(n_blocks,)](
        x._mx_packed, x._mx_scales, out_packed, out_scales,
        N, BS=BS, BLK=BS,
    )
    
    return mxt.mx_tensor(out_packed, out_scales, x._mx_dtype, x.shape, N, BS)


# ── Register Custom Kernel ──────────────────────────────────────────────────
@mxt.register_kernel(op="mx.relu", dtypes=["int8d"], hardware=["gfx1100", "sm_"], force="auto")
def my_relu_kernel():
    """Register custom kernel for automatic dispatch."""
    return custom_mx_relu
```

### Distributed Training

```python
import mxtorch as mxt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ── DDP with Quantized Gradients ────────────────────────────────────────────
model = mxt.to_mx(model, "int4d")
mxt.install_ddp_hooks(model)  # Pack gradients before all-reduce
model = DDP(model, device_ids=[local_rank])

# ── FSDP Policy for mx_tensor ───────────────────────────────────────────────
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

policy = mxt.make_fsdp_mx_policy("int4d", block=128)
model = FSDP(model, auto_wrap_policy=policy)

# ── mx_fsdp_wrapper for Save/Load ───────────────────────────────────────────
wrapper = mxt.mx_fsdp_wrapper(model, dtype="int4d")
wrapper.save("checkpoint.mx")
wrapper.load("checkpoint.mx")
```

### KV Cache Quantization

For memory-efficient long-context inference:

```python
import mxtorch as mxt

# ── Initialize KV Cache ─────────────────────────────────────────────────────
cache = mxt.kv_cache_quantizer(
    n_heads=32,
    head_dim=128,
    dtype="int8d",
    max_seq_len=32768,
    asymmetric_v=True  # Asymmetric quantization for V
)

# ── During Generation ───────────────────────────────────────────────────────
for step in range(max_new_tokens):
    # K, V: [batch, heads, 1, head_dim]
    k_new, v_new = attention_layer.compute_kv(x)
    
    # Quantize and append
    cache.append_kv(k_new, v_new)
    
    # Get full history
    k_hist, v_hist = cache.get()  # [batch, heads, seq_len, head_dim]
    
    # Or get quantized directly
    k_q, v_q = cache.get_quantized()  # Lists of mx_tensor
    
    # Compute attention
    out = attention(k_hist, v_hist, q_new)

# ── Memory Tracking ─────────────────────────────────────────────────────────
print(f"KV cache: {cache.memory_bytes() / 1024 / 1024:.1f} MB")
print(f"Compression vs fp16: {cache.compression_vs_fp16():.1f}x")
print(f"Sequence length: {cache.seq_len}")

# ── Reset Between Requests ──────────────────────────────────────────────────
cache.reset()
```

---

## Training with Quantization

### Quantization-Aware Training (QAT) with STE

```python
import mxtorch as mxt
import torch
import torch.nn as nn

# ── Model Setup ─────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Quantize weights
model = mxt.to_mx(model, "int4d")

# ── Training Loop with STE ──────────────────────────────────────────────────
optimizer = mxt.mx_adam_w(model.parameters(), lr=1e-3, state_dtype="int8d")

for epoch in range(10):
    for batch, target in train_loader:
        optimizer.zero_grad()
        
        # Forward pass (dequantize → compute → gradients flow via STE)
        output = model(batch)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update (states at int8, weights requantized)
        optimizer.step()
```

### Fine-Tuning with LoRA (QLoRA Style)

```python
import mxtorch as mxt
import torch.nn as nn

# ── Load Pretrained Model ───────────────────────────────────────────────────
model = load_pretrained_model()

# ── Replace Linears with QLoRA ──────────────────────────────────────────────
for name, module in list(model.named_modules()):
    if isinstance(module, nn.Linear):
        parent, child_name = get_parent(model, name)
        qlora = mxt.mx_lora_linear.from_linear(
            module,
            rank=16,
            base_dtype="int4d",    # Frozen base in int4
            lora_dtype=torch.bfloat16  # Trainable adapters in bf16
        )
        setattr(parent, child_name, qlora)

# ── Only LoRA Parameters Have Gradients ─────────────────────────────────────
lora_params = []
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False

optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

# ── Training Loop ───────────────────────────────────────────────────────────
for batch in train_loader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Stochastic Rounding for Training

```python
import mxtorch as mxt

# ── Gradient Accumulation with Stochastic Rounding ──────────────────────────
accumulated_grad = torch.zeros_like(param.data)

for i, batch in enumerate(train_loader):
    loss = model(batch)
    loss.backward()
    
    # Accumulate with stochastic rounding
    accumulated_grad = mxt.stochastic_round(
        accumulated_grad + param.grad,
        bits=8
    )
    
    if (i + 1) % accumulation_steps == 0:
        param.data -= lr * accumulated_grad
        accumulated_grad.zero_()
```

---

## Inference Optimization

### End-to-End Quantized Pipeline

```python
import mxtorch as mxt
import torch

# ── Load and Quantize Model ─────────────────────────────────────────────────
model = load_model()
model = mxt.to_mx(model, "int4d")
model.eval()

# ── Optional: Activation Quantization ───────────────────────────────────────
model = mxt.wrap_activations(model, "int8d")

# ── Inference ───────────────────────────────────────────────────────────────
with torch.no_grad():
    # Input can be float or mx_tensor
    input_q = mxt.dynamic_quantize(input, "int8d", "per_token")
    output = model(input_q)

# ── Clean Up ────────────────────────────────────────────────────────────────
mxt.unwrap_activations(model)
```

### Using Fused Operations for LLM Inference

```python
import mxtorch as mxt

# ── Transformer Block with Fused Ops ────────────────────────────────────────
class QuantizedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # Quantized weights
        self.wq = mxt.mx_tensor.quantize(torch.randn(d_model, d_model), mxt.int8d)
        self.wk = mxt.mx_tensor.quantize(torch.randn(d_model, d_model), mxt.int8d)
        self.wv = mxt.mx_tensor.quantize(torch.randn(d_model, d_model), mxt.int8d)
        self.wo = mxt.mx_tensor.quantize(torch.randn(d_model, d_model), mxt.int8d)
        
        # MLP weights
        self.gate_proj = mxt.mx_tensor.quantize(torch.randn(d_ff, d_model), mxt.int4d)
        self.up_proj = mxt.mx_tensor.quantize(torch.randn(d_ff, d_model), mxt.int4d)
        self.down_proj = mxt.mx_tensor.quantize(torch.randn(d_model, d_ff), mxt.int4d)
        
        # Norm weights
        self.norm_weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x, cos, sin):
        # Fused QKV projection (reads x once)
        Q, K, V = mxt.fused_qkv_projection(x, self.wq, self.wk, self.wv, n_heads=32)
        
        # Fused RoPE
        Q, K = mxt.fused_rope_int8(Q, K, cos, sin)
        
        # Fused attention
        attn_out = mxt.fused_sdpa_int8(Q, K, V)
        
        # Fused residual add + RMSNorm
        x = mxt.fused_add_rms_norm(attn_out, x, self.norm_weight)
        
        # MLP with SwiGLU
        gate = mxt.fused_int8_linear(x, self.gate_proj)
        up = mxt.fused_int8_linear(x, self.up_proj)
        mlp_out = mxt.fused_silu_and_mul(gate, up)
        mlp_out = mxt.fused_int8_linear(mlp_out, self.down_proj)
        
        return x + mlp_out
```

---

## Advanced Topics

### Mixed Precision Resolution

When operating on tensors with different dtypes, mxtorch resolves the output type:

```python
# ── Resolution Rules ────────────────────────────────────────────────────────
# mode = "u" if either is "u" (up-padded wins)
# bits = min(bits) (lower precision wins)
# kind = "float" if either is float
# variant = higher priority (h > v > s > b > "")

a = mxt.mx_tensor.quantize(torch.randn(64, 64), mxt.int4d)
b = mxt.mx_tensor.quantize(torch.randn(64, 64), mxt.int8d)
c = a + b  # Result is int4d (lower bits)

a = mxt.mx_tensor.quantize(torch.randn(64, 64), mxt.int4dh)  # Hadamard
b = mxt.mx_tensor.quantize(torch.randn(64, 64), mxt.int4d)   # Base
c = a @ b  # Result is int4dh (Hadamard has higher priority)
```

### Context Manager for Default Dtypes

```python
# ── mx_mode ─────────────────────────────────────────────────────────────────
with mxt.mx_mode("int4d", block=64):
    # All quantization inside uses int4d, block=64
    q1 = mxt.mx_quantize(x)
    q2 = mxt.mx_quantize(y)
```

### Boolean Quantization

```python
# ── 1-bit Neural Networks ───────────────────────────────────────────────────
# Convert to binary
binary = mxt.bool_to_mx(weight, "int1db", block=128)

# Logical operations
a = mxt.bool_to_mx(tensor_a, "int1db")
b = mxt.bool_to_mx(tensor_b, "int1db")
c = mxt.mx_logical_and(a, b)
d = mxt.mx_logical_or(a, b)
e = mxt.mx_logical_xor(a, b)
f = mxt.mx_logical_not(a)

# Via tensor methods
c = a.logical_and(b)
d = a.logical_or(b)
```

### Sparse + Quantized

```python
# ── Magnitude Pruning + Quantization ────────────────────────────────────────
sparse_w = mxt.prune_to_sparse(weight, sparsity=0.5, dtype="int4d")
print(f"Density: {sparse_w.density:.1%}")
print(f"Compression: {sparse_w.compression_vs_dense_fp32():.1f}x")

# Convert to PyTorch sparse formats
csr = sparse_w.to_torch_sparse_csr()
coo = sparse_w.to_torch_sparse_coo()

# ── Semi-Structured (2:4) Sparsity ──────────────────────────────────────────
# For NVIDIA A100/H100 sparse Tensor Cores
semi_sparse, mx_w = mxt.to_semi_structured_sparse(weight, dtype="int8d")
```

---

## Performance

### Hardware Support

| Hardware | Backend | Native int8 | Native int4 | Sparse 2:4 |
|----------|---------|-------------|-------------|------------|
| NVIDIA H100 | CUDA | ✓ | ✓ | ✓ |
| NVIDIA A100 | CUDA | ✓ | ✓ | ✓ |
| NVIDIA RTX 4090 | CUDA | ✓ | - | ✓ |
| AMD RX 7900 XTX | ROCm | ✓ | - | - |
| AMD MI300X | ROCm | ✓ | ✓ | - |

### Benchmark Results

On AMD RX 7900 XTX (gfx1100):

```
Dtype    :  1-bit,  32.0x vs fp32, 8x packed per 8-bit native op
int2d    :  2-bit,  16.0x vs fp32, 4x packed per 8-bit native op
int4d    :  4-bit,   8.0x vs fp32, 2x packed per 8-bit native op
int8d    :  8-bit,   4.0x vs fp32, 1x packed per 8-bit native op
float4d  :  4-bit,   8.0x vs fp32, 2x packed per 8-bit native op
float8u  :  8-bit,   4.0x vs fp32, 1x packed per 8-bit native op
```

### Optimization Tips

1. **Use larger block sizes** (128-256) for better throughput, smaller (32-64) for accuracy
2. **Enable kernel caching**: `mx_config.set_default("cache_kernels", True)`
3. **Use fused operations** to avoid intermediate dequantization
4. **For training**: Use `mx_adam_w` to reduce optimizer memory
5. **For inference**: Wrap activations with `wrap_activations()` for full quantization

---

## Troubleshooting

### Common Issues

#### "Triton not found" Warning

```
[mx_triton] Triton not found — pure-PyTorch fallback active.
```

Install Triton: `pip install triton`

#### Slow Performance

- Ensure tensors are on GPU: `tensor.cuda()`
- Check kernel caching is enabled
- Use fused operations instead of separate ops

#### Out of Memory

- Use smaller block sizes for scales
- Use `low_mem=True` in `to_mx()`
- Use `mx_adam_w` instead of standard AdamW

#### Gradient Issues During Training

- Use stochastic rounding: `stochastic_mx_quantize()`
- Check that STE is working: `loss.backward()` should propagate gradients
- Use `mx_adam_w` which handles quantized parameters

### Debug Mode

```python
import os
os.environ["MX_DEBUG"] = "1"
os.environ["MX_DEBUG_VERBOSE"] = "1"

import mxtorch as mxt
# Now mxtorch will print detailed debug information
```

### Strict Mode

```python
import os
os.environ["MX_STRICT"] = "1"

import mxtorch as mxt
# Now mxtorch will raise errors instead of falling back to fp32
```

---

## API Classes Summary

| Class | Purpose |
|-------|---------|
| `mx_config` | Global configuration |
| `mx_dtype` | Quantization data type |
| `mx_dtype_proxy` | Fake torch.dtype for compatibility |
| `mx_tensor` | Quantized tensor (torch.Tensor subclass) |
| `mx_quantizer` | Quantization methods (static class) |
| `mx_logical` | Boolean operations (static class) |
| `mx_fused_ops` | Fused kernel operations (static class) |
| `mx_specialized_matmul` | Specialized matmul ops (static class) |
| `mx_analysis` | Error analysis tools (static class) |
| `mx_model` | Model-level operations (static class) |
| `mx_info` | System/dtype info (static class) |
| `mx_context` | Context managers (static class) |
| `mx_kv_cache` | KV cache utilities (static class) |
| `mx_adam_w` | Quantized-state AdamW optimizer |

---

## Module Exports

```python
# Configuration
mx_config, get_version_info, print_module_info

# Type system
mx_dtype, mx_dtype_proxy, get_mx_dtype

# Common dtype aliases
int1d, int2d, int4d, int8d, float4d, float8u, float8d, ...
# Variant aliases
int4dh, int4dv, int4ds, int4db, float8us, ...

# Core classes
mx_tensor, nf4_tensor, sparse_mx_tensor, quantization_result

# Quantization functions
mx_quantize, mx_matmul, dynamic_quantize
stochastic_round, stochastic_mx_quantize
hadamard_quantize, hadamard_rotation
vector_quantize, vector_dequantize
nf4_quantize, nf4_dequantize
gptq_quantize, awq_quantize, ggml_quantize
double_quantize, smooth_quantize

# Boolean operations
bool_to_mx, mx_logical_and, mx_logical_or, mx_logical_not, mx_logical_xor

# Fused operations
fused_int8_linear, fused_linear_relu, fused_silu_and_mul
fused_rope_int8, fused_sdpa_int8, fused_add_rms_norm
fused_qkv_projection

# Model operations
to_mx, save_quantized, load_quantized
wrap_activations, unwrap_activations, calibrate

# nn.Module drop-ins
mx_linear, mx_conv2d, mx_conv1d, mx_embedding
mx_layer_norm, mx_rms_norm, mx_batch_norm2d
mx_multihead_attention, mx_transformer_encoder_layer
mx_lstm, mx_gru

# Advanced modules
mx_lora_linear, mx_sparse_linear, mx_dynamic_linear
mx_mixed_int8_linear

# Optimizer
mx_adam_w

# Distributed
install_ddp_hooks, make_fsdp_mx_policy, mx_fsdp_wrapper

# Analysis
snr, quantization_error, compare_dtypes
hardware_probe, hardware_profile, roofline_estimator
inspect_model, pack_strategy

# KV Cache
kv_cache_quantizer

# Sparse
prune_to_sparse, to_semi_structured_sparse

# Context
mx_mode, get_default_dtype

# Custom kernels
register_kernel
```

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---

## Citation

```bibtex
@software{mxtorch2026,
  title = {mxtorch: Sub-byte Quantization for PyTorch with Triton GPU Kernels},
  author = Daniel Derycke,
  year = {2026},
  url = {[https://github.com/your-org/mxtorch](https://github.com/DHDev0/mxtorch/)}
}
```
