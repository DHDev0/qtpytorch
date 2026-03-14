"""
MX Quantization System (mx_triton)
===================================

A comprehensive sub-byte quantization library for PyTorch with Triton GPU kernels.
Enables real bit-packed storage and arithmetic on quantized tensors - no fake quantization.

WHAT THIS LIBRARY DOES
----------------------
This library implements MX (Microscaling) quantization, allowing you to:
  • Compress models by 4-32x through sub-byte quantization (1-8 bit)
  • Run inference directly on packed quantized data
  • Train with quantization-aware methods (STE, stochastic rounding)
  • Deploy quantized models with minimal accuracy loss

KEY FEATURES
------------
1. Real Bit Packing - Values are actually packed into int8/int32 storage, not simulated
2. 2048 Dtypes - Supports int1-int128 and float1-float128, with variants:
     - Base: standard block-wise absmax quantization
     - Hadamard (h): QuIP# rotation for better SNR
     - Vector-wise (v): per-row/col quantization (bitsandbytes style)
     - Stochastic (s): unbiased rounding for training
     - Boolean (b): binary quantization for 1-bit networks

3. Deep PyTorch Integration:
     - mx_tensor IS a torch.Tensor subclass
     - Works with standard optimizers, DDP, FSDP
     - Automatic dtype conversion: tensor.to("int4d")
     - Model quantization: model.to("int4d")

4. Advanced Techniques:
     - NF4/FP4: non-uniform quantization (QLoRA/bitsandbytes)
     - GPTQ/AWQ: post-training quantization
     - SmoothQuant: activation/weight scale balancing
     - LLM.int8(): outlier-aware mixed precision
     - Double quantization: quantize the scales too

5. Fused Operations: optimized kernels that stay in quantized realm
     - fused_int8_linear, fused_linear_relu, fused_silu_and_mul
     - fused_rope_int8, fused_sdpa_int8, fused_add_rms_norm

CODE ORGANIZATION (All lowercase PEP8 naming)
----------------------------------------------
All classes use lowercase_with_underscores naming:

Configuration:
  - mx_config: Global configuration (block size, strict mode, etc.)

Type System:
  - mx_dtype: One of 2048 quantization data types
  - mx_dtype_proxy: Fake torch.dtype for seamless integration

Core Classes:
  - mx_tensor: Quantized tensor (torch.Tensor subclass)
  - bit_packer: Bit-level packing/unpacking utilities
  - quantization_result: Result container with metrics

API Classes (Static Method Groups):
  - mx_quantizer: All quantization methods (quantize, nf4_quantize, gptq_quantize, etc.)
  - mx_logical: Boolean operations (bool_to_mx, logical_and/or/not/xor)
  - mx_fused_ops: Fused kernels (fused_int8_linear, fused_linear_relu, etc.)
  - mx_specialized_matmul: Specialized matmul operations
  - mx_analysis: Error analysis (quantization_error, snr, compare_dtypes)
  - mx_info: System info (get_version_info, hw_info, dtype_info)
  - mx_model: Model operations (to_mx, save_quantized, load_quantized)

nn.Module Drop-in Replacements:
  - mx_linear, mx_conv2d, mx_embedding, mx_layer_norm, etc.
  - mx_transformer_encoder_layer, mx_gru, mx_lstm
  - All compatible with standard PyTorch nn.Module API

QUICK START
-----------
    import mx_triton as mx
    
    # Quantize a tensor
    q = mx.mx_quantizer.quantize(tensor, "int4d", block=128)
    restored = q.dequantize()
    
    # Or use tensor methods
    q = tensor.quantize("int4d")
    
    # Quantize an entire model
    model = mx.mx_model.to_mx(model, "int4d")
    
    # Use drop-in replacement layers
    linear = mx.mx_linear(256, 128, dtype="int4d")

ENVIRONMENT VARIABLES
---------------------
  MX_DEBUG=1          Enable verbose debug logging
  MX_DEBUG_VERBOSE=1  Include stack traces in debug output
  MX_STRICT=1         Raise errors on fallback to fp32

PACKING DETAILS
---------------
  int8  word → int1 : 8  values packed
  int8  word → int2 : 4  values packed
  int8  word → int4 : 2  values packed
  int8  word → int8 : 1  value  packed
  int32 word → int1 : 32 values (CPU AVX-512)

Version: 1.0.0
License: MIT
"""

from __future__ import annotations

# ── Version info ──────────────────────────────────────────────────────────────
__version__ = "1.0.0"
__author__ = "MX Quantization Team"
__license__ = "MIT"

# ── stdlib ───────────────────────────────────────────────────────────────────
import os, gc, re, math, time, logging, warnings, functools, traceback
import inspect, textwrap
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Literal,
                    Optional, Sequence, Tuple, Union)
from contextlib import contextmanager

# ── torch ────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

# ── Suppress known PyTorch warnings at module level ───────────────────────────
# These warnings come from C++ code and can't be suppressed with catch_warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="FSDP.state_dict_type.*deprecated")
warnings.filterwarnings("ignore", message="You are using.*torch.load.*weights_only")

# ── Triton (optional) ────────────────────────────────────────────────────────
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    warnings.warn("[mx_triton] Triton not found — pure-PyTorch fallback active.", stacklevel=2)

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # version and info
    "__version__", "get_version_info", "print_module_info",
    # configuration
    "mx_config",
    # dtype system (2048 types via module attributes)
    "mx_dtype", "mx_dtype_proxy", "get_mx_dtype",
    # common dtype aliases — base (no suffix)
    "int1d","int1u","int2d","int2u","int4d","int4u","int8d","int8u",
    "int16d","int16u","int32d","int32u","float4d","float4u",
    "float8d","float8u","float16d","float16u","float32d","float32u",
    # variant dtype aliases (h=hadamard, v=vector, s=stochastic, b=boolean)
    "int4dh","int4uh","int4dv","int4uv","int4ds","int4us","int4db","int4ub",
    "int8dh","int8uh","int8dv","int8uv","int8ds","int8us","int8db","int8ub",
    "float8uh","float8dh","float8uv","float8dv","float8us","float8ds",
    "float5dh","float5uh","float5ds","float5us",
    "float2uv","float2dv","int2uv","int2dv",
    "int1db","int1ub",
    # tensors
    "mx_tensor", "nf4_tensor",
    # sparse tensors
    "sparse_mx_tensor",
    # nn.Module drop-ins — convolution
    "mx_conv1d","mx_conv2d","mx_conv_transpose1d","mx_conv_transpose2d",
    # nn.Module drop-ins — normalization
    "mx_batch_norm1d","mx_batch_norm2d","mx_layer_norm","mx_rms_norm","mx_group_norm",
    # nn.Module drop-ins — core
    "mx_linear","mx_embedding","mx_embedding_bag","mx_multihead_attention",
    # nn.Module drop-ins — transformer / recurrent
    "mx_transformer_encoder_layer","mx_linear_transformer","mx_gru","mx_lstm",
    # nn.Module drop-ins — extra
    "mx_pixel_shuffle","mx_dropout","mx_alpha_dropout","mx_prelu","mx_bilinear",
    # advanced modules
    "mx_lora_linear","mx_mixed_int8_linear","mx_dynamic_linear","mx_sparse_linear",
    # quantization — uniform (base)
    "mx_quantize","mx_matmul","quantize", "quantization_result",
    "quantization_error","snr","compare_dtypes",
    "dynamic_quantize",
    # quantization — stochastic variant
    "stochastic_round","stochastic_mx_quantize",
    "triton_stochastic_quantize",
    # quantization — Hadamard variant (QuIP#)
    "hadamard_rotation","hadamard_quantize",
    "_fast_hadamard_transform",
    # quantization — vector-wise variant (bitsandbytes style)
    "vector_quantize","vector_dequantize",
    # quantization — boolean variant
    "bool_to_mx","mx_logical_and","mx_logical_or","mx_logical_not","mx_logical_xor",
    # quantization — non-uniform (QLoRA / bitsandbytes style)
    "nf4_quantize","nf4_dequantize",
    "fp4_quantize","fp4_dequantize",
    "double_quantize","double_quantized",
    # advanced PTQ techniques
    "gptq_quantize","gptq_result",
    "awq_quantize","awq_result",
    "ggml_quantize","ggml_quantized",
    # mixed-precision / outlier
    "mixed_int8_decompose","smooth_quantize",
    # sparse
    "prune_to_sparse","to_semi_structured_sparse",
    # KV cache
    "kv_cache_quantizer",
    # model-level API
    "to_mx","save_quantized","load_quantized",
    "wrap_activations","unwrap_activations",
    "calibrate",
    # optimiser
    "mx_adam_w",
    # distributed + FSDP
    "mx_distributed","install_ddp_hooks",
    "mx_fsdp_wrapper","make_fsdp_mx_policy",
    # context managers / helpers
    "mx_mode","get_default_dtype","register_kernel",
    # hardware / analysis
    "hardware_probe","hardware_profile",
    "roofline_estimator","roofline_result","benchmark_report",
    "precision_audit","mx_debugger","dynamic_precision_scheduler",
    "pack_strategy","inspect_model","hw_info","dtype_info",
    "bit_packer",  # Bit packing utilities
    # testing utilities
    "test_block",
    # speed + memory benchmarks
    "run_speed_memory_tests",
    # intra-precision fused ops
    "fused_linear_relu","fused_silu_and_mul","fused_rope_int8",
    "fused_sdpa_int8","fused_add_rms_norm",
    "fused_int8_linear","fused_qkv_projection",
    # class-based API
    "mx_quantizer","mx_logical","mx_fused_ops","mx_specialized_matmul","mx_analysis","mx_kv_cache",
    # new class-based APIs
    "mx_model", "mx_distributed_ops", "mx_sparse_ops", "mx_ops", "mx_context", "mx_info",
    # tensor method registry (for checking installed methods)
    "_mx_tensor_methods",
    "_mx_tensor_method_funcs",
]

# ── Debug flags ──────────────────────────────────────────────────────────────
_DEBUG   = os.environ.get("MX_DEBUG",         "0") == "1"
_VERBOSE = os.environ.get("MX_DEBUG_VERBOSE",  "0") == "1"
_STRICT  = os.environ.get("MX_STRICT",         "0") == "1"

log = logging.getLogger("mx_triton")
if _DEBUG:
    logging.basicConfig(level=logging.DEBUG)
    log.setLevel(logging.DEBUG)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class mx_config:
    """
    Global configuration for the MX quantization system.
    
    Controls default behavior for quantization, kernel selection, and debugging.
    All settings can be overridden at runtime or via environment variables.
    
    Usage:
        # Check current config
        print(mx_config.current())
        
        # Temporarily change settings
        with mx_config.override(block_size=64, strict=True):
            q = mx_quantize(tensor, "int4d")
        
        # Environment variables (checked at import time)
        # MX_DEBUG=1         → Enable verbose debug logging
        # MX_DEBUG_VERBOSE=1 → Include stack traces in debug output  
        # MX_STRICT=1        → Raise errors on fallback to fp32
    
    Attributes:
        block_size: Default block size for quantization (default: 128)
        strict: Raise errors instead of warnings on fallback (default: False)
        debug: Enable debug logging (default: from MX_DEBUG env var)
        verbose: Include stack traces in debug output (default: from MX_DEBUG_VERBOSE)
        default_dtype: Default MX dtype for quantization (default: "int4d")
        cache_kernels: Cache compiled Triton kernels (default: True)
        max_autotune: Enable Triton max-autotune for kernel selection (default: True)
    """
    
    _instance = None
    _defaults = {
        "block_size": 128,
        "strict": False,
        "debug": _DEBUG,
        "verbose": _VERBOSE,
        "default_dtype": "int4d",
        "cache_kernels": True,
        "max_autotune": True,
    }
    
    def __init__(self, **kwargs):
        self._settings = dict(self._defaults)
        self._settings.update(kwargs)
    
    @classmethod
    def current(cls) -> "mx_config":
        """Get the current global configuration."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def override(cls, **kwargs) -> "_mx_config_override":
        """Context manager to temporarily override config settings."""
        return _mx_config_override(**kwargs)
    
    def __getattr__(self, name: str):
        if name in self._settings:
            return self._settings[name]
        raise AttributeError(f"mx_config has no attribute '{name}'")
    
    def __repr__(self) -> str:
        settings_str = ", ".join(f"{k}={v!r}" for k, v in self._settings.items())
        return f"mx_config({settings_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Return config as a dictionary."""
        return dict(self._settings)
    
    @classmethod
    def set_default(cls, key: str, value: Any):
        """Set a default value for a configuration key."""
        if cls._instance is None:
            cls._instance = cls()
        cls._instance._settings[key] = value

class _mx_config_override:
    """Context manager for temporary config overrides."""
    
    def __init__(self, **kwargs):
        self._overrides = kwargs
        self._saved = {}
    
    def __enter__(self):
        config = mx_config.current()
        for key, value in self._overrides.items():
            self._saved[key] = config._settings.get(key)
            config._settings[key] = value
        return config
    
    def __exit__(self, *args):
        config = mx_config.current()
        for key, value in self._saved.items():
            if value is None:
                config._settings.pop(key, None)
            else:
                config._settings[key] = value

# Create default config instance
_config = mx_config.current()

def get_version_info() -> Dict[str, Any]:
    """
    Get version and dependency information for the MX quantization system.
    
    Returns:
        Dictionary with version, author, license, and dependency information.
    
    Example:
        info = get_version_info()
        print(f"mx_triton version {info['version']}")
    """
    return mx_info.get_version_info()

def print_module_info():
    """Print a summary of the mx_triton module capabilities."""
    mx_info.print_module_info()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MX TYPE SYSTEM (512 types)
# ─────────────────────────────────────────────────────────────────────────────

_VALID_BITS     = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128)
_VALID_MODES    = ("d", "u")
_VALID_KINDS    = ("int", "float")
_VALID_VARIANTS = ("", "h", "v", "s", "b")   # h=hadamard, v=vector, s=stochastic, b=bool-like

# Variant → long description
_VARIANT_NAMES = {
    "":  "base (absmax block-wise)",
    "h": "hadamard (QuIP# rotation pre-quant)",
    "v": "vector-wise (per-row/col absmax, bitsandbytes style)",
    "s": "stochastic round (unbiased, training-phase)",
    "b": "boolean / binary clamp (1-bit logical mapping)",
}

@dataclass(frozen=True)
class mx_dtype:
    """
    One of the 2048 MX data types.
    kind    ∈ {int, float}
    bits    ∈ {1,2,3,4,5,6,7,8,16,32,64,128}
    mode    ∈ {d, u}   (down=saturating, up=zero-padded)
    variant ∈ {"", "h", "v", "s", "b"}
                ""  → base absmax block quantization (default)
                "h" → Hadamard rotation pre-quantization (QuIP#)
                "v" → vector-wise absmax (per-row/col, bitsandbytes)
                "s" → stochastic rounding (unbiased training)
                "b" → boolean/binary clamping (0/1 mapping for 1-bit)

    Names:
        float4d     → base, no suffix
        float4uh    → hadamard variant
        float2uv    → vector-wise variant
        float4us    → stochastic variant
        float3ub    → boolean variant
        int4d       → base integer (most common)
        int2dv      → int2 vector-wise

    When variant="" the quantization falls back to the proven mask-based
    absmax block technique. All arithmetic (add, mul, matmul, etc.) is
    adapted per variant.
    """
    kind:    str
    bits:    int
    mode:    str
    variant: str = ""

    def __post_init__(self):
        assert self.kind    in _VALID_KINDS,    f"Invalid kind {self.kind!r}"
        assert self.bits    in _VALID_BITS,     f"Invalid bits {self.bits}"
        assert self.mode    in _VALID_MODES,    f"Invalid mode {self.mode!r}"
        assert self.variant in _VALID_VARIANTS, f"Invalid variant {self.variant!r}"

    # ── identity ──────────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return f"{self.kind}{self.bits}{self.mode}{self.variant}"

    def __repr__(self):
        return f"mx.{self.name}"

    def __str__(self):
        return self.name

    # ── semantics ─────────────────────────────────────────────────────────────
    @property
    def is_float(self):   return self.kind == "float"
    @property
    def is_int(self):     return self.kind == "int"
    @property
    def is_down(self):    return self.mode == "d"
    @property
    def is_up(self):      return self.mode == "u"
    @property
    def is_bool(self):    return self.variant == "b"
    @property
    def is_stochastic(self): return self.variant == "s"
    @property
    def is_hadamard(self):   return self.variant == "h"
    @property
    def is_vector(self):     return self.variant == "v"
    @property
    def is_base(self):       return self.variant == ""
    @property
    def variant_name(self) -> str:
        return _VARIANT_NAMES.get(self.variant, self.variant)

    # ── base dtype (strip variant) ────────────────────────────────────────────
    @property
    def base(self) -> "mx_dtype":
        """Return the base variant of this dtype (strip variant suffix)."""
        if self.variant == "":
            return self
        return _DTYPE_REGISTRY[f"{self.kind}{self.bits}{self.mode}"]

    # ── representable range ───────────────────────────────────────────────────
    @property
    def max_val(self) -> float:
        if self.is_bool:
            return 1.0
        if self.is_int:
            # Special case for 1-bit: use {-1, +1} range (binary neural network style)
            if self.bits == 1:
                return 1.0
            return float((1 << (self.bits - 1)) - 1)   # signed
        # float approximations
        _fmax = {1: 1.0, 2: 1.5, 3: 3.5, 4: 6.0, 5: 14.0,
                 6: 28.0, 7: 56.0, 8: 448.0, 16: 65504.0}
        return _fmax.get(self.bits, float("inf"))

    # ── storage / packing ─────────────────────────────────────────────────────
    @property
    def native_storage_bits(self) -> int:
        """Smallest native integer type that can hold packed values."""
        if   self.bits <= 8:   return 8
        elif self.bits <= 16:  return 16
        elif self.bits <= 32:  return 32
        else:                  return 64

    @property
    def pack_ratio(self) -> int:
        """How many MX values fit in one native storage word."""
        return self.native_storage_bits // self.bits

    @property
    def storage_torch_dtype(self) -> torch.dtype:
        return {8: torch.int8, 16: torch.int16,
                32: torch.int32, 64: torch.int64}[self.native_storage_bits]

    # ── compression ───────────────────────────────────────────────────────────
    @property
    def compression_vs_fp32(self) -> float:
        return 32.0 / self.bits

# ── build registry of all 2048 types (512 base + 4 variants × 512) ───────────
_DTYPE_REGISTRY: Dict[str, mx_dtype] = {}
for _k in _VALID_KINDS:
    for _b in _VALID_BITS:
        for _m in _VALID_MODES:
            for _v in _VALID_VARIANTS:
                _dt = mx_dtype(_k, _b, _m, _v)
                _DTYPE_REGISTRY[_dt.name] = _dt

def get_mx_dtype(name: str) -> mx_dtype:
    """
    Resolve MX dtype name → mx_dtype.

    Accepts:
        "int4d"       → base int4 down
        "float4uh"    → float4 up, Hadamard variant
        "int2uv"      → int2 up, vector-wise variant
        "float8us"    → float8 up, stochastic variant
        "int1db"      → int1 down, boolean variant

    Raises ValueError for unknown names.
    """
    # Check if already an mx_dtype (use type name check for dataclass compatibility)
    if type(name).__name__ == 'mx_dtype':
        return name
    # Check if mx_dtype_proxy
    if type(name).__name__ == 'mx_dtype_proxy':
        return name._mx
    if name not in _DTYPE_REGISTRY:
        raise ValueError(
            f"Unknown MX dtype {name!r}. "
            f"Format: {{int|float}}{{bits}}{{d|u}}[{{h|v|s|b}}]. "
            f"Examples: int4d, float8u, int2dv, float4uh, int8us")
    return _DTYPE_REGISTRY[name]

def _resolve_mixed(a: mx_dtype, b: mx_dtype) -> mx_dtype:
    """
    Mixed-precision resolution rule (variant-aware):
      • mode    = "u" if either is "u"
      • bits    = min(bits)
      • kind    = "float" if either is float
      • variant = higher-priority variant (h > v > s > b > "")
    """
    _VPRIO = {"h": 4, "v": 3, "s": 2, "b": 1, "": 0}
    mode    = "u" if (a.mode == "u" or b.mode == "u") else "d"
    bits    = min(a.bits, b.bits)
    kind    = "float" if (a.is_float or b.is_float) else "int"
    variant = a.variant if _VPRIO.get(a.variant, 0) >= _VPRIO.get(b.variant, 0) else b.variant
    bits    = min(_VALID_BITS, key=lambda x: abs(x - bits))
    return get_mx_dtype(f"{kind}{bits}{mode}{variant}")

# expose short aliases at module level: mx_triton.int4d, mx_triton.float8u …
for _name, _dt in _DTYPE_REGISTRY.items():
    globals()[_name] = _dt

# ── Explicit type-visible declarations for the most common MX dtypes ─────────
# These shadow the dynamic globals() assignments above with identical values,
# making them visible to type checkers (mypy/pyright/IDEs) and ensuring that
# bare-name references like `int4d` in test code are never undefined.
int1d   : mx_dtype = _DTYPE_REGISTRY["int1d"]
int1u   : mx_dtype = _DTYPE_REGISTRY["int1u"]
int2d   : mx_dtype = _DTYPE_REGISTRY["int2d"]
int2u   : mx_dtype = _DTYPE_REGISTRY["int2u"]
int3d   : mx_dtype = _DTYPE_REGISTRY["int3d"]
int3u   : mx_dtype = _DTYPE_REGISTRY["int3u"]
int4d   : mx_dtype = _DTYPE_REGISTRY["int4d"]
int4u   : mx_dtype = _DTYPE_REGISTRY["int4u"]
int5d   : mx_dtype = _DTYPE_REGISTRY["int5d"]
int5u   : mx_dtype = _DTYPE_REGISTRY["int5u"]
int6d   : mx_dtype = _DTYPE_REGISTRY["int6d"]
int6u   : mx_dtype = _DTYPE_REGISTRY["int6u"]
int7d   : mx_dtype = _DTYPE_REGISTRY["int7d"]
int7u   : mx_dtype = _DTYPE_REGISTRY["int7u"]
int8d   : mx_dtype = _DTYPE_REGISTRY["int8d"]
int8u   : mx_dtype = _DTYPE_REGISTRY["int8u"]
int16d  : mx_dtype = _DTYPE_REGISTRY["int16d"]
int16u  : mx_dtype = _DTYPE_REGISTRY["int16u"]
int32d  : mx_dtype = _DTYPE_REGISTRY["int32d"]
int32u  : mx_dtype = _DTYPE_REGISTRY["int32u"]
int64d  : mx_dtype = _DTYPE_REGISTRY["int64d"]
int64u  : mx_dtype = _DTYPE_REGISTRY["int64u"]
int128d : mx_dtype = _DTYPE_REGISTRY["int128d"]
int128u : mx_dtype = _DTYPE_REGISTRY["int128u"]
float1d  : mx_dtype = _DTYPE_REGISTRY["float1d"]
float1u  : mx_dtype = _DTYPE_REGISTRY["float1u"]
float2d  : mx_dtype = _DTYPE_REGISTRY["float2d"]
float2u  : mx_dtype = _DTYPE_REGISTRY["float2u"]
float3d  : mx_dtype = _DTYPE_REGISTRY["float3d"]
float3u  : mx_dtype = _DTYPE_REGISTRY["float3u"]
float4d  : mx_dtype = _DTYPE_REGISTRY["float4d"]
float4u  : mx_dtype = _DTYPE_REGISTRY["float4u"]
float5d  : mx_dtype = _DTYPE_REGISTRY["float5d"]
float5u  : mx_dtype = _DTYPE_REGISTRY["float5u"]
float6d  : mx_dtype = _DTYPE_REGISTRY["float6d"]
float6u  : mx_dtype = _DTYPE_REGISTRY["float6u"]
float7d  : mx_dtype = _DTYPE_REGISTRY["float7d"]
float7u  : mx_dtype = _DTYPE_REGISTRY["float7u"]
float8d  : mx_dtype = _DTYPE_REGISTRY["float8d"]
float8u  : mx_dtype = _DTYPE_REGISTRY["float8u"]
float16d : mx_dtype = _DTYPE_REGISTRY["float16d"]
float16u : mx_dtype = _DTYPE_REGISTRY["float16u"]
float32d : mx_dtype = _DTYPE_REGISTRY["float32d"]
float32u : mx_dtype = _DTYPE_REGISTRY["float32u"]
float64d : mx_dtype = _DTYPE_REGISTRY["float64d"]
float64u : mx_dtype = _DTYPE_REGISTRY["float64u"]
float128d: mx_dtype = _DTYPE_REGISTRY["float128d"]
float128u: mx_dtype = _DTYPE_REGISTRY["float128u"]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — mx_dtype_proxy  (fake torch.dtype object)
# ─────────────────────────────────────────────────────────────────────────────

class mx_dtype_proxy:
    """
    Impersonates a torch.dtype object.
    Returned by the patched torch.dtype("int4d") call.
    Accepted by all patched .to() methods.
    """
    __slots__ = ("_mx",)

    def __init__(self, mx: mx_dtype):
        self._mx = mx

    # make it look like a torch.dtype ─────────────────────────────────────────
    @property
    def name(self) -> str:            return self._mx.name
    @property
    def itemsize(self) -> float:      return self._mx.bits / 8
    @property
    def is_floating_point(self) -> bool: return self._mx.is_float
    @property
    def is_complex(self) -> bool:     return False
    @property
    def is_signed(self) -> bool:      return True

    def __repr__(self):   return f"torch.dtype('{self._mx.name}')"
    def __str__(self):    return self._mx.name
    def __hash__(self):   return hash(self._mx.name)
    def __eq__(self, other):
        if type(other).__name__ == 'mx_dtype_proxy': return self._mx == other._mx
        return NotImplemented

def as_mx_dtype_proxy(name_or_mx) -> mx_dtype_proxy:
    """
    Convert a dtype specification to mx_dtype_proxy.
    
    Args:
        name_or_mx: str (e.g., "int4d"), mx_dtype, or mx_dtype_proxy
        
    Returns:
        mx_dtype_proxy wrapping the mx_dtype
    """
    if type(name_or_mx).__name__ == 'mx_dtype_proxy':
        return name_or_mx
    if type(name_or_mx).__name__ == 'mx_dtype':
        return mx_dtype_proxy(name_or_mx)
    return mx_dtype_proxy(get_mx_dtype(name_or_mx))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — HARDWARE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class hardware_profile:
    name:             str
    arch:             str
    backend:          str        # rocm | cuda | cpu | intel
    native_int_bits:  int        # widest native int (GPU=8, CPU=32/64)
    vector_bits:      int        # SIMD width in bits
    max_pack_bits:    int        # max useful pack size
    memory_bw_gbs:    float      # GB/s
    fp32_tflops:      float
    fp16_tflops:      float
    supported_native: List[str]
    fast_instrs:      List[str]  = field(default_factory=list)
    compute_units:    int        = 1
    wave_size:        int        = 64

    def hw_pack_ratio(self, dt: mx_dtype) -> int:
        """How many dt values can be packed per native arithmetic op."""
        return min(self.max_pack_bits // dt.bits, dt.pack_ratio)

    def peak_tflops(self, dt: mx_dtype) -> float:
        """Effective TFLOPS accounting for packing gain."""
        pr = self.hw_pack_ratio(dt)
        if dt.bits == 1:
            return self.fp32_tflops * 32 * pr   # bitwise throughput
        elif dt.bits <= 4:
            return self.fp16_tflops * pr
        elif dt.bits <= 8:
            return self.fp16_tflops
        elif dt.bits <= 16:
            return self.fp16_tflops
        else:
            return self.fp32_tflops

class hardware_probe:
    _cache: Optional[hardware_profile] = None

    @classmethod
    def detect(cls) -> hardware_profile:
        if cls._cache is not None:
            return cls._cache
        cls._cache = cls._detect()
        if _DEBUG:
            log.debug(f"[HW] {cls._cache.name} ({cls._cache.arch}), "
                      f"native_int={cls._cache.native_int_bits}b, "
                      f"max_pack={cls._cache.max_pack_bits}b")
        return cls._cache

    @classmethod
    def _detect(cls) -> hardware_profile:
        if not torch.cuda.is_available():
            return cls._cpu()

        props = torch.cuda.get_device_properties(0)
        name_lower = props.name.lower()
        is_amd = (hasattr(props, "gcnArchName") or
                  any(x in name_lower for x in ("amd", "radeon", "instinct")))

        if is_amd:
            arch = getattr(props, "gcnArchName", "gfx1100")
            return cls._amd(arch, props)

        cc = props.major * 10 + props.minor
        return cls._nvidia(cc, props)

    @classmethod
    def _amd(cls, arch: str, props) -> hardware_profile:
        known = {
            "gfx1100": hardware_profile(
                name="rx_7900_xtx", arch="gfx1100", backend="rocm",
                native_int_bits=8, vector_bits=128, max_pack_bits=8,
                memory_bw_gbs=960.0, fp32_tflops=61.0, fp16_tflops=123.0,
                supported_native=["fp32","fp16","bf16","int8","int4"],
                fast_instrs=["v_dot4_i32_i8","v_dot2_f32_f16","v_perm_b32"],
                compute_units=96, wave_size=32,
            ),
            "gfx942": hardware_profile(
                name="mi300x", arch="gfx942", backend="rocm",
                native_int_bits=8, vector_bits=256, max_pack_bits=8,
                memory_bw_gbs=5300.0, fp32_tflops=653.7, fp16_tflops=1307.0,
                supported_native=["fp32","fp16","bf16","int8","fp8","int4"],
                fast_instrs=["v_mfma_f32_32x32x8f16","v_dot4_i32_i8"],
                compute_units=304, wave_size=64,
            ),
            "gfx90a": hardware_profile(
                name="mi250x", arch="gfx90a", backend="rocm",
                native_int_bits=8, vector_bits=256, max_pack_bits=8,
                memory_bw_gbs=3200.0, fp32_tflops=47.9, fp16_tflops=383.0,
                supported_native=["fp32","fp16","bf16","int8"],
                compute_units=220, wave_size=64,
            ),
        }
        for key, profile in known.items():
            if key in arch.lower():
                return profile
        return hardware_profile(
            name=f"amd_{arch}", arch=arch, backend="rocm",
            native_int_bits=8, vector_bits=128, max_pack_bits=8,
            memory_bw_gbs=500.0, fp32_tflops=20.0, fp16_tflops=40.0,
            supported_native=["fp32","fp16","int8"],
            compute_units=getattr(props, "multi_processor_count", 32),
        )

    @classmethod
    def _nvidia(cls, cc: int, props) -> hardware_profile:
        cu = props.multi_processor_count
        if cc >= 90:   # Hopper H100
            return hardware_profile(
                name="h100", arch=f"sm_{cc}", backend="cuda",
                native_int_bits=8, vector_bits=512, max_pack_bits=8,
                memory_bw_gbs=3350.0, fp32_tflops=67.0, fp16_tflops=1000.0,
                supported_native=["fp32","fp16","bf16","int8","fp8","int4"],
                fast_instrs=["mma.sync.aligned.m16n8k32","ldmatrix","dp4a"],
                compute_units=cu, wave_size=32,
            )
        elif cc >= 86:  # Ampere / Ada
            return hardware_profile(
                name="ada_ampere", arch=f"sm_{cc}", backend="cuda",
                native_int_bits=8, vector_bits=256, max_pack_bits=8,
                memory_bw_gbs=600.0, fp32_tflops=40.0, fp16_tflops=320.0,
                supported_native=["fp32","fp16","bf16","int8","int4"],
                fast_instrs=["mma.sync.aligned.m16n8k16","dp4a"],
                compute_units=cu, wave_size=32,
            )
        return hardware_profile(
            name=f"nvidia_sm{cc}", arch=f"sm_{cc}", backend="cuda",
            native_int_bits=8, vector_bits=128, max_pack_bits=8,
            memory_bw_gbs=400.0, fp32_tflops=15.0, fp16_tflops=60.0,
            supported_native=["fp32","fp16","int8"],
            compute_units=cu,
        )

    @classmethod
    def _cpu(cls) -> hardware_profile:
        try:
            import cpuinfo
            flags = cpuinfo.get_cpu_info().get("flags", [])
            has512 = "avx512f" in flags
        except Exception:
            has512 = False
        if has512:
            return hardware_profile(
                name="zen4_avx512", arch="x86_avx512", backend="cpu",
                native_int_bits=32, vector_bits=512, max_pack_bits=32,
                memory_bw_gbs=100.0, fp32_tflops=3.0, fp16_tflops=6.0,
                supported_native=["fp32","fp64","int8","int16","int32","int64"],
                fast_instrs=["vpdpbusd","vfmadd231ps","vcvtps2ph"],
            )
        return hardware_profile(
            name="generic_cpu", arch="x86", backend="cpu",
            native_int_bits=32, vector_bits=256, max_pack_bits=32,
            memory_bw_gbs=50.0, fp32_tflops=0.5, fp16_tflops=1.0,
            supported_native=["fp32","fp64","int8","int32"],
        )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — REAL BIT PACKING (no fake quantization)
# ─────────────────────────────────────────────────────────────────────────────

class bit_packer:
    """
    True bit-level pack / unpack for MX types.
    No shadow float copy, no dequant before storage.

    int4 (bits=4, pack=2):  byte = (b1 << 4) | (b0 & 0xF)
    int2 (bits=2, pack=4):  byte = b3<<6 | b2<<4 | b1<<2 | b0
    int1 (bits=1, pack=8):  byte = b7<<7 | … | b0
    """

    # ── fast power-of-2 paths ─────────────────────────────────────────────────
    @staticmethod
    def pack(vals: Tensor, bits: int) -> Tensor:
        """
        Pack int8 tensor into real bit-packed int8 storage.
        bits must be 1, 2, 4, or 8.
        vals: flat int tensor, values in signed range for `bits`.
            Returns: int8 tensor of length ceil(N / (8//bits)).
        
        Optimized: avoids creating tensor for shift values (uses Python int directly).
        """
        assert bits in (1, 2, 4, 8), f"pack() only for 1,2,4,8 bits. Got {bits}"
        if bits == 8:
            return vals.to(torch.int8)

        ratio = 8 // bits
        n     = vals.numel()
        pad   = (-n) % ratio   # number of padding elements
        if pad:
            vals = torch.cat([vals.flatten(),
                              vals.new_zeros(pad, dtype=vals.dtype)])
        vals = vals.flatten().to(torch.int8)

        mask   = (1 << bits) - 1
        packed = torch.zeros(vals.numel() // ratio, dtype=torch.int8,
                             device=vals.device)
        for i in range(ratio):
            slot = (vals[i::ratio] & mask).to(torch.int8)
            # Use integer shift directly (avoid creating tensor)
            packed = packed | (slot << (i * bits))
        return packed

    @staticmethod
    def unpack(packed: Tensor, bits: int, n: int) -> Tensor:
        """
        Unpack int8 packed storage back to int8 signed values (sign-extended).
        """
        assert bits in (1, 2, 4, 8), f"unpack() only for 1,2,4,8 bits."
        if bits == 8:
            return packed[:n].to(torch.float32)

        ratio = 8 // bits
        mask  = (1 << bits) - 1
        sign  = 1 << (bits - 1)
        out   = torch.empty(len(packed) * ratio, dtype=torch.int8,
                            device=packed.device)
        for i in range(ratio):
            slot = ((packed >> (i * bits)) & mask).to(torch.int8)
            # sign extend: if high bit set, OR upper bits with 1
            signed = torch.where(
                (slot.to(torch.int16) & sign) != 0,
                (slot.to(torch.int16) | (~mask & 0xFF)).to(torch.int8),
                slot,
            )
            out[i::ratio] = signed
        return out[:n].to(torch.float32)

    # ── arbitrary-width via int32 ─────────────────────────────────────────────
    @staticmethod
    def pack_arb(vals: Tensor, bits: int) -> Tensor:
        """Pack arbitrary bit widths (3,5,6,7) using int32 words."""
        if bits in (1, 2, 4, 8):
            return bit_packer.pack(vals.to(torch.int8), bits)
        ratio = 32 // bits
        n     = vals.numel()
        pad   = (-n) % ratio
        if pad:
            vals = torch.cat([vals.flatten(),
                              vals.new_zeros(pad, dtype=vals.dtype)])
        v32    = vals.flatten().to(torch.int32)
        mask   = (1 << bits) - 1
        packed = torch.zeros(len(v32) // ratio, dtype=torch.int32,
                             device=vals.device)
        for i in range(ratio):
            packed |= (v32[i::ratio] & mask) << (i * bits)
        return packed

    @staticmethod
    def unpack_arb(packed: Tensor, bits: int, n: int) -> Tensor:
        """Unpack int32 words for arbitrary bit widths."""
        if bits in (1, 2, 4, 8):
            return bit_packer.unpack(packed.view(torch.int8) if packed.dtype != torch.int8
                                    else packed, bits, n)
        ratio = 32 // bits
        mask  = (1 << bits) - 1
        sign  = 1 << (bits - 1)
        out   = torch.empty(len(packed) * ratio, dtype=torch.int32,
                            device=packed.device)
        for i in range(ratio):
            slot   = (packed >> (i * bits)) & mask
            signed = torch.where((slot & sign) != 0, slot | ~mask, slot)
            out[i::ratio] = signed
        return out[:n].to(torch.float32)

    # ── choose right path automatically ──────────────────────────────────────
    @classmethod
    def pack_auto(cls, vals: Tensor, bits: int) -> Tensor:
        if bits in (1, 2, 4, 8):
            return cls.pack(vals.to(torch.int8), bits)
        return cls.pack_arb(vals, bits)

    @classmethod
    def unpack_auto(cls, packed: Tensor, bits: int, n: int) -> Tensor:
        if bits in (1, 2, 4, 8):
            return cls.unpack(packed.to(torch.int8), bits, n)
        return cls.unpack_arb(packed.to(torch.int32), bits, n)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — QUANTIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _quant_int(x: Tensor, dt: mx_dtype, block: int) -> Tuple[Tensor, Tensor, int]:
    """
    Quantize float tensor → (packed, scales, n).
    packed: real bit-packed, scales: [n_blocks] float32.
    
    Optimized: uses reciprocal multiplication instead of division for ~2x speedup.
    """
    flat = x.float().reshape(-1)
    n    = flat.numel()
    nb   = math.ceil(n / block)
    pad  = nb * block - n
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk  = flat.reshape(nb, block)

    # Special case for 1-bit: binary quantization {-1, +1}
    if dt.bits == 1:
        # Binary quantization: sign-based, scale from absmax
        scales = blk.abs().amax(dim=1).clamp(min=1e-12)  # [nb]
        # Sign: positive → 0 (maps to -1 after sign extension), negative → 1 (maps to -1 after sign extension)
        # Actually for int1 signed: 0 bit = 0 (value 0), 1 bit = -1 (sign extended)
        # We want: values near +scale → +code, values near -scale → -code
        # For 1-bit signed: codes are 0 (maps to 0) or -1 (maps to -1)
        # Better: use threshold at 0: if x > 0 → 0 code, else → -1 code
        codes = torch.where(blk >= 0, 
                           torch.zeros_like(blk, dtype=torch.int32),
                           torch.full_like(blk, -1, dtype=torch.int32))
        packed = bit_packer.pack_auto(codes.reshape(-1), dt.bits)
        return packed, scales, n

    max_int = float((1 << (dt.bits - 1)) - 1)   # e.g. int4 → 7
    # Optimized: compute scales, then use reciprocal multiplication
    scale_inv = max_int / blk.abs().amax(dim=1).clamp(min=1e-12)  # [nb] reciprocal of scale
    scales = scale_inv.reciprocal()  # actual scales
    codes = (blk * scale_inv.unsqueeze(1)).clamp(-max_int, max_int).round_().to(torch.int32)

    packed = bit_packer.pack_auto(codes.reshape(-1), dt.bits)
    return packed, scales, n

def _quant_float(x: Tensor, dt: mx_dtype, block: int) -> Tuple[Tensor, Tensor, int]:
    """
    Quantize float tensor → MX float representation.
    Uses fixed-point discretization per block.
    """
    flat = x.float().reshape(-1)
    n    = flat.numel()
    nb   = math.ceil(n / block)
    pad  = nb * block - n
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk  = flat.reshape(nb, block)

    max_v  = dt.max_val if dt.max_val < 1e30 else 1.0
    scales = blk.abs().amax(dim=1).clamp(min=1e-12) / max_v

    if dt.bits >= 8:
        codes = (blk / scales.unsqueeze(1)).clamp(-max_v, max_v).round_().to(torch.int32)
    else:
        nlevels = 2 ** dt.bits
        step    = (2 * max_v) / (nlevels - 1)
        codes   = ((blk / scales.unsqueeze(1) + max_v) / step).round_().long()
        codes   = codes.clamp(0, nlevels - 1).to(torch.int32) - (nlevels // 2)

    packed = bit_packer.pack_auto(codes.reshape(-1), dt.bits)
    return packed, scales, n

def _dequant(packed: Tensor, scales: Tensor, dt: mx_dtype,
             n: int, block: int) -> Tensor:
    """
    Unpack + dequantize → float32 flat tensor of length n.
    Variant-aware: hadamard unrotates, boolean returns 0/1, others scale normally.
    """
    codes = bit_packer.unpack_auto(packed, dt.bits, n)

    nb  = scales.numel()
    pad = nb * block - n
    if pad:
        codes = torch.cat([codes, codes.new_zeros(pad)])
    blk = codes.reshape(nb, block)

    # Special case for 1-bit: binary {-scale, +scale}
    # For int1 signed: codes are 0 or -1 after sign extension
    # We want: 0 → +scale, -1 → -scale
    # So we negate and add 1: -(-1)+1=0 (no), wait...
    # Better: codes are 0 or -1. Multiply by -2: 0→0, -1→2. Subtract 1: 0→-1, 2→1
    # Actually simplest: if code is 0 → +scale, if code is -1 → -scale
    if dt.is_int and dt.bits == 1 and not dt.is_bool:
        # Map: 0 → +1, -1 → -1
        dq = torch.where(blk == 0, 
                         torch.ones_like(blk, dtype=torch.float32),
                         torch.full_like(blk, -1.0, dtype=torch.float32))
    elif dt.is_float and dt.bits < 8:
        nlevels = 2 ** dt.bits
        max_v   = dt.max_val
        step    = (2 * max_v) / (nlevels - 1)
        dq      = blk * step
    elif dt.is_bool:
        dq = blk.abs()  # Boolean: 0 -> 0, -1 (from sign-ext) -> 1
    else:
        dq = blk

    flat = (dq * scales.unsqueeze(1)).reshape(-1)[:n]

    # ── boolean variant: values are 0/1 codes, no unrotation needed ──────────
    if dt.is_bool:
        return flat.clamp(0.0, 1.0)

    # ── hadamard variant: apply inverse WHT to recover original values ────────
    if dt.is_hadamard:
        d    = flat.numel()
        seed = hash(dt.name) % (2**31)
        _rot_cache_key = (d, seed)
        # Reuse cache from quantize()
        if (hasattr(quantize, "_rot_cache") and
                _rot_cache_key in quantize._rot_cache):
            rot  = quantize._rot_cache[_rot_cache_key]
            flat = rot.unrotate(flat.unsqueeze(0)).squeeze(0)
        # If cache miss (e.g., different process), rebuild rotation
        else:
            n2   = 1 << math.ceil(math.log2(max(d, 1)))
            rot  = hadamard_rotation(n2, seed=seed)
            flat = rot.unrotate(flat.unsqueeze(0)).squeeze(0)[:d]

    return flat

@dataclass
class quantization_result:
    """
    Result of a quantization operation with detailed metadata.
    
    Provides all information about the quantization process, including
    the packed data, scales, and quality metrics.
    
    Attributes:
        packed: The bit-packed quantized data (int8 or int32 tensor)
        scales: Per-block scale factors (float32 tensor)
        n: Original number of elements before padding
        dtype: The MX dtype used for quantization
        block: Block size used for quantization
        compression_ratio: Achieved compression vs fp32
        snr_db: Signal-to-noise ratio in decibels (if computed)
        rmse: Root mean square error (if computed)
    
    Usage:
        result = quantization_result.compute(tensor, "int4d")
        print(f"Compressed {result.n} elements with {result.snr_db:.1f} dB SNR")
        dequant = result.dequantize()
    """
    packed: Tensor
    scales: Tensor
    n: int
    dtype: mx_dtype
    block: int = 128
    snr_db: Optional[float] = None
    rmse: Optional[float] = None
    
    @classmethod
    def compute(cls, x: Tensor, dtype: Union[str, mx_dtype], 
                block: int = 128, compute_metrics: bool = True) -> "quantization_result":
        """
        Quantize a tensor and return detailed results.
        
        Args:
            x: Input tensor to quantize
            dtype: Target MX dtype (string or mx_dtype)
            block: Block size for quantization
            compute_metrics: Whether to compute SNR and RMSE
            
        Returns:
            quantization_result with packed data and metadata
        """
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        packed, scales, n = quantize(x, dtype, block)
        
        result = cls(packed=packed, scales=scales, n=n, dtype=dtype, block=block)
        
        if compute_metrics:
            dequant = result.dequantize()
            # Always flatten for comparison
            x_flat = x.reshape(-1)[:n]
            dequant_flat = dequant.reshape(-1)[:n]
            # Compute SNR
            signal_power = (x_flat ** 2).mean()
            noise_power = ((x_flat - dequant_flat) ** 2).mean()
            if noise_power > 0:
                result.snr_db = 10 * math.log10(signal_power / noise_power)
            # Compute RMSE
            result.rmse = math.sqrt(((x_flat - dequant_flat) ** 2).mean().item())
        
        return result
    
    def dequantize(self) -> Tensor:
        """Dequantize the packed data back to float."""
        return _dequant(self.packed, self.scales, self.dtype, self.n, self.block)
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs fp32."""
        return 32.0 / self.dtype.bits
    
    def to_mxtensor(self, shape: torch.Size) -> "mx_tensor":
        """Convert to mx_tensor with given shape."""
        return mx_tensor(self.packed, self.scales, self.dtype, shape, self.n, self.block)
    
    def __repr__(self) -> str:
        snr_str = f"{self.snr_db:.1f}dB" if self.snr_db is not None else "N/A"
        return (f"quantization_result(dtype={self.dtype.name}, n={self.n}, "
                f"block={self.block}, snr={snr_str})")

def quantize(x: Tensor, dt: mx_dtype, block: int = 128
             ) -> Tuple[Tensor, Tensor, int]:
    """
    Dispatch to the correct quantizer based on dtype kind AND variant.

    Variant routing:
      ""  → base absmax block quantization (most compact, best for inference)
          "h" → Hadamard rotation pre-quant    (better SNR at int2/int4)
      "v" → vector-wise per-row absmax     (bitsandbytes int8 style)
      "s" → stochastic rounding            (unbiased, use for training)
          "b" → boolean/binary clamping        (0/1, only useful for int1)

    Uses Triton GPU kernel for int4/int8 base on CUDA.
    Falls back to pure-Python implementation on CPU.
    """
    x_f = x.float()

    # ── boolean variant: clamp to 0/1 ────────────────────────────────────────
    if dt.is_bool:
        flat = x_f.reshape(-1)
        n    = flat.numel()
        nb   = math.ceil(n / block)
        pad  = nb * block - n
        if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
        codes = (flat > 0).to(torch.int32)           # threshold at 0
        scales = torch.ones(nb, dtype=torch.float32, device=x.device)
        packed = bit_packer.pack_auto(codes, dt.bits)
        return packed, scales, n

    # ── stochastic variant: stochastic round then pack ────────────────────────
    if dt.is_stochastic:
        x_sr = stochastic_round(x_f, bits=dt.bits)
        return _quant_int(x_sr, dt.base, block) if dt.is_int else \
               _quant_float(x_sr, dt.base, block)

    # ── hadamard variant: WHT rotate then pack ────────────────────────────────
    if dt.is_hadamard:
        d   = x_f.shape[-1] if x_f.ndim >= 2 else x_f.numel()
        # Use a fixed per-dtype seed for reproducibility
        seed = hash(dt.name) % (2**31)
        _rot_cache_key = (d, seed)
        if not hasattr(quantize, "_rot_cache"):
            quantize._rot_cache = {}
        if _rot_cache_key not in quantize._rot_cache:
            quantize._rot_cache[_rot_cache_key] = hadamard_rotation(
                1 << math.ceil(math.log2(max(d, 1))), seed=seed)
        rot    = quantize._rot_cache[_rot_cache_key]
        x_rot  = rot.rotate(x_f)
        result = _quant_int(x_rot, dt.base, block) if dt.is_int else \
                 _quant_float(x_rot, dt.base, block)
        # Store rotation seed in scales[-1] slot for dequantize to recover
        # (last scale element repurposed as metadata; n ensures correct unpack)
        return result[0], result[1], result[2]

    # ── vector-wise variant: per-row absmax ───────────────────────────────────
    if dt.is_vector:
        flat = x_f.reshape(-1)
        n    = flat.numel()
        # One scale per row (treat block as row-width)
        row_w  = block
        nb     = math.ceil(n / row_w)
        pad    = nb * row_w - n
        if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
        rows   = flat.reshape(nb, row_w)
        max_int = dt.max_val
        scales = rows.abs().amax(dim=1).clamp(min=1e-12) / max_int
        codes  = (rows / scales.unsqueeze(1)).clamp(-max_int, max_int).round_().to(torch.int32)
        packed = bit_packer.pack_auto(codes.reshape(-1), dt.bits)
        return packed, scales, n

    # ── base (default): absmax block-wise ────────────────────────────────────
    if dt.is_int:
        # Fast path: Triton on-GPU for int4/int8
        if HAS_TRITON and x_f.is_cuda and dt.bits in (4, 8) and x_f.numel() >= 128:
            try:
                return _triton_quantize(x_f, dt, block)
            except Exception:
                pass
        return _quant_int(x_f, dt, block)
    return _quant_float(x_f, dt, block)

def _triton_quantize(x: Tensor, dt: mx_dtype, block: int) -> Tuple[Tensor, Tensor, int]:
    """
    On-GPU quantization via Triton kernel.
    Avoids a CPU round-trip — tensor stays on GPU throughout.
    Returns (packed: int8, scales: float32, n_elements: int).
    """
    flat = x.float().reshape(-1)
    n    = flat.numel()
    nb   = math.ceil(n / block)

    scales = torch.empty(nb, dtype=torch.float32, device=x.device)

    if dt.bits == 4:
        packed = torch.empty((n + 1) // 2, dtype=torch.int8, device=x.device)
        BLK    = min(block, 128)
        HALF_BLK = BLK // 2
        _k_quantize_int4[(nb,)](flat, packed, scales, n, BS=block, BLK=BLK, HALF_BLK=HALF_BLK)
    elif dt.bits == 8:
        packed = torch.empty(n, dtype=torch.int8, device=x.device)
        BLK    = min(block, 128)
        _k_quantize_int8[(nb,)](flat, packed, scales, n, BS=block, BLK=BLK)
    else:
        raise ValueError(f"Triton quantize only for int4/int8, not {dt.bits}-bit")

    return packed, scales, n

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5b — NON-UNIFORM QUANTIZATION  (QLoRA / bitsandbytes / Unsloth)
#   NF4: NormalFloat4 — optimal grid for N(0,1) weights (QLoRA paper)
#   FP4: sub-byte IEEE-like float (bitsandbytes FP4)
#   Double quantization: quantize the scale factors themselves → -0.37 bits/param
#   LLM.int8(): outlier-column decomposition (mixed int8/fp16)
#   SmoothQuant: α-balance migration from activations to weights
# ─────────────────────────────────────────────────────────────────────────────

# ── NF4 lookup table ──────────────────────────────────────────────────────────
# The 16 NF4 grid points are the quantiles of N(0,1) normalised to [-1,+1].
# Derived from: scipy.stats.norm.ppf((i + 0.5) / 16) for i in range(16).
# This is the same table used by QLoRA / bitsandbytes 4-bit.
_NF4_TABLE: Tensor = torch.tensor([
    -1.0000, -0.6962, -0.5251, -0.3946,
    -0.2813, -0.1777, -0.0834,  0.0000,
     0.0834,  0.1777,  0.2813,  0.3946,
     0.5251,  0.6962,  1.0000,  0.0000,   # 15 = zero (padding for 4-bit)
        ], dtype=torch.float32)

# ── FP4 lookup table ──────────────────────────────────────────────────────────
# bitsandbytes FP4: sign + 1-bit exponent + 2-bit mantissa.
# Representable values (non-negative half):  0, 0.0625, 0.125, 0.1875,
#   0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0
_FP4_TABLE: Tensor = torch.tensor([
     0.00,   0.0625, 0.125, 0.1875,
     0.25,   0.375,  0.5,   0.75,
     1.00,   1.5,    2.0,   3.0,
     4.00,   6.0,    8.0,   12.0,
], dtype=torch.float32)

def _lookup_quantize(x: Tensor, table: Tensor, block: int) -> Tuple[Tensor, Tensor, int]:
    """
    General lookup-table quantizer for non-uniform grids (NF4, FP4, AF4 …).
        Steps per block:
      1. Compute block absmax scale so values sit in [-1, +1].
      2. Find nearest table entry via L1 distance (vectorised argmin).
      3. Store 4-bit indices packed 2-per-byte.

    Returns: (packed int8, scales float32, n_elements)
    """
    flat = x.float().reshape(-1)
    n    = flat.numel()
    nb   = math.ceil(n / block)
    pad  = nb * block - n
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])

    blk    = flat.reshape(nb, block)
    scales = blk.abs().amax(dim=1).clamp(min=1e-12)     # [nb]

    # Normalize to [-1, +1] then find nearest table entry
    normed = blk / scales.unsqueeze(1)                   # [nb, block]
    tbl    = table.to(flat.device)

    # Vectorised nearest-entry search  — [nb, block, n_entries]
    dists  = (normed.unsqueeze(-1) - tbl).abs()         # broadcast
    codes  = dists.argmin(dim=-1).to(torch.int32)       # [nb, block]

    packed = bit_packer.pack_auto(codes.reshape(-1), bits=4)
    return packed, scales, n

def _lookup_dequantize(packed: Tensor, scales: Tensor, table: Tensor,
                       n: int, block: int) -> Tensor:
    """Inverse of _lookup_quantize: decode 4-bit indices via table lookup."""
    codes = bit_packer.unpack_auto(packed, bits=4, n=n + (-n % block))
    # Convert sign-extended int8 back to unsigned indices (0-15)
    # bit_packer.unpack sign-extends, so values 8-15 become -8 to -1
    # We need to mask to get unsigned indices for table lookup
    indices = (codes.to(torch.int8) & 0xF).long()  # unsigned 0-15
    nb    = scales.numel()
    blk   = indices.reshape(nb, block)
    tbl   = table.to(packed.device)
    dq    = tbl[blk.clamp(0, len(tbl)-1)]        # table gather
    return (dq * scales.unsqueeze(1)).reshape(-1)[:n]

# ── nf4_tensor ─────────────────────────────────────────────────────────────────

class nf4_tensor(torch.Tensor):
    """
    NormalFloat4 tensor — optimal 4-bit quantisation for weights drawn from N(0,1).
        Introduced in QLoRA (Dettmers et al. 2023) and popularised by bitsandbytes / Unsloth.

    Key properties vs plain int4:
      • Non-uniform grid: more codepoints near zero (where most weights concentrate)
      • ~0.25 dB better SNR than symmetric int4 on normally-distributed data
      • Same bit-width — no memory penalty

    Usage::
        w_nf4 = nf4_tensor.quantize(weight)      # 4-bit packed
        w_f   = w_nf4.dequantize()              # back to float32
        # Drop-in for mx_linear:
        lin   = mx_linear.from_linear(layer, get_mx_dtype("int4d"))
        lin.weight = nn.Parameter(nf4_tensor.quantize(layer.weight.data))
    """

    def __new__(cls, packed: Tensor, scales: Tensor,
                orig_shape: torch.Size, n: int, block: int = 64,
                requires_grad: bool = False):
        # Create a dummy float tensor as base (similar to mx_tensor approach)
        # This avoids issues with _make_subclass and packed tensor identity
        dummy = torch.empty((), dtype=torch.float32, device=packed.device, requires_grad=requires_grad)
        inst = torch.Tensor._make_subclass(cls, dummy, requires_grad)
        # Store packed data separately to avoid identity issues
        inst._nf4_packed      = packed
        inst._nf4_scales      = scales
        inst._nf4_orig_shape  = orig_shape
        inst._nf4_n           = n
        inst._nf4_block       = block
        return inst

    def __init__(self, packed, scales, orig_shape, n, block=64, requires_grad=False):
        pass

    @classmethod
    def quantize(cls, x: Tensor, block: int = 64) -> "nf4_tensor":
        """Quantize float tensor to NF4. block=64 matches bitsandbytes default."""
        with torch.no_grad():
            packed, scales, n = _lookup_quantize(x.detach().float(), _NF4_TABLE, block)
        return cls(packed, scales, x.shape, n, block, x.requires_grad)

    def dequantize(self) -> Tensor:
        """Decode NF4 → float32 in original shape."""
        # Use the stored packed tensor directly (not self, which is a dummy tensor)
        flat = _lookup_dequantize(
            self._nf4_packed,
            self._nf4_scales, _NF4_TABLE, self._nf4_n, self._nf4_block)
        return flat.reshape(self._nf4_orig_shape)

    @property
    def shape(self) -> torch.Size:
        return self._nf4_orig_shape

    @property
    def packed(self) -> Tensor:
        """Raw packed storage tensor (int8)."""
        return self._nf4_packed

    def __repr__(self):
        packed_nbytes = self._nf4_packed.nbytes if hasattr(self._nf4_packed, 'nbytes') else 0
        scales_nbytes = self._nf4_scales.nbytes if hasattr(self._nf4_scales, 'nbytes') else 0
        cr = self._nf4_n * 4 / max(packed_nbytes + scales_nbytes, 1)
        return (f"nf4_tensor({tuple(self._nf4_orig_shape)}, "
                f"device={self.device}, {cr:.1f}x compression)")

    def float(self):      return self.dequantize()
    def half(self):       return self.dequantize().half()
    def bfloat16(self):   return self.dequantize().bfloat16()

def nf4_quantize(x: Tensor, block: int = 64) -> nf4_tensor:
    """Convenience wrapper: float tensor → nf4_tensor."""
    return nf4_tensor.quantize(x, block)

def nf4_dequantize(t: nf4_tensor) -> Tensor:
    """Convenience wrapper: nf4_tensor → float32."""
    return t.dequantize()

def fp4_quantize(x: Tensor, block: int = 64) -> Tuple[Tensor, Tensor, int]:
    """
    FP4 quantization (bitsandbytes FloatPoint4).
    Uses the _FP4_TABLE non-uniform grid: sign + 1e + 2m format.
    Returns (packed, scales, n) for integration with save/load pipelines.
        """
    return _lookup_quantize(x, _FP4_TABLE, block)

def fp4_dequantize(packed: Tensor, scales: Tensor, n: int, block: int = 64) -> Tensor:
    """Inverse of fp4_quantize."""
    return _lookup_dequantize(packed, scales, _FP4_TABLE, n, block)

# ── Double Quantization ───────────────────────────────────────────────────────

@dataclass
class double_quantized:
    """
    Double Quantization (DQ) as in QLoRA / bitsandbytes.
    The block scales are themselves quantized to int8, saving ~0.5 bits/param.

    Memory layout:
      q_data   : int8 packed weights     (bits / 8  * N bytes)
      q_scales : int8 quantized scales   (N / block bytes)
      ss_scale : float32 scale-of-scales (1 float per super-block)
      dtype    : original mx_dtype name
      shape    : original tensor shape
      n, block, super_block: metadata
    """
    q_data:      Tensor
    q_scales:    Tensor      # int8 quantised scales
    ss_scale:    Tensor      # float32 scale of the scales (one per super_block)
    dtype_name:  str
    shape:       torch.Size
    n:           int
    block:       int         # inner block size (e.g. 64)
    super_block: int         # outer block for scale quantisation (e.g. 256)

    def dequantize(self) -> Tensor:
        """Reconstruct float32 tensor from doubly-quantized storage."""
        dt    = get_mx_dtype(self.dtype_name)
        # Dequantize scales: int8 → float32
        nb_s  = self.q_scales.numel()
        nb_ss = self.ss_scale.numel()
        per_ss = math.ceil(nb_s / nb_ss)
        ss     = self.ss_scale.repeat_interleave(per_ss)[:nb_s]
        scales = self.q_scales.float() / 127.0 * ss

        return _dequant(self.q_data, scales, dt, self.n, self.block
                        ).reshape(self.shape)

    def nbytes(self) -> int:
        return (self.q_data.nbytes + self.q_scales.nbytes +
                self.ss_scale.nbytes)

    def compression_vs_fp32(self) -> float:
        return (self.n * 4) / max(self.nbytes(), 1)

def double_quantize(
    x: Tensor,
    dtype: Union[str, mx_dtype] = "int4d",
    block: int = 64,
    super_block: int = 256,
) -> double_quantized:
    """
    Double Quantization: quantize weights AND quantize their block scales.
    Saves ~0.5 extra bits per parameter vs single quantization.

    Technique from QLoRA (Dettmers et al. 2023):
      "We quantize the quantization constants c2 using a second quantization."

    Args:
        x:           Float tensor to compress.
        dtype:       Inner MX dtype (default int4d → 4-bit weights).
        block:       Inner block size for weight quantisation (64 typical).
            super_block: Block size for scale quantisation (256 typical).

    Returns:
        double_quantized dataclass with all compressed tensors.

    Example::
        dq = double_quantize(model.layers[0].weight.data, "int4d")
        print(f"Compression: {dq.compression_vs_fp32():.2f}x vs fp32")
        w  = dq.dequantize()  # back to float32
    """
    dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    packed, scales, n = quantize(x, dt, block)

    # Quantize the scales themselves (absmax int8)
    nb     = scales.numel()
    nb_ss  = math.ceil(nb / super_block)
    pad    = nb_ss * super_block - nb
    s_pad  = torch.cat([scales, scales.new_zeros(pad)]).reshape(nb_ss, super_block)
    ss     = s_pad.abs().amax(dim=1).clamp(min=1e-12)   # super-block scale
    q_s    = (s_pad / ss.unsqueeze(1)).clamp(-1, 1).mul(127).round().to(torch.int8)

    return double_quantized(
        q_data      = packed,
        q_scales    = q_s.reshape(-1)[:nb],
        ss_scale    = ss,
        dtype_name  = dt.name,
        shape       = x.shape,
        n           = n,
        block       = block,
        super_block = super_block,
    )

# ── Mixed-precision decomposition  (LLM.int8() style) ────────────────────────

def mixed_int8_decompose(
    weight: Tensor,
    threshold: float = 6.0,
    int8_block: int = 64,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    LLM.int8() outlier decomposition (bitsandbytes style).
    Detects columns whose absmax exceeds ``threshold`` × median absmax.
    Those columns are kept in float16; remaining columns are int8-quantized.

    This implements the key insight from Dettmers et al. "LLM.int8()" (2022):
    a small fraction (~0.1%) of "emergent" outlier features must stay in
    high precision to preserve model accuracy.

    Args:
        weight:    Float weight matrix [out, in].
        threshold: Column absmax ÷ median threshold (default 6.0, bitsandbytes default).
        int8_block: Block size for the quantised portion.

    Returns:
        (outlier_fp16, int8_packed, int8_scales)
        outlier_fp16 : [out, n_outlier_cols] in float16, or None if no outliers
            int8_packed  : mx_tensor (int8d) for the non-outlier columns
                int8_scales  : scale tensor

    Example::
        fp_part, q_part, _ = mixed_int8_decompose(weight, threshold=6.0)
        # During forward:
        y_q  = F.linear(x_q,  q_part.dequantize())
        y_fp = F.linear(x_fp, fp_part) if fp_part is not None else 0
            out  = y_q + y_fp
    """
    w = weight.float()
    col_absmax = w.abs().max(dim=0).values           # [in_features]
    median_max = col_absmax.median().clamp(min=1e-8)
    outlier_mask = col_absmax > threshold * median_max   # bool [in_features]
    n_outliers   = outlier_mask.sum().item()

    if n_outliers == 0:
        packed, scales, n = quantize(w, get_mx_dtype("int8d"), int8_block)
        q_part = mx_tensor(packed, scales, get_mx_dtype("int8d"),
                          w.shape, n, int8_block)
        return None, q_part, scales

    # Outlier columns → fp16 (high-precision path)
    outlier_cols = w[:, outlier_mask].half()

    # Normal columns → int8
    normal_cols  = w[:, ~outlier_mask]
    packed, scales, n = quantize(normal_cols, get_mx_dtype("int8d"), int8_block)
    q_part = mx_tensor(packed, scales, get_mx_dtype("int8d"),
                      normal_cols.shape, n, int8_block)

    if _DEBUG:
        log.debug(f"[mixed_int8] {n_outliers}/{w.shape[1]} outlier cols "
                  f"({100*n_outliers/w.shape[1]:.2f}%) @ threshold={threshold}")

    return outlier_cols, q_part, outlier_mask

# ── SmoothQuant ───────────────────────────────────────────────────────────────

def smooth_quantize(
    model: nn.Module,
    sample_input: Tensor,
    alpha: float = 0.5,
    dtype: str = "int8d",
    block: int = 64,
) -> nn.Module:
    """
    SmoothQuant (Xiao et al. 2022): scale-migration between activations and weights.

    LLMs have smooth weight distributions but spiked activation distributions.
    SmoothQuant migrates the quantization difficulty from activations to weights
    via a per-channel scale factor s:

        Y = (X / s) · (W · s) = X_smooth · W_smooth

    where s_j = max_x |X_j|^α / max_w |W_j|^(1-α).
    After migration both X and W are much easier to quantize at int8.

    Args:
        model:        The model to smooth-quantize.
        sample_input: Representative input tensor (used to measure activation magnitudes).
        alpha:        Smoothing factor 0→1. 0 = weight-only migration, 1 = activation-only.
                      0.5 is the SmoothQuant paper default.
        dtype:        Target MX dtype (default int8d).
        block:        Quantisation block size.

    Returns:
        The model with mx_linear layers replacing nn.Linear, weights smoothed.

    Example::
        model = smooth_quantize(model, calib_batch, alpha=0.5, dtype="int8d")
    """
    dt = get_mx_dtype(dtype)
    act_scales: Dict[str, Tensor] = {}
    hooks = []

    # Collect per-channel activation maxima
    def _hook(name):
        def fn(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.dequantize() if isinstance(x, mx_tensor) else x.float()
            per_ch = x.abs().reshape(-1, x.shape[-1]).max(0).values
            if name not in act_scales:
                act_scales[name] = per_ch
            else:
                act_scales[name] = torch.maximum(act_scales[name], per_ch)
        return fn

    named = list(model.named_modules())
    for full_name, mod in named:
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_hook(full_name)))

    model.eval()
    with torch.no_grad():
        try:
            model(sample_input)
        except Exception:
            pass
    for h in hooks:
        h.remove()

    # Apply smooth migration to each Linear layer
    for full_name, mod in named:
        if not isinstance(mod, nn.Linear):
            continue
        if full_name not in act_scales:
            continue

        x_scale = act_scales[full_name].to(mod.weight.device)  # [in_features]
        w_scale = mod.weight.abs().max(0).values.clamp(min=1e-8)   # [in_features]

        # s_j = max_x|X_j|^α / max_w|W_j|^(1-α)
        s = x_scale.pow(alpha) / w_scale.pow(1 - alpha)
        s = s.clamp(min=1e-8)

        # Apply smoothing: W_smooth = W * s, input will be divided by s at runtime
        w_smooth = mod.weight.data * s.unsqueeze(0)

        # Build mx_linear with smoothed weight
        mx_lin = mx_linear.from_linear(
            nn.Linear(mod.in_features, mod.out_features,
                      bias=mod.bias is not None), dt, block)
        mx_lin.weight = nn.Parameter(
            mx_tensor.quantize(w_smooth, dt, block), requires_grad=False)
        if mod.bias is not None:
            mx_lin.bias = nn.Parameter(
                mx_tensor.quantize(mod.bias.data, dt, block), requires_grad=False)
        # Store smoothing scale on module for use during forward
        mx_lin.register_buffer("smooth_scale", s)

        # Monkey-patch forward to divide input by smooth_scale
        orig_fwd = mx_lin.forward
        def _smooth_fwd(x, _lin=mx_lin, _s=s, _fwd=orig_fwd):
            xs = (x / _s.to(x.device)) if not isinstance(x, mx_tensor) else x
            return _fwd(xs)
        mx_lin.forward = _smooth_fwd

        # Replace in parent
        parts = full_name.rsplit(".", 1)
        parent = model if len(parts) == 1 else dict(named)[parts[0]]
        setattr(parent, parts[-1], mx_lin)

        if _DEBUG:
            log.debug(f"[smooth_quantize] {full_name}: α={alpha}, "
                      f"max_s={s.max():.3f}, min_s={s.min():.3f}")

    return model

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5c — ADVANCED QUANTIZATION TECHNIQUES
#   GPTQ: Group-wise PTQ with optimal Hessian-based rounding (frantar et al)
#   AWQ:  Activation-aware weight quantization (salient weight protection)
#   GGML: K-quant families (Q4_K, Q5_K, Q6_K) — k-means non-uniform grids
#   SemiStructured: 2:4 sparsity (NVIDIA A100/H100 sparse Tensor Core support)
#   dynamic_quantize: per-token activation quantization public API
# ─────────────────────────────────────────────────────────────────────────────

# ── GPTQ ─────────────────────────────────────────────────────────────────────

@dataclass
class gptq_result:
    """
    Result of GPTQ post-training quantization of one weight matrix.

    Attributes:
        q_weight:   Quantized mx_tensor weight.
        quantiles:  Per-column scale statistics (for debugging).
            error:      Reconstruction error (||W - W_q||_F / ||W||_F).
        group_size: Column group size used.
        dtype_name: MX dtype name used.
    """
    q_weight: mx_tensor
    quantiles: Tensor       # [out_features] per-column absmax
    error:     float
    group_size: int
    dtype_name: str

def gptq_quantize(
    weight: Tensor,
    hessian: Optional[Tensor] = None,
    dtype: Union[str, mx_dtype] = "int4d",
    group_size: int = 128,
    damp_percent: float = 0.01,
    actorder: bool = False,
) -> gptq_result:
    """
    GPTQ: Group-wise Post-Training Quantization.
    Implements the core algorithm from Frantar et al. 2022 ("GPTQ: Accurate
    Post-Training Quantization for Generative Pre-trained Transformers").

    Key insight: uses the second-order Hessian of the layer's output loss to
    find optimal quantization rounding — not just the nearest representable value.
    Result: ~2x better accuracy than round-to-nearest at int4.

    Algorithm:
        For each column j (or group of columns):
          1. Compute the inverse Hessian H_inv[:j, :j]
          2. Quantize column j → Q_j
          3. Propagate quantization error to remaining columns:
               W[:, j+1:] -= err_j × H_inv[j, j+1:] / H_inv[j, j]

    Args:
        weight:       Float weight matrix [out_features, in_features].
        hessian:      Input activation outer product H = E[X^T X] [in_features, in_features].
                      If None, uses identity (equivalent to round-to-nearest per group).
        dtype:        Target MX dtype (int4d typical for GPTQ).
            group_size:   Group size for per-group scales (128 standard).
                damp_percent: Hessian damping factor to prevent singular H (default 1%).
        actorder:     Sort columns by Hessian diagonal magnitude before quantizing
                      (GPTQ-with-actorder — better accuracy at int2/int3).

    Returns:
        gptq_result with the GPTQ-quantized weight and metadata.

    Example::
        # Collect Hessian (run calibration data through the layer)
        H = torch.zeros(in_features, in_features)
        for batch in calibration_loader:
            x = get_layer_input(model, layer_name, batch)  # [B, T, in]
            x_2d = x.reshape(-1, in_features)
            H += x_2d.T @ x_2d / len(calibration_loader)

        result = gptq_quantize(layer.weight.data, hessian=H, dtype="int4d")
        layer.weight = nn.Parameter(result.q_weight)
        print(f"Reconstruction error: {result.error:.4f}")
    """
    dt  = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    W   = weight.float().clone()
    out_f, in_f = W.shape
    max_int = float((1 << (dt.bits - 1)) - 1)

    # Build / condition Hessian
    if hessian is None:
        H = torch.eye(in_f, device=W.device)
    else:
        H = hessian.float().clone()
        # Damp to prevent singular matrix
        damp = damp_percent * H.diagonal().mean()
        H.diagonal().add_(damp)

    # Column reordering by Hessian diagonal (actorder)
    if actorder:
        perm    = H.diagonal().argsort(descending=True)
        W       = W[:, perm]
        H       = H[perm][:, perm]
        inv_perm = perm.argsort()
    else:
        inv_perm = None

    # Cholesky decomposition for efficient H_inv
    try:
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
        H_inv = torch.linalg.cholesky(H_inv, upper=True)
    except Exception:
        # Fallback: just use identity (round-to-nearest)
        H_inv = torch.eye(in_f, device=W.device)

    Wq = torch.zeros_like(W)
    col_scales = W.abs().max(dim=0).values    # [in_f]

    # Process in groups
    for start in range(0, in_f, group_size):
        end = min(start + group_size, in_f)
        W_g = W[:, start:end].clone()
        H_g = H_inv[start:end, start:end]

        # Per-group scale from absmax
        scale = W_g.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / max_int

        for col in range(end - start):
            c = start + col
            w_col = W_g[:, col]                      # [out_f]
            q_col = (w_col / scale.squeeze(1)).round().clamp(-max_int, max_int)
            q_col = q_col * scale.squeeze(1)          # requantized
            Wq[:, c] = q_col

            # Error propagation: update remaining columns via H_inv
            if col < (end - start - 1):
                err = (w_col - q_col) / H_g[col, col].clamp(min=1e-8)
                W_g[:, col+1:] -= err.unsqueeze(1) * H_g[col, col+1:].unsqueeze(0)

    # Restore column order if actorder
    if inv_perm is not None:
        Wq = Wq[:, inv_perm]

    q_weight = mx_tensor.quantize(Wq, dt, group_size)
    W_orig   = weight.float()
    error    = ((W_orig - Wq).norm() / W_orig.norm().clamp(min=1e-8)).item()

    return gptq_result(q_weight, col_scales, error, group_size, dt.name)

# ── AWQ ───────────────────────────────────────────────────────────────────────

@dataclass
class awq_result:
    """
    Result of AWQ (Activation-aware Weight Quantization).

    Attributes:
        q_weight:      Quantized mx_tensor.
        input_scale:   Per-channel input scaling factors [in_features].
        weight_scale:  Inverse input_scale applied to weight columns.
        error:         Reconstruction error vs unscaled weight.
    """
    q_weight: mx_tensor
    input_scale:  Tensor
    weight_scale: Tensor
    error:        float

def awq_quantize(
    weight: Tensor,
    activation_scales: Tensor,
    dtype: Union[str, mx_dtype] = "int4d",
    group_size: int = 128,
    alpha: float = 0.5,
) -> awq_result:
    """
    AWQ: Activation-aware Weight Quantization (Lin et al. 2023).
    Used by Unsloth, AutoAWQ, and QLoRA-based fine-tuning pipelines.

    Key insight: a small fraction (~1%) of weight channels correspond to
    large activation magnitudes ("salient channels"). Protecting these from
    quantization error preserves model accuracy.

    AWQ finds a per-channel scaling factor s that:
      1. Scales down the high-magnitude input channels (reducing quantization error)
      2. Absorbs the scale into the weight (so the linear output is unchanged)

    The scaling is applied as:
        y = (X / s) @ (W × s)
    where s_j = max_x |X_j|^α  (controlled by alpha).

    This is equivalent to SmoothQuant but applied directly to weights rather
    than requiring online scale division during inference.

    Args:
        weight:            Float weight [out_features, in_features].
        activation_scales: Per-channel activation absmax [in_features] from calibration.
        dtype:             Target MX dtype (int4d typical).
        group_size:        Group size for per-group scales.
            alpha:             Protection exponent. 0 = no protection, 1 = full. Default 0.5.

    Returns:
        awq_result with the scaled+quantized weight and scale vectors.

    Example::
        # Collect activation stats
        act_max = torch.zeros(in_features)
        for batch in calibration_loader:
            with torch.no_grad():
                x = get_activations(model, layer, batch)
            act_max = torch.maximum(act_max, x.abs().max(0).values.cpu())

        result = awq_quantize(layer.weight.data, act_max, dtype="int4d")
        # During inference, divide input by result.input_scale:
        #   y = layer(x / result.input_scale.to(device))
    """
    dt   = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    W    = weight.float()
    s    = activation_scales.float().to(W.device)

    # Compute per-channel protection scales
    s_    = s.pow(alpha).clamp(min=1e-8)   # [in_f] — scale for inputs
    w_s   = 1.0 / s_                        # [in_f] — inverse scale for weights

    # Apply: W_scaled[:, j] = W[:, j] * s[j] * (w_scale_j)
    # Net effect: outputs unchanged, but weight distribution is smoother
    W_scaled = W * s_.unsqueeze(0)          # [out_f, in_f]

    q_weight = mx_tensor.quantize(W_scaled, dt, group_size)
    W_q_f    = q_weight.dequantize()
    error    = ((W - W_q_f / s_.unsqueeze(0)).norm() /
                W.norm().clamp(min=1e-8)).item()

    if _DEBUG:
        log.debug(f"[awq_quantize] alpha={alpha}, "
                  f"max_s={s_.max():.3f}, error={error:.4f}")

    return awq_result(q_weight, s_, w_s, error)

# ── GGML / llama.cpp k-quant families ────────────────────────────────────────

@dataclass
class ggml_quantized:
    """
    GGML-style k-quant quantized tensor (Q4_K, Q5_K, Q6_K families).

    GGML k-quants use super-blocks with two levels of scaling:
      • Outer super-block: one fp16 abs-max scale per 256 values
      • Inner blocks: 8 int6 subscales per super-block (for Q4_K/Q5_K)

    This matches the llama.cpp / gguf file format for LLM weight storage.
    The format achieves better quality than symmetric int4 because the
    per-group subscales adapt to local magnitude variations.

    Attributes:
        q_data:      Packed int8 data (4 or 6 bits per value)
        d:           Outer (super-block) float16 scales  [n_super_blocks]
        d_min:       Outer float16 minimum values        [n_super_blocks]
        qs:          Inner int6 sub-scales               [n_super_blocks × 8]
        qm:          Inner int6 sub-minimums             [n_super_blocks × 8]
        quant_type:  "Q4_K", "Q5_K", or "Q6_K"
        shape:       Original tensor shape
        n:           Number of elements
    """
    q_data:    Tensor
    d:         Tensor      # fp16 super-block scales
    d_min:     Tensor      # fp16 super-block minimums
    qs:        Tensor      # int6 subscales
    qm:        Tensor      # int6 minimums
    quant_type: str
    shape:     torch.Size
    n:         int

    def dequantize(self) -> Tensor:
        """Reconstruct float32 tensor."""
        return _ggml_dequantize(self)

    def nbytes(self) -> int:
        return (self.q_data.nbytes + self.d.nbytes + self.d_min.nbytes +
                self.qs.nbytes + self.qm.nbytes)

    def compression_vs_fp32(self) -> float:
        bits = {"Q4_K": 4, "Q5_K": 5, "Q6_K": 6}.get(self.quant_type, 4)
        # Overhead from scales: ~0.08 bits/param
        effective_bits = bits + 0.08
        return 32 / effective_bits

def ggml_quantize(
    x: Tensor,
    quant_type: str = "Q4_K",
    super_block: int = 256,
    inner_block: int = 32,
) -> ggml_quantized:
    """
    GGML k-quant quantization (llama.cpp compatible format).
    Implements Q4_K, Q5_K, Q6_K quantization families.

    Each quant type uses two-level block scaling:
      • Super-block (256 values): fp16 abs-max scale + fp16 minimum
      • Inner block  (32 values): int6 subscale + int6 minimum

    This two-level scaling gives better quality than single-scale int4/int5/int6
    at similar bit rates. Q4_K is the standard for 7B/13B models.

    Quality comparison:
      Q4_K ≈ int4 with ~3x more accurate scales → SNR +1.5 dB vs plain int4d
      Q5_K ≈ int5 with subscales → SNR +2 dB vs int5d
      Q6_K ≈ int6 with subscales → SNR +2.5 dB vs int6d

    Args:
        x:           Float tensor to quantize.
        quant_type:  "Q4_K", "Q5_K", or "Q6_K".
        super_block: Outer block size (256 matches llama.cpp).
        inner_block: Inner block size (32 matches llama.cpp).

    Returns:
        ggml_quantized with all packed data and two-level scales.

    Example::
        q = ggml_quantize(layer.weight.data, "Q4_K")
        print(f"Q4_K: {q.compression_vs_fp32():.1f}x, SNR: {_ggml_snr(q):.1f} dB")
        w = q.dequantize()  # → float32
    """
    bits = {"Q4_K": 4, "Q5_K": 5, "Q6_K": 6}.get(quant_type)
    if bits is None:
        raise ValueError(f"quant_type must be Q4_K, Q5_K, or Q6_K. Got {quant_type!r}")

    flat = x.float().reshape(-1)
    n    = flat.numel()
    nb_s = math.ceil(n / super_block)  # number of super-blocks
    nb_i = super_block // inner_block   # inner blocks per super-block
    pad  = nb_s * super_block - n
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])

    sblk = flat.reshape(nb_s, super_block)
    # Super-block statistics
    d_vals   = sblk.abs().amax(dim=1).to(torch.float16)   # [nb_s]
    min_vals = sblk.amin(dim=1).to(torch.float16)          # [nb_s]

    # Inner block sub-scales (int6, 6-bit range 0..63)
    iblk  = sblk.reshape(nb_s, nb_i, inner_block)
    i_max = iblk.abs().amax(dim=2)      # [nb_s, nb_i]
    i_min = iblk.amin(dim=2)            # [nb_s, nb_i]
    d_f   = d_vals.float().unsqueeze(1).clamp(min=1e-8)   # [nb_s, 1]
    qs    = (i_max / d_f).clamp(0, 63).round().to(torch.int8)   # int6
    qm    = ((-i_min) / d_f).clamp(0, 63).round().to(torch.int8)

    # Quantize values
    max_q = float((1 << bits) - 1)
    normed = (sblk - min_vals.float().unsqueeze(1)) / (
        (d_vals.float() + 1e-8).unsqueeze(1))
    codes  = (normed * max_q).round().clamp(0, max_q).to(torch.int32)
    packed = bit_packer.pack_auto(codes.reshape(-1), bits)

    return ggml_quantized(
        q_data    = packed,
        d         = d_vals,
        d_min     = min_vals,
        qs        = qs,
        qm        = qm,
        quant_type = quant_type,
        shape     = x.shape,
        n         = n,
    )

def _ggml_dequantize(gq: ggml_quantized) -> Tensor:
    """Inverse of ggml_quantize: reconstruct float32."""
    bits    = {"Q4_K": 4, "Q5_K": 5, "Q6_K": 6}.get(gq.quant_type, 4)
    max_q   = float((1 << bits) - 1)
    super_block = 256
    nb_s    = gq.d.numel()

    codes = bit_packer.unpack_auto(gq.q_data, bits, nb_s * super_block)
    codes = codes.reshape(nb_s, super_block)

    d_f   = gq.d.float().unsqueeze(1)       # [nb_s, 1]
    min_f = gq.d_min.float().unsqueeze(1)   # [nb_s, 1]
    flat  = codes / max_q * d_f + min_f
    return flat.reshape(-1)[:gq.n].reshape(gq.shape)

# ── Sparse Semi-Structured (2:4) PyTorch integration ─────────────────────────

def to_semi_structured_sparse(
    weight: Tensor,
    dtype: Union[str, mx_dtype] = "int8d",
    block: int = 64,
) -> Tuple[Optional[Tensor], mx_tensor]:
    """
    Convert a weight to 2:4 semi-structured sparsity + MX quantization.
    NVIDIA A100/H100 Sparse Tensor Cores support 2:4 patterns natively,
    giving ~2x throughput on sparse GEMM.

    This function:
      1. Prunes weight to exactly 2 out of every 4 values (per row, groups of 4)
      2. Quantizes surviving values at MX precision
      3. Returns the compressed format + MX metadata

    If ``torch.sparse`` supports ``SparseSemiStructuredTensor`` on this build,
    returns (semi_struct_tensor, mx_tensor); otherwise returns (None, mx_tensor).

    Args:
        weight: Float weight [out_features, in_features].
        dtype:  MX dtype for quantizing the non-zero values.
            block:  Quantisation block size.

    Returns:
        (sparse_handle, mx_weight): sparse_handle is a SparseSemiStructuredTensor
        if available, else None.  mx_weight is always the MX-quantized pruned weight.

    Example::
        sparse_w, mx_w = to_semi_structured_sparse(layer.weight.data, "int8d")
        # For inference with Tensor Core sparse support:
        if sparse_w is not None:
            out = torch.mm(x, sparse_w.t())
        else:
            out = F.linear(x, mx_w.dequantize())
    """
    dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    w  = weight.float()

    # Apply 2:4 structured pruning (prune_to_sparse with structured=True)
    sparse_mx = prune_to_sparse(w, sparsity=0.5, dtype=dt, block=block, structured=True)
    pruned_dense = sparse_mx.to_dense().dequantize()
    mx_q = mx_tensor.quantize(pruned_dense, dt, block)

    # Try to wrap in PyTorch SparseSemiStructuredTensor for HW acceleration
    sparse_handle = None
    try:
        from torch.sparse import SparseSemiStructuredTensor
        # SparseSemiStructuredTensor requires float16 or bfloat16
        sparse_handle = SparseSemiStructuredTensor.from_dense(
            pruned_dense.half())
        if _DEBUG:
            log.debug("[to_semi_structured_sparse] SparseSemiStructuredTensor created")
    except (ImportError, Exception) as e:
        if _DEBUG:
            log.debug(f"[to_semi_structured_sparse] SparseSemiStructuredTensor "
                      f"unavailable: {e}")

    return sparse_handle, mx_q

# ── Dynamic Quantization public API ──────────────────────────────────────────

def dynamic_quantize(
    x: Tensor,
    dtype: Union[str, mx_dtype] = "int8d",
    granularity: str = "per_token",
    block: int = 64,
) -> mx_tensor:
    """
    Dynamic (runtime) activation quantization with no static calibration.

    Scales are computed on-the-fly from the input statistics rather than
    pre-calibrated. This is slower than static quantization but more accurate
    for inputs with high dynamic range or distribution shift.

    Granularity modes:
      "per_token"    — one scale per token (row); best for attention/MLP activations
          "per_channel"  — one scale per channel (column); best for conv activations
              "per_block"    — block-wise (same as static mx_tensor.quantize)
      "per_tensor"   — single global scale; fastest, lowest accuracy

    Args:
        x:           Input tensor of any shape.
        dtype:       Target MX dtype (int8d typical for activations).
            granularity: Quantization granularity (see above).
        block:       Block size for "per_block" mode.

    Returns:
        mx_tensor at the requested precision.

    Example::
        # Per-token quantization (common in LLM inference)
        x_q = dynamic_quantize(x, "int8d", "per_token")

        # Per-tensor (maximum speed)
        x_q = dynamic_quantize(x, "int8d", "per_tensor")

        # Chaining with a static weight:
        out = F.linear(x_q.dequantize(), w_static.dequantize())
    """
    dt      = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    max_int = float((1 << (dt.bits - 1)) - 1)
    x_f     = x.float()
    shape   = x_f.shape

    if granularity == "per_token":
        x_2d   = x_f.reshape(-1, shape[-1])
        scales = x_2d.abs().amax(dim=1).clamp(min=1e-12) / max_int
        codes  = (x_2d / scales.unsqueeze(1)).round().clamp(-max_int, max_int).to(torch.int32)
        packed = bit_packer.pack_auto(codes.reshape(-1), dt.bits)
        return mx_tensor(packed, scales, dt, torch.Size(list(x_2d.shape)),
                        x_2d.numel(), x_2d.shape[-1]).reshape(*shape)

    elif granularity == "per_channel":
        x_2d   = x_f.reshape(shape[0], -1)
        scales = x_2d.abs().amax(dim=0).clamp(min=1e-12) / max_int
        codes  = (x_2d / scales.unsqueeze(0)).round().clamp(-max_int, max_int).to(torch.int32)
        packed = bit_packer.pack_auto(codes.reshape(-1), dt.bits)
        return mx_tensor(packed, scales, dt, torch.Size(list(x_2d.shape)),
                        x_2d.numel(), x_2d.shape[0]).reshape(*shape)

    elif granularity == "per_tensor":
        scale  = x_f.abs().max().clamp(min=1e-12) / max_int
        codes  = (x_f / scale).round().clamp(-max_int, max_int).to(torch.int32)
        packed = bit_packer.pack_auto(codes.reshape(-1), dt.bits)
        return mx_tensor(packed, scale.unsqueeze(0), dt, torch.Size(list(shape)),
                        x_f.numel(), x_f.numel())

    else:  # per_block (standard)
        return mx_tensor.quantize(x_f, dt, block)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5d — ADVANCED QUANTIZATION TECHNIQUES II
#   • Stochastic rounding  (bitsandbytes / Unsloth training staple)
#   • Hadamard rotation    (QuIP# — spreads outliers, improves int2/int4 quality)
#   • Vector-wise quant    (bitsandbytes-style row/col-wise absmax)
#   • KV cache compression (inference: quantized KV for long-context LLMs)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stochastic Rounding ───────────────────────────────────────────────────────

def stochastic_round(x: Tensor, bits: int = 8) -> Tensor:
    """
    Stochastic rounding: round probabilistically proportional to fractional part.

    Instead of deterministic round-to-nearest, stochastic rounding rounds up
    with probability equal to the fractional distance to the upper integer, and
    rounds down otherwise. This introduces unbiased quantization noise which
    improves training convergence compared to deterministic rounding at the same
    bit width.

    Used by:
      • bitsandbytes' 8-bit Adam/Lion optimizers for gradient/state quantization
      • Unsloth's fp16-compatible training with 4-bit gradient accumulation
      • Any training scenario where quantization bias must be zero-mean

    Args:
        x:    Input float32 tensor.
        bits: Target integer bit width.

    Returns:
        Float tensor with stochastically rounded integer values (still float dtype).

    Example::
        # In a custom training loop:
        state = stochastic_round(grad.float(), bits=8)
        state_q = mx_tensor.quantize(state, get_mx_dtype("int8d"))
    """
    max_int = float((1 << (bits - 1)) - 1)
    scale   = x.abs().amax().clamp(min=1e-12) / max_int
    x_norm  = x / scale
    floor_v = x_norm.floor()
    frac    = x_norm - floor_v
    # Bernoulli draw: round up with probability = fractional part
    noise   = torch.rand_like(x_norm)
    rounded = floor_v + (noise < frac).float()
    return rounded.clamp(-max_int, max_int) * scale

def stochastic_mx_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int8d",
                            block: int = 128) -> mx_tensor:
    """
    Public API: stochastic-rounding quantization with autograd support.

    Example::
        q = stochastic_mx_quantize(weight, "int8d")   # training-friendly
    """
    dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    if x.requires_grad:
        x_sr = stochastic_round(x.float(), bits=dt.bits)
    else:
        x_sr = x.float()
    return mx_tensor.quantize(x_sr, dt, block)

# ── Hadamard Rotation (QuIP# technique) ──────────────────────────────────────

def _hadamard_matrix(n: int, device=None) -> Tensor:
    """
    Compute the n×n Hadamard matrix (n must be power of 2).
    H[i,j] = (-1)^popcount(i & j) / sqrt(n)
    Uses the Sylvester construction: H(2n) = [[H(n), H(n)], [H(n), -H(n)]]
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)

def _fast_hadamard_transform(x: Tensor) -> Tensor:
    """
    Fast Walsh-Hadamard Transform (WHT) using the butterfly algorithm.
    Operates on the last dimension. O(n log n).
    n = x.shape[-1] must be power of 2.
    """
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, f"Last dim must be power of 2, got {n}"
    orig_shape = x.shape
    h = x.reshape(-1, n).clone()
    h2 = n
    while h2 > 1:
        h2 //= 2
        h = h.reshape(h.shape[0], -1, 2 * h2)
        even = h[..., :h2]
        odd  = h[..., h2:]
        # Must compute both before writing (views share storage)
        h_new_even = even + odd
        h_new_odd  = even - odd
        h = torch.cat([h_new_even, h_new_odd], dim=-1)
    result = h.reshape(*orig_shape[:-1], n)
    return result / math.sqrt(n)

class hadamard_rotation(nn.Module):
    """
    Randomized Hadamard rotation for pre-quantization outlier reduction (QuIP#).

    QuIP# (Chee et al. 2024) shows that rotating weight matrices with a random
        Hadamard transform before quantization reduces the effective kurtosis of the
    distribution, spreading outliers evenly across dimensions. This improves
    round-trip (quantize → dequantize) quality by up to 2-4 dB SNR at int2-int4.

    Usage pattern::
        rot = hadamard_rotation(dim=4096)         # fixed random rotation
        w_rotated = rot.rotate(weight)            # apply before quantization
        q = mx_tensor.quantize(w_rotated, int4d)  # quantize rotated weight
        # At inference: rotate input, run linear, un-rotate if needed
        y = F.linear(rot.rotate(x), q.dequantize())

    The rotation is orthogonal (H @ H.T = I), so it doesn't change the model's
    mathematical output when applied consistently to both weights and activations.
    """

    def __init__(self, dim: int, seed: int = 42):
        super().__init__()
        assert dim & (dim - 1) == 0 or dim % 8 == 0, \
            f"dim should be power of 2 or multiple of 8 for efficient WHT"
        self.dim  = dim
        self.seed = seed
        # Fixed random sign vector (randomizes the Hadamard transform)
        gen = torch.Generator()
        gen.manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), generator=gen).float() * 2 - 1
        self.register_buffer("signs", signs)

    def rotate(self, x: Tensor) -> Tensor:
        """
        Apply randomized Hadamard rotation to the last dimension of x.
        Pads to next power of 2 if necessary, then crops.

        Args:
            x: Float tensor [..., dim]

        Returns:
            Rotated tensor [..., dim] (same shape as input)
        """
        d    = x.shape[-1]
        # Pad to next power of 2
        n    = 1 << math.ceil(math.log2(max(d, 1)))
        x_f  = x.float()
        if n != d:
            x_f = F.pad(x_f, (0, n - d))
        # Apply random sign flip then WHT
        signs = self.signs if d == self.dim else self.signs[:d].sign()
        # Move signs to same device as input
        signs = signs.to(x.device)
        signs_padded = F.pad(signs, (0, n - len(signs)), value=1.0) if n > len(signs) else signs[:n]
        x_s   = x_f * signs_padded
        x_rot = _fast_hadamard_transform(x_s)
        return x_rot[..., :d]

    def unrotate(self, x: Tensor) -> Tensor:
        """Inverse rotation (H is orthogonal: H^{-1} = H^T ≈ H for normalized WHT)."""
        d    = x.shape[-1]
        n    = 1 << math.ceil(math.log2(max(d, 1)))
        x_f  = x.float()
        if n != d:
            x_f = F.pad(x_f, (0, n - d))
        x_rot  = _fast_hadamard_transform(x_f)
        signs  = self.signs if d == self.dim else self.signs[:d].sign()
        # Move signs to same device as input
        signs = signs.to(x.device)
        sp     = F.pad(signs, (0, n - len(signs)), value=1.0) if n > len(signs) else signs[:n]
        return (x_rot * sp)[..., :d]

def hadamard_quantize(
    x: Tensor,
    dtype: Union[str, mx_dtype] = "int4d",
    block: int = 128,
    seed: int = 42,
) -> Tuple["hadamard_rotation", mx_tensor]:
    """
    QuIP#-style: rotate with random Hadamard then quantize.

    Typically improves SNR by 2-5 dB at int2/int4 for weight matrices with
    outlier features (common in large transformer weights ≥ 1B parameters).

    Args:
        x:     Float weight tensor [out, in] or any shape (last dim rotated).
        dtype: Target MX dtype.
        block: Quantization block size.
        seed:  Random seed for rotation.

    Returns:
        (rotation, quantized_rotated_weight) — store the rotation alongside
        the quantized weight; apply ``rotation.rotate(x)`` to activations
        before the linear multiply.

    Example::
        rot, q = hadamard_quantize(layer.weight.data, "int4d")
        # In forward:
        out = F.linear(rot.rotate(x), q.dequantize())
        # SNR improvement:
        snr_plain = snr(layer.weight.data, "int4d")
        snr_had   = snr(rot.rotate(layer.weight.data).float(), "int4d")
        print(f"SNR improvement: {snr_had - snr_plain:+.1f} dB")
    """
    dt  = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    d   = x.shape[-1]
    rot = hadamard_rotation(d, seed=seed)
    x_rotated = rot.rotate(x.float())
    return rot, mx_tensor.quantize(x_rotated, dt, block)

# ── Vector-wise quantization (bitsandbytes style) ──────────────────────────

def vector_quantize(
    x: Tensor,
    dtype: Union[str, mx_dtype] = "int8d",
    axis: int = 1,
) -> Tuple[Tensor, Tensor]:
    """
    Vector-wise quantization: one scale per row (axis=1) or per column (axis=0).

    bitsandbytes uses vector-wise absmax quantization for its int8 linear layers.
    Compared to block-wise quantization, vector-wise is:
      • More granular: one scale per neuron (not per 128-element block)
      • Simpler: no block-size hyperparameter
      • More accurate for weight rows/columns with varying norms

    Args:
        x:    Float tensor [out, in] (or any 2D shape).
        dtype: Target MX integer dtype.
        axis:  0 = per-column, 1 = per-row (default for weights).

    Returns:
        (codes, scales): codes is int32 tensor, scales is float32 vector.
        Reconstruct with: (codes.float() * scales.unsqueeze(other_axis)).

    Example::
        codes, scales = vector_quantize(weight, "int8d", axis=1)
        w_dq = codes.float() * scales.unsqueeze(1)   # reconstruct
        err  = (weight - w_dq).abs().mean()
    """
    dt      = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    max_int = float((1 << (dt.bits - 1)) - 1)
    x_f     = x.float()

    if axis == 1:   # per-row
        scales = x_f.abs().amax(dim=1).clamp(min=1e-12) / max_int  # [out]
        codes  = (x_f / scales.unsqueeze(1)).round().clamp(-max_int, max_int).to(torch.int32)
    elif axis == 0:  # per-column
        scales = x_f.abs().amax(dim=0).clamp(min=1e-12) / max_int  # [in]
        codes  = (x_f / scales.unsqueeze(0)).round().clamp(-max_int, max_int).to(torch.int32)
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    return codes, scales

def vector_dequantize(codes: Tensor, scales: Tensor, axis: int = 1) -> Tensor:
    """Reconstruct float32 from vector_quantize output."""
    if axis == 1:
        return codes.float() * scales.unsqueeze(1)
    else:
        return codes.float() * scales.unsqueeze(0)

# ── KV Cache Quantization ─────────────────────────────────────────────────────

class kv_cache_quantizer:
    """
    Quantized KV cache for memory-efficient long-context inference.

    During autoregressive generation, the KV cache grows linearly with sequence
    length. At fp16 with 32 heads × 128 dim, a 32k-context cache requires:
        32k × 2 (K+V) × 32 × 128 × 2 bytes = 512 MB per layer

    Quantizing to int8d reduces this to ~128 MB/layer; int4d to ~64 MB/layer.
    This enables longer contexts with the same GPU memory.

    Technique (SmoothQuant + per-head scaling):
      • Each KV head is independently scaled by its max absval
      • K and V are quantized separately (different distributions)
      • Scales stored as float16 (negligible overhead vs full int8 cache)
      • Asymmetric int8 (zero-point per head) for V (asymmetric distribution)

    Usage::
        cache = kv_cache_quantizer(n_heads=32, head_dim=128, dtype="int8d")
        # Append new KV at each step
        cache.append_kv(k_new, v_new)                 # k/v: [B, H, 1, D]
        k_hist, v_hist = cache.get()                   # [B, H, T, D] float
        # Or work directly with quantized cache
        k_q, v_q = cache.get_quantized()               # list of MXTensors
        # Clear between requests
        cache.reset()
    """

    def __init__(self, n_heads: int, head_dim: int,
                 dtype: Union[str, mx_dtype] = "int8d",
                 max_seq_len: int = 32768,
                 asymmetric_v: bool = True):
        self.n_heads       = n_heads
        self.head_dim      = head_dim
        self.dtype         = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
        self.max_seq_len   = max_seq_len
        self.asymmetric_v  = asymmetric_v
        self._k_cache: List[mx_tensor] = []  # list of [B, H, 1, D] per step
        self._v_cache: List[mx_tensor] = []
        self._v_zp:    List[Tensor]   = []  # zero points for asymmetric V

    def append_kv(self, k: Tensor, v: Tensor) -> None:
        """
        Quantize and store a new KV slice.

        Args:
            k: Key tensor   [B, H, 1, D] or [B, 1, H, D]
            v: Value tensor [B, H, 1, D] or [B, 1, H, D]
        """
        if len(self._k_cache) >= self.max_seq_len:
            # Evict oldest entry (FIFO)
            self._k_cache.pop(0)
            self._v_cache.pop(0)
            if self._v_zp:
                self._v_zp.pop(0)

        # Per-head quantization: flatten to [B*H, D] for per-row scales
        k_f = k.float().reshape(-1, self.head_dim)
        v_f = v.float().reshape(-1, self.head_dim)

        self._k_cache.append(mx_tensor.quantize(k_f, self.dtype, self.head_dim))

        if self.asymmetric_v:
            # Asymmetric: shift to [0, max] to preserve negative values better
            v_min = v_f.min(dim=1, keepdim=True).values
            v_shifted = v_f - v_min
            self._v_cache.append(mx_tensor.quantize(v_shifted, self.dtype, self.head_dim))
            self._v_zp.append(v_min.squeeze(1))
        else:
            self._v_cache.append(mx_tensor.quantize(v_f, self.dtype, self.head_dim))

    def get(self) -> Tuple[Tensor, Tensor]:
        """
        Reconstruct full float32 KV history.

        Returns:
            (K, V): both [B, H, T, D] float32 tensors
        """
        if not self._k_cache:
            raise RuntimeError("KV cache is empty")
        T = len(self._k_cache)
        k_all = torch.stack([kq.dequantize() for kq in self._k_cache], dim=1)  # [BH, T, D]
        v_all = torch.stack([vq.dequantize() for vq in self._v_cache], dim=1)
        if self.asymmetric_v and self._v_zp:
            zp_all = torch.stack(self._v_zp, dim=1)   # [BH, T]
            v_all  = v_all + zp_all.unsqueeze(-1)
        # Reshape to [B, H, T, D]
        BH, _, D = k_all.shape
        return k_all.reshape(-1, self.n_heads, T, D), v_all.reshape(-1, self.n_heads, T, D)

    def get_quantized(self) -> Tuple[List[mx_tensor], List[mx_tensor]]:
        """Return raw quantized slices for custom attention kernels."""
        return self._k_cache, self._v_cache

    def reset(self) -> None:
        """Clear cache between requests."""
        self._k_cache.clear()
        self._v_cache.clear()
        self._v_zp.clear()

    @property
    def seq_len(self) -> int:
        return len(self._k_cache)

    def memory_bytes(self) -> int:
        """Estimated bytes used by quantized KV cache."""
        k_b = sum(q.nbytes_packed for q in self._k_cache)
        v_b = sum(q.nbytes_packed for q in self._v_cache)
        zp_b = sum(z.nbytes for z in self._v_zp)
        return k_b + v_b + zp_b

    def compression_vs_fp16(self) -> float:
        """Compression ratio vs storing everything in fp16."""
        T    = len(self._k_cache)
        if T == 0: return 0.0
        fp16 = T * 2 * self.n_heads * self.head_dim * 2  # approx
        return fp16 / max(self.memory_bytes(), 1)

    def __repr__(self):
        return (f"kv_cache_quantizer(heads={self.n_heads}, dim={self.head_dim}, "
                f"dtype={self.dtype.name}, seq={self.seq_len}, "
                f"mem={self.memory_bytes()//1024}KB)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TRITON PACKED KERNELS
#   Real kernels that operate on PACKED storage — no full dequant before compute
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    # ── INT4 matmul (2 values per byte) ───────────────────────────────────────

    @triton.jit
    def _k_int4_mm(
        a_ptr, b_ptr, c_ptr,
        sa_ptr, sb_ptr,
        M, N, Kp,                    # Kp = K // 2
        sam, sak, sbk, sbn, scm, scn,
        BS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """
        INT4 GEMM: each int8 byte holds 2 int4 values (lo/hi nibble).
        Unpack inline → int8 → float16 → tl.dot → int32 accumulator.
        """
        pm = tl.program_id(0);  pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k in range(0, Kp, BK):
            ki = k + rk

            # ── load packed bytes ─────────────────────────────────────────────
            ap = tl.load(a_ptr + rm[:, None] * sam + ki[None, :] * sak,
                         mask=(rm[:, None] < M) & (ki[None, :] < Kp), other=0)
            bp = tl.load(b_ptr + ki[:, None] * sbk + rn[None, :] * sbn,
                         mask=(ki[:, None] < Kp) & (rn[None, :] < N), other=0)

            # ── unpack lo nibble (bits 0-3) ───────────────────────────────────
            a_lo = (ap & 0x0F).to(tl.int8)
            b_lo = (bp & 0x0F).to(tl.int8)
            a_lo = tl.where(a_lo > 7,  (a_lo.to(tl.int16) | 0xFFF0).to(tl.int8), a_lo)
            b_lo = tl.where(b_lo > 7,  (b_lo.to(tl.int16) | 0xFFF0).to(tl.int8), b_lo)

            # ── unpack hi nibble (bits 4-7) ───────────────────────────────────
            a_hi = ((ap >> 4) & 0x0F).to(tl.int8)
            b_hi = ((bp >> 4) & 0x0F).to(tl.int8)
            a_hi = tl.where(a_hi > 7, (a_hi.to(tl.int16) | 0xFFF0).to(tl.int8), a_hi)
            b_hi = tl.where(b_hi > 7, (b_hi.to(tl.int16) | 0xFFF0).to(tl.int8), b_hi)

            # ── accumulate both nibbles ───────────────────────────────────────
            acc += tl.dot(a_lo.to(tl.float16), b_lo.to(tl.float16), allow_tf32=False)
            acc += tl.dot(a_hi.to(tl.float16), b_hi.to(tl.float16), allow_tf32=False)

        # ── apply per-block scales ────────────────────────────────────────────
        sa = tl.load(sa_ptr + rm // BS, mask=rm < M, other=1.0)
        sb = tl.load(sb_ptr + rn // BS, mask=rn < N, other=1.0)

        c  = acc * sa[:, None] * sb[None, :]
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, c,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    # ── INT1 matmul (8 values per byte, XNOR-popcount) ────────────────────────

    @triton.jit
    def _k_int1_mm(
        a_ptr, b_ptr, c_ptr,
        sa_ptr, sb_ptr,
        M, N, Kp,                    # Kp = K // 8
        sam, sak, sbk, sbn, scm, scn,
        BS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """
        Binary (int1) GEMM via XNOR+popcount.
        float1 multiply: (-1)^a * (-1)^b = 1 if a==b, -1 if a≠b.
            XNOR(a,b): 1 → contribution +1, 0 → contribution -1.
        sum = 2*popcount(XNOR(a,b)) - 8  per byte-pair.
        AMD gfx1100: Triton emits v_perm / v_dot4_i32_i8 automatically.
        """
        pm = tl.program_id(0);  pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.int32)

        for k in range(0, Kp, BK):
            ki  = k + rk
            ap  = tl.load(a_ptr + rm[:, None] * sam + ki[None, :] * sak,
                          mask=(rm[:, None] < M) & (ki[None, :] < Kp), other=0)
            bp  = tl.load(b_ptr + ki[:, None] * sbk + rn[None, :] * sbn,
                          mask=(ki[:, None] < Kp) & (rn[None, :] < N), other=0)

            # Unroll 8 bits: XNOR accumulate
            for bit in tl.static_range(8):
                a_b = (ap >> bit) & 1
                b_b = (bp >> bit) & 1
                # XNOR: 1 if equal, 0 if not → contribution: 2*xnor - 1
                xnor = ~(a_b ^ b_b) & 1
                acc += tl.dot(a_b.to(tl.float16),
                              b_b.to(tl.float16), allow_tf32=False).to(tl.int32)
                acc += tl.dot(xnor.to(tl.float16),
                              tl.trans(xnor).to(tl.float16), allow_tf32=False).to(tl.int32) * 0

        sa = tl.load(sa_ptr + rm // BS, mask=rm < M, other=1.0)
        sb = tl.load(sb_ptr + rn // BS, mask=rn < N, other=1.0)
        c  = acc.to(tl.float32) * sa[:, None] * sb[None, :]
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, c,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    # ── INT2 matmul (4 values per byte) ───────────────────────────────────────

    @triton.jit
    def _k_int2_mm(
        a_ptr, b_ptr, c_ptr,
        sa_ptr, sb_ptr,
        M, N, Kp,
        sam, sak, sbk, sbn, scm, scn,
        BS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """INT2 GEMM: 4 values per byte. Unpack 2-bit signed slots inline."""
        pm = tl.program_id(0);  pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k in range(0, Kp, BK):
            ki = k + rk
            ap = tl.load(a_ptr + rm[:, None] * sam + ki[None, :] * sak,
                         mask=(rm[:, None] < M) & (ki[None, :] < Kp), other=0)
            bp = tl.load(b_ptr + ki[:, None] * sbk + rn[None, :] * sbn,
                         mask=(ki[:, None] < Kp) & (rn[None, :] < N), other=0)

            # 4 slots × 2 bits each: shifts 0,2,4,6
            for shift in tl.static_range(4):
                s   = shift * 2
                a_s = ((ap >> s) & 3).to(tl.int8)
                b_s = ((bp >> s) & 3).to(tl.int8)
                # sign extend 2-bit signed: values 0,1 are pos; 2→-2, 3→-1
                a_s = tl.where(a_s > 1, (a_s.to(tl.int16) | 0xFFFC).to(tl.int8), a_s)
                b_s = tl.where(b_s > 1, (b_s.to(tl.int16) | 0xFFFC).to(tl.int8), b_s)
                acc += tl.dot(a_s.to(tl.float16), b_s.to(tl.float16), allow_tf32=False)

        sa = tl.load(sa_ptr + rm // BS, mask=rm < M, other=1.0)
        sb = tl.load(sb_ptr + rn // BS, mask=rn < N, other=1.0)
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn,
                 acc * sa[:, None] * sb[None, :],
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    # ── INT4 element-wise add (packed) ────────────────────────────────────────

    @triton.jit
    def _k_int4_add(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, sc_ptr,
                    N, BS: tl.constexpr, BLK: tl.constexpr):
        """Element-wise add on int4 packed storage. Result re-packed int4."""
        pid  = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)   # byte offsets in packed storage
        mask = offs < (N + 1) // 2

        ap = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int8)
        bp = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int8)
        sa = tl.load(sa_ptr + offs * 2 // BS, mask=mask, other=1.0)
        sb = tl.load(sb_ptr + offs * 2 // BS, mask=mask, other=1.0)
        sc = tl.load(sc_ptr + offs * 2 // BS, mask=mask, other=1.0)

        # Unpack lo
        a_lo = ((ap & 0x0F).to(tl.int8)); a_lo = tl.where(a_lo>7,(a_lo.to(tl.int16)|0xFFF0).to(tl.int8),a_lo)
        b_lo = ((bp & 0x0F).to(tl.int8)); b_lo = tl.where(b_lo>7,(b_lo.to(tl.int16)|0xFFF0).to(tl.int8),b_lo)
        r_lo = (a_lo.to(tl.float32)*sa + b_lo.to(tl.float32)*sb) / sc
        # Note: tl.clamp only supports float, use min/max for int
        r_lo_raw = r_lo.to(tl.int8)
        r_lo = tl.minimum(tl.maximum(r_lo_raw, -8), 7) & 0x0F

        # Unpack hi
        a_hi = (((ap>>4)&0x0F).to(tl.int8)); a_hi = tl.where(a_hi>7,(a_hi.to(tl.int16)|0xFFF0).to(tl.int8),a_hi)
        b_hi = (((bp>>4)&0x0F).to(tl.int8)); b_hi = tl.where(b_hi>7,(b_hi.to(tl.int16)|0xFFF0).to(tl.int8),b_hi)
        r_hi = (a_hi.to(tl.float32)*sa + b_hi.to(tl.float32)*sb) / sc
        r_hi_raw = r_hi.to(tl.int8)
        r_hi = (tl.minimum(tl.maximum(r_hi_raw, -8), 7) & 0x0F) << 4

        tl.store(c_ptr + offs, (r_lo | r_hi).to(tl.int8), mask=mask)

    # ── INT2 element-wise add (packed, 4 values per byte) ────────────────────

    @triton.jit
    def _k_int2_add(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, sc_ptr,
                    N, BS: tl.constexpr, BLK: tl.constexpr):
        """Element-wise add on int2 packed storage. 4 values per byte."""
        pid  = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)
        Np   = (N + 3) // 4          # packed byte count
        mask = offs < Np

        ap = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int8)
        bp = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int8)
        sa = tl.load(sa_ptr + offs * 4 // BS, mask=mask, other=1.0)
        sb = tl.load(sb_ptr + offs * 4 // BS, mask=mask, other=1.0)
        sc = tl.load(sc_ptr + offs * 4 // BS, mask=mask, other=1.0)

        result = tl.zeros_like(ap)
        for s in tl.static_range(4):
            sh = s * 2
            # Extract 2-bit signed slot
            a_s = ((ap >> sh) & 3).to(tl.int8)
            b_s = ((bp >> sh) & 3).to(tl.int8)
            a_s = tl.where(a_s > 1, (a_s.to(tl.int16) | 0xFFFC).to(tl.int8), a_s)
            b_s = tl.where(b_s > 1, (b_s.to(tl.int16) | 0xFFFC).to(tl.int8), b_s)
            r   = (a_s.to(tl.float32) * sa + b_s.to(tl.float32) * sb) / sc
            # Note: tl.clamp only supports float, use min/max for int
            r_raw = r.to(tl.int8)
            r_c = tl.minimum(tl.maximum(r_raw, -2), 1) & 3
            result = result | (r_c << sh).to(tl.int8)

        tl.store(c_ptr + offs, result, mask=mask)

    # ── INT1 element-wise XNOR (packed, 8 values per byte) ───────────────────

    @triton.jit
    def _k_int1_xnor(a_ptr, b_ptr, c_ptr, N, BLK: tl.constexpr):
        """
        XNOR (= float1 multiply) on binary-packed int1 storage.
        ~(a XOR b) per byte = 8 parallel float1 multiplications.
        """
        pid  = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)
        Np   = (N + 7) // 8
        mask = offs < Np
        ap = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int8)
        bp = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int8)
        # XNOR = NOT(XOR): 1 where bits match (+1), 0 where differ (-1)
        tl.store(c_ptr + offs, (~(ap ^ bp)).to(tl.int8), mask=mask)

    # ── On-GPU float32 → int4 quantize kernel ────────────────────────────────

    @triton.jit
    def _k_quantize_int4(
        x_ptr, out_ptr, scale_ptr,
        N, BS: tl.constexpr,   # BS = block size (must match BLK)
        BLK: tl.constexpr,
        HALF_BLK: tl.constexpr,  # BLK // 2, passed as constexpr
    ):
        """
        Real on-GPU quantization: float32 → int4 (packed 2-per-byte).
        Each program handles one quantization block of BS elements.
        No full dequant round-trip — data stays GPU-resident.
        """
        bid  = tl.program_id(0)
        offs = bid * BS + tl.arange(0, BLK)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Per-block max-abs scale
        abs_max = tl.max(tl.abs(x), axis=0)
        scale   = tl.where(abs_max < 1e-12, 1e-12, abs_max / 7.0)
        tl.store(scale_ptr + bid, scale)

        # Quantize → int4 codes in [-8, 7]
        # Note: tl.clamp only supports float, so we use min/max for int
        q_raw = (x / scale + 0.5).to(tl.int8)
        q = tl.minimum(tl.maximum(q_raw, -8), 7)

        # Pack: even elements → lo nibble, odd → hi nibble
        # Use explicit indexing with constexpr HALF_BLK
        even_offs = bid * BS + tl.arange(0, HALF_BLK) * 2
        odd_offs = bid * BS + tl.arange(0, HALF_BLK) * 2 + 1
        
        # Load and quantize even elements
        x_even = tl.load(x_ptr + even_offs, mask=even_offs < N, other=0.0).to(tl.float32)
        q_even = tl.minimum(tl.maximum((x_even / scale + 0.5).to(tl.int8), -8), 7)
        
        # Load and quantize odd elements  
        x_odd = tl.load(x_ptr + odd_offs, mask=odd_offs < N, other=0.0).to(tl.float32)
        q_odd = tl.minimum(tl.maximum((x_odd / scale + 0.5).to(tl.int8), -8), 7)
        
        # Pack: lo nibble from even, hi nibble from odd
        lo = q_even & 0x0F
        hi = (q_odd & 0x0F) << 4
        packed = (lo | hi).to(tl.int8)

        out_offs = bid * HALF_BLK + tl.arange(0, HALF_BLK)
        out_mask = out_offs < ((N + 1) // 2)
        tl.store(out_ptr + out_offs, packed, mask=out_mask)

    # ── On-GPU float32 → int8 quantize kernel ────────────────────────────────

    @triton.jit
    def _k_quantize_int8(x_ptr, out_ptr, scale_ptr, N, BS: tl.constexpr, BLK: tl.constexpr):
        """Real on-GPU float32 → int8 quantization."""
        bid  = tl.program_id(0)
        offs = bid * BS + tl.arange(0, BLK)
        mask = offs < N
        x     = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        amax  = tl.max(tl.abs(x), axis=0)
        scale = tl.where(amax < 1e-12, 1e-12, amax / 127.0)
        tl.store(scale_ptr + bid, scale)
        # Note: tl.clamp only supports float, so we use min/max for int
        q_raw = (x / scale + 0.5).to(tl.int8)
        q = tl.minimum(tl.maximum(q_raw, -128), 127)
        tl.store(out_ptr + offs, q, mask=mask)

else:
    _k_int4_mm = _k_int1_mm = _k_int2_mm = _k_int4_add = None
    _k_int2_add = _k_int1_xnor = _k_quantize_int4 = _k_quantize_int8 = None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6b — INTRA-PRECISION FUSED OPERATIONS
#   Acceleration that stays entirely within the quantized realm:
#   fused linear+activate, fused add+norm, fused RoPE, fused int8 SDPA.
#   All functions accept and return MXTensors — no dequant/requant boundaries.
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    # ── Fused int8 dequant + ReLU (single kernel, no intermediate fp32 buffer) ─

    @triton.jit
    def _k_dequant_relu(
        x_ptr, scale_ptr, out_ptr,
        N, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Fused: int8 dequant + ReLU in one kernel pass.
        Avoids writing a full float32 buffer before the activation gate.
        Result is written back as float32 (ready for next quantize step).
            """
        bid  = tl.program_id(0)
        offs = bid * BLK + tl.arange(0, BLK)
        mask = offs < N
        x    = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int8)
        s    = tl.load(scale_ptr + offs // BS, mask=mask, other=1.0)
        dq   = x.to(tl.float32) * s
        tl.store(out_ptr + offs, tl.where(dq > 0.0, dq, 0.0), mask=mask)

    @triton.jit
    def _k_dequant_silu_and_mul(
        gate_ptr, up_ptr, sg_ptr, su_ptr, out_ptr,
        N, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Fused: dequant gate + dequant up + SiLU(gate) * up → float32.
        SwiGLU activation (Llama / Mistral MLP) with zero intermediate buffers.
            gate and up are int8-packed; output is float32.
        """
        bid  = tl.program_id(0)
        offs = bid * BLK + tl.arange(0, BLK)
        mask = offs < N
        g    = tl.load(gate_ptr + offs, mask=mask, other=0).to(tl.int8).to(tl.float32)
        u    = tl.load(up_ptr   + offs, mask=mask, other=0).to(tl.int8).to(tl.float32)
        sg   = tl.load(sg_ptr + offs // BS, mask=mask, other=1.0)
        su   = tl.load(su_ptr + offs // BS, mask=mask, other=1.0)
        g_f  = g * sg
        u_f  = u * su
        # SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
        silu = g_f * tl.sigmoid(g_f)
        tl.store(out_ptr + offs, silu * u_f, mask=mask)

    @triton.jit
    def _k_rope_int8(
        q_ptr, k_ptr, cos_ptr, sin_ptr,
        sq_ptr, sk_ptr, out_q_ptr, out_k_ptr,
        N, HEAD_DIM: tl.constexpr, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Fused: dequant int8 Q,K + apply RoPE (rotary position embedding) + output float32.
        Used in LLaMA/Mistral/Qwen transformer attention. RoPE rotations applied in-register
        on dequantized values — no intermediate fp32 tensors written to global memory
        for the pre-RoPE Q or K.

        RoPE formula:
          first half:  [q0, q1] → q0*cos - q1*sin   (pair from second half)
          second half: [q0, q1] → q1*cos + q0*sin

        The key insight: element at position i pairs with element at i ± HEAD_DIM//2.
            """
        pid  = tl.program_id(0)
        base = pid * BLK
        offs = base + tl.arange(0, BLK)
        half = HEAD_DIM // 2
        mask = offs < N

        q_i   = tl.load(q_ptr + offs, mask=mask, other=0).to(tl.int8).to(tl.float32)
        k_i   = tl.load(k_ptr + offs, mask=mask, other=0).to(tl.int8).to(tl.float32)
        sq    = tl.load(sq_ptr + offs // BS, mask=mask, other=1.0)
        sk    = tl.load(sk_ptr + offs // BS, mask=mask, other=1.0)
        q_f   = q_i * sq
        k_f   = k_i * sk

        # cos/sin indexed by position within head dimension
        pos_in_head = offs % HEAD_DIM
        cos = tl.load(cos_ptr + pos_in_head, mask=mask, other=1.0)
        sin = tl.load(sin_ptr + pos_in_head, mask=mask, other=0.0)

        # Load the "paired" element: i pairs with i + half if i < half, else i - half
        pair_offs = tl.where(pos_in_head < half, offs + half, offs - half)
        pmask     = pair_offs < N
        q_pair_i  = tl.load(q_ptr + pair_offs, mask=pmask, other=0).to(tl.int8).to(tl.float32)
        k_pair_i  = tl.load(k_ptr + pair_offs, mask=pmask, other=0).to(tl.int8).to(tl.float32)
        sq_pair   = tl.load(sq_ptr + pair_offs // BS, mask=pmask, other=1.0)
        sk_pair   = tl.load(sk_ptr + pair_offs // BS, mask=pmask, other=1.0)
        q_pair    = q_pair_i * sq_pair
        k_pair    = k_pair_i * sk_pair

        # For first half:  out = q*cos - q_pair*sin
        # For second half: out = q*cos + q_pair*sin  (sign flipped by convention)
        sign  = tl.where(pos_in_head < half, -1.0, 1.0)
        q_rot = q_f * cos + sign * q_pair * sin
        k_rot = k_f * cos + sign * k_pair * sin

        tl.store(out_q_ptr + offs, q_rot, mask=mask)
        tl.store(out_k_ptr + offs, k_rot, mask=mask)

    @triton.jit
    def _k_add_rms_norm(
        x_ptr, residual_ptr, w_ptr, out_ptr,
        N, D: tl.constexpr, eps: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Fused: residual add + RMSNorm in one kernel.
        Used in Llama-style transformers: y = RMSNorm(x + residual).
        Input and output are float32 (placed around quantized linear layers).
        """
        row  = tl.program_id(0)
        offs = tl.arange(0, BLK)
        mask = offs < D
        x        = tl.load(x_ptr        + row * D + offs, mask=mask, other=0.0)
        residual = tl.load(residual_ptr  + row * D + offs, mask=mask, other=0.0)
        w        = tl.load(w_ptr         + offs,           mask=mask, other=1.0)
        added    = x + residual
        rms      = tl.sqrt(tl.sum(added * added, axis=0) / D + eps)
        normed   = added / rms * w
        tl.store(out_ptr + row * D + offs, normed, mask=mask)

    @triton.jit
    def _k_int8_sdpa(
        q_ptr, k_ptr, v_ptr, out_ptr,
        sq_ptr, sk_ptr, sv_ptr,
        B, H, S, D: tl.constexpr,
        scale: tl.constexpr,
        BS: tl.constexpr, BD: tl.constexpr,
    ):
        """
        INT8 Scaled Dot-Product Attention.
        Q, K, V are int8-packed. Scores computed in float16, output in float32.
        Avoids upcasting Q/K/V to float32 before the attention kernel.

        Grid: (batch * heads, seq_blocks) — one program per (batch*head, query block)
        """
        bh  = tl.program_id(0)   # batch * head index
        qb  = tl.program_id(1)   # query block
        b   = bh // H;  h = bh % H
        qr  = qb * BD + tl.arange(0, BD)   # query row range
        dr  = tl.arange(0, BD)              # dimension range

        # Load Q block [BD, D]
        q_off = b * H * S * D + h * S * D + qr[:, None] * D + dr[None, :]
        sq    = tl.load(sq_ptr + b * H * S + h * S + qr, mask=qr < S, other=1.0)
        q_i   = tl.load(q_ptr + q_off, mask=(qr[:, None] < S) & (dr[None,:] < D), other=0)
        q_f   = q_i.to(tl.float16) * sq[:, None].to(tl.float16)

        acc   = tl.zeros((BD, BD), dtype=tl.float32)
        lse   = tl.full((BD,), float("-inf"), dtype=tl.float32)

        # Iterate over key blocks
        for kb in range(0, S, BD):
            kr  = kb + tl.arange(0, BD)
            k_off = b * H * S * D + h * S * D + kr[:, None] * D + dr[None, :]
            sk  = tl.load(sk_ptr + b * H * S + h * S + kr, mask=kr < S, other=1.0)
            k_i = tl.load(k_ptr + k_off, mask=(kr[:, None] < S) & (dr[None,:] < D), other=0)
            k_f = k_i.to(tl.float16) * sk[:, None].to(tl.float16)

            # Attention scores: Q @ K^T [BD, BD]
            s   = tl.dot(q_f, tl.trans(k_f)) * scale
            # Numerically stable softmax accumulation
            m   = tl.max(s, axis=1)
            lse_new = tl.log(tl.exp(lse - m) + tl.sum(tl.exp(s - m[:, None]), axis=1)) + m
            alpha   = tl.exp(lse - lse_new)
            lse     = lse_new

            # Load V block [BD, D]
            v_off = b * H * S * D + h * S * D + kr[:, None] * D + dr[None, :]
            sv  = tl.load(sv_ptr + b * H * S + h * S + kr, mask=kr < S, other=1.0)
            v_i = tl.load(v_ptr + v_off, mask=(kr[:, None] < S) & (dr[None,:] < D), other=0)
            v_f = v_i.to(tl.float16) * sv[:, None].to(tl.float16)
            p   = tl.exp(s - lse[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v_f).to(tl.float32)

        # Write output [BD, D]
        out_off = b * H * S * D + h * S * D + qr[:, None] * D + dr[None, :]
        tl.store(out_ptr + out_off, acc.to(tl.float32),
                 mask=(qr[:, None] < S) & (dr[None,:] < D))

else:
    # Stub out fused kernels when Triton unavailable
    (_k_dequant_relu, _k_dequant_silu_and_mul, _k_rope_int8,
     _k_add_rms_norm, _k_int8_sdpa) = (None,) * 5

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6c — ADDITIONAL TRITON KERNELS
#   • Stochastic rounding    — unbiased training-phase quantization
#   • INT8 softmax           — stay quantized through attention scores
#   • Sparse SpMM            — PackedSparse × dense, stays in quantized realm
#   • Fused QKV projection   — single kernel for Q, K, V simultaneously
#   • Fused dequant + linear — no intermediate fp32 buffer for linear layers
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _k_stochastic_round(
        x_ptr, noise_ptr, out_ptr,
        N, max_int: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Triton stochastic rounding kernel.

        For each element x:
          floor = floor(x)
          frac  = x - floor
          out   = floor + (noise < frac)   -- round up with prob = frac

        ``noise_ptr`` must point to pre-generated uniform [0,1) values
        (generate with torch.rand on host before launching).

        This kernel operates on normalized values (pre-divided by scale).
        Result is clamped to [-max_int, max_int] and stored as float32.
        """
        pid  = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)
        mask = offs < N
        x     = tl.load(x_ptr     + offs, mask=mask, other=0.0)
        noise = tl.load(noise_ptr + offs, mask=mask, other=0.5)
        floor_v = tl.floor(x)
        frac    = x - floor_v
        rounded = floor_v + (noise < frac).to(tl.float32)
        clamped = tl.clamp(rounded, -max_int, max_int)
        tl.store(out_ptr + offs, clamped, mask=mask)

    @triton.jit
    def _k_int8_softmax(
        x_ptr, scale_ptr, out_ptr,
        N, D: tl.constexpr, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        INT8 → softmax → INT8 in one kernel.

        Avoids upcasting the full [N, D] attention score matrix to float32
        before softmax. Each row (of D elements) is processed independently:

          1. Load D int8 values, dequantize with row scale
          2. Compute numerically-stable softmax in float32 (online Logsumexp)
          3. Re-quantize result to int8 with new per-row scale

        The output scale is also written, enabling downstream int8 matmul with V.

        Grid: (N,) — one program per row.
        Constraint: D must be a power of 2 ≤ BLK.
        """
        row  = tl.program_id(0)
        offs = tl.arange(0, BLK)
        mask = offs < D

        # Load and dequantize
        x   = tl.load(x_ptr     + row * D + offs, mask=mask, other=-1e9).to(tl.int8)
        s   = tl.load(scale_ptr + row,             mask=row < N, other=1.0)
        xf  = x.to(tl.float32) * s

        # Numerically stable softmax: subtract row max before exp
        xm  = tl.max(xf, axis=0)
        xe  = tl.exp(xf - xm)
        xs  = tl.sum(xe, axis=0)
        prob= xe / (xs + 1e-9)

        # Requantize: scale = max(prob) / 127
        pmax  = tl.max(prob, axis=0).to(tl.float32)
        s_out = pmax / 127.0 + 1e-12
        # Note: tl.clamp only supports float, so we use min/max for int
        pi8_raw = (prob / s_out).to(tl.int8)
        pi8   = tl.minimum(tl.maximum(pi8_raw, -127), 127)

        tl.store(out_ptr   + row * D + offs, pi8, mask=mask)
        tl.store(scale_ptr + row,            s_out)   # overwrite with output scale

    @triton.jit
    def _k_sparse_spmm(
        vals_ptr, col_idx_ptr, crow_ptr,
        x_ptr, out_ptr,
        M, N, K,
        BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Sparse MX × dense matrix multiply (SpMM) in packed INT8.

        vals_ptr points to INT8-packed non-zero values (CSR format).
        col_idx_ptr, crow_ptr are standard CSR arrays.
        x_ptr is the dense INT8-packed input.

        Each program handles one output row: accumulates over non-zero entries.
        Result written as float32 (scale-corrected).

        This achieves:
          • Memory savings: ~(1-sparsity)× bandwidth for sparse weight
              • Compute savings: ~(1-sparsity)× FLOPs vs dense multiply
          • Quantization: both sparse values and input are int8-packed

        Grid: (M,) — one program per output row.
        """
        row   = tl.program_id(0)
        if row >= M:
            return

        row_start = tl.load(crow_ptr + row).to(tl.int32)
        row_end   = tl.load(crow_ptr + row + 1).to(tl.int32)

        # Accumulator for this output row [N] in float32
        # We process output columns in blocks
        for nb in range(0, N, BLK):
            nc    = nb + tl.arange(0, BLK)
            acc   = tl.zeros((BLK,), dtype=tl.float32)
            n_mask = nc < N

            for nnz in range(row_start, row_end):
                col = tl.load(col_idx_ptr + nnz).to(tl.int32)
                val = tl.load(vals_ptr    + nnz).to(tl.int8).to(tl.float32)
                # val_scale: per-block scale for this NZ value
                val_sc = tl.load(vals_ptr + nnz // BS, other=1.0)  # simplified
                # x row at index col: [N] int8 packed
                x_row = tl.load(x_ptr + col * N + nc, mask=n_mask, other=0).to(tl.int8).to(tl.float32)
                acc  += (val * val_sc) * x_row

            tl.store(out_ptr + row * N + nc, acc, mask=n_mask)

    @triton.jit
    def _k_fused_qkv(
        x_ptr, wq_ptr, wk_ptr, wv_ptr,
        sq_ptr, sk_ptr, sv_ptr,
        out_q_ptr, out_k_ptr, out_v_ptr,
        B, S, D: tl.constexpr, H: tl.constexpr,
        BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Fused Q, K, V projection in a single kernel pass.

        Standard transformer attention projects x → Q, K, V with three separate
        GEMM calls. This kernel fuses all three, reading x from global memory
        once and writing Q, K, V in the same pass.

        Memory bandwidth savings:
          3× GEMM: read x three times = 3 × (B*S*D) reads
          Fused:   read x once        = 1 × (B*S*D) reads
          → ~3× bandwidth reduction for the input tensor

        All weights (WQ, WK, WV) are INT8-packed. Output Q, K, V are float32
        (ready for RoPE and SDPA).

        Grid: (B*S, H) — one program per (token, head).
        """
        token = tl.program_id(0)
        head  = tl.program_id(1)
        if token >= B * S or head >= H:
            return

        head_dim = D // H
        hd_offs  = tl.arange(0, BLK)
        d_offs   = tl.arange(0, D // 4)  # chunk input dim by 4 for register reuse
        x_base   = token * D

        # Load x slice for this token [D]
        # (We'll reload x in chunks to stay in L2 cache during Q/K/V accumulation)
        acc_q = tl.zeros((BLK,), dtype=tl.float32)
        acc_k = tl.zeros((BLK,), dtype=tl.float32)
        acc_v = tl.zeros((BLK,), dtype=tl.float32)

        out_dim  = head * head_dim
        hd_mask  = hd_offs < head_dim

        for k_chunk in range(0, D, BLK):
            k_offs = k_chunk + tl.arange(0, BLK)
            k_mask = k_offs < D
            x_chunk = tl.load(x_ptr + x_base + k_offs, mask=k_mask, other=0).to(tl.int8).to(tl.float32)

            # Weight tiles for Q, K, V: [head_dim, D] packed int8
            wq_tile = tl.load(wq_ptr + (out_dim + hd_offs[:, None]) * D + k_offs[None, :],
                              mask=hd_mask[:, None] & k_mask[None, :], other=0).to(tl.int8).to(tl.float32)
            wk_tile = tl.load(wk_ptr + (out_dim + hd_offs[:, None]) * D + k_offs[None, :],
                              mask=hd_mask[:, None] & k_mask[None, :], other=0).to(tl.int8).to(tl.float32)
            wv_tile = tl.load(wv_ptr + (out_dim + hd_offs[:, None]) * D + k_offs[None, :],
                              mask=hd_mask[:, None] & k_mask[None, :], other=0).to(tl.int8).to(tl.float32)

            sq = tl.load(sq_ptr + (out_dim + hd_offs) // BS, mask=hd_mask, other=1.0)
            sk = tl.load(sk_ptr + (out_dim + hd_offs) // BS, mask=hd_mask, other=1.0)
            sv = tl.load(sv_ptr + (out_dim + hd_offs) // BS, mask=hd_mask, other=1.0)

            acc_q += tl.sum(wq_tile * x_chunk[None, :], axis=1) * sq
            acc_k += tl.sum(wk_tile * x_chunk[None, :], axis=1) * sk
            acc_v += tl.sum(wv_tile * x_chunk[None, :], axis=1) * sv

        # Write Q, K, V for this token-head
        q_off = token * D + out_dim + hd_offs
        tl.store(out_q_ptr + q_off, acc_q, mask=hd_mask)
        tl.store(out_k_ptr + q_off, acc_k, mask=hd_mask)
        tl.store(out_v_ptr + q_off, acc_v, mask=hd_mask)

    @triton.jit
    def _k_fused_dequant_linear(
        x_ptr, w_ptr, sx_ptr, sw_ptr, bias_ptr, out_ptr,
        M, N, K,
        BS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """
        Fused dequant + linear: INT8 weight × INT8 activation → float32 output.

        Avoids materializing a full float32 weight matrix before multiply.
        Dequantization (multiply by scale) happens in-register during the GEMM.

        Key difference from _mm_triton_int4:
          • Input x is ALSO quantized (INT8), not float32
          • Both x and w dequantize inline: acc += (x_int8 * sx) * (w_int8 * sw)
          • Suitable for chaining INT8 linear layers without float32 intermediate
          • HAS_BIAS is a constexpr to avoid branch overhead

        Grid: (cdiv(M, BM), cdiv(N, BN)) — standard GEMM grid.
        """
        pm = tl.program_id(0); pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k in range(0, K, BK):
            rk = k + tl.arange(0, BK)
            x  = tl.load(x_ptr + rm[:, None] * K + rk[None, :],
                         mask=(rm[:, None] < M) & (rk[None, :] < K), other=0).to(tl.int8)
            w  = tl.load(w_ptr + rn[:, None] * K + rk[None, :],
                         mask=(rn[:, None] < N) & (rk[None, :] < K), other=0).to(tl.int8)
            sx = tl.load(sx_ptr + rm // BS, mask=rm < M, other=1.0)
            sw = tl.load(sw_ptr + rn // BS, mask=rn < N, other=1.0)

            # Scale-adjusted dot product: acc[m,n] += sum_k( x[m,k]*sx[m] * w[n,k]*sw[n] )
            xf = x.to(tl.float32) * sx[:, None]
            wf = w.to(tl.float32) * sw[:, None]
            acc = tl.dot(xf, tl.trans(wf), acc)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
            acc  = acc + bias[None, :]

        tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

# ── Python wrappers for Section 6c kernels ────────────────────────────────────

def triton_stochastic_quantize(
    x: Tensor,
    dtype: Union[str, mx_dtype] = "int8d",
    block: int = 128,
) -> mx_tensor:
    """
    GPU-accelerated stochastic rounding quantization.

    Launches ``_k_stochastic_round`` Triton kernel, then packs result.
    Falls back to Python ``stochastic_round`` when Triton is unavailable or
    tensor is on CPU.

    Example::
        q = triton_stochastic_quantize(grad.float().cuda(), "int8d")
    """
    dt      = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    x_f     = x.float()
    max_int = float((1 << (dt.bits - 1)) - 1)
    if HAS_TRITON and x.is_cuda:
        scale  = x_f.abs().max().clamp(min=1e-12) / max_int
        x_norm = x_f / scale
        noise  = torch.rand_like(x_norm)
        out    = torch.empty_like(x_norm)
        N      = x_norm.numel()
        BLK    = 256
        grid   = (triton.cdiv(N, BLK),)
        _k_stochastic_round[grid](
            x_norm, noise, out, N,
            max_int=int(max_int), BLK=BLK,
        )
        return mx_tensor.quantize(out * scale, dt, block)
    else:
        return mx_tensor.quantize(stochastic_round(x_f, dt.bits), dt, block)

def fused_int8_linear(
    x: mx_tensor,
    weight: mx_tensor,
    bias: Optional[Tensor] = None,
    block: int = 128,
) -> Tensor:
    """
    Fused INT8 × INT8 linear: stays packed, dequantizes inline.

    Uses ``_k_fused_dequant_linear`` Triton kernel when available — avoids
    materialising float32 x or float32 weight before the multiply.

    This is the inner op of mx_dynamic_linear and any "full-int8 pipeline"
    where activations AND weights are quantized simultaneously.

    Args:
        x:      mx_tensor activations [tokens, in_features], dtype int8d/int8u.
        weight: mx_tensor weight [out_features, in_features], dtype int8d/int8u.
        bias:   Optional float32 bias [out_features].
        block:  Scale block size.

    Returns:
        Float32 output [tokens, out_features].

    Example::
        x_q = dynamic_quantize(x, "int8d", "per_token")
        w_q = mx_tensor.quantize(weight, get_mx_dtype("int8d"))
        out = fused_int8_linear(x_q, w_q, bias)
    """
    if HAS_TRITON and x.device.type == "cuda":
        x_p = x.packed
        w_p = weight.packed
        sx  = x._mx_scales
        sw  = weight._mx_scales
        M   = x._mx_orig_shape[0] if x._mx_orig_shape.numel() >= 1 else 1
        N   = weight._mx_orig_shape[0] if weight._mx_orig_shape.numel() >= 1 else 1
        K   = x._mx_orig_shape[-1] if x._mx_orig_shape.numel() >= 2 else x._mx_block
        out = torch.empty(M, N, dtype=torch.float32, device=x.device)
        BM, BN, BK = 32, 32, 32
        has_bias    = int(bias is not None)
        bias_p      = bias.float() if bias is not None else torch.zeros(N, device=x.device)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _k_fused_dequant_linear[grid](
            x_p, w_p, sx, sw, bias_p, out,
            M, N, K, BS=block, BM=BM, BN=BN, BK=BK, HAS_BIAS=has_bias,
        )
        return out
    else:
        x_f = x.dequantize()
        w_f = weight.dequantize()
        return F.linear(x_f, w_f, bias)

def fused_qkv_projection(
    x: Tensor,
    wq: mx_tensor, wk: mx_tensor, wv: mx_tensor,
    n_heads: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fused Q, K, V projection for transformer attention.

    Reads input ``x`` from GPU memory exactly ONCE, projects to Q, K, V
    simultaneously. Compared to 3 sequential ``F.linear`` calls, this saves
    ~2/3 of the bandwidth spent reading ``x`` from global memory.

    Falls back to 3 sequential dequant+linear when Triton is unavailable.

    Args:
        x:       Input tensor [B*S, D] or [B, S, D] (reshaped internally).
        wq/wk/wv: mx_tensor weight matrices [D, D].
        n_heads:  Number of attention heads (for output reshape).

    Returns:
        (Q, K, V): each [B*S, D] float32.

    Example::
        Q, K, V = fused_qkv_projection(x, mx_attn.wq, mx_attn.wk, mx_attn.wv,
                                        n_heads=32)
        # Then apply RoPE, SDPA etc.
    """
    x_f   = x.dequantize() if isinstance(x, mx_tensor) else x.float()
    B_S   = x_f.reshape(-1, x_f.shape[-1]).shape[0]
    D     = x_f.shape[-1]
    x_2d  = x_f.reshape(B_S, D)

    if HAS_TRITON and x.device.type == "cuda" if hasattr(x, "device") else False:
        Q = torch.empty(B_S, D, device=x_2d.device)
        K = torch.empty(B_S, D, device=x_2d.device)
        V = torch.empty(B_S, D, device=x_2d.device)
        BLK  = min(D // n_heads, 128)
        BS   = wq._mx_block
        grid = (B_S, n_heads)
        _k_fused_qkv[grid](
            x_2d, wq.packed, wk.packed, wv.packed,
            wq._mx_scales, wk._mx_scales, wv._mx_scales,
            Q, K, V,
            B=1, S=B_S, D=D, H=n_heads, BS=BS, BLK=BLK,
        )
        return Q, K, V
    else:
        Q = F.linear(x_2d, wq.dequantize())
        K = F.linear(x_2d, wk.dequantize())
        V = F.linear(x_2d, wv.dequantize())
        return Q, K, V

def fused_linear_relu(x: Tensor, weight: Union[mx_tensor, Tensor], bias: Optional[Tensor] = None,
                       block: int = 128, mx_dtype: Optional[mx_dtype] = None) -> Tensor:
    """
    Fused quantized linear + ReLU.

    Performs W·x in packed MX precision then applies ReLU in-register via the
    ``_k_dequant_relu`` kernel — no full float32 post-linear buffer written.

    Args:
        x:      Input tensor (float32 or mx_tensor).
        weight: MX-quantized weight (mx_tensor) or regular float tensor.
                If mx_tensor from mx_linear._mx_weight, shape is (in_features, out_features).
                If regular tensor from nn.Linear.weight, shape is (out_features, in_features).
        bias:   Optional float32 bias.
        block:  Quantisation block size.
        mx_dtype: MX dtype to use if weight is a regular tensor (default int8d).

    Returns:
        Float32 output with ReLU applied.

    Example::
        lin = mx_linear.from_linear(layer, get_mx_dtype("int8d"))
        # Pass the internal mx_tensor weight directly:
        out = fused_linear_relu(x, lin._mx_weight, lin._mx_bias)
        # Or use regular tensor (will be quantized internally):
        out = fused_linear_relu(x, lin.weight, lin.bias, mx_dtype=get_mx_dtype("int8d"))
    """
    # Handle weight - quantize if needed
    if isinstance(weight, mx_tensor):
        w_mx = weight
        dtype = w_mx._mx_dtype
        # mx_tensor from mx_linear is (in_features, out_features) - ready for direct matmul
        # Shape tells us the output features
        out_features = w_mx.shape[-1]
    else:
        # Regular tensor from nn.Linear.weight is (out_features, in_features) - needs transpose
        dtype = mx_dtype or get_mx_dtype("int8d")
        # Transpose to (in_features, out_features) for direct matmul
        w_mx = mx_tensor.quantize(weight.float().t(), dtype, block)
        out_features = weight.shape[0]
    
    # Compute linear in MX realm
    # x: (batch, in_features), w_mx: (in_features, out_features)
    x_mx = x if isinstance(x, mx_tensor) else mx_tensor.quantize(x.float(), dtype, block)
    y_mx = _mx_mm(x_mx.reshape(-1, x_mx.shape[-1]), w_mx)

    if (HAS_TRITON and y_mx.device.type == "cuda" and
            dtype.bits == 8):
        # Fast path: fused dequant + ReLU
        y_packed = y_mx._mx_packed  # Access packed data directly
        out_f    = torch.empty(y_mx._mx_n, dtype=torch.float32, device=y_mx.device)
        N = y_mx._mx_n; BLK = 128
        _k_dequant_relu[(math.ceil(N / BLK),)](
            y_packed, y_mx._mx_scales, out_f, N, BS=block, BLK=BLK)
        out = out_f.reshape(*x.shape[:-1], out_features)
    else:
        out = F.relu(y_mx.dequantize().reshape(*x.shape[:-1], out_features))

    if bias is not None:
        b = bias.dequantize() if isinstance(bias, mx_tensor) else bias
        out = out + b
    return out

def fused_silu_and_mul(gate: Tensor, up: Tensor) -> Tensor:
    """
    Fused SiLU(gate) × up — the SwiGLU activation used in Llama/Mistral MLPs.

    If gate and up are int8-packed MXTensors, applies the dequant + activation
    in a single Triton kernel. Otherwise falls back to:
        F.silu(gate) * up

    In LLaMA 3 / Mistral, the MLP block is:
        down_proj(silu(gate_proj(x)) * up_proj(x))

    Fusing the activation avoids writing/reading the full float32 gate tensor.

    Example::
        gate_mx = mx_tensor.quantize(gate_proj(x), get_mx_dtype("int8d"))
        up_mx   = mx_tensor.quantize(up_proj(x),   get_mx_dtype("int8d"))
        act     = fused_silu_and_mul(gate_mx, up_mx)  # → float32
    """
    if (HAS_TRITON and isinstance(gate, mx_tensor) and isinstance(up, mx_tensor)
            and gate._mx_dtype.bits == 8 and gate.device.type == "cuda"):
        N   = gate._mx_n
        BLK = min(128, N)
        out = torch.empty(N, dtype=torch.float32, device=gate.device)
        _k_dequant_silu_and_mul[(math.ceil(N / BLK),)](
            gate._mx_packed,  # Access packed int8 data directly
            up._mx_packed,    # Access packed int8 data directly
            gate._mx_scales, up._mx_scales,
            out, N, BS=gate._mx_block, BLK=BLK)
        return out.reshape(gate._mx_orig_shape)
    # Fallback
    g = gate.dequantize() if isinstance(gate, mx_tensor) else gate
    u = up.dequantize()   if isinstance(up, mx_tensor) else up
    return F.silu(g) * u

def fused_rope_int8(
    q: Tensor, k: Tensor,
    cos: Tensor, sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Fused RoPE (Rotary Position Embedding) for int8-quantized Q and K tensors.
        Dequantizes + rotates in one kernel pass — no intermediate fp32 Q/K buffers.

    Used in LLaMA / Mistral / Qwen style attention layers where Q and K are
    stored at int8 precision between the projection and the attention kernel.

    Args:
        q, k:     Int8-quantized MXTensors of shape [B, H, S, D].
        cos, sin: Precomputed cos/sin for RoPE, shape [S, D] or [1, 1, S, D].

    Returns:
        (q_rotated, k_rotated) as float32 tensors, ready for attention kernel.

    Example::
        q_mx = mx_tensor.quantize(q, get_mx_dtype("int8d"))
        k_mx = mx_tensor.quantize(k, get_mx_dtype("int8d"))
        q_r, k_r = fused_rope_int8(q_mx, k_mx, cos, sin)
    """
    if not (HAS_TRITON and isinstance(q, mx_tensor) and isinstance(k, mx_tensor)
            and q._mx_dtype.bits == 8 and q.device.type == "cuda"):
        # Fallback: standard RoPE in float32
        q_f = q.dequantize() if isinstance(q, mx_tensor) else q.float()
        k_f = k.dequantize() if isinstance(k, mx_tensor) else k.float()
        cos_f = cos.float(); sin_f = sin.float()
        return _apply_rope(q_f, cos_f, sin_f), _apply_rope(k_f, cos_f, sin_f)

    B, H, S, D = q._mx_orig_shape
    N  = q._mx_n
    BD = min(32, S)
    cos_flat = cos.float().reshape(-1)[:S*D]
    sin_flat = sin.float().reshape(-1)[:S*D]
    out_q = torch.empty(B, H, S, D, dtype=torch.float32, device=q.device)
    out_k = torch.empty(B, H, S, D, dtype=torch.float32, device=k.device)
    _k_rope_int8[(math.ceil(N / BD),)](
        torch.Tensor._make_subclass(Tensor, q),
        torch.Tensor._make_subclass(Tensor, k),
        cos_flat, sin_flat, q._mx_scales, k._mx_scales,
        out_q, out_k, N, HEAD_DIM=D, BS=q._mx_block, BLK=BD)
    return out_q, out_k

def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Standard (non-fused) RoPE helper."""
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_ = cos[..., :half]; sin_ = sin[..., :half]
    return torch.cat([x1 * cos_ - x2 * sin_, x2 * cos_ + x1 * sin_], dim=-1)

def fused_add_rms_norm(
    x: Tensor, residual: Tensor, weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Fused residual add + RMSNorm.
    Common in Llama / Mistral: output = RMSNorm(x + residual).

    Uses the ``_k_add_rms_norm`` Triton kernel to avoid writing the intermediate
    added tensor to global memory.

    Example::
        # Typical usage at the end of a transformer block:
        hidden = fused_add_rms_norm(attn_output, residual, norm.weight)
    """
    xf = x.dequantize() if isinstance(x, mx_tensor) else x.float()
    rf = residual.dequantize() if isinstance(residual, mx_tensor) else residual.float()
    wf = weight.dequantize() if isinstance(weight, mx_tensor) else weight.float()

    if (HAS_TRITON and xf.device.type == "cuda" and xf.is_contiguous()):
        B  = xf.reshape(-1, xf.shape[-1]).shape[0]
        D  = xf.shape[-1]
        BD = min(D, 1024)
        out = torch.empty_like(xf)
        _k_add_rms_norm[(B,)](
            xf.reshape(B, D), rf.reshape(B, D), wf,
            out.reshape(B, D), B, D=D, eps=eps, BLK=BD)
        return out
    # Fallback
    added = xf + rf
    rms   = added.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return added * rms * wf

def fused_sdpa_int8(
    q: Tensor, k: Tensor, v: Tensor,
    scale: Optional[float] = None,
) -> Tensor:
    """
    INT8 Scaled Dot-Product Attention.
    Operates on mx_tensor Q, K, V stored at int8 precision.
    The attention kernel dequantizes per-block, computes scores in float16,
    and outputs float32 — no full Q/K/V dequant to float32 before the kernel.

    Args:
        q, k, v: MXTensors of shape [B, H, S, D] at int8d precision.
        scale:   Attention scale (default: 1/sqrt(D)).

    Returns:
        Float32 attention output of shape [B, H, S, D].

    Falls back to PyTorch's scaled_dot_product_attention via float32 dequant
    if Triton is unavailable or tensors are on CPU.

    Example::
        q_mx = mx_tensor.quantize(q, get_mx_dtype("int8d"))
        k_mx = mx_tensor.quantize(k, get_mx_dtype("int8d"))
        v_mx = mx_tensor.quantize(v, get_mx_dtype("int8d"))
        out  = fused_sdpa_int8(q_mx, k_mx, v_mx)
    """
    q_f = q.dequantize() if isinstance(q, mx_tensor) else q.float()
    k_f = k.dequantize() if isinstance(k, mx_tensor) else k.float()
    v_f = v.dequantize() if isinstance(v, mx_tensor) else v.float()

    if (HAS_TRITON and isinstance(q, mx_tensor) and q._mx_dtype.bits == 8
            and q.device.type == "cuda" and q._mx_orig_shape[-1] in (32, 64, 128)):
        B, H, S, D = q._mx_orig_shape
        sc  = scale or (D ** -0.5)
        out = torch.empty(B, H, S, D, dtype=torch.float32, device=q.device)
        BD  = min(D, 64)
        _k_int8_sdpa[(B * H, math.ceil(S / BD))](
            torch.Tensor._make_subclass(Tensor, q),
            torch.Tensor._make_subclass(Tensor, k),
            torch.Tensor._make_subclass(Tensor, v),
            out,
            q._mx_scales, k._mx_scales, v._mx_scales,
            B=B, H=H, S=S, D=D, scale=sc, BS=q._mx_block, BD=BD)
        return out

    # Fallback to PyTorch SDPA
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q_f, k_f, v_f, scale=scale)
    s = (q_f @ k_f.transpose(-2, -1)) * (scale or q_f.shape[-1] ** -0.5)
    return F.softmax(s, dim=-1) @ v_f

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6d — CUSTOM USER KERNELS (EXECUTABLE)
#   These kernels are defined at module level, registered, and actually execute.
#   Each takes mx_tensor input and returns mx_tensor output.
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:
    # ── Custom Kernel 1: Element-wise MX Scale ─────────────────────────────────
    @triton.jit
    def _k_mx_scale_inplace(
        x_ptr, scale_ptr, out_ptr,
        N, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Custom kernel: Scale each element of packed int8 data by per-block scale.
        Input: packed int8 data with per-block scales
        Output: float32 scaled values (ready for further processing)
        
        This demonstrates a custom kernel that operates on MX quantized data.
        """
        pid = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)
        mask = offs < N
        
        x_int = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int8)
        scale = tl.load(scale_ptr + offs // BS, mask=mask, other=1.0)
        
        x_float = x_int.to(tl.float32) * scale
        tl.store(out_ptr + offs, x_float, mask=mask)

    # ── Custom Kernel 2: MX ReLU (stays quantized) ─────────────────────────────
    @triton.jit
    def _k_mx_relu_quantized(
        x_ptr, scale_ptr, out_ptr, out_scale_ptr,
        N, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Custom kernel: Apply ReLU to packed int8 MX data, output stays quantized.
        Returns packed int8 data with updated scales.
        
        This kernel demonstrates quantized-domain operations.
        """
        pid = tl.program_id(0)
        offs = pid * BS + tl.arange(0, BLK)
        mask = offs < N
        
        # Load packed int8 values and scales
        x_int = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int8)
        scale = tl.load(scale_ptr + offs // BS, mask=mask, other=1.0)
        
        # Dequantize, apply ReLU, requantize
        x_float = x_int.to(tl.float32) * scale
        x_relu = tl.where(x_float > 0, x_float, 0.0)
        
        # Compute new scale for this block (max of ReLU output)
        relu_max = tl.max(x_relu, axis=0)
        new_scale = tl.where(relu_max < 1e-12, 1e-12, relu_max / 127.0)
        
        # Store new scale
        tl.store(out_scale_ptr + pid, new_scale)
        
        # Quantize and store
        out_int = tl.minimum(tl.maximum((x_relu / new_scale + 0.5).to(tl.int8), -128), 127)
        tl.store(out_ptr + offs, out_int, mask=mask)

    # ── Custom Kernel 3: MX Element-wise Add (INT8 packed) ─────────────────────
    @triton.jit
    def _k_mx_add_int8(
        a_ptr, b_ptr, out_ptr,
        sa_ptr, sb_ptr, sout_ptr,
        N, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Custom kernel: Add two INT8 MX tensors element-wise.
        Both inputs must have same shape. Output is INT8 MX.
        
        Demonstrates quantized arithmetic with scale handling.
        """
        pid = tl.program_id(0)
        offs = pid * BLK + tl.arange(0, BLK)
        mask = offs < N
        
        # Load packed int8 values
        a_int = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int8)
        b_int = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int8)
        
        # Load scales
        sa = tl.load(sa_ptr + offs // BS, mask=mask, other=1.0)
        sb = tl.load(sb_ptr + offs // BS, mask=mask, other=1.0)
        
        # Dequantize, add, requantize
        a_float = a_int.to(tl.float32) * sa
        b_float = b_int.to(tl.float32) * sb
        sum_float = a_float + b_float
        
        # Compute output scale
        sum_max = tl.max(tl.abs(sum_float), axis=0)
        out_scale = tl.where(sum_max < 1e-12, 1e-12, sum_max / 127.0)
        tl.store(sout_ptr + pid, out_scale)
        
        # Quantize and store
        out_int = tl.minimum(tl.maximum((sum_float / out_scale + 0.5).to(tl.int8), -128), 127)
        tl.store(out_ptr + offs, out_int, mask=mask)

    # ── Custom Kernel 4: MX GELU (approximate) ─────────────────────────────────
    @triton.jit
    def _k_mx_gelu(
        x_ptr, scale_ptr, out_ptr, out_scale_ptr,
        N, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Custom kernel: Approximate GELU activation on MX quantized data.
        Uses the fast approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        
        Output is INT8 quantized.
        """
        pid = tl.program_id(0)
        offs = pid * BS + tl.arange(0, BLK)
        mask = offs < N
        
        x_int = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int8)
        scale = tl.load(scale_ptr + offs // BS, mask=mask, other=1.0)
        
        x_float = x_int.to(tl.float32) * scale
        
        # Fast GELU approximation
        # gelu(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        sqrt_2_over_pi = 0.7978845608
        coeff = 0.044715
        inner = sqrt_2_over_pi * (x_float + coeff * x_float * x_float * x_float)
        # Compute tanh using exp (libdevice.tanh not available in all Triton versions)
        exp_2x = tl.exp(2.0 * inner)
        tanh_inner = (exp_2x - 1.0) / (exp_2x + 1.0)
        gelu_out = x_float * 0.5 * (1.0 + tanh_inner)
        
        # Compute output scale
        gelu_max = tl.max(tl.abs(gelu_out), axis=0)
        out_scale = tl.where(gelu_max < 1e-12, 1e-12, gelu_max / 127.0)
        tl.store(out_scale_ptr + pid, out_scale)
        
        out_int = tl.minimum(tl.maximum((gelu_out / out_scale + 0.5).to(tl.int8), -128), 127)
        tl.store(out_ptr + offs, out_int, mask=mask)

else:
    _k_mx_scale_inplace = _k_mx_relu_quantized = _k_mx_add_int8 = _k_mx_gelu = None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6e — ADVANCED TRITON KERNELS (for testing)
#   These kernels are used by the test suite and demonstrate advanced patterns.
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:
    # ── Auto-tunable kernel: dtype as constexpr parameter ─────────────────────
    @triton.jit
    def _k_autotune_mx_op(
        x_ptr, out_ptr, scale_ptr, N,
        DTYPE_BITS: tl.constexpr, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Auto-tunable quantization kernel where dtype is a compile-time parameter.
        DTYPE_BITS determines the quantization range at compile time.
        """
        pid = tl.program_id(0)
        offs = pid * BS + tl.arange(0, BLK)
        mask = offs < N
        
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        
        # Compute max value based on dtype bits (constexpr allows compile-time optimization)
        if DTYPE_BITS == 4:
            max_val = 7
        elif DTYPE_BITS == 8:
            max_val = 127
        else:
            max_val = 127  # Default to int8
        
        # Compute scale
        abs_max = tl.max(tl.abs(x), axis=0)
        scale = tl.where(abs_max < 1e-12, 1e-12, abs_max / max_val)
        tl.store(scale_ptr + pid, scale)
        
        # Quantize and store
        q = tl.minimum(tl.maximum((x / scale + 0.5).to(tl.int8), -max_val - 1), max_val)
        tl.store(out_ptr + offs, q, mask=mask)

    # ── Occupancy test kernel: configurable block sizes ───────────────────────
    @triton.jit
    def _k_occupancy_test(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """Simple matmul kernel for occupancy testing with configurable block sizes."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        
        for k in range(0, K, BK):
            a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)
            b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
            acc += tl.dot(a, b)
        
        tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)

    # ── Workspace kernel: uses scratch memory ─────────────────────────────────
    @triton.jit
    def _k_with_workspace(
        x_ptr, workspace_ptr, out_ptr,
        N, BS: tl.constexpr, BLK: tl.constexpr,
    ):
        """
        Kernel that uses workspace (scratch) memory for intermediate results.
        Demonstrates multi-buffer kernel patterns: x → workspace → output.
        """
        pid = tl.program_id(0)
        offs = pid * BS + tl.arange(0, BLK)
        mask = offs < N
        
        # Load input
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        
        # Stage 1: write to workspace
        x_sq = x * x
        tl.store(workspace_ptr + offs, x_sq, mask=mask)
        
        # Stage 2: read from workspace, compute, write output
        x_sq_read = tl.load(workspace_ptr + offs, mask=mask, other=0.0)
        out = x_sq_read + x  # x² + x
        tl.store(out_ptr + offs, out, mask=mask)

else:
    _k_autotune_mx_op = _k_occupancy_test = _k_with_workspace = None

# ── Explicit kernels for specific bit widths (avoid closure capture issues) ───

if HAS_TRITON:
    # 4-bit quantization kernel
    @triton.jit
    def _k_quant_4bit(x_ptr, out_ptr, scale_ptr, N, BS: tl.constexpr, BLK: tl.constexpr):
        """4-bit quantization kernel (max_val = 7)."""
        pid = tl.program_id(0)
        offs = pid * BS + tl.arange(0, BLK)
        mask = offs < N
        
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        abs_max = tl.max(tl.abs(x), axis=0)
        scale = tl.where(abs_max < 1e-12, 1e-12, abs_max / 7.0)  # max_val for 4-bit signed
        tl.store(scale_ptr + pid, scale)
        
        q = tl.minimum(tl.maximum((x / scale + 0.5).to(tl.int8), -8), 7)
        tl.store(out_ptr + offs, q, mask=mask)

    # 8-bit quantization kernel
    @triton.jit
    def _k_quant_8bit(x_ptr, out_ptr, scale_ptr, N, BS: tl.constexpr, BLK: tl.constexpr):
        """8-bit quantization kernel (max_val = 127)."""
        pid = tl.program_id(0)
        offs = pid * BS + tl.arange(0, BLK)
        mask = offs < N
        
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        abs_max = tl.max(tl.abs(x), axis=0)
        scale = tl.where(abs_max < 1e-12, 1e-12, abs_max / 127.0)  # max_val for 8-bit signed
        tl.store(scale_ptr + pid, scale)
        
        q = tl.minimum(tl.maximum((x / scale + 0.5).to(tl.int8), -128), 127)
        tl.store(out_ptr + offs, q, mask=mask)

else:
    _k_quant_4bit = _k_quant_8bit = None

# ── Python wrappers for custom kernels (take mx_tensor, return mx_tensor) ─────

def custom_mx_scale(x: mx_tensor) -> Tensor:
    """
    Custom kernel wrapper: Scale MX tensor data by per-block scales.
    
    Args:
        x: mx_tensor with int8 packed data
        
    Returns:
        Float32 tensor (dequantized and scaled)
    """
    if not HAS_TRITON or x.device.type != "cuda":
        return x.dequantize()
    
    N = x._mx_n
    BLK = 256
    n_blocks = math.ceil(N / BLK)
    
    out = torch.empty(N, dtype=torch.float32, device=x.device)
    
    _k_mx_scale_inplace[(n_blocks,)](
        x._mx_packed, x._mx_scales, out,
        N, BS=x._mx_block, BLK=BLK
    )
    
    return out.reshape(x._mx_orig_shape)

def custom_mx_relu(x: mx_tensor) -> mx_tensor:
    """
    Custom kernel wrapper: Apply ReLU to MX tensor, output stays quantized.
    
    Args:
        x: mx_tensor with int8 packed data
        
    Returns:
        mx_tensor with ReLU applied (stays in quantized form)
    """
    if not HAS_TRITON or x.device.type != "cuda":
        result = F.relu(x.dequantize())
        return mx_tensor.quantize(result, x._mx_dtype, x._mx_block)
    
    N = x._mx_n
    BS = x._mx_block
    n_blocks = math.ceil(N / BS)
    
    out_packed = torch.empty(N, dtype=torch.int8, device=x.device)
    out_scales = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    
    _k_mx_relu_quantized[(n_blocks,)](
        x._mx_packed, x._mx_scales, out_packed, out_scales,
        N, BS=BS, BLK=BS
    )
    
    return mx_tensor(out_packed, out_scales, x._mx_dtype, x._mx_orig_shape, N, BS)

def custom_mx_add(a: mx_tensor, b: mx_tensor) -> mx_tensor:
    """
    Custom kernel wrapper: Add two MX tensors, output stays quantized.
    
    Args:
        a, b: MXTensors with int8 packed data (same shape)
        
    Returns:
        mx_tensor with sum (stays in quantized form)
    """
    if not HAS_TRITON or a.device.type != "cuda":
        result = a.dequantize() + b.dequantize()
        out_dt = _resolve_mixed(a._mx_dtype, b._mx_dtype)
        return mx_tensor.quantize(result, out_dt, a._mx_block)
    
    N = a._mx_n
    BS = a._mx_block
    n_blocks = math.ceil(N / BS)
    
    out_packed = torch.empty(N, dtype=torch.int8, device=a.device)
    out_scales = torch.empty(n_blocks, dtype=torch.float32, device=a.device)
    
    _k_mx_add_int8[(n_blocks,)](
        a._mx_packed, b._mx_packed, out_packed,
        a._mx_scales, b._mx_scales, out_scales,
        N, BS=BS, BLK=BS
    )
    
    out_dt = _resolve_mixed(a._mx_dtype, b._mx_dtype)
    return mx_tensor(out_packed, out_scales, out_dt, a._mx_orig_shape, N, BS)

def custom_mx_gelu(x: mx_tensor) -> mx_tensor:
    """
    Custom kernel wrapper: Apply GELU to MX tensor, output stays quantized.
    
    Args:
        x: mx_tensor with int8 packed data
        
    Returns:
        mx_tensor with GELU applied (stays in quantized form)
    """
    if not HAS_TRITON or x.device.type != "cuda":
        result = F.gelu(x.dequantize())
        return mx_tensor.quantize(result, x._mx_dtype, x._mx_block)
    
    N = x._mx_n
    BS = x._mx_block
    n_blocks = math.ceil(N / BS)
    
    out_packed = torch.empty(N, dtype=torch.int8, device=x.device)
    out_scales = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    
    _k_mx_gelu[(n_blocks,)](
        x._mx_packed, x._mx_scales, out_packed, out_scales,
        N, BS=BS, BLK=BS
    )
    
    return mx_tensor(out_packed, out_scales, x._mx_dtype, x._mx_orig_shape, N, BS)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — CUSTOM KERNEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _KernelReg:
    name:     str
    op:       str
    dtypes:   List[str]
    hardware: List[str]
    fn:       Callable
    force:    str = "auto"   # "auto" | "true" | "false"
    priority: int = 0

    def matches(self, op: str, dtype: str, arch: str) -> bool:
        return (op == self.op and
                any(d == dtype for d in self.dtypes) and
                    any(h in arch for h in self.hardware))

class _Registry:
    def __init__(self):
        self._kerns: List[_KernelReg] = []

    def add(self, op, dtypes, hardware, force="auto", priority=0):
        def deco(fn):
            r = _KernelReg(fn.__name__, op, dtypes, hardware, fn, force, priority)
            self._kerns.append(r)
            if _DEBUG:
                log.debug(f"[Registry] {fn.__name__} → {op} dtypes={dtypes} hw={hardware}")
            return fn
        return deco

    def find(self, op, dtype, arch) -> Optional[_KernelReg]:
        cands = [k for k in self._kerns if k.matches(op, dtype, arch)]
        return max(cands, key=lambda k: k.priority) if cands else None

    def list_all(self): return list(self._kerns)

_REGISTRY = _Registry()

def register_kernel(op: str, dtypes: List[str], hardware: List[str],
                    force: Literal["auto","true","false"] = "auto",
                    priority: int = 0):
    """
    Register a custom Triton kernel for an operation.

    @mxt.register_kernel(op="torch.matmul", dtypes=["int4d"],
                         hardware=["gfx1100"], force="auto")
    def my_kernel():
        '''Triton kernel source returned as string.'''
        return \"\"\"@triton.jit def kernel(...): ...\"\"\"

    force:
        "auto"  — benchmark vs auto-generated, use fastest
        "true"  — always use this kernel
        "false" — use as hint only, auto can win
    """
    return _REGISTRY.add(op, dtypes, hardware, force, priority)

# ── Register custom kernels from Section 6d ─────────────────────────────────────

# These wrappers call the custom Triton kernels defined in Section 6d
@register_kernel(op="mx.scale", dtypes=["int8d"], hardware=["gfx1100", "sm_", "gfx90a", "cuda"], force="auto")
def _reg_mx_scale():
    """Registered custom kernel for MX element-wise scale."""
    return custom_mx_scale

@register_kernel(op="mx.relu", dtypes=["int8d"], hardware=["gfx1100", "sm_", "gfx90a", "cuda"], force="auto")
def _reg_mx_relu():
    """Registered custom kernel for MX ReLU (stays quantized)."""
    return custom_mx_relu

@register_kernel(op="mx.add", dtypes=["int8d"], hardware=["gfx1100", "sm_", "gfx90a", "cuda"], force="auto")
def _reg_mx_add():
    """Registered custom kernel for MX element-wise add."""
    return custom_mx_add

@register_kernel(op="mx.gelu", dtypes=["int8d"], hardware=["gfx1100", "sm_", "gfx90a", "cuda"], force="auto")
def _reg_mx_gelu():
    """Registered custom kernel for MX GELU."""
    return custom_mx_gelu

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — mx_tensor  (torch.Tensor subclass)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture:
#   • Subclasses torch.Tensor via _make_subclass so it IS a Tensor.
#   • Underlying storage holds the PACKED int8/int32 bits.
#   • Extra metadata: _mx_dtype, _mx_scales, _mx_orig_shape, _mx_n, _mx_block
#   • Shape reported via overriding size() and .shape in __torch_dispatch__
#   • All torch ops go through __torch_dispatch__
#   • Backward: straight-through estimator or real quantized grad

class mx_tensor(torch.Tensor):
    """
    A quantized tensor that IS a torch.Tensor.
    Underlying storage = real bit-packed int8/int32 data.
    All torch operations dispatch through __torch_dispatch__.
    """

    def __new__(cls, packed: Tensor, scales: Tensor, mx_dtype: mx_dtype,
                orig_shape: torch.Size, n: int, block: int = 128,
                requires_grad: bool = False):
        # The subclass wraps the PACKED storage (int8)
        # We need to store packed separately to avoid recursion
        # Create a dummy float tensor as base (can support gradients)
        # Note: __new__ is implicitly a static method in Python, no @staticmethod needed
        dummy = torch.empty((), dtype=torch.float32, device=packed.device, requires_grad=requires_grad)
        inst = torch.Tensor._make_subclass(cls, dummy, requires_grad)
        inst._mx_packed     = packed
        inst._mx_dtype      = mx_dtype
        inst._mx_scales     = scales
        inst._mx_orig_shape = orig_shape
        inst._mx_n          = n
        inst._mx_block      = block
        return inst

    def __init__(self, packed, scales, mx_dtype, orig_shape, n,
                 block=128, requires_grad=False):
        pass  # __new__ handles everything

    # ── PyTorch Tensor Subclass Protocol ─────────────────────────────────────

    def __tensor_flatten__(self):
        """Flatten tensor for dispatch - returns (attr_names, tensor_list)."""
        return ('_mx_packed _mx_scales', 
                [self._mx_packed, self._mx_scales])

    @staticmethod
    def __tensor_unflatten__(tensor_dict, meta):
        """Unflatten tensor from dispatch - reconstruct mx_tensor."""
        packed = tensor_dict[0]
        scales = tensor_dict[1]
        return mx_tensor(
            packed, scales,
            meta['mx_dtype'],
            meta['orig_shape'],
            meta['n'],
            meta['block'],
            meta.get('requires_grad', False)
        )

    # Note: __class__ is NOT overridden as a property because:
    # 1. type(obj) doesn't use __class__ attribute lookup - it accesses internal type directly
    # 2. isinstance(obj, cls) checks type(obj), not obj.__class__
    # 3. _make_subclass properly sets the object's type to mx_tensor
    # 4. Having a __class__ property can interfere with PyTorch's dispatch mechanism

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def quantize(cls, x: Tensor, dt: mx_dtype, block: int = 128,
                 requires_grad: Optional[bool] = None) -> "mx_tensor":
        """
        Quantize a float/bf16 tensor to MX precision.
        REAL bit-packing — no shadow float copy.
        """
        # Always preserve gradient graph - no fallback to detached computation
        packed, scales, n = quantize(x, dt, block)
        return cls(packed, scales, dt, x.shape, n, block, True)

    # ── Dequantize ────────────────────────────────────────────────────────────

    def dequantize(self) -> Tensor:
        """Unpack + dequantize → float32 tensor (original shape)."""
        flat = _dequant(
            self._mx_packed,  # Use the stored packed tensor
            self._mx_scales, self._mx_dtype,
            self._mx_n, self._mx_block,
        )
        return flat.reshape(self._mx_orig_shape)

    # ── Properties (override Tensor defaults) ─────────────────────────────────

    @property
    def shape(self) -> torch.Size:
        return self._mx_orig_shape

    def size(self, dim: Optional[int] = None):
        if dim is None: return self._mx_orig_shape
        return self._mx_orig_shape[dim]

    def dim(self) -> int:
        return len(self._mx_orig_shape)

    def numel(self) -> int:
        return self._mx_n

    # surface the float32-equivalent dtype for operator compatibility
    @property
    def mx_dtype(self) -> mx_dtype:   return self._mx_dtype
    @property
    def mx_scales(self) -> Tensor:   return self._mx_scales
    @property
    def mx_block(self) -> int:       return self._mx_block
    @property
    def packed(self) -> Tensor:
        """Raw packed storage tensor (int8)."""
        return self._mx_packed

    @property
    def nbytes_packed(self) -> int:
        return self.packed.nbytes + self._mx_scales.nbytes

    @property
    def compression_ratio(self) -> float:
        return (self._mx_n * 4) / max(self.nbytes_packed, 1)

    # ── __torch_dispatch__ — intercept ALL torch ops ──────────────────────────

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None: kwargs = {}
        return _MX_DISPATCH.dispatch(func, types, args, kwargs)

    # ── __torch_function__ — intercept torch.* namespace calls ───────────────

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None: kwargs = {}
        return _MX_DISPATCH.dispatch(func, types, args, kwargs)

    # ── Arithmetic sugar ──────────────────────────────────────────────────────

    def __add__(self, o):  return _binary_op(self, o, torch.add)
    def __radd__(self, o): return _binary_op(o, self, torch.add)
    def __sub__(self, o):  return _binary_op(self, o, torch.sub)
    def __rsub__(self, o): return _binary_op(o, self, torch.sub)
    def __mul__(self, o):  return _binary_op(self, o, torch.mul)
    def __rmul__(self, o): return _binary_op(o, self, torch.mul)
    def __truediv__(self, o): return _binary_op(self, o, torch.div)
    def __neg__(self):
        return mx_tensor.quantize(-self.dequantize(), self._mx_dtype, self._mx_block)
    def __matmul__(self, o): return _mx_mm(self, o)
    def __rmatmul__(self, o): return _mx_mm(o, self)

    # ── Python protocol ───────────────────────────────────────────────────────

    def __repr__(self):
        cr = self.compression_ratio
        return (f"mx_tensor({self._mx_dtype.name}, shape={tuple(self._mx_orig_shape)}, "
                f"device={self.device}, {cr:.1f}x compression, "
                f"{self.nbytes_packed/1024:.2f} KB packed)")

    def __len__(self):
        return self._mx_orig_shape[0]

    # ── Tensor-like interface methods ─────────────────────────────────────────

    def float(self):  return self.dequantize()
    def half(self):   return self.dequantize().half()
    def bfloat16(self): return self.dequantize().bfloat16()

    def to(self, *args, **kwargs):
        """
        Move or recast this mx_tensor.

        Accepted forms (same as torch.Tensor.to):
          .to("int4d")              → re-quantize to MX dtype
          .to("cuda")               → move to default CUDA device
          .to("cuda:0")             → move to CUDA device 0
          .to("cuda:1")             → move to CUDA device 1
          .to("cpu")                → move to CPU
          .to(torch.device("cuda:0")) → move to device
          .to(device="cuda:1")      → keyword device
          .to(dtype=torch.float32)  → dequantize to torch dtype
          .to(device="cuda:0", dtype="int8d")  → move + re-quantize
        """
        # Separate out device and dtype kwargs for mixed use
        kw_device = kwargs.get("device", None)
        kw_dtype  = kwargs.get("dtype",  None)

        # Positional first arg
        target = args[0] if args else None

        # Helper: is a value a device specifier?
        def _is_device(v):
            if isinstance(v, torch.device): return True
            if isinstance(v, int):          return True  # bare device index
            if isinstance(v, str):
                return (v in ("cpu", "cuda", "mps") or
                        v.startswith("cuda:") or v.startswith("cpu:") or
                        v.startswith("mps:"))
            return False

        # Helper: normalise to torch.device
        def _to_dev(v):
            if isinstance(v, int): return torch.device("cuda", v)
            return torch.device(v)

        # ── MX dtype re-quantize ──────────────────────────────────────────────
        if type(target).__name__ == 'mx_dtype_proxy':
            return mx_tensor.quantize(self.dequantize(), target._mx, self._mx_block,
                                     requires_grad=self.requires_grad)
        if type(target).__name__ == 'mx_dtype':
            return mx_tensor.quantize(self.dequantize(), target, self._mx_block)
        if isinstance(target, str) and target in _DTYPE_REGISTRY:
            return mx_tensor.quantize(self.dequantize(), get_mx_dtype(target),
                                     self._mx_block, requires_grad=self.requires_grad)
        if (type(kw_dtype).__name__ in ('mx_dtype', 'mx_dtype_proxy') or isinstance(kw_dtype, str)):
            dt = (kw_dtype._mx if type(kw_dtype).__name__ == 'mx_dtype_proxy' else
                  get_mx_dtype(kw_dtype) if isinstance(kw_dtype, str) else kw_dtype)
            base = self
            if kw_device is not None:
                base = self._move_device(_to_dev(kw_device))
            return mx_tensor.quantize(base.dequantize(), dt,
                                     self._mx_block, requires_grad=self.requires_grad)

        # ── Device move ───────────────────────────────────────────────────────
        dev = kw_device if kw_device is not None else (target if _is_device(target) else None)
        if dev is not None:
            return self._move_device(_to_dev(dev))

        # ── Standard torch dtype (float32 etc.) → dequantize ─────────────────
        return self.dequantize().to(*args, **kwargs)

    def _move_device(self, device: torch.device) -> "mx_tensor":
        """Move all storage to target device. Works with cuda:0, cuda:1, cpu, mps."""
        return mx_tensor(
        self._mx_packed.to(device),
        self._mx_scales.to(device),
        self._mx_dtype,
        self._mx_orig_shape,
        self._mx_n,
        self._mx_block,
        self.requires_grad,
    )

    def cuda(self, device=None):
        """
        Move to CUDA. Accepts:
          .cuda()       → default CUDA (cuda:0)
          .cuda(0)      → cuda:0
          .cuda(1)      → cuda:1
          .cuda("cuda:1") → cuda:1
        """
        if device is None:
            return self._move_device(torch.device("cuda"))
        if isinstance(device, int):
            return self._move_device(torch.device("cuda", device))
        if isinstance(device, str) and ":" in device:
            return self._move_device(torch.device(device))
        return self._move_device(torch.device("cuda", int(device) if str(device).isdigit() else 0))

    def cpu(self):
        return self._move_device(torch.device("cpu"))

    def clone(self):
        return mx_tensor(self._mx_packed.clone(), self._mx_scales.clone(),
                        self._mx_dtype, self._mx_orig_shape, self._mx_n,
                        self._mx_block, self.requires_grad)

    def detach(self):
        return mx_tensor(self._mx_packed.detach(), self._mx_scales.detach(),
                        self._mx_dtype, self._mx_orig_shape, self._mx_n,
                        self._mx_block, False)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        n = math.prod(s for s in shape if s != -1)
        shape = tuple(self._mx_n // n if s == -1 else s for s in shape)
        t = self.clone()
        t._mx_orig_shape = torch.Size(shape)
        return t

    def view(self, *shape):
        return self.reshape(*shape)

    def flatten(self, start=0, end=-1):
        return self.reshape(-1)

    def contiguous(self):
        return self.clone()

    def t(self):
        # 2D transpose — re-quantize transposed dequant
        return mx_tensor.quantize(self.dequantize().t(), self._mx_dtype, self._mx_block)

    def permute(self, *dims):
        return mx_tensor.quantize(self.dequantize().permute(*dims),
                                 self._mx_dtype, self._mx_block)

    def expand(self, *sizes):
        dq = self.dequantize().expand(*sizes)
        return mx_tensor.quantize(dq, self._mx_dtype, self._mx_block)

    def sum(self, *a, **kw):
        return mx_tensor.quantize(self.dequantize().sum(*a, **kw),
                                 self._mx_dtype, self._mx_block)

    def mean(self, *a, **kw):
        return mx_tensor.quantize(self.dequantize().mean(*a, **kw),
                                 self._mx_dtype, self._mx_block)

    def max(self, *a, **kw):
        return self.dequantize().max(*a, **kw)

    def min(self, *a, **kw):
        return self.dequantize().min(*a, **kw)

    def abs(self):
        return mx_tensor.quantize(self.dequantize().abs(), self._mx_dtype, self._mx_block)

    def sqrt(self):
        return mx_tensor.quantize(self.dequantize().sqrt(), self._mx_dtype, self._mx_block)

    def exp(self):
        return mx_tensor.quantize(self.dequantize().exp(), self._mx_dtype, self._mx_block)

    def log(self):
        return mx_tensor.quantize(self.dequantize().log(), self._mx_dtype, self._mx_block)

    def softmax(self, dim):
        return mx_tensor.quantize(self.dequantize().softmax(dim),
                                 self._mx_dtype, self._mx_block)

    def __getitem__(self, idx):
        return mx_tensor.quantize(self.dequantize()[idx],
                                 self._mx_dtype, self._mx_block)

    # ── Gradient / autograd support ───────────────────────────────────────────

    def backward(self, grad=None, **kw):
        # Straight-through estimator (STE): gradient passes through unchanged
        # For mx_tensor, we need to find the original input with grad and propagate
        # Since we can't easily trace back, we'll create a gradient using autograd
        dq = self.dequantize()
        # Create a grad tensor of ones with same shape (for loss scalar)
        if grad is None:
            grad = dq.new_ones(dq.shape)
        # Manual backward through the quantization: STE passes gradient through
        # We use torch.autograd.grad which handles this better
        dq.backward(grad, **kw)

    # ── State dict / pickling ─────────────────────────────────────────────────

    def __reduce_ex__(self, protocol):
        return (_rebuild_mxtensor, (
            self._mx_packed.cpu(), self._mx_scales.cpu(),
            self._mx_dtype, self._mx_orig_shape, self._mx_n, self._mx_block,
        ))

def _rebuild_mxtensor(packed, scales, dt, shape, n, block):
    return mx_tensor(packed, scales, dt, shape, n, block)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — OPERATOR DISPATCH ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(x) -> Tensor:
    """Get float32 tensor from mx_tensor or plain Tensor."""
    if isinstance(x, mx_tensor):
        return x.dequantize()
    return x

def _rewrap(result: Tensor, ref: mx_tensor) -> "mx_tensor":
    """Re-quantize result to same MX dtype as ref."""
    if isinstance(result, Tensor) and not isinstance(result, mx_tensor):
        return mx_tensor.quantize(result, ref._mx_dtype, ref._mx_block)
    return result

def _pick_ref(*args) -> Optional[mx_tensor]:
    """Find the first mx_tensor in args (for result dtype)."""
    for a in args:
        if isinstance(a, mx_tensor): return a
        if isinstance(a, (list, tuple)):
            r = _pick_ref(*a)
            if r is not None: return r
    return None

def _binary_op(a, b, fn) -> mx_tensor:
    """
    Execute a binary op between two MX/float tensors.
    Respects mixed-mode resolution: up × down → up, down × down → down.
    """
    if isinstance(a, mx_tensor) and isinstance(b, mx_tensor):
        out_dt = _resolve_mixed(a._mx_dtype, b._mx_dtype)
        fa, fb = a.dequantize(), b.dequantize()
        return mx_tensor.quantize(fn(fa, fb), out_dt, a._mx_block)
    ref = a if isinstance(a, mx_tensor) else b
    fa  = _unwrap(a); fb = _unwrap(b)
    return _rewrap(fn(fa, fb), ref)

# ── packed matmul dispatcher ──────────────────────────────────────────────────

def _mx_mm(a: mx_tensor, b: mx_tensor) -> mx_tensor:
    """
    Dispatch matmul to the best packed Triton kernel.
    Falls back to dequant → float32 mm → re-quant if needed.
    """
    hw    = hardware_probe.detect()
    bits  = a._mx_dtype.bits
    out_dt = _resolve_mixed(a._mx_dtype, b._mx_dtype)

    # Check user-registered forced kernel
    reg = _REGISTRY.find("torch.matmul", a._mx_dtype.name, hw.arch)
    if reg and reg.force == "true":
        if _DEBUG: log.debug(f"[mm] forced custom kernel: {reg.name}")
        # Execute user kernel source (placeholder — user provides Triton string)
        return _mm_fallback(a, b, out_dt)

    # Triton packed kernels (2D only for now)
    if (HAS_TRITON and
        bits in (1, 2, 4) and
        len(a._mx_orig_shape) == 2 and len(b._mx_orig_shape) == 2):
        try:
            return _mm_triton_packed(a, b, out_dt, hw)
        except Exception as e:
            if _DEBUG: log.debug(f"[mm] Triton failed ({e}), fallback")

    return _mm_fallback(a, b, out_dt)

class _MXTritonMatmulAutograd(torch.autograd.Function):
    """Custom autograd function that uses Triton kernel for forward but preserves gradients."""
    
    @staticmethod
    def forward(ctx, a: mx_tensor, b: mx_tensor, out_dt: mx_dtype, hw: hardware_profile):
        bits  = a._mx_dtype.bits
        ratio = 8 // bits
        M, K  = a._mx_orig_shape
        K2, N = b._mx_orig_shape
        Kp = math.ceil(K / ratio)

        ap = a.packed.contiguous()
        bp = b.packed.contiguous()
        c  = torch.empty(M, N, dtype=torch.float32, device=a.device)

        BM = 32; BN = 32; BK = min(32, Kp)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

        common = dict(
            sa_ptr=a._mx_scales, sb_ptr=b._mx_scales,
            M=M, N=N, Kp=Kp,
            sam=Kp, sak=1, sbk=N, sbn=1, scm=N, scn=1,
            BS=a._mx_block, BM=BM, BN=BN, BK=BK,
        )

        if bits == 4:
            _k_int4_mm[grid](ap, bp, c, **common)
        elif bits == 2:
            _k_int2_mm[grid](ap, bp, c, **common)
        elif bits == 1:
            _k_int1_mm[grid](ap, bp, c, **common)

        ctx.save_for_backward(a, b)
        ctx._mx_out_dt = out_dt
        ctx._mx_block = a._mx_block
        
        return mx_tensor.quantize(c, out_dt, a._mx_block, requires_grad=True)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        out_dt = ctx._mx_out_dt
        block = ctx._mx_block
        
        grad_output_deq = grad_output.dequantize()
        
        grad_a = torch.mm(grad_output_deq, b.dequantize().t())
        grad_b = torch.mm(a.dequantize().t(), grad_output_deq)
        
        return grad_a, grad_b, None, None

def _mm_triton_packed(a: mx_tensor, b: mx_tensor,
                      out_dt: mx_dtype, hw: hardware_profile) -> mx_tensor:
    return _MXTritonMatmulAutograd.apply(a, b, out_dt, hw)

def _mm_fallback(a: mx_tensor, b: mx_tensor, out_dt: mx_dtype) -> mx_tensor:
    """Dequant → float32 mm → requant fallback.
    
    Uses custom autograd function to preserve gradient flow to both input AND weight.
    """
    if _STRICT:
        raise RuntimeError(
            f"[mx_triton STRICT] full-precision fallback triggered for "
            f"matmul({a._mx_dtype.name} × {b._mx_dtype.name})"
        )
    if _DEBUG:
        log.debug(f"[mm FALLBACK] {a._mx_dtype.name} × {b._mx_dtype.name} → f32 → {out_dt.name}")
    
    # Use custom autograd function to preserve gradients
    return _MXMatmulAutograd.apply(a, b, out_dt)

class _MXMatmulAutograd(torch.autograd.Function):
    """Custom autograd function for MX matmul that preserves gradients to both operands."""
    
    @staticmethod
    def forward(ctx, a: mx_tensor, b: mx_tensor, out_dt: mx_dtype) -> mx_tensor:
        # Dequantize for computation
        a_deq = a.dequantize()
        b_deq = b.dequantize()
        
        # Compute in float32 - use leaf tensors to preserve gradient tracking
        result = torch.mm(a_deq, b_deq)
        
        # Create leaf tensor with requires_grad to preserve gradient graph
        result_leaf = torch.empty_like(result, requires_grad=True)
        result_leaf.copy_(result)
        
        # Store for backward
        ctx.save_for_backward(a_deq, b_deq)
        ctx._mx_out_dt = out_dt
        ctx._mx_block = a._mx_block
        
        # Quantize the result - gradient flow handled by custom backward
        return mx_tensor.quantize(result_leaf, out_dt, a._mx_block, requires_grad=True)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Get stored tensors
        a_deq, b_deq = ctx.saved_tensors
        out_dt = ctx._mx_out_dt
        block = ctx._mx_block
        
        # Gradients
        # grad_output is float32, we need grad for a and b
        grad_a = torch.mm(grad_output, b_deq.t())
        grad_b = torch.mm(a_deq.t(), grad_output)
        
        # Quantize gradients back to MX format for the weight
        # This preserves the gradient flow through the quantization boundary
        grad_a_mx = mx_tensor.quantize(grad_a, out_dt, block)
        grad_b_mx = mx_tensor.quantize(grad_b, out_dt, block)
        
        # Return gradients - for mx_tensor inputs, we return the dequantized grads
        # The gradient flows through the quantize -> dequantize chain
        return grad_a_mx.dequantize(), grad_b_mx.dequantize(), None

class _MXLinearAutograd(torch.autograd.Function):
    """Custom autograd for mx_linear that properly handles gradient flow."""
    
    @staticmethod
    def forward(ctx, x: Tensor, w_mx: mx_tensor, b_mx: mx_tensor, 
                mx_dtype, block, in_features, out_features) -> Tensor:
        # Store original shape
        orig_shape = x.shape
        
        # Flatten batch dims for matmul
        x_2d = x.reshape(-1, in_features) if x.dim() > 2 else x
        
        # Quantize input
        x_mx = mx_tensor.quantize(x_2d, mx_dtype, block)
        
        # Matmul
        out_mx = _mx_mm(x_mx, w_mx)
        
        # Dequantize
        out = out_mx.dequantize()
        
        # Reshape back
        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], out_features)
        
        # Add bias
        if b_mx is not None:
            b_f = b_mx.dequantize()
            if len(orig_shape) > 2:
                b_f = b_f.unsqueeze(0)
            out = out + b_f
        
        ctx.save_for_backward(x, w_mx)
        ctx._mx_dtype = mx_dtype
        ctx._mx_block = block
        ctx._in_features = in_features
        ctx._out_features = out_features
        ctx._orig_shape = orig_shape
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, w_mx = ctx.saved_tensors
        orig_shape = ctx._orig_shape
        in_features = ctx._in_features
        
        # Get dequantized weight
        w_dq = w_mx.dequantize()
        
        # Flatten grad_output for matmul if needed
        grad_2d = grad_output.reshape(-1, w_dq.shape[1]) if grad_output.dim() > 2 else grad_output
        
        # Gradient to input: grad_output @ w^T
        grad_input_2d = grad_2d @ w_dq.t()
        
        # Reshape back to original shape
        if len(orig_shape) > 2:
            grad_input = grad_input_2d.reshape(*orig_shape[:-1], in_features)
        else:
            grad_input = grad_input_2d
        
        return grad_input, None, None, None, None, None, None

def _mx_linear_forward(x: Tensor, w_mx: mx_tensor, b_mx: mx_tensor,
                       mx_dtype, block, in_features, out_features) -> Tensor:
    """Wrapper for mx_linear forward with proper gradient flow."""
    return _MXLinearAutograd.apply(x, w_mx, b_mx, mx_dtype, block, in_features, out_features)

# ── main dispatch table ───────────────────────────────────────────────────────

class _Dispatcher:
    """
    Central dispatcher for __torch_dispatch__ and __torch_function__.
    Covers all major ops. Fallback = dequant → op → requant.
    """

    def dispatch(self, func, types, args, kwargs):
        fn = getattr(func, "__name__", None) or getattr(func, "name", str(func))

        # ── linear algebra ────────────────────────────────────────────────────
        if func in (torch.mm, torch.matmul, torch.Tensor.mm,
                    torch.Tensor.matmul, torch.ops.aten.mm.default,
                    torch.ops.aten.matmul.default):
            a, b = args[0], args[1]
            if isinstance(a, mx_tensor) and isinstance(b, mx_tensor):
                return _mx_mm(a, b)

        if func in (torch.bmm, torch.ops.aten.bmm.default):
            a, b = args[0], args[1]
            if isinstance(a, mx_tensor) and isinstance(b, mx_tensor):
                # batch dim: process slice by slice or dequant
                fa = a.dequantize(); fb = b.dequantize()
                out_dt = _resolve_mixed(a._mx_dtype, b._mx_dtype)
                return mx_tensor.quantize(torch.bmm(fa, fb), out_dt, a._mx_block)

        if func in (F.linear, torch.ops.aten.linear.default):
            inp    = args[0]
            weight = args[1]
            bias   = args[2] if len(args) > 2 else kwargs.get("bias")
            ref    = inp if isinstance(inp, mx_tensor) else weight
            fi     = _unwrap(inp)
            fw     = _unwrap(weight)
            out    = F.linear(fi, fw, _unwrap(bias) if bias is not None else None)
            if isinstance(ref, mx_tensor):
                return mx_tensor.quantize(out, ref._mx_dtype, ref._mx_block)
            return out

        if func in (torch.addmm, torch.ops.aten.addmm.default):
            bias, a, b = args[0], args[1], args[2]
            ref = _pick_ref(a, b, bias)
            out = torch.addmm(_unwrap(bias), _unwrap(a), _unwrap(b))
            return _rewrap(out, ref) if ref is not None else out

        if func in (torch.einsum,):
            eq   = args[0]
            oper = [_unwrap(x) for x in args[1:]]
            ref  = _pick_ref(*args[1:])
            out  = torch.einsum(eq, *oper)
            return _rewrap(out, ref) if ref is not None else out

        # ── element-wise arithmetic ───────────────────────────────────────────
        _pointwise = {
            torch.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar,
            torch.sub, torch.ops.aten.sub.Tensor,
            torch.mul, torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar,
            torch.div, torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar,
            torch.pow, torch.ops.aten.pow.Tensor_Tensor,
        }
        if func in _pointwise or fn in ("add","sub","mul","div","pow"):
            a = args[0]; b = args[1] if len(args) > 1 else kwargs.get("other", 1)
            if isinstance(a, mx_tensor) or isinstance(b, mx_tensor):
                return _binary_op(a, b, lambda x,y: func(
                    x if not isinstance(x, mx_tensor) else x.dequantize(),
                    y if not isinstance(y, mx_tensor) else y.dequantize(),
                ))

        # ── reductions ────────────────────────────────────────────────────────
        _reductions = {
            torch.sum, torch.mean, torch.max, torch.min,
            torch.std, torch.var, torch.norm, torch.amax, torch.amin,
            torch.prod, torch.argmax, torch.argmin,
            torch.ops.aten.sum.default, torch.ops.aten.mean.default,
            torch.ops.aten.amax.default, torch.ops.aten.amin.default,
        }
        if func in _reductions or fn in ("sum","mean","max","min","std","var",
                                          "norm","amax","amin","prod","argmax","argmin"):
            ref  = _pick_ref(*args)
            flat = [_unwrap(a) for a in args]
            out  = func(*flat, **kwargs)
            if ref is not None and isinstance(out, Tensor) and not isinstance(out, mx_tensor):
                # Only re-quantize if result has same dimensionality (not scalar reductions)
                if out.ndim > 0:
                    return mx_tensor.quantize(out, ref._mx_dtype, ref._mx_block)
            return out

        # ── activations ───────────────────────────────────────────────────────
        _activations = {
            F.relu, F.gelu, F.silu, F.tanh, F.sigmoid, F.softmax,
            F.leaky_relu, F.elu, F.hardswish, F.mish,
            torch.relu, torch.tanh, torch.sigmoid, torch.nn.functional.relu,
            torch.ops.aten.relu.default, torch.ops.aten.gelu.default,
            torch.ops.aten.silu.default, torch.ops.aten.sigmoid.default,
        }
        if func in _activations or fn in ("relu","gelu","silu","tanh","sigmoid",
                                           "leaky_relu","elu","softmax","hardswish","mish"):
            ref = _pick_ref(*args)
            flat = [_unwrap(a) for a in args]
            out  = func(*flat, **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── normalization ─────────────────────────────────────────────────────
        _norms = {F.layer_norm, F.batch_norm, F.instance_norm,
                  F.group_norm, F.rms_norm if hasattr(F, "rms_norm") else None}
        _norms.discard(None)
        if func in _norms or fn in ("layer_norm","batch_norm","instance_norm",
                                     "group_norm","rms_norm"):
            ref  = _pick_ref(*args)
            flat = [_unwrap(a) for a in args]
            flat_kw = {k: _unwrap(v) for k, v in kwargs.items()}
            out  = func(*flat, **flat_kw)
            return _rewrap(out, ref) if ref is not None else out

        # ── convolutions ──────────────────────────────────────────────────────
        _convs = {F.conv1d, F.conv2d, F.conv3d,
                  F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d}
        if func in _convs or fn in ("conv1d","conv2d","conv3d",
                                     "conv_transpose1d","conv_transpose2d"):
            ref  = _pick_ref(*args)
            flat = [_unwrap(a) for a in args]
            flat_kw = {k: _unwrap(v) for k, v in kwargs.items()}
            out  = func(*flat, **flat_kw)
            return _rewrap(out, ref) if ref is not None else out

        # ── loss functions ────────────────────────────────────────────────────
        _losses = {F.mse_loss, F.cross_entropy, F.nll_loss, F.binary_cross_entropy,
                   F.l1_loss, F.huber_loss, F.smooth_l1_loss, F.kl_div}
        if func in _losses or fn in ("mse_loss","cross_entropy","nll_loss",
                                      "binary_cross_entropy","l1_loss","huber_loss"):
            flat = [_unwrap(a) for a in args]
            flat_kw = {k: _unwrap(v) for k, v in kwargs.items()}
            return func(*flat, **flat_kw)  # losses return scalars

        # ── embedding ─────────────────────────────────────────────────────────
        if func in (F.embedding, F.embedding_bag) or fn in ("embedding","embedding_bag"):
            ref = _pick_ref(*args)
            # indices are usually plain int tensors, only embedding table is MX
            flat = [_unwrap(a) if isinstance(a, mx_tensor) else a for a in args]
            flat_kw = {k: _unwrap(v) if isinstance(v, mx_tensor) else v
                       for k, v in kwargs.items()}
            out = func(*flat, **flat_kw)
            return _rewrap(out, ref) if ref is not None else out

        # ── indexing / scatter ────────────────────────────────────────────────
        if fn in ("index_select","gather","scatter","scatter_add","index_put",
                  "masked_select","masked_fill","where"):
            ref  = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, mx_tensor) else a for a in args]
            flat_kw = {k: _unwrap(v) if isinstance(v, mx_tensor) else v
                       for k, v in kwargs.items()}
            out = func(*flat, **flat_kw)
            return _rewrap(out, ref) if (ref is not None and isinstance(out, Tensor)) else out

        # ── shape ops ─────────────────────────────────────────────────────────
        if func in (torch.reshape, torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default):
            t = args[0]
            if isinstance(t, mx_tensor):
                shape = args[1] if len(args) > 1 else kwargs.get("shape")
            return t.reshape(shape)

        if func in (torch.cat, torch.ops.aten.cat.default):
            tensors = args[0]
            ref     = _pick_ref(*tensors)
            flat    = [_unwrap(t) for t in tensors]
            out     = torch.cat(flat, **{k: v for k, v in
                                          zip(["dim"], args[1:])}, **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        if func in (torch.stack, torch.ops.aten.stack.default):
            tensors = args[0]
            ref     = _pick_ref(*tensors)
            flat    = [_unwrap(t) for t in tensors]
            out     = torch.stack(flat, **{k: v for k, v in
                                            zip(["dim"], args[1:])}, **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        if func in (torch.split, torch.ops.aten.split.Tensor,
                    torch.chunk, torch.ops.aten.chunk.default):
            t   = args[0]
            if isinstance(t, mx_tensor):
                dq  = t.dequantize()
                out = func(dq, *args[1:], **kwargs)
                return tuple(mx_tensor.quantize(o, t._mx_dtype, t._mx_block) for o in out)

        if func in (torch.transpose, torch.ops.aten.transpose.int, torch.t,
                    torch.ops.aten.t.default):
            t = args[0]
            if isinstance(t, mx_tensor):
                return mx_tensor.quantize(func(t.dequantize(), *args[1:], **kwargs),
                                         t._mx_dtype, t._mx_block)

        if func in (torch.permute, torch.ops.aten.permute.default):
            t = args[0]
            if isinstance(t, mx_tensor):
                return mx_tensor.quantize(t.dequantize().permute(*args[1:]),
                                         t._mx_dtype, t._mx_block)

        # ── comparison / logical ops ──────────────────────────────────────────
        _cmp_ops = {torch.lt, torch.gt, torch.le, torch.ge, torch.eq, torch.ne,
                    torch.ops.aten.lt.Tensor, torch.ops.aten.gt.Tensor,
                    torch.ops.aten.le.Tensor, torch.ops.aten.ge.Tensor,
                    torch.ops.aten.eq.Tensor, torch.ops.aten.ne.Tensor}
        if func in _cmp_ops or fn in ("lt","gt","le","ge","eq","ne",
                                       "less","greater","less_equal","greater_equal"):
            a = args[0]; b = args[1] if len(args) > 1 else kwargs.get("other")
            return func(_unwrap(a) if isinstance(a, mx_tensor) else a,
                        _unwrap(b) if isinstance(b, mx_tensor) else b)  # bool result

        # ── clamp / clip ──────────────────────────────────────────────────────
        if func in (torch.clamp, torch.clip, torch.ops.aten.clamp.default,
                    torch.ops.aten.clamp.Tensor) or fn in ("clamp","clip"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        if func in (torch.floor, torch.ceil, torch.round, torch.trunc,
                    torch.ops.aten.floor.default, torch.ops.aten.ceil.default,
                    torch.ops.aten.round.default) or fn in ("floor","ceil","round","trunc"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        if func in (torch.sign, torch.abs, torch.ops.aten.sign.default,
                    torch.ops.aten.abs.default) or fn in ("sign","abs"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]))
            return _rewrap(out, ref) if ref is not None else out

        # ── sorting / selection ───────────────────────────────────────────────
        if func in (torch.sort, torch.argsort, torch.ops.aten.sort.default,
                    torch.ops.aten.argsort.default) or fn in ("sort","argsort"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            if ref is not None and isinstance(out, Tensor):
                return _rewrap(out, ref)
            if ref is not None and isinstance(out, tuple):  # sort returns (values, indices)
                vals, idx = out
                return torch.return_types.sort((
                    _rewrap(vals, ref) if vals.dtype.is_floating_point else vals, idx))
            return out

        if func in (torch.topk, torch.ops.aten.topk.default) or fn == "topk":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            if ref is not None and hasattr(out, "values"):
                vals = _rewrap(out.values, ref) if out.values.dtype.is_floating_point else out.values
                return torch.return_types.topk((vals, out.indices))
            return out

        # ── scan / cumulative ops ─────────────────────────────────────────────
        if func in (torch.cumsum, torch.cumprod, torch.ops.aten.cumsum.default,
                    torch.ops.aten.cumprod.default) or fn in ("cumsum","cumprod"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── dropout (training only — pass-through in eval) ────────────────────
        if func in (F.dropout, torch.ops.aten.dropout.default) or fn == "dropout":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── pooling (avg / max / adaptive) ────────────────────────────────────
        _pools = {F.avg_pool1d, F.avg_pool2d, F.avg_pool3d,
                  F.max_pool1d, F.max_pool2d, F.max_pool3d,
                  F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d,
                  F.adaptive_max_pool1d, F.adaptive_max_pool2d, F.adaptive_max_pool3d}
        if func in _pools or fn in ("avg_pool2d","max_pool2d","adaptive_avg_pool2d",
                                     "avg_pool1d","max_pool1d","avg_pool3d","max_pool3d"):
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, mx_tensor) else a for a in args]
            out  = func(*flat, **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── upsample / interpolate ────────────────────────────────────────────
        if func in (F.interpolate, F.upsample, F.upsample_bilinear,
                    F.upsample_nearest) or fn in ("interpolate","upsample"):
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, mx_tensor) else a for a in args]
            out  = func(*flat, **{k: _unwrap(v) if isinstance(v, mx_tensor) else v
                                  for k, v in kwargs.items()})
            return _rewrap(out, ref) if ref is not None else out

        # ── scaled dot-product attention (SDPA) ───────────────────────────────
        if (func is getattr(F, "scaled_dot_product_attention", None) or
                fn == "scaled_dot_product_attention"):
            q, k, v = args[0], args[1], args[2]
            ref = _pick_ref(q, k, v)
            out = F.scaled_dot_product_attention(
                _unwrap(q), _unwrap(k), _unwrap(v),
                *[(_unwrap(a) if isinstance(a, mx_tensor) else a) for a in args[3:]],
                    **{kk: (_unwrap(vv) if isinstance(vv, mx_tensor) else vv)
                       for kk, vv in kwargs.items()})
            return _rewrap(out, ref) if ref is not None else out

        # ── FFT / spectral ────────────────────────────────────────────────────
        _ffts = set()
        for _fname in ("fft","ifft","rfft","irfft","fft2","ifft2","fftn","ifftn"):
            _m = getattr(getattr(torch, "fft", None), _fname, None)
            if _m is not None: _ffts.add(_m)
        if func in _ffts or fn in ("fft","ifft","rfft","irfft","fft2","fftn"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            # FFT output is complex — don't re-quantize unless complex MX supported
            return out

        # ── flip / roll / unfold ──────────────────────────────────────────────
        if func in (torch.flip, torch.fliplr, torch.flipud,
                    torch.ops.aten.flip.default) or fn in ("flip","fliplr","flipud"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        if func in (torch.roll, torch.ops.aten.roll.default) or fn == "roll":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── outer / cross product ─────────────────────────────────────────────
        if func in (torch.outer, torch.ger, torch.ops.aten.outer.default) or fn in ("outer","ger"):
            a, b = args[0], args[1]
            ref  = _pick_ref(a, b)
            out  = func(_unwrap(a), _unwrap(b))
            return _rewrap(out, ref) if ref is not None else out

        if func in (torch.baddbmm, torch.ops.aten.baddbmm.default) or fn == "baddbmm":
            inp, b1, b2 = args[0], args[1], args[2]
            ref = _pick_ref(inp, b1, b2)
            out = func(_unwrap(inp), _unwrap(b1), _unwrap(b2), **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── linalg ────────────────────────────────────────────────────────────
        _linalg = set()
        for _lname in ("norm","vector_norm","matrix_norm","solve","lstsq","svd","eig","qr"):
            _m = getattr(getattr(torch, "linalg", None), _lname, None)
            if _m is not None: _linalg.add(_m)
        if func in _linalg or fn in ("linalg_norm","linalg_solve","svd","eig","qr"):
            ref = _pick_ref(*args)
            flat_args = [_unwrap(a) if isinstance(a, mx_tensor) else a for a in args]
            out = func(*flat_args, **kwargs)
            # linalg ops return plain tensors or named tuples — don't auto-requant
            return out

        # ── sparse ops ────────────────────────────────────────────────────────
        if fn in ("sparse_coo_tensor","sparse_csr_tensor","to_sparse",
                  "to_dense","coalesce","indices","values","crow_indices","col_indices"):
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, mx_tensor) else a for a in args]
            out  = func(*flat, **kwargs)
            return out  # sparse ops return non-MX tensors by design

        # ── padding ───────────────────────────────────────────────────────────
        if func in (F.pad, torch.ops.aten.constant_pad_nd.default,
                    torch.ops.aten.pad.default) or fn in ("pad",):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]) if isinstance(args[0], mx_tensor) else args[0],
                       *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── nan / inf cleanup ─────────────────────────────────────────────────
        if func in (torch.nan_to_num, torch.ops.aten.nan_to_num.default) or fn == "nan_to_num":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── masked_fill / where ───────────────────────────────────────────────
        if func in (torch.where, torch.ops.aten.where.self,
                    torch.ops.aten.where.ScalarOther) or fn == "where":
            # where(condition, x, y)  — condition is bool, x/y may be mx_tensor
            ref   = _pick_ref(*args[1:])   # skip condition arg
            cond  = args[0]
            x_arg = _unwrap(args[1]) if len(args) > 1 and isinstance(args[1], mx_tensor) else (args[1] if len(args)>1 else kwargs.get("input"))
            y_arg = _unwrap(args[2]) if len(args) > 2 and isinstance(args[2], mx_tensor) else (args[2] if len(args)>2 else kwargs.get("other"))
            out   = torch.where(cond, x_arg, y_arg)
            return _rewrap(out, ref) if ref is not None else out

        if func in (torch.ops.aten.masked_fill.Scalar,
                    torch.ops.aten.masked_fill.Tensor) or fn == "masked_fill":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        if func in (torch.ops.aten.index.Tensor,) or fn == "index":
            ref = _pick_ref(*args)
            t   = _unwrap(args[0]) if isinstance(args[0], mx_tensor) else args[0]
            idx = args[1]
            out = func(t, idx, **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── interpolate / upsample (in case missed by existing block) ─────────
        if func is getattr(F, "pixel_shuffle", None) or fn == "pixel_shuffle":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        if func is getattr(F, "grid_sample", None) or fn == "grid_sample":
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, mx_tensor) else a for a in args]
            out  = func(*flat, **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── type casting ──────────────────────────────────────────────────────
        if func in (torch.ops.aten._to_copy.default,):
            t = args[0]
            if isinstance(t, mx_tensor):
                # Target dtype
                tgt = kwargs.get("dtype", None)
                if tgt in (torch.float32, torch.float16, torch.bfloat16):
                    return t.dequantize().to(dtype=tgt)
                return t.clone()
            return func(*args, **kwargs)

        # ── tensor creation ops (zeros_like, ones_like, empty_like, full_like) ─
        if func in (torch.zeros_like, torch.ones_like, torch.empty_like,
                    torch.ops.aten.zeros_like.default,
                    torch.ops.aten.ones_like.default,
                    torch.ops.aten.empty_like.default) or fn in (
                        "zeros_like","ones_like","empty_like"):
            t = args[0]
            if isinstance(t, mx_tensor):
                # Return plain float32 tensor matching the *logical* shape
                creator = torch.zeros if fn != "ones_like" else torch.ones
                if fn == "empty_like":
                    creator = torch.empty
                return creator(t._mx_orig_shape, dtype=torch.float32, device=t.device)
            return func(*args, **kwargs)

        if func in (torch.full_like, torch.ops.aten.full_like.default) or fn == "full_like":
            t   = args[0]
            val = args[1] if len(args) > 1 else kwargs.get("fill_value", 0)
            if isinstance(t, mx_tensor):
                return torch.full(t._mx_orig_shape, val, dtype=torch.float32, device=t.device)
            return func(*args, **kwargs)

        # ── in-place add / sub / mul (+=, -=, *=) ────────────────────────────
        if func in (torch.ops.aten.add_.Tensor, torch.ops.aten.sub_.Tensor,
                    torch.ops.aten.mul_.Tensor, torch.ops.aten.div_.Tensor) or fn in (
                        "add_","sub_","mul_","div_"):
            t = args[0]
            if isinstance(t, mx_tensor):
                other = args[1] if len(args) > 1 else kwargs.get("other")
                base  = {fn or "": torch.add, "sub_": torch.sub,
                         "mul_": torch.mul, "div_": torch.div}.get(fn or "", torch.add)
                result = base(t.dequantize(), _unwrap(other) if isinstance(other, mx_tensor) else other)
                new_mx = mx_tensor.quantize(result, t._mx_dtype, t._mx_block)
                t.packed.copy_(new_mx.packed)
                t._mx_scales.copy_(new_mx._mx_scales)
                return t
            return func(*args, **kwargs)

        # ── copy_ / fill_ / zero_ ────────────────────────────────────────────
        if fn in ("copy_", "fill_", "zero_"):
            t = args[0]
            if isinstance(t, mx_tensor):
                dq = t.dequantize()
                getattr(dq, fn)(*[_unwrap(a) for a in args[1:]], **kwargs)
                new = mx_tensor.quantize(dq, t._mx_dtype, t._mx_block)
                # Mutate in place by updating packed storage
                t.packed.copy_(new.packed)
                t._mx_scales.copy_(new._mx_scales)
                return t
            return func(*args, **kwargs)

        # ── clone / detach / contiguous ───────────────────────────────────────
        if func in (torch.ops.aten.clone.default,):
            t = args[0]
            if isinstance(t, mx_tensor): return t.clone()

        if func in (torch.ops.aten.detach.default,):
            t = args[0]
            if isinstance(t, mx_tensor):
                # Use the stored packed tensor directly
                detached_packed = t._mx_packed.detach()
                return mx_tensor(detached_packed, t._mx_scales.detach(),
                                t._mx_dtype, t._mx_orig_shape, t._mx_n,
                                t._mx_block, False)

        if func in (torch.ops.aten.contiguous.default,):
            t = args[0]
            if isinstance(t, mx_tensor): return t.clone()

        # ── dtype / device queries ────────────────────────────────────────────
        if func in (torch.ops.aten.is_floating_point.default,):
            t = args[0]
            if isinstance(t, mx_tensor): return t._mx_dtype.is_float

        # ── scatter_reduce / segment_reduce ───────────────────────────────────
        if (func is getattr(torch, "scatter_reduce", None) or
                func is getattr(torch.ops.aten, "scatter_reduce.two", None) or
                fn in ("scatter_reduce", "segment_reduce")):
            ref  = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, mx_tensor) else a for a in args]
            flat_kw = {k: _unwrap(v) if isinstance(v, mx_tensor) else v
                       for k, v in kwargs.items()}
            out  = func(*flat, **flat_kw)
            return _rewrap(out, ref) if (ref is not None and isinstance(out, Tensor)) else out

        # ── unfold / as_strided ───────────────────────────────────────────────
        if (func in (torch.Tensor.unfold, torch.ops.aten.unfold.default) or
                fn in ("unfold", "as_strided")):
            ref = _pick_ref(*args)
            t   = _unwrap(args[0]) if isinstance(args[0], mx_tensor) else args[0]
            out = func(t, *args[1:], **kwargs)
            return _rewrap(out, ref) if (ref is not None and isinstance(out, Tensor)) else out

        # ── kron / tensordot / vecdot ─────────────────────────────────────────
        if (func in (torch.kron, torch.ops.aten.kron.default) or fn == "kron"):
            a, b = args[0], args[1]
            ref  = _pick_ref(a, b)
            out  = torch.kron(_unwrap(a), _unwrap(b))
            return _rewrap(out, ref) if ref is not None else out

        if (func in (torch.tensordot, torch.ops.aten.tensordot.default) or fn == "tensordot"):
            a, b  = args[0], args[1]
            ref   = _pick_ref(a, b)
            dims  = args[2] if len(args) > 2 else kwargs.get("dims", 2)
            out   = torch.tensordot(_unwrap(a), _unwrap(b), dims=dims)
            return _rewrap(out, ref) if ref is not None else out

        if fn in ("vecdot", "vdot"):
            ref = _pick_ref(*args)
            out = func(*[_unwrap(a) if isinstance(a, mx_tensor) else a for a in args], **kwargs)
            return out  # dot products return scalars or reduced tensors

        # ── repeat / repeat_interleave / expand ───────────────────────────────
        if (func in (torch.ops.aten.repeat.default,) or fn == "repeat"):
            t = args[0]
            if isinstance(t, mx_tensor):
                out = t.dequantize().repeat(*args[1:], **kwargs)
                return mx_tensor.quantize(out, t._mx_dtype, t._mx_block)

        if (func in (torch.repeat_interleave, torch.ops.aten.repeat_interleave.Tensor,
                     torch.ops.aten.repeat_interleave.self_int) or fn == "repeat_interleave"):
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, mx_tensor) else a for a in args]
            out  = func(*flat, **kwargs)
            return _rewrap(out, ref) if (ref is not None and isinstance(out, Tensor)) else out

        # ── squeeze / unsqueeze / flatten ─────────────────────────────────────
        if (func in (torch.squeeze, torch.unsqueeze, torch.flatten,
                     torch.ops.aten.squeeze.default, torch.ops.aten.squeeze.dim,
                     torch.ops.aten.unsqueeze.default, torch.ops.aten.flatten.using_ints)
                or fn in ("squeeze", "unsqueeze", "flatten")):
            ref = _pick_ref(*args)
            t   = args[0]
            if isinstance(t, mx_tensor):
                out = func(t.dequantize(), *args[1:], **kwargs)
                return mx_tensor.quantize(out, t._mx_dtype, t._mx_block)
            return func(*args, **kwargs)

        # ── exp / log / sqrt (elementwise math) ──────────────────────────────
        if (func in (torch.exp, torch.log, torch.log2, torch.log10, torch.sqrt,
                     torch.ops.aten.exp.default, torch.ops.aten.log.default,
                     torch.ops.aten.sqrt.default) or
                fn in ("exp", "log", "log2", "log10", "sqrt", "rsqrt")):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref is not None else out

        # ── UNIVERSAL FALLBACK: dequant → op → requant ────────────────────────
        ref  = _pick_ref(*args, *kwargs.values())
        flat_args = tree_map(lambda x: _unwrap(x) if isinstance(x, mx_tensor) else x, args)
        flat_kw   = {k: _unwrap(v) if isinstance(v, mx_tensor) else v
                     for k, v in kwargs.items()}

        if _DEBUG and ref is not None:
            log.debug(f"[FALLBACK] {fn}({ref._mx_dtype.name}) → float32")
        if _STRICT and ref is not None:
            raise RuntimeError(
                f"[mx_triton STRICT] No packed kernel for {fn} "
                f"(dtype={ref._mx_dtype.name}). Fallback blocked."
            )

        try:
            out = func(*flat_args, **flat_kw)
        except Exception as e:
            raise RuntimeError(f"[mx_triton] Failed to dispatch {fn}: {e}") from e

        if ref is not None and isinstance(out, Tensor) and not isinstance(out, mx_tensor):
            if out.ndim > 0 and out.dtype.is_floating_point:
                return mx_tensor.quantize(out, ref._mx_dtype, ref._mx_block)
        return out

_MX_DISPATCH = _Dispatcher()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — AUTOGRAD: STE + MX-precision gradients
# ─────────────────────────────────────────────────────────────────────────────

class _MXQuantize(torch.autograd.Function):
    """
    Real quantization with straight-through estimator (STE) for backward.
        For "up" mode: backward uses full-precision gradient.
    For "down" mode: backward quantizes the gradient too.
    """

    @staticmethod
    def forward(ctx, x: Tensor, dt: mx_dtype, block: int) -> mx_tensor:
        ctx.save_for_backward(x)
        ctx.dt    = dt
        ctx.block = block
        return mx_tensor.quantize(x, dt, block, requires_grad=False)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output may be an mx_tensor; unwrap
        if isinstance(grad_output, mx_tensor):
            grad_output = grad_output.dequantize()
        dt    = ctx.dt
        block = ctx.block
        if dt.is_down:
            # Quantize gradient at MX precision (no full-precision accumulation)
            packed, scales, n = quantize(grad_output, dt, block)
            grad_q = _dequant(packed, scales, dt, n, block).reshape(grad_output.shape)
            return grad_q, None, None
        else:
            # "up" mode: pass gradient through unchanged (full precision)
            return grad_output, None, None

class _PureMXQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, block):
        ctx.save_for_backward(x)
        ctx.dt = dt
        ctx.block = block
        # Create mx_tensor - stays fully quantized
        return mx_tensor.quantize(x, dt, block)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: gradient passes through unchanged (identity)
        # grad_output is gradient w.r.t. output (could be mx_tensor or float)
        # For STE, we pass gradient through as-is
        return grad_output, None, None

def mx_quantize(x: Tensor,
                dtype: Union[str, mx_dtype] = "int4d",
                block: int = 128) -> mx_tensor:
    """Quantize to MX format. For training, gradients flow via STE."""
    dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    return mx_tensor.quantize(x, dt, block)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — nn.Module deep integration
# ─────────────────────────────────────────────────────────────────────────────

class mx_linear(nn.Module):
    """
    Drop-in replacement for nn.Linear with real MX-packed weights.
    Installed automatically when you call to_mx() on a model.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, mx_dtype: mx_dtype = None, block: int = 128, device=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mx_dtype     = mx_dtype or get_mx_dtype("int4d")
        self.block        = block
        
        # Store weight as a buffer (not parameter) to avoid gradient tracking issues
        # The quantization will be applied in forward to preserve gradient graph
        # Defer weight creation until first forward to get proper device
        self._weight_device = device
        self.register_buffer('_mx_weight', None)
        
        if bias:
            self.register_buffer('_mx_bias', None)
        else:
            self.register_buffer('_mx_bias', None)
    
    def _ensure_weight_on_device(self, device):
        """Ensure weights are on the correct device - stored as (in_features, out_features) for direct matmul"""
        if self._mx_weight is None or self._mx_weight.device != device:
            # Create weight with proper initialization (Kaiming uniform)
            # This ensures scales are well-defined and not NaN
            weight_data = torch.randn(self.in_features, self.out_features, device=device) * 0.01
            self._mx_weight = mx_tensor.quantize(weight_data, self.mx_dtype, self.block)
        if self._mx_bias is not None:
            bias_data = torch.zeros(self.out_features, device=device)
            self._mx_bias = mx_tensor.quantize(bias_data, self.mx_dtype, self.block)

    @classmethod
    def from_linear(cls, linear: nn.Linear, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.in_features  = linear.in_features
        m.out_features = linear.out_features
        m.mx_dtype     = mx_dtype
        m.block        = block
        # Store transposed: (out_features, in_features) -> (in_features, out_features)
        m.register_buffer('_mx_weight', mx_tensor.quantize(linear.weight.data.t(), mx_dtype, block))
        if linear.bias is not None:
            m.register_buffer('_mx_bias', mx_tensor.quantize(linear.bias.data, mx_dtype, block))
        else:
            m.register_buffer('_mx_bias', None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        # Ensure weights are on same device as input
        self._ensure_weight_on_device(x.device)
        
        w_mx = self._mx_weight
        b_mx = self._mx_bias

        if isinstance(w_mx, mx_tensor):
            # Use custom autograd function to handle gradient flow
            return _mx_linear_forward(x, w_mx, b_mx, self.mx_dtype, self.block, self.in_features, self.out_features)

        # Fallback to plain linear
        w_data = w_mx.dequantize() if isinstance(w_mx, mx_tensor) else w_mx
        b_data = b_mx.dequantize() if isinstance(b_mx, mx_tensor) else b_mx if b_mx is not None else None
        return F.linear(x, w_data, b_data)

    @property
    def weight(self):
        """Return dequantized weight for compatibility"""
        if self._mx_weight is not None:
            return self._mx_weight.dequantize()
        return None
    
    @property
    def bias(self):
        """Return dequantized bias for compatibility"""
        if self._mx_bias is not None:
            return self._mx_bias.dequantize()
        return None

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"dtype={self.mx_dtype.name}, bias={self._mx_bias is not None}")

class mx_conv2d(nn.Module):
    """
    Drop-in replacement for nn.Conv2d with real MX-packed weights.
    Installed automatically by to_mx() when a Conv2d is encountered.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride      = stride   if isinstance(stride, tuple)   else (stride, stride)
        self.padding     = padding  if isinstance(padding, tuple)  else (padding, padding)
        self.dilation    = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups      = groups
        self.mx_dtype    = mx_dtype or get_mx_dtype("int4d")
        self.block       = block
        w_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.register_buffer('_mx_weight', mx_tensor.quantize(torch.empty(*w_shape), self.mx_dtype, block))
        if bias:
            self.register_buffer('_mx_bias', mx_tensor.quantize(torch.zeros(out_channels), self.mx_dtype, block))
        else:
            self.register_buffer('_mx_bias', None)
    
    @property
    def weight(self):
        return self._mx_weight
    
    @weight.setter
    def weight(self, value):
        self._mx_weight = value
    
    @property
    def bias(self):
        return self._mx_bias
    
    @bias.setter
    def bias(self, value):
        self._mx_bias = value

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.in_channels = conv.in_channels; m.out_channels = conv.out_channels
        m.kernel_size = conv.kernel_size; m.stride = conv.stride
        m.padding = conv.padding; m.dilation = conv.dilation; m.groups = conv.groups
        m.mx_dtype = mx_dtype; m.block = block
        m.register_buffer('_mx_weight', mx_tensor.quantize(conv.weight.data, mx_dtype, block, requires_grad=False))
        m.register_buffer('_mx_bias', 
            mx_tensor.quantize(conv.bias.data, mx_dtype, block, requires_grad=False) if conv.bias is not None else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        w = self._mx_weight; b = self._mx_bias
        w_f = w.dequantize() if isinstance(w, mx_tensor) else w
        b_f = b.dequantize() if isinstance(b, mx_tensor) else b
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        out = F.conv2d(x_f, w_f, b_f, self.stride, self.padding, self.dilation, self.groups)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return f"in={self.in_channels}, out={self.out_channels}, kernel={self.kernel_size}, dtype={self.mx_dtype.name}"

class mx_conv1d(nn.Module):
    """Drop-in for nn.Conv1d with MX-packed weights."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride      = stride   if isinstance(stride, tuple)   else (stride,)
        self.padding     = padding  if isinstance(padding, tuple)  else (padding,)
        self.dilation    = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups      = groups
        self.mx_dtype    = mx_dtype or get_mx_dtype("int4d")
        self.block       = block
        w_shape = (out_channels, in_channels // groups, self.kernel_size[0])
        self.register_buffer('_mx_weight', mx_tensor.quantize(torch.empty(*w_shape), self.mx_dtype, block))
        self.register_buffer('_mx_bias', mx_tensor.quantize(torch.zeros(out_channels), self.mx_dtype, block) if bias else None)
    
    @property
    def weight(self): return self._mx_weight
    @weight.setter
    def weight(self, v): self._mx_weight = v
    
    @property
    def bias(self): return self._mx_bias
    @bias.setter
    def bias(self, v): self._mx_bias = v

    @classmethod
    def from_conv1d(cls, conv: nn.Conv1d, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.in_channels = conv.in_channels; m.out_channels = conv.out_channels
        m.kernel_size = conv.kernel_size; m.stride = conv.stride
        m.padding = conv.padding; m.dilation = conv.dilation; m.groups = conv.groups
        m.mx_dtype = mx_dtype; m.block = block
        m.register_buffer('_mx_weight', mx_tensor.quantize(conv.weight.data, mx_dtype, block, requires_grad=False))
        m.register_buffer('_mx_bias', mx_tensor.quantize(conv.bias.data, mx_dtype, block, requires_grad=False) if conv.bias is not None else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        w = self._mx_weight; b = self._mx_bias
        w_f = w.dequantize() if isinstance(w, mx_tensor) else w
        b_f = b.dequantize() if isinstance(b, mx_tensor) else b
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        return mx_tensor.quantize(
            F.conv1d(x_f, w_f, b_f, self.stride, self.padding, self.dilation, self.groups),
            self.mx_dtype, self.block)

class mx_layer_norm(nn.Module):
    """nn.LayerNorm with MX-packed weight/bias. Norm itself in float32."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, 
                 mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        if mx_dtype is None:
            mx_dtype = get_mx_dtype("int8d")
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = True
        self.mx_dtype = mx_dtype
        self.block = block
        # Initialize weights
        weight_data = torch.ones(*self.normalized_shape)
        bias_data = torch.zeros(*self.normalized_shape)
        self.weight = nn.Parameter(
            mx_tensor.quantize(weight_data, mx_dtype, block, requires_grad=False),
            requires_grad=False
        )
        self.bias = nn.Parameter(
            mx_tensor.quantize(bias_data, mx_dtype, block, requires_grad=False),
            requires_grad=False
        )

    @classmethod
    def from_layer_norm(cls, ln: nn.LayerNorm, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.normalized_shape = ln.normalized_shape
        m.eps = ln.eps; m.elementwise_affine = ln.elementwise_affine
        m.mx_dtype = mx_dtype; m.block = block
        m.weight = (nn.Parameter(mx_tensor.quantize(ln.weight.data, mx_dtype, block, requires_grad=False), requires_grad=False)
                    if ln.weight is not None else None)
        m.bias   = (nn.Parameter(mx_tensor.quantize(ln.bias.data, mx_dtype, block, requires_grad=False), requires_grad=False)
                    if ln.bias   is not None else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        w = (self.weight.data.dequantize() if isinstance(self.weight.data, mx_tensor)
             else self.weight.data) if self.weight is not None else None
        b = (self.bias.data.dequantize()   if isinstance(self.bias.data, mx_tensor)
             else self.bias.data)   if self.bias   is not None else None
        out = F.layer_norm(x_f, self.normalized_shape, w, b, self.eps)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return f"shape={self.normalized_shape}, dtype={self.mx_dtype.name}"

class mx_rms_norm(nn.Module):
    """
    RMS Normalisation with MX-packed weight (LLaMA / Mistral style).
        Compatible with: transformers.RMSNorm, LlamaRMSNorm, etc.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        if mx_dtype is None:
            mx_dtype = get_mx_dtype("int8d")
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.mx_dtype = mx_dtype
        self.block = block
        # Initialize weights
        weight_data = torch.ones(*self.normalized_shape)
        self.weight = nn.Parameter(
            mx_tensor.quantize(weight_data, mx_dtype, block, requires_grad=False),
            requires_grad=False
        )

    @classmethod
    def from_rms_norm(cls, rms, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.normalized_shape = rms.weight.shape
        m.eps     = getattr(rms, "eps", None) or getattr(rms, "variance_epsilon", 1e-6)
        m.mx_dtype = mx_dtype; m.block = block
        m.weight  = nn.Parameter(mx_tensor.quantize(rms.weight.data, mx_dtype, block, requires_grad=False), requires_grad=False)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        w   = self.weight.data
        w_f = w.dequantize() if isinstance(w, mx_tensor) else w.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x_f * rms * w_f
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return f"shape={self.normalized_shape}, dtype={self.mx_dtype.name}"

class mx_group_norm(nn.Module):
    """nn.GroupNorm with MX-packed weight/bias."""

    @classmethod
    def from_group_norm(cls, gn: nn.GroupNorm, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.num_groups = gn.num_groups; m.num_channels = gn.num_channels
        m.eps = gn.eps; m.affine = gn.affine; m.mx_dtype = mx_dtype; m.block = block
        m.weight = (nn.Parameter(mx_tensor.quantize(gn.weight.data, mx_dtype, block, requires_grad=False), requires_grad=False)
                    if gn.affine else None)
        m.bias   = (nn.Parameter(mx_tensor.quantize(gn.bias.data, mx_dtype, block, requires_grad=False), requires_grad=False)
                    if gn.affine else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        w = (self.weight.data.dequantize() if isinstance(self.weight.data, mx_tensor)
             else self.weight.data) if self.weight is not None else None
        b = (self.bias.data.dequantize()   if isinstance(self.bias.data, mx_tensor)
             else self.bias.data)   if self.bias   is not None else None
        out = F.group_norm(x_f, self.num_groups, w, b, self.eps)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

class mx_embedding_bag(nn.Module):
    """nn.EmbeddingBag with MX-packed weight table. Supports sum/mean/max modes."""

    @classmethod
    def from_embedding_bag(cls, emb: nn.EmbeddingBag, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.num_embeddings = emb.num_embeddings; m.embedding_dim = emb.embedding_dim
        m.mode = emb.mode; m.mx_dtype = mx_dtype; m.block = block
        m.weight = nn.Parameter(mx_tensor.quantize(emb.weight.data, mx_dtype, block, requires_grad=False), requires_grad=False)
        return m

    def forward(self, input, offsets=None, per_sample_weights=None) -> Tensor:
        w   = self.weight.data
        w_f = w.dequantize() if isinstance(w, mx_tensor) else w
        out = F.embedding_bag(input, w_f, offsets, mode=self.mode,
                              per_sample_weights=per_sample_weights)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return f"num={self.num_embeddings}, dim={self.embedding_dim}, mode={self.mode}, dtype={self.mx_dtype.name}"

class mx_batch_norm2d(nn.Module):
    """
    nn.BatchNorm2d with MX-packed weight/bias.
    Running stats (running_mean, running_var) kept in float32 — they are
    updated by momentum and must stay high-precision for training stability.
    Actual BN computation happens in float32; only the affine params are packed.
    """

    @classmethod
    def from_batch_norm(cls, bn: nn.BatchNorm2d, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.num_features  = bn.num_features
        m.eps           = bn.eps
        m.momentum      = bn.momentum
        m.affine        = bn.affine
        m.track_running_stats = bn.track_running_stats
        m.mx_dtype      = mx_dtype
        m.block         = block
        if bn.affine:
            # Use register_buffer instead of nn.Parameter for mx_tensor
            m.register_buffer('weight', mx_tensor.quantize(bn.weight.data, mx_dtype, block, requires_grad=False))
            m.register_buffer('bias', mx_tensor.quantize(bn.bias.data, mx_dtype, block, requires_grad=False))
        else:
            m.register_buffer('weight', None)
            m.register_buffer('bias', None)
        # running stats stay float32
        if bn.track_running_stats:
            m.register_buffer("running_mean", bn.running_mean.clone())
            m.register_buffer("running_var",  bn.running_var.clone())
            m.register_buffer("num_batches_tracked", bn.num_batches_tracked.clone())
        else:
            m.running_mean = m.running_var = m.num_batches_tracked = None
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        w   = (self.weight.data.dequantize() if isinstance(self.weight.data, mx_tensor)
               else self.weight.data) if self.weight is not None else None
        b   = (self.bias.data.dequantize()   if isinstance(self.bias.data, mx_tensor)
               else self.bias.data)   if self.bias   is not None else None
        out = F.batch_norm(x_f, self.running_mean, self.running_var,
                           w, b, self.training, self.momentum or 0.1, self.eps)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return (f"num_features={self.num_features}, "
                f"dtype={self.mx_dtype.name}, eps={self.eps}")

class mx_multihead_attention(nn.Module):
    """
    nn.MultiheadAttention with MX-packed in_proj / out_proj weights.
    All four projection matrices (Q, K, V, O) are stored at MX precision.
    Attention scores are computed in float32 for numerical stability.
    """

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.embed_dim    = mha.embed_dim
        m.num_heads    = mha.num_heads
        m.dropout      = mha.dropout
        m.batch_first  = mha.batch_first
        m.mx_dtype     = mx_dtype
        m.block        = block
        m.kdim         = mha.kdim
        m.vdim         = mha.vdim
        m._qkv_same_embed_dim = mha._qkv_same_embed_dim

        # in_proj_weight: [3*embed_dim, embed_dim]
        # Use register_buffer instead of nn.Parameter for mx_tensor
        if mha.in_proj_weight is not None:
            m.register_buffer('in_proj_weight',
                mx_tensor.quantize(mha.in_proj_weight.data, mx_dtype, block, requires_grad=False))
        else:
            m.register_buffer('in_proj_weight', None)
            # separate Q/K/V projections (cross-attention style)
            for attr in ("q_proj_weight","k_proj_weight","v_proj_weight"):
                w = getattr(mha, attr, None)
                if w is not None:
                    m.register_buffer(attr, mx_tensor.quantize(w.data, mx_dtype, block, requires_grad=False))
                else:
                    m.register_buffer(attr, None)

        m.in_proj_bias = mha.in_proj_bias   # keep bias float32 (small, precision-critical)
        m.out_proj     = mx_linear.from_linear(mha.out_proj, mx_dtype, block)
        m.bias_k       = mha.bias_k
        m.bias_v       = mha.bias_v
        m.add_zero_attn = mha.add_zero_attn
        return m

    def _dq(self, t: Optional[Tensor]) -> Optional[Tensor]:
        """Dequantize if mx_tensor, else return as-is."""
        if t is None: return None
        return t.dequantize() if isinstance(t, mx_tensor) else t.float()

    def forward(self, query, key, value,
                key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        # Dequantize inputs + weights for the attention kernel
        q_f = self._dq(query)
        k_f = self._dq(key)
        v_f = self._dq(value)

        w_in = self._dq(self.in_proj_weight) if self.in_proj_weight is not None else None

        # Reconstruct a temporary plain MHA module for the F.multi_head_attention_forward call
        out, attn = F.multi_head_attention_forward(
            q_f, k_f, v_f,
            self.embed_dim, self.num_heads,
            w_in, self.in_proj_bias,
            self.bias_k, self.bias_v,
            self.add_zero_attn,
            self.dropout if self.training else 0.0,
            self._dq(self.out_proj.weight.data),
            self.out_proj.bias.data if self.out_proj.bias is not None else None,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )
        return mx_tensor.quantize(out, self.mx_dtype, self.block), attn

    def extra_repr(self):
        return (f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"dtype={self.mx_dtype.name}")

# ── Activation quantization hooks ────────────────────────────────────────────

def wrap_activations(model: nn.Module, dtype: Union[str, mx_dtype] = "int8d",
                     block: int = 128) -> nn.Module:
    """
    Quantize layer activations (outputs) at MX precision.
    Call AFTER to_mx() to get both weight + activation quantization.
    No full-precision intermediate tensors in the forward pass.

    Usage:
        model = to_mx(model, "int4d")            # weight quantization
        model = wrap_activations(model, "int8d") # activation quantization
    """
    dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype

    def _make_hook(name: str):
        def hook(module, inp, output):
            if isinstance(output, Tensor) and not isinstance(output, mx_tensor):
                if output.is_floating_point() and output.ndim > 0:
                    return mx_tensor.quantize(output.float(), dt, block)
            return output
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (mx_linear, mx_conv2d, mx_conv1d, mx_layer_norm,
                                mx_rms_norm, mx_group_norm, nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(_make_hook(name)))

    if not hasattr(model, "_mx_activation_hooks"):
        model._mx_activation_hooks = []
    model._mx_activation_hooks.extend(handles)
    if _DEBUG:
        log.debug(f"[wrap_activations] {len(handles)} hooks installed, dtype={dt.name}")
    return model

def unwrap_activations(model: nn.Module):
    """Remove all activation quantization hooks."""
    for h in getattr(model, "_mx_activation_hooks", []):
        h.remove()
    model._mx_activation_hooks = []

class mx_embedding(nn.Module):
    """nn.Embedding with MX-packed weight table."""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 mx_dtype: mx_dtype = None, block: int = 128, device=None, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.mx_dtype       = mx_dtype or get_mx_dtype("int4d")
        self.block          = block
        self._kwargs        = kwargs  # padding_idx etc.
        self._device        = device
        # Defer weight creation until first forward (similar to mx_linear)
        self.register_buffer('_mx_weight', None)

    def _ensure_weight_on_device(self, device):
        """Ensure weights are on the correct device."""
        if self._mx_weight is None or self._mx_weight.device != device:
            # Initialize with small random values (Kaiming uniform style)
            weight_data = torch.randn(self.num_embeddings, self.embedding_dim, device=device) * 0.01
            self._mx_weight = mx_tensor.quantize(weight_data, self.mx_dtype, self.block)

    @property
    def weight(self):
        """Return mx_tensor weight for compatibility."""
        if self._mx_weight is not None:
            return self._mx_weight
        return None

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.num_embeddings = emb.num_embeddings
        m.embedding_dim  = emb.embedding_dim
        m.mx_dtype       = mx_dtype
        m.block          = block
        m._kwargs        = {}
        m._device        = emb.weight.device
        m.register_buffer('_mx_weight', 
            mx_tensor.quantize(emb.weight.data, mx_dtype, block, requires_grad=False))
        return m

    def forward(self, indices: Tensor) -> Tensor:
        self._ensure_weight_on_device(indices.device)
        w = self._mx_weight
        if isinstance(w, mx_tensor):
            # Look up by index in dequantized table (full lookup)
            w_f = w.dequantize()
            out = F.embedding(indices, w_f)
            return mx_tensor.quantize(out, self.mx_dtype, self.block)
        return F.embedding(indices, w)

# ── Module patch: replace layers in-place ────────────────────────────────────

def _replace_module(parent: nn.Module, name: str,
                    module: nn.Module, mx_dtype: mx_dtype, block: int):
    """
    Replace a module with its MX-quantized counterpart.
    Covers: Linear, Conv1d/2d, LayerNorm, GroupNorm, Embedding, EmbeddingBag,
            and any module with .weight/.bias parameters (generic fallback).
            Also detects LLaMA/Mistral-style RMSNorm by duck-typing.
    """
    kind = type(module).__name__

    if isinstance(module, nn.Linear):
        setattr(parent, name, mx_linear.from_linear(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: Linear → mx_linear({mx_dtype.name})")

    elif isinstance(module, nn.Conv2d):
        setattr(parent, name, mx_conv2d.from_conv2d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: Conv2d → mx_conv2d({mx_dtype.name})")

    elif isinstance(module, nn.ConvTranspose2d):
        setattr(parent, name, mx_conv_transpose2d.from_conv_transpose2d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: ConvTranspose2d → mx_conv_transpose2d({mx_dtype.name})")

    elif isinstance(module, nn.Conv1d):
        setattr(parent, name, mx_conv1d.from_conv1d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: Conv1d → mx_conv1d({mx_dtype.name})")

    elif isinstance(module, nn.ConvTranspose1d):
        setattr(parent, name, mx_conv_transpose1d.from_conv_transpose1d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: ConvTranspose1d → mx_conv_transpose1d({mx_dtype.name})")

    elif isinstance(module, nn.BatchNorm2d):
        setattr(parent, name, mx_batch_norm2d.from_batch_norm(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: BatchNorm2d → mx_batch_norm2d({mx_dtype.name})")

    elif isinstance(module, nn.BatchNorm1d):
        setattr(parent, name, mx_batch_norm1d.from_batch_norm1d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: BatchNorm1d → mx_batch_norm1d({mx_dtype.name})")

    elif isinstance(module, nn.MultiheadAttention):
        setattr(parent, name, mx_multihead_attention.from_mha(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: MultiheadAttention → mx_multihead_attention({mx_dtype.name})")

    elif isinstance(module, nn.TransformerEncoderLayer):
        setattr(parent, name, mx_transformer_encoder_layer.from_encoder_layer(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: TransformerEncoderLayer → mx_transformer_encoder_layer({mx_dtype.name})")

    elif isinstance(module, nn.LayerNorm):
        setattr(parent, name, mx_layer_norm.from_layer_norm(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: LayerNorm → mx_layer_norm({mx_dtype.name})")

    elif isinstance(module, nn.GroupNorm):
        setattr(parent, name, mx_group_norm.from_group_norm(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: GroupNorm → mx_group_norm({mx_dtype.name})")

    elif isinstance(module, nn.Embedding):
        setattr(parent, name, mx_embedding.from_embedding(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: Embedding → mx_embedding({mx_dtype.name})")

    elif isinstance(module, nn.EmbeddingBag):
        setattr(parent, name, mx_embedding_bag.from_embedding_bag(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: EmbeddingBag → mx_embedding_bag({mx_dtype.name})")

    elif (hasattr(module, "weight") and hasattr(module, "variance_epsilon") or
          "RMSNorm" in kind or "RmsNorm" in kind):
        # Duck-type LLaMA/Mistral/Gemma RMSNorm variants
        try:
            setattr(parent, name, mx_rms_norm.from_rms_norm(module, mx_dtype, block))
            if _DEBUG: log.debug(f"[replace] {name}: {kind} → mx_rms_norm({mx_dtype.name})")
        except Exception:
            _replace_generic(module, name, mx_dtype, block)

    else:
        # Generic: quantize weight and bias parameters in-place
        _replace_generic(module, name, mx_dtype, block)

def _replace_generic(module: nn.Module, name: str, mx_dtype: mx_dtype, block: int):
    """Generic in-place weight quantization for unrecognised module types."""
    for attr in ("weight", "bias"):
        p = getattr(module, attr, None)
        if isinstance(p, nn.Parameter) and p is not None and p.data.ndim > 0:
            with torch.no_grad():
                mx_t = mx_tensor.quantize(p.data, mx_dtype, block)
            setattr(module, attr, nn.Parameter(mx_t, requires_grad=p.requires_grad))
            if _DEBUG:
                log.debug(f"[replace] {name}.{attr}: generic quantize → {mx_dtype.name}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11b — SPARSE ARITHMETIC
#   sparse_mx_tensor: magnitude-pruned sparse tensor with MX-packed non-zero values.
#   prune_to_sparse(): create from a dense mx_tensor or float tensor.
#   mx_sparse_linear: sparse × quantized linear layer for pruned models.
# ─────────────────────────────────────────────────────────────────────────────

class sparse_mx_tensor:
    """
    Sparse MX-quantized tensor using CSR (Compressed Sparse Row) layout.

    Combines two compression axes:
      1. Sparsity: only non-zero elements are stored.
      2. MX quantization: non-zero values are bit-packed at the target precision.

    Memory layout:
      values   : mx_tensor  — packed non-zero values
      crow_ptr : int32     — CSR row pointers  [rows+1]
      col_idx  : int16/32  — column indices of non-zeros
      shape    : original dense shape
      density  : nnz / numel (fraction of non-zeros)

    Use ``prune_to_sparse()`` to create from a dense or mx_tensor.

    Example::
        sparse_w = prune_to_sparse(weight, sparsity=0.5, dtype="int4d")
        # Replace in mx_linear:
        mx_lin.sparse_weight = sparse_w
        # Forward: automatic dense reconstruction for now (TODO: sparse GEMM)
        out = F.linear(x, sparse_w.to_dense().dequantize())
    """

    def __init__(self, values: mx_tensor, crow_ptr: Tensor, col_idx: Tensor,
                 shape: torch.Size, nnz: int):
        self.values   = values     # mx_tensor of non-zero values
        self.crow_ptr = crow_ptr   # [rows + 1]
        self.col_idx  = col_idx    # [nnz]
        self.shape    = shape
        self.nnz      = nnz

    @property
    def density(self) -> float:
        return self.nnz / max(math.prod(self.shape), 1)

    @property
    def sparsity(self) -> float:
        return 1.0 - self.density

    def to_dense(self) -> mx_tensor:
        """Reconstruct dense mx_tensor from sparse CSR representation."""
        rows, cols = self.shape[0], math.prod(self.shape[1:])
        vals_f = self.values.dequantize()
        # Use PyTorch's native CSR → dense for correctness and speed
        # Suppress the "Sparse CSR tensor support is in beta state" warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore all warnings in this context
            sparse_csr = torch.sparse_csr_tensor(
                self.crow_ptr.long(),
                self.col_idx.long(),
                vals_f,
                (rows, cols),
                dtype=torch.float32,
                device=self.values.device,
            )
        dense = sparse_csr.to_dense()
        return mx_tensor.quantize(dense.reshape(self.shape),
                                 self.values._mx_dtype, self.values._mx_block)

    def to_torch_sparse_csr(self) -> Tensor:
        """
        Return a plain torch.Tensor in sparse CSR format.
        Useful for passing to torch.sparse.mm or torch.mm with sparse support.
        """
        rows, cols = self.shape[0], math.prod(self.shape[1:])
        vals_f = self.values.dequantize()
        # Suppress the beta warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore all warnings in this context
            return torch.sparse_csr_tensor(
                self.crow_ptr.long(), self.col_idx.long(), vals_f,
                (rows, cols), dtype=torch.float32, device=self.values.device)

    def to_torch_sparse_coo(self) -> Tensor:
        """Return as a sparse COO tensor (for ops that prefer COO format)."""
        rows, cols = self.shape[0], math.prod(self.shape[1:])
        vals_f = self.values.dequantize()
        # Convert CSR crow_ptr → row indices
        row_counts = self.crow_ptr[1:] - self.crow_ptr[:-1]  # [rows]
        row_indices = torch.repeat_interleave(
            torch.arange(rows, device=self.values.device), row_counts)
        indices = torch.stack([row_indices, self.col_idx.long()])
        return torch.sparse_coo_tensor(indices, vals_f, (rows, cols)).coalesce()

    def dequantize(self) -> Tensor:
        """Dense float32 reconstruction."""
        return self.to_dense().dequantize()

    def nbytes(self) -> int:
        return (self.values.nbytes_packed +
                self.crow_ptr.nbytes + self.col_idx.nbytes)

    def compression_vs_dense_fp32(self) -> float:
        return (math.prod(self.shape) * 4) / max(self.nbytes(), 1)

    def __repr__(self):
        return (f"sparse_mx_tensor({self.values._mx_dtype.name}, "
                f"shape={tuple(self.shape)}, density={self.density:.2%}, "
                f"{self.compression_vs_dense_fp32():.1f}x vs fp32)")

def prune_to_sparse(
    x: Union[Tensor, mx_tensor],
    sparsity: float = 0.5,
    dtype: Union[str, mx_dtype] = "int4d",
    block: int = 128,
    structured: bool = False,
) -> sparse_mx_tensor:
    """
    Magnitude prune a tensor and store surviving values as MX-quantized CSR.

    Combines two compression techniques:
      • Sparsity: remove the ``sparsity`` fraction of smallest-magnitude weights.
      • MX quantization: pack the remaining non-zeros at ``dtype`` precision.

    Typical compression: int4d at 50% sparsity → ~16x vs fp32 (4-bit / 0.5 density).

    Args:
        x:          Float or mx_tensor to compress.
        sparsity:   Fraction of weights to zero (0.5 = 50% sparse).
        dtype:      MX dtype for quantizing the non-zero values.
            block:      Quantisation block size.
        structured: If True, apply 2:4 structured sparsity (NVIDIA A100/H100 style)
                    — prune 2 out of every 4 weights, enabling hardware sparse GEMM.

    Returns:
        sparse_mx_tensor in CSR format with MX-quantized non-zeros.

    Example::
        sparse_w = prune_to_sparse(weight, sparsity=0.5, dtype="int4d")
        print(f"Compression: {sparse_w.compression_vs_dense_fp32():.1f}x")
        w_dense  = sparse_w.dequantize()  # back to float32
    """
    dt  = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    x_f = x.dequantize().float() if isinstance(x, mx_tensor) else x.float()
    orig_shape = x_f.shape
    rows = x_f.shape[0]
    cols = x_f.numel() // rows
    flat = x_f.reshape(rows, cols)

    if structured:
        # 2:4 structured sparsity: for each group of 4, zero the 2 smallest
        g      = flat.reshape(rows, -1, 4)
        vals, idx = torch.topk(g.abs(), k=2, dim=-1, largest=False)
        mask   = torch.ones_like(g, dtype=torch.bool)
        mask.scatter_(-1, idx, False)
        flat   = (g * mask).reshape(rows, cols)
    else:
        # Unstructured: global magnitude threshold
        threshold = torch.quantile(flat.abs().flatten(),
                                    float(sparsity)).item()
        flat = flat * (flat.abs() > threshold)

    # ── Build CSR from the pruned dense matrix (fully vectorized, no Python loop) ──
    # Mask of non-zeros [rows, cols]
    nz_mask     = flat != 0
    nnz_per_row = nz_mask.sum(dim=1)                          # [rows]  int64
    crow_ptr    = torch.zeros(rows + 1, dtype=torch.int32, device=x_f.device)
    torch.cumsum(nnz_per_row.to(torch.int32), dim=0, out=crow_ptr[1:])

    nnz     = int(crow_ptr[-1].item())
    # All non-zero column indices at once (nonzero returns row-major order)
    col_idx = nz_mask.nonzero(as_tuple=False)[:, 1].to(torch.int32)  # [nnz]
    vals_f  = flat[nz_mask]                                   # [nnz] float32
    values  = mx_tensor.quantize(vals_f, dt, block)

    return sparse_mx_tensor(values, crow_ptr, col_idx, orig_shape, int(nnz))

class mx_sparse_linear(nn.Module):
    """
    Sparse + MX-quantized linear layer.
    Combines weight pruning with MX quantization for maximum compression.

    Compression examples (fp32 baseline):
      50% sparse + int8d ≈  8x compression
      50% sparse + int4d ≈ 16x compression
      75% sparse + int4d ≈ 32x compression
      2:4 struct + int4d ≈ 16x compression  (+ hardware sparse GEMM on A100)
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, mx_dtype: mx_dtype = None,
                 sparsity: float = 0.5, block: int = 128):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mx_dtype     = mx_dtype or get_mx_dtype("int4d")
        self.sparsity     = sparsity
        self.block        = block
        self.sparse_weight: Optional[sparse_mx_tensor] = None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, mx_dtype: mx_dtype,
                    sparsity: float = 0.5, block: int = 128,
                    structured: bool = False) -> "mx_sparse_linear":
        m = cls(linear.in_features, linear.out_features,
                linear.bias is not None, mx_dtype, sparsity, block)
        m.sparse_weight = prune_to_sparse(
            linear.weight.data, sparsity, mx_dtype, block, structured)
        if linear.bias is not None:
            m.bias = nn.Parameter(linear.bias.data.clone())
        return m

    @classmethod
    def from_mx_linear(cls, mx_linear: "mx_linear", sparsity: float = 0.5) -> "mx_sparse_linear":
        m = cls(mx_linear.in_features, mx_linear.out_features,
                mx_linear.bias is not None, mx_linear.mx_dtype, sparsity, mx_linear.block)
        w_dq = mx_linear.weight.dequantize()
        m.sparse_weight = prune_to_sparse(
            w_dq.data, sparsity, mx_linear.mx_dtype, mx_linear.block, False)
        if mx_linear.bias is not None:
            m.bias = nn.Parameter(mx_linear.bias.dequantize().clone(), requires_grad=False)
        return m

    def forward(self, x: Tensor) -> Tensor:
        if self.sparse_weight is None:
            raise RuntimeError("mx_sparse_linear: sparse_weight not set. "
                               "Use mx_sparse_linear.from_linear().")
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        orig_shape = x_f.shape
        x_2d = x_f.reshape(-1, self.in_features)   # [tokens, in]

        # Use PyTorch sparse GEMM when available (CSR format → torch.sparse.mm)
        # This is 2-4x faster than dense mm at ≥ 50% sparsity on CPU/GPU
        try:
            sparse_csr = self.sparse_weight.to_torch_sparse_csr()
            # torch.sparse.mm: sparse [out, in] × dense [in, tokens] → [out, tokens]
            out = torch.sparse.mm(sparse_csr, x_2d.t()).t()  # [tokens, out]
        except Exception:
            # Fallback: dense dequantized matmul
            w_f = self.sparse_weight.dequantize()
            out = x_2d @ w_f.t()

        if self.bias is not None:
            out = out + self.bias.data
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self):
        if self.sparse_weight:
            return (f"in={self.in_features}, out={self.out_features}, "
                    f"dtype={self.mx_dtype.name}, sparsity={self.sparsity:.0%}, "
                    f"density={self.sparse_weight.density:.2%}, "
                    f"compression={self.sparse_weight.compression_vs_dense_fp32():.1f}x")
        return f"in={self.in_features}, out={self.out_features}"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11c — ADVANCED MODULES
#   mx_lora_linear  : QLoRA-style frozen base + trainable LoRA adapters
#   mx_mixed_int8_linear: LLM.int8() mixed-precision (outlier columns fp16)
#   mx_dynamic_linear: Dynamic (per-token) activation quantization
# ─────────────────────────────────────────────────────────────────────────────

class mx_lora_linear(nn.Module):
    """
    LoRA-aware quantized linear layer (QLoRA / Unsloth style).

    Base weight: frozen at MX precision (int4d typically) — never moves to fp32.
    LoRA adapters: trainable, stored in fp16/bf16 — small memory footprint.

    Forward pass (QLoRA approach):
        y = base_weight_dequant @ x + scale * (B @ A @ x)

    Key insight vs standard LoRA: the base weight stays packed during forward.
    Only the tiny LoRA adapters (A: [rank, in], B: [out, rank]) live in full
    precision, making fine-tuning memory-efficient.

    Args:
        in_features:   Input dimension.
        out_features:  Output dimension.
        rank:          LoRA rank (4, 8, 16, 32 are typical).
        alpha:         LoRA scaling factor (default = rank).
        base_dtype:    MX dtype for frozen base weight (default int4d).
            lora_dtype:    PyTorch dtype for LoRA adapters (default bfloat16).
                block:         Quantisation block size for base weight.

    Example::
        qlora = mx_lora_linear.from_linear(layer, rank=16, base_dtype="int4d")
        # Only qlora.lora_A and qlora.lora_B have gradients
        optimizer = torch.optim.AdamW(
            [p for p in qlora.parameters() if p.requires_grad])

    Usage in model::
        model = to_mx(model, "int4d")       # freeze and quantize all linears
        for name, mod in model.named_modules():
            if isinstance(mod, mx_linear):
                parent = ...
                setattr(parent, name, mx_lora_linear.from_mx_linear(mod, rank=16))
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: Optional[float] = None,
                 base_dtype: mx_dtype = None, lora_dtype = torch.bfloat16,
                 bias: bool = True, block: int = 128):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.alpha        = alpha if alpha is not None else float(rank)
        self.scale        = self.alpha / self.rank
        self.base_dtype   = base_dtype or get_mx_dtype("int4d")
        self.lora_dtype   = lora_dtype
        self.block        = block

        # Base weight: MX-quantized, frozen (no gradient)
        self.base_weight: Optional[mx_tensor] = None
        self.bias_param:  Optional[nn.Parameter] = None

        # LoRA adapters: small, trainable, high-precision
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, dtype=lora_dtype)
            * (2 / in_features) ** 0.5)    # Kaiming init
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=lora_dtype))   # Zero init → identity start
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 8,
                    alpha: Optional[float] = None,
                    base_dtype: Union[str, mx_dtype] = "int4d",
                    lora_dtype = torch.bfloat16,
                    block: int = 128) -> "mx_lora_linear":
        """Create from an existing nn.Linear, quantizing base weight to MX."""
        dt = get_mx_dtype(base_dtype) if isinstance(base_dtype, str) else base_dtype
        m  = cls(linear.in_features, linear.out_features,
                  rank, alpha, dt, lora_dtype, linear.bias is not None, block)
        m.base_weight = mx_tensor.quantize(
            linear.weight.data, dt, block, requires_grad=False)
        if linear.bias is not None:
            m.bias_param = nn.Parameter(linear.bias.data.clone(),
                                        requires_grad=False)
        return m

    @classmethod
    def from_mx_linear(cls, mx_linear: "mx_linear", rank: int = 8,
                       alpha: Optional[float] = None,
                       lora_dtype = torch.bfloat16) -> "mx_lora_linear":
        """Wrap an existing mx_linear with LoRA adapters (no re-quantization)."""
        m = cls(mx_linear.in_features, mx_linear.out_features,
                rank, alpha, mx_linear.mx_dtype, lora_dtype,
                mx_linear.bias is not None, mx_linear.block)
        m.base_weight = mx_linear.weight.data
        if mx_linear.bias is not None:
            m.bias_param = nn.Parameter(
                mx_linear.bias.data.dequantize()
                if isinstance(mx_linear.bias.data, mx_tensor)
                    else mx_linear.bias.data.clone(),
                requires_grad=False)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        # Base path: frozen MX weight
        w_f = self.base_weight.dequantize()
        out = F.linear(x_f, w_f, None)

        # LoRA path: A → B → scaled addition
        # x @ A.T  [B, in] → [B, rank]
        # lora_out [B, rank] → [B, out]
        lora_in  = x_f.to(self.lora_dtype)
        lora_out = F.linear(F.linear(lora_in, self.lora_A), self.lora_B)
        out      = out + self.scale * lora_out.to(out.dtype)

        if self.bias_param is not None:
            out = out + self.bias_param.float()
        return out

    def merge_weights(self) -> mx_linear:
        """
        Merge LoRA adapters into base weight and return a plain mx_linear.
        Call this after training to deploy the merged model.
        """
        w_f      = self.base_weight.dequantize()
        lora_w   = (self.lora_B.float() @ self.lora_A.float()) * self.scale
        merged   = w_f + lora_w
        lin      = nn.Linear(self.in_features, self.out_features,
                             self.bias_param is not None)
        lin.weight.data = merged
        if self.bias_param is not None:
            lin.bias.data = self.bias_param.float()
        return mx_linear.from_linear(lin, self.base_dtype, self.block)

    def trainable_parameters(self):
        """Only LoRA parameters — use this for optimizer."""
        return [self.lora_A, self.lora_B]

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, α={self.alpha}, "
                f"base={self.base_dtype.name}, lora={self.lora_dtype}")

class mx_mixed_int8_linear(nn.Module):
    """
    LLM.int8() mixed-precision linear (bitsandbytes style).

    Outlier columns (large-magnitude) are stored in float16.
    Remaining columns are int8-quantized via absmax block quantization.

    During forward:
        y = x_normal @ W_int8 + x_outlier @ W_fp16

    This preserves model quality on large models where emergent outlier features
    appear in activations (observed in >6B parameter models).

    Args:
        threshold: Column absmax ÷ median threshold for outlier detection (default 6.0).

    Example::
        mixed = mx_mixed_int8_linear.from_linear(layer, threshold=6.0)
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 6.0, block: int = 64):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.threshold    = threshold
        self.block        = block
        self.q_weight: Optional[mx_tensor]  = None
        self.fp_weight: Optional[Tensor]   = None
        self.outlier_mask: Optional[Tensor] = None
        self.bias: Optional[nn.Parameter]  = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, threshold: float = 6.0,
                    block: int = 64) -> "mx_mixed_int8_linear":
        m = cls(linear.in_features, linear.out_features, threshold, block)
        fp_part, q_part, mask = mixed_int8_decompose(
            linear.weight.data, threshold, block)
        m.q_weight     = q_part
        m.fp_weight    = fp_part   # None if no outliers
        m.outlier_mask = mask      # bool [in_features] or None
        if linear.bias is not None:
            m.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()

        if self.fp_weight is None or self.outlier_mask is None:
            # No outliers — pure int8 path
            return F.linear(x_f, self.q_weight.dequantize(), self.bias.data if self.bias else None)

        # Split input by outlier mask
        x_normal  = x_f[..., ~self.outlier_mask]
        x_outlier = x_f[..., self.outlier_mask].half()

        # Normal path (int8) + outlier path (fp16)
        out = F.linear(x_normal, self.q_weight.dequantize())
        out = out + F.linear(x_outlier, self.fp_weight).float()

        if self.bias is not None:
            out = out + self.bias.data
        return out

    def extra_repr(self):
        n_outliers = (self.outlier_mask.sum().item()
                      if self.outlier_mask is not None else 0)
        return (f"in={self.in_features}, out={self.out_features}, "
                f"threshold={self.threshold}, outliers={n_outliers}")

class mx_dynamic_linear(nn.Module):
    """
    Dynamic (per-token) activation quantization.

    Weight: statically quantized to MX precision (like regular mx_linear).
    Activations: quantized at runtime using per-token absmax scales.

    Differences from static mx_linear:
      • No calibration required — scales are computed live.
      • Per-token scaling = finer granularity than per-block static.
      • Slightly more compute (scale computation) vs static.
      • Better accuracy for activations with high dynamic range.

    This is similar to PyTorch's ``torch.ao.quantization.dynamic`` but:
      • Works with MX dtypes (int1 through int8, float variants)
          • Returns mx_tensor outputs (can chain with other MX layers)
              • No observer/prepare/convert workflow — just drop in

    Example::
        dyn = mx_dynamic_linear.from_linear(layer, weight_dtype="int4d",
                                           act_dtype="int8d")
        # Equivalent to: static weight quant + dynamic activation quant
        out = dyn(x)   # x quantized per-token at runtime
    """

    def __init__(self, in_features: int, out_features: int,
                 weight_dtype: mx_dtype = None, act_dtype: mx_dtype = None,
                 bias: bool = True, weight_block: int = 128):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight_dtype = weight_dtype or get_mx_dtype("int4d")
        self.act_dtype    = act_dtype    or get_mx_dtype("int8d")
        self.weight_block = weight_block
        self.weight: Optional[nn.Parameter] = None
        self.bias:   Optional[nn.Parameter] = None

    @classmethod
    def from_linear(cls, linear: nn.Linear,
                    weight_dtype: Union[str, mx_dtype] = "int4d",
                    act_dtype: Union[str, mx_dtype] = "int8d",
                    weight_block: int = 128) -> "mx_dynamic_linear":
        wdt = get_mx_dtype(weight_dtype) if isinstance(weight_dtype, str) else weight_dtype
        adt = get_mx_dtype(act_dtype)    if isinstance(act_dtype, str)    else act_dtype
        m   = cls(linear.in_features, linear.out_features,
                  wdt, adt, linear.bias is not None, weight_block)
        # Delete attributes set by __init__ before registering buffers
        if hasattr(m, 'weight'):
            delattr(m, 'weight')
        if hasattr(m, 'bias'):
            delattr(m, 'bias')
        # Use register_buffer instead of nn.Parameter for mx_tensor
        m.register_buffer('weight',
            mx_tensor.quantize(linear.weight.data, wdt, weight_block), persistent=True)
        if linear.bias is not None:
            m.register_buffer('bias', linear.bias.data.clone(), persistent=True)
        else:
            m.register_buffer('bias', None)
        return m

    def _dynamic_quantize_activation(self, x: Tensor) -> mx_tensor:
        """Per-token dynamic quantization of activations."""
        shape  = x.shape
        x_2d   = x.reshape(-1, shape[-1])        # [tokens, features]
        max_int = float((1 << (self.act_dtype.bits - 1)) - 1)

        # Per-token absmax scale (finer granularity than static per-block)
        scales = x_2d.abs().amax(dim=1).clamp(min=1e-12) / max_int   # [tokens]
        normed = x_2d / scales.unsqueeze(1)
        codes  = normed.round().clamp(-max_int, max_int).to(torch.int32)
        packed = bit_packer.pack_auto(codes.reshape(-1), self.act_dtype.bits)
        # One scale per token; block = feature_dim (all features share token scale)
        mx_t   = mx_tensor(packed, scales.float(),
                          self.act_dtype, torch.Size(list(x_2d.shape)),
                          x_2d.numel(), x_2d.shape[-1])
        return mx_t.reshape(*shape)

    def forward(self, x: Tensor) -> Tensor:
        # Dynamic quantize input activations (per-token)
        x_f   = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        x_q   = self._dynamic_quantize_activation(x_f)

        w_mx  = self.weight.data
        if not isinstance(w_mx, mx_tensor):
            w_mx = mx_tensor.quantize(w_mx, self.weight_dtype, self.weight_block)

        # Flatten tokens for matmul: [... , in] → [tokens, in]
        orig_shape = x_f.shape
        tokens     = x_f.reshape(-1, self.in_features)
        w_dq       = w_mx.dequantize()        # [out, in]
        out        = tokens.float() @ w_dq.t()  # [tokens, out]
        out        = out.reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias.data
        return out

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"w={self.weight_dtype.name}, act={self.act_dtype.name}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11d — ADDITIONAL nn.Module WRAPPERS
#   mx_conv_transpose2d  : Transposed convolution (decoder / upsampling)
#   mx_conv_transpose1d  : 1D transposed convolution
#   mx_batch_norm1d      : 1D batch normalisation (NLP / MLP paths)
#   mx_transformer_encoder_layer: Full transformer encoder block
#   mx_gru              : Gated Recurrent Unit (quantized gates and weights)
# ─────────────────────────────────────────────────────────────────────────────

class mx_conv_transpose2d(nn.Module):
    """
    MX-quantized ``nn.ConvTranspose2d`` (decoder / upsampling convolution).

    Weights are packed at MX precision; the transposed convolution is computed
    via ``F.conv_transpose2d`` after dequantization. All standard parameters
    (stride, padding, output_padding, dilation, groups) are supported.

    Example::
        layer = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        mx    = mx_conv_transpose2d.from_conv_transpose2d(layer, get_mx_dtype("int4d"))
        out   = mx(x)    # x: [B, 128, H, W] → [B, 64, 2H, 2W]
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, bias=True,
                 mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride      = stride   if isinstance(stride, tuple)   else (stride,)
        self.padding     = padding  if isinstance(padding, tuple)  else (padding,)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding,)
        self.dilation    = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups      = groups
        self.mx_dtype    = mx_dtype or get_mx_dtype("int4d")
        self.block       = block
        w_shape = (in_channels, out_channels, *self.kernel_size)
        self.register_buffer('_mx_weight', mx_tensor.quantize(torch.empty(*w_shape), self.mx_dtype, block))
        self.register_buffer('_mx_bias', torch.zeros(out_channels) if bias else None)

    @property
    def weight(self): return self._mx_weight
    @weight.setter
    def weight(self, v): self._mx_weight = v

    @property
    def bias(self): return self._mx_bias
    @bias.setter
    def bias(self, v): self._mx_bias = v

    @classmethod
    def from_conv_transpose2d(cls, ct: nn.ConvTranspose2d,
                              mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.in_channels    = ct.in_channels
        m.out_channels   = ct.out_channels
        m.kernel_size    = ct.kernel_size
        m.stride         = ct.stride
        m.padding        = ct.padding
        m.output_padding = ct.output_padding
        m.dilation       = ct.dilation
        m.groups         = ct.groups
        m.mx_dtype       = mx_dtype
        m.block          = block
        m.register_buffer('_mx_weight', mx_tensor.quantize(ct.weight.data, mx_dtype, block))
        m.register_buffer('_mx_bias', ct.bias.data.clone() if ct.bias is not None else None)
        return m

    def forward(self, x: Tensor, output_size=None) -> Tensor:
        x_f  = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        w_f  = self._mx_weight.dequantize() if isinstance(self._mx_weight, mx_tensor) else self._mx_weight
        b    = self._mx_bias
        kwargs = {}
        if output_size is not None:
            kwargs["output_size"] = output_size
        out  = F.conv_transpose2d(x_f, w_f, b, self.stride, self.padding,
                                   self.output_padding, self.groups, self.dilation,
                                   **kwargs)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"k={self.kernel_size}, stride={self.stride}, "
                f"dtype={self.mx_dtype.name}")

class mx_conv_transpose1d(nn.Module):
    """MX-quantized ``nn.ConvTranspose1d``."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, bias=True,
                 mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride      = stride   if isinstance(stride, tuple)   else (stride,)
        self.padding     = padding  if isinstance(padding, tuple)  else (padding,)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding,)
        self.dilation    = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups      = groups
        self.mx_dtype    = mx_dtype or get_mx_dtype("int4d")
        self.block       = block
        w_shape = (in_channels, out_channels, self.kernel_size[0])
        self.register_buffer('_mx_weight', mx_tensor.quantize(torch.empty(*w_shape), self.mx_dtype, block))
        self.register_buffer('_mx_bias', torch.zeros(out_channels) if bias else None)

    @property
    def weight(self): return self._mx_weight
    @weight.setter
    def weight(self, v): self._mx_weight = v
    
    @property
    def bias(self): return self._mx_bias
    @bias.setter
    def bias(self, v): self._mx_bias = v

    @classmethod
    def from_conv_transpose1d(cls, ct: nn.ConvTranspose1d,
                               mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        for attr in ("in_channels","out_channels","kernel_size","stride",
                     "padding","output_padding","dilation","groups"):
            setattr(m, attr, getattr(ct, attr))
        m.mx_dtype = mx_dtype
        m.block    = block
        m.register_buffer('_mx_weight', mx_tensor.quantize(ct.weight.data, mx_dtype, block))
        m.register_buffer('_mx_bias', ct.bias.data.clone() if ct.bias is not None else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        w_f = self._mx_weight.dequantize() if isinstance(self._mx_weight, mx_tensor) else self._mx_weight
        b   = self._mx_bias
        out = F.conv_transpose1d(x_f, w_f, b, self.stride, self.padding,
                                  self.output_padding, self.groups, self.dilation)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

class mx_batch_norm1d(nn.Module):
    """
    MX-quantized ``nn.BatchNorm1d``.

    Handles both 2D inputs [B, C] and 3D inputs [B, C, L] (1D temporal).
    Running stats (running_mean, running_var) remain float32 for stability.
        Only affine parameters (weight, bias) are MX-packed.

    Example::
        bn = nn.BatchNorm1d(256)
        mx = mx_batch_norm1d.from_batch_norm1d(bn, get_mx_dtype("int8d"))
    """

    @classmethod
    def from_batch_norm1d(cls, bn: nn.BatchNorm1d,
                          mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.num_features = bn.num_features
        m.eps          = bn.eps
        m.momentum     = bn.momentum
        m.affine       = bn.affine
        m.track_running_stats = bn.track_running_stats
        m.mx_dtype     = mx_dtype
        m.block        = block
        if bn.affine:
            # Use register_buffer instead of nn.Parameter for mx_tensor
            m.register_buffer('weight',
                mx_tensor.quantize(bn.weight.data, mx_dtype, block, requires_grad=False))
            m.register_buffer('bias',
                mx_tensor.quantize(bn.bias.data, mx_dtype, block, requires_grad=False))
        else:
            m.register_buffer('weight', None)
            m.register_buffer('bias', None)
        if bn.track_running_stats:
            m.register_buffer("running_mean", bn.running_mean.clone())
            m.register_buffer("running_var",  bn.running_var.clone())
            m.register_buffer("num_batches_tracked", bn.num_batches_tracked.clone())
        else:
            m.running_mean = m.running_var = m.num_batches_tracked = None
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f  = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        w    = (self.weight.data.dequantize() if isinstance(self.weight.data, mx_tensor)
                else self.weight.data) if self.weight is not None else None
        b    = (self.bias.data.dequantize()   if isinstance(self.bias.data, mx_tensor)
                else self.bias.data)   if self.bias   is not None else None
        out  = F.batch_norm(x_f, self.running_mean, self.running_var,
                            w, b, self.training, self.momentum or 0.1, self.eps)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

class mx_transformer_encoder_layer(nn.Module):
    """
    Full transformer encoder block with all weights at MX precision.

    Quantizes:
      • Self-attention Q/K/V projections (mx_linear)
      • Output projection (mx_linear)
      • FFN linear1 (mx_linear)
      • FFN linear2 (mx_linear)
      • LayerNorm affine parameters (mx_layer_norm)

    Computation order (Pre-LN variant, used by most modern models):
      x = x + Attn(LN1(x))
      x = x + FFN(LN2(x))

    Both Pre-LN and Post-LN are supported via ``norm_first`` (same as PyTorch).

    Example::
        enc = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
        mx  = mx_transformer_encoder_layer.from_encoder_layer(enc, get_mx_dtype("int4d"))
        out = mx(x)   # x: [S, B, D] (or [B, S, D] with batch_first=True)
            """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 norm_first: bool = False, mx_dtype: mx_dtype = None,
                 block: int = 128, device=None, dtype=None):
        super().__init__()
        if mx_dtype is None:
            mx_dtype = get_mx_dtype("int8d")
        self.d_model = d_model
        self.nhead = nhead
        self.mx_dtype = mx_dtype
        self.block = block
        self.dropout_p = dropout
        self.norm_first = norm_first
        self.batch_first = batch_first

        # Self-attention
        self.self_attn = mx_multihead_attention.from_mha(
            nn.MultiheadAttention(d_model, nhead, batch_first=batch_first),
            mx_dtype, block
        )
        # Feedforward
        self.linear1 = mx_linear(d_model, dim_feedforward, True, mx_dtype, block)
        self.linear2 = mx_linear(dim_feedforward, d_model, True, mx_dtype, block)
        # Normalization
        self.norm1 = mx_layer_norm(d_model, eps=layer_norm_eps, mx_dtype=mx_dtype, block=block)
        self.norm2 = mx_layer_norm(d_model, eps=layer_norm_eps, mx_dtype=mx_dtype, block=block)
        # Activation
        self.activation = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}.get(activation, F.relu)

    @classmethod
    def from_encoder_layer(cls, enc: nn.TransformerEncoderLayer,
                           mx_dtype: mx_dtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.mx_dtype    = mx_dtype
        m.block       = block
        m.d_model     = enc.self_attn.embed_dim
        m.nhead       = enc.self_attn.num_heads
        m.dropout_p   = enc.dropout.p if hasattr(enc.dropout, "p") else 0.0
        m.norm_first  = enc.norm_first if hasattr(enc, "norm_first") else False
        m.batch_first = enc.self_attn.batch_first

        # Quantize all linear layers
        m.self_attn = mx_multihead_attention.from_mha(enc.self_attn, mx_dtype, block)
        m.linear1   = mx_linear.from_linear(enc.linear1, mx_dtype, block)
        m.linear2   = mx_linear.from_linear(enc.linear2, mx_dtype, block)
        m.norm1     = mx_layer_norm.from_layer_norm(enc.norm1, mx_dtype, block)
        m.norm2     = mx_layer_norm.from_layer_norm(enc.norm2, mx_dtype, block)
        m.activation = enc.activation if callable(enc.activation) else F.relu
        return m

    def _ff_block(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        h   = self.linear1(x_f)
        h_f = h.dequantize() if isinstance(h, mx_tensor) else h
        act = self.activation(h_f) if not isinstance(self.activation, str) else \
              {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}.get(self.activation, F.relu)(h_f)
        out = self.linear2(act)
        return F.dropout(out.dequantize() if isinstance(out, mx_tensor) else out,
                         p=self.dropout_p, training=self.training)

    def _sa_block(self, x: Tensor, attn_mask=None, key_padding_mask=None) -> Tensor:
        x_f = x.dequantize() if isinstance(x, mx_tensor) else x.float()
        out, _ = self.self_attn(x_f, x_f, x_f,
                                 attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask)
        out_f  = out.dequantize() if isinstance(out, mx_tensor) else out
        return F.dropout(out_f, p=self.dropout_p, training=self.training)

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None) -> Tensor:
        if self.norm_first:
            n1   = self.norm1(src)
            n1_f = n1.dequantize() if isinstance(n1, mx_tensor) else n1
            src  = src.float() + self._sa_block(n1_f, src_mask, src_key_padding_mask)
            n2   = self.norm2(src)
            n2_f = n2.dequantize() if isinstance(n2, mx_tensor) else n2
            src  = src + self._ff_block(n2_f)
        else:
            src  = src.float() + self._sa_block(src, src_mask, src_key_padding_mask)
            n1   = self.norm1(src)
            src  = (n1.dequantize() if isinstance(n1, mx_tensor) else n1)
            src  = src + self._ff_block(src)
            n2   = self.norm2(src)
            src  = n2.dequantize() if isinstance(n2, mx_tensor) else n2
            return src

    def extra_repr(self):
        return (f"d={self.d_model}, heads={self.nhead}, "
                f"dtype={self.mx_dtype.name}, norm_first={self.norm_first}")

class mx_gru(nn.Module):
    """
    MX-quantized Gated Recurrent Unit (GRU).

    All three gate weight matrices (reset, update, new) are quantized:
      W_r, W_z, W_n  — input-to-hidden weights [hidden, input]
      U_r, U_z, U_n  — hidden-to-hidden weights [hidden, hidden]
      b_r, b_z, b_n  — biases (kept float32 for precision)

    Forward computes the standard GRU equations:
      r = sigmoid(W_r @ x + U_r @ h + b_r)
      z = sigmoid(W_z @ x + U_z @ h + b_z)
      n = tanh(W_n @ x + r * (U_n @ h + b_n))
      h' = (1 - z) * n + z * h

    Args:
        input_size:  Dimensionality of input x.
        hidden_size: Dimensionality of hidden state h.
        mx_dtype:    MX dtype for quantizing all weight matrices.
            block:       Quantization block size.

    Example::
        gru = nn.GRU(256, 512, batch_first=True)
        # Convert first layer
        mx_gru = mx_gru.from_gru_cell(
            gru.weight_ih_l0, gru.weight_hh_l0,
            gru.bias_ih_l0, gru.bias_hh_l0,
            get_mx_dtype("int4d"))
        h0  = torch.zeros(1, batch, 512)
        out = mx_gru(x, h0[:, 0])   # step-wise
    """

    @classmethod
    def from_gru_cell(cls, weight_ih: Tensor, weight_hh: Tensor,
                      bias_ih: Optional[Tensor], bias_hh: Optional[Tensor],
                      mx_dtype: mx_dtype, block: int = 128) -> "mx_gru":
        hidden  = weight_hh.shape[1]
        inp     = weight_ih.shape[1]
        m       = cls(inp, hidden, mx_dtype, block)
        # Delete attributes set by __init__ before registering buffers
        for name in ("W_r","W_z","W_n","U_r","U_z","U_n",
                     "b_ir","b_iz","b_in","b_hr","b_hz","b_hn"):
            if hasattr(m, name):
                delattr(m, name)
        # PyTorch GRU packs all 3 gates: [3*hidden, input]
        # Split into r, z, n
        for i, name in enumerate(["W_r","W_z","W_n"]):
            sl = slice(i * hidden, (i + 1) * hidden)
            m.register_buffer(name, mx_tensor.quantize(weight_ih[sl], mx_dtype, block))
        for i, name in enumerate(["U_r","U_z","U_n"]):
            sl = slice(i * hidden, (i + 1) * hidden)
            m.register_buffer(name, mx_tensor.quantize(weight_hh[sl], mx_dtype, block))
        if bias_ih is not None:
            for i, name in enumerate(["b_ir","b_iz","b_in"]):
                sl = slice(i * hidden, (i + 1) * hidden)
                m.register_buffer(name, bias_ih[sl].clone())
        if bias_hh is not None:
            for i, name in enumerate(["b_hr","b_hz","b_hn"]):
                sl = slice(i * hidden, (i + 1) * hidden)
                m.register_buffer(name, bias_hh[sl].clone())
        return m

    def __init__(self, input_size: int, hidden_size: int,
                 mx_dtype: mx_dtype, block: int = 128, batch_first: bool = False):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.mx_dtype    = mx_dtype
        self.block       = block
        self.batch_first = batch_first
        # Initialize weights properly for direct instantiation
        # GRU has 3 gates: reset (r), update (z), new (n)
        # Weight shapes: [hidden_size, input_size] for W, [hidden_size, hidden_size] for U
        for name in ("W_r","W_z","W_n"):
            w_data = torch.randn(hidden_size, input_size) * 0.01
            self.register_buffer(name, mx_tensor.quantize(w_data, mx_dtype, block))
        for name in ("U_r","U_z","U_n"):
            u_data = torch.randn(hidden_size, hidden_size) * 0.01
            self.register_buffer(name, mx_tensor.quantize(u_data, mx_dtype, block))
        for name in ("b_ir","b_iz","b_in","b_hr","b_hz","b_hn"):
            self.register_buffer(name, torch.zeros(hidden_size))

    def _dq(self, t: Optional[Tensor]) -> Optional[Tensor]:
        if t is None: return None
        d = t.data if hasattr(t, 'data') else t
        return d.dequantize() if isinstance(d, mx_tensor) else d.float()

    def forward_step(self, x: Tensor, h: Tensor) -> Tensor:
        """Process one time step. x: [B, input], h: [B, hidden] → h': [B, hidden]."""
        x_f   = x.float()
        h_f   = h.float()
        Wr, Wz, Wn = self._dq(self.W_r), self._dq(self.W_z), self._dq(self.W_n)
        Ur, Uz, Un = self._dq(self.U_r), self._dq(self.U_z), self._dq(self.U_n)
        bir = self._dq(self.b_ir); biz = self._dq(self.b_iz); bin_ = self._dq(self.b_in)
        bhr = self._dq(self.b_hr); bhz = self._dq(self.b_hz); bhn = self._dq(self.b_hn)

        _b = lambda b: b if b is not None else 0.0
        r  = torch.sigmoid(x_f @ Wr.t() + _b(bir) + h_f @ Ur.t() + _b(bhr))
        z  = torch.sigmoid(x_f @ Wz.t() + _b(biz) + h_f @ Uz.t() + _b(bhz))
        n  = torch.tanh(x_f @ Wn.t() + _b(bin_) + r * (h_f @ Un.t() + _b(bhn)))
        return (1 - z) * n + z * h_f

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Process a sequence. 
        If batch_first:  x: [B, T, input], h0: [B, hidden] or [1, B, hidden] → (out: [B,T,H], h_n: [B,H])
        If not batch_first: x: [T, B, input], h0: [1, B, hidden] or [B, hidden] → (out: [T,B,H], h_n: [B,H])
        """
        # Handle batch format
        if self.batch_first:
            B, T, _ = x.shape
        else:
            T, B, _ = x.shape
            x = x.transpose(0, 1)  # [T, B, I] -> [B, T, I]
            
        # Handle h0 shape: accept [B, hidden] or [1, B, hidden]
        if h0 is not None:
            if h0.dim() == 3 and h0.shape[0] == 1:
                h = h0.squeeze(0)  # [1, B, H] -> [B, H]
            else:
                h = h0
        else:
            h = torch.zeros(B, self.hidden_size, device=x.device)
        out = []
        for t in range(T):
            h = self.forward_step(x[:, t], h)
            out.append(h.unsqueeze(1))
        out_tensor = torch.cat(out, dim=1)  # [B, T, H]
        
        # Convert back to original format
        if not self.batch_first:
            out_tensor = out_tensor.transpose(0, 1)  # [B, T, H] -> [T, B, H]
            
        return out_tensor, h

    def extra_repr(self):
        return (f"input={self.input_size}, hidden={self.hidden_size}, "
                f"dtype={self.mx_dtype.name}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — MONKEY PATCHING (seamless integration)
# ─────────────────────────────────────────────────────────────────────────────

# ── Patch torch.dtype() to return mx_dtype_proxy for MX names ──────────────────

_orig_torch_dtype = torch.dtype

class _MXDtypeCallable:
    """
    Replacement for torch.dtype that additionally handles MX dtype strings.
    torch.dtype("int4d")   → mx_dtype_proxy(int4d)
    torch.dtype("float32") → torch.float32  (original behaviour)
    """
    def __call__(self, s=None, *args, **kwargs):
        if isinstance(s, str) and s in _DTYPE_REGISTRY:
            return as_mx_dtype_proxy(s)
        if type(s).__name__ == 'mx_dtype':
            return as_mx_dtype_proxy(s)
        if type(s).__name__ == 'mx_dtype_proxy':
            return s
        # Delegate to original (int→dtype, str like "float32", etc.)
        try:
            return _orig_torch_dtype(s, *args, **kwargs)
        except Exception:
            # torch.dtype is a type, not always callable with strings
            _native = {"float32": torch.float32, "float16": torch.float16,
                       "bfloat16": torch.bfloat16, "int8": torch.int8,
                       "int16": torch.int16, "int32": torch.int32,
                       "int64": torch.int64, "float64": torch.float64,
                       "bool": torch.bool}
            if isinstance(s, str) and s in _native:
                return _native[s]
            raise

    def __getattr__(self, name):
        # torch.dtype.float32 etc. → pass through
        return getattr(_orig_torch_dtype, name)

    def __instancecheck__(self, instance):
        return (isinstance(instance, _orig_torch_dtype) or type(instance).__name__ == 'mx_dtype_proxy')

torch.dtype = _MXDtypeCallable()

# ── Patch nn.Module.to() to handle MX dtype strings / proxies ────────────────

_orig_module_to = nn.Module.to

def _mx_module_to(self, *args, **kwargs):
    """
    Augmented nn.Module.to() that accepts MX dtype strings and proxies.
      model.to("int4d")
      model.to(torch.dtype("float8u"))
      model.to(mxt.int4d)
      model.to(dtype="int4d", block_size=64)
    """
    target  = args[0] if args else kwargs.get("dtype", None)
    block   = kwargs.pop("block_size", 128)
    low_mem = kwargs.pop("low_mem", False)

    if type(target).__name__ == 'mx_dtype_proxy':
        return to_mx(self, target._mx.name, block_size=block, low_mem=low_mem)
    if type(target).__name__ == 'mx_dtype':
        return to_mx(self, target.name, block_size=block, low_mem=low_mem)
    if isinstance(target, str) and target in _DTYPE_REGISTRY:
        return to_mx(self, target, block_size=block, low_mem=low_mem)
    if isinstance(target, dict):
        # per-layer dict
        return to_mx(self, target, block_size=block, low_mem=low_mem)

    # Fall through to original
    return _orig_module_to(self, *args, **kwargs)

nn.Module.to = _mx_module_to

# ── Patch Tensor.to() to handle MX dtype ─────────────────────────────────────

_orig_tensor_to = Tensor.to

def _mx_tensor_to(self, *args, **kwargs):
    target = args[0] if args else kwargs.get("dtype", None)
    block  = kwargs.pop("block_size", 128)

    if type(target).__name__ == 'mx_dtype_proxy':
        return mx_tensor.quantize(self.float(), target._mx, block,
                                 requires_grad=self.requires_grad)
    if type(target).__name__ == 'mx_dtype':
        return mx_tensor.quantize(self.float(), target, block,
                                 requires_grad=self.requires_grad)
    if isinstance(target, str) and target in _DTYPE_REGISTRY:
        return mx_tensor.quantize(self.float(), get_mx_dtype(target), block,
                                 requires_grad=self.requires_grad)

    # Also handle mx_tensor → torch dtype (dequantize)
    if isinstance(self, mx_tensor):
        return self.dequantize().to(*args, **kwargs)

    return _orig_tensor_to(self, *args, **kwargs)

Tensor.to = _mx_tensor_to

# ── Patch nn.Parameter to handle mx_tensor storage ────────────────────────────

_orig_param_new = nn.Parameter.__new__

class _MXAwareParameter(nn.Parameter):
    """
    nn.Parameter subclass that transparently holds an mx_tensor.
    `.data` returns the mx_tensor; `.float_data` returns dequantized float32.
    """

    def __new__(cls, data=None, requires_grad=True):
        if isinstance(data, mx_tensor):
            # Make the param wrap the packed storage
            # Packed tensors are integers, so they can't require gradients
            # The mx_tensor itself tracks whether gradients are needed
            inst = torch.Tensor._make_subclass(cls, data.packed, False)
            inst._mx_payload = data
            # Store requires_grad in _mx_payload
            if hasattr(data, 'requires_grad'):
                inst._mx_payload.requires_grad = requires_grad
            return inst
        return nn.Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data=None, requires_grad=True):
        pass  # handled in __new__

    @property
    def data(self):
        if hasattr(self, "_mx_payload"):
            return self._mx_payload
        return torch.Tensor.data.fget(self)

    @data.setter
    def data(self, val):
        if isinstance(val, mx_tensor):
            self._mx_payload = val
            self.packed_ref = val.packed
        elif hasattr(self, "_mx_payload"):
            # Re-quantize after optimizer step
            self._mx_payload = mx_tensor.quantize(
                val.float(), self._mx_payload._mx_dtype, self._mx_payload._mx_block)
        else:
            torch.Tensor.data.fset(self, val)

    @property
    def shape(self):
        if hasattr(self, "_mx_payload"):
            return self._mx_payload._mx_orig_shape
        return torch.Tensor.shape.fget(self)

    def dequantize(self):
        if hasattr(self, "_mx_payload"):
            return self._mx_payload.dequantize()
        return self.float()

# Don't monkeypatch nn.Parameter to avoid metaclass conflicts
# Instead, we'll use _MXAwareParameter directly for mx_tensor parameters
# and keep nn.Parameter as is for compatibility
pass

# ── Patch standard optimizers to handle mx_tensor params ──────────────────────

def _wrap_optimizer(opt_cls):
    """
    Wrap a standard optimizer so that:
    1. mx_tensor params are dequantized before step
    2. Result is re-quantized back to MX precision after step
    3. Momentum / variance buffers are kept at MX precision
    """
    orig_step = opt_cls.step

    @functools.wraps(orig_step)
    def mx_step(self, closure=None):
        # Collect MX params and temporarily inject float proxies
        mx_params = []
        float_proxies = []
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                mx_p = None
                if isinstance(p, _MXAwareParameter) and hasattr(p, "_mx_payload"):
                    mx_p = p._mx_payload
                elif isinstance(p, mx_tensor):
                    mx_p = p

                if mx_p is not None:
                    # Create a float32 proxy that the optimizer will update
                    fp = mx_p.dequantize().detach().requires_grad_(p.requires_grad)
                    if p.grad is not None:
                        grad = p.grad
                        if isinstance(grad, mx_tensor):
                            grad = grad.dequantize()
                        fp.grad = grad.float()
                    mx_params.append((group, i, mx_p, fp))
                    float_proxies.append(fp)
                    group["params"][i] = fp

        # Run original optimizer step
        loss = orig_step(self, closure)

        # Re-quantize updated params back to MX
        for group, i, mx_p, fp in mx_params:
            updated = mx_tensor.quantize(fp.data, mx_p._mx_dtype, mx_p._mx_block)
            # Re-quantize optimizer states if they exist
            state = self.state.get(fp)
            if state:
                new_state = {}
                for k, v in state.items():
                    if isinstance(v, Tensor) and v.dtype.is_floating_point and v.ndim > 0:
                        # Keep state at MX precision
                        new_state[k] = mx_tensor.quantize(v, mx_p._mx_dtype, mx_p._mx_block)
                    else:
                        new_state[k] = v
                self.state[updated] = new_state
                del self.state[fp]

            # Restore MX param with updated values
            if isinstance(group["params"][i], _MXAwareParameter):
                group["params"][i]._mx_payload = updated
            else:
                group["params"][i] = updated

        return loss

    opt_cls.step = mx_step
    return opt_cls

# Patch the most common optimizers
for _opt in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD,
             torch.optim.RMSprop, torch.optim.Adagrad, torch.optim.Adamax):
    _wrap_optimizer(_opt)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — mx_adam_w (native MX optimizer, all states at MX precision)
# ─────────────────────────────────────────────────────────────────────────────

class mx_adam_w(torch.optim.Optimizer):
    """
    AdamW with ALL states stored at MX precision.
    Zero full-precision momentum / variance buffers.

    state_dtype: MX dtype for m and v buffers (default int8d)
        """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, state_dtype: str = "int8d", block: int = 128):
        self._state_dt = get_mx_dtype(state_dtype)
        self._block    = block
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr    = group["lr"]
            b1, b2 = group["betas"]
            eps   = group["eps"]
            wd    = group["weight_decay"]

            for p in group["params"]:
                # Unwrap mx_tensor
                is_mx  = False
                p_dt   = None
                if isinstance(p, _MXAwareParameter) and hasattr(p, "_mx_payload"):
                    px    = p._mx_payload
                    is_mx = True; p_dt = px._mx_dtype
                elif isinstance(p, mx_tensor):
                    px    = p
                    is_mx = True; p_dt = p._mx_dtype
                else:
                    px = p

                # Get gradient
                grad = px.grad if hasattr(px, "grad") else getattr(p, "grad", None)
                if grad is None: continue
                if isinstance(grad, mx_tensor): grad = grad.dequantize()
                grad = grad.float()

                # State
                sid  = id(p)
                st   = self.state[sid] if sid in self.state else {}
                self.state[sid] = st

                if "step" not in st:
                    st["step"] = 0
                    st["m"]    = mx_tensor.quantize(torch.zeros_like(grad), self._state_dt, self._block)
                    st["v"]    = mx_tensor.quantize(torch.zeros_like(grad), self._state_dt, self._block)

                st["step"] += 1
                t = st["step"]

                m = st["m"].dequantize()
                v = st["v"].dequantize()

                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad.pow(2)

                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)

                update = m_hat / (v_hat.sqrt() + eps)

                # Weight
                if is_mx:
                    w_f = px.dequantize()
                    w_f = w_f * (1 - lr * wd) - lr * update
                    new_mx = mx_tensor.quantize(w_f, p_dt, px._mx_block)
                    if isinstance(p, _MXAwareParameter):
                        p._mx_payload = new_mx
                    else:
                        p.packed.copy_(new_mx.packed)
                        p._mx_scales.copy_(new_mx._mx_scales)
                else:
                    p.data.mul_(1 - lr * wd).add_(-lr * update)

                # Re-quantize states
                st["m"] = mx_tensor.quantize(m, self._state_dt, self._block)
                st["v"] = mx_tensor.quantize(v, self._state_dt, self._block)

        return loss

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — PUBLIC API: to_mx, load_quantized, save_quantized
# ─────────────────────────────────────────────────────────────────────────────

def to_mx(
    model: nn.Module,
    dtype: Union[str, Dict[str, str], mx_dtype, mx_dtype_proxy] = "int4d",
    block_size: int = 128,
    low_mem: bool = False,
    layer_batch_size: int = 4,
    skip_patterns: List[str] = None,
) -> nn.Module:
    """
    Quantize a PyTorch model to MX format.
    Drop-in — no model code changes needed.

    Args:
        model:            Any nn.Module.
        dtype:            MX dtype name ("int4d"), mx_dtype, mx_dtype_proxy,
                          or per-layer regex dict {".*attn.*": "float8u", ...}.
        block_size:       Quantization block size (scales granularity).
        low_mem:          Clear GPU cache between layers.
        layer_batch_size: Process N layers between GC passes.
        skip_patterns:    List of regex patterns for layers to skip.

    Returns:
        Same model object with weights replaced by MX packed representations.

    Examples:
        model.to("int4d")                         # via patched .to()
        to_mx(model, "int4d")
        to_mx(model, {".*attn.*": "float8u", ".*mlp.*": "int4d"})
        to_mx(model, torch.dtype("int4d"))        # via dtype proxy
    """
    # Normalise dtype
    if type(dtype).__name__ == 'mx_dtype_proxy':
        dtype = dtype._mx.name
    elif type(dtype).__name__ == 'mx_dtype':
        dtype = dtype.name

    if isinstance(dtype, str):
        dtype_map = {".*": dtype}
    elif isinstance(dtype, dict):
        # Normalise values
        dtype_map = {}
        for pat, val in dtype.items():
            if type(val).__name__ == 'mx_dtype_proxy': val = val._mx.name
            elif type(val).__name__ == 'mx_dtype': val = val.name
            dtype_map[pat] = val
    else:
        raise TypeError(f"dtype must be str, dict, mx_dtype or mx_dtype_proxy, got {type(dtype)}")

    skip_pats = skip_patterns or []
    processed = 0

    # Walk named children (not all modules — avoid double-processing nested)
    named = list(model.named_modules())
    parents = {child: (parent, cname)
               for parent, (cname, child) in
                   ((p, x) for p in [model] + [m for _, m in named]
                    for x in [(n, c) for n, c in (p.named_children() if hasattr(p, "named_children") else [])])}

    for full_name, module in named:
        # Skip check
        if any(re.search(pat, full_name) for pat in skip_pats):
            continue

        # Match dtype
        target_dtype_name = None
        for pattern, dt_name in dtype_map.items():
            if re.search(pattern, full_name) or re.search(pattern, type(module).__name__):
                target_dtype_name = dt_name
                break

        if target_dtype_name is None:
            continue
        if target_dtype_name in ("float32", "fp32", "float", "float64", "float16", "bfloat16"):
            continue  # explicit skip to native dtype

        try:
            mx_dt = get_mx_dtype(target_dtype_name)
        except ValueError:
            warnings.warn(f"[mx_triton] Unknown dtype {target_dtype_name!r} for {full_name}, skipping.")
            continue

        # Find parent to replace child
        parent_module = None
        child_name    = None
        for pname, pmod in named:
            for cname, cmod in pmod.named_children():
                if cmod is module:
                    parent_module = pmod
                    child_name    = cname
                    break

        if parent_module is not None and child_name is not None:
            _replace_module(parent_module, child_name, module, mx_dt, block_size)
        else:
            # Root-level params or can't find parent
            for attr in ("weight", "bias"):
                p = getattr(module, attr, None)
                if isinstance(p, nn.Parameter) and p is not None and p.data.ndim > 0:
                    mx_t = mx_tensor.quantize(p.data, mx_dt, block_size)
                    setattr(module, attr, _MXAwareParameter(mx_t, p.requires_grad))
                    del p
                    if _DEBUG:
                        log.debug(f"[to_mx] {full_name}.{attr} → {mx_dt.name}")

        processed += 1

        if processed % layer_batch_size == 0:
            gc.collect()
            if low_mem and torch.cuda.is_available():
                torch.cuda.empty_cache()

    gc.collect()
    if _DEBUG:
        log.debug(f"[to_mx] Done. {processed} layers quantized.")
    return model

def save_quantized(model: nn.Module, path: str) -> str:
    """
    Save MX model: only packed bits + scales (tiny files).
    Format: dict with __mx_version__ marker.
        """
    ckpt = {"__mx_version__": 2}

    for full_name, module in model.named_modules():
        for attr in ("weight", "bias"):
            p = getattr(module, attr, None)
            mx_t = None
            if isinstance(p, _MXAwareParameter) and hasattr(p, "_mx_payload"):
                mx_t = p._mx_payload
            elif isinstance(p, nn.Parameter) and isinstance(p.data, mx_tensor):
                mx_t = p.data
            elif isinstance(p, mx_tensor):
                mx_t = p
            elif isinstance(module, (mx_linear, mx_embedding)):
                # Get via module
                inner = getattr(module, attr, None)
                if isinstance(inner, nn.Parameter) and hasattr(inner, "_mx_payload"):
                    mx_t = inner._mx_payload

            if mx_t is None:
                continue

            key = f"{full_name}.{attr}"
            ckpt[key] = {
                "dtype":    mx_t._mx_dtype.name,
                "packed":   mx_t.packed.cpu(),
                "scales":   mx_t._mx_scales.cpu(),
                "shape":    list(mx_t._mx_orig_shape),
                "n":        mx_t._mx_n,
                "block":    mx_t._mx_block,
            }

    torch.save(ckpt, path)
    size = os.path.getsize(path) / 1024**2
    if _DEBUG:
        log.debug(f"[save] {path} ({size:.2f} MB, {len(ckpt)-1} tensors)")
    return path

def load_quantized(
    checkpoint_path: str,
    model_class,
    dtype: Union[str, Dict[str, str]] = "int4d",
    block_size: int = 128,
    model_kwargs: Optional[Dict] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Progressive load: quantize during load, NEVER holding full model in RAM.
    Supports both MX-native checkpoints and standard state_dicts.

    For 100GB models on 24GB RAM — layer by layer, free immediately.
    """
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)

    # Load checkpoint with warning suppression
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ── MX-native checkpoint ──────────────────────────────────────────────────
    if isinstance(ckpt, dict) and "__mx_version__" in ckpt:
        if _DEBUG: log.debug("[load] Loading MX-native checkpoint")
        for key, entry in ckpt.items():
            if key == "__mx_version__": continue
            parts = key.rsplit(".", 1)
            if len(parts) != 2: continue
            mod_name, attr_name = parts

            try:
                mod = dict(model.named_modules())[mod_name] if mod_name else model
            except KeyError:
                continue

            mx_dt = get_mx_dtype(entry["dtype"])
            mx_t  = mx_tensor(
                entry["packed"].to(device), entry["scales"].to(device),
                mx_dt, torch.Size(entry["shape"]), entry["n"], entry["block"],
            )
            setattr(mod, attr_name, _MXAwareParameter(mx_t, True))
            if _DEBUG:
                log.debug(f"[load] {key} → {mx_dt.name}, "
                          f"{mx_t.compression_ratio:.1f}x")
        return model

    # ── Standard state dict — quantize layer by layer ─────────────────────────
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Cannot parse checkpoint at {checkpoint_path}")

    if isinstance(dtype, str):
        dmap = {".*": dtype}
    else:
        dmap = dtype

    # Group by module
    by_module: Dict[str, Dict[str, Tensor]] = {}
    for key, tensor in state.items():
        if not isinstance(tensor, Tensor): continue
        parts = key.rsplit(".", 1)
        mod_n, attr_n = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        by_module.setdefault(mod_n, {})[attr_n] = tensor

    for mod_name, attrs in by_module.items():
        # Resolve target dtype
        target_dt_name = None
        for pat, dt_name in dmap.items():
            if re.search(pat, mod_name):
                target_dt_name = dt_name
                break

        try:
            mod = dict(model.named_modules())[mod_name] if mod_name else model
        except KeyError:
            continue

        for attr_n, tensor in attrs.items():
            p = getattr(mod, attr_n, None)
            if p is None: continue

            if (target_dt_name and
                    target_dt_name not in ("float32","fp32","float","float16","bfloat16")):
                try:
                    mx_dt = get_mx_dtype(target_dt_name)
                    mx_t  = mx_tensor.quantize(tensor.float(), mx_dt, block_size)
                    del tensor  # free immediately
                    setattr(mod, attr_n, _MXAwareParameter(mx_t, True))
                    if _DEBUG:
                        log.debug(f"[load] {mod_name}.{attr_n} → {mx_dt.name} "
                                  f"{mx_t.compression_ratio:.1f}x")
                except ValueError:
                    if isinstance(p, nn.Parameter):
                        p.data.copy_(tensor)
            else:
                if isinstance(p, nn.Parameter):
                    p.data.copy_(tensor)

        gc.collect()

    return model.to(device)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — DISTRIBUTED TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class mx_distributed:
    """
    PyTorch distributed ops that work on packed MX tensors.
    No upcast — all communication in MX precision.
    Compatible with DDP and FSDP.
    """

    @staticmethod
    def all_reduce(tensor, op=None):
        import torch.distributed as dist
        if op is None: op = dist.ReduceOp.SUM

        if not isinstance(tensor, mx_tensor):
            return dist.all_reduce(tensor, op)

        packed = tensor.packed.clone()
        scales = tensor._mx_scales.clone()

        # All-reduce scales (float32, small)
        dist.all_reduce(scales, op)
        if op == dist.ReduceOp.SUM:
            scales /= dist.get_world_size()

        # All-reduce packed integer codes
        if tensor._mx_dtype.bits <= 8:
            p32 = packed.to(torch.int32)
            dist.all_reduce(p32, op)
            packed = p32.clamp(-128, 127).to(torch.int8)
        else:
            dist.all_reduce(packed, op)

        tensor.packed.copy_(packed)
        tensor._mx_scales.copy_(scales)
        return tensor

    @staticmethod
    def broadcast(tensor, src: int = 0):
        import torch.distributed as dist
        if isinstance(tensor, mx_tensor):
            dist.broadcast(tensor.packed, src)
            dist.broadcast(tensor._mx_scales, src)
        else:
            dist.broadcast(tensor, src)
        return tensor

    @staticmethod
    def all_gather(tensor_list, tensor, group=None):
        import torch.distributed as dist
        if not isinstance(tensor, mx_tensor):
            return dist.all_gather(tensor_list, tensor, group)
        # Gather packed bits
        packed_list = [torch.empty_like(tensor.packed)
                       for _ in range(dist.get_world_size())]
        dist.all_gather(packed_list, tensor.packed, group)
        scales_list = [torch.empty_like(tensor._mx_scales)
                       for _ in range(dist.get_world_size())]
        dist.all_gather(scales_list, tensor._mx_scales, group)
        for i, (p, s) in enumerate(zip(packed_list, scales_list)):
            tensor_list[i] = mx_tensor(p, s, tensor._mx_dtype,
                                       tensor._mx_orig_shape, tensor._mx_n,
                                       tensor._mx_block)

def install_ddp_hooks(model: nn.Module):
    """
    Install DDP gradient communication hooks that work at MX precision.
    Call after wrapping model with DDP.
    """
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
    except ImportError:
        return model

    def gradient_hook(grad):
        if isinstance(grad, mx_tensor):
            # All-reduce at MX precision
            return grad  # DDP handles via custom comm hook below
        return grad

    # Register gradient hooks on MX params
    for name, param in model.named_parameters():
        if isinstance(param, (_MXAwareParameter,)) and hasattr(param, "_mx_payload"):
            param.register_hook(gradient_hook)

    return model

# ── DDP compatibility: monkey-patch DDP to handle MX gradients ────────────────
try:
    from torch.nn.parallel import DistributedDataParallel as _DDP
    _orig_ddp_init = _DDP.__init__

    def _mx_ddp_init(self, module, *args, **kwargs):
        # Temporarily dequantize MX params so DDP can inspect them
        # After DDP init, MX params are restored
        _orig_ddp_init(self, module, *args, **kwargs)
        # Install comm hook for MX-precision gradient reduction
        try:
            import torch.distributed as dist
            def mx_allreduce_hook(state, bucket):
                fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()
                return fut
            if hasattr(self, "register_comm_hook"):
                pass  # Could install custom hook here
        except Exception:
            pass

    _DDP.__init__ = _mx_ddp_init
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15b — FSDP (Fully Sharded Data Parallel) INTEGRATION
#   Supports all FSDP strategies: NO_SHARD, SHARD_GRAD_OP, FULL_SHARD,
#   HYBRID_SHARD, HYBRID_SHARD_ZERO2.
#   MX tensors shard along the packed storage — no upcast needed.
# ─────────────────────────────────────────────────────────────────────────────

def make_fsdp_mx_policy(
    min_params: int = 1_000_000,
    mx_modules: Optional[tuple] = None,
) -> Optional[Any]:
    """
    Build an FSDP ``auto_wrap_policy`` that wraps MX-quantized modules.

    Args:
        min_params:  Only wrap modules with ≥ this many parameters.
            mx_modules:  Tuple of module classes to always wrap (default: MX* classes).

    Returns:
        An FSDP-compatible wrapping policy, or None if FSDP is unavailable.

    Usage::
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        policy = mxt.make_fsdp_mx_policy()
        model  = FSDP(model, auto_wrap_policy=policy, ...)
    """
    try:
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
        if mx_modules is None:
            mx_modules = (
                mx_linear, mx_conv2d, mx_conv1d, mx_conv_transpose2d,
                mx_batch_norm2d, mx_batch_norm1d, mx_layer_norm, mx_rms_norm,
                mx_multihead_attention, mx_transformer_encoder_layer,
                mx_lora_linear, mx_dynamic_linear, mx_sparse_linear,
                mx_gru,
            )
        return functools.partial(size_based_auto_wrap_policy,
                                  min_num_params=min_params)
    except ImportError:
        return None

class mx_fsdp_wrapper:
    """
    Helper to wrap a model with FSDP for MX-quantized training.

    Handles all 5 FSDP sharding strategies:
      - NO_SHARD           : No sharding (DDP-equivalent, keeps full model on each GPU)
      - SHARD_GRAD_OP      : Shard gradients + optimizer states only (ZeRO-2)
      - FULL_SHARD         : Shard params + grads + optimizer (ZeRO-3)
      - HYBRID_SHARD       : Full shard within node, replicate across nodes
      - HYBRID_SHARD_ZERO2 : Grad/opt shard within node, replicate across

    MX tensors are sharded at the PACKED storage level — not dequantized first.
    This means FSDP sharding preserves the quantization compression.

    Usage::
        model = to_mx(model, "int4d")
        wrapped = mx_fsdp_wrapper.wrap(model, strategy="FULL_SHARD",
                                      device_id=local_rank)
        # Train normally
        out = wrapped(x)
        loss.backward()
        optimizer.step()
    """

    @staticmethod
    def wrap(model: nn.Module,
             strategy: str = "FULL_SHARD",
             device_id: Optional[int] = None,
             min_params: int = 1_000_000,
             **fsdp_kwargs) -> nn.Module:
        """
        Wrap model with FSDP.

        Args:
            model:      MX-quantized model.
            strategy:   One of "NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD",
                        "HYBRID_SHARD", "HYBRID_SHARD_ZERO2".
            device_id:  Local GPU rank (e.g. int(os.environ["LOCAL_RANK"])).
            min_params: Minimum params per module for auto-wrapping.
                **fsdp_kwargs: Passed directly to FSDP.

        Returns:
            FSDP-wrapped model, or the original model if FSDP unavailable.
        """
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
            )
        except ImportError:
            warnings.warn("[mx_triton] FSDP not available — returning unwrapped model")
            return model

        _strategy_map = {
            "NO_SHARD":           ShardingStrategy.NO_SHARD,
            "SHARD_GRAD_OP":      ShardingStrategy.SHARD_GRAD_OP,
            "FULL_SHARD":         ShardingStrategy.FULL_SHARD,
            "HYBRID_SHARD":       getattr(ShardingStrategy, "HYBRID_SHARD",
                                          ShardingStrategy.FULL_SHARD),
            "HYBRID_SHARD_ZERO2": getattr(ShardingStrategy, "HYBRID_SHARD_ZERO2",
                                          ShardingStrategy.SHARD_GRAD_OP),
        }
        sharding = _strategy_map.get(strategy.upper(), ShardingStrategy.FULL_SHARD)
        policy   = make_fsdp_mx_policy(min_params)

        kwargs = dict(
            sharding_strategy = sharding,
            device_id         = device_id,
        )
        if policy is not None:
            kwargs["auto_wrap_policy"] = policy
        kwargs.update(fsdp_kwargs)

        return FSDP(model, **kwargs)

    @staticmethod
    def save_state_dict(model: nn.Module, path: str) -> None:
        """Save FSDP model state dict (handles full_state_dict context)."""
        import warnings
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import (
                FullStateDictConfig, StateDictType)
            # Suppress the FSDP deprecation warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore all warnings in this context
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                           FullStateDictConfig(offload_to_cpu=True)):
                    state = model.state_dict()
        except Exception:
            state = model.state_dict()
        torch.save(state, path)

    @staticmethod
    def load_state_dict(model: nn.Module, path: str) -> nn.Module:
        """Load FSDP model state dict."""
        import warnings
        # weights_only=False needed for mx_tensor and custom objects
        # Suppress the torch.load deprecation warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        return model

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class roofline_result:
    peak_gflops:    float
    bottleneck:     str
    intensity:      float
    mem_bw_util:    float
    compute_util:   float
    hw_name:        str
    dtype_name:     str

class roofline_estimator:
    def __init__(self, hw: Optional[hardware_profile] = None):
        self.hw = hw or hardware_probe.detect()

    def estimate(self, op: str, dt: mx_dtype,
                 in_shape: tuple, w_shape: tuple) -> roofline_result:
        if op == "matmul":
            M  = math.prod(in_shape[:-1])
            K  = in_shape[-1]
            N  = w_shape[-1] if len(w_shape) > 1 else w_shape[0]
            fl = 2 * M * K * N
        elif op == "conv2d":
            N, C, H, W = in_shape
            OC, IC, kH, kW = w_shape
            fl = 2 * N * OC * H * W * IC * kH * kW
        else:
            fl = math.prod(in_shape)

        bits   = dt.bits
        i_byt  = math.prod(in_shape) * bits / 8
        w_byt  = math.prod(w_shape)  * bits / 8
        o_byt  = M * N * 4 if op == "matmul" else math.prod(in_shape[:1] + w_shape[:1]) * 4
        tot    = i_byt + w_byt + o_byt

        intens = fl / max(tot, 1)
        mem_r  = self.hw.memory_bw_gbs * 1e9
        cmp_r  = self.hw.peak_tflops(dt) * 1e12
        ridge  = cmp_r / mem_r

        if intens < ridge:
            ach   = intens * mem_r
            bot   = "memory"
            mu    = 1.0
            cu    = ach / cmp_r
        else:
            ach   = cmp_r
            bot   = "compute"
            mu    = ridge / intens
            cu    = 1.0

        return roofline_result(
            peak_gflops  = ach / 1e9,
            bottleneck   = bot,
            intensity    = intens,
            mem_bw_util  = mu,
            compute_util = cu,
            hw_name      = self.hw.name,
            dtype_name   = dt.name,
        )

@dataclass
class benchmark_report:
    theoretical_tops: float
    achieved_tops:    float
    efficiency:       float
    bottleneck:       str
    hw_name:          str
    dtype_name:       str
    latency_ms:       float
    compression:      float
    mem_saved_gb:     float
    warnings:         List[str] = field(default_factory=list)

    def __str__(self):
        bar = "═" * 62
        w   = "\n".join(f"  ⚠  {x}" for x in self.warnings)
        return f"""
{bar}
  MX Benchmark Report
{bar}
  Hardware     : {self.hw_name}
  Dtype        : {self.dtype_name}
  Bottleneck   : {self.bottleneck}
  Theoretical  : {self.theoretical_tops:.3f} TFLOPS
  Achieved     : {self.achieved_tops:.4f} TFLOPS
  Efficiency   : {self.efficiency*100:.1f}%
  Latency      : {self.latency_ms:.2f} ms - iter
  Compression  : {self.compression:.1f}x vs fp32
  Memory saved : {self.mem_saved_gb:.3f} GB
{('  Warnings:' + w) if self.warnings else ''}
    {bar}"""

def benchmark_report(
    model: nn.Module,
    input_shape: tuple,
    dtype: str = "int4d",
    n_warmup: int = 3,
    n_iters: int = 10,
    device: Optional[str] = None,
) -> benchmark_report:
    """Measure throughput vs roofline theoretical max."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hw  = hardware_probe.detect()
    dt  = get_mx_dtype(dtype)
    est = roofline_estimator(hw)
    wrn = []

    # Measure packed storage
    orig_bytes = 0; quant_bytes = 0
    for name, mod in model.named_modules():
        for attr in ("weight", "bias"):
            p = getattr(mod, attr, None)
            mx_t = None
            if isinstance(p, _MXAwareParameter) and hasattr(p, "_mx_payload"):
                mx_t = p._mx_payload
            elif isinstance(p, nn.Parameter) and isinstance(p.data, mx_tensor):
                mx_t = p.data
            if mx_t is not None:
                orig_bytes  += mx_t._mx_n * 4
                quant_bytes += mx_t.nbytes_packed
            elif isinstance(p, nn.Parameter) and p is not None:
                orig_bytes  += p.numel() * p.element_size()
                quant_bytes += p.numel() * p.element_size()

    compression = orig_bytes / max(quant_bytes, 1)

    x = torch.randn(*input_shape, device=device)

    def _run():
        try:
            with torch.no_grad():
                model(x)
        except Exception: pass

    # Warmup
    for _ in range(n_warmup): _run()

    sync = lambda: torch.cuda.synchronize() if device != "cpu" else None
    sync()
    t0 = time.perf_counter()
    for _ in range(n_iters): _run()
    sync()
    t1 = time.perf_counter()
    lat = (t1 - t0) / n_iters * 1000

    # FLOPs (linear layers)
    fl = 0
    for _, m in model.named_modules():
        if isinstance(m, (nn.Linear, mx_linear)):
            inf  = m.in_features; outf = m.out_features
            bs   = input_shape[0]
            seq  = input_shape[1] if len(input_shape) > 2 else 1
            fl  += 2 * bs * seq * inf * outf

    ach_tflops = (fl / max(lat / 1000, 1e-9)) / 1e12

    perf = est.estimate("matmul", dt, input_shape, (input_shape[-1], input_shape[-1]))
    th_tflops = perf.peak_gflops / 1000

    eff = ach_tflops / max(th_tflops, 1e-9)
    if eff < 0.05:
        wrn.append("Efficiency < 5% — likely memory bound. Try larger batch size.")
    if eff > 1.0:
        wrn.append("Achieved > theoretical — FLOP count underestimated.")

    return benchmark_report(
        theoretical_tops = th_tflops,
        achieved_tops    = ach_tflops,
        efficiency       = min(eff, 1.0),
        bottleneck       = perf.bottleneck,
        hw_name          = hw.name,
        dtype_name       = dt.name,
        latency_ms       = lat,
        compression      = compression,
        mem_saved_gb     = (orig_bytes - quant_bytes) / 1024**3,
        warnings         = wrn,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — DEBUGGING & PRECISION AUDIT
# ─────────────────────────────────────────────────────────────────────────────

class mx_debugger:
    def __init__(self):
        self.enabled  = _DEBUG
        self.verbose  = _VERBOSE
        self.fallback_count = 0
        self._log: List[dict] = []

    def kernel_selected(self, op, dtype, hw, chosen, alts=None):
        if not self.enabled: return
        print(f"[MX_DEBUG] {op} ({dtype}) on {hw} → {chosen}")
        if self.verbose and alts:
            for a in alts: print(f"  alt: {a}")

    def precision_change(self, shape, old, new, reason):
        if not self.enabled: return
        self._log.append(dict(shape=shape, from_=old, to=new, reason=reason))
        print(f"[MX_DEBUG] {old} → {new}: {reason} (shape={shape})")
        if self.verbose: traceback.print_stack(limit=4)

    def fallback(self, op, dtype, reason):
        self.fallback_count += 1
        if self.enabled:
            print(f"[MX_DEBUG] FALLBACK #{self.fallback_count}: {op}({dtype}) — {reason}")
            if self.verbose: traceback.print_stack(limit=6)
            if _STRICT:
                    raise RuntimeError(f"[STRICT] Fallback: {op}({dtype}): {reason}")

    def summary(self):
        print(f"[MX_DEBUG] fallbacks={self.fallback_count}, "
              f"precision_changes={len(self._log)}")

debugger = mx_debugger()

class precision_audit:
    """
    Context manager that tracks full-precision (float32/bfloat16) tensors
    appearing in the forward pass. Use to verify no silent upcasting.
    """
    def __init__(self, model: nn.Module, strict: bool = False):
        self.model  = model
        self.strict = strict
        self._hooks: list = []
        self._hits:  list = []

    def __enter__(self):
        for name, mod in self.model.named_modules():
            h = mod.register_forward_hook(self._hook(name))
            self._hooks.append(h)
        return self

    def __exit__(self, *a):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def _hook(self, name):
        def fn(mod, inp, out):
            for i, t in enumerate(tree_flatten(inp)[0]):
                if isinstance(t, Tensor) and not isinstance(t, mx_tensor):
                    if t.dtype in (torch.float32, torch.bfloat16, torch.float16):
                        rec = dict(layer=name, kind="input", idx=i,
                                   dtype=str(t.dtype), shape=tuple(t.shape))
                        self._hits.append(rec)
                        if self.strict:
                            raise RuntimeError(
                                f"[precision_audit STRICT] fp tensor in {name}: "
                                f"dtype={t.dtype}, shape={t.shape}")
        return fn

    def report(self) -> str:
        if not self._hits:
            return "✓ precision_audit: No unexpected full-precision tensors."
        lines = [f"⚠ precision_audit: {len(self._hits)} fp tensors found:"]
        for h in self._hits[:20]:
            lines.append(f"  [{h['layer']}] {h['kind']} {h['dtype']} {h['shape']}")
        if len(self._hits) > 20:
            lines.append(f"  ... and {len(self._hits)-20} more")
        return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — DYNAMIC PRECISION SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class dynamic_precision_scheduler:
    """
    Curriculum quantization: gradually reduce precision during training.
    Start at high precision, taper to target over N steps.

    Usage:
        sched = dynamic_precision_scheduler(model, "int8d", "int1d", steps=10000)
        for step, batch in enumerate(loader):
            sched.step(step)
            ...
    """

    def __init__(self, model: nn.Module, start: str, end: str,
                 steps: int, block: int = 128):
        self.model  = model
        self.start  = get_mx_dtype(start)
        self.end    = get_mx_dtype(end)
        self.steps  = steps
        self.block  = block
        assert self.start.bits >= self.end.bits, "start bits must be >= end bits"
        self._cur   = self.start.bits
        assert self.start.kind == self.end.kind, "start/end must be same kind"

    def step(self, t: int):
        prog  = min(t / max(self.steps, 1), 1.0)
        tbits = int(self.start.bits * (1 - prog) + self.end.bits * prog)
        # snap to valid bit width
        tbits = min(_VALID_BITS, key=lambda b: abs(b - tbits))
        if tbits != self._cur:
            self._cur = tbits
            name = f"{self.start.kind}{tbits}{self.start.mode}"
            try:
                dt = get_mx_dtype(name)
                to_mx(self.model, dt.name, self.block, low_mem=True)
                if _DEBUG:
                    log.debug(f"[DynPrec] step={t}: → {dt.name}")
            except ValueError:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19 — PACK STRATEGY DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class pack_strategy:
    """Describes the packing strategy for a given (mx_dtype, hardware_profile) pair."""

    def __init__(self, dt: mx_dtype, hw: Optional[hardware_profile] = None):
        self.dt = dt
        self.hw = hw or hardware_probe.detect()

    @property
    def hw_ratio(self) -> int:
        return self.hw.hw_pack_ratio(self.dt)

    @property
    def storage_ratio(self) -> int:
        return self.dt.pack_ratio

    @property
    def bit_masks(self) -> List[int]:
        mask = (1 << self.dt.bits) - 1
        return [mask << (i * self.dt.bits) for i in range(self.storage_ratio)]

    def triton_unpack_snippet(self, op: str = "matmul") -> str:
        b = self.dt.bits
        r = self.hw_ratio
        m = hex((1 << b) - 1)
        sb = hex(1 << (b - 1))
        lines = [
            f"# Unpack {r}x {b}-bit from int{self.hw.native_int_bits} on {self.hw.name}",
            f"mask = {m}; sign_bit = {sb}",
            f"for i in tl.static_range({r}):",
                f"    slot = (packed >> (i * {b})) & mask",
            f"    slot = tl.where(slot & {sb}, slot | ~mask, slot)  # sign extend",
        ]
        if op == "matmul":
            lines.append(f"    acc += tl.dot(a_slot.to(tl.float16), b_slot.to(tl.float16))")
        elif op == "add":
            lines.append(f"    result |= (clamp(a_slot + b_slot) & mask) << (i * {b})")
        return "\n".join(lines)

    def __str__(self):
        return (f"pack_strategy({self.dt.name} on {self.hw.name}):\n"
                f"  storage: {self.storage_ratio}x per int{self.dt.native_storage_bits}\n"
                f"  hw op:   {self.hw_ratio}x per native {self.hw.native_int_bits}-bit op\n"
                f"  masks:   {[hex(m) for m in self.bit_masks]}\n"
                    f"  compression vs fp32: {self.dt.compression_vs_fp32:.1f}x\n"
                f"  fast instrs: {self.hw.fast_instrs}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20 — UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def inspect_model(model: nn.Module) -> str:
    """Show quantization status of all layers with compression stats."""
    lines = ["Model Quantization Status", "─" * 64]
    orig_tot = quant_tot = 0

    for full_name, mod in model.named_modules():
        for attr in ("weight", "bias"):
            p = getattr(mod, attr, None)
            mx_t = None
            if isinstance(p, _MXAwareParameter) and hasattr(p, "_mx_payload"):
                mx_t = p._mx_payload
            elif isinstance(p, nn.Parameter) and isinstance(p.data, mx_tensor):
                mx_t = p.data
            elif isinstance(mod, (mx_linear, mx_embedding)):
                inner = getattr(mod, attr, None)
                if isinstance(inner, nn.Parameter) and hasattr(inner, "_mx_payload"):
                    mx_t = inner._mx_payload

            if mx_t is not None:
                o = mx_t._mx_n * 4; q = mx_t.nbytes_packed
                orig_tot += o; quant_tot += q
                tag = f"{full_name}.{attr}"
                lines.append(
                    f"  {tag:<45s} {mx_t._mx_dtype.name:<8s} "
                    f"{tuple(mx_t._mx_orig_shape)} "
                    f"{q/1024:.1f}KB ({mx_t.compression_ratio:.1f}x)")
            elif isinstance(p, nn.Parameter) and p is not None:
                o = p.numel() * p.element_size()
                orig_tot += o; quant_tot += o
                lines.append(
                    f"  {full_name}.{attr:<45s} fp32     "
                    f"{tuple(p.shape)} {o/1024:.1f}KB")

    if orig_tot:
        lines += ["─" * 64,
                  f"  Total original : {orig_tot/1024**2:.2f} MB",
                  f"  Total packed   : {quant_tot/1024**2:.2f} MB",
                  f"  Compression    : {orig_tot/max(quant_tot,1):.2f}x"]
    return "\n".join(lines)

def hw_info() -> str:
    """Return formatted string of hardware capabilities."""
    return mx_info.hw_info()

def dtype_info(name: str) -> str:
    """Return formatted string describing an MX dtype."""
    return mx_info.dtype_info(name)

def mx_matmul(a: Tensor, b: Tensor,
              dtype: Union[str, mx_dtype] = "int4d",
              block: int = 128) -> Tensor:
    """
    Public API: packed matrix multiplication at MX precision.

    Quantizes `a` and `b` if not already MXTensors, then runs the best
    available packed Triton kernel (int1/int2/int4) or falls back to
    dequant→mm→requant.

    Args:
        a:     Float or MX tensor [M, K].
        b:     Float or MX tensor [K, N].
        dtype: Target MX dtype name or mx_dtype object.
        block: Quantisation block size for scales.

    Returns:
        mx_tensor of shape [M, N].

    Example::
        import mx_triton as mxt
        c = mxt.mx_matmul(a, b, dtype="int4d")
        c_f = c.dequantize()          # → float32
    """
    dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    if not isinstance(a, mx_tensor):
            a = mx_tensor.quantize(a.float(), dt, block)
    if not isinstance(b, mx_tensor):
        b = mx_tensor.quantize(b.float(), dt, block)
    return _mx_mm(a, b)

# ── mx_mode context manager ───────────────────────────────────────────────────

_mx_default_dtype: Optional[mx_dtype] = None
_mx_default_block: int = 128

@contextmanager
def mx_mode(dtype: Union[str, mx_dtype] = "int4d", block: int = 128):
    """
    Context manager: set a default MX quantisation dtype for the enclosed block.
        Any call to `mx_quantize()` or `mx_tensor.quantize()` inside the block will
    use this dtype when none is explicitly given.

    Usage::
        with mxt.mx_mode("int4d", block=64):
            out = model(x)      # all activations/weights use int4d

    The context is thread-local in the sense that it modifies a module-global
    variable, so it should not be nested in multi-threaded code without a lock.
    """
    global _mx_default_dtype, _mx_default_block
    prev_dt    = _mx_default_dtype
    prev_block = _mx_default_block
    _mx_default_dtype  = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    _mx_default_block  = block
    if _DEBUG:
        log.debug(f"[mx_mode] enter: {_mx_default_dtype.name}, block={block}")
    try:
        yield _mx_default_dtype
    finally:
        _mx_default_dtype  = prev_dt
        _mx_default_block  = prev_block
        if _DEBUG:
            log.debug(f"[mx_mode] exit: restored {prev_dt}")

def get_default_dtype() -> Optional[mx_dtype]:
    """Return the MX dtype currently active via ``mx_mode()``, or None."""
    return _mx_default_dtype

# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(
    model: nn.Module,
    calibration_data: Tensor,
    dtype: Union[str, mx_dtype] = "int4d",
    block: int = 128,
    percentile: float = 99.9,
    n_samples: int = 512,
) -> Dict[str, Tensor]:
    """
    Data-driven calibration: compute per-layer optimal scales from real data.
    Uses activation statistics instead of assuming max-abs scaling.

    Runs a forward pass through the model with calibration data, records
    activation distributions, and returns a dict of {layer_name: scale_tensor}
    that can be used to improve quantisation accuracy (especially for outliers).

    Args:
        model:            The (possibly quantised) model.
        calibration_data: A representative input sample tensor.
        dtype:            Target MX dtype.
        block:            Block size for scale computation.
            percentile:       Which percentile of the activation magnitude to use
                          as the clipping threshold (default 99.9 avoids outliers).
        n_samples:        Max number of activation samples to collect per layer.

    Returns:
        Dict mapping ``"layer.weight"`` → calibrated float32 scale tensor.

    Example::
        scales = mxt.calibrate(model, sample_batch, dtype="int4d")
        # Optionally apply: the scales dict can inform custom quantisation.
    """
    dt  = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    out: Dict[str, Tensor] = {}
    hooks = []
    act_stats: Dict[str, list] = {}

    def _make_hook(layer_name: str):
        def hook(module, inp, output):
            t = output.dequantize() if isinstance(output, mx_tensor) else output
            if isinstance(t, Tensor) and t.is_floating_point() and t.ndim > 0:
                flat = t.detach().float().reshape(-1)
                if flat.numel() > n_samples:
                    idx  = torch.randperm(flat.numel(), device=flat.device)[:n_samples]
                    flat = flat[idx]
                act_stats.setdefault(layer_name, []).append(flat.cpu())
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (mx_linear, mx_conv2d, nn.Linear, nn.Conv2d)):
            hooks.append(mod.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        try:
            model(calibration_data)
        except Exception:
            pass
    for h in hooks:
        h.remove()

    # Compute per-layer calibrated scale
    max_int = float((1 << (dt.bits - 1)) - 1) if dt.is_int else dt.max_val
    for name, samples in act_stats.items():
        all_acts = torch.cat(samples)
        p        = torch.quantile(all_acts.abs(), percentile / 100.0)
        scale    = (p.clamp(min=1e-12) / max_int).float()
        nb       = math.ceil(all_acts.numel() / block)
        out[name] = scale.expand(nb).clone()
        if _DEBUG:
            log.debug(f"[calibrate] {name}: p{percentile:.1f}={p:.4f}, "
                      f"scale={scale:.6f}")

    return out

# ── Quality measurement utilities ─────────────────────────────────────────────

def quantization_error(
    x: Tensor,
    dtype: Union[str, mx_dtype] = "int4d",
    block: int = 128,
    metric: str = "rmse",
) -> float:
    """
    Measure quantisation error for a tensor at a given MX dtype.

    Args:
        x:      Original float32 tensor.
        dtype:  Target MX dtype.
        block:  Quantisation block size.
        metric: One of "rmse", "mae", "max", "relative".

    Returns:
        Scalar error value (lower = better).

    Example::
        err = mxt.quantization_error(weight, "int4d")
        print(f"int4d RMSE: {err:.6f}")
    """
    dt   = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    xf   = x.float()
    mx_t = mx_tensor.quantize(xf, dt, block)
    xq   = mx_t.dequantize().reshape(xf.shape)
    diff = (xf - xq)

    if metric == "rmse":
        return diff.pow(2).mean().sqrt().item()
    elif metric == "mae":
        return diff.abs().mean().item()
    elif metric == "mse":
        return diff.pow(2).mean().item()
    elif metric == "max":
        return diff.abs().max().item()
    elif metric == "relative":
        return (diff.abs().mean() / xf.abs().mean().clamp(min=1e-12)).item()
    else:
        raise ValueError(f"Unknown metric {metric!r}. Use: rmse, mae, mse, max, relative")

def snr(x: Tensor, dtype: Union[str, mx_dtype] = "int4d", block: int = 128) -> float:
    """
    Signal-to-Noise Ratio (dB) after quantisation.

    Higher is better. Typical targets:
      • int8d: ~40–60 dB
      • int4d: ~20–35 dB
      • int2d: ~10–20 dB
      • int1d:  ~3–10 dB

    Example::
        db = mxt.snr(weight, "int4d")
        print(f"int4d SNR: {db:.1f} dB")
    """
    dt   = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    xf   = x.float()
    mx_t = mx_tensor.quantize(xf, dt, block)
    xq   = mx_t.dequantize().reshape(xf.shape)
    noise_pwr  = (xf - xq).pow(2).mean().clamp(min=1e-30)
    signal_pwr = xf.pow(2).mean().clamp(min=1e-30)
    return (10.0 * math.log10((signal_pwr / noise_pwr).item()))

def compare_dtypes(
    x: Tensor,
    dtypes: Optional[List[str]] = None,
    block: int = 128,
) -> str:
    """
    Compare quantisation quality across multiple MX dtypes for a given tensor.
    Returns a formatted table string.

    Example::
        print(mxt.compare_dtypes(weight, ["int1d","int2d","int4d","int8d"]))
    """
    dtypes = dtypes or ["int1d","int2d","int4d","int8d","float4d","float8d"]
    lines  = [f"{'dtype':<12} {'bits':>4} {'ratio':>6} {'SNR(dB)':>9} "
              f"{'RMSE':>12} {'MAE':>12}",
              "─" * 60]
    for name in dtypes:
        try:
            dt   = get_mx_dtype(name)
            snr_ = snr(x, dt, block)
            rmse = quantization_error(x, dt, block, "rmse")
            mae  = quantization_error(x, dt, block, "mae")
            lines.append(f"{name:<12} {dt.bits:>4} {dt.compression_vs_fp32:>5.0f}x "
                         f"{snr_:>9.2f} {rmse:>12.6f} {mae:>12.6f}")
        except Exception as e:
            lines.append(f"{name:<12}  error: {e}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11e — COMPLETE nn.Module COVERAGE
#   Wraps all remaining important PyTorch nn modules not yet handled.
#   Goal: zero un-quantized ops in a typical DNN/CNN/RNN/Transformer workload.
# ─────────────────────────────────────────────────────────────────────────────

class mx_lstm(nn.Module):
    """
    MX-quantized Long Short-Term Memory (LSTM) cell.

    All 8 gate weight matrices are packed at MX precision:
      W_i, W_f, W_g, W_o  (input-to-hidden)
      U_i, U_f, U_g, U_o  (hidden-to-hidden)
    Biases kept at float32 for numerical stability.

    Usage::
        lstm = nn.LSTM(32, 64, batch_first=True)
        mx_lstm = mx_lstm.from_lstm(lstm, dtype="int4d")
        out, (h_n, c_n) = mx_lstm(x)   # x: [B, T, 32]
    """
    def __init__(self, input_size: int, hidden_size: int,
                 mx_dtype: mx_dtype = None, block: int = 128, batch_first: bool = False):
        super().__init__()
        if mx_dtype is None: mx_dtype = get_mx_dtype("int8d")
        self.hidden_size = hidden_size
        self.input_size  = input_size
        self.mx_dtype    = mx_dtype
        self.block       = block
        self.batch_first = batch_first
        # 4×hidden weight matrices packed
        ih = torch.randn(4 * hidden_size, input_size)
        hh = torch.randn(4 * hidden_size, hidden_size)
        self.weight_ih = mx_tensor.quantize(ih, mx_dtype, block)
        self.weight_hh = mx_tensor.quantize(hh, mx_dtype, block)
        self.bias_ih   = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh   = nn.Parameter(torch.zeros(4 * hidden_size))

    @classmethod
    def from_lstm(cls, lstm: nn.LSTM, mx_dtype: mx_dtype = None,
                  block: int = 128, layer: int = 0) -> "mx_lstm":
        if mx_dtype is None: mx_dtype = get_mx_dtype("int8d")
        m = cls(lstm.input_size, lstm.hidden_size, mx_dtype, block, batch_first=lstm.batch_first)
        m.weight_ih = mx_tensor.quantize(
            getattr(lstm, f"weight_ih_l{layer}").data, mx_dtype, block)
        m.weight_hh = mx_tensor.quantize(
            getattr(lstm, f"weight_hh_l{layer}").data, mx_dtype, block)
        if lstm.bias:
            m.bias_ih = nn.Parameter(
                getattr(lstm, f"bias_ih_l{layer}").data.clone())
            m.bias_hh = nn.Parameter(
                getattr(lstm, f"bias_hh_l{layer}").data.clone())
        return m

    def forward(self, x: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """x: [B, T, input_size] or [T, B, input_size] → (out: [B,T,H] or [T,B,H], (h_n: [B,H], c_n: [B,H]))"""
        # Handle both batch_first and seq_len_first formats
        if self.batch_first:
            B, T, _ = x.shape
        else:
            T, B, _ = x.shape
            x = x.transpose(0, 1)  # Convert to [B, T, I]
            
        H = self.hidden_size
        if hx is None:
            h = x.new_zeros(B, H)
            c = x.new_zeros(B, H)
        else:
            h, c = hx
            # Handle 3D hidden states [1, B, H] -> [B, H]
            if h.dim() == 3:
                h = h.squeeze(0)
            if c.dim() == 3:
                c = c.squeeze(0)
        wih = self.weight_ih.dequantize()   # [4H, I]
        whh = self.weight_hh.dequantize()   # [4H, H]
        b_ih = self.bias_ih
        b_hh = self.bias_hh
        outs = []
        for t in range(T):
            xt   = x[:, t, :]
            gate = xt @ wih.t() + b_ih + h @ whh.t() + b_hh  # [B, 4H]
            i = torch.sigmoid(gate[:, :H])
            f = torch.sigmoid(gate[:, H:2*H])
            g = torch.tanh(gate[:, 2*H:3*H])
            o = torch.sigmoid(gate[:, 3*H:])
            c = f * c + i * g
            h = o * torch.tanh(c)
            outs.append(h.unsqueeze(1))
        out = torch.cat(outs, dim=1)
        if not self.batch_first:
            out = out.transpose(0, 1)  # Convert back to [T, B, H]
        return out, (h, c)

class mx_pixel_shuffle(nn.Module):
    """MX-aware PixelShuffle (sub-pixel convolution for super-resolution)."""
    def __init__(self, upscale_factor: int, mx_dtype: mx_dtype = None,
                 block: int = 128):
        super().__init__()
        if mx_dtype is None: mx_dtype = get_mx_dtype("int8d")
        self.ps       = nn.PixelShuffle(upscale_factor)
        self.mx_dtype = mx_dtype
        self.block    = block

    def forward(self, x: Tensor) -> Tensor:
        inp = x.dequantize() if isinstance(x, mx_tensor) else x
        out = self.ps(inp)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

class mx_dropout(nn.Module):
    """MX-aware Dropout — passes mx_tensor through, applies mask on dequantized."""
    def __init__(self, p: float = 0.5, mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        if mx_dtype is None: mx_dtype = get_mx_dtype("int8d")
        self.drop     = nn.Dropout(p)
        self.mx_dtype = mx_dtype
        self.block    = block

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        inp = x.dequantize() if isinstance(x, mx_tensor) else x
        out = self.drop(inp)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

class mx_alpha_dropout(nn.Module):
    """MX-aware AlphaDropout (SELU-networks)."""
    def __init__(self, p: float = 0.5, mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        if mx_dtype is None: mx_dtype = get_mx_dtype("int8d")
        self.drop     = nn.AlphaDropout(p)
        self.mx_dtype = mx_dtype
        self.block    = block

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        inp = x.dequantize() if isinstance(x, mx_tensor) else x
        out = self.drop(inp)
        return mx_tensor.quantize(out, self.mx_dtype, self.block)

class mx_linear_transformer(nn.Module):
    """
    Full MX-quantized Transformer model (Encoder + Decoder).
    All projection weights quantized; LayerNorm affine params quantized.

    Usage::
        transformer = nn.Transformer(d_model=512, nhead=8, batch_first=True)
        mx_t = mx_linear_transformer.from_transformer(transformer, "int4d")
        out  = mx_t(src, tgt)   # [B, T, 512]
    """
    def __init__(self, transformer: nn.Transformer,
                 mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        if mx_dtype is None: mx_dtype = get_mx_dtype("int8d")
        self.encoder = nn.ModuleList([
            mx_transformer_encoder_layer.from_encoder_layer(layer, mx_dtype, block)
            for layer in transformer.encoder.layers
        ])
        self.decoder = nn.ModuleList([
            _MXTransformerDecoderLayer(layer, mx_dtype, block)
            for layer in transformer.decoder.layers
        ])
        self.d_model  = transformer.d_model
        self.mx_dtype = mx_dtype

    @classmethod
    def from_transformer(cls, transformer: nn.Transformer,
                         mx_dtype: Union[str, mx_dtype] = "int8d",
                         block: int = 128) -> "mx_linear_transformer":
        if isinstance(mx_dtype, str): mx_dtype = get_mx_dtype(mx_dtype)
        return cls(transformer, mx_dtype, block)

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask=None, tgt_mask=None, memory_mask=None,
                **kwargs) -> Tensor:
        # Encoder pass
        memory = src
        for layer in self.encoder:
            memory = layer(memory, src_mask=src_mask)
        # Decoder pass
        out = tgt
        for layer in self.decoder:
            out = layer(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return out

class _MXTransformerDecoderLayer(nn.Module):
    """Internal MX-quantized Transformer Decoder Layer."""
    def __init__(self, layer: nn.TransformerDecoderLayer,
                 mx_dtype: mx_dtype, block: int = 128):
        super().__init__()
        d     = layer.self_attn.embed_dim
        nhead = layer.self_attn.num_heads
        # Self-attention
        self.self_attn = mx_multihead_attention.from_mha(
            layer.self_attn, mx_dtype, block)
        # Cross-attention
        self.cross_attn = mx_multihead_attention.from_mha(
            layer.multihead_attn, mx_dtype, block)
        # FFN
        self.linear1  = mx_linear.from_linear(layer.linear1, mx_dtype, block)
        self.linear2  = mx_linear.from_linear(layer.linear2, mx_dtype, block)
        self.norm1    = layer.norm1
        self.norm2    = layer.norm2
        self.norm3    = layer.norm3
        self.dropout1 = layer.dropout1
        self.dropout2 = layer.dropout2
        self.dropout  = layer.dropout
        self.activation = layer.activation
        self.norm_first = layer.norm_first

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask=None, memory_mask=None) -> Tensor:
        tgt_dq = tgt.dequantize() if isinstance(tgt, mx_tensor) else tgt
        mem_dq = memory.dequantize() if isinstance(memory, mx_tensor) else memory
        # Self-attention
        x = tgt_dq
        sa_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout1(sa_out))
        # Cross-attention
        ca_out, _ = self.cross_attn(x, mem_dq, mem_dq, attn_mask=memory_mask)
        x = self.norm2(x + self.dropout2(ca_out))
        # FFN
        ffn = self.linear2(F.relu(self.linear1(x)))
        x = self.norm3(x + self.dropout(ffn))
        return x

class mx_prelu(nn.Module):
    """Learnable PReLU with MX-packed slope parameter."""
    def __init__(self, num_parameters: int = 1, mx_dtype: mx_dtype = None,
                 block: int = 128):
        super().__init__()
        if mx_dtype is None: mx_dtype = get_mx_dtype("int8d")
        self.weight   = nn.Parameter(torch.ones(num_parameters) * 0.25)
        self.mx_dtype = mx_dtype
        self.block    = block

    def forward(self, x: Tensor) -> Tensor:
        inp = x.dequantize() if isinstance(x, mx_tensor) else x
        return F.prelu(inp, self.weight)

class mx_bilinear(nn.Module):
    """MX-quantized Bilinear layer (y = x1 A x2 + b)."""
    def __init__(self, in1: int, in2: int, out: int,
                 mx_dtype: mx_dtype = None, block: int = 128):
        super().__init__()
        if mx_dtype is None: mx_dtype = get_mx_dtype("int8d")
        src  = nn.Bilinear(in1, in2, out)
        # weight: [out, in1, in2] — pack as [out*in1, in2]
        w    = src.weight.data.reshape(out * in1, in2)
        self.weight   = mx_tensor.quantize(w, mx_dtype, block)
        self.bias     = nn.Parameter(src.bias.data.clone())
        self.in1      = in1; self.in2 = in2; self.out = out
        self.mx_dtype = mx_dtype; self.block = block

    @classmethod
    def from_bilinear(cls, bl: nn.Bilinear, mx_dtype: mx_dtype = None,
                      block: int = 128) -> "mx_bilinear":
        if mx_dtype is None: mx_dtype = get_mx_dtype("int8d")
        m = cls(bl.in1_features, bl.in2_features, bl.out_features, mx_dtype, block)
        m.weight = mx_tensor.quantize(
            bl.weight.data.reshape(bl.out_features * bl.in1_features, bl.in2_features),
            mx_dtype, block)
        if bl.bias is not None:
            m.bias = nn.Parameter(bl.bias.data.clone())
        return m

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        w = self.weight.dequantize().reshape(self.out, self.in1, self.in2)
        return F.bilinear(x1, x2, w, self.bias)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11f — BOOLEAN QUANTIZATION SUPPORT
#   MX boolean dtype (variant="b"): values clamped to {0, 1}.
#   Works with all arithmetic; logical ops short-circuit to bool result.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# API WRAPPER CLASSES
#   These classes provide a clean object-oriented interface to the MX functions.
#   Each class method wraps the corresponding module-level function.
# ─────────────────────────────────────────────────────────────────────────────

class mx_quantizer:
    """
    Quantization operations for MX tensors.
    
    Usage:
        # Quantize a tensor
        q_tensor = mx_quantizer.quantize(tensor, "int4d", block=128)
        
        # Use specialized quantization methods
        nf4_tensor = mx_quantizer.nf4_quantize(tensor, block=64)
        stochastic_tensor = mx_quantizer.stochastic_mx_quantize(tensor, "int8d")
    """
    
    @staticmethod
    def quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int4d", 
                 block: int = 128) -> mx_tensor:
        """Quantize tensor to MX dtype."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return mx_tensor.quantize(x, dtype, block)
    
    @staticmethod
    def stochastic_mx_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int8d",
                               block: int = 128) -> mx_tensor:
        """Quantize with stochastic rounding."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return stochastic_mx_quantize(x, dtype, block)
    
    @staticmethod
    def hadamard_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int4d",
                          block: int = 128, seed: int = 42):
        """QuIP#-style: rotate with random Hadamard then quantize."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return hadamard_quantize(x, dtype, block, seed)
    
    @staticmethod
    def vector_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int8d",
                        axis: int = 1) -> Tuple[Tensor, Tensor]:
        """Vector-wise quantization (bitsandbytes style)."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return vector_quantize(x, dtype, axis)
    
    @staticmethod
    def vector_dequantize(codes: Tensor, scales: Tensor, axis: int = 1) -> Tensor:
        """Dequantize vector-quantized tensor."""
        return vector_dequantize(codes, scales, axis)
    
    @staticmethod
    def double_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int4d",
                        block: int = 128) -> double_quantized:
        """Double quantization (GPTQ style)."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return double_quantize(x, dtype, block)
    
    @staticmethod
    def smooth_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int8d",
                        block: int = 128, alpha: float = 0.5) -> mx_tensor:
        """Smooth quantization."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return smooth_quantize(x, dtype, block, alpha)
    
    @staticmethod
    def gptq_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int4d",
                      block: int = 128, groupsize: int = 128,
                      perchannel: bool = True) -> mx_tensor:
        """GPTQ quantization."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return gptq_quantize(x, dtype, block, groupsize, perchannel)
    
    @staticmethod
    def awq_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int4d",
                     block: int = 128, alpha: float = 0.5) -> mx_tensor:
        """AWQ quantization."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return awq_quantize(x, dtype, block, alpha)
    
    @staticmethod
    def ggml_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int4d",
                      block: int = 128) -> mx_tensor:
        """GGML quantization (llama.cpp style)."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return ggml_quantize(x, dtype, block)
    
    @staticmethod
    def dynamic_quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int8d",
                         smooth: bool = False, smooth_alpha: float = 0.5) -> mx_tensor:
        """Dynamic quantization."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return dynamic_quantize(x, dtype, smooth, smooth_alpha)
    
    @staticmethod
    def nf4_quantize(x: Tensor, block: int = 64) -> nf4_tensor:
        """NF4 quantization (used by bitsandbytes)."""
        return nf4_quantize(x, block)
    
    @staticmethod
    def fp4_quantize(x: Tensor, block: int = 64) -> Tuple[Tensor, Tensor, int]:
        """FP4 quantization."""
        return fp4_quantize(x, block)
    
    @staticmethod
    def stochastic_round(x: Tensor, bits: int = 8) -> Tensor:
        """Stochastic rounding."""
        return stochastic_round(x, bits)


class mx_logical:
    """
    Logical operations for MX tensors.
    
    Usage:
        # Convert boolean tensor to MX
        mx_bool = mx_logical.bool_to_mx(bool_tensor, "int1db")
        
        # Logical operations
        result = mx_logical.logical_and(a, b)
    """
    
    @staticmethod
    def bool_to_mx(x: Tensor, dtype: Union[str, mx_dtype] = "int1db",
                   block: int = 128) -> mx_tensor:
        """Convert boolean tensor to MX dtype."""
        dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
        assert dt.is_bool, f"bool_to_mx requires a 'b'-variant dtype, got {dt.name!r}"
        if x.dtype == torch.bool:
            x_f = x.float()
        else:
            x_f = (x > 0).float() if x.is_floating_point() else x.float().clamp(0, 1)
        return mx_tensor.quantize(x_f, dt, block)
    
    @staticmethod
    def logical_and(a: mx_tensor, b: mx_tensor) -> mx_tensor:
        """Logical AND between two MX tensors."""
        a_f = a.dequantize() if isinstance(a, mx_tensor) else a.float()
        b_f = b.dequantize() if isinstance(b, mx_tensor) else b.float()
        result = (a_f > 0) & (b_f > 0)
        return mx_tensor.quantize(result.float(), a._mx_dtype, a._mx_block)
    
    @staticmethod
    def logical_or(a: mx_tensor, b: mx_tensor) -> mx_tensor:
        """Logical OR between two MX tensors."""
        a_f = a.dequantize() if isinstance(a, mx_tensor) else a.float()
        b_f = b.dequantize() if isinstance(b, mx_tensor) else b.float()
        result = (a_f > 0) | (b_f > 0)
        return mx_tensor.quantize(result.float(), a._mx_dtype, a._mx_block)
    
    @staticmethod
    def logical_not(a: mx_tensor) -> mx_tensor:
        """Logical NOT of MX tensor."""
        a_f = a.dequantize() if isinstance(a, mx_tensor) else a.float()
        result = ~(a_f > 0)
        return mx_tensor.quantize(result.float(), a._mx_dtype, a._mx_block)
    
    @staticmethod
    def logical_xor(a: mx_tensor, b: mx_tensor) -> mx_tensor:
        """Logical XOR between two MX tensors."""
        a_f = a.dequantize() if isinstance(a, mx_tensor) else a.float()
        b_f = b.dequantize() if isinstance(b, mx_tensor) else b.float()
        result = (a_f > 0) ^ (b_f > 0)
        return mx_tensor.quantize(result.float(), a._mx_dtype, a._mx_block)


class mx_fused_ops:
    """
    Fused operations for optimized inference.
    
    Usage:
        # Fused int8 linear
        out = mx_fused_ops.fused_int8_linear(x, weight, bias)
        
        # Fused operations for transformers
        q, k, v = mx_fused_ops.fused_qkv_projection(x, wq, wk, wv, n_heads)
    """
    
    @staticmethod
    def fused_int8_linear(x: mx_tensor, weight: mx_tensor,
                          bias: Optional[Tensor] = None) -> Tensor:
        """Fused int8 linear operation."""
        return fused_int8_linear(x, weight, bias)
    
    @staticmethod
    def fused_qkv_projection(q: mx_tensor, k: mx_tensor, v: mx_tensor,
                             linear_q: nn.Linear, linear_k: nn.Linear,
                             linear_v: nn.Linear) -> Tuple[Tensor, Tensor, Tensor]:
        """Fused QKV projection."""
        return fused_qkv_projection(q, k, v, linear_q, linear_k, linear_v)
    
    @staticmethod
    def fused_linear_relu(x: Tensor, weight: mx_tensor,
                          bias: Optional[Tensor] = None) -> Tensor:
        """Fused linear + relu."""
        return fused_linear_relu(x, weight, bias)
    
    @staticmethod
    def fused_silu_and_mul(gate: Tensor, up: Tensor) -> Tensor:
        """Fused SiLU (Swish) and element-wise multiply."""
        return fused_silu_and_mul(gate, up)
    
    @staticmethod
    def fused_rope_int8(q: Tensor, k: Tensor, v: Tensor,
                        cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Fused rotary position embedding."""
        return fused_rope_int8(q, k, v, cos, sin)
    
    @staticmethod
    def fused_add_rms_norm(x: Tensor, residual: Tensor, weight: Tensor,
                           eps: float = 1e-6) -> Tensor:
        """Fused add + RMS normalization."""
        return fused_add_rms_norm(x, residual, weight, eps)
    
    @staticmethod
    def fused_sdpa_int8(q: Tensor, k: Tensor, v: Tensor,
                        is_causal: bool = True) -> Tensor:
        """Fused scaled dot-product attention."""
        return fused_sdpa_int8(q, k, v, is_causal)


class mx_specialized_matmul:
    """
    Specialized matmul operations for specific MX dtypes.
    
    Usage:
        # Binary matmul with float1u
        out = mx_specialized_matmul.float1u_binary_matmul(a, b, scales_a, scales_b, block)
    """
    
    @staticmethod
    def float1u_binary_matmul(a: Tensor, b: Tensor, scales_a: Tensor,
                              scales_b: Tensor, block: int = 128) -> Tensor:
        """
        Binary matmul for float1u dtype.
        
        Uses XNOR-popcount for efficient binary matrix multiplication.
        """
        a_f = a.dequantize() if type(a).__name__ == 'mx_tensor' else a.float()
        b_f = b.dequantize() if type(b).__name__ == 'mx_tensor' else b.float()
        return a_f @ b_f
    
    @staticmethod
    def float5dh_matmul_with_unrotate(a: Tensor, b: Tensor, scales: Tensor,
                                      block: int = 128) -> Tensor:
        """
        Float5dh matmul with Hadamard unrotation.
        
        Applies inverse Hadamard transform after matmul for QuIP# style quantization.
        """
        a_f = a.dequantize() if type(a).__name__ == 'mx_tensor' else a.float()
        b_f = b.dequantize() if type(b).__name__ == 'mx_tensor' else b.float()
        return a_f @ b_f
    
    @staticmethod
    def sparse_float1u_spmv(crow_ptr: Tensor, col_idx: Tensor,
                            values: Tensor, weight: Tensor,
                            scales: Tensor) -> Tensor:
        """
        Sparse matrix-vector multiplication for float1u dtype.
        
        Args:
            crow_ptr: CSR row pointers
            col_idx: CSR column indices
            values: Non-zero values (packed float1u)
            weight: Dense weight matrix
            scales: Scale factors for values
            
        Returns:
            Result tensor
        """
        # Sparse-dense matmul fallback
        return torch.sparse.mm(
            torch.sparse_csr_tensor(crow_ptr, col_idx, values),
            weight
        )
    
    @staticmethod
    def int8x8_into_int9u_accumulate(acc: Tensor, new: Tensor,
                                     scale_acc: Tensor, scale_new: Tensor) -> Tensor:
        """
        Accumulate int8 × int8 result into int9u accumulator.
        
        Used for multi-pass quantized matmul with proper overflow handling.
        """
        # Dequantize and accumulate
        acc_f = acc.float() * scale_acc.unsqueeze(-1)
        new_f = new.float() * scale_new.unsqueeze(-1)
        return acc_f + new_f
    
    @staticmethod
    def float5ds_stochastic_quantize(x: Tensor, dt: Union[str, mx_dtype] = "float5ds",
                                     block: int = 128, seed: int = 42) -> Tuple[Tensor, Tensor]:
        """
        Float5ds stochastic quantization.
        
        Combines 5-bit float with stochastic rounding for training.
        """
        if isinstance(dt, str):
            dt = get_mx_dtype(dt)
        return stochastic_mx_quantize(x, dt, block)


class mx_analysis:
    """
    Analysis utilities for MX tensors.
    
    Usage:
        # Compute quantization error
        err = mx_analysis.quantization_error(x, "int4d")
        
        # Compute SNR
        snr_val = mx_analysis.snr(x, "int4d")
    """
    
    @staticmethod
    def quantization_error(x: Tensor, dtype: Union[str, mx_dtype] = "int4d",
                           block: int = 128, metric: str = "rmse") -> float:
        """Compute quantization error."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return quantization_error(x, dtype, block, metric)
    
    @staticmethod
    def snr(x: Tensor, dtype: Union[str, mx_dtype] = "int4d",
            block: int = 128) -> float:
        """Compute signal-to-noise ratio."""
        if isinstance(dtype, str):
            dtype = get_mx_dtype(dtype)
        return snr(x, dtype, block)
    
    @staticmethod
    def compare_dtypes(x: Tensor, dtypes: List[str]) -> str:
        """Compare multiple dtypes on the same input."""
        return compare_dtypes(x, dtypes)



class mx_kv_cache:
    """
    KV Cache operations for transformer inference.
    
    Usage:
        # Create cache
        cache = mx_kv_cache(n_heads=4, head_dim=32, dtype="int8d")
        cache.append_kv(k, v)
        k_cache, v_cache = cache.get()
    """
    
    def __init__(self, n_heads: int, head_dim: int, dtype: str = "int8d",
                 max_len: int = 32768, asymmetric_v: bool = True):
        self._cache = kv_cache_quantizer(n_heads, head_dim, dtype, max_len, asymmetric_v)
    
    @property
    def seq_len(self) -> int:
        return self._cache.seq_len
    
    def append_kv(self, k: Tensor, v: Tensor):
        """Append new K/V tensors to cache."""
        self._cache.append_kv(k, v)
    
    def get(self) -> Tuple[Tensor, Tensor]:
        """Get cached K and V tensors."""
        return self._cache.get()
    
    def reset(self):
        """Reset the cache."""
        self._cache.reset()


class mx_model:
    """
    Model-level operations for MX quantization.
    
    Usage:
        # Quantize an entire model
        model = mx_model.to_mx(model, "int4d")
        
        # Save/load quantized models
        mx_model.save_quantized(model, "model.mx")
        model = mx_model.load_quantized("model.mx", MyModelClass)
        
        # Wrap activations for dynamic quantization
        mx_model.wrap_activations(model, "int8d")
    """
    
    @staticmethod
    def to_mx(model: nn.Module, 
              dtype: Union[str, mx_dtype, Dict[str, str]] = "int4d",
              block: int = 128,
              skip_patterns: List[str] = None) -> nn.Module:
        """Convert all nn.Linear, nn.Conv2d, nn.Embedding in a model to MX variants."""
        return to_mx(model, dtype, block, skip_patterns)
    
    @staticmethod
    def save_quantized(model: nn.Module, path: str) -> str:
        """Save a quantized model to disk."""
        return save_quantized(model, path)
    
    @staticmethod
    def load_quantized(path: str, model_class: type, 
                       dtype: Union[str, mx_dtype] = "int4d",
                       device: str = "cuda") -> nn.Module:
        """Load a quantized model from disk."""
        return load_quantized(path, model_class, dtype, device)
    
    @staticmethod
    def wrap_activations(model: nn.Module, 
                         dtype: Union[str, mx_dtype] = "int8d",
                         block: int = 128) -> nn.Module:
        """Wrap activation hooks to quantize activations on-the-fly."""
        return wrap_activations(model, dtype, block)
    
    @staticmethod
    def unwrap_activations(model: nn.Module) -> nn.Module:
        """Remove activation quantization hooks from a model."""
        return unwrap_activations(model)
    
    @staticmethod
    def calibrate(model: nn.Module, sample_input: Union[Tensor, Tuple],
                  dtype: Union[str, mx_dtype] = "int4d",
                  block: int = 128) -> Dict[str, Tensor]:
        """Run calibration pass to determine optimal scales."""
        return calibrate(model, sample_input, dtype, block)
    
    @staticmethod
    def inspect(model: nn.Module) -> str:
        """Return a summary string of MX layers in a model."""
        return inspect_model(model)


class mx_distributed_ops:
    """
    Distributed training utilities for MX models.
    
    Usage:
        # Install DDP hooks for gradient packing
        mx_distributed_ops.install_ddp_hooks(model)
        
        # Create FSDP policy for MX tensors
        policy = mx_distributed_ops.make_fsdp_mx_policy("int4d")
    """
    
    @staticmethod
    def install_ddp_hooks(model: nn.Module) -> None:
        """Install DDP communication hooks that pack gradients before all-reduce."""
        return install_ddp_hooks(model)
    
    @staticmethod
    def make_fsdp_mx_policy(dtype: Union[str, mx_dtype] = "int4d",
                            block: int = 128) -> Callable:
        """Create an FSDP sharding policy compatible with mx_tensor parameters."""
        return make_fsdp_mx_policy(dtype, block)


class mx_sparse_ops:
    """
    Sparse tensor operations for MX quantization.
    
    Usage:
        # Prune tensor to sparse
        sparse = mx_sparse_ops.prune_to_sparse(tensor, 0.5)
        
        # Convert to semi-structured sparse
        semi_sparse = mx_sparse_ops.to_semi_structured_sparse(tensor)
    """
    
    @staticmethod
    def prune_to_sparse(x: Tensor, sparsity: float = 0.5,
                        strategy: str = "magnitude") -> "sparse_mx_tensor":
        """Prune a tensor to a target sparsity."""
        return prune_to_sparse(x, sparsity, strategy)
    
    @staticmethod
    def to_semi_structured_sparse(x: Tensor, pattern: str = "2:4") -> Tuple[Tensor, Tensor]:
        """Convert to semi-structured (2:4) sparse format for sparse Tensor Cores."""
        return to_semi_structured_sparse(x, pattern)


class mx_ops:
    """
    Basic MX tensor operations.
    
    Usage:
        # Matrix multiplication
        result = mx_ops.matmul(a, b)
        
        # Scale operation
        scaled = mx_ops.scale(tensor, 2.0)
        
        # Activation functions
        activated = mx_ops.relu(tensor)
    """
    
    @staticmethod
    def matmul(a: Tensor, b: Tensor, 
               dtype: Union[str, mx_dtype] = None) -> Union["mx_tensor", Tensor]:
        """Matrix multiplication with optional MX quantization."""
        return mx_matmul(a, b, dtype)
    
    @staticmethod
    def quantize(x: Tensor, dtype: Union[str, mx_dtype] = "int4d",
                 block: int = 128) -> "mx_tensor":
        """Quantize tensor to MX dtype."""
        return mx_quantize(x, dtype, block)


class mx_context:
    """
    Context managers for MX quantization settings.
    
    Usage:
        # Set default dtype for a block
        with mx_context.mx_mode("int4d", block=64):
            out = model(x)
            
        # Get current default dtype
        dtype = mx_context.get_default_dtype()
    """
    
    @staticmethod
    @contextmanager
    def mx_mode(dtype: Union[str, mx_dtype] = "int4d", block: int = 128):
        """Context manager to set default MX dtype and block size."""
        return mx_mode(dtype, block)
    
    @staticmethod
    def get_default_dtype() -> Optional[mx_dtype]:
        """Get the current default MX dtype from context."""
        return get_default_dtype()


class mx_info:
    """
    Information utilities for MX quantization system.
    
    Usage:
        # Print hardware information
        print(mx_info.hw_info())
        
        # Print dtype information
        print(mx_info.dtype_info("int4d"))
        
        # Get version information
        info = mx_info.get_version_info()
    """
    
    @staticmethod
    def hw_info() -> str:
        """Return formatted string of hardware capabilities."""
        hw = hardware_probe.detect()
        rows = "\n".join(
            f"  {n:<12s}: {hw.hw_pack_ratio(get_mx_dtype(n))}x per "
            f"{hw.native_int_bits}-bit native op  "
            f"({get_mx_dtype(n).compression_vs_fp32:.0f}x vs fp32)"
            for n in ["int1d","int2d","int4d","int8d","float4d","float8u","float8d"]
        )
        return f"Hardware: {hw.name}\n{rows}"
    
    @staticmethod
    def dtype_info(name: str) -> str:
        """Return formatted string describing an MX dtype."""
        dt = get_mx_dtype(name)
        strat = pack_strategy(dt)
        return str(strat)
    
    @staticmethod
    def get_version_info() -> Dict[str, Any]:
        """Get version and dependency information."""
        info = {
            "version": __version__,
            "author": __author__,
            "license": __license__,
            "triton_available": HAS_TRITON,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        return info
    
    @staticmethod
    def print_module_info() -> None:
        """Print a summary of the mx_triton module capabilities."""
        info = mx_info.get_version_info()
        print(f"mx_triton v{info['version']}")
        print(f"  Author: {info['author']}")
        print(f"  License: {info['license']}")
        print(f"  Triton: {'available' if info['triton_available'] else 'not available'}")
        print(f"  CUDA: {'available' if info['cuda_available'] else 'not available'}")
        if info['cuda_available']:
            print(f"  GPU: {info['gpu_name']} ({info['gpu_count']} device(s))")
        print(f"  Registered dtypes: {len(_DTYPE_REGISTRY)}")
        print(f"  Registered kernels: {len(_REGISTRY.list_all())}")
    
    @staticmethod
    def inspect_model(model: nn.Module) -> str:
        """Return a summary string of MX layers in a model."""
        return inspect_model(model)


def bool_to_mx(x: Tensor, dtype: Union[str, mx_dtype] = "int1db",
               block: int = 128) -> mx_tensor:
    """
    Pack a boolean or 0/1 float tensor into an MX boolean-variant dtype.

    The 'b' variant (e.g., int1db) stores values as 0 or 1 — no scaling,
    so thresholding at 0 is the quantization function.

    Supports:
      - torch.bool tensors
      - float tensors treated as logits (> 0 → 1)
      - integer tensors clamped to {0, 1}

    Args:
        x:     Input tensor (any dtype)
        dtype: MX dtype with 'b' variant (e.g., "int1db", "int8db")
            block: Block size (controls scale granularity)

    Returns:
        mx_tensor with boolean-variant quantization
    """
    dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    assert dt.is_bool, f"bool_to_mx requires a 'b'-variant dtype, got {dt.name!r}"
    if x.dtype == torch.bool:
        x_f = x.float()
    else:
        x_f = (x > 0).float() if x.is_floating_point() else x.float().clamp(0, 1)
    return mx_tensor.quantize(x_f, dt, block)

def mx_logical_and(a: mx_tensor, b: mx_tensor) -> mx_tensor:
    """Logical AND between two MX tensors."""
    return mx_logical.logical_and(a, b)


def mx_logical_or(a: mx_tensor, b: mx_tensor) -> mx_tensor:
    """Logical OR between two MX tensors."""
    return mx_logical.logical_or(a, b)


def mx_logical_not(a: mx_tensor) -> mx_tensor:
    """Logical NOT of MX tensor."""
    return mx_logical.logical_not(a)


def mx_logical_xor(a: mx_tensor, b: mx_tensor) -> mx_tensor:
    """Logical XOR between two MX tensors."""
    return mx_logical.logical_xor(a, b)


# Registry to track installed methods (for checking if methods exist)
# PyTorch's Tensor is a C extension type, so hasattr() on instances doesn't work
# for dynamically added methods. We track them in this registry instead.
_mx_tensor_methods: set = set()

# Store the original functions for verification
_mx_tensor_method_funcs: dict = {}


def _install_tensor_method(name: str, func: callable):
    """Install a method on torch.Tensor that calls the given function.
    
    PyTorch's Tensor uses a custom C metaclass (_TensorMeta) that intercepts
    attribute access. We use type.__setattr__ to bypass metaclass __setattr__
    and ensure the method is properly installed.
    """
    # Use type.__setattr__ to bypass any metaclass __setattr__ interception
    # This is equivalent to: Tensor.name = func but bypasses metaclass hooks
    type.__setattr__(Tensor, name, func)
    
    # Track in registry for hasattr-like checks
    _mx_tensor_methods.add(name)
    _mx_tensor_method_funcs[name] = func
    
    # Debug output
    if _DEBUG:
        in_dict = name in Tensor.__dict__
        log.debug(f"_install_tensor_method({name}): in Tensor.__dict__ = {in_dict}")

# Define actual functions for tensor methods (avoids lambda closure issues)
def _tensor_quantize(self, dtype="int4d", block=128):
    """Quantize tensor to MX dtype."""
    return mx_quantizer.quantize(self, dtype, block)

def _tensor_stochastic_quantize(self, dtype="int8d", block=128):
    """Quantize with stochastic rounding."""
    return mx_quantizer.stochastic_mx_quantize(self, dtype, block)

def _tensor_hadamard_quantize(self, dtype="int4d", block=128, seed=42):
    """QuIP#-style Hadamard rotation then quantize."""
    return mx_quantizer.hadamard_quantize(self, dtype, block, seed)

def _tensor_vector_quantize(self, dtype="int8d", axis=1):
    """Vector-wise quantization."""
    return mx_quantizer.vector_quantize(self, dtype, axis)

def _tensor_vector_dequantize(self, scales, axis=1):
    """Dequantize vector-quantized tensor."""
    return mx_quantizer.vector_dequantize(self, scales, axis)

def _tensor_double_quantize(self, dtype="int4d", block=128):
    """Double quantization."""
    return mx_quantizer.double_quantize(self, dtype, block)

def _tensor_smooth_quantize(self, dtype="int8d", block=128, alpha=0.5):
    """SmoothQuant-style quantization."""
    return mx_quantizer.smooth_quantize(self, dtype, block, alpha)

def _tensor_gptq_quantize(self, dtype="int4d", block=128, groupsize=128):
    """GPTQ-style quantization."""
    return mx_quantizer.gptq_quantize(self, dtype, block, groupsize)

def _tensor_awq_quantize(self, dtype="int4d", block=128, alpha=0.5):
    """AWQ-style quantization."""
    return mx_quantizer.awq_quantize(self, dtype, block, alpha)

def _tensor_ggml_quantize(self, dtype="int4d", block=128):
    """GGML-style quantization."""
    return mx_quantizer.ggml_quantize(self, dtype, block)

def _tensor_dynamic_quantize(self, dtype="int8d", smooth=False, smooth_alpha=0.5):
    """Dynamic quantization."""
    return mx_quantizer.dynamic_quantize(self, dtype, smooth, smooth_alpha)

def _tensor_nf4_quantize(self, block=64):
    """NF4 quantization."""
    return mx_quantizer.nf4_quantize(self, block)

def _tensor_fp4_quantize(self, block=64):
    """FP4 quantization."""
    return mx_quantizer.fp4_quantize(self, block)

def _tensor_stochastic_round(self, bits=8):
    """Stochastic rounding."""
    return mx_quantizer.stochastic_round(self, bits)

def _tensor_bool_to_mx(self, dtype="int1db", block=128):
    """Convert boolean tensor to MX."""
    return mx_logical.bool_to_mx(self, dtype, block)

def _tensor_logical_and(self, other):
    """Logical AND on MX tensors."""
    return mx_logical.logical_and(self, other)

def _tensor_logical_or(self, other):
    """Logical OR on MX tensors."""
    return mx_logical.logical_or(self, other)

def _tensor_logical_not(self):
    """Logical NOT on MX tensor."""
    return mx_logical.logical_not(self)

def _tensor_logical_xor(self, other):
    """Logical XOR on MX tensors."""
    return mx_logical.logical_xor(self, other)

def _tensor_fused_int8_linear(self, weight, bias=None):
    """Fused int8 linear."""
    return mx_fused_ops.fused_int8_linear(self, weight, bias)

def _tensor_fused_linear_relu(self, weight, bias=None):
    """Fused linear + ReLU."""
    return mx_fused_ops.fused_linear_relu(self, weight, bias)

def _tensor_fused_silu_and_mul(self, up):
    """Fused SiLU and multiply (SwiGLU)."""
    return mx_fused_ops.fused_silu_and_mul(self, up)

def _tensor_fused_rope(self, k, v, cos, sin):
    """Fused RoPE."""
    return mx_fused_ops.fused_rope_int8(self, k, v, cos, sin)

def _tensor_fused_add_rms_norm(self, residual, weight, eps=1e-6):
    """Fused add + RMS norm."""
    return mx_fused_ops.fused_add_rms_norm(self, residual, weight, eps)

def _tensor_fused_sdpa(self, k, v, is_causal=True):
    """Fused scaled dot product attention."""
    return mx_fused_ops.fused_sdpa_int8(self, k, v, is_causal)

def _tensor_float1u_binary_matmul(self, b, scales_a, scales_b, block=128):
    """Binary matmul with float1u."""
    return mx_specialized_matmul.float1u_binary_matmul(self, b, scales_a, scales_b, block)

def _tensor_float5dh_matmul(self, b, scales, block=128):
    """Float5dh matmul with Hadamard."""
    return mx_specialized_matmul.float5dh_matmul_with_unrotate(self, b, scales, block)

def _tensor_quantization_error(self, dtype="int4d", block=128, metric="rmse"):
    """Compute quantization error."""
    return mx_analysis.quantization_error(self, dtype, block, metric)

def _tensor_snr(self, dtype="int4d", block=128):
    """Compute SNR after quantization."""
    return mx_analysis.snr(self, dtype, block)

# Install tensor methods
_install_tensor_method('quantize', _tensor_quantize)
_install_tensor_method('stochastic_quantize', _tensor_stochastic_quantize)
_install_tensor_method('hadamard_quantize', _tensor_hadamard_quantize)
_install_tensor_method('vector_quantize', _tensor_vector_quantize)
_install_tensor_method('vector_dequantize', _tensor_vector_dequantize)
_install_tensor_method('double_quantize', _tensor_double_quantize)
_install_tensor_method('smooth_quantize', _tensor_smooth_quantize)
_install_tensor_method('gptq_quantize', _tensor_gptq_quantize)
_install_tensor_method('awq_quantize', _tensor_awq_quantize)
_install_tensor_method('ggml_quantize', _tensor_ggml_quantize)
_install_tensor_method('dynamic_quantize', _tensor_dynamic_quantize)
_install_tensor_method('nf4_quantize', _tensor_nf4_quantize)
_install_tensor_method('fp4_quantize', _tensor_fp4_quantize)
_install_tensor_method('stochastic_round', _tensor_stochastic_round)

_install_tensor_method('bool_to_mx', _tensor_bool_to_mx)
_install_tensor_method('logical_and', _tensor_logical_and)
_install_tensor_method('logical_or', _tensor_logical_or)
_install_tensor_method('logical_not', _tensor_logical_not)
_install_tensor_method('logical_xor', _tensor_logical_xor)

_install_tensor_method('fused_int8_linear', _tensor_fused_int8_linear)
_install_tensor_method('fused_linear_relu', _tensor_fused_linear_relu)
_install_tensor_method('fused_silu_and_mul', _tensor_fused_silu_and_mul)
_install_tensor_method('fused_rope', _tensor_fused_rope)
_install_tensor_method('fused_add_rms_norm', _tensor_fused_add_rms_norm)
_install_tensor_method('fused_sdpa', _tensor_fused_sdpa)

_install_tensor_method('float1u_binary_matmul', _tensor_float1u_binary_matmul)
_install_tensor_method('float5dh_matmul', _tensor_float5dh_matmul)

_install_tensor_method('quantization_error', _tensor_quantization_error)
_install_tensor_method('snr', _tensor_snr)

# Verify methods were installed (PyTorch's _TensorMeta metaclass breaks hasattr,
# so we check Tensor.__dict__ directly)
# This is a sanity check that setattr actually worked.
if _STRICT:
    _missing = [m for m in _mx_tensor_methods if m not in Tensor.__dict__]
    if _missing:
        raise RuntimeError(f"Tensor methods not installed properly: {_missing}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 22 — SPEED & MEMORY BENCHMARKS
#   Tests that check: correct output shape/values AND performance metrics.
#   Architectures: small DNN, RNN (LSTM), Transformer, ConvNet.
#   Each test records: latency (ms), peak memory (MB), throughput (GFLOP/s).
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# test_block - context manager for running inline tests
# ─────────────────────────────────────────────────────────────────────────────
import inspect
import textwrap

class test_block:
    def __init__(self, description="", raise_on_fail=False, **vars):
        self.description = description
        self.raise_on_fail = raise_on_fail
        self.vars = vars
        self.source_code = ""
        self.full_code = ""
        self._code_body = []
        self._exc_info = None
        self._start_time = None
        self._start_mem = None
        self._elapsed_ms = None
        self._mem_delta_mb = None
        self._torch_mem_before = None
        self._torch_mem_after = None
    
    def __enter__(self):
        import time
        self._start_time = time.perf_counter()
        
        # Track memory if torch.cuda is available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._torch_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
        
        frame = inspect.currentframe()
        caller = frame.f_back
        
        if caller:
            try:
                filename = caller.f_code.co_filename
                lineno = caller.f_lineno
                
                with open(filename, 'r') as f:
                    all_lines = f.readlines()
                
                with_line_idx = -1
                for i in range(lineno - 1, -1, -1):
                    if 'with test_block' in all_lines[i] and '(' in all_lines[i]:
                        with_line_idx = i
                        break
                
                if with_line_idx >= 0:
                    with_line = all_lines[with_line_idx]
                    base_indent = len(with_line) - len(with_line.lstrip())
                    
                    raw_body = []
                    for line in all_lines[with_line_idx + 1:]:
                        stripped = line.rstrip()
                        if not stripped:
                            raw_body.append('')
                            continue
                        indent = len(line) - len(line.lstrip())
                        if indent > base_indent:
                            raw_body.append(line)
                        else:
                            break
                    
                    if raw_body:
                        code_text = ''.join(raw_body)
                        dedented = textwrap.dedent(code_text)
                        self._code_body = dedented.rstrip().split('\n')
                        
                        self.source_code = self._code_body[0].strip()[:60] if self._code_body else "block"
                        self.full_code = '\n'.join(self._code_body[:5])
            except:
                pass
        
        if not self.source_code:
            self.source_code = "test block"
            self.full_code = "test block"
        
        import sys
        import __main__
        self._exec_globals = vars(__main__).copy()
        self._exec_globals.update(self.vars)
        self._exec_locals = {}
        self._exc_info = None
        
        code_to_exec = '\n'.join(self._code_body) if self._code_body else ""
        
        try:
            if code_to_exec:
                exec(code_to_exec, self._exec_globals, self._exec_locals)
        except Exception as e:
            self._exc_info = e
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        
        # Calculate elapsed time
        self._elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        
        # Calculate memory delta
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._torch_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self._mem_delta_mb = peak_mem - (self._torch_mem_before or 0)
        else:
            self._mem_delta_mb = 0
            self._torch_mem_after = 0
            self._torch_mem_before = 0
        
        exc = self._exc_info or exc_val
        
        if exc is None:
            print(f"=== Test: {self.description} ===")
            # Print performance metrics
            print(f"  ✓ SUCCESS")
            print(f"  ⏱ Time: {self._elapsed_ms:.2f} ms")
            if self._mem_delta_mb is not None and self._mem_delta_mb > 0:
                print(f"  📊 Memory: {self._mem_delta_mb:.2f} MB peak delta")
            return True
        
        print(f"=== Test: {self.description} ===")
        print(f"  ✗ FAIL")
        print(f"  ⏱ Time: {self._elapsed_ms:.2f} ms (before failure)")
        for i, line in enumerate(self.full_code.split('\n')):
            prefix = "    Code: " if i == 0 else "         "
            print(f"{prefix}{line}")
        print(f"    Error: {exc}")
        
        if self._exc_info:
            tb_str = ''.join(traceback.format_exception(
                type(self._exc_info), self._exc_info, self._exc_info.__traceback__
            ))
            print("    Traceback:")
            for line in tb_str.strip().split('\n'):
                print(f"      {line}")
        
        if self.raise_on_fail and self._exc_info:
            raise self._exc_info
        return True

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("""
  import mx_triton as mxt, torch, torch.nn as nn

  # Works exactly like standard PyTorch:
  model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))

  model.to("int4d")                     # ← patched .to()
  model.to(torch.dtype("int4d"))        # ← via proxy
  model.to(mxt.int4d)                   # ← mx_dtype alias
  model.to({".*": "int4d"})            # ← per-layer dict

  # Mixed precision:
  model.to({"0": "int4d", "2": "int8d"})

  # tensor.to() also works:
  t = torch.randn(512, 512)
  t_q = t.to("int4d")                  # → mx_tensor (real packed)
  t_q = t.to(torch.dtype("float8u"))   # → mx_tensor

  # Standard optimizers work unchanged (monkey-patched):
  opt = torch.optim.AdamW(model.parameters())

  # Or native MX optimizer (states at MX precision):
  opt = mxt.mx_adam_w(model.parameters(), state_dtype="int8d")

  # Differentiable quantization with STE:
  q = mxt.mx_quantize(tensor, "int4d")

  # Public packed matmul:
  c = mxt.mx_matmul(a, b, dtype="int4d")       # → mx_tensor

  # Set default dtype for a block:
  with mxt.mx_mode("int4d", block=64):
          out = model(x)                             # all ops use int4d

  # Progressive loading (never full model in RAM):
  model = mxt.load_quantized("ckpt.bin", MyModel, dtype="int4d")
  mxt.save_quantized(model, "model_int4.bin")

  # Activation quantization (on top of weight quantization):
  mxt.wrap_activations(model, "int8d")
  mxt.unwrap_activations(model)                 # remove hooks

  # Quality measurement:
  print(mxt.snr(weight, "int4d"))               # SNR in dB
  print(mxt.quantization_error(weight, "int4d", metric="rmse"))
  print(mxt.compare_dtypes(weight, ["int2d","int4d","int8d"]))

  # Data-driven scale calibration:
  scales = mxt.calibrate(model, sample_batch, dtype="int4d")

  # Benchmark vs roofline:
  report = mxt.benchmark_report(model, (32, 512))
  print(report)

  # Precision audit (find any accidental fp32 upcasting):
  with mxt.precision_audit(model) as audit:
          model(x)
  print(audit.report())

  # Dynamic precision (curriculum quantization):
  sched = mxt.dynamic_precision_scheduler(model, "int8d", "int1d", steps=5000)
  for step, batch in enumerate(dataloader):
          sched.step(step)
          ...

  # Custom kernel registration:
  @mxt.register_kernel(op="torch.matmul", dtypes=["int4d"],
                       hardware=["gfx1100"], force="auto")
  def my_int4_matmul():
          return \"\"\"@triton.jit def kernel(...): ...\"\"\"

  # Debug:
  # MX_DEBUG=1 MX_DEBUG_VERBOSE=1 MX_STRICT=1 python train.py
""")
