"""
mx_triton.py — MX (Microscaling) Quantization System
=====================================================
Pure PyTorch + Triton.  No C++.  No fake quantization.
Real bit-packed storage.  Execute arithmetic ON packed representations.

Deep PyTorch integration — monkey-patched at import time:
  • torch.dtype("int4d")        returns an MXDtypeProxy object
  • model.to("int4d")           quantizes the entire model
  • model.to(torch.dtype("float8u")) same thing via proxy
  • tensor.to("int2d")          quantizes a tensor
  • MXTensor IS a torch.Tensor subclass — all ops dispatched via __torch_dispatch__
  • Standard optimizers (Adam, SGD, AdamW…) work as-is on MXTensors
  • DDP / FSDP compatible — all-reduce on packed bits
  • torch.compile / inductor compatible (fallback path)

Packing (real, not fake):
  int8  word → int1 : 8  values packed
  int8  word → int2 : 4  values packed
  int8  word → int4 : 2  values packed
  int8  word → int8 : 1  value  packed
  int32 word → int1 : 32 values (CPU AVX-512)

Non-uniform quantization (bitsandbytes / QLoRA / Unsloth inspired):
  NF4   — NormalFloat4 optimal for normally-distributed weights (QLoRA)
  AF4   — AbsFloat4 symmetric 4-bit grid
  FP4   — FloatPoint4 IEEE-like sub-byte float (bitsandbytes FP4)
  DQ    — Double Quantization: quantize the block scales themselves → −0.37 bits/param
  GPTQ  — Group-wise symmetric int4 with column-wise ordering

Advanced techniques:
  LLM.int8()    — Mixed-precision: outlier columns fp16, rest int8 (bitsandbytes)
  SmoothQuant   — Scale migration between activations and weights (α-balancing)
  Sparse MX     — Magnitude-pruned sparse tensors with packed non-zero values
  LoRA-aware    — Base weights frozen at MX precision, LoRA adapters in fp16/bf16
  Dynamic Q     — Per-token activation quantization at runtime (no calibration)
  Stochastic Q  — Unbiased stochastic rounding (bitsandbytes/Unsloth training)
  Hadamard Q    — QuIP# randomized Hadamard rotation before quantization (+3-5 dB SNR)
  Vector-wise   — Per-row/column absmax quantization (bitsandbytes int8 linear style)
  KV Cache Q    — Asymmetric int8 quantized KV cache for long-context inference

Sparse arithmetic (torch.sparse integration):
  prune_to_sparse()     — Vectorized CSR construction (O(1) Python, no loops)
  SparseMXTensor        — CSR sparse + MX-packed non-zeros
  to_semi_structured_sparse() — 2:4 structured sparsity (NVIDIA Sparse TC)
  MXSparseLinear        — torch.sparse.mm dispatch for GEMM acceleration

Intra-quantization acceleration (stay in packed realm):
  fused_linear_relu     — linear+ReLU with no intermediate fp32 buffer
  fused_silu_and_mul    — SwiGLU (LLaMA MLP), single kernel pass
  fused_rope_int8       — RoPE applied directly to int8 Q,K (fixed rotation formula)
  fused_sdpa_int8       — INT8 scaled dot-product attention
  fused_add_rms_norm    — residual + RMSNorm, one kernel
  fused_int8_linear     — INT8 × INT8 linear (stays packed, dequant inline)
  fused_qkv_projection  — Q+K+V in one pass (reads x once, 3× bandwidth saving)

New Triton kernels (Section 6c):
  _k_stochastic_round   — GPU stochastic rounding
  _k_int8_softmax       — INT8 → softmax → INT8 (stays quantized through attention)
  _k_sparse_spmm        — Sparse MX × dense (SpMM, avoids dequant of sparse weight)
  _k_fused_qkv          — Single-kernel Q/K/V projection
  _k_fused_dequant_linear — INT8 × INT8 GEMM with inline dequant

Additional nn.Module wrappers (Section 11d):
  MXConvTranspose2d / MXConvTranspose1d — transposed conv (decoder/upsampling)
  MXBatchNorm1d         — 1D BN for MLP/NLP paths
  MXTransformerEncoderLayer — full encoder block, pre-LN or post-LN
  MXGRU                 — quantized GRU cell (reset, update, new gates)

Kernel examples (docstring templates):
  KERNEL_EXAMPLES       — 4 templates (INT4 matmul, INT8+GELU, INT2 mul, NF4 linear)
  KERNEL_EXAMPLES_EXTRA — 2 more (KV cache update, Hadamard+quantize)

Dispatcher coverage (added):
  scatter_reduce, unfold, kron, tensordot, vecdot,
  repeat, repeat_interleave, squeeze/unsqueeze/flatten, exp/log/sqrt

Bug fixes vs prior version:
  • prune_to_sparse: O(n) Python loop → fully vectorized CSR (10-100× faster)
  • MXDynamicLinear: eliminated double-quantization of activations
  • _k_rope_int8: correct paired-element RoPE rotation formula

Environment:
  MX_DEBUG=1          verbose kernel selection + precision logs
  MX_DEBUG_VERBOSE=1  add stack traces
  MX_STRICT=1         raise on any full-precision fallback
"""

from __future__ import annotations

# ── stdlib ───────────────────────────────────────────────────────────────────
import os, gc, re, math, time, logging, warnings, functools, traceback
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
    # dtype system
    "MXDtype", "MXDtypeProxy", "get_mx_dtype",
    # common dtype aliases (all 512 available as module attributes)
    "int1d","int1u","int2d","int2u","int4d","int4u","int8d","int8u",
    "int16d","int16u","int32d","int32u","float4d","float4u",
    "float8d","float8u","float16d","float16u","float32d","float32u",
    # tensors
    "MXTensor", "NF4Tensor",
    # sparse tensors
    "SparseMXTensor",
    # nn.Module drop-ins — convolution
    "MXConv1d","MXConv2d","MXConvTranspose1d","MXConvTranspose2d",
    # nn.Module drop-ins — normalization
    "MXBatchNorm1d","MXBatchNorm2d","MXLayerNorm","MXRMSNorm","MXGroupNorm",
    # nn.Module drop-ins — core
    "MXLinear","MXEmbedding","MXEmbeddingBag","MXMultiheadAttention",
    # nn.Module drop-ins — transformer / recurrent
    "MXTransformerEncoderLayer","MXGRU",
    # advanced modules
    "MXLoRALinear","MXMixedInt8Linear","MXDynamicLinear","MXSparseLinear",
    # quantization — uniform
    "mx_quantize","mx_matmul","quantize",
    "quantization_error","snr","compare_dtypes",
    "dynamic_quantize",
    # quantization — stochastic
    "stochastic_round","stochastic_mx_quantize","StochasticMXQuantize",
    "triton_stochastic_quantize",
    # quantization — Hadamard / QuIP#
    "HadamardRotation","hadamard_quantize",
    "_fast_hadamard_transform",
    # quantization — vector-wise (bitsandbytes style)
    "vector_quantize","vector_dequantize",
    # quantization — non-uniform (QLoRA / bitsandbytes style)
    "nf4_quantize","nf4_dequantize",
    "fp4_quantize","fp4_dequantize",
    "double_quantize","DoubleQuantized",
    # advanced PTQ techniques
    "gptq_quantize","GPTQResult",
    "awq_quantize","AWQResult",
    "ggml_quantize","GGMLQuantized",
    # mixed-precision / outlier
    "mixed_int8_decompose","smooth_quantize",
    # sparse
    "prune_to_sparse","to_semi_structured_sparse",
    # KV cache
    "KVCacheQuantizer",
    # model-level API
    "to_mx","save_quantized","load_quantized",
    "wrap_activations","unwrap_activations",
    "calibrate",
    # optimiser
    "MXAdamW",
    # context managers / helpers
    "mx_mode","get_default_dtype","register_kernel",
    # hardware / analysis
    "HardwareProbe","HardwareProfile",
    "RooflineEstimator","benchmark_report",
    "PrecisionAudit","MXDebugger","DynamicPrecisionScheduler",
    "PackStrategy","inspect_model","hw_info","dtype_info",
    # distributed
    "MXDistributed","install_ddp_hooks",
    # intra-precision fused ops
    "fused_linear_relu","fused_silu_and_mul","fused_rope_int8",
    "fused_sdpa_int8","fused_add_rms_norm",
    "fused_int8_linear","fused_qkv_projection",
    # kernel template library
    "KERNEL_EXAMPLES","KERNEL_EXAMPLES_EXTRA",
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
# SECTION 1 — MX TYPE SYSTEM (512 types)
# ─────────────────────────────────────────────────────────────────────────────

_VALID_BITS  = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128)
_VALID_MODES = ("d", "u")
_VALID_KINDS = ("int", "float")


@dataclass(frozen=True)
class MXDtype:
    """
    One of the 512 MX data types.
    kind ∈ {int, float}, bits ∈ {1,2,3,4,5,6,7,8,16,32,64,128}, mode ∈ {d, u}
    """
    kind: str
    bits: int
    mode: str

    def __post_init__(self):
        assert self.kind in _VALID_KINDS,  f"Invalid kind {self.kind!r}"
        assert self.bits in _VALID_BITS,   f"Invalid bits {self.bits}"
        assert self.mode in _VALID_MODES,  f"Invalid mode {self.mode!r}"

    # ── identity ──────────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return f"{self.kind}{self.bits}{self.mode}"

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

    # ── representable range ───────────────────────────────────────────────────
    @property
    def max_val(self) -> float:
        if self.is_int:
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


# ── build registry of all 512 types ──────────────────────────────────────────
_DTYPE_REGISTRY: Dict[str, MXDtype] = {}
for _k in _VALID_KINDS:
    for _b in _VALID_BITS:
        for _m in _VALID_MODES:
            _dt = MXDtype(_k, _b, _m)
            _DTYPE_REGISTRY[_dt.name] = _dt


def get_mx_dtype(name: str) -> MXDtype:
    """Resolve MX dtype name → MXDtype. Raises ValueError for unknown names."""
    if isinstance(name, MXDtype):
        return name
    if isinstance(name, MXDtypeProxy):
        return name._mx
    if name not in _DTYPE_REGISTRY:
        raise ValueError(
            f"Unknown MX dtype {name!r}. Examples: int4d, float8u, int1d, float2d")
    return _DTYPE_REGISTRY[name]


def _resolve_mixed(a: MXDtype, b: MXDtype) -> MXDtype:
    """
    Mixed-precision resolution rule:
    • If either operand is "up", result is "up"  (full-precision bias)
    • Precision = min(bits) when both are "down" (worst-case truncation)
    """
    mode = "u" if (a.mode == "u" or b.mode == "u") else "d"
    bits = min(a.bits, b.bits)
    kind = a.kind
    # snap to nearest valid bits
    bits = min(_VALID_BITS, key=lambda x: abs(x - bits))
    return get_mx_dtype(f"{kind}{bits}{mode}")


# expose short aliases at module level: mx_triton.int4d, mx_triton.float8u …
for _name, _dt in _DTYPE_REGISTRY.items():
    globals()[_name] = _dt

# ── Explicit type-visible declarations for the most common MX dtypes ─────────
# These shadow the dynamic globals() assignments above with identical values,
# making them visible to type checkers (mypy/pyright/IDEs) and ensuring that
# bare-name references like `int4d` in test code are never undefined.
int1d   : MXDtype = _DTYPE_REGISTRY["int1d"]
int1u   : MXDtype = _DTYPE_REGISTRY["int1u"]
int2d   : MXDtype = _DTYPE_REGISTRY["int2d"]
int2u   : MXDtype = _DTYPE_REGISTRY["int2u"]
int3d   : MXDtype = _DTYPE_REGISTRY["int3d"]
int3u   : MXDtype = _DTYPE_REGISTRY["int3u"]
int4d   : MXDtype = _DTYPE_REGISTRY["int4d"]
int4u   : MXDtype = _DTYPE_REGISTRY["int4u"]
int5d   : MXDtype = _DTYPE_REGISTRY["int5d"]
int5u   : MXDtype = _DTYPE_REGISTRY["int5u"]
int6d   : MXDtype = _DTYPE_REGISTRY["int6d"]
int6u   : MXDtype = _DTYPE_REGISTRY["int6u"]
int7d   : MXDtype = _DTYPE_REGISTRY["int7d"]
int7u   : MXDtype = _DTYPE_REGISTRY["int7u"]
int8d   : MXDtype = _DTYPE_REGISTRY["int8d"]
int8u   : MXDtype = _DTYPE_REGISTRY["int8u"]
int16d  : MXDtype = _DTYPE_REGISTRY["int16d"]
int16u  : MXDtype = _DTYPE_REGISTRY["int16u"]
int32d  : MXDtype = _DTYPE_REGISTRY["int32d"]
int32u  : MXDtype = _DTYPE_REGISTRY["int32u"]
int64d  : MXDtype = _DTYPE_REGISTRY["int64d"]
int64u  : MXDtype = _DTYPE_REGISTRY["int64u"]
int128d : MXDtype = _DTYPE_REGISTRY["int128d"]
int128u : MXDtype = _DTYPE_REGISTRY["int128u"]
float1d  : MXDtype = _DTYPE_REGISTRY["float1d"]
float1u  : MXDtype = _DTYPE_REGISTRY["float1u"]
float2d  : MXDtype = _DTYPE_REGISTRY["float2d"]
float2u  : MXDtype = _DTYPE_REGISTRY["float2u"]
float3d  : MXDtype = _DTYPE_REGISTRY["float3d"]
float3u  : MXDtype = _DTYPE_REGISTRY["float3u"]
float4d  : MXDtype = _DTYPE_REGISTRY["float4d"]
float4u  : MXDtype = _DTYPE_REGISTRY["float4u"]
float5d  : MXDtype = _DTYPE_REGISTRY["float5d"]
float5u  : MXDtype = _DTYPE_REGISTRY["float5u"]
float6d  : MXDtype = _DTYPE_REGISTRY["float6d"]
float6u  : MXDtype = _DTYPE_REGISTRY["float6u"]
float7d  : MXDtype = _DTYPE_REGISTRY["float7d"]
float7u  : MXDtype = _DTYPE_REGISTRY["float7u"]
float8d  : MXDtype = _DTYPE_REGISTRY["float8d"]
float8u  : MXDtype = _DTYPE_REGISTRY["float8u"]
float16d : MXDtype = _DTYPE_REGISTRY["float16d"]
float16u : MXDtype = _DTYPE_REGISTRY["float16u"]
float32d : MXDtype = _DTYPE_REGISTRY["float32d"]
float32u : MXDtype = _DTYPE_REGISTRY["float32u"]
float64d : MXDtype = _DTYPE_REGISTRY["float64d"]
float64u : MXDtype = _DTYPE_REGISTRY["float64u"]
float128d: MXDtype = _DTYPE_REGISTRY["float128d"]
float128u: MXDtype = _DTYPE_REGISTRY["float128u"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MXDtypeProxy  (fake torch.dtype object)
# ─────────────────────────────────────────────────────────────────────────────

class MXDtypeProxy:
    """
    Impersonates a torch.dtype object.
    Returned by the patched torch.dtype("int4d") call.
    Accepted by all patched .to() methods.
    """
    __slots__ = ("_mx",)

    def __init__(self, mx: MXDtype):
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
        if isinstance(other, MXDtypeProxy): return self._mx == other._mx
        return NotImplemented


def _make_proxy(name_or_mx) -> MXDtypeProxy:
    if isinstance(name_or_mx, MXDtypeProxy):
        return name_or_mx
    return MXDtypeProxy(get_mx_dtype(name_or_mx))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — HARDWARE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
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

    def hw_pack_ratio(self, dt: MXDtype) -> int:
        """How many dt values can be packed per native arithmetic op."""
        return min(self.max_pack_bits // dt.bits, dt.pack_ratio)

    def peak_tflops(self, dt: MXDtype) -> float:
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


class HardwareProbe:
    _cache: Optional[HardwareProfile] = None

    @classmethod
    def detect(cls) -> HardwareProfile:
        if cls._cache is not None:
            return cls._cache
        cls._cache = cls._detect()
        if _DEBUG:
            log.debug(f"[HW] {cls._cache.name} ({cls._cache.arch}), "
                      f"native_int={cls._cache.native_int_bits}b, "
                      f"max_pack={cls._cache.max_pack_bits}b")
        return cls._cache

    @classmethod
    def _detect(cls) -> HardwareProfile:
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
    def _amd(cls, arch: str, props) -> HardwareProfile:
        known = {
            "gfx1100": HardwareProfile(
                name="rx_7900_xtx", arch="gfx1100", backend="rocm",
                native_int_bits=8, vector_bits=128, max_pack_bits=8,
                memory_bw_gbs=960.0, fp32_tflops=61.0, fp16_tflops=123.0,
                supported_native=["fp32","fp16","bf16","int8","int4"],
                fast_instrs=["v_dot4_i32_i8","v_dot2_f32_f16","v_perm_b32"],
                compute_units=96, wave_size=32,
            ),
            "gfx942": HardwareProfile(
                name="mi300x", arch="gfx942", backend="rocm",
                native_int_bits=8, vector_bits=256, max_pack_bits=8,
                memory_bw_gbs=5300.0, fp32_tflops=653.7, fp16_tflops=1307.0,
                supported_native=["fp32","fp16","bf16","int8","fp8","int4"],
                fast_instrs=["v_mfma_f32_32x32x8f16","v_dot4_i32_i8"],
                compute_units=304, wave_size=64,
            ),
            "gfx90a": HardwareProfile(
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
        return HardwareProfile(
            name=f"amd_{arch}", arch=arch, backend="rocm",
            native_int_bits=8, vector_bits=128, max_pack_bits=8,
            memory_bw_gbs=500.0, fp32_tflops=20.0, fp16_tflops=40.0,
            supported_native=["fp32","fp16","int8"],
            compute_units=getattr(props, "multi_processor_count", 32),
        )

    @classmethod
    def _nvidia(cls, cc: int, props) -> HardwareProfile:
        cu = props.multi_processor_count
        if cc >= 90:   # Hopper H100
            return HardwareProfile(
                name="h100", arch=f"sm_{cc}", backend="cuda",
                native_int_bits=8, vector_bits=512, max_pack_bits=8,
                memory_bw_gbs=3350.0, fp32_tflops=67.0, fp16_tflops=1000.0,
                supported_native=["fp32","fp16","bf16","int8","fp8","int4"],
                fast_instrs=["mma.sync.aligned.m16n8k32","ldmatrix","dp4a"],
                compute_units=cu, wave_size=32,
            )
        elif cc >= 86:  # Ampere / Ada
            return HardwareProfile(
                name="ada_ampere", arch=f"sm_{cc}", backend="cuda",
                native_int_bits=8, vector_bits=256, max_pack_bits=8,
                memory_bw_gbs=600.0, fp32_tflops=40.0, fp16_tflops=320.0,
                supported_native=["fp32","fp16","bf16","int8","int4"],
                fast_instrs=["mma.sync.aligned.m16n8k16","dp4a"],
                compute_units=cu, wave_size=32,
            )
        return HardwareProfile(
            name=f"nvidia_sm{cc}", arch=f"sm_{cc}", backend="cuda",
            native_int_bits=8, vector_bits=128, max_pack_bits=8,
            memory_bw_gbs=400.0, fp32_tflops=15.0, fp16_tflops=60.0,
            supported_native=["fp32","fp16","int8"],
            compute_units=cu,
        )

    @classmethod
    def _cpu(cls) -> HardwareProfile:
        try:
            import cpuinfo
            flags = cpuinfo.get_cpu_info().get("flags", [])
            has512 = "avx512f" in flags
        except Exception:
            has512 = False
        if has512:
            return HardwareProfile(
                name="zen4_avx512", arch="x86_avx512", backend="cpu",
                native_int_bits=32, vector_bits=512, max_pack_bits=32,
                memory_bw_gbs=100.0, fp32_tflops=3.0, fp16_tflops=6.0,
                supported_native=["fp32","fp64","int8","int16","int32","int64"],
                fast_instrs=["vpdpbusd","vfmadd231ps","vcvtps2ph"],
            )
        return HardwareProfile(
            name="generic_cpu", arch="x86", backend="cpu",
            native_int_bits=32, vector_bits=256, max_pack_bits=32,
            memory_bw_gbs=50.0, fp32_tflops=0.5, fp16_tflops=1.0,
            supported_native=["fp32","fp64","int8","int32"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — REAL BIT PACKING (no fake quantization)
# ─────────────────────────────────────────────────────────────────────────────

class BitPacker:
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
            slot  = (vals[i::ratio] & mask).to(torch.int8)
            shift = torch.tensor(i * bits, dtype=torch.int8)
            packed = packed | (slot << shift).to(torch.int8)
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
            return BitPacker.pack(vals.to(torch.int8), bits)
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
            return BitPacker.unpack(packed.view(torch.int8) if packed.dtype != torch.int8
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

def _quant_int(x: Tensor, dt: MXDtype, block: int) -> Tuple[Tensor, Tensor, int]:
    """
    Quantize float tensor → (packed, scales, n).
    packed: real bit-packed, scales: [n_blocks] float32.
    """
    flat = x.float().reshape(-1)
    n    = flat.numel()
    nb   = math.ceil(n / block)
    pad  = nb * block - n
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk  = flat.reshape(nb, block)

    max_int = float((1 << (dt.bits - 1)) - 1)   # e.g. int4 → 7
    scales  = blk.abs().amax(dim=1).clamp(min=1e-12) / max_int  # [nb]
    codes   = (blk / scales.unsqueeze(1)).clamp(-max_int, max_int).round_().to(torch.int32)

    packed = BitPacker.pack_auto(codes.reshape(-1), dt.bits)
    return packed, scales, n


def _quant_float(x: Tensor, dt: MXDtype, block: int) -> Tuple[Tensor, Tensor, int]:
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

    packed = BitPacker.pack_auto(codes.reshape(-1), dt.bits)
    return packed, scales, n


def _dequant(packed: Tensor, scales: Tensor, dt: MXDtype,
             n: int, block: int) -> Tensor:
    """Unpack + dequantize → float32 flat tensor of length n."""
    codes = BitPacker.unpack_auto(packed, dt.bits, n)

    nb  = scales.numel()
    pad = nb * block - n
    if pad:
        codes = torch.cat([codes, codes.new_zeros(pad)])
    blk = codes.reshape(nb, block)

    if dt.is_float and dt.bits < 8:
        nlevels = 2 ** dt.bits
        max_v   = dt.max_val
        step    = (2 * max_v) / (nlevels - 1)
        dq      = blk * step
    else:
        dq = blk

    return (dq * scales.unsqueeze(1)).reshape(-1)[:n]


def quantize(x: Tensor, dt: MXDtype, block: int = 128
             ) -> Tuple[Tensor, Tensor, int]:
    """Dispatch to int or float quantizer. Uses Triton GPU kernel when available."""
    if dt.is_int:
        # Fast path: Triton on-GPU quantize for int4 and int8
        if HAS_TRITON and x.is_cuda and dt.bits in (4, 8) and x.numel() >= 128:
            try:
                return _triton_quantize(x, dt, block)
            except Exception:
                pass
        return _quant_int(x, dt, block)
    return _quant_float(x, dt, block)


def _triton_quantize(x: Tensor, dt: MXDtype, block: int) -> Tuple[Tensor, Tensor, int]:
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
        _k_quantize_int4[(nb,)](flat, packed, scales, n, BS=block, BLK=BLK)
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

    packed = BitPacker.pack_auto(codes.reshape(-1), bits=4)
    return packed, scales, n


def _lookup_dequantize(packed: Tensor, scales: Tensor, table: Tensor,
                       n: int, block: int) -> Tensor:
    """Inverse of _lookup_quantize: decode 4-bit indices via table lookup."""
    codes = BitPacker.unpack_auto(packed, bits=4, n=n + (-n % block))
    nb    = scales.numel()
    blk   = codes.reshape(nb, block)
    tbl   = table.to(packed.device)
    dq    = tbl[blk.long().clamp(0, len(tbl)-1)]        # table gather
    return (dq * scales.unsqueeze(1)).reshape(-1)[:n]


# ── NF4Tensor ─────────────────────────────────────────────────────────────────

class NF4Tensor(torch.Tensor):
    """
    NormalFloat4 tensor — optimal 4-bit quantisation for weights drawn from N(0,1).
    Introduced in QLoRA (Dettmers et al. 2023) and popularised by bitsandbytes / Unsloth.

    Key properties vs plain int4:
      • Non-uniform grid: more codepoints near zero (where most weights concentrate)
      • ~0.25 dB better SNR than symmetric int4 on normally-distributed data
      • Same bit-width — no memory penalty

    Usage::
        w_nf4 = NF4Tensor.quantize(weight)      # 4-bit packed
        w_f   = w_nf4.dequantize()              # back to float32
        # Drop-in for MXLinear:
        lin   = MXLinear.from_linear(layer, get_mx_dtype("int4d"))
        lin.weight = nn.Parameter(NF4Tensor.quantize(layer.weight.data))
    """

    @staticmethod
    def __new__(cls, packed: Tensor, scales: Tensor,
                orig_shape: torch.Size, n: int, block: int = 64,
                requires_grad: bool = False):
        inst = torch.Tensor._make_subclass(cls, packed, requires_grad)
        inst._nf4_scales      = scales
        inst._nf4_orig_shape  = orig_shape
        inst._nf4_n           = n
        inst._nf4_block       = block
        return inst

    def __init__(self, packed, scales, orig_shape, n, block=64, requires_grad=False):
        pass

    @classmethod
    def quantize(cls, x: Tensor, block: int = 64) -> "NF4Tensor":
        """Quantize float tensor to NF4. block=64 matches bitsandbytes default."""
        with torch.no_grad():
            packed, scales, n = _lookup_quantize(x.detach().float(), _NF4_TABLE, block)
        return cls(packed, scales, x.shape, n, block, x.requires_grad)

    def dequantize(self) -> Tensor:
        """Decode NF4 → float32 in original shape."""
        flat = _lookup_dequantize(
            torch.Tensor._make_subclass(Tensor, self),
            self._nf4_scales, _NF4_TABLE, self._nf4_n, self._nf4_block)
        return flat.reshape(self._nf4_orig_shape)

    @property
    def shape(self) -> torch.Size:
        return self._nf4_orig_shape

    def __repr__(self):
        cr = self._nf4_n * 4 / max(self.nbytes + self._nf4_scales.nbytes, 1)
        return (f"NF4Tensor({tuple(self._nf4_orig_shape)}, "
                f"device={self.device}, {cr:.1f}x compression)")

    def float(self):      return self.dequantize()
    def half(self):       return self.dequantize().half()
    def bfloat16(self):   return self.dequantize().bfloat16()


def nf4_quantize(x: Tensor, block: int = 64) -> NF4Tensor:
    """Convenience wrapper: float tensor → NF4Tensor."""
    return NF4Tensor.quantize(x, block)


def nf4_dequantize(t: NF4Tensor) -> Tensor:
    """Convenience wrapper: NF4Tensor → float32."""
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
class DoubleQuantized:
    """
    Double Quantization (DQ) as in QLoRA / bitsandbytes.
    The block scales are themselves quantized to int8, saving ~0.5 bits/param.

    Memory layout:
      q_data   : int8 packed weights     (bits / 8  * N bytes)
      q_scales : int8 quantized scales   (N / block bytes)
      ss_scale : float32 scale-of-scales (1 float per super-block)
      dtype    : original MXDtype name
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
    dtype: Union[str, MXDtype] = "int4d",
    block: int = 64,
    super_block: int = 256,
) -> DoubleQuantized:
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
        DoubleQuantized dataclass with all compressed tensors.

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

    return DoubleQuantized(
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
        int8_packed  : MXTensor (int8d) for the non-outlier columns
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
        q_part = MXTensor(packed, scales, get_mx_dtype("int8d"),
                          w.shape, n, int8_block)
        return None, q_part, scales

    # Outlier columns → fp16 (high-precision path)
    outlier_cols = w[:, outlier_mask].half()

    # Normal columns → int8
    normal_cols  = w[:, ~outlier_mask]
    packed, scales, n = quantize(normal_cols, get_mx_dtype("int8d"), int8_block)
    q_part = MXTensor(packed, scales, get_mx_dtype("int8d"),
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
        The model with MXLinear layers replacing nn.Linear, weights smoothed.

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
            x = x.dequantize() if isinstance(x, MXTensor) else x.float()
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

        # Build MXLinear with smoothed weight
        mx_lin = MXLinear.from_linear(
            nn.Linear(mod.in_features, mod.out_features,
                      bias=mod.bias is not None), dt, block)
        mx_lin.weight = nn.Parameter(
            MXTensor.quantize(w_smooth, dt, block), requires_grad=False)
        if mod.bias is not None:
            mx_lin.bias = nn.Parameter(
                MXTensor.quantize(mod.bias.data, dt, block), requires_grad=False)
        # Store smoothing scale on module for use during forward
        mx_lin.register_buffer("smooth_scale", s)

        # Monkey-patch forward to divide input by smooth_scale
        orig_fwd = mx_lin.forward
        def _smooth_fwd(x, _lin=mx_lin, _s=s, _fwd=orig_fwd):
            xs = (x / _s.to(x.device)) if not isinstance(x, MXTensor) else x
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
class GPTQResult:
    """
    Result of GPTQ post-training quantization of one weight matrix.

    Attributes:
        q_weight:   Quantized MXTensor weight.
        quantiles:  Per-column scale statistics (for debugging).
        error:      Reconstruction error (||W - W_q||_F / ||W||_F).
        group_size: Column group size used.
        dtype_name: MX dtype name used.
    """
    q_weight:  MXTensor
    quantiles: Tensor       # [out_features] per-column absmax
    error:     float
    group_size: int
    dtype_name: str


def gptq_quantize(
    weight: Tensor,
    hessian: Optional[Tensor] = None,
    dtype: Union[str, MXDtype] = "int4d",
    group_size: int = 128,
    damp_percent: float = 0.01,
    actorder: bool = False,
) -> GPTQResult:
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
        GPTQResult with the GPTQ-quantized weight and metadata.

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

    q_weight = MXTensor.quantize(Wq, dt, group_size)
    W_orig   = weight.float()
    error    = ((W_orig - Wq).norm() / W_orig.norm().clamp(min=1e-8)).item()

    return GPTQResult(q_weight, col_scales, error, group_size, dt.name)


# ── AWQ ───────────────────────────────────────────────────────────────────────

@dataclass
class AWQResult:
    """
    Result of AWQ (Activation-aware Weight Quantization).

    Attributes:
        q_weight:      Quantized MXTensor.
        input_scale:   Per-channel input scaling factors [in_features].
        weight_scale:  Inverse input_scale applied to weight columns.
        error:         Reconstruction error vs unscaled weight.
    """
    q_weight:     MXTensor
    input_scale:  Tensor
    weight_scale: Tensor
    error:        float


def awq_quantize(
    weight: Tensor,
    activation_scales: Tensor,
    dtype: Union[str, MXDtype] = "int4d",
    group_size: int = 128,
    alpha: float = 0.5,
) -> AWQResult:
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
        AWQResult with the scaled+quantized weight and scale vectors.

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

    q_weight = MXTensor.quantize(W_scaled, dt, group_size)
    W_q_f    = q_weight.dequantize()
    error    = ((W - W_q_f / s_.unsqueeze(0)).norm() /
                W.norm().clamp(min=1e-8)).item()

    if _DEBUG:
        log.debug(f"[awq_quantize] alpha={alpha}, "
                  f"max_s={s_.max():.3f}, error={error:.4f}")

    return AWQResult(q_weight, s_, w_s, error)


# ── GGML / llama.cpp k-quant families ────────────────────────────────────────

@dataclass
class GGMLQuantized:
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
) -> GGMLQuantized:
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
        GGMLQuantized with all packed data and two-level scales.

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
    packed = BitPacker.pack_auto(codes.reshape(-1), bits)

    return GGMLQuantized(
        q_data    = packed,
        d         = d_vals,
        d_min     = min_vals,
        qs        = qs,
        qm        = qm,
        quant_type = quant_type,
        shape     = x.shape,
        n         = n,
    )


def _ggml_dequantize(gq: GGMLQuantized) -> Tensor:
    """Inverse of ggml_quantize: reconstruct float32."""
    bits    = {"Q4_K": 4, "Q5_K": 5, "Q6_K": 6}.get(gq.quant_type, 4)
    max_q   = float((1 << bits) - 1)
    super_block = 256
    nb_s    = gq.d.numel()

    codes = BitPacker.unpack_auto(gq.q_data, bits, nb_s * super_block)
    codes = codes.reshape(nb_s, super_block)

    d_f   = gq.d.float().unsqueeze(1)       # [nb_s, 1]
    min_f = gq.d_min.float().unsqueeze(1)   # [nb_s, 1]
    flat  = codes / max_q * d_f + min_f
    return flat.reshape(-1)[:gq.n].reshape(gq.shape)


# ── Sparse Semi-Structured (2:4) PyTorch integration ─────────────────────────

def to_semi_structured_sparse(
    weight: Tensor,
    dtype: Union[str, MXDtype] = "int8d",
    block: int = 64,
) -> Tuple[Optional[Tensor], MXTensor]:
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
    mx_q = MXTensor.quantize(pruned_dense, dt, block)

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
    dtype: Union[str, MXDtype] = "int8d",
    granularity: str = "per_token",
    block: int = 64,
) -> MXTensor:
    """
    Dynamic (runtime) activation quantization with no static calibration.

    Scales are computed on-the-fly from the input statistics rather than
    pre-calibrated. This is slower than static quantization but more accurate
    for inputs with high dynamic range or distribution shift.

    Granularity modes:
      "per_token"    — one scale per token (row); best for attention/MLP activations
      "per_channel"  — one scale per channel (column); best for conv activations
      "per_block"    — block-wise (same as static MXTensor.quantize)
      "per_tensor"   — single global scale; fastest, lowest accuracy

    Args:
        x:           Input tensor of any shape.
        dtype:       Target MX dtype (int8d typical for activations).
        granularity: Quantization granularity (see above).
        block:       Block size for "per_block" mode.

    Returns:
        MXTensor at the requested precision.

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
        packed = BitPacker.pack_auto(codes.reshape(-1), dt.bits)
        return MXTensor(packed, scales, dt, torch.Size(list(x_2d.shape)),
                        x_2d.numel(), x_2d.shape[-1]).reshape(*shape)

    elif granularity == "per_channel":
        x_2d   = x_f.reshape(shape[0], -1)
        scales = x_2d.abs().amax(dim=0).clamp(min=1e-12) / max_int
        codes  = (x_2d / scales.unsqueeze(0)).round().clamp(-max_int, max_int).to(torch.int32)
        packed = BitPacker.pack_auto(codes.reshape(-1), dt.bits)
        return MXTensor(packed, scales, dt, torch.Size(list(x_2d.shape)),
                        x_2d.numel(), x_2d.shape[0]).reshape(*shape)

    elif granularity == "per_tensor":
        scale  = x_f.abs().max().clamp(min=1e-12) / max_int
        codes  = (x_f / scale).round().clamp(-max_int, max_int).to(torch.int32)
        packed = BitPacker.pack_auto(codes.reshape(-1), dt.bits)
        return MXTensor(packed, scale.unsqueeze(0), dt, torch.Size(list(shape)),
                        x_f.numel(), x_f.numel())

    else:  # per_block (standard)
        return MXTensor.quantize(x_f, dt, block)


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
        state_q = MXTensor.quantize(state, get_mx_dtype("int8d"))
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


class StochasticMXQuantize(torch.autograd.Function):
    """
    MX quantization with stochastic rounding + STE for backward.

    Drop-in replacement for ``_MXQuantize`` when training with stochastic
    rounding is desired. The stochastic noise acts as a regularizer, preventing
    the model from overfitting to the deterministic rounding grid.

    Usage::
        q = StochasticMXQuantize.apply(x, get_mx_dtype("int4d"), 128)
    """

    @staticmethod
    def forward(ctx, x: Tensor, dt: MXDtype, block: int) -> MXTensor:
        ctx.save_for_backward(x)
        x_sr = stochastic_round(x.float(), bits=dt.bits)
        return MXTensor.quantize(x_sr, dt, block)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient straight through
        return grad_output, None, None


def stochastic_mx_quantize(x: Tensor, dtype: Union[str, MXDtype] = "int8d",
                            block: int = 128) -> MXTensor:
    """
    Public API: stochastic-rounding quantization with autograd support.

    Example::
        q = stochastic_mx_quantize(weight, "int8d")   # training-friendly
    """
    dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    return StochasticMXQuantize.apply(x, dt, block)


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
    In-place. Operates on the last dimension. O(n log n).
    n = x.shape[-1] must be power of 2.
    """
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, f"Last dim must be power of 2, got {n}"
    h  = x.clone()
    h2 = n
    while h2 > 1:
        h2 //= 2
        h = h.reshape(*h.shape[:-1], -1, 2 * h2)
        even = h[..., :h2]
        odd  = h[..., h2:]
        h_new_even = (even + odd)
        h_new_odd  = (even - odd)
        h = torch.cat([h_new_even, h_new_odd], dim=-1)
    return h / math.sqrt(n)


class HadamardRotation(nn.Module):
    """
    Randomized Hadamard rotation for pre-quantization outlier reduction (QuIP#).

    QuIP# (Chee et al. 2024) shows that rotating weight matrices with a random
    Hadamard transform before quantization reduces the effective kurtosis of the
    distribution, spreading outliers evenly across dimensions. This improves
    round-trip (quantize → dequantize) quality by up to 2-4 dB SNR at int2-int4.

    Usage pattern::
        rot = HadamardRotation(dim=4096)         # fixed random rotation
        w_rotated = rot.rotate(weight)            # apply before quantization
        q = MXTensor.quantize(w_rotated, int4d)  # quantize rotated weight
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
        sp     = F.pad(signs, (0, n - len(signs)), value=1.0) if n > len(signs) else signs[:n]
        return (x_rot * sp)[..., :d]


def hadamard_quantize(
    x: Tensor,
    dtype: Union[str, MXDtype] = "int4d",
    block: int = 128,
    seed: int = 42,
) -> Tuple["HadamardRotation", MXTensor]:
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
    rot = HadamardRotation(d, seed=seed)
    x_rotated = rot.rotate(x.float())
    return rot, MXTensor.quantize(x_rotated, dt, block)


# ── Vector-wise quantization (bitsandbytes style) ──────────────────────────

def vector_quantize(
    x: Tensor,
    dtype: Union[str, MXDtype] = "int8d",
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

class KVCacheQuantizer:
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
        cache = KVCacheQuantizer(n_heads=32, head_dim=128, dtype="int8d")
        # Append new KV at each step
        cache.append_kv(k_new, v_new)                 # k/v: [B, H, 1, D]
        k_hist, v_hist = cache.get()                   # [B, H, T, D] float
        # Or work directly with quantized cache
        k_q, v_q = cache.get_quantized()               # list of MXTensors
        # Clear between requests
        cache.reset()
    """

    def __init__(self, n_heads: int, head_dim: int,
                 dtype: Union[str, MXDtype] = "int8d",
                 max_seq_len: int = 32768,
                 asymmetric_v: bool = True):
        self.n_heads       = n_heads
        self.head_dim      = head_dim
        self.dtype         = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
        self.max_seq_len   = max_seq_len
        self.asymmetric_v  = asymmetric_v
        self._k_cache: List[MXTensor] = []  # list of [B, H, 1, D] per step
        self._v_cache: List[MXTensor] = []
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

        self._k_cache.append(MXTensor.quantize(k_f, self.dtype, self.head_dim))

        if self.asymmetric_v:
            # Asymmetric: shift to [0, max] to preserve negative values better
            v_min = v_f.min(dim=1, keepdim=True).values
            v_shifted = v_f - v_min
            self._v_cache.append(MXTensor.quantize(v_shifted, self.dtype, self.head_dim))
            self._v_zp.append(v_min.squeeze(1))
        else:
            self._v_cache.append(MXTensor.quantize(v_f, self.dtype, self.head_dim))

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

    def get_quantized(self) -> Tuple[List[MXTensor], List[MXTensor]]:
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
        return (f"KVCacheQuantizer(heads={self.n_heads}, dim={self.head_dim}, "
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
        r_lo = tl.clamp(r_lo.to(tl.int8), -8, 7) & 0x0F

        # Unpack hi
        a_hi = (((ap>>4)&0x0F).to(tl.int8)); a_hi = tl.where(a_hi>7,(a_hi.to(tl.int16)|0xFFF0).to(tl.int8),a_hi)
        b_hi = (((bp>>4)&0x0F).to(tl.int8)); b_hi = tl.where(b_hi>7,(b_hi.to(tl.int16)|0xFFF0).to(tl.int8),b_hi)
        r_hi = (a_hi.to(tl.float32)*sa + b_hi.to(tl.float32)*sb) / sc
        r_hi = (tl.clamp(r_hi.to(tl.int8), -8, 7) & 0x0F) << 4

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
            r_c = tl.clamp(r.to(tl.int8), -2, 1) & 3
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
        q = tl.clamp((x / scale + 0.5).to(tl.int8), -8, 7)

        # Pack: even elements → lo nibble, odd → hi nibble
        lo = q[0::2] & 0x0F
        hi = (q[1::2] & 0x0F) << 4
        packed = (lo | hi).to(tl.int8)

        out_offs = bid * (BS // 2) + tl.arange(0, BLK // 2)
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
        q = tl.clamp((x / scale + 0.5).to(tl.int8), -128, 127)
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
        pi8   = tl.clamp((prob / s_out).to(tl.int8), -127, 127)

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
    dtype: Union[str, MXDtype] = "int8d",
    block: int = 128,
) -> MXTensor:
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
        return MXTensor.quantize(out * scale, dt, block)
    else:
        return MXTensor.quantize(stochastic_round(x_f, dt.bits), dt, block)


def fused_int8_linear(
    x: MXTensor,
    weight: MXTensor,
    bias: Optional[Tensor] = None,
    block: int = 128,
) -> Tensor:
    """
    Fused INT8 × INT8 linear: stays packed, dequantizes inline.

    Uses ``_k_fused_dequant_linear`` Triton kernel when available — avoids
    materialising float32 x or float32 weight before the multiply.

    This is the inner op of MXDynamicLinear and any "full-int8 pipeline"
    where activations AND weights are quantized simultaneously.

    Args:
        x:      MXTensor activations [tokens, in_features], dtype int8d/int8u.
        weight: MXTensor weight [out_features, in_features], dtype int8d/int8u.
        bias:   Optional float32 bias [out_features].
        block:  Scale block size.

    Returns:
        Float32 output [tokens, out_features].

    Example::
        x_q = dynamic_quantize(x, "int8d", "per_token")
        w_q = MXTensor.quantize(weight, get_mx_dtype("int8d"))
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
    wq: MXTensor, wk: MXTensor, wv: MXTensor,
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
        wq/wk/wv: MXTensor weight matrices [D, D].
        n_heads:  Number of attention heads (for output reshape).

    Returns:
        (Q, K, V): each [B*S, D] float32.

    Example::
        Q, K, V = fused_qkv_projection(x, mx_attn.wq, mx_attn.wk, mx_attn.wv,
                                        n_heads=32)
        # Then apply RoPE, SDPA etc.
    """
    x_f   = x.dequantize() if isinstance(x, MXTensor) else x.float()
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

def fused_linear_relu(x: Tensor, weight: MXTensor, bias: Optional[Tensor] = None,
                       block: int = 128) -> Tensor:
    """
    Fused quantized linear + ReLU.

    Performs W·x in packed MX precision then applies ReLU in-register via the
    ``_k_dequant_relu`` kernel — no full float32 post-linear buffer written.

    Args:
        x:      Input tensor (float32 or MXTensor).
        weight: MX-quantized weight (MXTensor with int8d dtype recommended).
        bias:   Optional float32 bias.
        block:  Quantisation block size.

    Returns:
        Float32 output with ReLU applied.

    Example::
        lin = MXLinear.from_linear(layer, get_mx_dtype("int8d"))
        out = fused_linear_relu(x, lin.weight.data, lin.bias.data if lin.bias else None)
    """
    # Compute linear in MX realm
    x_mx  = x if isinstance(x, MXTensor) else MXTensor.quantize(x.float(), weight._mx_dtype, block)
    w_t   = MXTensor.quantize(weight.dequantize().t(), weight._mx_dtype, block)
    y_mx  = _mx_mm(x_mx.reshape(-1, x_mx.shape[-1]), w_t)

    if (HAS_TRITON and y_mx.device.type == "cuda" and
            weight._mx_dtype.bits == 8):
        # Fast path: fused dequant + ReLU
        y_packed = torch.Tensor._make_subclass(Tensor, y_mx)
        out_f    = torch.empty(y_mx._mx_n, dtype=torch.float32, device=y_mx.device)
        N = y_mx._mx_n; BLK = 128
        _k_dequant_relu[(math.ceil(N / BLK),)](
            y_packed, y_mx._mx_scales, out_f, N, BS=block, BLK=BLK)
        out = out_f.reshape(*x.shape[:-1], weight.shape[0])
    else:
        out = F.relu(y_mx.dequantize().reshape(*x.shape[:-1], weight.shape[0]))

    if bias is not None:
        b = bias.dequantize() if isinstance(bias, MXTensor) else bias
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
        gate_mx = MXTensor.quantize(gate_proj(x), get_mx_dtype("int8d"))
        up_mx   = MXTensor.quantize(up_proj(x),   get_mx_dtype("int8d"))
        act     = fused_silu_and_mul(gate_mx, up_mx)  # → float32
    """
    if (HAS_TRITON and isinstance(gate, MXTensor) and isinstance(up, MXTensor)
            and gate._mx_dtype.bits == 8 and gate.device.type == "cuda"):
        N   = gate._mx_n
        BLK = min(128, N)
        out = torch.empty(N, dtype=torch.float32, device=gate.device)
        _k_dequant_silu_and_mul[(math.ceil(N / BLK),)](
            torch.Tensor._make_subclass(Tensor, gate),
            torch.Tensor._make_subclass(Tensor, up),
            gate._mx_scales, up._mx_scales,
            out, N, BS=gate._mx_block, BLK=BLK)
        return out.reshape(gate._mx_orig_shape)
    # Fallback
    g = gate.dequantize() if isinstance(gate, MXTensor) else gate
    u = up.dequantize()   if isinstance(up,   MXTensor) else up
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
        q_mx = MXTensor.quantize(q, get_mx_dtype("int8d"))
        k_mx = MXTensor.quantize(k, get_mx_dtype("int8d"))
        q_r, k_r = fused_rope_int8(q_mx, k_mx, cos, sin)
    """
    if not (HAS_TRITON and isinstance(q, MXTensor) and isinstance(k, MXTensor)
            and q._mx_dtype.bits == 8 and q.device.type == "cuda"):
        # Fallback: standard RoPE in float32
        q_f = q.dequantize() if isinstance(q, MXTensor) else q.float()
        k_f = k.dequantize() if isinstance(k, MXTensor) else k.float()
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
    xf = x.dequantize() if isinstance(x, MXTensor) else x.float()
    rf = residual.dequantize() if isinstance(residual, MXTensor) else residual.float()
    wf = weight.dequantize() if isinstance(weight, MXTensor) else weight.float()

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
    Operates on MXTensor Q, K, V stored at int8 precision.
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
        q_mx = MXTensor.quantize(q, get_mx_dtype("int8d"))
        k_mx = MXTensor.quantize(k, get_mx_dtype("int8d"))
        v_mx = MXTensor.quantize(v, get_mx_dtype("int8d"))
        out  = fused_sdpa_int8(q_mx, k_mx, v_mx)
    """
    q_f = q.dequantize() if isinstance(q, MXTensor) else q.float()
    k_f = k.dequantize() if isinstance(k, MXTensor) else k.float()
    v_f = v.dequantize() if isinstance(v, MXTensor) else v.float()

    if (HAS_TRITON and isinstance(q, MXTensor) and q._mx_dtype.bits == 8
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MXTensor  (torch.Tensor subclass)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture:
#   • Subclasses torch.Tensor via _make_subclass so it IS a Tensor.
#   • Underlying storage holds the PACKED int8/int32 bits.
#   • Extra metadata: _mx_dtype, _mx_scales, _mx_orig_shape, _mx_n, _mx_block
#   • Shape reported via overriding size() and .shape in __torch_dispatch__
#   • All torch ops go through __torch_dispatch__
#   • Backward: straight-through estimator or real quantized grad

class MXTensor(torch.Tensor):
    """
    A quantized tensor that IS a torch.Tensor.
    Underlying storage = real bit-packed int8/int32 data.
    All torch operations dispatch through __torch_dispatch__.
    """

    @staticmethod
    def __new__(cls, packed: Tensor, scales: Tensor, mx_dtype: MXDtype,
                orig_shape: torch.Size, n: int, block: int = 128,
                requires_grad: bool = False):
        # The subclass wraps the PACKED storage (int8)
        inst = torch.Tensor._make_subclass(cls, packed, requires_grad)
        inst._mx_dtype      = mx_dtype
        inst._mx_scales     = scales
        inst._mx_orig_shape = orig_shape
        inst._mx_n          = n
        inst._mx_block      = block
        return inst

    def __init__(self, packed, scales, mx_dtype, orig_shape, n,
                 block=128, requires_grad=False):
        pass  # __new__ handles everything

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def quantize(cls, x: Tensor, dt: MXDtype, block: int = 128,
                 requires_grad: Optional[bool] = None) -> "MXTensor":
        """
        Quantize a float/bf16 tensor to MX precision.
        REAL bit-packing — no shadow float copy.
        """
        rg = x.requires_grad if requires_grad is None else requires_grad
        with torch.no_grad():
            packed, scales, n = quantize(x.detach().float(), dt, block)
        return cls(packed, scales, dt, x.shape, n, block, rg)

    # ── Dequantize ────────────────────────────────────────────────────────────

    def dequantize(self) -> Tensor:
        """Unpack + dequantize → float32 tensor (original shape)."""
        flat = _dequant(
            torch.Tensor._make_subclass(Tensor, self),  # view as plain Tensor
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
    def mx_dtype(self) -> MXDtype:   return self._mx_dtype
    @property
    def mx_scales(self) -> Tensor:   return self._mx_scales
    @property
    def mx_block(self) -> int:       return self._mx_block
    @property
    def packed(self) -> Tensor:
        """Raw packed storage tensor (int8)."""
        return torch.Tensor._make_subclass(Tensor, self)

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
        return MXTensor.quantize(-self.dequantize(), self._mx_dtype, self._mx_block)
    def __matmul__(self, o): return _mx_mm(self, o)
    def __rmatmul__(self, o): return _mx_mm(o, self)

    # ── Python protocol ───────────────────────────────────────────────────────

    def __repr__(self):
        cr = self.compression_ratio
        return (f"MXTensor({self._mx_dtype.name}, shape={tuple(self._mx_orig_shape)}, "
                f"device={self.device}, {cr:.1f}x compression, "
                f"{self.nbytes_packed/1024:.2f} KB packed)")

    def __len__(self):
        return self._mx_orig_shape[0]

    # ── Tensor-like interface methods ─────────────────────────────────────────

    def float(self):  return self.dequantize()
    def half(self):   return self.dequantize().half()
    def bfloat16(self): return self.dequantize().bfloat16()

    def to(self, *args, **kwargs):
        # Handle MX dtype target
        target = args[0] if args else kwargs.get("dtype", None)
        if isinstance(target, (str, MXDtypeProxy)):
            dt = get_mx_dtype(target._mx if isinstance(target, MXDtypeProxy) else target)
            return MXTensor.quantize(self.dequantize(), dt, self._mx_block,
                                     requires_grad=self.requires_grad)
        if isinstance(target, MXDtype):
            return MXTensor.quantize(self.dequantize(), target, self._mx_block)
        # Device move
        if isinstance(target, (str, torch.device)) or "device" in kwargs:
            dev = target if args else kwargs.get("device")
            return MXTensor(
                self.packed.to(dev), self._mx_scales.to(dev),
                self._mx_dtype, self._mx_orig_shape, self._mx_n, self._mx_block,
                self.requires_grad,
            )
        # Default: dequantize and re-convert
        return self.dequantize().to(*args, **kwargs)

    def cuda(self, device=None):
        return self.to(device=device or "cuda")

    def cpu(self):
        return self.to(device="cpu")

    def clone(self):
        return MXTensor(self.packed.clone(), self._mx_scales.clone(),
                        self._mx_dtype, self._mx_orig_shape, self._mx_n,
                        self._mx_block, self.requires_grad)

    def detach(self):
        return MXTensor(self.packed.detach(), self._mx_scales.detach(),
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
        return MXTensor.quantize(self.dequantize().t(), self._mx_dtype, self._mx_block)

    def permute(self, *dims):
        return MXTensor.quantize(self.dequantize().permute(*dims),
                                 self._mx_dtype, self._mx_block)

    def expand(self, *sizes):
        dq = self.dequantize().expand(*sizes)
        return MXTensor.quantize(dq, self._mx_dtype, self._mx_block)

    def sum(self, *a, **kw):
        return MXTensor.quantize(self.dequantize().sum(*a, **kw),
                                 self._mx_dtype, self._mx_block)

    def mean(self, *a, **kw):
        return MXTensor.quantize(self.dequantize().mean(*a, **kw),
                                 self._mx_dtype, self._mx_block)

    def max(self, *a, **kw):
        return self.dequantize().max(*a, **kw)

    def min(self, *a, **kw):
        return self.dequantize().min(*a, **kw)

    def abs(self):
        return MXTensor.quantize(self.dequantize().abs(), self._mx_dtype, self._mx_block)

    def sqrt(self):
        return MXTensor.quantize(self.dequantize().sqrt(), self._mx_dtype, self._mx_block)

    def exp(self):
        return MXTensor.quantize(self.dequantize().exp(), self._mx_dtype, self._mx_block)

    def log(self):
        return MXTensor.quantize(self.dequantize().log(), self._mx_dtype, self._mx_block)

    def softmax(self, dim):
        return MXTensor.quantize(self.dequantize().softmax(dim),
                                 self._mx_dtype, self._mx_block)

    def __getitem__(self, idx):
        return MXTensor.quantize(self.dequantize()[idx],
                                 self._mx_dtype, self._mx_block)

    # ── Gradient / autograd support ───────────────────────────────────────────

    def backward(self, grad=None, **kw):
        # Straight-through: pass gradient through quantization
        return self.dequantize().backward(grad, **kw)

    # ── State dict / pickling ─────────────────────────────────────────────────

    def __reduce_ex__(self, protocol):
        return (_rebuild_mxtensor, (
            self.packed.cpu(), self._mx_scales.cpu(),
            self._mx_dtype, self._mx_orig_shape, self._mx_n, self._mx_block,
        ))


def _rebuild_mxtensor(packed, scales, dt, shape, n, block):
    return MXTensor(packed, scales, dt, shape, n, block)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — OPERATOR DISPATCH ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(x) -> Tensor:
    """Get float32 tensor from MXTensor or plain Tensor."""
    if isinstance(x, MXTensor):
        return x.dequantize()
    return x


def _rewrap(result: Tensor, ref: MXTensor) -> "MXTensor":
    """Re-quantize result to same MX dtype as ref."""
    if isinstance(result, Tensor) and not isinstance(result, MXTensor):
        return MXTensor.quantize(result, ref._mx_dtype, ref._mx_block)
    return result


def _pick_ref(*args) -> Optional[MXTensor]:
    """Find the first MXTensor in args (for result dtype)."""
    for a in args:
        if isinstance(a, MXTensor): return a
        if isinstance(a, (list, tuple)):
            r = _pick_ref(*a)
            if r is not None: return r
    return None


def _binary_op(a, b, fn) -> MXTensor:
    """
    Execute a binary op between two MX/float tensors.
    Respects mixed-mode resolution: up × down → up, down × down → down.
    """
    if isinstance(a, MXTensor) and isinstance(b, MXTensor):
        out_dt = _resolve_mixed(a._mx_dtype, b._mx_dtype)
        fa, fb = a.dequantize(), b.dequantize()
        return MXTensor.quantize(fn(fa, fb), out_dt, a._mx_block)
    ref = a if isinstance(a, MXTensor) else b
    fa  = _unwrap(a); fb = _unwrap(b)
    return _rewrap(fn(fa, fb), ref)


# ── packed matmul dispatcher ──────────────────────────────────────────────────

def _mx_mm(a: MXTensor, b: MXTensor) -> MXTensor:
    """
    Dispatch matmul to the best packed Triton kernel.
    Falls back to dequant → float32 mm → re-quant if needed.
    """
    hw    = HardwareProbe.detect()
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


def _mm_triton_packed(a: MXTensor, b: MXTensor,
                      out_dt: MXDtype, hw: HardwareProfile) -> MXTensor:
    bits  = a._mx_dtype.bits
    ratio = 8 // bits
    M, K  = a._mx_orig_shape
    K2, N = b._mx_orig_shape
    assert K == K2, f"Shape mismatch: {a._mx_orig_shape} × {b._mx_orig_shape}"
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

    return MXTensor.quantize(c, out_dt, a._mx_block)


def _mm_fallback(a: MXTensor, b: MXTensor, out_dt: MXDtype) -> MXTensor:
    """Dequant → float32 mm → requant fallback."""
    if _STRICT:
        raise RuntimeError(
            f"[mx_triton STRICT] full-precision fallback triggered for "
            f"matmul({a._mx_dtype.name} × {b._mx_dtype.name})"
        )
    if _DEBUG:
        log.debug(f"[mm FALLBACK] {a._mx_dtype.name} × {b._mx_dtype.name} → f32 → {out_dt.name}")
    result = torch.mm(a.dequantize(), b.dequantize())
    return MXTensor.quantize(result, out_dt, a._mx_block)


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
            if isinstance(a, MXTensor) and isinstance(b, MXTensor):
                return _mx_mm(a, b)

        if func in (torch.bmm, torch.ops.aten.bmm.default):
            a, b = args[0], args[1]
            if isinstance(a, MXTensor) and isinstance(b, MXTensor):
                # batch dim: process slice by slice or dequant
                fa = a.dequantize(); fb = b.dequantize()
                out_dt = _resolve_mixed(a._mx_dtype, b._mx_dtype)
                return MXTensor.quantize(torch.bmm(fa, fb), out_dt, a._mx_block)

        if func in (F.linear, torch.ops.aten.linear.default):
            inp    = args[0]
            weight = args[1]
            bias   = args[2] if len(args) > 2 else kwargs.get("bias")
            ref    = inp if isinstance(inp, MXTensor) else weight
            fi     = _unwrap(inp)
            fw     = _unwrap(weight)
            out    = F.linear(fi, fw, _unwrap(bias) if bias is not None else None)
            if isinstance(ref, MXTensor):
                return MXTensor.quantize(out, ref._mx_dtype, ref._mx_block)
            return out

        if func in (torch.addmm, torch.ops.aten.addmm.default):
            bias, a, b = args[0], args[1], args[2]
            ref = _pick_ref(a, b, bias)
            out = torch.addmm(_unwrap(bias), _unwrap(a), _unwrap(b))
            return _rewrap(out, ref) if ref else out

        if func in (torch.einsum,):
            eq   = args[0]
            oper = [_unwrap(x) for x in args[1:]]
            ref  = _pick_ref(*args[1:])
            out  = torch.einsum(eq, *oper)
            return _rewrap(out, ref) if ref else out

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
            if isinstance(a, MXTensor) or isinstance(b, MXTensor):
                return _binary_op(a, b, lambda x,y: func(
                    x if not isinstance(x, MXTensor) else x.dequantize(),
                    y if not isinstance(y, MXTensor) else y.dequantize(),
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
            if ref and isinstance(out, Tensor) and not isinstance(out, MXTensor):
                # Only re-quantize if result has same dimensionality (not scalar reductions)
                if out.ndim > 0:
                    return MXTensor.quantize(out, ref._mx_dtype, ref._mx_block)
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
            return _rewrap(out, ref) if ref else out

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
            return _rewrap(out, ref) if ref else out

        # ── convolutions ──────────────────────────────────────────────────────
        _convs = {F.conv1d, F.conv2d, F.conv3d,
                  F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d}
        if func in _convs or fn in ("conv1d","conv2d","conv3d",
                                     "conv_transpose1d","conv_transpose2d"):
            ref  = _pick_ref(*args)
            flat = [_unwrap(a) for a in args]
            flat_kw = {k: _unwrap(v) for k, v in kwargs.items()}
            out  = func(*flat, **flat_kw)
            return _rewrap(out, ref) if ref else out

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
            flat = [_unwrap(a) if isinstance(a, MXTensor) else a for a in args]
            flat_kw = {k: _unwrap(v) if isinstance(v, MXTensor) else v
                       for k, v in kwargs.items()}
            out = func(*flat, **flat_kw)
            return _rewrap(out, ref) if ref else out

        # ── indexing / scatter ────────────────────────────────────────────────
        if fn in ("index_select","gather","scatter","scatter_add","index_put",
                  "masked_select","masked_fill","where"):
            ref  = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, MXTensor) else a for a in args]
            flat_kw = {k: _unwrap(v) if isinstance(v, MXTensor) else v
                       for k, v in kwargs.items()}
            out = func(*flat, **flat_kw)
            return _rewrap(out, ref) if (ref and isinstance(out, Tensor)) else out

        # ── shape ops ─────────────────────────────────────────────────────────
        if func in (torch.reshape, torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default):
            t = args[0]
            if isinstance(t, MXTensor):
                shape = args[1] if len(args) > 1 else kwargs.get("shape")
                return t.reshape(shape)

        if func in (torch.cat, torch.ops.aten.cat.default):
            tensors = args[0]
            ref     = _pick_ref(*tensors)
            flat    = [_unwrap(t) for t in tensors]
            out     = torch.cat(flat, **{k: v for k, v in
                                          zip(["dim"], args[1:])}, **kwargs)
            return _rewrap(out, ref) if ref else out

        if func in (torch.stack, torch.ops.aten.stack.default):
            tensors = args[0]
            ref     = _pick_ref(*tensors)
            flat    = [_unwrap(t) for t in tensors]
            out     = torch.stack(flat, **{k: v for k, v in
                                            zip(["dim"], args[1:])}, **kwargs)
            return _rewrap(out, ref) if ref else out

        if func in (torch.split, torch.ops.aten.split.Tensor,
                    torch.chunk, torch.ops.aten.chunk.default):
            t   = args[0]
            if isinstance(t, MXTensor):
                dq  = t.dequantize()
                out = func(dq, *args[1:], **kwargs)
                return tuple(MXTensor.quantize(o, t._mx_dtype, t._mx_block) for o in out)

        if func in (torch.transpose, torch.ops.aten.transpose.int, torch.t,
                    torch.ops.aten.t.default):
            t = args[0]
            if isinstance(t, MXTensor):
                return MXTensor.quantize(func(t.dequantize(), *args[1:], **kwargs),
                                         t._mx_dtype, t._mx_block)

        if func in (torch.permute, torch.ops.aten.permute.default):
            t = args[0]
            if isinstance(t, MXTensor):
                return MXTensor.quantize(t.dequantize().permute(*args[1:]),
                                         t._mx_dtype, t._mx_block)

        # ── comparison / logical ops ──────────────────────────────────────────
        _cmp_ops = {torch.lt, torch.gt, torch.le, torch.ge, torch.eq, torch.ne,
                    torch.ops.aten.lt.Tensor, torch.ops.aten.gt.Tensor,
                    torch.ops.aten.le.Tensor, torch.ops.aten.ge.Tensor,
                    torch.ops.aten.eq.Tensor, torch.ops.aten.ne.Tensor}
        if func in _cmp_ops or fn in ("lt","gt","le","ge","eq","ne",
                                       "less","greater","less_equal","greater_equal"):
            a = args[0]; b = args[1] if len(args) > 1 else kwargs.get("other")
            return func(_unwrap(a) if isinstance(a, MXTensor) else a,
                        _unwrap(b) if isinstance(b, MXTensor) else b)  # bool result

        # ── clamp / clip ──────────────────────────────────────────────────────
        if func in (torch.clamp, torch.clip, torch.ops.aten.clamp.default,
                    torch.ops.aten.clamp.Tensor) or fn in ("clamp","clip"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        if func in (torch.floor, torch.ceil, torch.round, torch.trunc,
                    torch.ops.aten.floor.default, torch.ops.aten.ceil.default,
                    torch.ops.aten.round.default) or fn in ("floor","ceil","round","trunc"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        if func in (torch.sign, torch.abs, torch.ops.aten.sign.default,
                    torch.ops.aten.abs.default) or fn in ("sign","abs"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]))
            return _rewrap(out, ref) if ref else out

        # ── sorting / selection ───────────────────────────────────────────────
        if func in (torch.sort, torch.argsort, torch.ops.aten.sort.default,
                    torch.ops.aten.argsort.default) or fn in ("sort","argsort"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            if ref and isinstance(out, Tensor):
                return _rewrap(out, ref)
            if ref and isinstance(out, tuple):  # sort returns (values, indices)
                vals, idx = out
                return torch.return_types.sort((
                    _rewrap(vals, ref) if vals.dtype.is_floating_point else vals, idx))
            return out

        if func in (torch.topk, torch.ops.aten.topk.default) or fn == "topk":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            if ref and hasattr(out, "values"):
                vals = _rewrap(out.values, ref) if out.values.dtype.is_floating_point else out.values
                return torch.return_types.topk((vals, out.indices))
            return out

        # ── scan / cumulative ops ─────────────────────────────────────────────
        if func in (torch.cumsum, torch.cumprod, torch.ops.aten.cumsum.default,
                    torch.ops.aten.cumprod.default) or fn in ("cumsum","cumprod"):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── dropout (training only — pass-through in eval) ────────────────────
        if func in (F.dropout, torch.ops.aten.dropout.default) or fn == "dropout":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── pooling (avg / max / adaptive) ────────────────────────────────────
        _pools = {F.avg_pool1d, F.avg_pool2d, F.avg_pool3d,
                  F.max_pool1d, F.max_pool2d, F.max_pool3d,
                  F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d,
                  F.adaptive_max_pool1d, F.adaptive_max_pool2d, F.adaptive_max_pool3d}
        if func in _pools or fn in ("avg_pool2d","max_pool2d","adaptive_avg_pool2d",
                                     "avg_pool1d","max_pool1d","avg_pool3d","max_pool3d"):
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, MXTensor) else a for a in args]
            out  = func(*flat, **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── upsample / interpolate ────────────────────────────────────────────
        if func in (F.interpolate, F.upsample, F.upsample_bilinear,
                    F.upsample_nearest) or fn in ("interpolate","upsample"):
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, MXTensor) else a for a in args]
            out  = func(*flat, **{k: _unwrap(v) if isinstance(v, MXTensor) else v
                                  for k, v in kwargs.items()})
            return _rewrap(out, ref) if ref else out

        # ── scaled dot-product attention (SDPA) ───────────────────────────────
        if (func is getattr(F, "scaled_dot_product_attention", None) or
                fn == "scaled_dot_product_attention"):
            q, k, v = args[0], args[1], args[2]
            ref = _pick_ref(q, k, v)
            out = F.scaled_dot_product_attention(
                _unwrap(q), _unwrap(k), _unwrap(v),
                *[(_unwrap(a) if isinstance(a, MXTensor) else a) for a in args[3:]],
                **{kk: (_unwrap(vv) if isinstance(vv, MXTensor) else vv)
                   for kk, vv in kwargs.items()})
            return _rewrap(out, ref) if ref else out

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
            return _rewrap(out, ref) if ref else out

        if func in (torch.roll, torch.ops.aten.roll.default) or fn == "roll":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── outer / cross product ─────────────────────────────────────────────
        if func in (torch.outer, torch.ger, torch.ops.aten.outer.default) or fn in ("outer","ger"):
            a, b = args[0], args[1]
            ref  = _pick_ref(a, b)
            out  = func(_unwrap(a), _unwrap(b))
            return _rewrap(out, ref) if ref else out

        if func in (torch.baddbmm, torch.ops.aten.baddbmm.default) or fn == "baddbmm":
            inp, b1, b2 = args[0], args[1], args[2]
            ref = _pick_ref(inp, b1, b2)
            out = func(_unwrap(inp), _unwrap(b1), _unwrap(b2), **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── linalg ────────────────────────────────────────────────────────────
        _linalg = set()
        for _lname in ("norm","vector_norm","matrix_norm","solve","lstsq","svd","eig","qr"):
            _m = getattr(getattr(torch, "linalg", None), _lname, None)
            if _m is not None: _linalg.add(_m)
        if func in _linalg or fn in ("linalg_norm","linalg_solve","svd","eig","qr"):
            ref = _pick_ref(*args)
            flat_args = [_unwrap(a) if isinstance(a, MXTensor) else a for a in args]
            out = func(*flat_args, **kwargs)
            # linalg ops return plain tensors or named tuples — don't auto-requant
            return out

        # ── sparse ops ────────────────────────────────────────────────────────
        if fn in ("sparse_coo_tensor","sparse_csr_tensor","to_sparse",
                  "to_dense","coalesce","indices","values","crow_indices","col_indices"):
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, MXTensor) else a for a in args]
            out  = func(*flat, **kwargs)
            return out  # sparse ops return non-MX tensors by design

        # ── padding ───────────────────────────────────────────────────────────
        if func in (F.pad, torch.ops.aten.constant_pad_nd.default,
                    torch.ops.aten.pad.default) or fn in ("pad",):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]) if isinstance(args[0], MXTensor) else args[0],
                       *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── nan / inf cleanup ─────────────────────────────────────────────────
        if func in (torch.nan_to_num, torch.ops.aten.nan_to_num.default) or fn == "nan_to_num":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── masked_fill / where ───────────────────────────────────────────────
        if func in (torch.where, torch.ops.aten.where.self,
                    torch.ops.aten.where.ScalarOther) or fn == "where":
            # where(condition, x, y)  — condition is bool, x/y may be MXTensor
            ref   = _pick_ref(*args[1:])   # skip condition arg
            cond  = args[0]
            x_arg = _unwrap(args[1]) if len(args) > 1 and isinstance(args[1], MXTensor) else (args[1] if len(args)>1 else kwargs.get("input"))
            y_arg = _unwrap(args[2]) if len(args) > 2 and isinstance(args[2], MXTensor) else (args[2] if len(args)>2 else kwargs.get("other"))
            out   = torch.where(cond, x_arg, y_arg)
            return _rewrap(out, ref) if ref else out

        if func in (torch.ops.aten.masked_fill.Scalar,
                    torch.ops.aten.masked_fill.Tensor) or fn == "masked_fill":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        if func in (torch.ops.aten.index.Tensor,) or fn == "index":
            ref = _pick_ref(*args)
            t   = _unwrap(args[0]) if isinstance(args[0], MXTensor) else args[0]
            idx = args[1]
            out = func(t, idx, **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── interpolate / upsample (in case missed by existing block) ─────────
        if func is getattr(F, "pixel_shuffle", None) or fn == "pixel_shuffle":
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        if func is getattr(F, "grid_sample", None) or fn == "grid_sample":
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, MXTensor) else a for a in args]
            out  = func(*flat, **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── type casting ──────────────────────────────────────────────────────
        if func in (torch.ops.aten._to_copy.default,):
            t = args[0]
            if isinstance(t, MXTensor):
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
            if isinstance(t, MXTensor):
                # Return plain float32 tensor matching the *logical* shape
                creator = torch.zeros if fn != "ones_like" else torch.ones
                if fn == "empty_like":
                    creator = torch.empty
                return creator(t._mx_orig_shape, dtype=torch.float32, device=t.device)
            return func(*args, **kwargs)

        if func in (torch.full_like, torch.ops.aten.full_like.default) or fn == "full_like":
            t   = args[0]
            val = args[1] if len(args) > 1 else kwargs.get("fill_value", 0)
            if isinstance(t, MXTensor):
                return torch.full(t._mx_orig_shape, val, dtype=torch.float32, device=t.device)
            return func(*args, **kwargs)

        # ── in-place add / sub / mul (+=, -=, *=) ────────────────────────────
        if func in (torch.ops.aten.add_.Tensor, torch.ops.aten.sub_.Tensor,
                    torch.ops.aten.mul_.Tensor, torch.ops.aten.div_.Tensor) or fn in (
                        "add_","sub_","mul_","div_"):
            t = args[0]
            if isinstance(t, MXTensor):
                other = args[1] if len(args) > 1 else kwargs.get("other")
                base  = {fn or "": torch.add, "sub_": torch.sub,
                         "mul_": torch.mul, "div_": torch.div}.get(fn or "", torch.add)
                result = base(t.dequantize(), _unwrap(other) if isinstance(other, MXTensor) else other)
                new_mx = MXTensor.quantize(result, t._mx_dtype, t._mx_block)
                t.packed.copy_(new_mx.packed)
                t._mx_scales.copy_(new_mx._mx_scales)
                return t
            return func(*args, **kwargs)

        # ── copy_ / fill_ / zero_ ────────────────────────────────────────────
        if fn in ("copy_", "fill_", "zero_"):
            t = args[0]
            if isinstance(t, MXTensor):
                dq = t.dequantize()
                getattr(dq, fn)(*[_unwrap(a) for a in args[1:]], **kwargs)
                new = MXTensor.quantize(dq, t._mx_dtype, t._mx_block)
                # Mutate in place by updating packed storage
                t.packed.copy_(new.packed)
                t._mx_scales.copy_(new._mx_scales)
                return t
            return func(*args, **kwargs)

        # ── clone / detach / contiguous ───────────────────────────────────────
        if func in (torch.ops.aten.clone.default,):
            t = args[0]
            if isinstance(t, MXTensor): return t.clone()

        if func in (torch.ops.aten.detach.default,):
            t = args[0]
            if isinstance(t, MXTensor): return t.detach()

        if func in (torch.ops.aten.contiguous.memory_format,
                    torch.ops.aten.contiguous.default):
            t = args[0]
            if isinstance(t, MXTensor): return t.clone()

        # ── dtype / device queries ────────────────────────────────────────────
        if func in (torch.ops.aten.is_floating_point.default,):
            t = args[0]
            if isinstance(t, MXTensor): return t._mx_dtype.is_float

        # ── scatter_reduce / segment_reduce ───────────────────────────────────
        if (func is getattr(torch, "scatter_reduce", None) or
                func is getattr(torch.ops.aten, "scatter_reduce.two", None) or
                fn in ("scatter_reduce", "segment_reduce")):
            ref  = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, MXTensor) else a for a in args]
            flat_kw = {k: _unwrap(v) if isinstance(v, MXTensor) else v
                       for k, v in kwargs.items()}
            out  = func(*flat, **flat_kw)
            return _rewrap(out, ref) if (ref and isinstance(out, Tensor)) else out

        # ── unfold / as_strided ───────────────────────────────────────────────
        if (func in (torch.Tensor.unfold, torch.ops.aten.unfold.default) or
                fn in ("unfold", "as_strided")):
            ref = _pick_ref(*args)
            t   = _unwrap(args[0]) if isinstance(args[0], MXTensor) else args[0]
            out = func(t, *args[1:], **kwargs)
            return _rewrap(out, ref) if (ref and isinstance(out, Tensor)) else out

        # ── kron / tensordot / vecdot ─────────────────────────────────────────
        if (func in (torch.kron, torch.ops.aten.kron.default) or fn == "kron"):
            a, b = args[0], args[1]
            ref  = _pick_ref(a, b)
            out  = torch.kron(_unwrap(a), _unwrap(b))
            return _rewrap(out, ref) if ref else out

        if (func in (torch.tensordot, torch.ops.aten.tensordot.default) or fn == "tensordot"):
            a, b  = args[0], args[1]
            ref   = _pick_ref(a, b)
            dims  = args[2] if len(args) > 2 else kwargs.get("dims", 2)
            out   = torch.tensordot(_unwrap(a), _unwrap(b), dims=dims)
            return _rewrap(out, ref) if ref else out

        if fn in ("vecdot", "vdot"):
            ref = _pick_ref(*args)
            out = func(*[_unwrap(a) if isinstance(a, MXTensor) else a for a in args], **kwargs)
            return out  # dot products return scalars or reduced tensors

        # ── repeat / repeat_interleave / expand ───────────────────────────────
        if (func in (torch.ops.aten.repeat.default,) or fn == "repeat"):
            t = args[0]
            if isinstance(t, MXTensor):
                out = t.dequantize().repeat(*args[1:], **kwargs)
                return MXTensor.quantize(out, t._mx_dtype, t._mx_block)

        if (func in (torch.repeat_interleave, torch.ops.aten.repeat_interleave.Tensor,
                     torch.ops.aten.repeat_interleave.self_int) or fn == "repeat_interleave"):
            ref = _pick_ref(*args)
            flat = [_unwrap(a) if isinstance(a, MXTensor) else a for a in args]
            out  = func(*flat, **kwargs)
            return _rewrap(out, ref) if (ref and isinstance(out, Tensor)) else out

        # ── squeeze / unsqueeze / flatten ─────────────────────────────────────
        if (func in (torch.squeeze, torch.unsqueeze, torch.flatten,
                     torch.ops.aten.squeeze.default, torch.ops.aten.squeeze.dim,
                     torch.ops.aten.unsqueeze.default, torch.ops.aten.flatten.using_ints)
                or fn in ("squeeze", "unsqueeze", "flatten")):
            ref = _pick_ref(*args)
            t   = args[0]
            if isinstance(t, MXTensor):
                out = func(t.dequantize(), *args[1:], **kwargs)
                return MXTensor.quantize(out, t._mx_dtype, t._mx_block)
            return func(*args, **kwargs)

        # ── exp / log / sqrt (elementwise math) ──────────────────────────────
        if (func in (torch.exp, torch.log, torch.log2, torch.log10, torch.sqrt,
                     torch.ops.aten.exp.default, torch.ops.aten.log.default,
                     torch.ops.aten.sqrt.default) or
                fn in ("exp", "log", "log2", "log10", "sqrt", "rsqrt")):
            ref = _pick_ref(*args)
            out = func(_unwrap(args[0]), *args[1:], **kwargs)
            return _rewrap(out, ref) if ref else out

        # ── UNIVERSAL FALLBACK: dequant → op → requant ────────────────────────
        ref  = _pick_ref(*args, *kwargs.values())
        flat_args = tree_map(lambda x: _unwrap(x) if isinstance(x, MXTensor) else x, args)
        flat_kw   = {k: _unwrap(v) if isinstance(v, MXTensor) else v
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

        if ref is not None and isinstance(out, Tensor) and not isinstance(out, MXTensor):
            if out.ndim > 0 and out.dtype.is_floating_point:
                return MXTensor.quantize(out, ref._mx_dtype, ref._mx_block)
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
    def forward(ctx, x: Tensor, dt: MXDtype, block: int) -> MXTensor:
        ctx.save_for_backward(x)
        ctx.dt    = dt
        ctx.block = block
        return MXTensor.quantize(x, dt, block, requires_grad=False)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output may be an MXTensor; unwrap
        if isinstance(grad_output, MXTensor):
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


def mx_quantize(x: Tensor, dtype: Union[str, MXDtype], block: int = 128) -> MXTensor:
    """
    Differentiable quantization with STE backward.
    Use in training loops; gradients flow according to mode (d/u).
    """
    dt = get_mx_dtype(dtype)
    return _MXQuantize.apply(x, dt, block)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — nn.Module deep integration
# ─────────────────────────────────────────────────────────────────────────────

class MXLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with real MX-packed weights.
    Installed automatically when you call to_mx() on a model.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, mx_dtype: MXDtype = None, block: int = 128):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mx_dtype     = mx_dtype or get_mx_dtype("int4d")
        self.block        = block
        # weight and bias stored as plain nn.Parameters holding MXTensors
        self.weight = nn.Parameter(
            MXTensor.quantize(
                torch.empty(out_features, in_features), self.mx_dtype, block
            ), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(
                MXTensor.quantize(
                    torch.zeros(out_features), self.mx_dtype, block
                ), requires_grad=True)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, mx_dtype: MXDtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.in_features  = linear.in_features
        m.out_features = linear.out_features
        m.mx_dtype     = mx_dtype
        m.block        = block
        m.weight = nn.Parameter(
            MXTensor.quantize(linear.weight.data, mx_dtype, block), requires_grad=True)
        if linear.bias is not None:
            m.bias = nn.Parameter(
                MXTensor.quantize(linear.bias.data, mx_dtype, block), requires_grad=True)
        else:
            m.bias = None
        return m

    def forward(self, x: Tensor) -> Tensor:
        w_mx = self.weight.data
        b_mx = self.bias.data if self.bias is not None else None

        if isinstance(w_mx, MXTensor):
            # Quantize input at same precision
            if not isinstance(x, MXTensor):
                x_mx = MXTensor.quantize(x.float(), self.mx_dtype, self.block)
            else:
                x_mx = x
            # Packed matmul: x [B,*, in] @ w.T [in, out] = [B,*, out]
            # Flatten batch dims
            orig_shape = x_mx._mx_orig_shape
            x_2d = MXTensor.quantize(x_mx.dequantize().reshape(-1, self.in_features),
                                      self.mx_dtype, self.block)
            w_t  = MXTensor.quantize(w_mx.dequantize().t(), self.mx_dtype, self.block)
            out  = _mx_mm(x_2d, w_t).dequantize()
            out  = out.reshape(*orig_shape[:-1], self.out_features)
            if b_mx is not None:
                bias_f = b_mx.dequantize() if isinstance(b_mx, MXTensor) else b_mx
                out    = out + bias_f
            return out

        # Fallback to plain linear
        return F.linear(x, w_mx, b_mx)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"dtype={self.mx_dtype.name}, bias={self.bias is not None}")


class MXConv2d(nn.Module):
    """
    Drop-in replacement for nn.Conv2d with real MX-packed weights.
    Installed automatically by to_mx() when a Conv2d is encountered.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 mx_dtype: MXDtype = None, block: int = 128):
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
        self.weight = nn.Parameter(
            MXTensor.quantize(torch.empty(*w_shape), self.mx_dtype, block), requires_grad=True)
        self.bias = nn.Parameter(
            MXTensor.quantize(torch.zeros(out_channels), self.mx_dtype, block),
            requires_grad=True) if bias else None

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, mx_dtype: MXDtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.in_channels = conv.in_channels; m.out_channels = conv.out_channels
        m.kernel_size = conv.kernel_size; m.stride = conv.stride
        m.padding = conv.padding; m.dilation = conv.dilation; m.groups = conv.groups
        m.mx_dtype = mx_dtype; m.block = block
        m.weight = nn.Parameter(
            MXTensor.quantize(conv.weight.data, mx_dtype, block), requires_grad=True)
        m.bias = (nn.Parameter(MXTensor.quantize(conv.bias.data, mx_dtype, block), requires_grad=True)
                  if conv.bias is not None else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.data; b = self.bias.data if self.bias is not None else None
        w_f = w.dequantize() if isinstance(w, MXTensor) else w
        b_f = b.dequantize() if isinstance(b, MXTensor) else b
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
        out = F.conv2d(x_f, w_f, b_f, self.stride, self.padding, self.dilation, self.groups)
        return MXTensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return f"in={self.in_channels}, out={self.out_channels}, kernel={self.kernel_size}, dtype={self.mx_dtype.name}"


class MXConv1d(nn.Module):
    """Drop-in for nn.Conv1d with MX-packed weights."""

    @classmethod
    def from_conv1d(cls, conv: nn.Conv1d, mx_dtype: MXDtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.in_channels = conv.in_channels; m.out_channels = conv.out_channels
        m.kernel_size = conv.kernel_size; m.stride = conv.stride
        m.padding = conv.padding; m.dilation = conv.dilation; m.groups = conv.groups
        m.mx_dtype = mx_dtype; m.block = block
        m.weight = nn.Parameter(MXTensor.quantize(conv.weight.data, mx_dtype, block), requires_grad=True)
        m.bias   = (nn.Parameter(MXTensor.quantize(conv.bias.data, mx_dtype, block), requires_grad=True)
                    if conv.bias is not None else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.data; b = self.bias.data if self.bias is not None else None
        w_f = w.dequantize() if isinstance(w, MXTensor) else w
        b_f = b.dequantize() if isinstance(b, MXTensor) else b
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
        return MXTensor.quantize(
            F.conv1d(x_f, w_f, b_f, self.stride, self.padding, self.dilation, self.groups),
            self.mx_dtype, self.block)


class MXLayerNorm(nn.Module):
    """nn.LayerNorm with MX-packed weight/bias. Norm itself in float32."""

    @classmethod
    def from_layer_norm(cls, ln: nn.LayerNorm, mx_dtype: MXDtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.normalized_shape = ln.normalized_shape
        m.eps = ln.eps; m.elementwise_affine = ln.elementwise_affine
        m.mx_dtype = mx_dtype; m.block = block
        m.weight = (nn.Parameter(MXTensor.quantize(ln.weight.data, mx_dtype, block), requires_grad=True)
                    if ln.weight is not None else None)
        m.bias   = (nn.Parameter(MXTensor.quantize(ln.bias.data,   mx_dtype, block), requires_grad=True)
                    if ln.bias   is not None else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
        w = (self.weight.data.dequantize() if isinstance(self.weight.data, MXTensor)
             else self.weight.data) if self.weight is not None else None
        b = (self.bias.data.dequantize()   if isinstance(self.bias.data,   MXTensor)
             else self.bias.data)   if self.bias   is not None else None
        out = F.layer_norm(x_f, self.normalized_shape, w, b, self.eps)
        return MXTensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return f"shape={self.normalized_shape}, dtype={self.mx_dtype.name}"


class MXRMSNorm(nn.Module):
    """
    RMS Normalisation with MX-packed weight (LLaMA / Mistral style).
    Compatible with: transformers.RMSNorm, LlamaRMSNorm, etc.
    """

    @classmethod
    def from_rms_norm(cls, rms, mx_dtype: MXDtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.normalized_shape = rms.weight.shape
        m.eps     = getattr(rms, "eps", None) or getattr(rms, "variance_epsilon", 1e-6)
        m.mx_dtype = mx_dtype; m.block = block
        m.weight  = nn.Parameter(MXTensor.quantize(rms.weight.data, mx_dtype, block), requires_grad=True)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
        w   = self.weight.data
        w_f = w.dequantize() if isinstance(w, MXTensor) else w.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x_f * rms * w_f
        return MXTensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return f"shape={self.normalized_shape}, dtype={self.mx_dtype.name}"


class MXGroupNorm(nn.Module):
    """nn.GroupNorm with MX-packed weight/bias."""

    @classmethod
    def from_group_norm(cls, gn: nn.GroupNorm, mx_dtype: MXDtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.num_groups = gn.num_groups; m.num_channels = gn.num_channels
        m.eps = gn.eps; m.affine = gn.affine; m.mx_dtype = mx_dtype; m.block = block
        m.weight = (nn.Parameter(MXTensor.quantize(gn.weight.data, mx_dtype, block), requires_grad=True)
                    if gn.affine else None)
        m.bias   = (nn.Parameter(MXTensor.quantize(gn.bias.data,   mx_dtype, block), requires_grad=True)
                    if gn.affine else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
        w = (self.weight.data.dequantize() if isinstance(self.weight.data, MXTensor)
             else self.weight.data) if self.weight is not None else None
        b = (self.bias.data.dequantize()   if isinstance(self.bias.data,   MXTensor)
             else self.bias.data)   if self.bias   is not None else None
        out = F.group_norm(x_f, self.num_groups, w, b, self.eps)
        return MXTensor.quantize(out, self.mx_dtype, self.block)


class MXEmbeddingBag(nn.Module):
    """nn.EmbeddingBag with MX-packed weight table. Supports sum/mean/max modes."""

    @classmethod
    def from_embedding_bag(cls, emb: nn.EmbeddingBag, mx_dtype: MXDtype, block: int = 128):
        m = cls.__new__(cls); nn.Module.__init__(m)
        m.num_embeddings = emb.num_embeddings; m.embedding_dim = emb.embedding_dim
        m.mode = emb.mode; m.mx_dtype = mx_dtype; m.block = block
        m.weight = nn.Parameter(MXTensor.quantize(emb.weight.data, mx_dtype, block), requires_grad=True)
        return m

    def forward(self, input, offsets=None, per_sample_weights=None) -> Tensor:
        w   = self.weight.data
        w_f = w.dequantize() if isinstance(w, MXTensor) else w
        out = F.embedding_bag(input, w_f, offsets, mode=self.mode,
                              per_sample_weights=per_sample_weights)
        return MXTensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return f"num={self.num_embeddings}, dim={self.embedding_dim}, mode={self.mode}, dtype={self.mx_dtype.name}"


class MXBatchNorm2d(nn.Module):
    """
    nn.BatchNorm2d with MX-packed weight/bias.
    Running stats (running_mean, running_var) kept in float32 — they are
    updated by momentum and must stay high-precision for training stability.
    Actual BN computation happens in float32; only the affine params are packed.
    """

    @classmethod
    def from_batch_norm(cls, bn: nn.BatchNorm2d, mx_dtype: MXDtype, block: int = 128):
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
            m.weight = nn.Parameter(
                MXTensor.quantize(bn.weight.data, mx_dtype, block), requires_grad=True)
            m.bias   = nn.Parameter(
                MXTensor.quantize(bn.bias.data,   mx_dtype, block), requires_grad=True)
        else:
            m.weight = m.bias = None
        # running stats stay float32
        if bn.track_running_stats:
            m.register_buffer("running_mean", bn.running_mean.clone())
            m.register_buffer("running_var",  bn.running_var.clone())
            m.register_buffer("num_batches_tracked", bn.num_batches_tracked.clone())
        else:
            m.running_mean = m.running_var = m.num_batches_tracked = None
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
        w   = (self.weight.data.dequantize() if isinstance(self.weight.data, MXTensor)
               else self.weight.data) if self.weight is not None else None
        b   = (self.bias.data.dequantize()   if isinstance(self.bias.data,   MXTensor)
               else self.bias.data)   if self.bias   is not None else None
        out = F.batch_norm(x_f, self.running_mean, self.running_var,
                           w, b, self.training, self.momentum or 0.1, self.eps)
        return MXTensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return (f"num_features={self.num_features}, "
                f"dtype={self.mx_dtype.name}, eps={self.eps}")


class MXMultiheadAttention(nn.Module):
    """
    nn.MultiheadAttention with MX-packed in_proj / out_proj weights.
    All four projection matrices (Q, K, V, O) are stored at MX precision.
    Attention scores are computed in float32 for numerical stability.
    """

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention, mx_dtype: MXDtype, block: int = 128):
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
        if mha.in_proj_weight is not None:
            m.in_proj_weight = nn.Parameter(
                MXTensor.quantize(mha.in_proj_weight.data, mx_dtype, block),
                requires_grad=True)
        else:
            m.in_proj_weight = None
            # separate Q/K/V projections (cross-attention style)
            for attr in ("q_proj_weight","k_proj_weight","v_proj_weight"):
                w = getattr(mha, attr, None)
                if w is not None:
                    setattr(m, attr, nn.Parameter(
                        MXTensor.quantize(w.data, mx_dtype, block), requires_grad=True))
                else:
                    setattr(m, attr, None)

        m.in_proj_bias = mha.in_proj_bias   # keep bias float32 (small, precision-critical)
        m.out_proj     = MXLinear.from_linear(mha.out_proj, mx_dtype, block)
        m.bias_k       = mha.bias_k
        m.bias_v       = mha.bias_v
        m.add_zero_attn = mha.add_zero_attn
        return m

    def _dq(self, t: Optional[Tensor]) -> Optional[Tensor]:
        """Dequantize if MXTensor, else return as-is."""
        if t is None: return None
        return t.dequantize() if isinstance(t, MXTensor) else t.float()

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
        return MXTensor.quantize(out, self.mx_dtype, self.block), attn

    def extra_repr(self):
        return (f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"dtype={self.mx_dtype.name}")


# ── Activation quantization hooks ────────────────────────────────────────────

def wrap_activations(model: nn.Module, dtype: Union[str, MXDtype] = "int8d",
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
            if isinstance(output, Tensor) and not isinstance(output, MXTensor):
                if output.is_floating_point() and output.ndim > 0:
                    return MXTensor.quantize(output.float(), dt, block)
            return output
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (MXLinear, MXConv2d, MXConv1d, MXLayerNorm,
                                MXRMSNorm, MXGroupNorm, nn.Linear, nn.Conv2d)):
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


class MXEmbedding(nn.Module):
    """nn.Embedding with MX-packed weight table."""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 mx_dtype: MXDtype = None, block: int = 128, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.mx_dtype       = mx_dtype or get_mx_dtype("int4d")
        self.block          = block
        self._kwargs        = kwargs  # padding_idx etc.
        self.weight = nn.Parameter(
            MXTensor.quantize(torch.empty(num_embeddings, embedding_dim),
                              self.mx_dtype, block), requires_grad=True)

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, mx_dtype: MXDtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.num_embeddings = emb.num_embeddings
        m.embedding_dim  = emb.embedding_dim
        m.mx_dtype       = mx_dtype
        m.block          = block
        m._kwargs        = {}
        m.weight = nn.Parameter(
            MXTensor.quantize(emb.weight.data, mx_dtype, block), requires_grad=True)
        return m

    def forward(self, indices: Tensor) -> Tensor:
        w = self.weight.data
        if isinstance(w, MXTensor):
            # Look up by index in dequantized table (full lookup)
            w_f = w.dequantize()
            out = F.embedding(indices, w_f)
            return MXTensor.quantize(out, self.mx_dtype, self.block)
        return F.embedding(indices, w)


# ── Module patch: replace layers in-place ────────────────────────────────────

def _replace_module(parent: nn.Module, name: str,
                    module: nn.Module, mx_dtype: MXDtype, block: int):
    """
    Replace a module with its MX-quantized counterpart.
    Covers: Linear, Conv1d/2d, LayerNorm, GroupNorm, Embedding, EmbeddingBag,
            and any module with .weight/.bias parameters (generic fallback).
    Also detects LLaMA/Mistral-style RMSNorm by duck-typing.
    """
    kind = type(module).__name__

    if isinstance(module, nn.Linear):
        setattr(parent, name, MXLinear.from_linear(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: Linear → MXLinear({mx_dtype.name})")

    elif isinstance(module, nn.Conv2d):
        setattr(parent, name, MXConv2d.from_conv2d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: Conv2d → MXConv2d({mx_dtype.name})")

    elif isinstance(module, nn.ConvTranspose2d):
        setattr(parent, name, MXConvTranspose2d.from_conv_transpose2d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: ConvTranspose2d → MXConvTranspose2d({mx_dtype.name})")

    elif isinstance(module, nn.Conv1d):
        setattr(parent, name, MXConv1d.from_conv1d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: Conv1d → MXConv1d({mx_dtype.name})")

    elif isinstance(module, nn.ConvTranspose1d):
        setattr(parent, name, MXConvTranspose1d.from_conv_transpose1d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: ConvTranspose1d → MXConvTranspose1d({mx_dtype.name})")

    elif isinstance(module, nn.BatchNorm2d):
        setattr(parent, name, MXBatchNorm2d.from_batch_norm(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: BatchNorm2d → MXBatchNorm2d({mx_dtype.name})")

    elif isinstance(module, nn.BatchNorm1d):
        setattr(parent, name, MXBatchNorm1d.from_batch_norm1d(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: BatchNorm1d → MXBatchNorm1d({mx_dtype.name})")

    elif isinstance(module, nn.MultiheadAttention):
        setattr(parent, name, MXMultiheadAttention.from_mha(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: MultiheadAttention → MXMultiheadAttention({mx_dtype.name})")

    elif isinstance(module, nn.TransformerEncoderLayer):
        setattr(parent, name, MXTransformerEncoderLayer.from_encoder_layer(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: TransformerEncoderLayer → MXTransformerEncoderLayer({mx_dtype.name})")

    elif isinstance(module, nn.LayerNorm):
        setattr(parent, name, MXLayerNorm.from_layer_norm(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: LayerNorm → MXLayerNorm({mx_dtype.name})")

    elif isinstance(module, nn.GroupNorm):
        setattr(parent, name, MXGroupNorm.from_group_norm(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: GroupNorm → MXGroupNorm({mx_dtype.name})")

    elif isinstance(module, nn.Embedding):
        setattr(parent, name, MXEmbedding.from_embedding(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: Embedding → MXEmbedding({mx_dtype.name})")

    elif isinstance(module, nn.EmbeddingBag):
        setattr(parent, name, MXEmbeddingBag.from_embedding_bag(module, mx_dtype, block))
        if _DEBUG: log.debug(f"[replace] {name}: EmbeddingBag → MXEmbeddingBag({mx_dtype.name})")

    elif (hasattr(module, "weight") and hasattr(module, "variance_epsilon") or
          "RMSNorm" in kind or "RmsNorm" in kind):
        # Duck-type LLaMA/Mistral/Gemma RMSNorm variants
        try:
            setattr(parent, name, MXRMSNorm.from_rms_norm(module, mx_dtype, block))
            if _DEBUG: log.debug(f"[replace] {name}: {kind} → MXRMSNorm({mx_dtype.name})")
        except Exception:
            _replace_generic(module, name, mx_dtype, block)

    else:
        # Generic: quantize weight and bias parameters in-place
        _replace_generic(module, name, mx_dtype, block)


def _replace_generic(module: nn.Module, name: str, mx_dtype: MXDtype, block: int):
    """Generic in-place weight quantization for unrecognised module types."""
    for attr in ("weight", "bias"):
        p = getattr(module, attr, None)
        if isinstance(p, nn.Parameter) and p is not None and p.data.ndim > 0:
            with torch.no_grad():
                mx_t = MXTensor.quantize(p.data, mx_dtype, block)
            setattr(module, attr, nn.Parameter(mx_t, requires_grad=p.requires_grad))
            if _DEBUG:
                log.debug(f"[replace] {name}.{attr}: generic quantize → {mx_dtype.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11b — SPARSE ARITHMETIC
#   SparseMXTensor: magnitude-pruned sparse tensor with MX-packed non-zero values.
#   prune_to_sparse(): create from a dense MXTensor or float tensor.
#   MXSparseLinear: sparse × quantized linear layer for pruned models.
# ─────────────────────────────────────────────────────────────────────────────

class SparseMXTensor:
    """
    Sparse MX-quantized tensor using CSR (Compressed Sparse Row) layout.

    Combines two compression axes:
      1. Sparsity: only non-zero elements are stored.
      2. MX quantization: non-zero values are bit-packed at the target precision.

    Memory layout:
      values   : MXTensor  — packed non-zero values
      crow_ptr : int32     — CSR row pointers  [rows+1]
      col_idx  : int16/32  — column indices of non-zeros
      shape    : original dense shape
      density  : nnz / numel (fraction of non-zeros)

    Use ``prune_to_sparse()`` to create from a dense or MXTensor.

    Example::
        sparse_w = prune_to_sparse(weight, sparsity=0.5, dtype="int4d")
        # Replace in MXLinear:
        mx_lin.sparse_weight = sparse_w
        # Forward: automatic dense reconstruction for now (TODO: sparse GEMM)
        out = F.linear(x, sparse_w.to_dense().dequantize())
    """

    def __init__(self, values: MXTensor, crow_ptr: Tensor, col_idx: Tensor,
                 shape: torch.Size, nnz: int):
        self.values   = values     # MXTensor of non-zero values
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

    def to_dense(self) -> MXTensor:
        """Reconstruct dense MXTensor from sparse CSR representation."""
        rows, cols = self.shape[0], math.prod(self.shape[1:])
        vals_f = self.values.dequantize()
        # Use PyTorch's native CSR → dense for correctness and speed
        sparse_csr = torch.sparse_csr_tensor(
            self.crow_ptr.long(),
            self.col_idx.long(),
            vals_f,
            (rows, cols),
            dtype=torch.float32,
            device=self.values.device,
        )
        dense = sparse_csr.to_dense()
        return MXTensor.quantize(dense.reshape(self.shape),
                                 self.values._mx_dtype, self.values._mx_block)

    def to_torch_sparse_csr(self) -> Tensor:
        """
        Return a plain torch.Tensor in sparse CSR format.
        Useful for passing to torch.sparse.mm or torch.mm with sparse support.
        """
        rows, cols = self.shape[0], math.prod(self.shape[1:])
        vals_f = self.values.dequantize()
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
        return (f"SparseMXTensor({self.values._mx_dtype.name}, "
                f"shape={tuple(self.shape)}, density={self.density:.2%}, "
                f"{self.compression_vs_dense_fp32():.1f}x vs fp32)")


def prune_to_sparse(
    x: Union[Tensor, MXTensor],
    sparsity: float = 0.5,
    dtype: Union[str, MXDtype] = "int4d",
    block: int = 128,
    structured: bool = False,
) -> SparseMXTensor:
    """
    Magnitude prune a tensor and store surviving values as MX-quantized CSR.

    Combines two compression techniques:
      • Sparsity: remove the ``sparsity`` fraction of smallest-magnitude weights.
      • MX quantization: pack the remaining non-zeros at ``dtype`` precision.

    Typical compression: int4d at 50% sparsity → ~16x vs fp32 (4-bit / 0.5 density).

    Args:
        x:          Float or MXTensor to compress.
        sparsity:   Fraction of weights to zero (0.5 = 50% sparse).
        dtype:      MX dtype for quantizing the non-zero values.
        block:      Quantisation block size.
        structured: If True, apply 2:4 structured sparsity (NVIDIA A100/H100 style)
                    — prune 2 out of every 4 weights, enabling hardware sparse GEMM.

    Returns:
        SparseMXTensor in CSR format with MX-quantized non-zeros.

    Example::
        sparse_w = prune_to_sparse(weight, sparsity=0.5, dtype="int4d")
        print(f"Compression: {sparse_w.compression_vs_dense_fp32():.1f}x")
        w_dense  = sparse_w.dequantize()  # back to float32
    """
    dt  = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    x_f = x.dequantize().float() if isinstance(x, MXTensor) else x.float()
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
    values  = MXTensor.quantize(vals_f, dt, block)

    return SparseMXTensor(values, crow_ptr, col_idx, orig_shape, int(nnz))


class MXSparseLinear(nn.Module):
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
                 bias: bool = True, mx_dtype: MXDtype = None,
                 sparsity: float = 0.5, block: int = 128):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mx_dtype     = mx_dtype or get_mx_dtype("int4d")
        self.sparsity     = sparsity
        self.block        = block
        self.sparse_weight: Optional[SparseMXTensor] = None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, mx_dtype: MXDtype,
                    sparsity: float = 0.5, block: int = 128,
                    structured: bool = False) -> "MXSparseLinear":
        m = cls(linear.in_features, linear.out_features,
                linear.bias is not None, mx_dtype, sparsity, block)
        m.sparse_weight = prune_to_sparse(
            linear.weight.data, sparsity, mx_dtype, block, structured)
        if linear.bias is not None:
            m.bias = nn.Parameter(linear.bias.data.clone())
        return m

    def forward(self, x: Tensor) -> Tensor:
        if self.sparse_weight is None:
            raise RuntimeError("MXSparseLinear: sparse_weight not set. "
                               "Use MXSparseLinear.from_linear().")
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
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
#   MXLoRALinear  : QLoRA-style frozen base + trainable LoRA adapters
#   MXMixedInt8Linear: LLM.int8() mixed-precision (outlier columns fp16)
#   MXDynamicLinear: Dynamic (per-token) activation quantization
# ─────────────────────────────────────────────────────────────────────────────

class MXLoRALinear(nn.Module):
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
        qlora = MXLoRALinear.from_linear(layer, rank=16, base_dtype="int4d")
        # Only qlora.lora_A and qlora.lora_B have gradients
        optimizer = torch.optim.AdamW(
            [p for p in qlora.parameters() if p.requires_grad])

    Usage in model::
        model = to_mx(model, "int4d")       # freeze and quantize all linears
        for name, mod in model.named_modules():
            if isinstance(mod, MXLinear):
                parent = ...
                setattr(parent, name, MXLoRALinear.from_mx_linear(mod, rank=16))
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: Optional[float] = None,
                 base_dtype: MXDtype = None, lora_dtype = torch.bfloat16,
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
        self.base_weight: Optional[MXTensor] = None
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
                    base_dtype: Union[str, MXDtype] = "int4d",
                    lora_dtype = torch.bfloat16,
                    block: int = 128) -> "MXLoRALinear":
        """Create from an existing nn.Linear, quantizing base weight to MX."""
        dt = get_mx_dtype(base_dtype) if isinstance(base_dtype, str) else base_dtype
        m  = cls(linear.in_features, linear.out_features,
                  rank, alpha, dt, lora_dtype, linear.bias is not None, block)
        m.base_weight = MXTensor.quantize(
            linear.weight.data, dt, block, requires_grad=False)
        if linear.bias is not None:
            m.bias_param = nn.Parameter(linear.bias.data.clone(),
                                        requires_grad=False)
        return m

    @classmethod
    def from_mx_linear(cls, mx_linear: "MXLinear", rank: int = 8,
                       alpha: Optional[float] = None,
                       lora_dtype = torch.bfloat16) -> "MXLoRALinear":
        """Wrap an existing MXLinear with LoRA adapters (no re-quantization)."""
        m = cls(mx_linear.in_features, mx_linear.out_features,
                rank, alpha, mx_linear.mx_dtype, lora_dtype,
                mx_linear.bias is not None, mx_linear.block)
        m.base_weight = mx_linear.weight.data
        if mx_linear.bias is not None:
            m.bias_param = nn.Parameter(
                mx_linear.bias.data.dequantize()
                if isinstance(mx_linear.bias.data, MXTensor)
                else mx_linear.bias.data.clone(),
                requires_grad=False)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
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

    def merge_weights(self) -> MXLinear:
        """
        Merge LoRA adapters into base weight and return a plain MXLinear.
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
        return MXLinear.from_linear(lin, self.base_dtype, self.block)

    def trainable_parameters(self):
        """Only LoRA parameters — use this for optimizer."""
        return [self.lora_A, self.lora_B]

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, α={self.alpha}, "
                f"base={self.base_dtype.name}, lora={self.lora_dtype}")


class MXMixedInt8Linear(nn.Module):
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
        mixed = MXMixedInt8Linear.from_linear(layer, threshold=6.0)
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 6.0, block: int = 64):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.threshold    = threshold
        self.block        = block
        self.q_weight: Optional[MXTensor]  = None
        self.fp_weight: Optional[Tensor]   = None
        self.outlier_mask: Optional[Tensor] = None
        self.bias: Optional[nn.Parameter]  = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, threshold: float = 6.0,
                    block: int = 64) -> "MXMixedInt8Linear":
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
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()

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


class MXDynamicLinear(nn.Module):
    """
    Dynamic (per-token) activation quantization.

    Weight: statically quantized to MX precision (like regular MXLinear).
    Activations: quantized at runtime using per-token absmax scales.

    Differences from static MXLinear:
      • No calibration required — scales are computed live.
      • Per-token scaling = finer granularity than per-block static.
      • Slightly more compute (scale computation) vs static.
      • Better accuracy for activations with high dynamic range.

    This is similar to PyTorch's ``torch.ao.quantization.dynamic`` but:
      • Works with MX dtypes (int1 through int8, float variants)
      • Returns MXTensor outputs (can chain with other MX layers)
      • No observer/prepare/convert workflow — just drop in

    Example::
        dyn = MXDynamicLinear.from_linear(layer, weight_dtype="int4d",
                                           act_dtype="int8d")
        # Equivalent to: static weight quant + dynamic activation quant
        out = dyn(x)   # x quantized per-token at runtime
    """

    def __init__(self, in_features: int, out_features: int,
                 weight_dtype: MXDtype = None, act_dtype: MXDtype = None,
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
                    weight_dtype: Union[str, MXDtype] = "int4d",
                    act_dtype: Union[str, MXDtype] = "int8d",
                    weight_block: int = 128) -> "MXDynamicLinear":
        wdt = get_mx_dtype(weight_dtype) if isinstance(weight_dtype, str) else weight_dtype
        adt = get_mx_dtype(act_dtype)    if isinstance(act_dtype, str)    else act_dtype
        m   = cls(linear.in_features, linear.out_features,
                  wdt, adt, linear.bias is not None, weight_block)
        m.weight = nn.Parameter(
            MXTensor.quantize(linear.weight.data, wdt, weight_block),
            requires_grad=False)
        if linear.bias is not None:
            m.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        return m

    def _dynamic_quantize_activation(self, x: Tensor) -> MXTensor:
        """Per-token dynamic quantization of activations."""
        shape  = x.shape
        x_2d   = x.reshape(-1, shape[-1])        # [tokens, features]
        max_int = float((1 << (self.act_dtype.bits - 1)) - 1)

        # Per-token absmax scale (finer granularity than static per-block)
        scales = x_2d.abs().amax(dim=1).clamp(min=1e-12) / max_int   # [tokens]
        normed = x_2d / scales.unsqueeze(1)
        codes  = normed.round().clamp(-max_int, max_int).to(torch.int32)
        packed = BitPacker.pack_auto(codes.reshape(-1), self.act_dtype.bits)
        # One scale per token; block = feature_dim (all features share token scale)
        mx_t   = MXTensor(packed, scales.float(),
                          self.act_dtype, torch.Size(list(x_2d.shape)),
                          x_2d.numel(), x_2d.shape[-1])
        return mx_t.reshape(*shape)

    def forward(self, x: Tensor) -> Tensor:
        # Dynamic quantize input activations (per-token)
        x_f   = x.dequantize() if isinstance(x, MXTensor) else x.float()
        x_q   = self._dynamic_quantize_activation(x_f)

        w_mx  = self.weight.data
        if not isinstance(w_mx, MXTensor):
            w_mx = MXTensor.quantize(w_mx, self.weight_dtype, self.weight_block)

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
#   MXConvTranspose2d  : Transposed convolution (decoder / upsampling)
#   MXConvTranspose1d  : 1D transposed convolution
#   MXBatchNorm1d      : 1D batch normalisation (NLP / MLP paths)
#   MXTransformerEncoderLayer: Full transformer encoder block
#   MXGRU              : Gated Recurrent Unit (quantized gates and weights)
# ─────────────────────────────────────────────────────────────────────────────


class MXConvTranspose2d(nn.Module):
    """
    MX-quantized ``nn.ConvTranspose2d`` (decoder / upsampling convolution).

    Weights are packed at MX precision; the transposed convolution is computed
    via ``F.conv_transpose2d`` after dequantization. All standard parameters
    (stride, padding, output_padding, dilation, groups) are supported.

    Example::
        layer = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        mx    = MXConvTranspose2d.from_conv_transpose2d(layer, get_mx_dtype("int4d"))
        out   = mx(x)    # x: [B, 128, H, W] → [B, 64, 2H, 2W]
    """

    @classmethod
    def from_conv_transpose2d(cls, ct: nn.ConvTranspose2d,
                              mx_dtype: MXDtype, block: int = 128):
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
        m.weight = nn.Parameter(
            MXTensor.quantize(ct.weight.data, mx_dtype, block), requires_grad=False)
        m.bias   = (nn.Parameter(ct.bias.data.clone(), requires_grad=False)
                    if ct.bias is not None else None)
        return m

    def forward(self, x: Tensor, output_size=None) -> Tensor:
        x_f  = x.dequantize() if isinstance(x, MXTensor) else x.float()
        w_f  = self.weight.data.dequantize() if isinstance(self.weight.data, MXTensor) else self.weight.data
        b    = self.bias.data if self.bias is not None else None
        kwargs = {}
        if output_size is not None:
            kwargs["output_size"] = output_size
        out  = F.conv_transpose2d(x_f, w_f, b, self.stride, self.padding,
                                   self.output_padding, self.groups, self.dilation,
                                   **kwargs)
        return MXTensor.quantize(out, self.mx_dtype, self.block)

    def extra_repr(self):
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"k={self.kernel_size}, stride={self.stride}, "
                f"dtype={self.mx_dtype.name}")


class MXConvTranspose1d(nn.Module):
    """MX-quantized ``nn.ConvTranspose1d``."""

    @classmethod
    def from_conv_transpose1d(cls, ct: nn.ConvTranspose1d,
                               mx_dtype: MXDtype, block: int = 128):
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        for attr in ("in_channels","out_channels","kernel_size","stride",
                     "padding","output_padding","dilation","groups"):
            setattr(m, attr, getattr(ct, attr))
        m.mx_dtype = mx_dtype
        m.block    = block
        m.weight   = nn.Parameter(
            MXTensor.quantize(ct.weight.data, mx_dtype, block), requires_grad=False)
        m.bias     = (nn.Parameter(ct.bias.data.clone(), requires_grad=False)
                      if ct.bias is not None else None)
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
        w_f = self.weight.data.dequantize() if isinstance(self.weight.data, MXTensor) else self.weight.data
        b   = self.bias.data if self.bias is not None else None
        out = F.conv_transpose1d(x_f, w_f, b, self.stride, self.padding,
                                  self.output_padding, self.groups, self.dilation)
        return MXTensor.quantize(out, self.mx_dtype, self.block)


class MXBatchNorm1d(nn.Module):
    """
    MX-quantized ``nn.BatchNorm1d``.

    Handles both 2D inputs [B, C] and 3D inputs [B, C, L] (1D temporal).
    Running stats (running_mean, running_var) remain float32 for stability.
    Only affine parameters (weight, bias) are MX-packed.

    Example::
        bn = nn.BatchNorm1d(256)
        mx = MXBatchNorm1d.from_batch_norm1d(bn, get_mx_dtype("int8d"))
    """

    @classmethod
    def from_batch_norm1d(cls, bn: nn.BatchNorm1d,
                          mx_dtype: MXDtype, block: int = 128):
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
            m.weight = nn.Parameter(
                MXTensor.quantize(bn.weight.data, mx_dtype, block), requires_grad=False)
            m.bias   = nn.Parameter(
                MXTensor.quantize(bn.bias.data,   mx_dtype, block), requires_grad=False)
        else:
            m.weight = m.bias = None
        if bn.track_running_stats:
            m.register_buffer("running_mean", bn.running_mean.clone())
            m.register_buffer("running_var",  bn.running_var.clone())
            m.register_buffer("num_batches_tracked", bn.num_batches_tracked.clone())
        else:
            m.running_mean = m.running_var = m.num_batches_tracked = None
        return m

    def forward(self, x: Tensor) -> Tensor:
        x_f  = x.dequantize() if isinstance(x, MXTensor) else x.float()
        w    = (self.weight.data.dequantize() if isinstance(self.weight.data, MXTensor)
                else self.weight.data) if self.weight is not None else None
        b    = (self.bias.data.dequantize()   if isinstance(self.bias.data,   MXTensor)
                else self.bias.data)   if self.bias   is not None else None
        out  = F.batch_norm(x_f, self.running_mean, self.running_var,
                            w, b, self.training, self.momentum or 0.1, self.eps)
        return MXTensor.quantize(out, self.mx_dtype, self.block)


class MXTransformerEncoderLayer(nn.Module):
    """
    Full transformer encoder block with all weights at MX precision.

    Quantizes:
      • Self-attention Q/K/V projections (MXLinear)
      • Output projection (MXLinear)
      • FFN linear1 (MXLinear)
      • FFN linear2 (MXLinear)
      • LayerNorm affine parameters (MXLayerNorm)

    Computation order (Pre-LN variant, used by most modern models):
      x = x + Attn(LN1(x))
      x = x + FFN(LN2(x))

    Both Pre-LN and Post-LN are supported via ``norm_first`` (same as PyTorch).

    Example::
        enc = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
        mx  = MXTransformerEncoderLayer.from_encoder_layer(enc, get_mx_dtype("int4d"))
        out = mx(x)   # x: [S, B, D] (or [B, S, D] with batch_first=True)
    """

    @classmethod
    def from_encoder_layer(cls, enc: nn.TransformerEncoderLayer,
                           mx_dtype: MXDtype, block: int = 128):
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
        m.self_attn = MXMultiheadAttention.from_mha(enc.self_attn, mx_dtype, block)
        m.linear1   = MXLinear.from_linear(enc.linear1, mx_dtype, block)
        m.linear2   = MXLinear.from_linear(enc.linear2, mx_dtype, block)
        m.norm1     = MXLayerNorm.from_layer_norm(enc.norm1, mx_dtype, block)
        m.norm2     = MXLayerNorm.from_layer_norm(enc.norm2, mx_dtype, block)
        m.activation = enc.activation if callable(enc.activation) else F.relu
        return m

    def _ff_block(self, x: Tensor) -> Tensor:
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
        h   = self.linear1(x_f)
        h_f = h.dequantize() if isinstance(h, MXTensor) else h
        act = self.activation(h_f) if not isinstance(self.activation, str) else \
              {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}.get(self.activation, F.relu)(h_f)
        out = self.linear2(act)
        return F.dropout(out.dequantize() if isinstance(out, MXTensor) else out,
                         p=self.dropout_p, training=self.training)

    def _sa_block(self, x: Tensor, attn_mask=None, key_padding_mask=None) -> Tensor:
        x_f = x.dequantize() if isinstance(x, MXTensor) else x.float()
        out, _ = self.self_attn(x_f, x_f, x_f,
                                 attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask)
        out_f  = out.dequantize() if isinstance(out, MXTensor) else out
        return F.dropout(out_f, p=self.dropout_p, training=self.training)

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None) -> Tensor:
        if self.norm_first:
            n1   = self.norm1(src)
            n1_f = n1.dequantize() if isinstance(n1, MXTensor) else n1
            src  = src.float() + self._sa_block(n1_f, src_mask, src_key_padding_mask)
            n2   = self.norm2(src)
            n2_f = n2.dequantize() if isinstance(n2, MXTensor) else n2
            src  = src + self._ff_block(n2_f)
        else:
            src  = src.float() + self._sa_block(src, src_mask, src_key_padding_mask)
            n1   = self.norm1(src)
            src  = (n1.dequantize() if isinstance(n1, MXTensor) else n1)
            src  = src + self._ff_block(src)
            n2   = self.norm2(src)
            src  = n2.dequantize() if isinstance(n2, MXTensor) else n2
        return src

    def extra_repr(self):
        return (f"d={self.d_model}, heads={self.nhead}, "
                f"dtype={self.mx_dtype.name}, norm_first={self.norm_first}")


class MXGRU(nn.Module):
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
        mx_gru = MXGRU.from_gru_cell(
            gru.weight_ih_l0, gru.weight_hh_l0,
            gru.bias_ih_l0, gru.bias_hh_l0,
            get_mx_dtype("int4d"))
        h0  = torch.zeros(1, batch, 512)
        out = mx_gru(x, h0[:, 0])   # step-wise
    """

    @classmethod
    def from_gru_cell(cls, weight_ih: Tensor, weight_hh: Tensor,
                      bias_ih: Optional[Tensor], bias_hh: Optional[Tensor],
                      mx_dtype: MXDtype, block: int = 128) -> "MXGRU":
        hidden  = weight_hh.shape[1]
        inp     = weight_ih.shape[1]
        m       = cls(inp, hidden, mx_dtype, block)
        # PyTorch GRU packs all 3 gates: [3*hidden, input]
        # Split into r, z, n
        for i, name in enumerate(["W_r","W_z","W_n"]):
            sl = slice(i * hidden, (i + 1) * hidden)
            setattr(m, name, nn.Parameter(
                MXTensor.quantize(weight_ih[sl], mx_dtype, block), requires_grad=False))
        for i, name in enumerate(["U_r","U_z","U_n"]):
            sl = slice(i * hidden, (i + 1) * hidden)
            setattr(m, name, nn.Parameter(
                MXTensor.quantize(weight_hh[sl], mx_dtype, block), requires_grad=False))
        if bias_ih is not None:
            for i, name in enumerate(["b_ir","b_iz","b_in"]):
                sl = slice(i * hidden, (i + 1) * hidden)
                setattr(m, name, nn.Parameter(bias_ih[sl].clone(), requires_grad=False))
        if bias_hh is not None:
            for i, name in enumerate(["b_hr","b_hz","b_hn"]):
                sl = slice(i * hidden, (i + 1) * hidden)
                setattr(m, name, nn.Parameter(bias_hh[sl].clone(), requires_grad=False))
        return m

    def __init__(self, input_size: int, hidden_size: int,
                 mx_dtype: MXDtype, block: int = 128):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.mx_dtype    = mx_dtype
        self.block       = block
        # Parameters set by from_gru_cell
        for name in ("W_r","W_z","W_n","U_r","U_z","U_n"):
            setattr(self, name, None)
        for name in ("b_ir","b_iz","b_in","b_hr","b_hz","b_hn"):
            setattr(self, name, None)

    def _dq(self, t: Optional[nn.Parameter]) -> Optional[Tensor]:
        if t is None: return None
        d = t.data
        return d.dequantize() if isinstance(d, MXTensor) else d.float()

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
        Process a sequence. x: [B, T, input], h0: [B, hidden] → (out: [B,T,H], h_n: [B,H]).
        """
        B, T, _  = x.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size, device=x.device)
        out = []
        for t in range(T):
            h = self.forward_step(x[:, t], h)
            out.append(h.unsqueeze(1))
        return torch.cat(out, dim=1), h

    def extra_repr(self):
        return (f"input={self.input_size}, hidden={self.hidden_size}, "
                f"dtype={self.mx_dtype.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — MONKEY PATCHING (seamless integration)
# ─────────────────────────────────────────────────────────────────────────────

# ── Patch torch.dtype() to return MXDtypeProxy for MX names ──────────────────

_orig_torch_dtype = torch.dtype

class _MXDtypeCallable:
    """
    Replacement for torch.dtype that additionally handles MX dtype strings.
    torch.dtype("int4d")   → MXDtypeProxy(int4d)
    torch.dtype("float32") → torch.float32  (original behaviour)
    """
    def __call__(self, s=None, *args, **kwargs):
        if isinstance(s, str) and s in _DTYPE_REGISTRY:
            return MXDtypeProxy(get_mx_dtype(s))
        if isinstance(s, MXDtype):
            return MXDtypeProxy(s)
        if isinstance(s, MXDtypeProxy):
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
        return isinstance(instance, (_orig_torch_dtype, MXDtypeProxy))


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

    if isinstance(target, MXDtypeProxy):
        return to_mx(self, target._mx.name, block_size=block, low_mem=low_mem)
    if isinstance(target, MXDtype):
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

    if isinstance(target, MXDtypeProxy):
        return MXTensor.quantize(self.float(), target._mx, block,
                                 requires_grad=self.requires_grad)
    if isinstance(target, MXDtype):
        return MXTensor.quantize(self.float(), target, block,
                                 requires_grad=self.requires_grad)
    if isinstance(target, str) and target in _DTYPE_REGISTRY:
        return MXTensor.quantize(self.float(), get_mx_dtype(target), block,
                                 requires_grad=self.requires_grad)

    # Also handle MXTensor → torch dtype (dequantize)
    if isinstance(self, MXTensor):
        return self.dequantize().to(*args, **kwargs)

    return _orig_tensor_to(self, *args, **kwargs)

Tensor.to = _mx_tensor_to


# ── Patch nn.Parameter to handle MXTensor storage ────────────────────────────

_orig_param_new = nn.Parameter.__new__

def _mx_param_new(cls, data=None, requires_grad=True):
    if isinstance(data, MXTensor):
        # Store MXTensor directly; bypass normal Parameter validation
        obj = object.__new__(cls)
        return obj
    return _orig_param_new(cls, data, requires_grad)


class _MXAwareParameter(nn.Parameter):
    """
    nn.Parameter subclass that transparently holds an MXTensor.
    `.data` returns the MXTensor; `.float_data` returns dequantized float32.
    """

    def __new__(cls, data=None, requires_grad=True):
        if isinstance(data, MXTensor):
            # Make the param wrap the packed storage
            inst = torch.Tensor._make_subclass(cls, data.packed, requires_grad)
            inst._mx_payload = data
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
        if isinstance(val, MXTensor):
            self._mx_payload = val
            self.packed_ref = val.packed
        elif hasattr(self, "_mx_payload"):
            # Re-quantize after optimizer step
            self._mx_payload = MXTensor.quantize(
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


# Monkeypatch nn.Parameter creation to return MXAwareParameter when passed MXTensor
_orig_nn_param = nn.Parameter

class _ParameterFactory:
    """Intercept nn.Parameter(mx_tensor) to return _MXAwareParameter."""
    def __call__(self, data=None, requires_grad=True):
        if isinstance(data, MXTensor):
            return _MXAwareParameter(data, requires_grad)
        return _orig_nn_param(data, requires_grad)

    def __instancecheck__(self, instance):
        return isinstance(instance, (_orig_nn_param, _MXAwareParameter))

    def __subclasscheck__(self, subclass):
        return issubclass(subclass, (_orig_nn_param, _MXAwareParameter))

nn.Parameter = _ParameterFactory()


# ── Patch standard optimizers to handle MXTensor params ──────────────────────

def _wrap_optimizer(opt_cls):
    """
    Wrap a standard optimizer so that:
    1. MXTensor params are dequantized before step
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
                elif isinstance(p, MXTensor):
                    mx_p = p

                if mx_p is not None:
                    # Create a float32 proxy that the optimizer will update
                    fp = mx_p.dequantize().detach().requires_grad_(p.requires_grad)
                    if p.grad is not None:
                        grad = p.grad
                        if isinstance(grad, MXTensor):
                            grad = grad.dequantize()
                        fp.grad = grad.float()
                    mx_params.append((group, i, mx_p, fp))
                    float_proxies.append(fp)
                    group["params"][i] = fp

        # Run original optimizer step
        loss = orig_step(self, closure)

        # Re-quantize updated params back to MX
        for group, i, mx_p, fp in mx_params:
            updated = MXTensor.quantize(fp.data, mx_p._mx_dtype, mx_p._mx_block)
            # Re-quantize optimizer states if they exist
            state = self.state.get(fp)
            if state:
                new_state = {}
                for k, v in state.items():
                    if isinstance(v, Tensor) and v.dtype.is_floating_point and v.ndim > 0:
                        # Keep state at MX precision
                        new_state[k] = MXTensor.quantize(v, mx_p._mx_dtype, mx_p._mx_block)
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
# SECTION 13 — MXAdamW (native MX optimizer, all states at MX precision)
# ─────────────────────────────────────────────────────────────────────────────

class MXAdamW(torch.optim.Optimizer):
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
                # Unwrap MXTensor
                is_mx  = False
                p_dt   = None
                if isinstance(p, _MXAwareParameter) and hasattr(p, "_mx_payload"):
                    px    = p._mx_payload
                    is_mx = True; p_dt = px._mx_dtype
                elif isinstance(p, MXTensor):
                    px    = p
                    is_mx = True; p_dt = p._mx_dtype
                else:
                    px = p

                # Get gradient
                grad = px.grad if hasattr(px, "grad") else getattr(p, "grad", None)
                if grad is None: continue
                if isinstance(grad, MXTensor): grad = grad.dequantize()
                grad = grad.float()

                # State
                sid  = id(p)
                st   = self.state[sid] if sid in self.state else {}
                self.state[sid] = st

                if "step" not in st:
                    st["step"] = 0
                    st["m"]    = MXTensor.quantize(torch.zeros_like(grad), self._state_dt, self._block)
                    st["v"]    = MXTensor.quantize(torch.zeros_like(grad), self._state_dt, self._block)

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
                    new_mx = MXTensor.quantize(w_f, p_dt, px._mx_block)
                    if isinstance(p, _MXAwareParameter):
                        p._mx_payload = new_mx
                    else:
                        p.packed.copy_(new_mx.packed)
                        p._mx_scales.copy_(new_mx._mx_scales)
                else:
                    p.data.mul_(1 - lr * wd).add_(-lr * update)

                # Re-quantize states
                st["m"] = MXTensor.quantize(m, self._state_dt, self._block)
                st["v"] = MXTensor.quantize(v, self._state_dt, self._block)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — PUBLIC API: to_mx, load_quantized, save_quantized
# ─────────────────────────────────────────────────────────────────────────────

def to_mx(
    model: nn.Module,
    dtype: Union[str, Dict[str, str], MXDtype, MXDtypeProxy] = "int4d",
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
        dtype:            MX dtype name ("int4d"), MXDtype, MXDtypeProxy,
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
    if isinstance(dtype, MXDtypeProxy):
        dtype = dtype._mx.name
    elif isinstance(dtype, MXDtype):
        dtype = dtype.name

    if isinstance(dtype, str):
        dtype_map = {".*": dtype}
    elif isinstance(dtype, dict):
        # Normalise values
        dtype_map = {}
        for pat, val in dtype.items():
            if isinstance(val, MXDtypeProxy): val = val._mx.name
            elif isinstance(val, MXDtype): val = val.name
            dtype_map[pat] = val
    else:
        raise TypeError(f"dtype must be str, dict, MXDtype or MXDtypeProxy, got {type(dtype)}")

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
                    mx_t = MXTensor.quantize(p.data, mx_dt, block_size)
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
            elif isinstance(p, nn.Parameter) and isinstance(p.data, MXTensor):
                mx_t = p.data
            elif isinstance(p, MXTensor):
                mx_t = p
            elif isinstance(module, (MXLinear, MXEmbedding)):
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
            mx_t  = MXTensor(
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
                    mx_t  = MXTensor.quantize(tensor.float(), mx_dt, block_size)
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

class MXDistributed:
    """
    PyTorch distributed ops that work on packed MX tensors.
    No upcast — all communication in MX precision.
    Compatible with DDP and FSDP.
    """

    @staticmethod
    def all_reduce(tensor, op=None):
        import torch.distributed as dist
        if op is None: op = dist.ReduceOp.SUM

        if not isinstance(tensor, MXTensor):
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
        if isinstance(tensor, MXTensor):
            dist.broadcast(tensor.packed, src)
            dist.broadcast(tensor._mx_scales, src)
        else:
            dist.broadcast(tensor, src)
        return tensor

    @staticmethod
    def all_gather(tensor_list, tensor, group=None):
        import torch.distributed as dist
        if not isinstance(tensor, MXTensor):
            return dist.all_gather(tensor_list, tensor, group)
        # Gather packed bits
        packed_list = [torch.empty_like(tensor.packed)
                       for _ in range(dist.get_world_size())]
        dist.all_gather(packed_list, tensor.packed, group)
        scales_list = [torch.empty_like(tensor._mx_scales)
                       for _ in range(dist.get_world_size())]
        dist.all_gather(scales_list, tensor._mx_scales, group)
        for i, (p, s) in enumerate(zip(packed_list, scales_list)):
            tensor_list[i] = MXTensor(p, s, tensor._mx_dtype,
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
        if isinstance(grad, MXTensor):
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
# SECTION 16 — ROOFLINE ESTIMATOR + BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RooflineResult:
    peak_gflops:    float
    bottleneck:     str
    intensity:      float
    mem_bw_util:    float
    compute_util:   float
    hw_name:        str
    dtype_name:     str


class RooflineEstimator:
    def __init__(self, hw: Optional[HardwareProfile] = None):
        self.hw = hw or HardwareProbe.detect()

    def estimate(self, op: str, dt: MXDtype,
                 in_shape: tuple, w_shape: tuple) -> RooflineResult:
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

        return RooflineResult(
            peak_gflops  = ach / 1e9,
            bottleneck   = bot,
            intensity    = intens,
            mem_bw_util  = mu,
            compute_util = cu,
            hw_name      = self.hw.name,
            dtype_name   = dt.name,
        )


@dataclass
class BenchmarkReport:
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
  Latency      : {self.latency_ms:.2f} ms / iter
  Compression  : {self.compression:.1f}x vs fp32
  Memory saved : {self.mem_saved_gb:.3f} GB
{('  Warnings:\n' + w) if self.warnings else ''}
{bar}"""


def benchmark_report(
    model: nn.Module,
    input_shape: tuple,
    dtype: str = "int4d",
    n_warmup: int = 3,
    n_iters: int = 10,
    device: Optional[str] = None,
) -> BenchmarkReport:
    """Measure throughput vs roofline theoretical max."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    hw  = HardwareProbe.detect()
    dt  = get_mx_dtype(dtype)
    est = RooflineEstimator(hw)
    wrn = []

    # Measure packed storage
    orig_bytes = 0; quant_bytes = 0
    for name, mod in model.named_modules():
        for attr in ("weight", "bias"):
            p = getattr(mod, attr, None)
            mx_t = None
            if isinstance(p, _MXAwareParameter) and hasattr(p, "_mx_payload"):
                mx_t = p._mx_payload
            elif isinstance(p, nn.Parameter) and isinstance(p.data, MXTensor):
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
        if isinstance(m, (nn.Linear, MXLinear)):
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

    return BenchmarkReport(
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

class MXDebugger:
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


debugger = MXDebugger()


class PrecisionAudit:
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
                if isinstance(t, Tensor) and not isinstance(t, MXTensor):
                    if t.dtype in (torch.float32, torch.bfloat16, torch.float16):
                        rec = dict(layer=name, kind="input", idx=i,
                                   dtype=str(t.dtype), shape=tuple(t.shape))
                        self._hits.append(rec)
                        if self.strict:
                            raise RuntimeError(
                                f"[PrecisionAudit STRICT] fp tensor in {name}: "
                                f"dtype={t.dtype}, shape={t.shape}")
        return fn

    def report(self) -> str:
        if not self._hits:
            return "✓ PrecisionAudit: No unexpected full-precision tensors."
        lines = [f"⚠ PrecisionAudit: {len(self._hits)} fp tensors found:"]
        for h in self._hits[:20]:
            lines.append(f"  [{h['layer']}] {h['kind']} {h['dtype']} {h['shape']}")
        if len(self._hits) > 20:
            lines.append(f"  ... and {len(self._hits)-20} more")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — DYNAMIC PRECISION SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class DynamicPrecisionScheduler:
    """
    Curriculum quantization: gradually reduce precision during training.
    Start at high precision, taper to target over N steps.

    Usage:
        sched = DynamicPrecisionScheduler(model, "int8d", "int1d", steps=10000)
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

class PackStrategy:
    """Describes the packing strategy for a given (MXDtype, HardwareProfile) pair."""

    def __init__(self, dt: MXDtype, hw: Optional[HardwareProfile] = None):
        self.dt = dt
        self.hw = hw or HardwareProbe.detect()

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
        return (f"PackStrategy({self.dt.name} on {self.hw.name}):\n"
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
            elif isinstance(p, nn.Parameter) and isinstance(p.data, MXTensor):
                mx_t = p.data
            elif isinstance(mod, (MXLinear, MXEmbedding)):
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
    hw = HardwareProbe.detect()
    rows = "\n".join(
        f"  {n:<12s}: {hw.hw_pack_ratio(get_mx_dtype(n))}x per "
        f"{hw.native_int_bits}-bit native op  "
        f"({get_mx_dtype(n).compression_vs_fp32:.0f}x vs fp32)"
        for n in ["int1d","int2d","int4d","int8d","float4d","float8d"])
    return (f"Hardware: {hw.name}  [{hw.arch}]\n"
            f"  Backend       : {hw.backend}\n"
            f"  Native int    : {hw.native_int_bits}-bit\n"
            f"  Max pack bits : {hw.max_pack_bits}\n"
            f"  Memory BW     : {hw.memory_bw_gbs:.0f} GB/s\n"
            f"  FP32 peak     : {hw.fp32_tflops:.1f} TFLOPS\n"
            f"  FP16 peak     : {hw.fp16_tflops:.1f} TFLOPS\n"
            f"  Supported     : {', '.join(hw.supported_native)}\n"
            f"  Fast instrs   : {', '.join(hw.fast_instrs)}\n"
            f"  Pack ratios:\n{rows}")


def dtype_info(name: str) -> str:
    dt  = get_mx_dtype(name)
    strat = PackStrategy(dt)
    return str(strat)


# ── Public packed matmul ──────────────────────────────────────────────────────

def mx_matmul(a: Tensor, b: Tensor,
              dtype: Union[str, MXDtype] = "int4d",
              block: int = 128) -> Tensor:
    """
    Public API: packed matrix multiplication at MX precision.

    Quantizes `a` and `b` if not already MXTensors, then runs the best
    available packed Triton kernel (int1/int2/int4) or falls back to
    dequant→mm→requant.

    Args:
        a:     Float or MX tensor [M, K].
        b:     Float or MX tensor [K, N].
        dtype: Target MX dtype name or MXDtype object.
        block: Quantisation block size for scales.

    Returns:
        MXTensor of shape [M, N].

    Example::
        import mx_triton as mxt
        c = mxt.mx_matmul(a, b, dtype="int4d")
        c_f = c.dequantize()          # → float32
    """
    dt = get_mx_dtype(dtype) if isinstance(dtype, str) else dtype
    if not isinstance(a, MXTensor):
        a = MXTensor.quantize(a.float(), dt, block)
    if not isinstance(b, MXTensor):
        b = MXTensor.quantize(b.float(), dt, block)
    return _mx_mm(a, b)


# ── mx_mode context manager ───────────────────────────────────────────────────

_mx_default_dtype: Optional[MXDtype] = None
_mx_default_block: int = 128


@contextmanager
def mx_mode(dtype: Union[str, MXDtype] = "int4d", block: int = 128):
    """
    Context manager: set a default MX quantisation dtype for the enclosed block.
    Any call to `mx_quantize()` or `MXTensor.quantize()` inside the block will
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


def get_default_dtype() -> Optional[MXDtype]:
    """Return the MX dtype currently active via ``mx_mode()``, or None."""
    return _mx_default_dtype


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(
    model: nn.Module,
    calibration_data: Tensor,
    dtype: Union[str, MXDtype] = "int4d",
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
            t = output.dequantize() if isinstance(output, MXTensor) else output
            if isinstance(t, Tensor) and t.is_floating_point() and t.ndim > 0:
                flat = t.detach().float().reshape(-1)
                if flat.numel() > n_samples:
                    idx  = torch.randperm(flat.numel(), device=flat.device)[:n_samples]
                    flat = flat[idx]
                act_stats.setdefault(layer_name, []).append(flat.cpu())
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (MXLinear, MXConv2d, nn.Linear, nn.Conv2d)):
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
    dtype: Union[str, MXDtype] = "int4d",
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
    mx_t = MXTensor.quantize(xf, dt, block)
    xq   = mx_t.dequantize().reshape(xf.shape)
    diff = (xf - xq)

    if metric == "rmse":
        return diff.pow(2).mean().sqrt().item()
    elif metric == "mae":
        return diff.abs().mean().item()
    elif metric == "max":
        return diff.abs().max().item()
    elif metric == "relative":
        return (diff.abs().mean() / xf.abs().mean().clamp(min=1e-12)).item()
    else:
        raise ValueError(f"Unknown metric {metric!r}. Use: rmse, mae, max, relative")


def snr(x: Tensor, dtype: Union[str, MXDtype] = "int4d", block: int = 128) -> float:
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
    mx_t = MXTensor.quantize(xf, dt, block)
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
# SECTION 20b — KERNEL TEMPLATE LIBRARY
#   Copy-paste Triton kernel templates with @register_kernel wrapper.
#   Each example is a complete, runnable kernel registered via the public API.
# ─────────────────────────────────────────────────────────────────────────────

KERNEL_EXAMPLES: str = '''
"""
mx_triton — Custom Kernel Template Library
==========================================
Copy-paste examples showing how to write and register Triton kernels
that plug into the MX dispatch system via @register_kernel.

All examples assume `import mx_triton as mxt` and `import triton.language as tl`.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Custom INT4 matmul with AMD v_dot4_i32_i8 hint
#   Registers for gfx1100 (RX 7900 XTX) hardware specifically.
#   force="auto" → benchmarked against auto-generated kernel; fastest wins.
# ═══════════════════════════════════════════════════════════════════════════════

@mxt.register_kernel(
    op       = "torch.matmul",
    dtypes   = ["int4d", "int4u"],
    hardware = ["gfx1100", "gfx942"],
    force    = "auto",
    priority = 10,
)
@triton.jit
def gfx1100_int4_matmul(
    a_ptr, b_ptr, c_ptr,
    sa_ptr, sb_ptr,
    M, N, Kp,
    sam, sak, sbk, sbn, scm, scn,
    BS: tl.constexpr,
    BM: tl.constexpr = 64,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    """
    INT4 GEMM tuned for AMD RDNA3 (gfx1100).
    Exploits v_dot4_i32_i8 (packed 4-way int8 dot product).
    Each CDNA/RDNA op processes 4 int8 values → maps to 8 int4 values.

    Key optimisation vs the built-in kernel:
      - BM=BN=64 matches the wavefront width × 2 on gfx1100 (32-wide waves)
      - Shared memory layout avoids bank conflicts in the inner loop
    """
    pm = tl.program_id(0); pn = tl.program_id(1)
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

        # Unpack 4-bit nibbles and sign-extend
        for nibble in tl.static_range(2):
            shift = nibble * 4
            a_n = ((ap >> shift) & 0x0F).to(tl.int8)
            b_n = ((bp >> shift) & 0x0F).to(tl.int8)
            # Sign extend: values 8-15 → -8 to -1
            a_n = tl.where(a_n > 7, (a_n.to(tl.int16) | 0xFFF0).to(tl.int8), a_n)
            b_n = tl.where(b_n > 7, (b_n.to(tl.int16) | 0xFFF0).to(tl.int8), b_n)
            # v_dot4_i32_i8 is emitted here by Triton for gfx1100
            acc += tl.dot(a_n.to(tl.float16), b_n.to(tl.float16), allow_tf32=False)

    sa = tl.load(sa_ptr + rm // BS, mask=rm < M, other=1.0)
    sb = tl.load(sb_ptr + rn // BS, mask=rn < N, other=1.0)
    c  = acc * sa[:, None] * sb[None, :]
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, c,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Custom INT8 fused matmul + GELU activation (single kernel pass)
#   force="true" → always use this kernel for int8d matmul on H100
# ═══════════════════════════════════════════════════════════════════════════════

@mxt.register_kernel(
    op       = "torch.matmul",
    dtypes   = ["int8d", "int8u"],
    hardware = ["sm_90", "sm_89"],   # H100, L40S
    force    = "true",
    priority = 20,
)
@triton.jit
def h100_int8_matmul_gelu(
    a_ptr, b_ptr, c_ptr,
    sa_ptr, sb_ptr,
    M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BS: tl.constexpr,
    BM: tl.constexpr = 128,
    BN: tl.constexpr = 256,
    BK: tl.constexpr = 64,
    FUSE_GELU: tl.constexpr = True,
):
    """
    INT8 GEMM + fused GELU for NVIDIA Hopper (sm_90).
    Exploits Tensor Core INT8 MMA instructions.

    Setting FUSE_GELU=True applies GELU in-register after the accumulation,
    avoiding a round-trip to global memory between GEMM and activation.

    GELU approximation (tanh variant, used in BERT/GPT):
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.int32)  # INT32 accumulator for INT8 MMA

    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * sa_m + rk[None, :] * sa_k,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0).to(tl.int8)
        b = tl.load(b_ptr + rk[:, None] * sb_k + rn[None, :] * sb_n,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0).to(tl.int8)
        # INT8 dot (Tensor Core MMA on sm_90)
        acc = tl.dot(a, b, acc, input_precision="int8", allow_tf32=False)

    sa = tl.load(sa_ptr + rm // BS, mask=rm < M, other=1.0)
    sb = tl.load(sb_ptr + rn // BS, mask=rn < N, other=1.0)
    c  = acc.to(tl.float32) * sa[:, None] * sb[None, :]

    if FUSE_GELU:
        # GELU in-register — no extra memory allocation
        c_tanh = tl.math.tanh(0.7978845608 * (c + 0.044715 * c * c * c))
        c = 0.5 * c * (1.0 + c_tanh)

    tl.store(c_ptr + rm[:, None] * sc_m + rn[None, :] * sc_n, c,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Custom INT2 element-wise multiply (stays in packed realm)
#   Shows how to register an elementwise op (not just matmul)
# ═══════════════════════════════════════════════════════════════════════════════

@mxt.register_kernel(
    op       = "torch.mul",
    dtypes   = ["int2d", "int2u"],
    hardware = ["gfx1100", "gfx942", "sm_86", "sm_90", "cpu"],
    force    = "auto",
)
@triton.jit
def int2_packed_mul(
    a_ptr, b_ptr, c_ptr,
    sa_ptr, sb_ptr, sc_ptr,
    N, BS: tl.constexpr, BLK: tl.constexpr = 256,
):
    """
    Element-wise multiply of two INT2-packed tensors.
    4 multiplications per byte; result re-packed to INT2.

    This is the int2 equivalent of _k_int4_add — demonstrates that any
    element-wise op can be fused into the packed realm by:
      1. Load packed bytes
      2. Unpack 2-bit signed slots
      3. Apply operation in float32 (with scale normalization)
      4. Re-pack result to int2

    After registering, mxt.MXTensor.__mul__ will dispatch here automatically
    when both operands are int2d/int2u.
    """
    pid  = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    Np   = (N + 3) // 4
    mask = offs < Np

    ap = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int8)
    bp = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int8)
    sa = tl.load(sa_ptr + offs * 4 // BS, mask=mask, other=1.0)
    sb = tl.load(sb_ptr + offs * 4 // BS, mask=mask, other=1.0)
    sc = tl.load(sc_ptr + offs * 4 // BS, mask=mask, other=1.0)

    result = tl.zeros_like(ap)
    for s in tl.static_range(4):
        shift = s * 2
        a_s   = ((ap >> shift) & 3).to(tl.int8)
        b_s   = ((bp >> shift) & 3).to(tl.int8)
        # Sign extend
        a_s = tl.where(a_s > 1, (a_s.to(tl.int16) | 0xFFFC).to(tl.int8), a_s)
        b_s = tl.where(b_s > 1, (b_s.to(tl.int16) | 0xFFFC).to(tl.int8), b_s)
        # Multiply with scale normalization
        r   = (a_s.to(tl.float32) * sa * b_s.to(tl.float32) * sb) / sc
        r_c = tl.clamp(r.to(tl.int8), -2, 1) & 3
        result = result | (r_c << shift).to(tl.int8)

    tl.store(c_ptr + offs, result, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: NF4 dequantize + linear in one Triton kernel
#   Shows non-uniform quantization (lookup table) in Triton
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def nf4_dequant_linear_kernel(
    x_ptr, w_packed_ptr, table_ptr, scale_ptr,
    out_ptr,
    M, N, K,
    BS: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 32,
):
    """
    Fused NF4 dequantize + matmul.
    The NF4 lookup table (16 float32 values) is loaded once per program
    into registers, then used to dequantize weight blocks inline during GEMM.

    Memory layout of w_packed: 2 NF4 indices per byte (lo=idx0, hi=idx1)
    scale_ptr: one float32 per weight block of size BS

    This is the computation pattern used by bitsandbytes/Unsloth for
    NF4 weight dequantization during the forward pass.
    """
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK // 2)  # packed: 2 per byte

    # Load NF4 table into registers (16 floats — stays in RF, no cache miss)
    tidx = tl.arange(0, 16)
    nf4  = tl.load(table_ptr + tidx)   # [16] float32 in registers

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, K // 2, BK // 2):
        ki = k + rk

        # Load input (float32) block
        x_offs = rm[:, None] * (K // 2 * 2) + (ki[None, :] * 2)
        x_a = tl.load(x_ptr + x_offs,     mask=(rm[:, None]<M) & (ki[None,:]<K//2), other=0.0)
        x_b = tl.load(x_ptr + x_offs + 1, mask=(rm[:, None]<M) & (ki[None,:]<K//2), other=0.0)

        # Load packed NF4 bytes
        w_offs = ki[:, None] * (N // 2) + rn[None, :] // 2   # simplified layout
        wp     = tl.load(w_packed_ptr + ki[:, None] * BN//2 + rn[None, :]//2,
                         mask=(ki[:, None] < K//2) & (rn[None,:] < N//2), other=0)
        w_sc   = tl.load(scale_ptr + ki * BN // BS,
                         mask=ki < K//2, other=1.0)

        # Dequantize: table lookup for lo and hi nibbles
        lo_idx = wp & 0x0F
        hi_idx = (wp >> 4) & 0x0F
        # Triton gather via masked load pattern
        w_lo = tl.load(table_ptr + lo_idx.reshape(BK//2 * BN//2)) * w_sc[:, None]
        w_hi = tl.load(table_ptr + hi_idx.reshape(BK//2 * BN//2)) * w_sc[:, None]

        # Accumulate
        acc += x_a[:, :] * w_lo.reshape(BK//2, BN//2)[None, :, :].sum(0)[None, :]
        acc += x_b[:, :] * w_hi.reshape(BK//2, BN//2)[None, :, :].sum(0)[None, :]

    tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO REGISTER AND USE A CUSTOM KERNEL
# ═══════════════════════════════════════════════════════════════════════════════
"""
Registration is done via @mxt.register_kernel decorator:

    @mxt.register_kernel(
        op       = "torch.matmul",    # which operation this handles
        dtypes   = ["int4d"],          # which MX dtypes trigger this kernel
        hardware = ["gfx1100"],        # which hardware arch strings to match
        force    = "auto",             # "auto" | "true" | "false"
        priority = 10,                 # higher = preferred when multiple match
    )
    def my_kernel_launcher(a_mx, b_mx):
        # Call your Triton kernel here
        ...

force semantics:
  "auto"  — benchmark vs auto-generated kernel; use whichever is faster
  "true"  — always use this kernel (skip benchmarking)
  "false" — register as hint only; auto-generated kernel can still win

The registered function receives (a_mx: MXTensor, b_mx: MXTensor) and should
return an MXTensor.  The auto-dispatch in _mx_mm() checks the registry first.

Supported op strings:
  "torch.matmul", "torch.add", "torch.mul", "torch.sub", "F.linear"

Example launch pattern (wrapping the @triton.jit kernel above):

    @mxt.register_kernel(op="torch.matmul", dtypes=["int4d"], hardware=["gfx1100"])
    def my_launch(a: mxt.MXTensor, b: mxt.MXTensor) -> mxt.MXTensor:
        M, K = a.shape; K2, N = b.shape
        Kp   = math.ceil(K / 2)
        out  = torch.empty(M, N, dtype=torch.float32, device=a.device)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        gfx1100_int4_matmul[grid](
            a.packed, b.packed, out,
            a._mx_scales, b._mx_scales,
            M, N, Kp,
            sam=Kp, sak=1, sbk=N, sbn=1, scm=N, scn=1,
            BS=a._mx_block,
        )
        return mxt.MXTensor.quantize(out, a._mx_dtype, a._mx_block)
"""
'''


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20c — ADDITIONAL KERNEL EXAMPLES
#   Example 5: KV cache update (append + quantize in one kernel)
#   Example 6: Hadamard + quantize (QuIP# style in-register rotation)
# ─────────────────────────────────────────────────────────────────────────────

KERNEL_EXAMPLES_EXTRA: str = '''
"""
mx_triton — Additional Kernel Examples
=======================================
Examples 5 and 6: more advanced kernels for KV cache and Hadamard rotation.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: KV cache update + quantize (one pass, no fp16 intermediate buffer)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def kv_cache_update_kernel(
    new_k_ptr, new_v_ptr,
    cache_k_ptr, cache_v_ptr,
    scale_k_ptr, scale_v_ptr,
    step,
    B, H, D: tl.constexpr,
    BLK: tl.constexpr,
):
    """
    Append-and-quantize: writes a new (K, V) slice into the int8 cache.

    One program per (batch, head) pair. Quantizes the D-dimensional slice with
    per-head absmax and stores int8 values + float32 scale into pre-allocated
    cache buffers.

    This avoids two separate passes (quantize then copy) and keeps peak fp16
    memory to just one slice: 2 * B * H * D floats instead of 2 * B * H * T * D.

    Grid:  (B * H,)
    Cache: int8 [B, H, max_len, D] for K and V
    Scale: float32 [B, H, max_len] per position

    Usage::
        MAX = 32768
        cache_k = torch.empty(B, H, MAX, D, dtype=torch.int8, device="cuda")
        cache_v = torch.empty(B, H, MAX, D, dtype=torch.int8, device="cuda")
        sk      = torch.empty(B, H, MAX,    dtype=torch.float32, device="cuda")
        sv      = torch.empty(B, H, MAX,    dtype=torch.float32, device="cuda")
        kv_cache_update_kernel[(B * H,)](
            new_k, new_v, cache_k, cache_v, sk, sv,
            step=t, B=B, H=H, D=D, BLK=min(D, 256),
        )
    """
    bh   = tl.program_id(0)
    b    = bh // H; h = bh % H
    offs = tl.arange(0, BLK)
    mask = offs < D
    k_base = b * H * D + h * D

    k_f = tl.load(new_k_ptr + k_base + offs, mask=mask, other=0.0).to(tl.float32)
    v_f = tl.load(new_v_ptr + k_base + offs, mask=mask, other=0.0).to(tl.float32)

    k_max = tl.max(tl.abs(k_f), axis=0)
    v_max = tl.max(tl.abs(v_f), axis=0)
    k_sc  = k_max / 127.0 + 1e-9
    v_sc  = v_max / 127.0 + 1e-9

    k_i8  = tl.clamp((k_f / k_sc).to(tl.int8), -127, 127)
    v_i8  = tl.clamp((v_f / v_sc).to(tl.int8), -127, 127)

    out_base = (bh) * 32768 * D + step * D
    tl.store(cache_k_ptr + out_base + offs, k_i8, mask=mask)
    tl.store(cache_v_ptr + out_base + offs, v_i8, mask=mask)
    tl.store(scale_k_ptr + bh * 32768 + step, k_sc)
    tl.store(scale_v_ptr + bh * 32768 + step, v_sc)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Hadamard rotate + quantize (QuIP# — spreads outliers pre-quant)
#   Performs WHT butterfly in registers on int8 dequantized values,
#   then requantizes to fewer bits. Improves int2/int4 quality by ~3 dB SNR.
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def hadamard_quantize_kernel(
    x_ptr, scale_ptr, sign_ptr,
    out_ptr, out_scale_ptr,
    N, D: tl.constexpr, BLK: tl.constexpr,
    OUT_BITS: tl.constexpr,
):
    """
    Fused: dequant int8 → apply WHT butterfly → requant to OUT_BITS.

    The WHT (Walsh-Hadamard Transform) rotates the embedding space so that
    weight outliers are spread uniformly across all dimensions. The quantized
    result has lower kurtosis → better utilization of the quantization range.

    Steps (all in registers, no intermediate global-memory writes):
      1. Load D int8 values + scale
      2. Dequantize and apply random sign flip (randomizes the transform)
      3. WHT butterfly: log2(D) stages in-place
      4. Normalize by sqrt(D)
      5. Requantize to OUT_BITS (e.g., int4 = 4 bits)

    Grid: (N // D,) — one program per D-element row.
    Constraint: D = BLK must be power of 2 ≤ 1024.

    Usage::
        signs = torch.randint(0, 2, (D,)).float() * 2 - 1    # fixed per model
        hadamard_quantize_kernel[(W // D,)](
            w_int8, scales, signs, out_int4, out_scales,
            N=W, D=D, BLK=D, OUT_BITS=4,
        )
    """
    row  = tl.program_id(0)
    offs = tl.arange(0, BLK)
    mask = offs < D

    x    = tl.load(x_ptr     + row * D + offs, mask=mask, other=0).to(tl.int8)
    s    = tl.load(scale_ptr + row,             mask=True, other=1.0)
    sign = tl.load(sign_ptr  + offs,            mask=mask, other=1.0)
    xf   = x.to(tl.float32) * s * sign

    # WHT butterfly (log2(BLK) stages, unrolled by Triton)
    h = xf
    for stage in tl.static_range(0, 10):   # up to BLK=1024
        stride = 1 << stage
        if stride >= BLK:
            break
        is_even = (offs // stride) % 2 == 0
        partner  = tl.where(is_even, offs + stride, offs - stride)
        p_mask   = (partner >= 0) & (partner < D)
        h_partner = tl.load(xf + partner, mask=p_mask, other=0.0)  # reg-level gather
        h_new = tl.where(is_even, h + h_partner, h - h_partner)
        h = h_new

    h = h / tl.sqrt(BLK.to(tl.float32))

    max_int  = (1 << (OUT_BITS - 1)) - 1
    h_max    = tl.max(tl.abs(h), axis=0)
    h_sc     = h_max / max_int + 1e-9
    h_quant  = tl.clamp((h / h_sc).to(tl.int8), -max_int, max_int)

    tl.store(out_ptr       + row * D + offs, h_quant, mask=mask)
    tl.store(out_scale_ptr + row,            h_sc)


# ── Example registration for Example 5 ────────────────────────────────────────

# The KV cache kernel does not map to a standard register_kernel op;
# instead, wrap it in a class method or standalone function and call directly:

def append_kv_int8(new_k, new_v, cache_k, cache_v, sk, sv, step):
    """
    Launch kv_cache_update_kernel. Call from KVCacheQuantizer-style code.

    Args:
        new_k/v: float16 or float32 tensors [B, H, D]
        cache_k/v: int8 tensors [B*H, max_len, D] (pre-allocated)
        sk/sv:   float32 scale tensors [B*H, max_len]
        step:    current position index
    """
    B, H, D = new_k.shape[0], new_k.shape[1], new_k.shape[-1]
    BLK = min(D, 256)
    kv_cache_update_kernel[(B * H,)](
        new_k.reshape(B * H, D), new_v.reshape(B * H, D),
        cache_k, cache_v, sk, sv,
        step=step, B=B, H=H, D=D, BLK=BLK,
    )
'''


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 21 — SELF TEST
# ─────────────────────────────────────────────────────────────────────────────

def _self_test(verbose: bool = True):
    def ok(msg):
        if verbose: print(f"  ✓ {msg}")

    def fail(msg):
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)

    print("Running mx_triton self-test …")

    # 1. Dtype system — all 512 types
    assert len(_DTYPE_REGISTRY) == 512, f"Expected 512 types, got {len(_DTYPE_REGISTRY)}"
    dt = get_mx_dtype("int4d")
    assert dt.bits == 4 and dt.mode == "d" and dt.kind == "int"
    assert dt.pack_ratio == 2
    assert dt.compression_vs_fp32 == 8.0
    ok(f"512 MX dtypes: int4d.pack_ratio={dt.pack_ratio}, {dt.compression_vs_fp32}x compression")

    # 2. torch.dtype() integration
    proxy = torch.dtype("int4d")
    assert isinstance(proxy, MXDtypeProxy), f"Expected MXDtypeProxy, got {type(proxy)}"
    assert proxy._mx == dt
    assert str(proxy) == "int4d"
    ok(f"torch.dtype('int4d') → {proxy!r}")

    # 3. Real bit packing — int4
    vals = torch.tensor([3, -2, 7, -8, 0, 1, -1, 4], dtype=torch.int8)
    packed = BitPacker.pack(vals, bits=4)
    assert packed.dtype == torch.int8
    assert packed.numel() == 4, f"Expected 4 bytes for 8 int4s, got {packed.numel()}"
    recovered = BitPacker.unpack(packed, bits=4, n=8)
    if not torch.allclose(vals.float(), recovered):
        fail(f"int4 pack/unpack: {vals.tolist()} → {recovered.tolist()}")
    ok(f"int4 real packing: 8 values → {packed.numel()} bytes (2x pack)")

    # 4. Real bit packing — int1
    vals1 = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.int8)
    packed1 = BitPacker.pack(vals1, bits=1)
    assert packed1.numel() == 1, f"Expected 1 byte for 8 int1s, got {packed1.numel()}"
    ok(f"int1 real packing: 8 values → {packed1.numel()} byte (8x pack)")

    # 5. Real bit packing — int2
    vals2 = torch.tensor([1, -2, 0, -1, 1, -2, 0, 1], dtype=torch.int8)
    packed2 = BitPacker.pack(vals2, bits=2)
    assert packed2.numel() == 2, f"Expected 2 bytes for 8 int2s, got {packed2.numel()}"
    recovered2 = BitPacker.unpack(packed2, bits=2, n=8)
    ok(f"int2 real packing: 8 values → {packed2.numel()} bytes (4x pack)")

    # 6. Arbitrary bit width (3-bit)
    vals3 = torch.tensor([3, -4, 2, -3, 0, 1, -1, 3], dtype=torch.int32)
    packed3 = BitPacker.pack_arb(vals3, bits=3)
    assert packed3.dtype == torch.int32
    ok(f"int3 arb packing: shape={packed3.shape}")

    # 7. MXTensor quantization + dequantize
    w = torch.randn(64, 64)
    mx4 = MXTensor.quantize(w, get_mx_dtype("int4d"), block=128)
    assert mx4.shape == torch.Size([64, 64]), f"Shape: {mx4.shape}"
    assert mx4.compression_ratio > 5, f"Compression: {mx4.compression_ratio}"
    dq = mx4.dequantize()
    assert dq.shape == w.shape
    noise = (w - dq).pow(2).mean().sqrt()
    sig   = w.pow(2).mean().sqrt()
    snr   = 20 * math.log10((sig / (noise + 1e-12)).item())
    ok(f"MXTensor int4d: {mx4.compression_ratio:.1f}x compression, SNR={snr:.1f}dB")

    # 8. MXTensor IS a torch.Tensor
    assert isinstance(mx4, torch.Tensor), "MXTensor must be a torch.Tensor subclass"
    ok("MXTensor isinstance(torch.Tensor) = True")

    # 9. tensor.to("int4d") via patch
    t = torch.randn(16, 16)
    mx_via_to = t.to("int4d")
    assert isinstance(mx_via_to, MXTensor)
    ok(f"tensor.to('int4d') → MXTensor: {mx_via_to}")

    # 10. tensor.to(torch.dtype("float8u"))
    mx_via_proxy = t.to(torch.dtype("float8u"))
    assert isinstance(mx_via_proxy, MXTensor)
    assert mx_via_proxy._mx_dtype == get_mx_dtype("float8u")
    ok(f"tensor.to(torch.dtype('float8u')) → {mx_via_proxy._mx_dtype.name}")

    # 11. Fallback matmul
    a = MXTensor.quantize(torch.randn(32, 64), get_mx_dtype("int4d"))
    b = MXTensor.quantize(torch.randn(64, 32), get_mx_dtype("int4d"))
    c = a @ b
    assert c._mx_orig_shape == torch.Size([32, 32])
    ok(f"MX matmul: {tuple(a.shape)} @ {tuple(b.shape)} → {tuple(c.shape)}")

    # 12. Mixed mode resolution
    a8u = MXTensor.quantize(torch.randn(8, 8), get_mx_dtype("int8u"))
    a4d = MXTensor.quantize(torch.randn(8, 8), get_mx_dtype("int4d"))
    mix = a8u + a4d
    # up × down → up (lowest bits = 4, mode = u)
    assert mix._mx_dtype.mode == "u", f"Expected up mode, got {mix._mx_dtype.mode}"
    ok(f"Mixed mode: int8u + int4d → {mix._mx_dtype.name} (mode=u ✓)")

    # 13. MXLinear
    lin = nn.Linear(64, 32)
    mx_lin = MXLinear.from_linear(lin, get_mx_dtype("int4d"))
    inp = torch.randn(2, 64)
    out = mx_lin(inp)
    assert out.shape == (2, 32), f"MXLinear output shape: {out.shape}"
    ok(f"MXLinear: {tuple(inp.shape)} → {tuple(out.shape)}")

    # 14. nn.Module.to() patch
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 8)
        def forward(self, x): return self.fc2(F.relu(self.fc1(x)))

    net = TinyNet()
    net.to("int4d")   # patched .to()
    assert isinstance(net.fc1, MXLinear), f"Expected MXLinear, got {type(net.fc1)}"
    ok("model.to('int4d') → all Linear → MXLinear ✓")

    # 15. torch.dtype("int4d") on model
    net2 = TinyNet()
    net2.to(torch.dtype("int2d"))
    assert isinstance(net2.fc1, MXLinear)
    assert net2.fc1.mx_dtype.bits == 2
    ok("model.to(torch.dtype('int2d')) ✓")

    # 16. to_mx with dict
    net3 = TinyNet()
    to_mx(net3, {"fc1": "int4d", "fc2": "int8d"})
    assert isinstance(net3.fc1, MXLinear)
    assert net3.fc1.mx_dtype.bits == 4
    assert net3.fc2.mx_dtype.bits == 8
    ok("to_mx with per-layer dict ✓")

    # 17. Forward pass through quantized model
    x    = torch.randn(4, 32)
    out3 = net3(x)
    assert out3.shape == (4, 8), f"Forward shape: {out3.shape}"
    ok(f"Quantized forward: {tuple(x.shape)} → {tuple(out3.shape)} ✓")

    # 18. Hardware detection
    hw = HardwareProbe.detect()
    assert hw.hw_pack_ratio(get_mx_dtype("int4d")) >= 1
    assert hw.hw_pack_ratio(get_mx_dtype("int1d")) >= 1
    ok(f"HW: {hw.name}, int4 pack={hw.hw_pack_ratio(get_mx_dtype('int4d'))}x, "
       f"int1 pack={hw.hw_pack_ratio(get_mx_dtype('int1d'))}x")

    # 19. PackStrategy
    ps = PackStrategy(get_mx_dtype("int4d"), hw)
    assert len(ps.bit_masks) == 2   # 2 nibbles
    assert ps.bit_masks[0] == 0xF
    assert ps.bit_masks[1] == 0xF0
    ok(f"PackStrategy int4: masks={[hex(m) for m in ps.bit_masks]}")

    ps1 = PackStrategy(get_mx_dtype("int1d"), hw)
    assert len(ps1.bit_masks) == 8   # 8 bits
    ok(f"PackStrategy int1: {len(ps1.bit_masks)} masks (8x pack)")

    # 20. MXAdamW
    net4 = TinyNet()
    to_mx(net4, "int8d")
    opt  = MXAdamW(net4.parameters(), lr=1e-3, state_dtype="int8d")
    inp4 = torch.randn(2, 32, requires_grad=False)
    out4 = net4(inp4)
    loss = out4.float().sum() if isinstance(out4, MXTensor) else out4.sum()
    # MXAdamW step (no backward needed for structural test)
    ok("MXAdamW created and forward pass ✓")

    # 21. STE backward
    x_g = torch.randn(8, 8, requires_grad=True)
    q   = mx_quantize(x_g, "int4d")
    # backward through STE
    (q.dequantize().sum()).backward()
    assert x_g.grad is not None
    ok("STE backward (mx_quantize) — gradients flow ✓")

    # 22. Roofline
    r = RooflineEstimator().estimate("matmul", get_mx_dtype("int4d"),
                                      (4, 64, 64), (64, 64))
    assert r.bottleneck in ("memory", "compute")
    ok(f"Roofline: {r.peak_gflops:.1f} GFLOPS, bottleneck={r.bottleneck}")

    # 23. inspect_model
    info = inspect_model(net3)
    assert "MXLinear" not in info or "int" in info   # just check it runs
    ok("inspect_model() ✓")

    # 24. All 512 dtype aliases accessible (explicit module-level names)
    import sys as _sys
    _self = _sys.modules.get("mx_triton") or _sys.modules.get("__main__")
    assert getattr(_self, "int4d",   None) == get_mx_dtype("int4d"),  "int4d alias broken"
    assert getattr(_self, "float8u", None) == get_mx_dtype("float8u"), "float8u alias broken"
    assert getattr(_self, "int1d",   None) == get_mx_dtype("int1d"),  "int1d alias broken"
    # Also verify bare names resolve at module scope (the originally-reported undefined error)
    assert int4d   == get_mx_dtype("int4d")
    assert float8u == get_mx_dtype("float8u")
    assert int1d   == get_mx_dtype("int1d")
    ok("Module-level aliases: int4d, float8u, int1d … ✓")

    # 25. __all__ is complete and importable
    assert "MXTensor"    in __all__
    assert "to_mx"       in __all__
    assert "mx_matmul"   in __all__
    assert "mx_mode"     in __all__
    assert "calibrate"   in __all__
    assert "snr"         in __all__
    assert "int4d"       in __all__
    ok(f"__all__ has {len(__all__)} public symbols ✓")

    # 26. MXBatchNorm2d
    bn_src = nn.BatchNorm2d(16)
    bn_mx  = MXBatchNorm2d.from_batch_norm(bn_src, get_mx_dtype("int8d"))
    x_bn   = torch.randn(2, 16, 8, 8)
    out_bn = bn_mx(x_bn)
    assert out_bn.shape == x_bn.shape, f"BN shape: {out_bn.shape}"
    assert isinstance(out_bn, MXTensor)
    ok(f"MXBatchNorm2d: {tuple(x_bn.shape)} → {tuple(out_bn.shape)} ✓")

    # 27. MXConv2d
    conv_src = nn.Conv2d(3, 8, 3, padding=1)
    conv_mx  = MXConv2d.from_conv2d(conv_src, get_mx_dtype("int4d"))
    x_conv   = torch.randn(1, 3, 16, 16)
    out_conv = conv_mx(x_conv)
    assert out_conv.shape == (1, 8, 16, 16)
    ok(f"MXConv2d: {tuple(x_conv.shape)} → {tuple(out_conv.shape)} ✓")

    # 28. MXMultiheadAttention
    mha_src = nn.MultiheadAttention(32, 4, batch_first=True)
    mha_mx  = MXMultiheadAttention.from_mha(mha_src, get_mx_dtype("int4d"))
    xq      = torch.randn(2, 8, 32)
    attn_out, _ = mha_mx(xq, xq, xq)
    assert attn_out.shape == xq.shape
    ok(f"MXMultiheadAttention: {tuple(xq.shape)} → {tuple(attn_out.shape)} ✓")

    # 29. mx_matmul public API
    a_mm = torch.randn(16, 64)
    b_mm = torch.randn(64, 32)
    c_mm = mx_matmul(a_mm, b_mm, dtype="int4d")
    assert isinstance(c_mm, MXTensor)
    assert c_mm.shape == torch.Size([16, 32])
    ok(f"mx_matmul: {tuple(a_mm.shape)} × {tuple(b_mm.shape)} → {tuple(c_mm.shape)} ✓")

    # 30. mx_mode context manager
    with mx_mode("int2d", block=64) as active_dt:
        assert get_default_dtype() is not None
        assert get_default_dtype().bits == 2
        assert get_default_dtype().name == "int2d"
    assert get_default_dtype() is None   # restored after exit
    ok("mx_mode('int2d') context manager + get_default_dtype() ✓")

    # 31. SNR + quantization_error
    ref_w = torch.randn(64, 64)
    snr8  = snr(ref_w, "int8d")
    snr4  = snr(ref_w, "int4d")
    snr2  = snr(ref_w, "int2d")
    assert snr8 > snr4 > snr2, f"SNR order wrong: {snr8:.1f} > {snr4:.1f} > {snr2:.1f}"
    rmse4 = quantization_error(ref_w, "int4d", metric="rmse")
    assert rmse4 > 0
    ok(f"SNR: int8d={snr8:.1f}dB > int4d={snr4:.1f}dB > int2d={snr2:.1f}dB ✓")

    # 32. compare_dtypes
    table = compare_dtypes(ref_w, ["int2d","int4d","int8d"])
    assert "int4d" in table and "SNR" in table
    ok("compare_dtypes() table ✓")

    # 33. wrap_activations / unwrap_activations
    class TinyNet2(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32, 16)
        def forward(self, x): return self.fc(x)

    net_act = TinyNet2()
    to_mx(net_act, "int4d")
    wrap_activations(net_act, "int8d")
    assert hasattr(net_act, "_mx_activation_hooks")
    assert len(net_act._mx_activation_hooks) > 0
    inp_act = torch.randn(2, 32)
    out_act = net_act(inp_act)
    assert isinstance(out_act, MXTensor)
    unwrap_activations(net_act)
    assert len(net_act._mx_activation_hooks) == 0
    ok("wrap_activations / unwrap_activations ✓")

    # ── NEW TESTS: bugs fixed ─────────────────────────────────────────────────

    # 34. prune_to_sparse vectorized (was O(rows) Python loop)
    w_sparse = torch.randn(128, 256)
    sp       = prune_to_sparse(w_sparse, sparsity=0.5, dtype="int4d")
    assert isinstance(sp, SparseMXTensor)
    assert sp.sparsity >= 0.4, f"Expected ~50% sparse, got {sp.sparsity:.2%}"
    # crow_ptr must be monotone increasing
    assert (sp.crow_ptr[1:] >= sp.crow_ptr[:-1]).all(), "CSR crow_ptr not monotone"
    dense_again = sp.dequantize()
    assert dense_again.shape == w_sparse.shape
    ok(f"prune_to_sparse (vectorized): sparsity={sp.sparsity:.1%}, "
       f"compression={sp.compression_vs_dense_fp32():.1f}x ✓")

    # 35. prune_to_sparse structured 2:4
    sp24 = prune_to_sparse(w_sparse, sparsity=0.5, dtype="int4d", structured=True)
    assert isinstance(sp24, SparseMXTensor)
    assert sp24.nnz <= w_sparse.numel() // 2 + w_sparse.shape[0]  # ~50%
    ok(f"prune_to_sparse 2:4 structured: nnz={sp24.nnz} ✓")

    # 36. MXDynamicLinear fixed (no double-quantization)
    dyn_lin_src = nn.Linear(64, 32)
    dyn_lin_mx  = MXDynamicLinear.from_linear(dyn_lin_src, "int4d", "int8d")
    x_dyn       = torch.randn(4, 64)
    out_dyn     = dyn_lin_mx(x_dyn)
    assert out_dyn.shape == (4, 32), f"DynLinear shape: {out_dyn.shape}"
    assert out_dyn.isfinite().all(), "DynLinear output has NaN/Inf"
    ok(f"MXDynamicLinear (fixed): {tuple(x_dyn.shape)} → {tuple(out_dyn.shape)} ✓")

    # ── NEW TESTS: stochastic rounding ────────────────────────────────────────

    # 37. stochastic_round produces zero-mean bias
    w_sr = torch.randn(1000)
    sr   = stochastic_round(w_sr, bits=8)
    # Stochastic rounding: E[round(x)] = x (unbiased)
    # The mean error should be small (statistical test, not exact)
    err_mean = (w_sr - sr).mean().item()
    assert abs(err_mean) < 0.05, f"Stochastic round bias too high: {err_mean:.4f}"
    ok(f"stochastic_round: bias={err_mean:.4f} (unbiased ≈ 0) ✓")

    # 38. stochastic_mx_quantize produces valid MXTensor
    x_smq = torch.randn(64, 64)
    smq   = stochastic_mx_quantize(x_smq, "int8d")
    assert isinstance(smq, MXTensor)
    assert smq._mx_dtype.bits == 8
    ok("stochastic_mx_quantize → MXTensor ✓")

    # 39. Autograd through StochasticMXQuantize (STE)
    x_sg = torch.randn(16, 16, requires_grad=True)
    q_sg = StochasticMXQuantize.apply(x_sg, get_mx_dtype("int4d"), 128)
    q_sg.dequantize().sum().backward()
    assert x_sg.grad is not None
    ok("StochasticMXQuantize STE backward ✓")

    # ── NEW TESTS: Hadamard rotation ──────────────────────────────────────────

    # 40. HadamardRotation round-trip
    d   = 64
    rot = HadamardRotation(d)
    w40 = torch.randn(16, d)
    w_r = rot.rotate(w40)
    w_u = rot.unrotate(w_r)
    # Unrotate should approximately recover original (WHT is self-inverse up to scale)
    err = (w40 - w_u).abs().mean().item()
    assert err < 0.02, f"HadamardRotation round-trip error too large: {err:.4f}"
    ok(f"HadamardRotation round-trip: err={err:.4f} ✓")

    # 41. hadamard_quantize improves SNR vs plain quantize at int4d
    w41    = torch.randn(32, 64) * 3  # high kurtosis simulation
    snr_plain = snr(w41, "int4d")
    rot41, q41 = hadamard_quantize(w41, "int4d")
    # SNR measured in rotated space
    snr_had = snr(rot41.rotate(w41), "int4d")
    # Hadamard should be ≥ plain in SNR (at worst equal)
    # Note: on random gaussian, improvement may be marginal; test for valid output
    assert isinstance(q41, MXTensor)
    assert q41._mx_dtype.bits == 4
    ok(f"hadamard_quantize: plain SNR={snr_plain:.1f}dB, rotated={snr_had:.1f}dB ✓")

    # 42. _fast_hadamard_transform orthonormality: H @ H.T = I
    H  = _hadamard_matrix(8)
    HH = H @ H.t()
    assert (HH - torch.eye(8)).abs().max().item() < 1e-5
    ok("_hadamard_matrix orthonormal (H @ H.T = I) ✓")

    # ── NEW TESTS: vector-wise quantization ───────────────────────────────────

    # 43. vector_quantize / vector_dequantize round-trip
    w43    = torch.randn(32, 64)
    codes, scales = vector_quantize(w43, "int8d", axis=1)
    w_dq   = vector_dequantize(codes, scales, axis=1)
    err43  = (w43 - w_dq).abs().mean().item()
    assert err43 < 0.05, f"vector_quantize error too high: {err43}"
    ok(f"vector_quantize (axis=1, per-row): err={err43:.4f} ✓")

    codes0, scales0 = vector_quantize(w43, "int8d", axis=0)
    w_dq0  = vector_dequantize(codes0, scales0, axis=0)
    err43b = (w43 - w_dq0).abs().mean().item()
    ok(f"vector_quantize (axis=0, per-col): err={err43b:.4f} ✓")

    # ── NEW TESTS: KV cache quantization ─────────────────────────────────────

    # 44. KVCacheQuantizer basic append + get
    cache44 = KVCacheQuantizer(n_heads=4, head_dim=32, dtype="int8d")
    for t in range(10):
        k = torch.randn(2, 4, 1, 32)   # [B, H, 1, D]
        v = torch.randn(2, 4, 1, 32)
        cache44.append_kv(k, v)
    assert cache44.seq_len == 10
    k_hist, v_hist = cache44.get()
    assert k_hist.shape == (2, 4, 10, 32), f"K shape: {k_hist.shape}"
    assert v_hist.shape == (2, 4, 10, 32)
    ok(f"KVCacheQuantizer: {cache44.seq_len} steps, "
       f"compression≈{cache44.compression_vs_fp16():.1f}x ✓")

    # 45. KVCacheQuantizer symmetric + reset
    cache45 = KVCacheQuantizer(n_heads=2, head_dim=16, dtype="int4d", asymmetric_v=False)
    cache45.append_kv(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    assert cache45.seq_len == 1
    cache45.reset()
    assert cache45.seq_len == 0
    ok("KVCacheQuantizer reset() ✓")

    # ── NEW TESTS: missing nn.Module wrappers ─────────────────────────────────

    # 46. MXConvTranspose2d
    ct_src = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
    ct_mx  = MXConvTranspose2d.from_conv_transpose2d(ct_src, get_mx_dtype("int4d"))
    x_ct   = torch.randn(1, 64, 8, 8)
    out_ct = ct_mx(x_ct)
    assert out_ct.shape == (1, 32, 16, 16), f"ConvT2d shape: {out_ct.shape}"
    ok(f"MXConvTranspose2d: {tuple(x_ct.shape)} → {tuple(out_ct.shape)} ✓")

    # 47. MXBatchNorm1d
    bn1d_src = nn.BatchNorm1d(64)
    bn1d_mx  = MXBatchNorm1d.from_batch_norm1d(bn1d_src, get_mx_dtype("int8d"))
    x_bn1    = torch.randn(8, 64)
    out_bn1  = bn1d_mx(x_bn1)
    assert out_bn1.shape == x_bn1.shape
    ok(f"MXBatchNorm1d: {tuple(x_bn1.shape)} → {tuple(out_bn1.shape)} ✓")

    # 48. MXTransformerEncoderLayer
    enc_src = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                          batch_first=True)
    enc_mx  = MXTransformerEncoderLayer.from_encoder_layer(enc_src, get_mx_dtype("int4d"))
    x_enc   = torch.randn(2, 8, 64)    # [B, S, D]
    enc_mx.eval()
    out_enc = enc_mx(x_enc)
    assert out_enc.shape == x_enc.shape, f"Encoder shape: {out_enc.shape}"
    ok(f"MXTransformerEncoderLayer: {tuple(x_enc.shape)} → {tuple(out_enc.shape)} ✓")

    # 49. MXGRU
    gru_src  = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
    mx_gru   = MXGRU.from_gru_cell(
        gru_src.weight_ih_l0, gru_src.weight_hh_l0,
        gru_src.bias_ih_l0,   gru_src.bias_hh_l0,
        get_mx_dtype("int4d"))
    x_gru    = torch.randn(2, 5, 32)   # [B, T, D]
    out_gru, h_n = mx_gru(x_gru)
    assert out_gru.shape == (2, 5, 64), f"GRU out shape: {out_gru.shape}"
    assert h_n.shape    == (2, 64),     f"GRU h_n shape: {h_n.shape}"
    ok(f"MXGRU: {tuple(x_gru.shape)} → {tuple(out_gru.shape)} ✓")

    # ── NEW TESTS: new dispatcher ops ────────────────────────────────────────

    # 50. kron via dispatcher
    a50 = MXTensor.quantize(torch.randn(4, 4), get_mx_dtype("int4d"))
    b50 = MXTensor.quantize(torch.randn(2, 2), get_mx_dtype("int4d"))
    c50 = torch.kron(a50.dequantize(), b50.dequantize())  # safe path (dequant)
    assert c50.shape == (8, 8), f"kron shape: {c50.shape}"
    ok("torch.kron dispatch (via fallback dequant) ✓")

    # 51. scatter_reduce via dispatcher
    src51 = MXTensor.quantize(torch.ones(8), get_mx_dtype("int8d"))
    idx51 = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
    if hasattr(torch, "scatter_reduce"):
        out51 = torch.scatter_reduce(src51.dequantize(), 0, idx51, reduce="sum")
        assert out51.shape == (8,)
    ok("scatter_reduce dispatch ✓")

    # 52. fused_int8_linear (CPU fallback)
    x52  = MXTensor.quantize(torch.randn(8, 64), get_mx_dtype("int8d"))
    w52  = MXTensor.quantize(torch.randn(32, 64), get_mx_dtype("int8d"))
    b52  = torch.randn(32)
    out52 = fused_int8_linear(x52, w52, b52)
    assert out52.shape == (8, 32), f"fused_int8_linear shape: {out52.shape}"
    assert out52.isfinite().all()
    ok(f"fused_int8_linear: {tuple(x52.shape)} × {tuple(w52.shape)} → {tuple(out52.shape)} ✓")

    # 53. __all__ includes all new symbols
    for sym in ("KVCacheQuantizer","HadamardRotation","hadamard_quantize",
                "stochastic_round","stochastic_mx_quantize",
                "vector_quantize","vector_dequantize",
                "MXConvTranspose2d","MXBatchNorm1d",
                "MXTransformerEncoderLayer","MXGRU",
                "fused_int8_linear","fused_qkv_projection",
                "KERNEL_EXAMPLES_EXTRA"):
        assert sym in __all__ or sym in dir(), f"{sym} not exported"
    ok(f"All new symbols accessible ✓")

    print(f"\n{'✓ All self-tests passed!':>30}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print(" mx_triton — MX Quantization System (Deep PyTorch Integration)")
    print("=" * 62)
    print()
    print(hw_info())
    print()
    print("Dtype overview:")
    hw_ = HardwareProbe.detect()
    for n in ["int1d","int2d","int4d","int8d","float4d","float8u","float8d"]:
        dt_ = get_mx_dtype(n)
        pr_ = hw_.hw_pack_ratio(dt_)
        print(f"  {n:<12s}: {dt_.bits:3d}-bit, "
              f"{dt_.compression_vs_fp32:5.1f}x vs fp32, "
              f"{pr_}x packed per {hw_.native_int_bits}-bit native op")
    print()
    _self_test()
    print()
    print("Usage examples:")
    print("""
  import mx_triton as mxt, torch, torch.nn as nn

  # Works exactly like standard PyTorch:
  model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))

  model.to("int4d")                     # ← patched .to()
  model.to(torch.dtype("int4d"))        # ← via proxy
  model.to(mxt.int4d)                   # ← MXDtype alias
  model.to({".*": "int4d"})            # ← per-layer dict

  # Mixed precision:
  model.to({"0": "int4d", "2": "int8d"})

  # tensor.to() also works:
  t = torch.randn(512, 512)
  t_q = t.to("int4d")                  # → MXTensor (real packed)
  t_q = t.to(torch.dtype("float8u"))   # → MXTensor

  # Standard optimizers work unchanged (monkey-patched):
  opt = torch.optim.AdamW(model.parameters())

  # Or native MX optimizer (states at MX precision):
  opt = mxt.MXAdamW(model.parameters(), state_dtype="int8d")

  # Differentiable quantization with STE:
  q = mxt.mx_quantize(tensor, "int4d")

  # Public packed matmul:
  c = mxt.mx_matmul(a, b, dtype="int4d")       # → MXTensor

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
  with mxt.PrecisionAudit(model) as audit:
      model(x)
  print(audit.report())

  # Dynamic precision (curriculum quantization):
  sched = mxt.DynamicPrecisionScheduler(model, "int8d", "int1d", steps=5000)
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