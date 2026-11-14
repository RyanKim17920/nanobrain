# cuBLASLt GEMM epilogues (ctypes, the libcublasLt torch links): the MLP's fc1 + bias + GELU with the pre-activation
# kept as aux, and fc2's input gradient times GELU' of that aux, each one GEMM instead of a GEMM + an elementwise
# pass over the 4x wide hidden activations. cuBLASLt's GELU is the tanh approximation (NUMERICS-CHANGING vs nn.GELU's
# erf, |diff| < 5e-4). training speed only (Mlp.lt_gelu, set by train.py native_lt_gelu)
import ctypes
import os

import torch
from torch import Tensor

_lib = None
_state = {}
_one, _zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
GELU_AUX_BIAS, DGELU = 32 | 128 | 4, 64 | 128


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heur(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("ws", ctypes.c_size_t), ("state", ctypes.c_int), ("waves", ctypes.c_float),
                ("reserved", ctypes.c_int * 4)]


def _L():
    global _lib
    if _lib is None:
        p = os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cu13", "lib", "libcublasLt.so.13")
        L = ctypes.CDLL(p if os.path.exists(p) else "libcublasLt.so.13")
        vp, i32, u64, i64, sz = ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint64, ctypes.c_int64, ctypes.c_size_t
        L.cublasLtCreate.argtypes = [vp]
        L.cublasLtMatmulDescCreate.argtypes = [vp, i32, i32]
        L.cublasLtMatmulDescSetAttribute.argtypes = [vp, i32, vp, sz]
        L.cublasLtMatrixLayoutCreate.argtypes = [vp, i32, u64, u64, i64]
        L.cublasLtMatmulPreferenceCreate.argtypes = [vp]
        L.cublasLtMatmulPreferenceSetAttribute.argtypes = [vp, i32, vp, sz]
        L.cublasLtMatmulAlgoGetHeuristic.argtypes = [vp, vp, vp, vp, vp, vp, vp, i32, vp, vp]
        L.cublasLtMatmul.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, sz, vp]
        _lib = L
    return _lib


def _ck(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"cublasLt {what}: status {status}")


def init(device: torch.device) -> None:
    # handle + 32 MB workspace, made before any cuda graph capture
    idx = torch.cuda.current_device() if device.index is None else device.index
    if idx not in _state:
        h = ctypes.c_void_p()
        _ck(_L().cublasLtCreate(ctypes.byref(h)), "create")
        _state[idx] = (h, torch.empty(32 << 20, dtype=torch.uint8, device=f"cuda:{idx}"), {})


def _set(desc, attr: int, val, ctype) -> None:
    v = ctype(val)
    _ck(_L().cublasLtMatmulDescSetAttribute(desc, attr, ctypes.byref(v), ctypes.sizeof(v)), f"attribute {attr}")


def _layout(rows: int, cols: int):
    lay = ctypes.c_void_p()
    _ck(_L().cublasLtMatrixLayoutCreate(ctypes.byref(lay), 14, rows, cols, rows), "layout")  # bf16, column major
    return lay


def gemm(A: Tensor, B: Tensor, tb: bool, epi: int, aux: Tensor, bias: Tensor | None = None) -> Tensor:
    # row-major bf16 C = A @ op(B) (op = transpose when tb) with epilogue epi, as the column-major C^T = op(B)^T A^T
    M, K = A.shape
    N = B.shape[0] if tb else B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    h, ws, plans = _state[A.device.index]
    key = (M, N, K, tb, epi)
    if key not in plans:
        L = _L()
        desc = ctypes.c_void_p()
        _ck(L.cublasLtMatmulDescCreate(ctypes.byref(desc), 68, 0), "desc")  # fp32 compute and scale
        _set(desc, 3, int(tb), ctypes.c_int32)
        _set(desc, 7, epi, ctypes.c_uint32)
        _set(desc, 22, 14, ctypes.c_int32)  # aux bf16, laid out as C
        _set(desc, 12, N, ctypes.c_int64)
        _set(desc, 11, ws.data_ptr(), ctypes.c_void_p)  # aligned placeholders for the heuristic
        if bias is not None:
            _set(desc, 26, 14, ctypes.c_int32)
            _set(desc, 8, ws.data_ptr(), ctypes.c_void_p)
        lays = (_layout(B.shape[1], B.shape[0]), _layout(K, M), _layout(N, M))
        pref = ctypes.c_void_p()
        _ck(L.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "preference")
        wsz = ctypes.c_uint64(ws.numel())
        _ck(L.cublasLtMatmulPreferenceSetAttribute(pref, 1, ctypes.byref(wsz), 8), "preference workspace")
        res, cnt = (_Heur * 1)(), ctypes.c_int(0)
        _ck(L.cublasLtMatmulAlgoGetHeuristic(h, desc, *lays, lays[2], pref, 1, res, ctypes.byref(cnt)), "heuristic")
        if cnt.value < 1:
            raise RuntimeError(f"cublasLt: no algorithm for {key}")
        plans[key] = (desc, lays, _Algo.from_buffer_copy(res[0].algo))
    desc, (a_l, b_l, d_l), algo = plans[key]
    if bias is not None:
        _set(desc, 8, bias.data_ptr(), ctypes.c_void_p)
    _set(desc, 11, aux.data_ptr(), ctypes.c_void_p)
    vp = ctypes.c_void_p
    _ck(_L().cublasLtMatmul(h, desc, ctypes.byref(_one), vp(B.data_ptr()), a_l, vp(A.data_ptr()), b_l, ctypes.byref(_zero),
                            vp(C.data_ptr()), d_l, vp(C.data_ptr()), d_l, ctypes.byref(algo), vp(ws.data_ptr()),
                            ws.numel(), vp(torch.cuda.current_stream(A.device).cuda_stream)), "matmul")
    return C


@torch.library.custom_op("nanobrain::lt_gelu", mutates_args=())
def lt_gelu(x: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    # (gelu(x w^T + b), x w^T + b), x [M, K], w [N, K], b [N], bf16
    h = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
    return gemm(x.contiguous(), w.contiguous(), True, GELU_AUX_BIAS, h, b.contiguous()), h


@lt_gelu.register_fake
def _(x, w, b):
    return x.new_empty(x.shape[0], w.shape[0]), x.new_empty(x.shape[0], w.shape[0])


@torch.library.custom_op("nanobrain::lt_dgelu", mutates_args=())
def lt_dgelu(dy: Tensor, w: Tensor, h: Tensor) -> Tensor:
    # gelu'(h) * (dy w), dy [M, K], w [K, N] (fc2 weight), h [M, N] the pre-activation
    return gemm(dy.contiguous(), w.contiguous(), False, DGELU, h.contiguous())


@lt_dgelu.register_fake
def _(dy, w, h):
    return torch.empty_like(h)


class LtMlp(torch.autograd.Function):
    # fc2(gelu(fc1(x))) on bf16 [M, D] rows; the weights / biases come in bf16
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        a, h = lt_gelu(x, w1, b1)
        ctx.save_for_backward(x, w1, w2, h, a)
        return torch.addmm(b2, a, w2.t())

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2, h, a = ctx.saved_tensors
        dy = dy.contiguous()
        dh = lt_dgelu(dy, w2, h)
        return dh @ w1, dh.t() @ x, dh.sum(0), dy.t() @ a, dy.sum(0)
