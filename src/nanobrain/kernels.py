# training-speed kernels (torch.library custom ops, so they sit inside the compiled / cuda-graphed regions):
# syrk (same math; train.py native_muon_syrk): muon's symmetric newton-schulz products as upper-triangle tile GEMMs
import torch
import triton
import triton.language as tl
from torch import Tensor


# muon: X X^T of a stack of bf16 [b, m, n] matrices (m a multiple of 128) computing only the upper triangle of 128 x 128
# output tiles and writing each also mirrored (the product is symmetric: the mirrored tile is the same fp32 dot products)
@triton.jit
def _syrk_kernel(x_ptr, o_ptr, ti_ptr, tj_ptr, K, ca, cb, cc, M: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                 POLY: tl.constexpr):
    # o = x x^T, or with POLY (x = A symmetric, K = M): o = ca I + cb A + cc A A^T, on the upper tiles, mirrored
    t = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    i = tl.load(ti_ptr + t)
    j = tl.load(tj_ptr + t)
    rm = i * BM + tl.arange(0, BM)
    rn = j * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    xa = x_ptr + b * M * K + rm[:, None] * K + rk[None, :]
    xb = x_ptr + b * M * K + rn[:, None] * K + rk[None, :]
    acc = tl.zeros([BM, BM], dtype=tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(xa), tl.trans(tl.load(xb)), acc)
        xa += BK
        xb += BK
    if POLY:
        A = tl.load(x_ptr + b * M * K + rm[:, None] * K + rn[None, :]).to(tl.float32)
        acc = cc * acc + cb * A + tl.where(rm[:, None] == rn[None, :], ca, 0.0)
    c = acc.to(tl.bfloat16)
    ob = o_ptr + b * M * M
    tl.store(ob + rm[:, None] * M + rn[None, :], c)
    if i != j:
        tl.store(ob + rn[:, None] * M + rm[None, :], tl.trans(c))


_TRI = {}


def _syrk(x: Tensor, poly: tuple[float, float, float] | None) -> Tensor:
    b, m, n = x.shape
    out = torch.empty(b, m, m, device=x.device, dtype=x.dtype)
    T = m // 128
    if (T, x.device) not in _TRI:
        ij = [(i, j) for i in range(T) for j in range(i, T)]
        _TRI[(T, x.device)] = tuple(torch.tensor(v, dtype=torch.int32, device=x.device) for v in zip(*ij))
    ti, tj = _TRI[(T, x.device)]
    _syrk_kernel[(len(ti), b)](x, out, ti, tj, n, *(poly or (0.0, 0.0, 0.0)), m, 128, 64, poly is not None,
                               num_warps=4, num_stages=3)
    return out


@torch.library.custom_op("nanobrain::syrk", mutates_args=())
def syrk(x: Tensor) -> Tensor:
    # x @ x.mT for a contiguous bf16 [b, m, n] stack (m a multiple of 128)
    return _syrk(x, None)


@syrk.register_fake
def _(x):
    return x.new_empty(x.shape[0], x.shape[1], x.shape[1])


@torch.library.custom_op("nanobrain::syrk_poly", mutates_args=())
def syrk_poly(A: Tensor, ca: float, cb: float, cc: float) -> Tensor:
    # ca I + cb A + cc A @ A for a stack of symmetric bf16 [b, m, m] (one rounding, from the fp32 accumulator)
    return _syrk(A, (ca, cb, cc))


@syrk_poly.register_fake
def _(A, ca, cb, cc):
    return torch.empty_like(A)
