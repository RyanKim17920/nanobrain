# training-speed kernels (torch.library custom ops, so they sit inside the compiled / cuda-graphed regions):
# - in-graph fp8 block linears (NUMERICS-CHANGING; train.py native_fp8_lt): cuBLASLt fp8 GEMMs on cublaslt's ctypes
#   handle with rowwise current scaling (one fp32 scale per token row / weight row: cuBLASLt OUTER_VEC_32F). fprop and
#   dgrad run in e4m3, wgrad stays bf16 (the bf16 path's layouts, no transposed fp8 copies). layer norm + cast is left
#   to inductor; GELU / GELU' / the qkv grad gather and their row casts are one triton pass each.
# - syrk (same math; train.py native_muon_syrk): muon's symmetric newton-schulz products as upper-triangle tile GEMMs
import ctypes

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from nanobrain import cublaslt as C

E4 = torch.float8_e4m3fn
BIAS = 4  # cuBLASLt epilogue


def _gemm8(P: Tensor, Q: Tensor, sp: Tensor, sq: Tensor, bias: Tensor | None) -> Tensor:
    # row-major bf16 out[M, N] = diag(sp) P[M, K] @ (diag(sq) Q[N, K])^T (+ bias), both e4m3 K-contiguous (the fp8 TN
    # layout): the column-major out^T = op(Q) P^T with Q in cuBLASLt's A slot. sp [M] / sq [N]: fp32 row dequant scales
    M, K = P.shape
    N = Q.shape[0]
    out = torch.empty(M, N, device=P.device, dtype=torch.bfloat16)
    h, ws, plans = C._state[P.device.index]
    key = ("f8", M, N, K, bias is not None)
    if key not in plans:
        L = C._L()
        desc = ctypes.c_void_p()
        C._ck(L.cublasLtMatmulDescCreate(ctypes.byref(desc), 68, 0), "desc")  # fp32 compute / scale
        C._set(desc, 3, 1, ctypes.c_int32)  # transA
        C._set(desc, 25, 1, ctypes.c_int8)  # fast accumulation
        C._set(desc, 31, 3, ctypes.c_uint32)  # A / B scales: outer vectors (OUTER_VEC_32F)
        C._set(desc, 32, 3, ctypes.c_uint32)
        attrs = (17, 18, 8) if bias is not None else (17, 18)
        if bias is not None:
            C._set(desc, 7, BIAS, ctypes.c_uint32)
            C._set(desc, 26, 14, ctypes.c_int32)
        for attr in attrs:  # aligned placeholders for the heuristic
            C._set(desc, attr, ws.data_ptr(), ctypes.c_void_p)
        lays = []
        for dt, rows, cols in ((28, K, N), (28, K, M), (14, N, M)):
            lay = ctypes.c_void_p()
            C._ck(L.cublasLtMatrixLayoutCreate(ctypes.byref(lay), dt, rows, cols, rows), "layout")
            lays.append(lay)
        pref = ctypes.c_void_p()
        C._ck(L.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "preference")
        wsz = ctypes.c_uint64(ws.numel())
        C._ck(L.cublasLtMatmulPreferenceSetAttribute(pref, 1, ctypes.byref(wsz), 8), "preference workspace")
        res, cnt = (C._Heur * 1)(), ctypes.c_int(0)
        C._ck(L.cublasLtMatmulAlgoGetHeuristic(h, desc, *lays, lays[2], pref, 1, res, ctypes.byref(cnt)), "heuristic")
        if cnt.value < 1:
            raise RuntimeError(f"cublasLt: no fp8 algorithm for {key}")
        plans[key] = (desc, lays, C._Algo.from_buffer_copy(res[0].algo))
    desc, (a_l, b_l, d_l), algo = plans[key]
    for attr, t in ((17, sq), (18, sp), (8, bias)):
        if t is not None:
            C._set(desc, attr, t.data_ptr(), ctypes.c_void_p)
    vp = ctypes.c_void_p
    C._ck(C._L().cublasLtMatmul(h, desc, ctypes.byref(C._one), vp(Q.data_ptr()), a_l, vp(P.data_ptr()), b_l,
                                ctypes.byref(C._zero), vp(out.data_ptr()), d_l, vp(out.data_ptr()), d_l, ctypes.byref(algo),
                                vp(ws.data_ptr()), ws.numel(), vp(torch.cuda.current_stream(P.device).cuda_stream)),
          "fp8 matmul")
    return out


@torch.library.custom_op("nanobrain::mm8", mutates_args=())
def mm8(P: Tensor, Q: Tensor, sp: Tensor, sq: Tensor, bias: Tensor | None = None) -> Tensor:
    # P Q^T (+ bias), bf16 out
    return _gemm8(P.contiguous(), Q.contiguous(), sp.contiguous(), sq.contiguous(), bias)


@mm8.register_fake
def _(P, Q, sp, sq, bias=None):
    return P.new_empty(P.shape[0], Q.shape[0], dtype=torch.bfloat16)


def quant(t: Tensor) -> tuple[Tensor, Tensor]:
    # rowwise current scaling: (t / s) in e4m3 with s = row amax / 448; returns (fp8 [M, K], s [M] fp32)
    s = (t.detach().abs().amax(-1, keepdim=True).float() / 448.0).clamp_min(1e-30)
    return (t.float() / s).clamp(-448.0, 448.0).to(E4).contiguous(), s.view(-1)


_K0, _K1 = tl.constexpr(0.7978845608028654), tl.constexpr(0.044715)  # tanh GELU: sqrt(2 / pi), cubic coefficient


@triton.jit
def _quant_row(v, s_ptr, q_ptr, row, cols, mask, N: tl.constexpr):
    # rowwise e4m3 of the bf16-rounded row v (fp32 values): scale = amax / 448
    s = tl.maximum(tl.max(tl.abs(v), 0) / 448.0, 1e-30)
    tl.store(q_ptr + row * N + cols, tl.clamp(v / s, -448.0, 448.0).to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row, s)


@triton.jit
def _gelu_q_kernel(h_ptr, a_ptr, q_ptr, s_ptr, N: tl.constexpr, BN: tl.constexpr):
    # a = gelu_tanh(h) (bf16) and its rowwise e4m3 copy, one row per program
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BN)
    mask = cols < N
    h = tl.load(h_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    a = (0.5 * h * (1.0 + libdevice.tanh(_K0 * (h + _K1 * h * h * h)))).to(tl.bfloat16)
    tl.store(a_ptr + row * N + cols, a, mask=mask)
    _quant_row(a.to(tl.float32), s_ptr, q_ptr, row, cols, mask, N)


@triton.jit
def _dgelu_q_kernel(g_ptr, h_ptr, d_ptr, q_ptr, s_ptr, db_ptr, M, N: tl.constexpr, BN: tl.constexpr, RPP: tl.constexpr):
    # d = g * gelu_tanh'(h) (bf16), its rowwise e4m3 copy, and per program column sums of d (the bias grad partials)
    pid = tl.program_id(0)
    cols = tl.arange(0, BN)
    mask = cols < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for r in range(RPP):
        row = (pid * RPP + r).to(tl.int64)
        if row < M:
            g = tl.load(g_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
            h = tl.load(h_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
            t = libdevice.tanh(_K0 * (h + _K1 * h * h * h))
            dg = 0.5 * (1.0 + t) + 0.5 * h * (1.0 - t * t) * _K0 * (1.0 + 3.0 * _K1 * h * h)
            d = (g * dg).to(tl.bfloat16)
            tl.store(d_ptr + row * N + cols, d, mask=mask)
            df = d.to(tl.float32)
            acc += df
            _quant_row(df, s_ptr, q_ptr, row, cols, mask, N)
    tl.store(db_ptr + pid * N + cols, acc, mask=mask)


@torch.library.custom_op("nanobrain::gelu_q", mutates_args=())
def gelu_q(h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # (gelu_tanh(h) bf16, its rowwise e4m3, the row scales)
    M, N = h.shape
    a, q, s = torch.empty_like(h), torch.empty(M, N, device=h.device, dtype=E4), torch.empty(M, device=h.device)
    _gelu_q_kernel[(M,)](h, a, q, s, N, triton.next_power_of_2(N), num_warps=8)
    return a, q, s


@gelu_q.register_fake
def _(h):
    return torch.empty_like(h), torch.empty_like(h, dtype=E4), h.new_empty(h.shape[0], dtype=torch.float32)


_RPP = 32


@torch.library.custom_op("nanobrain::dgelu_q", mutates_args=())
def dgelu_q(g: Tensor, h: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # (d = g gelu_tanh'(h) bf16, its rowwise e4m3, the row scales, d.sum(0) fp32)
    M, N = h.shape
    P = triton.cdiv(M, _RPP)
    d, q, s = torch.empty_like(h), torch.empty(M, N, device=h.device, dtype=E4), torch.empty(M, device=h.device)
    db = torch.empty(P, N, device=h.device)
    _dgelu_q_kernel[(P,)](g.contiguous(), h, d, q, s, db, M, N, triton.next_power_of_2(N), _RPP, num_warps=16)
    return d, q, s, db.sum(0)


@dgelu_q.register_fake
def _(g, h):
    return torch.empty_like(h), torch.empty_like(h, dtype=E4), h.new_empty(h.shape[0], dtype=torch.float32), \
        h.new_empty(h.shape[1], dtype=torch.float32)


@triton.jit
def _qkv_q_kernel(q_ptr, k_ptr, v_ptr, sb, sh, sn, d_ptr, q8_ptr, s_ptr, db_ptr, M, NT, H: tl.constexpr,
                  HD: tl.constexpr, RPP: tl.constexpr):
    # the attention input grads dq / dk / dv ([B, H, N, hd] views with shared strides sb / sh / sn, hd contiguous) as
    # one token-major [M, 3 H hd] bf16 row (the qkv GEMM's output layout), its rowwise e4m3 copy, and per program
    # column sums (the qkv bias grad partials)
    pid = tl.program_id(0)
    hh = tl.arange(0, H)[:, None]
    dd = tl.arange(0, HD)[None, :]
    off_in = hh * sh + dd
    off_out = hh * HD + dd
    W: tl.constexpr = H * HD
    a0 = tl.zeros([H, HD], dtype=tl.float32)
    a1 = tl.zeros([H, HD], dtype=tl.float32)
    a2 = tl.zeros([H, HD], dtype=tl.float32)
    for r in range(RPP):
        row = pid * RPP + r
        if row < M:
            b = row // NT
            base = b.to(tl.int64) * sb + (row - b * NT).to(tl.int64) * sn + off_in
            g0 = tl.load(q_ptr + base).to(tl.float32)
            g1 = tl.load(k_ptr + base).to(tl.float32)
            g2 = tl.load(v_ptr + base).to(tl.float32)
            out = row.to(tl.int64) * (3 * W) + off_out
            tl.store(d_ptr + out, g0.to(tl.bfloat16))
            tl.store(d_ptr + out + W, g1.to(tl.bfloat16))
            tl.store(d_ptr + out + 2 * W, g2.to(tl.bfloat16))
            a0 += g0
            a1 += g1
            a2 += g2
            amax = tl.maximum(tl.maximum(tl.max(tl.max(tl.abs(g0), 1), 0), tl.max(tl.max(tl.abs(g1), 1), 0)),
                              tl.max(tl.max(tl.abs(g2), 1), 0))
            sc = tl.maximum(amax / 448.0, 1e-30)
            tl.store(q8_ptr + out, tl.clamp(g0 / sc, -448.0, 448.0).to(tl.float8e4nv))
            tl.store(q8_ptr + out + W, tl.clamp(g1 / sc, -448.0, 448.0).to(tl.float8e4nv))
            tl.store(q8_ptr + out + 2 * W, tl.clamp(g2 / sc, -448.0, 448.0).to(tl.float8e4nv))
            tl.store(s_ptr + row, sc)
    pb = pid.to(tl.int64) * (3 * W) + off_out
    tl.store(db_ptr + pb, a0)
    tl.store(db_ptr + pb + W, a1)
    tl.store(db_ptr + pb + 2 * W, a2)


@torch.library.custom_op("nanobrain::qkv_q", mutates_args=())
def qkv_q(dq: Tensor, dk: Tensor, dv: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # ([M, 3 D] bf16 token rows of (dq, dk, dv), rowwise e4m3, row scales, column sums fp32) of [B, H, N, hd] grads
    B, H, N, hd = dq.shape
    if dq.stride(-1) != 1 or dk.stride() != dq.stride() or dv.stride() != dq.stride():
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
    M, W = B * N, 3 * H * hd
    P = triton.cdiv(M, _RPP)
    d, q8 = torch.empty(M, W, device=dq.device, dtype=torch.bfloat16), torch.empty(M, W, device=dq.device, dtype=E4)
    s = torch.empty(M, device=dq.device)
    db = torch.empty(P, W, device=dq.device)
    _qkv_q_kernel[(P,)](dq, dk, dv, dq.stride(0), dq.stride(1), dq.stride(2), d, q8, s, db, M, N, H, hd, _RPP,
                        num_warps=4)
    return d, q8, s, db.sum(0)


@qkv_q.register_fake
def _(dq, dk, dv):
    B, H, N, hd = dq.shape
    M, W = B * N, 3 * H * hd
    return dq.new_empty(M, W), dq.new_empty(M, W, dtype=E4), dq.new_empty(M, dtype=torch.float32), \
        dq.new_empty(W, dtype=torch.float32)


class QKV8(torch.autograd.Function):
    # q, k, v ([B, H, N, hd] views) of one fp8 GEMM x [M, D] @ cat(Wq, Wk, Wv)^T + b: fp8 fprop / dgrad, bf16 wgrad.
    # backward takes dq / dk / dv straight into one fused row pass (qkv_q)
    @staticmethod
    def forward(ctx, x, w, b, B, H):
        x8, sx = quant(x)
        w8, sw = quant(w)
        ctx.save_for_backward(x, w)
        ctx.has_b = b is not None
        y = mm8(x8, w8, sx, sw, None if b is None else b.to(torch.bfloat16))
        y = y.view(B, x.shape[0] // B, 3, H, -1).permute(2, 0, 3, 1, 4)
        return y[0], y[1], y[2]

    @staticmethod
    def backward(ctx, dq, dk, dv):
        x, w = ctx.saved_tensors
        d, d8, sd, db = qkv_q(dq, dk, dv)
        wt8, swt = quant(w.t())
        return mm8(d8, wt8, sd, swt), d.t() @ x, db if ctx.has_b else None, None, None


class Linear8(torch.autograd.Function):
    # y = x W^T + b with x [M, K] bf16, W [N, K], b [N]: fp8 fprop / dgrad, bf16 wgrad
    @staticmethod
    def forward(ctx, x, w, b):
        x8, sx = quant(x)
        w8, sw = quant(w)
        ctx.save_for_backward(x, w)
        ctx.has_b = b is not None
        return mm8(x8, w8, sx, sw, None if b is None else b.to(torch.bfloat16))

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dy8, sdy = quant(dy)
        wt8, swt = quant(w.t())
        dx = mm8(dy8, wt8, sdy, swt)
        return dx, dy.t() @ x, dy.sum(0) if ctx.has_b else None


class Mlp8(torch.autograd.Function):
    # fc2(gelu_tanh(fc1(x))) on bf16 [M, D] rows: fp8 fprop / dgrad, bf16 wgrad. GELU (+ the rowwise cast of its
    # output) and GELU' (+ the cast, the fc1 bias grad) are one triton pass each
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        x8, sx = quant(x)
        w18, s1 = quant(w1)
        h = mm8(x8, w18, sx, s1, b1.to(torch.bfloat16))
        a, a8, sa = gelu_q(h)
        w28, s2 = quant(w2)
        ctx.save_for_backward(x, h, a, w1, w2)
        return mm8(a8, w28, sa, s2, b2.to(torch.bfloat16))

    @staticmethod
    def backward(ctx, dy):
        x, h, a, w1, w2 = ctx.saved_tensors
        dy8, sdy = quant(dy)
        w2t8, s2t = quant(w2.t())
        dh, dh8, sdh, db1 = dgelu_q(mm8(dy8, w2t8, sdy, s2t), h)
        w1t8, s1t = quant(w1.t())
        dx = mm8(dh8, w1t8, sdh, s1t)
        return dx, dh.t() @ x, db1, dy.t() @ a, dy.sum(0)


def linear8(x: Tensor, lin: torch.nn.Linear) -> Tensor:
    y = Linear8.apply(x.reshape(-1, x.shape[-1]).to(torch.bfloat16), lin.weight, lin.bias)
    return y.view(*x.shape[:-1], -1)


def mlp8(x: Tensor, mlp: torch.nn.Module) -> Tensor:
    y = Mlp8.apply(x.reshape(-1, x.shape[-1]).to(torch.bfloat16), mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight, mlp.fc2.bias)
    return y.view(*x.shape[:-1], -1)


# muon: X X^T of a stack of bf16 [b, m, n] matrices (m a multiple of 128) computing only the upper triangle of 128 x 128
# output tiles and writing each also mirrored (the product is symmetric: the mirrored tile is the same fp32 dot products)
@triton.jit
def _syrk_kernel(x_ptr, o_ptr, ti_ptr, tj_ptr, K, M: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
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
    c = acc.to(tl.bfloat16)
    ob = o_ptr + b * M * M
    tl.store(ob + rm[:, None] * M + rn[None, :], c)
    if i != j:
        tl.store(ob + rn[:, None] * M + rm[None, :], tl.trans(c))


_TRI = {}


@torch.library.custom_op("nanobrain::syrk", mutates_args=())
def syrk(x: Tensor) -> Tensor:
    # x @ x.mT for a contiguous bf16 [b, m, n] stack
    b, m, n = x.shape
    out = torch.empty(b, m, m, device=x.device, dtype=x.dtype)
    T = m // 128
    if (T, x.device) not in _TRI:
        ij = [(i, j) for i in range(T) for j in range(i, T)]
        _TRI[(T, x.device)] = tuple(torch.tensor(v, dtype=torch.int32, device=x.device) for v in zip(*ij))
    ti, tj = _TRI[(T, x.device)]
    _syrk_kernel[(len(ti), b)](x, out, ti, tj, n, m, 128, 64, num_warps=4, num_stages=3)
    return out


@syrk.register_fake
def _(x):
    return x.new_empty(x.shape[0], x.shape[1], x.shape[1])
