"""native_sparse_tokens (speed): the native prep without dense volumes. A batch stays as its int16 in-mask values
plus the mask as a bitmap with per-word prefix counts (built from the run lengths on the gpu); every token voxel the
model reads (each spacing group's mask fractions, the sampled sequences, the swap partners) is looked up there in
one triton kernel: ras reorientation, box fit, normalization and the synthetic tumour paste applied per voxel. Same
values as native_batch_transform + native_tokens' dense volumes (the per-image mean / std are fp64 sums, the
tumour's knot noise a direct trilinear read: float-rounding differences only)."""

import math

import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice

from .data import native_spacing
from .prep import ras_orientation


@triton.jit
def _moments_kernel(vals, blk, scale, clip, mean, out, BLOCK: tl.constexpr, CENTER: tl.constexpr):
    # per block of one image's values: sum of clamp(v * scale) (or of its squared deviation from the mean)
    pid = tl.program_id(0)
    img = tl.load(blk + pid * 3)
    start = tl.load(blk + pid * 3 + 1)
    end = tl.load(blk + pid * 3 + 2)
    offs = start + tl.arange(0, BLOCK)
    m = offs < end
    v = tl.load(vals + offs, mask=m, other=0).to(tl.float32) * tl.load(scale + img)
    v = tl.minimum(tl.maximum(v, tl.load(clip + img * 2)), tl.load(clip + img * 2 + 1))
    if CENTER:
        d = v - tl.load(mean + img)
        v = d * d
    tl.store(out + pid, tl.sum(tl.where(m, v, 0.0), axis=0))


@triton.jit
def _bits_kernel(runs, ends, blk, bits, cnt, BLOCK: tl.constexpr):
    # the true (odd) runs of a block of one image's runs as bits of the global voxel bitmap + per word counts
    pid = tl.program_id(0)
    first = tl.load(blk + pid * 3)
    start = tl.load(blk + pid * 3 + 1)
    end = tl.load(blk + pid * 3 + 2)
    r = start + tl.arange(0, BLOCK)
    m = (r < end) & (((r - first) % 2) == 1)
    ln = tl.load(runs + r, mask=m, other=0).to(tl.int64)
    e = tl.load(ends + r, mask=m, other=0)
    s = e - ln
    m = m & (ln > 0)
    w0 = s >> 5
    for j in tl.static_range(9):  # a run of <= 255 voxels spans <= 9 words
        w = w0 + j
        lo = tl.maximum(s, w * 32)
        hi = tl.minimum(e, w * 32 + 32)
        mm = m & (hi > lo)
        n = hi - lo
        word = (((tl.full(n.shape, 1, tl.int64) << n) - 1) << (lo - w * 32)) & 0xFFFFFFFF
        tl.atomic_or(bits + w, word.to(tl.int32), mask=mm)
        tl.atomic_add(cnt + w, n.to(tl.int32), mask=mm)


@triton.jit
def _lookup(f, valid, bits, pre, vals, img, nrm):
    # (normalized value, in mask) of global voxels f
    w = f >> 5
    bit = (f & 31).to(tl.int32)
    word = tl.load(bits + w, mask=valid, other=0)
    inm = valid & (((word >> bit) & 1) == 1)
    below = word & ((tl.full(bit.shape, 1, tl.int32) << bit) - 1)
    vidx = tl.load(pre + w, mask=inm, other=0) + libdevice.popc(below)
    v = tl.load(vals + vidx, mask=inm, other=0).to(tl.float32) * tl.load(nrm + img * 5)
    v = tl.minimum(tl.maximum(v, tl.load(nrm + img * 5 + 1)), tl.load(nrm + img * 5 + 2))
    v = (v - tl.load(nrm + img * 5 + 3)) / tl.load(nrm + img * 5 + 4)
    return tl.where(inm, v, 0.0), inm


@triton.jit
def _token_voxels(tid, q, geo, img, PX, PY, PZ, GY, GZ):
    # global voxel index of voxel q of box token tid (row major tokens and voxels), and whether it is in the image
    t0 = tid // (GY * GZ)
    t1 = (tid // GZ) % GY
    t2 = tid % GZ
    b0 = t0 * PX + q // (PY * PZ)
    b1 = t1 * PY + (q // PZ) % PY
    b2 = t2 * PZ + q % PZ
    g = geo + img * 10
    valid = (b0 >= tl.load(g + 4)) & (b0 < tl.load(g + 5)) & (b1 >= tl.load(g + 6)) & (b1 < tl.load(g + 7))
    valid = valid & (b2 >= tl.load(g + 8)) & (b2 < tl.load(g + 9))
    f = tl.load(g) + tl.load(g + 1) * b0 + tl.load(g + 2) * b1 + tl.load(g + 3) * b2
    return f, valid, b0, b1, b2


@triton.jit
def _rank(f, bits, pre, m):
    # in-mask voxels before global voxel f (the value index of f)
    w = f >> 5
    word = tl.load(bits + w, mask=m, other=0)
    below = word & ((tl.full(f.shape, 1, tl.int32) << (f & 31).to(tl.int32)) - 1)
    return tl.load(pre + w, mask=m, other=0) + libdevice.popc(below)


@triton.jit(do_not_specialize=["NT", "P", "PX", "PY", "PZ", "GY", "GZ", "NUM"])
def _frac_kernel(rimg, geo, line, bits, pre, out, NT, P, PX, PY, PZ, GY, GZ, NUM, TB: tl.constexpr, LB: tl.constexpr):
    # in-mask voxel fraction of every box token of the group's images (exact count / P), TB tokens per program: each
    # token's voxel lines along the image's storage-contiguous ras axis (coef +-1) counted as two ranks of the bitmap
    tok = tl.program_id(0) * TB + tl.arange(0, TB)
    tm = tok < NT
    img = tl.load(rimg + tok // NUM, mask=tm, other=0)
    tid = tok % NUM
    g = geo + img * 10
    la = tl.load(line + img, mask=tm, other=0)[:, None]  # the ras axis with coef +-1
    t0 = (tid // (GY * GZ))[:, None]
    t1 = ((tid // GZ) % GY)[:, None]
    t2 = (tid % GZ)[:, None]
    lo0 = tl.load(g + 4, mask=tm, other=0)[:, None]
    hi0 = tl.load(g + 5, mask=tm, other=0)[:, None]
    lo1 = tl.load(g + 6, mask=tm, other=0)[:, None]
    hi1 = tl.load(g + 7, mask=tm, other=0)[:, None]
    lo2 = tl.load(g + 8, mask=tm, other=0)[:, None]
    hi2 = tl.load(g + 9, mask=tm, other=0)[:, None]
    f0 = tl.load(g, mask=tm, other=0)[:, None]
    k0 = tl.load(g + 1, mask=tm, other=0)[:, None]
    k1 = tl.load(g + 2, mask=tm, other=0)[:, None]
    k2 = tl.load(g + 3, mask=tm, other=0)[:, None]
    PA = tl.where(la == 0, PY, PX)
    PB = tl.where(la == 2, PY, PZ)
    PL = tl.where(la == 0, PX, tl.where(la == 1, PY, PZ))
    c = tl.where(la == 0, k0, tl.where(la == 1, k1, k2))
    llo = tl.where(la == 0, lo0, tl.where(la == 1, lo1, lo2))
    lhi = tl.where(la == 0, hi0, tl.where(la == 1, hi1, hi2))
    acc = tl.zeros([TB, LB], dtype=tl.int32)
    nl = P // tl.minimum(tl.minimum(PX, PY), PZ)  # >= every token's line count
    for i0 in range(0, nl, LB):
        i = i0 + tl.arange(0, LB)[None, :]
        ia = i // PB
        ib = i % PB
        b0 = t0 * PX + tl.where(la == 0, 0, ia)
        b1 = t1 * PY + tl.where(la == 0, ia, tl.where(la == 1, 0, ib))
        b2 = t2 * PZ + tl.where(la == 2, 0, ib)
        ok = tm[:, None] & (i < PA * PB)
        ok = ok & ((la == 0) | ((b0 >= lo0) & (b0 < hi0)))
        ok = ok & ((la == 1) | ((b1 >= lo1) & (b1 < hi1)))
        ok = ok & ((la == 2) | ((b2 >= lo2) & (b2 < hi2)))
        base_l = tl.where(la == 0, b0, tl.where(la == 1, b1, b2))
        s = tl.maximum(base_l, llo)
        e = tl.minimum(base_l + PL, lhi)
        ok = ok & (e > s)
        fb = f0 + k0 * b0 + k1 * b1 + k2 * b2 - c * base_l
        fs = fb + c * s
        fe = fb + c * (e - 1)
        n = _rank(tl.maximum(fs, fe) + 1, bits, pre, ok) - _rank(tl.minimum(fs, fe), bits, pre, ok)
        acc += tl.where(ok, n, 0)
    tl.store(out + tok, tl.sum(acc, axis=1).to(tl.float32) / P, mask=tm)


@triton.jit(do_not_specialize=["per_img", "P", "PX", "PY", "PZ", "GY", "GZ", "NK"])
def _gather_kernel(
    tok, rimg, per_img, geo, nrm, bits, pre, vals, mass, ti, tf, knots, img_out, msk_out, les_out,
    P, PX, PY, PZ, GY, GZ, NK, TUMOR: tl.constexpr, LES: tl.constexpr, BLOCK: tl.constexpr,
):
    # rows of (image, box token): the normalized image / mask voxels (+ the pasted tumour, its soft mask)
    row = tl.program_id(0)
    q = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    qm = q < P
    tid = tl.load(tok + row)
    img = tl.load(rimg + row // per_img)
    f, valid, b0, b1, b2 = _token_voxels(tid, q, geo, img, PX, PY, PZ, GY, GZ)
    v, inm = _lookup(f, valid & qm, bits, pre, vals, img, nrm)
    les = tl.zeros([BLOCK], dtype=tl.float32)
    if TUMOR:
        k = tl.load(mass + img)
        if k >= 0:
            # plan_tumors' mass value at these box voxels (the knot noise trilinear on the mass' 5^3 block)
            l0 = b0 - tl.load(ti + k * 6)
            l1 = b1 - tl.load(ti + k * 6 + 1)
            l2 = b2 - tl.load(ti + k * 6 + 2)
            inbox = (l0 >= 0) & (l0 < tl.load(ti + k * 6 + 3)) & (l1 >= 0) & (l1 < tl.load(ti + k * 6 + 4))
            inbox = inbox & (l2 >= 0) & (l2 < tl.load(ti + k * 6 + 5))
            fp = tf + k * 19
            r0 = (l0.to(tl.float32) - tl.load(fp + 14)) * tl.load(fp)
            r1 = (l1.to(tl.float32) - tl.load(fp + 15)) * tl.load(fp + 1)
            r2 = (l2.to(tl.float32) - tl.load(fp + 16)) * tl.load(fp + 2)
            a0 = r0 * tl.load(fp + 3)
            a1 = r1 * tl.load(fp + 4)
            a2 = r2 * tl.load(fp + 5)
            d = tl.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            ib = tl.load(fp + 6)
            u0 = (tl.minimum(tl.maximum(r0 * ib, -1.0), 1.0) + 1.0) * 2.0
            u1 = (tl.minimum(tl.maximum(r1 * ib, -1.0), 1.0) + 1.0) * 2.0
            u2 = (tl.minimum(tl.maximum(r2 * ib, -1.0), 1.0) + 1.0) * 2.0
            i0 = tl.minimum(u0.to(tl.int32), 3)
            i1 = tl.minimum(u1.to(tl.int32), 3)
            i2 = tl.minimum(u2.to(tl.int32), 3)
            f0 = u0 - i0.to(tl.float32)
            f1 = u1 - i1.to(tl.float32)
            f2 = u2 - i2.to(tl.float32)
            base = knots + (k * 5 + i0) * 25 + i1 * 5 + i2
            bump = tl.zeros([BLOCK], dtype=tl.float32)
            thick = tl.zeros([BLOCK], dtype=tl.float32)
            for c in tl.static_range(8):
                w = (f0 if c & 4 else 1.0 - f0) * (f1 if c & 2 else 1.0 - f1) * (f2 if c & 1 else 1.0 - f2)
                o = (25 if c & 4 else 0) + (5 if c & 2 else 0) + (1 if c & 1 else 0)
                bump += w * tl.load(base + o, mask=qm, other=0.0)
                thick += w * tl.load(base + NK * 125 + o, mask=qm, other=0.0)
            depth = (1.0 - d * (1.0 + bump)) * tl.load(fp + 7)
            outer = tl.sigmoid(depth * tl.load(fp + 8))
            core = tl.sigmoid((depth - tl.load(fp + 9) * (1.0 + 2.0 * thick)) * 1.1) * tl.load(fp + 10)
            val = tl.load(fp + 11) * (outer - core) - tl.load(fp + 12) * core
            inside = inm & inbox
            new = tl.minimum(tl.maximum(v + val, tl.load(fp + 17)), tl.load(fp + 18))
            v = tl.where(inside, new, v)
            les = tl.where(inside, outer, 0.0)
    o = row * P + q
    tl.store(img_out + o, v, mask=qm)
    tl.store(msk_out + o, inm.to(tl.float32), mask=qm)
    if LES:
        tl.store(les_out + o, les, mask=qm)


@triton.jit(do_not_specialize=["R", "P", "T"])
def _pool_kernel(x, m, les, thr, kb, kf, out, R, P, T, BR: tl.constexpr, BK: tl.constexpr, NB: tl.constexpr):
    # bin sums of channel c of rows [R, P] over the adaptive pool bins (0/1 indicator kb / kf [P, NB]):
    # c = 0 the mask, 1..T (x > thr) x mask, T + 1 lesion x mask (0/1 channels exact in bf16, the lesion in tf32x3)
    rb = tl.program_id(0)
    c = tl.program_id(1)
    rows = rb * BR + tl.arange(0, BR)
    rm = rows < R
    acc = tl.zeros([BR, NB], dtype=tl.float32)
    t = tl.load(thr + tl.maximum(c - 1, 0), mask=(c >= 1) & (c <= T), other=0.0)
    for k0 in range(0, P, BK):
        ks = k0 + tl.arange(0, BK)
        km = ks < P
        o = rows[:, None] * P + ks[None, :]
        lm = rm[:, None] & km[None, :]
        mv = tl.load(m + o, mask=lm, other=0.0)
        if c == T + 1:
            a = tl.load(les + o, mask=lm, other=0.0) * mv
            kf_ = tl.load(kf + ks[:, None] * NB + tl.arange(0, NB)[None, :], mask=km[:, None], other=0.0)
            acc += tl.dot(a, kf_, input_precision="tf32x3")
        else:
            if c > 0:
                mv = tl.where(tl.load(x + o, mask=lm, other=0.0) > t, mv, 0.0)
            kb_ = tl.load(kb + ks[:, None] * NB + tl.arange(0, NB)[None, :], mask=km[:, None], other=0.0)
            acc += tl.dot(mv.to(tl.bfloat16), kb_)
    oo = (rows[:, None] * (T + 2) + c) * NB + tl.arange(0, NB)[None, :]
    tl.store(out + oo, acc, mask=rm[:, None])


def pool_targets(x: Tensor, m: Tensor, les: Tensor | None, thr: Tensor, patch, sub: int) -> Tensor:
    # sep_pool over (mask, (x > thr_t) x mask, lesion x mask) of rows [R, P] in one kernel: [R, T + 2, sub^3]
    from .model import pool_kron

    P = math.prod(patch)
    x, m = x.reshape(-1, P).contiguous(), m.reshape(-1, P).contiguous()
    R, T = x.shape[0], thr.numel()
    k, lx, ly, lz = pool_kron(tuple(patch), sub, x.device)
    kf = k.t().contiguous()
    out = torch.empty(R, T + 2, sub**3, device=x.device, dtype=torch.float32)
    C = T + 2 if les is not None else T + 1
    _pool_kernel[(triton.cdiv(R, 64), C)](
        x, m, les.reshape(-1, P).contiguous() if les is not None else m, thr.float().contiguous(), kf.to(torch.bfloat16),
        kf, out, R, P, T, BR=64, BK=64, NB=sub**3,
    )
    out = out.view(R, T + 2, sub, sub, sub)
    return out / lx.view(-1, 1, 1) / ly.view(1, -1, 1) / lz.view(1, 1, -1)


@triton.jit
def _rnd(v, BF16: tl.constexpr):
    # the autocast compute dtype's rounding (corrupt_apply_batched's rnd)
    if BF16:
        return v.to(tl.bfloat16).to(tl.float32)
    return v


@triton.jit(do_not_specialize=["seed", "NV", "K", "P", "PX", "PY", "PZ", "KX", "KY", "KZ"])
def _corrupt_kernel(
    vis, msk, out, cand, kind, lo, seed, NV, K, P, PX, PY, PZ, KX, KY, KZ, CB: tl.constexpr, BF16: tl.constexpr,
):
    # corrupt_apply_batched's per candidate math, one program per candidate token (kind 0 = untouched), CB voxels at
    # a time: 1 bias ramp, 2 masked gaussian blur (sigma 1 voxel, replicate padded, the three passes as one 5^3
    # stencil), 3 rician noise, 4 gamma, 5 2x block mean. random draws from the philox stream (seed, candidate)
    cid = tl.program_id(0)
    kd = tl.load(kind + cid)
    if kd > 0:
        r = cid // K
        base = (r * NV + tl.load(cand + cid)).to(tl.int64) * P
        L = tl.load(lo + r)
        roff = cid.to(tl.int64) * 8
        # per token statistics first (noise: in-mask mean / std; gamma: in-mask range)
        s1 = tl.zeros([CB], dtype=tl.float32)
        s2 = tl.zeros([CB], dtype=tl.float32)
        big = tl.full([CB], -float("inf"), tl.float32)
        small = tl.full([CB], float("inf"), tl.float32)
        if kd == 3 or kd == 4:
            for q0 in range(0, P, CB):
                q = q0 + tl.arange(0, CB)
                qm = q < P
                x = tl.load(vis + base + q, mask=qm, other=0.0)
                m = tl.load(msk + base + q, mask=qm, other=0.0)
                s1 += m
                s2 += x * m
                inb = (m > 0) & qm
                big = tl.maximum(big, tl.where(inb, x, -float("inf")))
                small = tl.minimum(small, tl.where(inb, x, float("inf")))
        cnt = tl.maximum(tl.sum(s1, axis=0), 1.0)
        mean = tl.sum(s2, axis=0) / cnt
        sd = 0.0
        if kd == 3:
            s3 = tl.zeros([CB], dtype=tl.float32)
            for q0 in range(0, P, CB):
                q = q0 + tl.arange(0, CB)
                qm = q < P
                x = tl.load(vis + base + q, mask=qm, other=0.0)
                m = tl.load(msk + base + q, mask=qm, other=0.0)
                s3 += (x - mean) * (x - mean) * m
            sd = 0.1 * tl.sqrt(tl.sum(s3, axis=0) / cnt)
        bg = tl.max(big, axis=0)
        sm_ = tl.min(small, axis=0)
        fin = tl.sum(s1, axis=0) > 0
        span = tl.where(fin, tl.maximum(bg - sm_, 1e-6), 1.0)
        sm_ = tl.where(fin, sm_, 0.0)
        u0 = tl.randn(seed, roff)
        u1 = tl.randn(seed, roff + 1)
        u2 = tl.randn(seed, roff + 2)
        nrm = tl.maximum(tl.abs(u0) + tl.abs(u1) + tl.abs(u2), 1e-6)
        u0 = _rnd(u0 / nrm, BF16)
        u1 = _rnd(u1 / nrm, BF16)
        u2 = _rnd(u2 / nrm, BF16)
        amp = 0.2 + 0.1 * tl.rand(seed, roff + 3)
        g = tl.exp(tl.where(tl.rand(seed, roff + 4) < 0.5, -1.0, 1.0) * (0.2 + 0.3 * tl.rand(seed, roff + 5)))
        for q0 in range(0, P, CB):
            q = q0 + tl.arange(0, CB)
            qm = q < P
            x = tl.load(vis + base + q, mask=qm, other=0.0)
            m = tl.load(msk + base + q, mask=qm, other=0.0)
            c0 = q // (PY * PZ)
            c1 = (q // PZ) % PY
            c2 = q % PZ
            res = x
            if kd == 1:
                r0 = _rnd((c0.to(tl.float32) + 0.5) / PX * 2 - 1, BF16)
                r1 = _rnd((c1.to(tl.float32) + 0.5) / PY * 2 - 1, BF16)
                r2 = _rnd((c2.to(tl.float32) + 0.5) / PZ * 2 - 1, BF16)
                res = (x - L) * (1 + amp * _rnd(u0 * r0 + u1 * r1 + u2 * r2, BF16)) + L
            elif kd == 2:
                num = tl.zeros([CB], dtype=tl.float32)
                den = tl.zeros([CB], dtype=tl.float32)
                for t in range(125):
                    a = t // 25
                    bb = (t // 5) % 5
                    cc = t % 5
                    wa = tl.where(PX >= 2, _rnd(tl.exp(-((a - 2.0) * (a - 2.0)) / 2) / 2.4837318859, BF16), tl.where(a == 2, 1.0, 0.0))
                    wb = tl.where(PY >= 2, _rnd(tl.exp(-((bb - 2.0) * (bb - 2.0)) / 2) / 2.4837318859, BF16), tl.where(bb == 2, 1.0, 0.0))
                    wc = tl.where(PZ >= 2, _rnd(tl.exp(-((cc - 2.0) * (cc - 2.0)) / 2) / 2.4837318859, BF16), tl.where(cc == 2, 1.0, 0.0))
                    w = wa * wb * wc
                    if w != 0.0:
                        ia = tl.minimum(tl.maximum(c0 + (a - 2), 0), PX - 1)
                        ib = tl.minimum(tl.maximum(c1 + (bb - 2), 0), PY - 1)
                        ic = tl.minimum(tl.maximum(c2 + (cc - 2), 0), PZ - 1)
                        o = base + (ia * PY + ib) * PZ + ic
                        xv = tl.load(vis + o, mask=qm, other=0.0)
                        mv = tl.load(msk + o, mask=qm, other=0.0)
                        num += w * _rnd(xv * mv, BF16)
                        den += w * _rnd(mv, BF16)
                res = _rnd(_rnd(num, BF16) / tl.maximum(_rnd(den, BF16), 1e-3), BF16)
            elif kd == 3:
                off = cid.to(tl.int64) * 16384 + q * 2
                n1 = tl.randn(seed + 1, off)
                n2 = tl.randn(seed + 1, off + 1)
                a_ = x - L + n1 * sd
                res = tl.sqrt(a_ * a_ + (n2 * sd) * (n2 * sd)) + L
            elif kd == 4:
                v = tl.minimum(tl.maximum((x - sm_) / span, 0.0), 1.0)
                res = sm_ + span * libdevice.pow(v, g)
            elif kd == 5:
                s0 = (c0 // KX) * KX
                b1 = (c1 // KY) * KY
                b2 = (c2 // KZ) * KZ
                sxm = tl.zeros([CB], dtype=tl.float32)
                smm = tl.zeros([CB], dtype=tl.float32)
                n = tl.zeros([CB], dtype=tl.float32)
                for o in range(8):
                    d0 = (o >> 2) & 1
                    d1 = (o >> 1) & 1
                    d2 = o & 1
                    ok = qm & (d0 < KX) & (d1 < KY) & (d2 < KZ) & (s0 + d0 < PX) & (b1 + d1 < PY) & (b2 + d2 < PZ)
                    oo = base + ((s0 + d0) * PY + (b1 + d1)) * PZ + b2 + d2
                    xv = tl.load(vis + oo, mask=ok, other=0.0)
                    mv = tl.load(msk + oo, mask=ok, other=0.0)
                    sxm += xv * mv
                    smm += mv
                    n += ok.to(tl.float32)
                n = tl.maximum(n, 1.0)
                res = (sxm / n) / tl.maximum(smm / n, 1e-3)
            tl.store(out + base + q, res * m, mask=qm)


def corrupt_tokens(vis: Tensor, vmask: Tensor, patch, cand: Tensor, kind: Tensor, min_vox: int) -> Tensor:
    # corrupt_apply_batched for one group in one kernel (the same per candidate math; draws from triton's philox)
    rows, nv, P = vis.shape
    k = cand.shape[1]
    lo = torch.where(vmask > 0, vis, math.inf).amin(dim=(1, 2))
    lo = torch.where(torch.isfinite(lo), lo, 0.0).float().contiguous()
    out = vis.clone()
    ks = [2 if p >= min_vox else 1 for p in patch]
    bf16 = torch.is_autocast_enabled(vis.device.type) and torch.get_autocast_dtype(vis.device.type) == torch.bfloat16
    seed = int(torch.randint(0, 2**30, (1,)).item())
    _corrupt_kernel[(rows * k,)](
        vis.contiguous(), vmask.contiguous(), out, cand.contiguous(), kind.contiguous(), lo, seed, nv, k, P, *patch, *ks,
        CB=512, BF16=bf16,
    )
    return out


def _blocks(counts: list[int], block: int, first: list[int] | None = None) -> np.ndarray:
    # per block of `block` items of one image: (image or its first item, start, end) on the concatenated items
    rows, off = [], 0
    for i, n in enumerate(counts):
        starts = np.arange(off, off + n, block)
        rows.append(np.stack([np.full(len(starts), i if first is None else first[i]), starts,
                              np.minimum(starts + block, off + n)], 1))
        off += n
    return np.concatenate(rows).astype(np.int64)


def _to(dev, a, dtype=None) -> Tensor:
    t = torch.as_tensor(a, dtype=dtype)
    return (t.pin_memory() if dev.type == "cuda" else t).to(dev, non_blocking=True)


class SparseVolumes:
    """A native_concat batch on the gpu as (values, bitmap): ras geometry per image on the host, normalization and
    the bitmap on the device. Stands in for native_batch_transform's dense image / mask lists."""

    def __init__(self, batch: dict):
        samples = batch["samples"]
        vals = batch["values"]
        dev = self.device = vals.device
        self.B = B = len(samples)
        nv = [s["n_values"] for s in samples]
        nr = [s["n_runs"] for s in samples]
        shapes = [tuple(int(v) for v in s["shape"]) for s in samples]
        vox = [math.prod(sh) for sh in shapes]
        self.vals = vals
        # normalization: (clamp(v * scale, clip) - mean) / std per image, the moments as fp64 sums of block sums
        blk = _to(dev, _blocks(nv, 4096))
        img_of = blk[:, 0]
        scale, clip = batch["scale"], batch["clip"]
        n = batch["num_values"].double()
        part = torch.empty(len(blk), device=dev, dtype=torch.float32)
        _moments_kernel[(len(blk),)](vals, blk, scale, clip, scale, part, BLOCK=4096, CENTER=False)
        mean = (torch.zeros(B, device=dev, dtype=torch.float64).index_add_(0, img_of, part.double()) / n).float()
        _moments_kernel[(len(blk),)](vals, blk, scale, clip, mean, part, BLOCK=4096, CENTER=True)
        var = torch.zeros(B, device=dev, dtype=torch.float64).index_add_(0, img_of, part.double()) / n
        std = var.sqrt().float().clamp_min(1e-6)
        self.nrm = torch.stack([scale, clip[:, 0], clip[:, 1], mean, std], 1).contiguous()
        self.lims = list(((clip - mean[:, None]) / std[:, None]).unbind(0))
        # the mask bitmap over the concatenated voxels (+ per word in-mask counts before it = the value index)
        runs = batch["mask_rle"]
        ends = runs.long().cumsum(0)
        words = -(-sum(vox) // 32) + 1
        self.bits = torch.zeros(words, device=dev, dtype=torch.int32)
        cnt = torch.zeros(words, device=dev, dtype=torch.int32)
        first = np.concatenate([[0], np.cumsum(nr)[:-1]]).tolist()
        rblk = _to(dev, _blocks(nr, 1024, first))
        _bits_kernel[(len(rblk),)](runs, ends, rblk, self.bits, cnt, BLOCK=1024)
        self.pre = cnt.cumsum(0, dtype=torch.int32) - cnt
        # per image ras geometry: ras shape and the ras voxel -> global voxel map (const + coef . r)
        self.ras, self.const, self.coef, self.spacing = [], [], [], []
        off = 0
        for s, sh, nvox in zip(samples, shapes, vox):
            flip_axes, order, affine = s["ras"] if "ras" in s else ras_orientation(s["affine"], sh)
            stride = (sh[1] * sh[2], sh[2], 1)
            self.ras.append(tuple(sh[o] for o in order))
            self.coef.append([-stride[o] if o in flip_axes else stride[o] for o in order])
            self.const.append(off + sum((sh[k] - 1) * stride[k] for k in flip_axes))
            self.spacing.append(affine)
            off += nvox

    def mask_at(self, owners: list[int], ras: Tensor) -> Tensor:
        # bool mask at ras voxels ras [k, ..., 3] (long) of the images owners (len k)
        g = _to(self.device, [[self.const[i], *self.coef[i]] for i in owners], torch.int64)
        f = g[:, 0].view(-1, *[1] * (ras.dim() - 2)) + (ras * g[:, 1:].view(-1, *[1] * (ras.dim() - 2), 3)).sum(-1)
        word = self.bits[f >> 5]
        return ((word >> (f & 31).int()) & 1).bool()

    def geometry(self, groups: dict, geoms: dict) -> None:
        # per image (batch order) its box fit for its group: const, coef (3), box lo / hi per axis
        from .model import fit_slices

        rows = [None] * self.B
        for sp, ids in groups.items():
            box = geoms[sp][1]
            for ii in ids:
                src, dst = fit_slices(self.ras[ii], box)
                c = self.const[ii] + sum(cf * (s.start - d.start) for cf, s, d in zip(self.coef[ii], src, dst))
                rows[ii] = [c, *self.coef[ii], *(v for d in dst for v in (d.start, d.stop))]
        self.geo = _to(self.device, rows, torch.int64)
        self.line = _to(self.device, [[abs(c) for c in cf].index(1) for cf in self.coef], torch.int32)
        self.mass = None

    def tumors(self, plan: dict | None) -> None:
        # plan_tumors' sparse plan: per mass box (lo, num) and its float parameters, the knot noise
        if not plan:
            return
        self.mass = torch.full((self.B,), -1, dtype=torch.int32)
        self.mass[plan["owners"]] = torch.arange(len(plan["owners"]), dtype=torch.int32)
        self.mass = _to(self.device, self.mass)
        self.ti = torch.cat([plan["lo"], plan["num"]], 1).contiguous()
        self.tf = plan["par"].contiguous()
        self.knots = plan["knots"].contiguous()

    def group(self, ids: list[int], patch, grid) -> "SparseGroup":
        return SparseGroup(self, ids, tuple(patch), tuple(grid))


class SparseGroup:
    """One spacing group's images as LazyTokens stand-in: frac() and the row gathers."""

    def __init__(self, vol: SparseVolumes, ids: list[int], patch, grid):
        self.vol, self.ids, self.patch, self.grid = vol, ids, patch, grid
        self.num = math.prod(grid)
        self.P = math.prod(patch)
        self.rimg = _to(vol.device, ids, torch.int64)

    def _args(self):
        (px, py, pz), (_, gy, gz) = self.patch, self.grid
        return self.P, px, py, pz, gy, gz

    def frac(self) -> Tensor:
        v = self.vol
        out = torch.empty(len(self.ids), self.num, device=v.device, dtype=torch.float32)
        _frac_kernel[(triton.cdiv(out.numel(), 32),)](
            self.rimg, v.geo, v.line, v.bits, v.pre, out, out.numel(), *self._args(), self.num, TB=32, LB=32,
        )
        return out

    def gather(self, tok: Tensor, rimg: Tensor, per_img: int, lesion: bool = False):
        # rows (rimg[row // per_img], tok[row]) -> image, mask (float 0 / 1) [R, P] (+ the soft lesion mask)
        v = self.vol
        tok = tok.reshape(-1).contiguous()
        R = tok.numel()
        img = torch.empty(R, self.P, device=v.device, dtype=torch.float32)
        msk = torch.empty_like(img)
        tumor = v.mass is not None
        les = torch.empty_like(img) if lesion and tumor else None
        block = 128 if self.P <= 128 else 512
        dummy = v.bits
        _gather_kernel[(R, triton.cdiv(self.P, block))](
            tok, rimg, per_img, v.geo, v.nrm, v.bits, v.pre, v.vals,
            v.mass if tumor else dummy, v.ti if tumor else dummy, v.tf if tumor else dummy,
            v.knots if tumor else dummy, img, msk, les if les is not None else img,
            *self._args(), (v.knots.shape[2] // 5) if tumor else 1,
            TUMOR=tumor, LES=les is not None, BLOCK=block,
        )
        return img, msk, les


def sparse_batch_transform(batch: dict, target_spacing: float = 1.0) -> dict:
    # native_batch_transform's output for native_sparse_tokens: the SparseVolumes as "image" (no dense volumes)
    samples = batch["samples"]
    vol = SparseVolumes(batch)
    out = {"image": vol, "mask": None, "lim": vol.lims}
    out["spacing"] = [native_spacing(affine, target_spacing) for affine in vol.spacing]
    if "fino" in samples[0]:
        out["fino"] = torch.from_numpy(np.stack([s["fino"] for s in samples]))
    if "subject" in samples[0]:
        out["subject"] = torch.tensor([s["subject"] for s in samples], dtype=torch.int64)
    return out
