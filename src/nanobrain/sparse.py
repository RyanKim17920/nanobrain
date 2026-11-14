"""native_sparse_tokens (speed): the native prep without dense volumes. A batch stays as its int16 in-mask values
plus the mask as a bitmap with per-word prefix counts (built from the run lengths on the gpu); every token voxel the
model reads (each spacing group's mask fractions, the sampled sequences, the swap partners) is looked up there in
one triton kernel: ras reorientation, box fit, normalization and the synthetic tumour paste applied per voxel. Same
values as native_batch_transform + native_tokens' dense volumes (the per-image mean / std are fp64 sums, the
tumour's knot noise a direct trilinear read: float-rounding differences only)."""

import functools
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


@triton.jit
def _group_of(row, gtab, G, NC: tl.constexpr, COL: tl.constexpr):
    # group of row(s): the number of groups j >= 1 whose first row (column COL of the group table) is <= row
    g = row * 0
    for j in range(1, G):
        g += (row >= tl.load(gtab + j * NC + COL)).to(g.dtype)
    return g


@triton.jit(do_not_specialize=["NT", "G"])
def _frac_kernel(gtab, G, rimg, geo, line, bits, pre, out, NT, TB: tl.constexpr, LB: tl.constexpr):
    # in-mask voxel fraction of every box token of every group (exact count / P), TB tokens per program. group table
    # rows (NC 9): first token, P, PX, PY, PZ, GY, GZ, NUM, first image id. each token's voxel lines along the image's
    # storage-contiguous ras axis (coef +-1) counted as two ranks of the bitmap
    tok = tl.program_id(0) * TB + tl.arange(0, TB)
    tm = tok < NT
    gp = gtab + _group_of(tok, gtab, G, 9, 0) * 9
    P = tl.load(gp + 1, mask=tm, other=1)[:, None]
    PX = tl.load(gp + 2, mask=tm, other=1)[:, None]
    PY = tl.load(gp + 3, mask=tm, other=1)[:, None]
    PZ = tl.load(gp + 4, mask=tm, other=1)[:, None]
    GY = tl.load(gp + 5, mask=tm, other=1)
    GZ = tl.load(gp + 6, mask=tm, other=1)
    NUM = tl.load(gp + 7, mask=tm, other=1)
    loc = tok - tl.load(gp, mask=tm, other=0)
    img = tl.load(rimg + tl.load(gp + 8, mask=tm, other=0) + loc // NUM, mask=tm, other=0)
    tid = loc % NUM
    g = geo + img * 10
    la = tl.load(line + img, mask=tm, other=0)[:, None]
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
    nl = tl.max(tl.where(tm[:, None], PA * PB, 0))
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
    tl.store(out + tok, tl.sum(acc, axis=1).to(tl.float32) / tl.reshape(P, [TB]), mask=tm)


@triton.jit(do_not_specialize=["G", "NK"])
def _gather_kernel(
    gtab, G, tok, rimg, geo, nrm, bits, pre, vals, mass, ti, tf, knots, img_out, msk_out, les_out, NK,
    TUMOR: tl.constexpr, LES: tl.constexpr, BLOCK: tl.constexpr,
):
    # one row of (image, box token) per program, rows of every group: the normalized image / mask voxels (+ the
    # pasted tumour, its soft mask). group table rows (NC 10): first row, P, PX, PY, PZ, GY, GZ, rows per image,
    # first image id, first output element
    row = tl.program_id(0)
    gp = gtab + _group_of(row, gtab, G, 10, 0) * 10
    P = tl.load(gp + 1)
    PX = tl.load(gp + 2)
    PY = tl.load(gp + 3)
    PZ = tl.load(gp + 4)
    GY = tl.load(gp + 5)
    GZ = tl.load(gp + 6)
    loc = row - tl.load(gp)
    tid = tl.load(tok + row)
    img = tl.load(rimg + tl.load(gp + 8) + loc // tl.load(gp + 7))
    obase = tl.load(gp + 9) + loc.to(tl.int64) * P
    k = tl.load(mass + img) if TUMOR else -1
    for q0 in range(0, P, BLOCK):
        q = q0 + tl.arange(0, BLOCK)
        qm = q < P
        f, valid, b0, b1, b2 = _token_voxels(tid, q, geo, img, PX, PY, PZ, GY, GZ)
        v, inm = _lookup(f, valid & qm, bits, pre, vals, img, nrm)
        les = tl.zeros([BLOCK], dtype=tl.float32)
        if TUMOR:
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
                    oc = (25 if c & 4 else 0) + (5 if c & 2 else 0) + (1 if c & 1 else 0)
                    bump += w * tl.load(base + oc, mask=qm, other=0.0)
                    thick += w * tl.load(base + NK * 125 + oc, mask=qm, other=0.0)
                depth = (1.0 - d * (1.0 + bump)) * tl.load(fp + 7)
                outer = tl.sigmoid(depth * tl.load(fp + 8))
                core = tl.sigmoid((depth - tl.load(fp + 9) * (1.0 + 2.0 * thick)) * 1.1) * tl.load(fp + 10)
                val = tl.load(fp + 11) * (outer - core) - tl.load(fp + 12) * core
                inside = inm & inbox
                new = tl.minimum(tl.maximum(v + val, tl.load(fp + 17)), tl.load(fp + 18))
                v = tl.where(inside, new, v)
                les = tl.where(inside, outer, 0.0)
        o = obase + q
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


@functools.lru_cache(maxsize=1024)
def _pool_mats(patch, sub: int, device):
    # model.pool_kron's 0/1 bin indicator as [P, sub^3] fp32 and bf16 (exact), + the bin lengths
    from .model import _cached, pool_kron

    k, lx, ly, lz = pool_kron(patch, sub, device)
    kf = k.t().contiguous()
    return _cached(kf.to(torch.bfloat16)), _cached(kf), lx, ly, lz


def pool_targets(x: Tensor, m: Tensor, les: Tensor | None, thr: Tensor, patch, sub: int) -> Tensor:
    # sep_pool over (mask, (x > thr_t) x mask, lesion x mask) of rows [R, P] in one kernel: [R, T + 2, sub^3]
    from .model import pool_kron

    P = math.prod(patch)
    x, m = x.reshape(-1, P).contiguous(), m.reshape(-1, P).contiguous()
    R, T = x.shape[0], thr.numel()
    kb, kf, lx, ly, lz = _pool_mats(tuple(patch), sub, x.device)
    out = torch.empty(R, T + 2, sub**3, device=x.device, dtype=torch.float32)
    C = T + 2 if les is not None else T + 1
    _pool_kernel[(triton.cdiv(R, 64), C)](
        x, m, les.reshape(-1, P).contiguous() if les is not None else m, thr.float().contiguous(), kb,
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


@triton.jit(do_not_specialize=["G", "R"])
def _rowmin_kernel(gtab, G, lo, R, CB: tl.constexpr):
    # each group's row minimum over the in-mask visible voxels (0 for a row without one). group table rows (NC 5):
    # first row, row length, vis pointer, mask pointer, (unused)
    row = tl.program_id(0)
    gp = gtab + _group_of(row, gtab, G, 5, 0) * 5
    n = tl.load(gp + 1)
    base = (row - tl.load(gp)).to(tl.int64) * n
    vis = tl.load(gp + 2).to(tl.pointer_type(tl.float32))
    msk = tl.load(gp + 3).to(tl.pointer_type(tl.float32))
    acc = tl.full([CB], float("inf"), tl.float32)
    for q0 in range(0, n, CB):
        q = q0 + tl.arange(0, CB)
        qm = q < n
        x = tl.load(vis + base + q, mask=qm, other=0.0)
        m = tl.load(msk + base + q, mask=qm, other=0.0)
        acc = tl.minimum(acc, tl.where(qm & (m > 0), x, float("inf")))
    v = tl.min(acc, axis=0)
    tl.store(lo + row, tl.where(v < float("inf"), v, 0.0))


@triton.jit(do_not_specialize=["seed", "G"])
def _corrupt_kernel(gtab, G, cand, kind, lo, seed, CB: tl.constexpr, BF16: tl.constexpr):
    # corrupt_apply_batched's per candidate math, one program per candidate token (kind 0 = untouched), CB voxels at
    # a time: 1 bias ramp, 2 masked gaussian blur (sigma 1 voxel, replicate padded, the three passes as one 5^3
    # stencil), 3 rician noise, 4 gamma, 5 2x block mean. random draws from the philox stream (seed, candidate)
    # group table rows (NC 14): first candidate, NV, K, P, PX, PY, PZ, KX, KY, KZ, vis / mask / out pointers, first row
    cid = tl.program_id(0)
    kd = tl.load(kind + cid)
    if kd > 0:
        gp = gtab + _group_of(cid, gtab, G, 14, 0) * 14
        NV = tl.load(gp + 1)
        K = tl.load(gp + 2)
        P = tl.load(gp + 3)
        PX = tl.load(gp + 4)
        PY = tl.load(gp + 5)
        PZ = tl.load(gp + 6)
        KX = tl.load(gp + 7)
        KY = tl.load(gp + 8)
        KZ = tl.load(gp + 9)
        vis = tl.load(gp + 10).to(tl.pointer_type(tl.float32))
        msk = tl.load(gp + 11).to(tl.pointer_type(tl.float32))
        out = tl.load(gp + 12).to(tl.pointer_type(tl.float32))
        r = (cid - tl.load(gp)) // K
        base = (r * NV + tl.load(cand + cid)).to(tl.int64) * P
        L = tl.load(lo + tl.load(gp + 13) + r)
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


def corrupt_all(items: list[tuple], min_vox: int) -> list[Tensor]:
    # corrupt_apply_batched in one kernel over every group's candidates (the same per candidate math; draws from
    # triton's philox): items (vis [rows, nv, P], mask, patch, cand [rows, k], kind) per group -> corrupted vis
    dev = items[0][0].device
    outs, keep, ctab, rtab, c0, r0 = [], [], [], [], 0, 0
    for vis, vmask, patch, cand, kind in items:
        rows, nv, P = vis.shape
        vis, vmask = vis.float().contiguous(), vmask.float().contiguous()
        keep += [vis, vmask]  # the pointer tables reference them until the kernels ran (stream order)
        out = vis.clone()
        outs.append(out)
        ks = [2 if p >= min_vox else 1 for p in patch]
        ctab.append([c0, nv, cand.shape[1], P, *patch, *ks, vis.data_ptr(), vmask.data_ptr(), out.data_ptr(), r0])
        rtab.append([r0, nv * P, vis.data_ptr(), vmask.data_ptr(), 0])
        c0 += cand.numel()
        r0 += rows
    ctab, rtab = _pack(dev, [ctab, rtab])
    lo = torch.empty(r0, device=dev, dtype=torch.float32)
    _rowmin_kernel[(r0,)](rtab, len(items), lo, r0, CB=1024)
    cand = torch.cat([it[3].reshape(-1) for it in items])
    kind = torch.cat([it[4].reshape(-1) for it in items])
    bf16 = torch.is_autocast_enabled(dev.type) and torch.get_autocast_dtype(dev.type) == torch.bfloat16
    seed = int(torch.randint(0, 2**30, (1,)).item())
    _corrupt_kernel[(c0,)](ctab, len(items), cand, kind, lo, seed, CB=512, BF16=bf16)
    return outs


def _blocks(counts: list[int], block: int, first: list[int] | None = None) -> np.ndarray:
    # per block of `block` items of one image: (image or its first item, start, end) on the concatenated items
    rows, off = [], 0
    for i, n in enumerate(counts):
        starts = np.arange(off, off + n, block)
        rows.append(np.stack([np.full(len(starts), i if first is None else first[i]), starts,
                              np.minimum(starts + block, off + n)], 1))
        off += n
    return np.concatenate(rows).astype(np.int64)


def _pack(dev, arrays: list) -> list[Tensor]:
    # int64 host arrays in one pinned upload, each view 16-byte aligned (triton specializes on pointer alignment)
    flat = [np.asarray(a, np.int64).ravel() for a in arrays]
    offs = np.cumsum([0] + [-(-len(f) // 2) * 2 for f in flat])
    buf = np.zeros(offs[-1], np.int64)
    for f, o in zip(flat, offs):
        buf[o : o + len(f)] = f
    t = _to(dev, buf)
    return [t[o : o + len(f)].view(np.shape(a)) for a, f, o in zip(arrays, flat, offs)]


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
        first = np.concatenate([[0], np.cumsum(nr)[:-1]]).tolist()
        blk, rblk = _pack(dev, [_blocks(nv, 4096), _blocks(nr, 1024, first)])
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
        line = [[abs(c) for c in cf].index(1) for cf in self.coef]
        self.geo, self.line, *rimg = _pack(self.device, [np.asarray(rows, np.int64), line, *groups.values()])
        self.rimg = dict(zip(groups, rimg))
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

    def gather(self, specs: list[tuple]) -> list[tuple]:
        # specs (group, tok, rimg, rows per image, lesion) -> (image, mask, lesion | None) [R, P] each, in one kernel
        tab, rimg_off, toks, rimgs, r0, o0, n_img = [], [], [], [], 0, 0, 0
        for G, tok, rimg, per_img, _ in specs:
            (_, gy, gz) = G.grid
            R = tok.numel()
            tab.append([r0, G.P, *G.patch, gy, gz, per_img, n_img, o0])
            toks.append(tok.reshape(-1))
            rimgs.append(rimg)
            r0, o0, n_img = r0 + R, o0 + R * G.P, n_img + rimg.numel()
        (tab,) = _pack(self.device, [tab])
        tok = toks[0].contiguous() if len(toks) == 1 else torch.cat(toks)
        rimg = rimgs[0].contiguous() if len(rimgs) == 1 else torch.cat(rimgs)
        img = torch.empty(o0, device=self.device, dtype=torch.float32)
        msk = torch.empty_like(img)
        tumor = self.mass is not None
        lesion = tumor and any(s[4] for s in specs)
        les = torch.empty_like(img) if lesion else img
        d = self.bits
        _gather_kernel[(r0,)](
            tab, len(specs), tok, rimg, self.geo, self.nrm, self.bits, self.pre, self.vals,
            self.mass if tumor else d, self.ti if tumor else d, self.tf if tumor else d, self.knots if tumor else d,
            img, msk, les, (self.knots.shape[2] // 5) if tumor else 1, TUMOR=tumor, LES=lesion, BLOCK=512,
        )
        out, o0 = [], 0
        for G, tok, _, _, want in specs:
            n = tok.numel() * G.P
            sl = slice(o0, o0 + n)
            out.append((img[sl].view(-1, G.P), msk[sl].view(-1, G.P), les[sl].view(-1, G.P) if want and lesion else None))
            o0 += n
        return out

    def group(self, sp, ids: list[int], patch, grid) -> "SparseGroup":
        return SparseGroup(self, sp, ids, tuple(patch), tuple(grid))


class SparseGroup:
    """One spacing group's images (LazyTokens stand-in): its token geometry and row gathers."""

    def __init__(self, vol: SparseVolumes, sp, ids: list[int], patch, grid):
        self.vol, self.ids, self.patch, self.grid = vol, ids, tuple(patch), tuple(grid)
        self.num = math.prod(grid)
        self.P = math.prod(patch)
        self.rimg = vol.rimg[sp]

    def gather(self, tok: Tensor, rimg: Tensor, per_img: int, lesion: bool = False):
        # rows (rimg[row // per_img], tok[row]) -> image, mask (float 0 / 1) [R, P] (+ the soft lesion mask)
        return self.vol.gather([(self, tok, rimg, per_img, lesion)])[0]


def frac_all(vol: "SparseVolumes", groups: list[SparseGroup]) -> list[Tensor]:
    # every group's per token in-mask fraction [n, num] in one kernel
    tab, rimg, t0 = [], [], 0
    for G in groups:
        (_, gy, gz) = G.grid
        tab.append([t0, G.P, *G.patch, gy, gz, G.num, len(rimg)])
        rimg += G.ids
        t0 += len(G.ids) * G.num
    tab, rimg = _pack(vol.device, [tab, rimg])
    out = torch.empty(t0, device=vol.device, dtype=torch.float32)
    _frac_kernel[(triton.cdiv(t0, 32),)](tab, len(groups), rimg, vol.geo, vol.line, vol.bits, vol.pre, out, t0, TB=32, LB=32)
    return [o.view(len(G.ids), G.num) for o, G in zip(out.split([len(G.ids) * G.num for G in groups]), groups)]


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
