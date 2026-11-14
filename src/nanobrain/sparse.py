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
def _bits_kernel(runs, ends, blk, bits, cnt, BLOCK: tl.constexpr, FULL: tl.constexpr = False):
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
        if FULL:
            # FAST_BITS: a word the run covers whole is this run's alone: plain stores, atomics at its two ends only
            full = n == 32
            tl.store(bits + w, tl.full(n.shape, -1, tl.int32), mask=mm & full)
            tl.store(cnt + w, tl.full(n.shape, 32, tl.int32), mask=mm & full)
            mm = mm & (n < 32)
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
def _frac_kernel(gtab, G, rimg, geo, line, bits, pre, out, NT, TB: tl.constexpr, LB: tl.constexpr,
                 SHORT: tl.constexpr = False):
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
        if SHORT:
            # FAST_FRAC: lines of <= 32 voxels span <= 2 words: the count from those words alone (no prefix loads)
            a = tl.minimum(fs, fe)
            z = tl.maximum(fs, fe) + 1
            wa = a >> 5
            wz = z >> 5
            ba = tl.load(bits + wa, mask=ok, other=0)
            bz = tl.load(bits + wz, mask=ok, other=0)
            one = tl.full(a.shape, 1, tl.int32)
            n = libdevice.popc(bz & ((one << (z & 31).to(tl.int32)) - 1)) - libdevice.popc(ba & ((one << (a & 31).to(tl.int32)) - 1))
            n += tl.where(wz != wa, libdevice.popc(ba), 0)
        else:
            n = _rank(tl.maximum(fs, fe) + 1, bits, pre, ok) - _rank(tl.minimum(fs, fe), bits, pre, ok)
        acc += tl.where(ok, n, 0)
    tl.store(out + tok, tl.sum(acc, axis=1).to(tl.float32) / tl.reshape(P, [TB]), mask=tm)


@triton.jit(do_not_specialize=["G", "NK", "V", "SEQ"])
def _gather_kernel(
    gtab, G, tok, rimg, geo, nrm, bits, pre, vals, mass, ti, tf, knots, img_out, msk_out, les_out, NK,
    t_img, t_w, V, SEQ,
    TUMOR: tl.constexpr, LES: tl.constexpr, BLOCK: tl.constexpr, SPLIT: tl.constexpr = False, NC: tl.constexpr = 10,
):
    # one row of (image, box token) per program, rows of every group: the normalized image / mask voxels (+ the
    # pasted tumour, its soft mask). group table rows (NC 10): first row, P, PX, PY, PZ, GY, GZ, rows per image,
    # first image id, first output element. SPLIT (FAST_TOK, NC 13; rows = sequence rows of SEQ tokens): + first
    # target element, lesion wanted, 1 / P (fp32 bits); a row's first V tokens go to img / msk / les_out, the rest to
    # t_img and t_w (the mask / P: flat_loss' weights)
    row = tl.program_id(0)
    gp = gtab + _group_of(row, gtab, G, NC, 0) * NC
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
    o1, o2 = img_out, msk_out
    if SPLIT:
        sr = loc // SEQ
        j = loc % SEQ
        vis = j < V
        obase = tl.where(vis, tl.load(gp + 9) + (sr * V + j).to(tl.int64) * P,
                         tl.load(gp + 10) + (sr * (SEQ - V) + j - V).to(tl.int64) * P)
        want = vis & (tl.load(gp + 11) != 0)
        invp = tl.load(gp + 12).to(tl.int32).to(tl.float32, bitcast=True)
        on = tl.where(vis, 1.0, invp)
        o1 = tl.where(vis, img_out, t_img)
        o2 = tl.where(vis, msk_out, t_w)
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
        if SPLIT:
            tl.store(o1 + o, v, mask=qm)
            tl.store(o2 + o, tl.where(inm, on, 0.0), mask=qm)
            if LES:
                tl.store(les_out + o, les, mask=qm & want)
        else:
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


# FAST_BITS (speed): _bits_kernel's whole words as plain stores, BITS_CFG (runs per block, num_warps) (the same
# bitmap and counts)
FAST_BITS = False
BITS_CFG = (128, 4)


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
        bb = BITS_CFG[0] if FAST_BITS else 1024
        blk, rblk = _pack(dev, [_blocks(nv, 4096), _blocks(nr, bb, first)])
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
        _bits_kernel[(len(rblk),)](runs, ends, rblk, self.bits, cnt, BLOCK=bb, FULL=FAST_BITS,
                                   num_warps=BITS_CFG[1] if FAST_BITS else 4)
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

    def gather(self, specs: list[tuple], cfg: tuple = (512, 4)) -> list[tuple]:
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
            img, msk, les, (self.knots.shape[2] // 5) if tumor else 1, img, img, 0, 1, TUMOR=tumor, LES=lesion, BLOCK=cfg[0],
            num_warps=cfg[1],
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


# FAST_FRAC (speed): _frac_kernel's line counts from the line's (<= 2) bitmap words, not two prefix ranks, and its
# tile / warps FRAC_CFG (TB, LB, num_warps) (the same exact counts)
FAST_FRAC = False
FRAC_CFG = (32, 4, 4)


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
    short = FAST_FRAC and max(max(G.patch) for G in groups) <= 32
    tb, lb, nw = FRAC_CFG if FAST_FRAC else (32, 32, 4)
    _frac_kernel[(triton.cdiv(t0, tb),)](tab, len(groups), rimg, vol.geo, vol.line, vol.bits, vol.pre, out, t0, TB=tb, LB=lb,
                                         SHORT=short, num_warps=nw)
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


# FAST_TOK (speed, model.native_tokens): after the sequence orders, every spacing group's sequence rows, unified
# swaps, corruptions and level / lesion targets as a few launches over flat buffers (the per group torch ops of the
# loop + rtd_replace + corrupt_all + finish + flat_loss before; the same values)
FAST_TOK = False
GATHER_CFG = (32, 1)  # the sequence gather's BLOCK, num_warps (any: per voxel math)
# FAST_POOL (speed, with FAST_TOK): the level targets' 0/1 channels in one batched dot per row block (exact sums)
FAST_POOL = False
POOL_CFG = (8, 64, 4)  # its rows per program, k block, num_warps


# IEEE fp32 ops as their own ptx instructions (round to nearest, subnormals kept, no fma contraction): torch's
# elementwise kernels' results bit for bit (libdevice's *_rn flush subnormals under triton's ftz reflect)
@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sub(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _div(a, b):
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _qkeys_kernel(vb, koff, lens, vis, msk, keys, cnt, BLOCK: tl.constexpr):
    # rtd_replace's row_quantiles sort keys: a row's visible values in its brain mask (> 0.5), +inf elsewhere, at
    # koff (rows of a length one after another: each length class a [rows, length] view), and its in-mask count
    r = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = i < tl.load(lens + r)
    o = tl.load(vb + r) + i
    x = tl.load(vis + o, mask=ok, other=0.0)
    m = tl.load(msk + o, mask=ok, other=0.0) > 0.5
    tl.store(keys + tl.load(koff + r) + i, tl.where(m, x, float("inf")), mask=ok)
    tl.atomic_add(cnt + r, tl.sum((m & ok).to(tl.int32), axis=0))


@triton.jit(do_not_specialize=["V"])
def _tokmin_kernel(rbase, rp, vis, msk, tmin, V, BLOCK: tl.constexpr):
    # each visible token's minimum over its in-mask voxels (+inf without one): their row minimum is corrupt_all's lo
    t = tl.program_id(0)
    r = t // V
    P = tl.load(rp + r)
    base = tl.load(rbase + r) + (t % V).to(tl.int64) * P
    mn = tl.full([BLOCK], float("inf"), tl.float32)
    for q0 in range(0, P, BLOCK):
        q = q0 + tl.arange(0, BLOCK)
        qm = q < P
        x = tl.load(vis + base + q, mask=qm, other=0.0)
        m = tl.load(msk + base + q, mask=qm, other=0.0)
        mn = tl.minimum(mn, tl.where(qm & (m > 0), x, float("inf")))
    tl.store(tmin + t, tl.min(mn, axis=0))


@triton.jit(do_not_specialize=["K", "V"])
def _rtd_kernel(grow, vb, prs, pb, prow, cand, labels, q, p_img, p_msk, vis, msk, done, K, V,
                NQ: tl.constexpr, BLOCK: tl.constexpr):
    # rtd_replace (unified swap) of candidate c of row r2: the partner patch (>= 50% brain) quantile mapped to the
    # host row's quantiles into the visible row in place where the label selects it, and done
    pid = tl.program_id(0)
    r2 = pid // K
    c = pid % K
    r = tl.load(grow + r2)
    P = tl.load(prs + r2)
    cd = tl.load(cand + r * K + c)
    sel = tl.load(labels + r * V + cd) == 1
    pbase = tl.load(pb + r2) + c.to(tl.int64) * P
    n = tl.zeros([BLOCK], dtype=tl.int32)
    for q0 in range(0, P, BLOCK):
        qq = q0 + tl.arange(0, BLOCK)
        n += (tl.load(p_msk + pbase + qq, mask=qq < P, other=0.0) > 0).to(tl.int32)
    rep = sel & (2 * tl.sum(n, axis=0) >= P)
    tl.store(done + r * V + cd, rep.to(tl.int8))
    if rep:
        qp = q + tl.load(prow + r2) * NQ
        qh = q + r2 * NQ
        vbase = tl.load(vb + r2) + cd * P
        for q0 in range(0, P, BLOCK):
            qq = q0 + tl.arange(0, BLOCK)
            qm = qq < P
            x = tl.load(p_img + pbase + qq, mask=qm, other=0.0)
            pm = tl.load(p_msk + pbase + qq, mask=qm, other=0.0)
            pos = tl.zeros([BLOCK], dtype=tl.int32)
            for i in tl.static_range(NQ):
                pos += (tl.load(qp + i) < x).to(tl.int32)
            pos = tl.minimum(tl.maximum(pos, 1), NQ - 1)
            lo = tl.load(qp + pos - 1)
            hi = tl.load(qp + pos)
            hl = tl.load(qh + pos - 1)
            hh = tl.load(qh + pos)
            t = _div(_sub(x, lo), _sub(hi, lo))
            t = tl.where(t != t, t, tl.minimum(tl.maximum(t, 0.0), 1.0))
            y = _add(hl, _mul(t, _sub(hh, hl)))
            tl.store(vis + vbase + qq, y, mask=qm)
            tl.store(msk + vbase + qq, pm, mask=qm)


@triton.jit(do_not_specialize=["G", "T", "C0"])
def _pool_all_kernel(gtab, G, x, m, les, thr, out, T, C0, BR: tl.constexpr, BK: tl.constexpr, NB: tl.constexpr,
                     SUB: tl.constexpr, NC: tl.constexpr):
    # _pool_kernel over every group's token rows (the same per row block math) + pool_targets' bin length divisions.
    # group table rows (NC 8 + 3 SUB): first row block, rows, P, kb / kf pointers, first voxel, first output row,
    # lesion (0 / 1), the bin lengths along x, y, z
    rb = tl.program_id(0)
    c = tl.program_id(1) + C0
    gp = gtab + _group_of(rb, gtab, G, NC, 0) * NC
    R = tl.load(gp + 1)
    P = tl.load(gp + 2)
    kb = tl.load(gp + 3).to(tl.pointer_type(tl.bfloat16))
    kf = tl.load(gp + 4).to(tl.pointer_type(tl.float32))
    x0 = tl.load(gp + 5)
    rows = (rb - tl.load(gp)) * BR + tl.arange(0, BR)
    rm = rows < R
    acc = tl.zeros([BR, NB], dtype=tl.float32)
    t = tl.load(thr + tl.maximum(c - 1, 0), mask=(c >= 1) & (c <= T), other=0.0)
    if (c <= T) | (tl.load(gp + 7) != 0):
        for k0 in range(0, P, BK):
            ks = k0 + tl.arange(0, BK)
            km = ks < P
            o = x0 + rows[:, None] * P + ks[None, :]
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
    nb = tl.arange(0, NB)
    lx = tl.load(gp + 8 + nb // (SUB * SUB)).to(tl.float32)
    ly = tl.load(gp + 8 + SUB + (nb // SUB) % SUB).to(tl.float32)
    lz = tl.load(gp + 8 + 2 * SUB + nb % SUB).to(tl.float32)
    acc = _div(_div(_div(acc, lx[None, :] + tl.zeros_like(acc)), ly[None, :] + tl.zeros_like(acc)), lz[None, :] + tl.zeros_like(acc))
    oo = ((tl.load(gp + 6) + rows[:, None]) * (T + 2) + c) * NB + nb[None, :]
    tl.store(out + oo, acc, mask=rm[:, None])


@triton.jit(do_not_specialize=["G", "T"])
def _pool_exact_kernel(gtab, G, x, m, thr, out, T, BR: tl.constexpr, BK: tl.constexpr, NB: tl.constexpr,
                       SUB: tl.constexpr, NC: tl.constexpr, TC: tl.constexpr):
    # FAST_POOL: _pool_all_kernel's 0/1 channels (mask, (x > thr_t) x mask; exact integer bin sums) of BR rows in one
    # dot of the channels stacked along the rows (x and the mask read once, not once per channel); the table's row
    # blocks are of 64 rows
    pid = tl.program_id(0)
    rb = pid // (64 // BR)
    gp = gtab + _group_of(rb, gtab, G, NC, 0) * NC
    R = tl.load(gp + 1)
    P = tl.load(gp + 2)
    kb = tl.load(gp + 3).to(tl.pointer_type(tl.bfloat16))
    x0 = tl.load(gp + 5)
    rows = (rb - tl.load(gp)) * 64 + (pid % (64 // BR)) * BR + tl.arange(0, BR)
    rm = rows < R
    ch = tl.arange(0, TC)
    t = tl.load(thr + tl.maximum(ch - 1, 0), mask=(ch >= 1) & (ch <= T), other=0.0)
    acc = tl.zeros([TC * BR, NB], dtype=tl.float32)
    for k0 in range(0, P, BK):
        ks = k0 + tl.arange(0, BK)
        km = ks < P
        o = x0 + rows[:, None] * P + ks[None, :]
        lm = rm[:, None] & km[None, :]
        mv = tl.load(m + o, mask=lm, other=0.0)
        xv = tl.load(x + o, mask=lm, other=0.0)
        a = tl.where((ch[:, None, None] == 0) | (xv[None, :, :] > t[:, None, None]), mv[None, :, :], 0.0)
        kb_ = tl.load(kb + ks[:, None] * NB + tl.arange(0, NB)[None, :], mask=km[:, None], other=0.0)
        acc += tl.dot(tl.reshape(a.to(tl.bfloat16), (TC * BR, BK)), kb_)
    acc = tl.reshape(acc, (TC, BR, NB))
    nb = tl.arange(0, NB)
    lx = tl.load(gp + 8 + nb // (SUB * SUB)).to(tl.float32)
    ly = tl.load(gp + 8 + SUB + (nb // SUB) % SUB).to(tl.float32)
    lz = tl.load(gp + 8 + 2 * SUB + nb % SUB).to(tl.float32)
    z = tl.zeros_like(acc)
    acc = _div(_div(_div(acc, lx[None, None, :] + z), ly[None, None, :] + z), lz[None, None, :] + z)
    oo = ((tl.load(gp + 6) + rows[None, :, None]) * (T + 2) + ch[:, None, None]) * NB + nb[None, None, :]
    tl.store(out + oo, acc, mask=rm[None, :, None] & (ch[:, None, None] <= T))


def _bins(size: int, sub: int) -> list[int]:
    # pool_kron's adaptive bin lengths along one axis
    return [-((-(i + 1) * size) // sub) - (i * size) // sub for i in range(sub)]


def fast_tokens(model, vol: "SparseVolumes", parts: list[dict], order_all: Tensor, uni: tuple, geoms: dict,
                S: int, V: int, NP: int) -> list[dict]:
    # native_tokens' sparse loop + corrupt_all + finish + flat_loss for the unified / level / lesion setup, batched
    dev = vol.device
    seq = V + NP
    cand, labels, kind = uni
    K = cand.shape[1]
    T = model.level_thresholds.numel()
    sub = model.patch_size[0] // model.level_pool
    NB = sub**3
    nq = model.rtd_quantiles + 1
    min_vox = max(2, model.patch_size[0] // 2)
    gs, r0, vb, tb = [], 0, 0, 0
    for p in parts:
        G = p["vols"]
        b = len(p["ids"])
        rows = b * S
        gs.append(dict(G=G, b=b, rows=rows, r0=r0, vb=vb, tb=tb, P=G.P, les=p.pop("sparse_les", False)))
        r0, vb, tb = r0 + rows, vb + rows * V * G.P, tb + rows * NP * G.P
    R = r0
    tumor = vol.mass is not None
    lesion = tumor and any(g["les"] for g in gs)
    f32 = dict(device=dev, dtype=torch.float32)
    vimg, vmsk = torch.empty(vb, **f32), torch.empty(vb, **f32)
    vles = torch.empty(vb, **f32) if lesion else vimg
    timg, tw = torch.empty(tb, **f32), torch.empty(tb, **f32)
    tmin = torch.empty(R * V, **f32)
    out = torch.empty(vb, **f32)
    pooled = torch.empty(R * V, T + 2, NB, **f32)
    # host tables: sequence gather, coord row offsets, pool, corrupt; rtd ones below
    gtab, rimg, coff, ptab, ctab, n0, nimg, blk, rbase, rp = [], [], [], [], [], 0, 0, 0, [], []
    kb_kf = {}
    for g in gs:
        G = g["G"]
        (_, gy, gz) = G.grid
        gtab.append([g["r0"] * seq, G.P, *G.patch, gy, gz, S * seq, nimg, g["vb"], g["tb"], int(g["les"]),
                     int((np.float32(1.0) / np.float32(G.P)).view(np.int32))])
        rbase += [g["vb"] + i * V * G.P for i in range(g["rows"])]
        rp += [G.P] * g["rows"]
        rimg += G.ids
        nimg += len(G.ids)
        coff += [n0] * g["rows"]
        n0 += G.num
        if G.patch not in kb_kf:
            kb_kf[G.patch] = _pool_mats(G.patch, sub, dev)[:2]
        kb, kf = kb_kf[G.patch]
        lens = [v for size in G.patch for v in _bins(size, sub)]
        ptab.append([blk, g["rows"] * V, G.P, kb.data_ptr(), kf.data_ptr(), g["vb"], g["r0"] * V,
                     int(g["les"] and lesion), *lens])
        blk += triton.cdiv(g["rows"] * V, 64)
        ks = [2 if p_ >= min_vox else 1 for p_ in G.patch]
        o = 4 * g["vb"]
        ctab.append([g["r0"] * K, V, K, G.P, *G.patch, *ks, vimg.data_ptr() + o, vmsk.data_ptr() + o, out.data_ptr() + o,
                     g["r0"]])
    # rtd rows (groups of >= 2 images): partner rows of the next image, same sample index
    two = sorted((g for g in gs if g["b"] >= 2), key=lambda g: g["P"])
    grow, rvb, rps, rpb, prow, rlen, pspecs, pb, s2 = [], [], [], [], [], [], [], 0, 0
    for g in two:
        b, P = g["b"], g["P"]
        for rl in range(g["rows"]):
            grow.append(g["r0"] + rl)
            rvb.append(g["vb"] + rl * V * P)
            rps.append(P)
            rpb.append(pb + rl * K * P)
            prow.append(s2 + (rl // S + 1) % b * S + rl % S)
            rlen.append(V * P)
        s2 += g["rows"]
        pb += g["rows"] * K * P
        ids = g["G"].ids
        pspecs.append(ids[1:] + ids[:1])
    rlen_ = rlen
    koff_ = np.cumsum([0] + rlen[:-1]).tolist() if rlen else []
    tabs = _pack(dev, [gtab, rimg, coff, ptab, ctab, rbase, rp, grow, rvb, rps, rpb, prow, rlen, koff_, *pspecs])
    gtab, rimg, coff, ptab, ctab, rbase, rp, grow, rvb, rps, rpb, prow, rlen, koff, *prims = tabs
    d = vol.bits
    _gather_kernel[(R * seq,)](
        gtab, len(gs), order_all.reshape(-1) if order_all.is_contiguous() else order_all.contiguous().view(-1),
        rimg, vol.geo, vol.nrm, vol.bits, vol.pre, vol.vals,
        vol.mass if tumor else d, vol.ti if tumor else d, vol.tf if tumor else d, vol.knots if tumor else d,
        vimg, vmsk, vles, (vol.knots.shape[2] // 5) if tumor else 1, timg, tw, V, seq,
        TUMOR=tumor, LES=lesion, BLOCK=GATHER_CFG[0], SPLIT=True, NC=13, num_warps=GATHER_CFG[1],
    )
    # token centres: every group's table in one gather
    ctr = geoms[parts[0]["spacing"]][3][0] if len(parts) == 1 else torch.cat([geoms[p["spacing"]][3][0] for p in parts])
    coord = ctr[order_all + coff[:, None]]
    if two:
        # partner rows, quantile keys + one sort, the mapping in place
        vo = order_all[:, :V].gather(1, cand)
        pg = vol.gather([(g["G"], vo[g["r0"] : g["r0"] + g["rows"]], pr, S * K, False) for g, pr in zip(two, prims)],
                        GATHER_CFG)
        # (gather's outputs are views of one buffer each, groups in order)
        p_img, p_msk = (torch.as_strided(t, (pb,), (1,)) for t in pg[0][:2])
        # (rows of one patch size are adjacent here (two by P), and so are their sort keys)
        qs = torch.linspace(0, 1, nq, device=dev)
        keys = torch.empty(sum(rlen_), **f32)
        cnt = torch.zeros(s2, device=dev, dtype=torch.int32)
        _qkeys_kernel[(s2, triton.cdiv(max(rlen_), 4096))](rvb, koff, rlen, vimg, vmsk, keys, cnt, BLOCK=4096)
        cnt = cnt.long().clamp_min(1)
        idx = (qs[None] * (cnt[:, None] - 1).float()).round().long()
        qv, a = [], 0
        for P in sorted({g["P"] for g in two}):
            n = sum(g["rows"] for g in two if g["P"] == P)
            k0 = koff_[a]
            qv.append(keys[k0 : k0 + n * V * P].view(n, V * P).sort(dim=1).values.gather(1, idx[a : a + n]))
            a += n
        qv = qv[0] if len(qv) == 1 else torch.cat(qv)
        qv = torch.where(torch.isfinite(qv), qv, torch.zeros_like(qv))
        qv = (qv + 1e-5 * torch.arange(nq, device=dev)[None]).contiguous()
        done = torch.zeros(R, V, device=dev, dtype=torch.int8)
        _rtd_kernel[(s2 * K,)](grow, rvb, rps, rpb, prow, cand, labels.contiguous(), qv, p_img, p_msk, vimg, vmsk, done,
                               K, V, NQ=nq, BLOCK=512)
        labels = torch.where((labels == 1) & ~done.bool(), 0, labels)
    else:
        labels = torch.where(labels == 1, 0, labels)
    # corruptions (corrupt_all's kernel, the row minima from the token minima)
    out.copy_(vimg)
    _tokmin_kernel[(R * V,)](rbase, rp, vimg, vmsk, tmin, V, BLOCK=512)
    lo = tmin.view(R, V).amin(1)
    lo = torch.where(lo < float("inf"), lo, 0.0)
    bf16 = torch.is_autocast_enabled(dev.type) and torch.get_autocast_dtype(dev.type) == torch.bfloat16
    seed = int(torch.randint(0, 2**30, (1,)).item())
    _corrupt_kernel[(R * K,)](ctab, len(gs), cand, kind, lo, seed, CB=512, BF16=bf16)
    # level + lesion targets (pool_targets), the token weights
    thr = model.level_thresholds.float().contiguous()
    if FAST_POOL:
        br, bk, nw = POOL_CFG
        _pool_exact_kernel[(blk * (64 // br),)](ptab, len(gs), out, vmsk, thr, pooled, T, BR=br, BK=bk, NB=NB, SUB=sub,
                                                NC=8 + 3 * sub, TC=triton.next_power_of_2(T + 1), num_warps=nw)
    _pool_all_kernel[(blk, 1 if FAST_POOL else T + 2)](ptab, len(gs), out, vmsk, vles, thr, pooled, T,
                                                       T + 1 if FAST_POOL else 0, BR=64, BK=64, NB=NB, SUB=sub,
                                                       NC=8 + 3 * sub)
    pooled = pooled.view(R * V, T + 2, sub, sub, sub)
    w = pooled[:, :1]
    lvl_t = pooled[:, 1 : 1 + T] / w.clamp_min(1e-6)
    w = w.flatten(1)
    altered = (labels > 0).bool()
    keep = ~altered.reshape(-1, 1)
    les_w = torch.where(keep, w, 0.0)
    les_t = torch.where(keep, pooled[:, 1 + T].flatten(1) / w.clamp_min(1e-6), 0.0)
    syn = ~altered
    for p, g, order in zip(parts, gs, order_all.split([g["rows"] for g in gs])):
        a, z, P = g["r0"], g["r0"] + g["rows"], g["P"]
        del p["vols"], p["scores"]
        p.pop("lesion", None)
        p["order"] = order.unflatten(0, (g["b"], S))
        p["unified_label"] = labels[a:z]
        p["vis_coord"], p["target_coord"] = coord[a:z].split([V, NP], 1)
        p["vis_tok"] = out[g["vb"] : g["vb"] + g["rows"] * V * P].view(g["rows"], V, P)
        p["lvl_w"], p["lvl_t"] = pooled[a * V : z * V, :1], lvl_t[a * V : z * V]
        p["syn_clean"] = syn[a:z]
        p["les_w"], p["les_t"] = les_w[a * V : z * V], les_t[a * V : z * V]
    parts[0]["flat_w"], parts[0]["flat_t"] = tw, timg
    return parts



# FAST_SCORES (speed, model.native_tokens): sequence_scores' masking (>= min_mask_frac else 2.0) and the 3.0 / 0.0
# padded scores / mask fraction rows of every group in one launch (per group where + copies before; the same values)
FAST_SCORES = False


@triton.jit(do_not_specialize=["G", "N", "S"])
def _scores_kernel(tab, G, frac, sc, fr, N, S, lo, BLOCK: tl.constexpr):
    # group table rows (NC 5): first sequence row, tokens, rand pointer, first fraction element, first image
    row = tl.program_id(0)
    gp = tab + _group_of(row, tab, G, 5, 0) * 5
    loc = row - tl.load(gp)
    n = tl.load(gp + 1)
    rnd = tl.load(gp + 2).to(tl.pointer_type(tl.float32))
    img = loc // S
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cm = c < N
    inb = c < n
    f = tl.load(frac + tl.load(gp + 3) + img * n + c, mask=inb, other=0.0)
    r = tl.load(rnd + loc.to(tl.int64) * n + c, mask=inb, other=0.0)
    tl.store(sc + row.to(tl.int64) * N + c, tl.where(inb, tl.where(f >= lo, r, 2.0), 3.0), mask=cm)
    if loc % S == 0:
        tl.store(fr + (tl.load(gp + 4) + img).to(tl.int64) * N + c, tl.where(inb, f, 0.0), mask=cm)


def pad_scores(parts: list[dict], S: int, min_frac: float) -> tuple[Tensor, Tensor]:
    # parts' "rand" [b, S, n] + mask_frac [b, n] (views of frac_all's buffer) -> scores [rows, N], frac [B, N]
    tab, r0, f0, i0 = [], 0, 0, 0
    fr0 = parts[0]["mask_frac"]
    N = max(p["mask_frac"].shape[1] for p in parts)
    for p in parts:
        b, n = p["mask_frac"].shape
        tab.append([r0, n, p["rand"].data_ptr(), p["mask_frac"].data_ptr() // 4 - fr0.data_ptr() // 4, i0])
        r0, i0 = r0 + b * S, i0 + b
    (tab,) = _pack(fr0.device, [tab])
    sc = torch.empty(r0, N, device=fr0.device, dtype=torch.float32)
    fr = torch.empty(i0, N, device=fr0.device, dtype=torch.float32)
    _scores_kernel[(r0, triton.cdiv(N, 1024))](tab, len(parts), fr0, sc, fr, N, S, min_frac, BLOCK=1024)
    for p in parts:
        p.pop("rand")
    return sc, fr
