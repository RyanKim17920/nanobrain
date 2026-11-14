# Muon for the transformer block matrices, AdamW for everything else.
# References:
# muon: https://github.com/KellerJordan/Muon
# moonlight (match_rms_adamw scaling): https://arxiv.org/abs/2502.16982

from collections import defaultdict

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    # batched over leading dims, bf16 like the reference implementation
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transpose = G.size(-2) > G.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon with the update scaled to AdamW's rms (0.2 sqrt(max(A, B))), so it takes AdamW's lr
    and decoupled weight decay. Same shape matrices are orthogonalized in one batch."""

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        fused=False,
        graph=False,
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps
        )
        super().__init__(params, defaults)
        # fused: one compiled step per batch of same shape matrices (a matrix and its transpose
        # share a batch), with the momentum, bf16 cast, stacking and weight update fused into a
        # few kernels, and the newton-schulz polynomial applied as one matrix (_newtonschulz5_poly).
        # same math as the unfused step up to bf16 rounding
        self.fused = fused
        # with fused, misc.clip_grad leaves these grads unclipped and sets the clip coefficient
        # as grad_scale, which the step applies on the fly
        self.defers_grad_clip = fused
        self.grad_scale: Tensor | None = None
        # graph (speed, off when unset; with fused): after a few eager fused steps the whole fused step is
        # captured as one cuda graph and replayed on static copies of the grads / clip scale (the same compiled
        # kernels on the same values, ~1 launch instead of the per shape compiled calls' python + launches)
        self.graph = graph and fused
        self._graph = None
        self._graph_steps = 0

    @torch.no_grad()
    def step(self):
        if self.graph:
            self._graph_step()
            return
        if self.fused:
            self._fused_step()
            return
        for group in self.param_groups:
            by_shape = defaultdict(list)
            for p in group["params"]:
                if p.grad is not None:
                    assert p.ndim == 2, "muon only takes matrices"
                    by_shape[p.shape].append(p)

            for shape, params in by_shape.items():
                grads = [p.grad for p in params]
                bufs = []
                for p in params:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    bufs.append(state["momentum_buffer"])

                momentum = group["momentum"]
                torch._foreach_lerp_(bufs, grads, 1 - momentum)
                if group["nesterov"]:
                    updates = torch._foreach_lerp(grads, bufs, momentum)
                else:
                    updates = bufs
                update = zeropower_via_newtonschulz5(torch.stack(updates), group["ns_steps"])

                lr = group["lr"]
                scale = 0.2 * max(shape) ** 0.5
                torch._foreach_mul_(params, 1 - lr * group["weight_decay"])
                torch._foreach_add_(params, list(update.to(params[0].dtype).unbind(0)), alpha=-lr * scale)

    def _graph_step(self):
        params = [p for group in self.param_groups for p in group["params"]]
        grads = [p.grad for p in params]
        if self._graph is None and self._graph_steps < 3:
            # eager fused steps first: compiles, creates the momentum buffers. a tensor clip scale (ones if none)
            # so the compiled graphs are the ones replayed later
            if self.grad_scale is None:
                self.grad_scale = torch.ones((), device=params[0].device)
            self._graph_steps += 1
            self._fused_step()
            return
        assert all(g is not None for g in grads), "muon graph: every muon matrix needs a grad"
        scale = self.grad_scale if self.grad_scale is not None else torch.ones((), device=params[0].device)
        self.grad_scale = None
        if self._graph is None:
            self._static_grads = [torch.empty_like(g) for g in grads]
            self._static_scale = torch.empty((), device=scale.device, dtype=scale.dtype)
            torch._foreach_copy_(self._static_grads, grads)
            self._static_scale.copy_(scale)
            self._set_coefs()
            # capture records without running; the replay below is this step's update
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            self._graph = torch.cuda.CUDAGraph()
            # thread_local: the prep threads' cuda calls (other streams) must not invalidate this capture
            with torch.cuda.graph(self._graph, stream=stream, capture_error_mode="thread_local"):
                self._fused_step(grads=self._static_grads, grad_scale=self._static_scale, set_coefs=False)
            torch.cuda.current_stream().wait_stream(stream)
        else:
            torch._foreach_copy_(self._static_grads, grads)
            self._static_scale.copy_(scale)
            self._set_coefs()
        self._graph.replay()

    def _set_coefs(self):
        for group in self.param_groups:
            shapes = {tuple(sorted(p.shape)) for p in group["params"] if p.grad is not None}
            lr, wd = group["lr"], group["weight_decay"]
            for shape in shapes:
                coefs = self._coefs((id(group), shape), group["params"][0].device)
                coefs[0].fill_(1 - lr * wd)
                coefs[1].fill_(-lr * 0.2 * max(shape) ** 0.5)

    def _fused_step(self, grads=None, grad_scale=None, set_coefs=True):
        # grads (graph): the grads to use, in param_groups order (default p.grad); grad_scale likewise
        if grad_scale is None:
            grad_scale = self.grad_scale
        self.grad_scale = None
        gi = iter(grads) if grads is not None else None
        for group in self.param_groups:
            # batch by the orientation newton-schulz runs in (rows <= cols)
            by_shape = defaultdict(list)
            grad_of = {}
            for p in group["params"]:
                g = next(gi) if gi is not None else p.grad
                if g is not None:
                    assert p.ndim == 2, "muon only takes matrices"
                    by_shape[tuple(sorted(p.shape))].append(p)
                    grad_of[p] = g

            lr, wd = group["lr"], group["weight_decay"]
            for shape, params in by_shape.items():
                bufs = []
                for p in params:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    bufs.append(state["momentum_buffer"])
                # python float scalars would be baked into the compiled graph, so they go in
                # as 0-d tensors (fill_ doesn't sync). graph: one pair per (group, shape), filled before replay
                coefs = self._coefs((id(group), shape) if self.graph else shape, params[0].device)
                if set_coefs:
                    coefs[0].fill_(1 - lr * wd)
                    coefs[1].fill_(-lr * 0.2 * max(shape) ** 0.5)
                _fused_muon_update(
                    params,
                    [grad_of[p] for p in params],
                    bufs,
                    coefs[0],
                    coefs[1],
                    grad_scale,
                    momentum=group["momentum"],
                    nesterov=group["nesterov"],
                    ns_steps=group["ns_steps"],
                )

    def _coefs(self, shape, device) -> tuple[Tensor, Tensor]:
        cache = self.__dict__.setdefault("_coef_cache", {})
        if shape not in cache:
            cache[shape] = (torch.zeros((), device=device), torch.zeros((), device=device))
        return cache[shape]


def _newtonschulz5_poly(X: Tensor, steps: int) -> Tensor:
    # zeropower_via_newtonschulz5 on a stack of bf16 rows <= cols matrices. the same iteration
    # X = a X + (b A + c A^2) X evaluated as X = P X, P = a I + b A + c A^2: one fused pointwise
    # kernel on the small square matrix between plain gemms. rounds fewer intermediates to bf16,
    # so it lands closer to an fp64 reference than the unfused bf16 version
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    eye = torch.eye(X.size(-2), device=X.device, dtype=X.dtype)
    for _ in range(steps):
        A = X @ X.mT
        P = a * eye + b * A + c * (A @ A)
        X = P @ X
    return X


# one graph per distinct matrix shape: models with many shapes (hiera stages) exceed dynamo's
# default limit of 8
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)


@torch.compile(fullgraph=True, dynamic=False)
def _fused_muon_update(
    params: list[Tensor],
    grads: list[Tensor],
    bufs: list[Tensor],
    decay: Tensor,
    alpha: Tensor,
    grad_scale: Tensor | None,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
):
    updates = []
    for g, buf in zip(grads, bufs):
        if grad_scale is not None:
            g = g * grad_scale
        buf.lerp_(g, 1 - momentum)
        u = g.lerp(buf, momentum) if nesterov else buf
        u = u.bfloat16()
        updates.append(u.mT if u.size(0) > u.size(1) else u)
    X = _newtonschulz5_poly(torch.stack(updates), ns_steps)
    for p, u in zip(params, X.unbind(0)):
        if p.size(0) > p.size(1):
            u = u.mT
        p.mul_(decay).add_(u.to(p.dtype) * alpha)


class CombinedOptimizer:
    """Steps several optimizers as one. param_groups is the concatenation of theirs, so the lr/wd
    updates and grad clipping see every parameter."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [group for opt in self.optimizers for group in opt.param_groups]

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


def is_muon_param(
    name: str,
    param: Tensor,
    encoder_only: bool = False,
    decoder_only: bool = False,
    all_matrices: bool = False,
) -> bool:
    # hidden matrices of the encoder and decoder blocks (encoder_only / decoder_only: one side).
    # embeddings, the input/output projections, norms and biases stay on AdamW, unless
    # all_matrices (every 2d weight except the position tables)
    if encoder_only and name.startswith("decoder_"):
        return False
    if decoder_only and not name.startswith("decoder_"):
        return False
    if all_matrices:
        return param.ndim == 2 and "pos_embed" not in name
    return param.ndim == 2 and "blocks." in name
