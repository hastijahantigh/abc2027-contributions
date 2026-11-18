import torch
import torch.nn.functional as F

from .base import BaseLoss


class MSELoss(BaseLoss):
    """
    LeWM-style latent prediction error: mean squared error between predicted and target
    latents. It also penalizes norm errors, which matter once SIGReg gives the latents a
    fixed (isotropic Gaussian) scale.
    """
    def __init__(self, pred_key: str, target_key: str) -> None:
        super().__init__()
        self.pred_key = pred_key
        self.target_key = target_key

    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = F.mse_loss(output[self.pred_key].float(), output[self.target_key].float())
        return loss


class CosineLoss(BaseLoss):
    """
    1 - cosine similarity between predicted and target latents, averaged over rows. Unlike MSE
    it ignores errors in the norm (the ablation of MSELoss).
    """
    def __init__(self, pred_key: str, target_key: str) -> None:
        super().__init__()
        self.pred_key = pred_key
        self.target_key = target_key

    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        cos = F.cosine_similarity(output[self.pred_key].float(), output[self.target_key].float(), dim=-1)
        return (1 - cos).mean()


class StateCELoss(BaseLoss):
    """
    Weighted cross-entropy over the fall states (none / begin / middle / end), one row per
    supervised timestep: sum(weight * class_weight[y] * CE) / sum(weight * class_weight[y]).
    Rows of weight 0 (padding past the stream end) do not count; flagged falls come with a
    smaller weight (see data/labels.py). class_weights offsets the imbalance (about 3 % of
    timesteps are falls).
    """
    def __init__(self, logits_key: str, target_key: str, weight_key: str,
                 class_weights: list[float] | None = None) -> None:
        super().__init__()
        self.logits_key = logits_key
        self.target_key = target_key
        self.weight_key = weight_key
        self.register_buffer("class_weights", torch.tensor(class_weights or [1.0, 3.0, 3.0, 3.0]), persistent=False)

    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = output[self.logits_key].float()
        target = output[self.target_key].long()
        weight = output[self.weight_key].float() * self.class_weights[target]
        ce = F.cross_entropy(logits, target, reduction="none")
        return (weight * ce).sum() / weight.sum().clamp_min(1e-6)


class BoundaryDIoULoss(BaseLoss):
    """
    1D DIoU loss of boundary regression (ActionFormer): each row is a timestep t inside a fall with
    predicted and target distances (d_start, d_end) >= 0, i.e. the intervals [t - d_start, t + d_end].
    loss = 1 - IoU + (distance of the centres / length of the enclosing interval)^2, averaged with the
    row weights (0 outside falls, the label weight inside).
    """
    def __init__(self, pred_key: str, target_key: str, weight_key: str) -> None:
        super().__init__()
        self.pred_key = pred_key
        self.target_key = target_key
        self.weight_key = weight_key

    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        weight = output[self.weight_key].float()
        keep = weight > 0
        if not keep.any():
            return output[self.pred_key].sum() * 0.0
        pred, target, weight = output[self.pred_key].float()[keep], output[self.target_key].float()[keep], weight[keep]
        inter = torch.minimum(pred[:, 0], target[:, 0]) + torch.minimum(pred[:, 1], target[:, 1])
        union = pred.sum(-1) + target.sum(-1) - inter
        iou = inter / union.clamp_min(1e-6)
        enclosing = torch.maximum(pred[:, 0], target[:, 0]) + torch.maximum(pred[:, 1], target[:, 1])
        centre = ((pred[:, 1] - pred[:, 0]) - (target[:, 1] - target[:, 0])) / 2
        loss = 1 - iou + centre.square() / enclosing.square().clamp_min(1e-6)
        return (weight * loss).sum() / weight.sum()


class SIGRegLoss(BaseLoss):
    """
    Prevents latent collapse via Epps-Pulley Statistic.
    If `fixed_length` is given, latents are zero-padded to that many rows so the large
    [N, num_projections, num_nodes] intermediates keep a constant shape across batches
    (a variable N fragments the CUDA caching allocator and GPU memory keeps growing).
    """
    def __init__(
        self,
        latent_key: str,
        batch_size: int,
        num_projections: int = 1024,
        num_nodes: int = 17,
        fixed_length: int | None = None
    ) -> None:
        super().__init__()

        self.latent_key = latent_key
        self.num_projections = num_projections
        self.batch_size = batch_size
        self.fixed_length = fixed_length

        t = torch.linspace(0, 3, num_nodes, dtype=torch.float32)
        dt = 3 / (num_nodes - 1)

        weights = torch.full((num_nodes,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        # Constants, rebuilt from the arguments: kept out of checkpoints.
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        z = output[self.latent_key]
        N, D = z.shape

        num_pad = 0
        if self.fixed_length is not None:
            z = z[:self.fixed_length]
            N = z.size(0)
            num_pad = self.fixed_length - N
            z = F.pad(z, (0, 0, 0, num_pad), value=0.0)

        A = torch.randn(D, self.num_projections, device=z.device)
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (z @ A).unsqueeze(-1) * self.t
        # Zero padded rows give x_t == 0 exactly, i.e. cos == 1 and sin == 0, so their
        # contribution is removed from the sums analytically (no full-size mask tensor needed).
        cos_mean = (x_t.cos().sum(dim=0) - num_pad) / N
        sin_mean = x_t.sin().sum(dim=0) / N
        err = (cos_mean - self.phi).square() + sin_mean.square()

        statistic = (err @ self.weights) * self.batch_size
        loss = statistic.mean()
        return loss


class UnitBCELoss(BaseLoss):
    """
    The recording head's fall loss: weighted BCE of P(the unit holds a fall), one row per unit
    (model/recording_head.py). Weight 0 for units the scorer ignores or that cut a fall. A batch with
    no unit gives 0.
    """
    def __init__(self, logit_key: str = "unit_logit", target_key: str = "unit_fall", weight_key: str = "unit_fall_w",
                 pos_weight: float = 1.0) -> None:
        super().__init__()
        self.logit_key, self.target_key, self.weight_key = logit_key, target_key, weight_key
        self.pos_weight = pos_weight

    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.target_key not in output:
            return output["step_logits"].sum() * 0.0
        logit, target = output[self.logit_key].float(), output[self.target_key].float()
        weight = output[self.weight_key].float() * torch.where(target > 0, self.pos_weight, 1.0)
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        return (weight * bce).sum() / weight.sum().clamp_min(1e-6)


class SpanSoftCELoss(BaseLoss):
    """
    The recording head's span loss: cross-entropy of the start and the end distributions (softmax
    over the unit) against Gaussians of std `sigma` timesteps around the target start and end,
    restricted to the unit's domain; one row per unit, weighted (0: no target). Used twice: on the
    labelled falls (span_target) and, as rule distillation, on the official rule's interval in
    non-fall units (rule_target).
    """
    def __init__(self, target_key: str = "span_target", weight_key: str = "span_w", logits_key: str = "span_logits",
                 domain_key: str = "span_domain", sigma: float = 5.0) -> None:
        super().__init__()
        self.target_key, self.weight_key, self.logits_key, self.domain_key = target_key, weight_key, logits_key, domain_key
        self.sigma = sigma

    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.target_key not in output:
            return output["step_logits"].sum() * 0.0
        weight = output[self.weight_key].float()
        keep = weight > 0
        if not keep.any():
            return output[self.logits_key].sum() * 0.0
        logits = output[self.logits_key][keep].float()                        # [n, L, 2]
        domain = output[self.domain_key][keep]
        target = output[self.target_key][keep].float().clamp(0, logits.size(1) - 1)   # [n, 2]
        pos = torch.arange(logits.size(1), device=logits.device, dtype=torch.float32)
        soft = torch.exp(-0.5 * ((pos[None, :, None] - target[:, None, :]) / self.sigma) ** 2) * domain[..., None]
        soft = soft / soft.sum(1, keepdim=True).clamp_min(1e-12)
        ce = -(soft * torch.log_softmax(logits, dim=1)).sum(1).mean(-1)      # mean of start and end
        return (weight[keep] * ce).sum() / weight[keep].sum()


class SpanIoULoss(BaseLoss):
    """
    1 - tIoU between the target span and the expected span of the start and end distributions (softmax
    over the domain; the expected positions are differentiable), one row per prediction, weighted (0:
    no target). Next to SpanSoftCELoss it rewards a span that covers the whole fall over one whose
    start sits on the impact: the CE only scores the probability at the target steps.
    """
    def __init__(self, target_key: str = "span_target", weight_key: str = "span_w", logits_key: str = "span_logits",
                 domain_key: str = "span_domain") -> None:
        super().__init__()
        self.target_key, self.weight_key, self.logits_key, self.domain_key = target_key, weight_key, logits_key, domain_key

    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.target_key not in output:
            return output["step_logits"].sum() * 0.0
        weight = output[self.weight_key].float()
        keep = weight > 0
        if not keep.any():
            return output[self.logits_key].sum() * 0.0
        logits = output[self.logits_key][keep].float()                        # [n, L, 2]
        domain = output[self.domain_key][keep]
        logits = logits.masked_fill(~domain[..., None].bool(), -1e4)
        pos = torch.arange(logits.size(1), device=logits.device, dtype=torch.float32)
        expected = (torch.softmax(logits, dim=1) * pos[None, :, None]).sum(1)   # [n, 2]
        target = output[self.target_key][keep].float()
        s, e = expected[:, 0], torch.maximum(expected[:, 1], expected[:, 0] + 1.0)
        inter = (torch.minimum(e, target[:, 1]) - torch.maximum(s, target[:, 0])).clamp_min(0)
        union = (e - s) + (target[:, 1] - target[:, 0]) - inter
        iou = inter / union.clamp_min(1e-6)
        return (weight[keep] * (1 - iou)).sum() / weight[keep].sum()


class SpanPairIoULoss(BaseLoss):
    """
    JF7: 1 - the expected gain of a span drawn from the start and end distributions (taken independent),
    over every (start, end) pair at token resolution (`stride` steps; each token's start or end sits at
    the probability-weighted mean of its steps) with the end 1 to `max_len` steps' worth of tokens after
    the start (mbr_span's limit; longer and shorter pairs count 0). The gain of a pair is `mix` x its tIoU
    with the target plus (1 - mix) x the mean over `thetas` of sigmoid((tIoU - theta) / tau): the
    challenge score that mbr_span maximises at inference, made smooth. One row per prediction, weighted
    (0: no target), as SpanIoULoss. Pairs are a band [n, T, max_len / stride], not [n, T, T].
    """
    def __init__(self, target_key: str = "span_target", weight_key: str = "span_w", logits_key: str = "span_logits",
                 domain_key: str = "span_domain", stride: int = 4, thetas: tuple[float, ...] = (0.1, 0.3, 0.5),
                 tau: float = 0.05, mix: float = 0.5, max_len: int = 300) -> None:
        super().__init__()
        self.target_key, self.weight_key, self.logits_key, self.domain_key = target_key, weight_key, logits_key, domain_key
        self.stride, self.thetas, self.tau, self.mix = stride, tuple(thetas), tau, mix
        self.band = max(1, max_len // stride)

    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.target_key not in output:
            return output["step_logits"].sum() * 0.0
        weight = output[self.weight_key].float()
        keep = weight > 0
        if not keep.any():
            return output[self.logits_key].sum() * 0.0
        logits = output[self.logits_key][keep].float()                        # [n, L, 2]
        domain = output[self.domain_key][keep].bool()
        n, L, _ = logits.shape
        s = self.stride
        pad = -L % s
        p = torch.softmax(logits.masked_fill(~domain[..., None], -1e4), dim=1)
        pos = torch.arange(L, device=p.device, dtype=torch.float32)
        if pad:
            p, pos = F.pad(p, (0, 0, 0, pad)), F.pad(pos, (0, pad), value=float(L))
        T = (L + pad) // s
        pt = p.view(n, T, s, 2)
        mass = pt.sum(2)                                                        # [n, T, 2]
        at = (pt * pos.view(1, T, s, 1)).sum(2) / mass.clamp_min(1e-12)
        at = torch.where(mass > 1e-12, at, pos.view(1, T, s, 1)[:, :, 0].expand(n, T, 2))   # empty token: its first step
        D = min(self.band, T)
        # the end token i + d (d = 1 .. D) of each start token i; beyond the last token the mass is 0
        me = F.pad(mass[..., 1], (0, D))[:, 1:].unfold(1, D, 1)[:, :T]        # [n, T, D]
        en = F.pad(at[..., 1], (0, D), value=float(L))[:, 1:].unfold(1, D, 1)[:, :T]
        st = at[..., 0, None]                                                   # [n, T, 1]
        target = output[self.target_key][keep].float()
        a, b = target[:, 0, None, None], target[:, 1, None, None]
        inter = (torch.minimum(en, b) - torch.maximum(st, a)).clamp_min(0)
        union = (en - st).clamp_min(0) + (b - a) - inter
        iou = inter / union.clamp_min(1e-6)
        gain = self.mix * iou + (1 - self.mix) * sum(torch.sigmoid((iou - th) / self.tau) for th in self.thetas) / len(self.thetas)
        gain = torch.where(en > st, gain, torch.zeros_like(gain))
        expected = (mass[..., 0, None] * me * gain).sum((1, 2))
        return (weight[keep] * (1 - expected)).sum() / weight[keep].sum()
