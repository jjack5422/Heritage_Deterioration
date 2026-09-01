"""Pure-PyTorch DER and decomposed FFN experts used inside SAM3 fusion layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DecomposedLinearDelta(nn.Module):
    """Rank-limited weight delta, random down projection and zero up projection."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.up(self.down(inputs))


class DecomposedFfnExpert(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, rank: int) -> None:
        super().__init__()
        self.linear1_delta = DecomposedLinearDelta(d_model, dim_feedforward, rank)
        self.linear2_delta = DecomposedLinearDelta(dim_feedforward, d_model, rank)


class DomainContextAttention(nn.Module):
    """Attend a concept query over the full spatial visual-memory sequence."""

    def __init__(self, d_model: int, heads: int = 8) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)

    def forward(self, concept: Tensor, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.ndim != 3 or visual_tokens.shape[1] < 2:
            raise ValueError("domain context requires at least two full spatial visual tokens")
        if concept.ndim != 2:
            raise ValueError("concept embedding must be [B,D]")
        context, _ = self.attention(concept[:, None], visual_tokens, visual_tokens, need_weights=False)
        return context[:, 0]


@dataclass(frozen=True)
class RouterDiagnostics:
    logits: Tensor
    soft_probabilities: Tensor
    hard_assignments: Tensor
    weights: Tensor
    entropy: Tensor
    top1_usage: Tensor
    top2_usage: Tensor


class DynamicExpertRouter(nn.Module):
    def __init__(self, d_model: int, experts: int = 4, top_k: int = 2, heads: int = 8) -> None:
        super().__init__()
        if top_k != 2 or experts != 4:
            raise ValueError("approved DER contract requires four experts and top-k=2")
        self.experts = experts
        self.top_k = top_k
        self.context_attention = DomainContextAttention(d_model, heads=heads)
        self.projection = nn.Sequential(
            nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, experts)
        )

    def forward(self, local_tokens: Tensor, concept: Tensor, visual_tokens: Tensor, *, temperature: float) -> RouterDiagnostics:
        if local_tokens.ndim != 3:
            raise ValueError("local tokens must be [B,N,D]")
        context = self.context_attention(concept, visual_tokens)
        count = local_tokens.shape[1]
        features = torch.cat((
            local_tokens,
            context[:, None].expand(-1, count, -1),
            concept.detach()[:, None].expand(-1, count, -1),
        ), dim=-1)
        logits = self.projection(features) / float(temperature)
        soft = logits.softmax(dim=-1)
        _, indices = torch.topk(logits, self.top_k, dim=-1)
        active = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, indices, True)
        masked = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
        weights = masked.softmax(dim=-1)
        hard = active.to(logits.dtype)
        entropy = -(soft * soft.clamp_min(1e-8).log()).sum(dim=-1).mean()
        top1 = F.one_hot(indices[..., 0], self.experts).float().mean(dim=(0, 1))
        top2 = hard.mean(dim=(0, 1))
        return RouterDiagnostics(logits, soft, hard, weights, entropy, top1, top2)


class DaMoeFfn(nn.Module):
    """Frozen shared FFN bases plus four routed low-rank expert deltas."""

    def __init__(
        self,
        base_linear1: nn.Linear,
        base_linear2: nn.Linear,
        *,
        activation: nn.Module | None = None,
        dropout: nn.Module | None = None,
        experts: int = 4,
        rank: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.base_linear1 = base_linear1
        self.base_linear2 = base_linear2
        for parameter in (*base_linear1.parameters(), *base_linear2.parameters()):
            parameter.requires_grad_(False)
        self.activation = activation or nn.GELU()
        self.dropout = dropout or nn.Identity()
        self.experts = nn.ModuleList([
            DecomposedFfnExpert(base_linear1.in_features, base_linear1.out_features, rank)
            for _ in range(experts)
        ])
        self.router = DynamicExpertRouter(base_linear1.in_features, experts=experts, top_k=top_k)

    def forward(
        self,
        tokens: Tensor,
        visual_tokens: Tensor,
        concept: Tensor,
        *,
        temperature: float,
    ) -> tuple[Tensor, RouterDiagnostics]:
        diagnostics = self.router(tokens, concept, visual_tokens, temperature=temperature)
        expert_outputs = []
        for expert in self.experts:
            hidden = self.activation(self.base_linear1(tokens) + expert.linear1_delta(tokens))
            hidden = self.dropout(hidden)
            expert_outputs.append(self.base_linear2(hidden) + expert.linear2_delta(hidden))
        stacked = torch.stack(expert_outputs, dim=-2)
        output = (stacked * diagnostics.weights.unsqueeze(-1)).sum(dim=-2)
        return output, diagnostics


def router_temperature(epoch: int) -> float:
    """Linearly anneal from 2.0 to 1.0 over epoch indices 0..4."""

    return 1.0 if epoch >= 4 else 2.0 - epoch / 4.0
