"""多類別語意分割 loss: CE (含 class weight) + soft Dice (macro, 含/不含背景)。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss。輸入 logits [B,C,H,W], target long [B,H,W]。

    ignore_index_in_dice: 計算 Dice 時忽略的類別 (預設 0=背景, 與 mIoU 計算一致)。
    """

    def __init__(self, num_classes: int, ignore_index_in_dice: Optional[int] = 0,
                 smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index_in_dice = ignore_index_in_dice
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target.clamp_min(0), num_classes=self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        inter = (probs * target_oh).sum(dim=dims)
        denom = probs.sum(dim=dims) + target_oh.sum(dim=dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)  # [C]

        mask = torch.ones(self.num_classes, dtype=torch.bool, device=logits.device)
        if self.ignore_index_in_dice is not None:
            mask[self.ignore_index_in_dice] = False
        return 1.0 - dice[mask].mean()


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skel(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Differentiable skeleton via iterative morphological opening (Shit et al., CVPR 2021).

    Input x: (B, 1, H, W) in [0, 1].
    """
    img1 = _soft_open(x)
    skel = F.relu(x - img1)
    for _ in range(iters):
        x = _soft_erode(x)
        img1 = _soft_open(x)
        delta = F.relu(x - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """centerline-Dice loss for tubular structure (Shit et al., 2021).

    Operates on the positive class of a binary segmentation. Expects logits [B,C,H,W],
    target long [B,H,W], C=2 (background, positive_class). For multi-class, only the
    `positive_class` index is used.
    """

    def __init__(self, positive_class: int = 1, iters: int = 3, smooth: float = 1e-5):
        super().__init__()
        self.positive_class = positive_class
        self.iters = iters
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        p = probs[:, self.positive_class:self.positive_class + 1, :, :]  # (B,1,H,W)
        t = (target == self.positive_class).float().unsqueeze(1)  # (B,1,H,W)
        sp = soft_skel(p, iters=self.iters)
        st = soft_skel(t, iters=self.iters)
        cl_tprec = (sp * t).sum(dim=(2, 3)) + self.smooth
        cl_tprec = cl_tprec / (sp.sum(dim=(2, 3)) + self.smooth)
        cl_trec = (st * p).sum(dim=(2, 3)) + self.smooth
        cl_trec = cl_trec / (st.sum(dim=(2, 3)) + self.smooth)
        cldice = 2.0 * cl_tprec * cl_trec / (cl_tprec + cl_trec + self.smooth)
        return 1.0 - cldice.mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Reduces loss for well-classified examples, focusing training on hard negatives.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma * ce).mean()
        return focal


class CEDiceLoss(nn.Module):
    """加權 CE + Dice (+ 可選 clDice) 的組合 loss。

    Args:
        num_classes
        class_weights: tensor [C] (給 CE 用), None 表示不加權
        ce_weight, dice_weight, cldice_weight: 三個分量的相對權重 (cldice_weight=0 關閉)
        ignore_index_in_dice: Dice 平均時忽略的類別
        cldice_iters: soft skeleton 的迭代次數 (預設 3)
        cldice_positive_class: clDice 對應的正類別 index (預設 1，即 bg=0 / fg=1 之 fg)
        focal_weight: focal loss 權重 (0 = 關閉, 取代 CE)
        focal_gamma: focal loss gamma 參數
    """

    def __init__(self,
                 num_classes: int,
                 class_weights: Optional[torch.Tensor] = None,
                 ce_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 cldice_weight: float = 0.0,
                 ignore_index_in_dice: Optional[int] = 0,
                 cldice_iters: int = 3,
                 cldice_positive_class: int = 1,
                 focal_weight: float = 0.0,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.focal_weight = focal_weight
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes,
                             ignore_index_in_dice=ignore_index_in_dice)
        self.cldice = SoftClDiceLoss(positive_class=cldice_positive_class,
                                     iters=cldice_iters)
        self.focal = FocalLoss(alpha=class_weights, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = self.ce(logits, target)
        dice = self.dice(logits, target)
        parts = {"ce": float(ce.detach()), "dice": float(dice.detach())}
        total = self.ce_weight * ce + self.dice_weight * dice
        if self.focal_weight > 0:
            fl = self.focal(logits, target)
            total = total + self.focal_weight * fl
            parts["focal"] = float(fl.detach())
        if self.cldice_weight > 0:
            cl = self.cldice(logits, target)
            total = total + self.cldice_weight * cl
            parts["cldice"] = float(cl.detach())
        return total, parts


class SegLoss(nn.Module):
    """Ignore-aware weighted CE, foreground Dice and directional confusion cost."""

    def __init__(
        self,
        num_classes: int,
        ignore_value: int = 255,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        cost_weight: float = 0.0,
        craquelure_to_crack_cost: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_value = ignore_value
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cost_weight = cost_weight
        self.craquelure_to_crack_cost = craquelure_to_crack_cost
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_value,
        )
        valid = target != self.ignore_value
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        probabilities = F.softmax(logits, dim=1) * valid.unsqueeze(1)
        one_hot = F.one_hot(safe_target, self.num_classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(probabilities.dtype) * valid.unsqueeze(1)
        intersection = (probabilities * one_hot).sum((0, 2, 3))
        denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
        dice = 1 - ((2 * intersection[1:] + 1) / (denominator[1:] + 1)).mean()
        cost = logits.new_zeros(())
        if self.num_classes == 3 and self.craquelure_to_crack_cost > 0:
            craquelure = valid & (target == 2)
            if craquelure.any():
                cost = (
                    probabilities[:, 1][craquelure].mean()
                    * self.craquelure_to_crack_cost
                    / 10.0
                )
        loss = self.ce_weight * ce + self.dice_weight * dice + self.cost_weight * cost
        return loss, {
            "ce": float(ce.detach()),
            "dice": float(dice.detach()),
            "cost": float(cost.detach()),
        }
