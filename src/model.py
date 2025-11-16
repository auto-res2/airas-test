"""src/model.py – architectures + optimiser factories"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

# ---------------------------------------------------------------------------
# ResNet-20 (classic CIFAR variant)
# ---------------------------------------------------------------------------

def _conv3x3(inp, out, stride=1):
    return nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(inp, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inp != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = _conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 3)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [_BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Tiny Vision-Transformer for CIFAR-10 (optional)
# ---------------------------------------------------------------------------
class ViTSmall(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        img_size, patch = 32, 4
        dim = cfg.model.d_model
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_patches, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=cfg.model.num_heads,
            dim_feedforward=cfg.model.d_ff,
            dropout=cfg.model.dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.model.num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, cfg.dataset.num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


# ---------------------------------------------------------------------------
# Transformer-Small NMT
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class TransformerSmallNMT(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, cfg: DictConfig):
        super().__init__()
        d = cfg.model.d_model
        self.src_emb = nn.Embedding(src_vocab_size, d)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d)
        self.pos = PositionalEncoding(d)
        self.transformer = nn.Transformer(
            d_model=d,
            nhead=cfg.model.num_heads,
            num_encoder_layers=cfg.model.num_layers,
            num_decoder_layers=cfg.model.num_layers,
            dim_feedforward=cfg.model.d_ff,
            dropout=cfg.model.dropout,
        )
        self.generator = nn.Linear(d, tgt_vocab_size)

    def forward(self, src, tgt):
        src = self.pos(self.src_emb(src))
        tgt = self.pos(self.tgt_emb(tgt))
        memory = self.transformer.encoder(src)
        out = self.transformer.decoder(tgt, memory)
        return self.generator(out), out


# ---------------------------------------------------------------------------
# Optimisers: G-AGD & AGD
# ---------------------------------------------------------------------------
class GAGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, delta=1.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, delta=delta))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            delta = group["delta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["var"] = torch.tensor(0.0, device=p.device, dtype=p.dtype)
                    state["prev_m_hat"] = torch.zeros_like(p)
                exp_avg, var = state["exp_avg"], state["var"]
                state["step"] += 1
                step = state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                m_hat = exp_avg / (1 - beta1 ** step)
                s_t = m_hat - state["prev_m_hat"] if step > 1 else m_hat
                state["prev_m_hat"].copy_(m_hat)
                var.mul_(beta2).add_(s_t.pow(2).mean(), alpha=1 - beta2)
                var_hat = var / (1 - beta2 ** step)
                denom = torch.maximum(var_hat.sqrt(), torch.tensor(delta * math.sqrt(1 - beta2 ** step), device=p.device))
                p.addcdiv_(m_hat, denom, value=-group["lr"])
        return loss


class AGD(GAGD):
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            delta = group["delta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["var"] = torch.zeros_like(p)
                    state["prev_m_hat"] = torch.zeros_like(p)
                exp_avg, var = state["exp_avg"], state["var"]
                state["step"] += 1
                step = state["step"]
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                m_hat = exp_avg / (1 - beta1 ** step)
                s_t = m_hat - state["prev_m_hat"] if step > 1 else m_hat
                state["prev_m_hat"].copy_(m_hat)
                var.mul_(beta2).addcmul_(s_t, s_t, value=1 - beta2)
                var_hat = var / (1 - beta2 ** step)
                denom = torch.maximum(var_hat.sqrt(), torch.full_like(var_hat, delta * math.sqrt(1 - beta2 ** step)))
                p.addcdiv_(m_hat, denom, value=-group["lr"])
        return loss


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig, extra: Dict):
    if cfg.model.name == "resnet20":
        return ResNet20(num_classes=extra.get("num_classes", 10))
    if cfg.model.name == "transformer_small":
        if extra["task"] == "nmt":
            return TransformerSmallNMT(len(extra["src_vocab"]), len(extra["tgt_vocab"]), cfg)
        return ViTSmall(cfg)
    raise ValueError(cfg.model.name)


def get_optimizer(model: nn.Module, cfg: DictConfig):
    name = cfg.training.optimizer.lower()
    if name == "g_agd":
        return GAGD(model.parameters(), lr=cfg.training.initial_learning_rate, betas=tuple(cfg.training.betas), weight_decay=cfg.training.weight_decay, delta=cfg.training.delta)
    if name == "agd":
        return AGD(model.parameters(), lr=cfg.training.initial_learning_rate, betas=tuple(cfg.training.betas), weight_decay=cfg.training.weight_decay, delta=cfg.training.delta)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.training.initial_learning_rate, betas=tuple(cfg.training.betas), weight_decay=cfg.training.weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.training.initial_learning_rate, momentum=cfg.training.betas[0], weight_decay=cfg.training.weight_decay)
    raise ValueError(name)


def get_lr_scheduler(opt, cfg: DictConfig, total_steps: int):
    sched_cfg = cfg.training.scheduler
    if sched_cfg.name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    if sched_cfg.name == "inverse_sqrt":
        warm = sched_cfg.warmup_updates
        return optim.lr_scheduler.LambdaLR(opt, lambda step: (step / warm) if step < warm else (warm ** 0.5) / (step ** 0.5))
    return None