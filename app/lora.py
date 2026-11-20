"""Milestones 2 & 5: From-scratch LoRA (lfllm-lflora) adapter injection, training, and merging.

Uses PyTorch when available; falls back cleanly to a pure-NumPy LoRA engine if PyTorch DLLs
are unavailable on the host machine.
"""
from __future__ import annotations

import math
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class SimpleWordTokenizer:
    """Lightweight word/punctuation tokenizer for offline laptop & CI LoRA training."""

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, texts: list[str] | None = None):
        self.token_to_id: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        if texts:
            self.fit(texts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def fit(self, texts: list[str]) -> "SimpleWordTokenizer":
        for text in texts:
            for tok in self._tokenize(text):
                if tok not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[tok] = idx
                    self.id_to_token[idx] = tok
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    def encode(self, text: str, max_length: int = 128, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id.get(t, self.token_to_id[self.UNK_TOKEN]) for t in self._tokenize(text)]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids[: max_length - 2] + [self.eos_token_id]
        else:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in special:
                continue
            tokens.append(self.id_to_token.get(int(idx), ""))
        text = " ".join(t for t in tokens if t)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text.strip()


@dataclass
class TrainingResult:
    loss_history: list[float] = field(default_factory=list)
    initial_loss: float = 0.0
    final_loss: float = 0.0
    epochs: int = 0
    steps: int = 0
    replaced_modules: list[str] = field(default_factory=list)
    param_stats: dict[str, float | int] = field(default_factory=dict)
    elapsed_sec: float = 0.0


if TORCH_AVAILABLE:

    class TinyTransformerBlock(nn.Module):
        """Compact causal Transformer block with named projections for LoRA injection."""

        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.ln1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, d_model * 4)
            self.fc2 = nn.Linear(d_model * 4, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, seq_len, dim = x.shape
            h = self.ln1(x)
            q = self.q_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
            x = x + self.out_proj(context)

            h2 = self.ln2(x)
            x = x + self.fc2(F.gelu(self.fc1(h2)))
            return x

    class TinyCausalLM(nn.Module):
        """Lightweight causal Transformer LM compatible with lfllm-lflora LoRA injection & merging."""

        def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, max_seq_len: int = 256):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.max_seq_len = max_seq_len
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.blocks = nn.ModuleList([TinyTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            bsz, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
            x = self.token_emb(input_ids) + self.pos_emb(positions)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            return self.lm_head(x)

    class LoRALinear(nn.Module):
        """Low-Rank Adaptation wrapper implementing h = W_0 x + (alpha / r) * B(A(x))."""

        def __init__(self, base_layer: nn.Module, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
            super().__init__()
            if rank <= 0:
                raise ValueError("LoRA rank must be a positive integer.")

            self.rank = rank
            self.alpha = float(alpha)
            self.scaling = self.alpha / self.rank
            self.merged = False
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

            if isinstance(base_layer, nn.Linear):
                self.in_features = base_layer.in_features
                self.out_features = base_layer.out_features
                self.is_conv1d = False
            elif hasattr(base_layer, "weight") and base_layer.weight.ndim == 2:
                self.in_features, self.out_features = base_layer.weight.shape
                self.is_conv1d = True
            else:
                raise TypeError(f"Unsupported layer type for LoRALinear: {type(base_layer)}")

            self.base_layer = base_layer
            for param in self.base_layer.parameters():
                param.requires_grad = False

            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def delta_weight(self) -> torch.Tensor:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            return delta.T if self.is_conv1d else delta

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_layer(x)
            if self.merged:
                return base_out
            lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
            return base_out + lora_out

        @torch.no_grad()
        def merge(self) -> None:
            if self.merged:
                return
            self.base_layer.weight.add_(self.delta_weight())
            self.merged = True

        @torch.no_grad()
        def unmerge(self) -> None:
            if not self.merged:
                return
            self.base_layer.weight.sub_(self.delta_weight())
            self.merged = False

        @torch.no_grad()
        def to_merged_base(self) -> nn.Module:
            self.merge()
            return self.base_layer

else:
    # Pure-NumPy fallback when PyTorch DLLs are unavailable (e.g. Windows without VC++ Redistributable)

    class _NpLinear:
        def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, bias: bool = True):
            self.in_features = in_features
            self.out_features = out_features
            scale = 1.0 / math.sqrt(in_features)
            self.weight = rng.uniform(-scale, scale, size=(out_features, in_features)).astype(np.float32)
            self.bias = np.zeros(out_features, dtype=np.float32) if bias else None

        def __call__(self, x: np.ndarray) -> np.ndarray:
            out = x @ self.weight.T
            if self.bias is not None:
                out = out + self.bias
            return out

        def numel(self) -> int:
            return int(self.weight.size + (self.bias.size if self.bias is not None else 0))

    class LoRALinear:  # type: ignore[no-redef]
        """Pure-NumPy LoRA adapter implementing h = W_0 x + (alpha / r) * B(A(x))."""

        def __init__(self, base_layer: _NpLinear, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
            if rank <= 0:
                raise ValueError("LoRA rank must be a positive integer.")
            self.base_layer = base_layer
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
            self.rank = rank
            self.alpha = float(alpha)
            self.scaling = self.alpha / self.rank
            self.merged = False
            rng = np.random.default_rng(42)
            self.lora_A = rng.normal(0.0, 1.0 / math.sqrt(self.in_features), size=(rank, self.in_features)).astype(np.float32)
            self.lora_B = np.zeros((self.out_features, rank), dtype=np.float32)

        def delta_weight(self) -> np.ndarray:
            return (self.lora_B @ self.lora_A) * self.scaling

        def __call__(self, x: np.ndarray) -> np.ndarray:
            base_out = self.base_layer(x)
            if self.merged:
                return base_out
            return base_out + ((x @ self.lora_A.T) @ self.lora_B.T) * self.scaling

        def merge(self) -> None:
            if not self.merged:
                self.base_layer.weight = self.base_layer.weight + self.delta_weight()
                self.merged = True

        def unmerge(self) -> None:
            if self.merged:
                self.base_layer.weight = self.base_layer.weight - self.delta_weight()
                self.merged = False

        def to_merged_base(self) -> _NpLinear:
            self.merge()
            return self.base_layer

    class TinyCausalLM:  # type: ignore[no-redef]
        """Pure-NumPy compact causal LM with LoRA-injectable projections."""

        def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, max_seq_len: int = 256, seed: int = 42):
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.max_seq_len = max_seq_len
            rng = np.random.default_rng(seed)
            self.token_emb = rng.normal(0.0, 0.08, size=(vocab_size, d_model)).astype(np.float32)
            self.q_proj: _NpLinear | LoRALinear = _NpLinear(d_model, d_model, rng)
            self.v_proj: _NpLinear | LoRALinear = _NpLinear(d_model, d_model, rng)
            self.lm_head = _NpLinear(d_model, vocab_size, rng, bias=False)

        def eval(self) -> "TinyCausalLM":
            return self

        def train(self) -> "TinyCausalLM":
            return self

        def modules(self) -> list[Any]:
            return [self, self.q_proj, self.v_proj, self.lm_head]

        def named_modules(self) -> list[tuple[str, Any]]:
            return [("q_proj", self.q_proj), ("v_proj", self.v_proj), ("lm_head", self.lm_head)]

        def __call__(self, input_ids: Any) -> np.ndarray:
            ids = np.asarray(input_ids, dtype=np.int64)
            x = self.token_emb[ids]
            h = x + 0.5 * np.tanh(self.q_proj(x) + self.v_proj(x))
            return self.lm_head(h)


def inject_lora(
    model: Any,
    rank: int = 4,
    alpha: float = 8.0,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "c_attn"),
    dropout: float = 0.0,
) -> tuple[Any, list[str]]:
    """Freeze all base parameters and inject LoRALinear adapters into matching modules."""
    if not TORCH_AVAILABLE:
        replaced = []
        for name in ("q_proj", "v_proj"):
            if name in target_modules and hasattr(model, name):
                layer = getattr(model, name)
                if not isinstance(layer, LoRALinear):
                    setattr(model, name, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
                    replaced.append(name)
        return model, replaced

    for param in model.parameters():
        param.requires_grad = False

    replaced: list[str] = []

    def _replace(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            is_target = any(child_name == t or full_name.endswith(t) for t in target_modules)
            is_linear_like = isinstance(child, nn.Linear) or child.__class__.__name__ == "Conv1D"
            if is_target and is_linear_like:
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(full_name)
            else:
                _replace(child, full_name)

    _replace(model)
    return model, replaced


def count_parameters(model: Any) -> dict[str, float | int]:
    """Return total, trainable, frozen parameter counts and trainable percentage."""
    if not TORCH_AVAILABLE:
        base_total = int(model.token_emb.size + model.lm_head.numel())
        trainable = 0
        for name in ("q_proj", "v_proj"):
            layer = getattr(model, name)
            if isinstance(layer, LoRALinear):
                base_total += layer.base_layer.numel()
                t = int(layer.lora_A.size + layer.lora_B.size)
                trainable += t
                base_total += t
            else:
                base_total += layer.numel()
        frozen = base_total - trainable
        pct = (trainable / base_total * 100.0) if base_total > 0 else 0.0
        return {"total": base_total, "trainable": trainable, "frozen": frozen, "trainable_pct": round(pct, 4)}

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": round(pct, 4),
    }


def extract_adapter_state_dict(model: Any) -> dict[str, Any]:
    """Extract only LoRA adapter parameters (lora_A, lora_B) for lightweight persistence."""
    if not TORCH_AVAILABLE:
        state = {}
        for name, m in model.named_modules():
            if isinstance(m, LoRALinear):
                state[f"{name}.lora_A"] = m.lora_A.copy()
                state[f"{name}.lora_B"] = m.lora_B.copy()
        return state

    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def save_adapter(model: Any, path: str | Path) -> Path:
    """Save LoRA adapter weights to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = extract_adapter_state_dict(model)
    if TORCH_AVAILABLE:
        torch.save(state, out)
    else:
        with out.open("wb") as f:
            pickle.dump(state, f)
    return out


def merge_lora_adapters(model: Any) -> tuple[Any, int]:
    """Milestone 5: Merge all LoRALinear adapters into base layers and unwrap for deployment."""
    if not TORCH_AVAILABLE:
        merged_count = 0
        for name in ("q_proj", "v_proj"):
            layer = getattr(model, name, None)
            if isinstance(layer, LoRALinear):
                setattr(model, name, layer.to_merged_base())
                merged_count += 1
        return model, merged_count

    merged_count = 0

    def _unwrap(module: nn.Module) -> None:
        nonlocal merged_count
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.to_merged_base())
                merged_count += 1
            else:
                _unwrap(child)

    _unwrap(model)
    return model, merged_count


def verify_merge_equivalence(
    model: Any,
    sample_input_ids: Any,
    atol: float = 1e-4,
) -> dict[str, float | bool | int]:
    """Merge LoRA adapters and verify numerical equivalence + inference latency."""
    model.eval()
    if not TORCH_AVAILABLE:
        ids = np.asarray(sample_input_ids, dtype=np.int64)
        t0 = time.perf_counter()
        unmerged_logits = model(ids)
        unmerged_ms = (time.perf_counter() - t0) * 1000.0

        _, merged_layers = merge_lora_adapters(model)

        t1 = time.perf_counter()
        merged_logits = model(ids)
        merged_ms = (time.perf_counter() - t1) * 1000.0

        max_abs_diff = float(np.max(np.abs(unmerged_logits - merged_logits)))
        return {
            "merged_layers": merged_layers,
            "max_abs_diff": max_abs_diff,
            "is_equivalent": max_abs_diff <= atol,
            "unmerged_ms": round(unmerged_ms, 3),
            "merged_ms": round(merged_ms, 3),
        }

    with torch.no_grad():
        t0 = time.perf_counter()
        unmerged_out = model(sample_input_ids)
        unmerged_ms = (time.perf_counter() - t0) * 1000.0
        unmerged_logits = unmerged_out.logits if hasattr(unmerged_out, "logits") else unmerged_out

        _, merged_layers = merge_lora_adapters(model)

        t1 = time.perf_counter()
        merged_out = model(sample_input_ids)
        merged_ms = (time.perf_counter() - t1) * 1000.0
        merged_logits = merged_out.logits if hasattr(merged_out, "logits") else merged_out

        max_abs_diff = float((unmerged_logits - merged_logits).abs().max().item())
        return {
            "merged_layers": merged_layers,
            "max_abs_diff": max_abs_diff,
            "is_equivalent": max_abs_diff <= atol,
            "unmerged_ms": round(unmerged_ms, 3),
            "merged_ms": round(merged_ms, 3),
        }


def _np_train_lora(
    model: TinyCausalLM,
    tokenizer: SimpleWordTokenizer,
    texts: list[str],
    epochs: int,
    lr: float,
    max_length: int,
    progress_callback: Callable[[int, int, float], None] | None,
) -> TrainingResult:
    pad_id = tokenizer.pad_token_id
    encoded = [tokenizer.encode(t, max_length=max_length) for t in texts]
    encoded = [seq for seq in encoded if len(seq) >= 2]
    max_len = max(len(seq) for seq in encoded)
    batch = np.full((len(encoded), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(encoded):
        batch[i, : len(seq)] = seq

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    mask = (targets != pad_id).astype(np.float32)
    denom = float(mask.sum())

    loss_history: list[float] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        x = model.token_emb[inputs]  # (B, T, D)
        pre = model.q_proj(x) + model.v_proj(x)
        th = np.tanh(pre)
        h = x + 0.5 * th
        logits = model.lm_head(h)  # (B, T, V)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(shifted)
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

        b_idx = np.arange(targets.shape[0])[:, None]
        t_idx = np.arange(targets.shape[1])[None, :]
        target_probs = np.clip(probs[b_idx, t_idx, targets], 1e-9, 1.0)
        loss_val = float((-np.log(target_probs) * mask).sum() / denom)
        loss_history.append(round(loss_val, 4))

        # Backprop to LoRA A & B matrices
        dlogits = probs
        dlogits[b_idx, t_idx, targets] -= 1.0
        dlogits = dlogits * (mask[..., None] / denom)

        dh = dlogits @ model.lm_head.weight  # (B, T, D)
        dpre = dh * 0.5 * (1.0 - th * th)  # (B, T, D)
        x_flat = x.reshape(-1, x.shape[-1])
        dpre_flat = dpre.reshape(-1, dpre.shape[-1])

        for layer in (model.q_proj, model.v_proj):
            if isinstance(layer, LoRALinear):
                xa = x_flat @ layer.lora_A.T  # (N, r)
                grad_B = (dpre_flat.T @ xa) * layer.scaling
                grad_A = ((dpre_flat @ layer.lora_B).T @ x_flat) * layer.scaling
                norm_B = float(np.linalg.norm(grad_B)) + 1e-8
                norm_A = float(np.linalg.norm(grad_A)) + 1e-8
                layer.lora_B -= (lr * 4.0) * (grad_B / norm_B)
                layer.lora_A -= (lr * 4.0) * (grad_A / norm_A)

        if progress_callback is not None:
            progress_callback(epoch, epochs, loss_val)

    elapsed = time.perf_counter() - start_time
    replaced = [name for name, m in model.named_modules() if isinstance(m, LoRALinear)]
    return TrainingResult(
        loss_history=loss_history,
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        epochs=epochs,
        steps=epochs,
        replaced_modules=replaced,
        param_stats=count_parameters(model),
        elapsed_sec=round(elapsed, 3),
    )


def train_lora(
    model: Any,
    tokenizer: SimpleWordTokenizer,
    texts: list[str],
    epochs: int = 6,
    lr: float = 5e-3,
    max_length: int = 96,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> TrainingResult:
    """Fine-tune LoRA parameters on causal LM next-token prediction loss."""
    if not texts:
        raise ValueError("Training texts must not be empty.")

    if not TORCH_AVAILABLE:
        return _np_train_lora(model, tokenizer, texts, epochs, lr, max_length, progress_callback)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Call inject_lora() before training.")

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    pad_id = tokenizer.pad_token_id

    encoded = [tokenizer.encode(t, max_length=max_length) for t in texts]
    encoded = [seq for seq in encoded if len(seq) >= 2]
    if not encoded:
        raise ValueError("Encoded training sequences are too short.")

    max_len = max(len(seq) for seq in encoded)
    batch = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(encoded):
        batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    model.train()
    loss_history: list[float] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(inputs)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        loss_val = float(loss.item())
        loss_history.append(round(loss_val, 4))
        if progress_callback is not None:
            progress_callback(epoch, epochs, loss_val)

    elapsed = time.perf_counter() - start_time
    replaced = [name for name, m in model.named_modules() if isinstance(m, LoRALinear)]
    return TrainingResult(
        loss_history=loss_history,
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        epochs=epochs,
        steps=epochs,
        replaced_modules=replaced,
        param_stats=count_parameters(model),
        elapsed_sec=round(elapsed, 3),
    )


def build_tiny_lora_model(
    texts: list[str],
    rank: int = 4,
    alpha: float = 8.0,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    seed: int = 42,
) -> tuple[TinyCausalLM, SimpleWordTokenizer, list[str]]:
    """Convenience builder: tokenizer + TinyCausalLM + LoRA injection."""
    tokenizer = SimpleWordTokenizer(texts)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        model = TinyCausalLM(vocab_size=tokenizer.vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    else:
        model = TinyCausalLM(vocab_size=tokenizer.vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, seed=seed)
    model, replaced = inject_lora(model, rank=rank, alpha=alpha, target_modules=("q_proj", "v_proj"))
    return model, tokenizer, replaced


def make_input_batch(ids: list[int]) -> Any:
    """Wrap token IDs into a batch tensor (torch.Tensor or np.ndarray)."""
    if TORCH_AVAILABLE:
        return torch.tensor([ids], dtype=torch.long)
    return np.array([ids], dtype=np.int64)
