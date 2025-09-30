# -*- coding: utf-8 -*-
from minbpe import BasicTokenizer
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import sys, io

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------- Tokenizer ---------------- #
tokenizer = BasicTokenizer()
tokenizer.load(model_file="output/tokenizer/my_tokenizer.model")

def get_vocab_size(tok: BasicTokenizer) -> int:
    vocab = tok.vocab
    special_tokens = tok.special_tokens
    return len(vocab) + len(special_tokens)

# ---------------- Hyperparameters ---------------- #
torch.manual_seed(3647)

block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = get_vocab_size(tokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- Model Components ---------------- #
class Head(nn.Module):
    """Single self-attention head"""

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """Simple linear layer followed by non-linearity"""

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


# ---------------- GPT Language Model ---------------- #
class GPTLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_tokens.shape

        token_embedding = self.token_embedding_table(input_tokens)  # (B,T,C)
        positional_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,C)
        x = token_embedding + positional_embedding  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.final_layer_norm(x)  # (B,T,C)
        logits = self.final_linear_layer(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -block_size:]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)  # (B, T+1)
        return input_tokens


# ---------------- Initialize Model ---------------- #
model = GPTLanguageModel()
model = model.to(device)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# Quick test
batch_size = 1
seq_length = 6
x = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
logits, loss = model(x)
print(logits.shape, loss)

# ---------------- Utils ---------------- #
def print_model_structure(model: torch.nn.Module, indent: str = "") -> None:
    """Print model structure in a hierarchical format"""
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(
            f"{indent}├─ {name}: {child.__class__.__name__} ({params:,} parameters)"
        )
        print_model_structure(child, indent + "│  ")


print_model_structure(model)


def get_model_stats(model: torch.nn.Module) -> pd.DataFrame:
    """Create a DataFrame with detailed layer statistics"""
    stats = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            params = sum(p.numel() for p in module.parameters())
            stats.append(
                {
                    "Layer Name": name,
                    "Type": module.__class__.__name__,
                    "Parameters": params,
                    "Trainable": sum(
                        p.numel() for p in module.parameters() if p.requires_grad
                    ),
                }
            )
    return pd.DataFrame(stats)


stats_df = get_model_stats(model)
print(stats_df.to_string(index=False))  # UTF-8 safe
