
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout):

        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError(
                "Embedding size must be divisible by attention heads."
            )

        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(
            B, T, self.n_head, self.head_size
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_head, self.head_size
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_head, self.head_size
        ).transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_size)

        mask = self.mask[:T, :T]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        weights = self.attention_dropout(weights)

        output = weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, C)

        output = self.proj(output)

        return self.residual_dropout(output)


class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head,
        block_size,
        dropout
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.attention = CausalSelfAttention(
            n_embd,
            n_head,
            block_size,
            dropout
        )

        self.ln2 = nn.LayerNorm(n_embd)

        self.feed_forward = FeedForward(
            n_embd,
            dropout
        )

    def forward(self, x):

        x = x + self.attention(self.ln1(x))

        x = x + self.feed_forward(self.ln2(x))

        return x


class AstralLM(nn.Module):

    def __init__(self, vocab_size, config):

        super().__init__()

        self.block_size = config["block_size"]

        self.token_embedding = nn.Embedding(
            vocab_size,
            config["n_embd"]
        )

        self.position_embedding = nn.Embedding(
            self.block_size,
            config["n_embd"]
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                config["n_embd"],
                config["n_head"],
                self.block_size,
                config["dropout"]
            )
            for _ in range(config["n_layer"])
        ])

        self.final_norm = nn.LayerNorm(config["n_embd"])

        self.language_head = nn.Linear(
            config["n_embd"],
            vocab_size,
            bias=False
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0,
                std=0.02
            )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > self.block_size:
            raise ValueError("Input exceeds context length.")

        positions = torch.arange(
            T,
            device=idx.device
        )

        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)
        )

        x = self.blocks(x)

        x = self.final_norm(x)

        logits = self.language_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.1,
        stop_token_ids=None
    ):

        self.eval()

        stop_ids = set(stop_token_ids or [])

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            # Repetition penalty
            if repetition_penalty > 1:

                recent_tokens = set(
                    idx[0, -self.block_size:].tolist()
                )

                for token_id in recent_tokens:

                    value = logits[:, token_id]

                    logits[:, token_id] = torch.where(
                        value > 0,
                        value / repetition_penalty,
                        value * repetition_penalty
                    )

            # Temperature
            logits = logits / max(temperature, 0.01)

            # Top-k filtering
            if top_k is not None and top_k > 0:

                k = min(top_k, logits.size(-1))

                values, _ = torch.topk(logits, k)

                threshold = values[:, -1].unsqueeze(-1)

                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf")
                )

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

            # Stop at end-of-response token
            if int(next_token[0, 0]) in stop_ids:
                break

        return idx
