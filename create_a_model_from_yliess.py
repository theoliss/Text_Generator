from __future__ import annotations

from itertools import cycle
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import builtins
import os
import requests
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F



class HarryPotter:
    def __init__(self, path: str) -> None:
        self.path = path
        self.content = open(self.path, 'r').read()

    def __len__(self) -> int: return len(self.content)


class Vocab:
    def __init__(self) -> None:
        self.encoder = tiktoken.get_encoding("gpt2")

    def __len__(self) -> int: return self.encoder.n_vocab
    def __getitem__(self, idx: str | int) -> int | str:
        match type(idx):
            case builtins.str: return self.encoder.encode_single_token(idx)
            case builtins.int: return self.encoder.decode([idx])

    def encode(self, chars: str) -> list[int]: return self.encoder.encode_ordinary(chars)
    def decode(self, idxs: list[int]) -> str: return self.encoder.decode(idxs)
    

class HarryPotterDataset(Dataset):
    def __init__(self, context_length: int, books: HarryPotter, vocab: Vocab, train: bool) -> None:
        super().__init__()
        self.context_length = context_length
        self.train = train
        self.data = torch.tensor(vocab.encode(books.content), dtype=torch.long)
        
        n = int(0.9 * len(self.data))
        self.data = self.data[:n] if train else self.data[n:]

    def __len__(self) -> int: return len(self.data) - self.context_length
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y
    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head: int, embed_dim: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = map(lambda x: x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2), torch.chunk(self.qkv(x), 3, dim=-1))
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        h_dim = 4 * embed_dim
        self.l1 = nn.Linear(embed_dim, h_dim, bias=False)
        self.l2 = nn.Linear(h_dim, embed_dim, bias=False)
        self.l3 = nn.Linear(embed_dim, h_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(F.silu(self.l1(x)) * self.l3(x))


class Block(nn.Module):
    def __init__(self, n_head: int, embed_dim: int) -> None:
        super().__init__()
        self.mhsa = nn.Sequential(RMSNorm(embed_dim), MultiHeadSelfAttention(n_head, embed_dim))
        self.ffwd = nn.Sequential(RMSNorm(embed_dim), FeedForward(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mhsa(x)
        x = x + self.ffwd(x)
        return x


class LLM(nn.Module):
    def __init__(self, vocab: Vocab, context_length: int, embed_dim: int, n_head: int, n_layer: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_emb = nn.Embedding(len(vocab), embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.blocks = nn.Sequential(*[Block(n_head, embed_dim) for _ in range(n_layer)])
        self.rmsn = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, len(vocab))

        # https://paperswithcode.com/method/weight-tying
        self.token_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

        self.num_parameters = sum(p.numel() for p in self.parameters())
        self.num_buffers = sum(b.numel() for b in self.buffers())

    def _init_weights(self, module: nn.Module) -> None:
        match type(module):
            case nn.Embedding: nn.init.normal_(module.weight, mean=0, std=0.02)
            case nn.Linear:
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None: nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        t = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(t)
        x = self.blocks(x)
        x = self.lm_head(self.rmsn(x))
        return x
    
    def fit(self, loader: DataLoader, optimizer: Optimizer, scheduler: LRScheduler, steps: int, accumulate: int, device: torch.device) -> list[float]:
        self.train()
        history = []
        batches = iter(cycle(loader))
        pbar = tqdm(range(steps), desc='Training')
        for _ in pbar:
            with autocast(dtype=torch.float16):
                loss = 0.0
                for _ in range(accumulate):
                    x, y = map(lambda t: t.to(device), next(batches))
                    logits = self(x)
                    B, T, C = logits.shape
                    loss += F.cross_entropy(logits.view(B * T, C), y.view(B * T))
                loss = loss / accumulate
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(nll=f"{loss.item():.2f}", lr=f"{scheduler.get_last_lr()[-1]:.2e}")
            history.append(loss.item())
        return history

    @torch.inference_mode()
    def evaluate_loss(self, loader: DataLoader, device: torch.device) -> float:
        self.eval()
        loss = 0.0
        pbar = tqdm(loader, desc='Evaluating')
        for batch in pbar:
            x, y = map(lambda t: t.to(device), batch)
            logits = self(x)
            B, T, C = logits.shape
            loss += F.cross_entropy(logits.view(B * T, C), y.view(B * T), reduction='sum').item()
        pbar.set_postfix(nll=f"{loss / (len(loader.dataset) * self.context_length):.2f}")
        return loss / (len(loader.dataset) * self.context_length)
    
if __name__ == '__main__':
    from onnxruntime.quantization import quantize_dynamic
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    import matplotlib.pyplot as plt


    device = torch.device('cuda')
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
    torch.backends.cudnn.benchmark = True

    context_length = 32
    embed_dim = 32
    n_head = 1
    n_layer = 1
    batch_size = 1
    accumulate = 268 // batch_size
    lr = 3e-4
    weight_decay = 1e-1

    print("opening_book")
    books = HarryPotter('./dataset/processed_dataset.txt')
    print("creating_vocab")
    vocab = Vocab()
    
    print("creating_train_set")
    train_set = HarryPotterDataset(context_length, books, vocab, train=True)
    val_set = HarryPotterDataset(context_length, books, vocab, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)

    steps = len(train_loader) // accumulate
    model = LLM(vocab, context_length, embed_dim, n_head, n_layer).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim_groups = [
        {'params': [p for p in params if p.dim() >= 2], 'weight_decay': weight_decay},
        {'params': [p for p in params if p.dim() <  2], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optim_groups, lr=lr, betas=(0.9, 0.99))
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1)

    print(f"Statistics")
    print(f"---------------------------")
    print(f"Vocab          {f'{len(vocab):,}':>12}")
    print(f"Tokens         {f'{len(train_set.data):,}':>12s}")
    print(f"---------------------------")
    print(f"Batch Size     {f'{batch_size:,}':>12}")
    print(f"Accumulate     {f'{accumulate:,}':>12}")
    print(f"Context Length {f'{context_length:,}':>12}")
    print(f"---------------------------")
    print(f"Parameters     {f'{model.num_parameters:,}':>12}")
    print(f"Buffers        {f'{model.num_buffers:,}':>12}")
    print(f"Footprint      {f'{(model.num_parameters + model.num_buffers) * 32 * 1.25e-10:.2f} GB':>12}")
    print(f"---------------------------")

    nlls = model.fit(train_loader, optimizer, scheduler, steps, accumulate, device)
    plt.figure(figsize=(8, 4))
    plt.plot(nlls)
    plt.title("Training Loss over Time")
    plt.xlabel("step")
    plt.ylabel("nll")
    plt.savefig("harry_potter.png")

    nll = model.evaluate_loss(val_loader, device)
    print(f"---------------------")
    print(f"Validation NLL {f'{nll:.2f}':>6}")
    print(f"---------------------")

    device = torch.device('cpu')
    model = model.to(device)
    
    torch.save(model.state_dict(), 'harry_potter.pt')
    torch.onnx.export(
        model.cpu(),
        torch.zeros((1, context_length), dtype=torch.long),
        'harry_potter.onnx',
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 1: 'context'}, 'output': {0: 'batch_size', 1: 'context'}},
    )
    from onnxruntime.quantization import quantize_dynamic
    quantize_dynamic('harry_potter.onnx', 'harry_potter.8bit.onnx')