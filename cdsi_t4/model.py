"""CDSI T4 — 微型 Transformer LM。压缩率 = 2^(-bits/token)，loss(以2为底) 即压缩长度。"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d=256, h=8, layers=6, ctx=256, drop=0.1):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(d, h, drop) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), 1)
        for b in self.blocks:
            x = b(x, mask)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        # 交叉熵(自然对数) -> 下游换算 bits/token
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def bits_per_token(model, data, device, ctx, batch=16, max_batches=64):
    """核心度量：在给定数据上的平均 bits/token = 压缩后长度。越低=压得越短=越理解。"""
    model.eval()
    nats, ntok, nb = 0.0, 0, 0
    i = 0
    while i + ctx + 1 <= len(data) and nb < max_batches:
        xs, ys = [], []
        for _ in range(batch):
            if i + ctx + 1 > len(data):
                break
            xs.append(data[i:i + ctx])
            ys.append(data[i + 1:i + ctx + 1])
            i += ctx
        if not xs:
            break
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        logits, _ = model(x)
        ll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
        nats += ll.item()
        ntok += y.numel()
        nb += 1
    model.train()
    bpt = (nats / max(ntok, 1)) / math.log(2)  # nats -> bits
    return bpt  # 压缩率 = 2 ** (-bpt)
