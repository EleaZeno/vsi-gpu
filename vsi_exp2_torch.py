# -*- coding: utf-8 -*-
"""
VSI EXPERIMENT 2 (PyTorch, Colab T4 / local 3060): THE CORE TEST of self-evolution.
Question: can a SMALL model, training ONLY on its OWN self-judged data (NO ground-truth
labels), keep getting better on an UNTOUCHABLE hold-out for MANY rounds WITHOUT saturating?
Persistent non-saturating rise = kappa>0 = REAL self-evolution. Flatten = FAKE.

Mechanism: model self-generates carry-add problems; solves each k times with sampling;
MAJORITY VOTE = pseudo-label (NOT real answer); keep high-agreement -> train on them.
Hold-out is NEVER trained/self-labeled on -> only honest yardstick (anti-wireheading).
Arms: SELF (self-eval loop) / ORACLE (true labels, upper) / FROZEN (train r0 only, floor).
Verdict (condition-gated, NO template): kappa>0 iff hold-out net rises persistently
(2nd-half slope>0 AND final>>round-3) AND beats FROZEN floor.
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

DEV = "cuda" if torch.cuda.is_available() else "cpu"

B = 10; SEQ = 64
DIGITS = [str(i) for i in range(B)]
WORDS = ["add", "plus", "eq", "carry", "|", "sep"] + DIGITS
stoi = {w: i for i, w in enumerate(WORDS)}
itos = {i: w for w, i in stoi.items()}
Vt = len(WORDS); PAD = Vt; Vt_full = Vt + 1


def add_with_trace(a, b):
    n = max(len(a), len(b))
    a = a + [0] * (n - len(a)); b = b + [0] * (n - len(b))
    carry = 0; out = []; tr = []
    for i in range(n):
        s = a[i] + b[i] + carry
        d = s % B; carry = s // B
        out.append(d); tr.append((d, carry))
    if carry:
        out.append(carry); tr.append((carry, 0))
    return out, tr


def true_sum_str(a, b):
    out, _ = add_with_trace(a, b)
    return "".join(str(d) for d in reversed(out))


def gen_problem(rng, g):
    a = [int(rng.integers(0, B)) for _ in range(g)]
    b = [int(rng.integers(0, B)) for _ in range(g)]
    prompt = ["add"] + [str(d) for d in reversed(a)] + ["plus"] + [str(d) for d in reversed(b)] + ["eq"]
    return a, b, prompt


def final_of(ans):
    if "sep" not in ans:
        return "<e>"
    i = ans.index("sep")
    digs = [t for t in ans[i + 1:] if t in stoi and t in DIGITS]
    return "".join(digs) if digs else "<e>"


def build_answer(a, b, rho, rng):
    out, tr = add_with_trace(a, b)
    inter = []
    for (d, c) in tr:
        if rng.random() < rho:
            inter += ["carry", str(c), str(d), "|"]
    return inter + ["sep"] + [str(d) for d in reversed(out)]


def make_pseudo_answer(final_str):
    return ["sep"] + list(final_str)


def encode_pair(prompt, ans):
    seq = prompt + ans
    ids = [stoi[w] for w in seq if w in stoi]
    ids = (ids + [PAD] * (SEQ - len(ids)))[:SEQ]
    mask = [0] * SEQ
    for i in range(len(prompt) - 1, len(prompt) + len(ans)):
        if i < SEQ:
            mask[i] = 1
    return ids, mask

class LM(nn.Module):
    def __init__(self, d=192, nhead=8, nlayer=6):
        super().__init__()
        self.emb = nn.Embedding(Vt_full, d)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, d))
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0, activation="relu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.head = nn.Linear(d, Vt_full)

    def forward(self, x):
        T = x.size(1)
        h = self.emb(x) + self.pos[:, :T]
        m = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        return self.head(self.enc(h, mask=m))


def train_on(model, opt, pairs, steps, bs=256):
    if not pairs:
        return
    model.train()
    enc = [encode_pair(p, a) for p, a in pairs]
    toks = torch.tensor([e[0] for e in enc], dtype=torch.long, device=DEV)
    masks = torch.tensor([e[1] for e in enc], dtype=torch.float32, device=DEV)
    n = toks.size(0)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), device=DEV)
        x = toks[idx]; mk = masks[idx][:, :-1]
        logits = model(x[:, :-1]); tgt = x[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, Vt_full), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        loss = (ce * mk).sum() / mk.sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def _decode(model, prompts, temp=0.0):
    model.eval()
    cur = [[stoi[w] for w in p] for p in prompts]
    done = [False] * len(cur); outs = [[] for _ in cur]
    for _ in range(SEQ - 1):
        L = max(len(c) for c in cur)
        if L >= SEQ:
            break
        batch = np.full((len(cur), L), PAD, dtype="int64")
        for i, c in enumerate(cur):
            cc = c[-SEQ:]; batch[i, :len(cc)] = cc
        lg = model(torch.tensor(batch, device=DEV))
        for i, c in enumerate(cur):
            if done[i]:
                continue
            logit = lg[i, len(c) - 1]
            if temp <= 0.0:
                nx = int(torch.argmax(logit))
            else:
                p = F.softmax(logit / temp, dim=-1).float().cpu().numpy()
                nx = int(np.random.choice(len(p), p=p / p.sum()))
            if nx == PAD:
                done[i] = True; continue
            outs[i].append(itos.get(nx, "?")); c.append(nx)
            if len(c) >= SEQ:
                done[i] = True
        if all(done):
            break
    return outs


def greedy_finals(model, prompts):
    return [final_of(o) for o in _decode(model, prompts, temp=0.0)]

def self_label(model, prompts, k=5, temp=0.7, agree_thresh=0.6):
    votes = [Counter() for _ in prompts]
    for _ in range(k):
        finals = [final_of(o) for o in _decode(model, prompts, temp=temp)]
        for i, f in enumerate(finals):
            if f != "<e>":
                votes[i][f] += 1
    kept = []
    for i, p in enumerate(prompts):
        if not votes[i]:
            continue
        lab, cnt = votes[i].most_common(1)[0]
        agree = cnt / k
        if agree >= agree_thresh:
            kept.append((p, lab, agree))
    return kept


def build_holdout(grades, per_g=300, gseed=777777):
    g = np.random.default_rng(gseed); H = {}
    for gg in grades:
        items = []
        for _ in range(per_g):
            a, b, prompt = gen_problem(g, gg)
            items.append((prompt, true_sum_str(a, b)))
        H[gg] = items
    return H


def eval_holdout(model, H, grades):
    res = {}
    for gg in grades:
        items = H[gg]
        prompts = [p for p, t in items]; gold = [t for p, t in items]
        preds = greedy_finals(model, prompts)
        ok = sum(1 for pr, gd in zip(preds, gold) if pr == gd)
        tot = len(items)
        maj = Counter(gold).most_common(1)[0][1] / tot
        res[gg] = (ok / tot, maj, ok / tot - maj)
    return res


def holdout_score(res, grades):
    nets = [max(0.0, res[g][2]) for g in grades]
    return float(np.mean(nets))


def dstar(res, grades):
    ds = 0
    for i, g in enumerate(grades):
        acc, mj, net = res[g]
        if acc >= 0.6 and net >= 0.20 and ds == i:
            ds = i + 1
        else:
            break
    return ds


def run_arm(mode, rounds, seed, grades, nq=600, steps=200, lr=2e-3, k=5):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    model = LM().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(rounds, 2), eta_min=lr * 0.2)
    H = build_holdout(grades)
    traj = []; dtraj = []; kept_frac = []
    seed_pairs = []
    for _ in range(400):
        a, b, prompt = gen_problem(rng, 1)
        seed_pairs.append((prompt, build_answer(a, b, 1.0, rng)))
    train_on(model, opt, seed_pairs, steps=120)

    for r in range(rounds):
        cur = eval_holdout(model, H, grades)
        front = grades[0]
        for g in grades:
            if cur[g][0] >= 0.6:
                front = g
        gen_grades = [g for g in grades if g <= min(front + 1, grades[-1])]

        if mode == "FROZEN" and r >= 1:
            pass
        else:
            prompts = []; meta = []
            for _ in range(nq):
                gg = int(rng.choice(gen_grades))
                a, b, prompt = gen_problem(rng, gg)
                prompts.append(prompt); meta.append((a, b))
            if mode == "ORACLE":
                pairs = [(p, build_answer(a, b, 1.0, rng)) for p, (a, b) in zip(prompts, meta)]
                kf = 1.0
            else:
                kept = self_label(model, prompts, k=k, temp=0.7, agree_thresh=0.6)
                pairs = [(p, make_pseudo_answer(lab)) for (p, lab, ag) in kept]
                kf = len(kept) / max(len(prompts), 1)
            train_on(model, opt, pairs, steps=steps)
            kept_frac.append(round(kf, 3))
            sched.step()

        res = eval_holdout(model, H, grades)
        traj.append(round(holdout_score(res, grades), 4))
        dtraj.append(dstar(res, grades))
    final = eval_holdout(model, H, grades)
    return {"traj": traj, "dtraj": dtraj, "kept_frac": kept_frac,
            "final_dstar": dstar(final, grades),
            "final_net_by_g": {g: round(final[g][2], 3) for g in grades}}


def slope(ys):
    n = len(ys)
    if n < 2:
        return 0.0
    xs = np.arange(n)
    return float(np.polyfit(xs, ys, 1)[0])

if __name__ == "__main__":
    rounds = 24; seeds = [0, 1]; grades = [1, 2, 3, 4, 5, 6]
    arms = ["SELF", "ORACLE", "FROZEN"]
    print("=" * 80)
    print("VSI EXPERIMENT 2 (PyTorch): self-evolution non-saturation test  device=%s" % DEV)
    print("rounds=%d seeds=%s grades=%s model=d192/h8/L6" % (rounds, seeds, grades))
    print("SELF=majority-vote self-labels (NO ground truth) | ORACLE=true labels | FROZEN=train r0 only")
    print("=" * 80, flush=True)

    agg = {}
    for arm in arms:
        trajs = []; dstars = []; t0 = time.time()
        for sd in seeds:
            res = run_arm(arm, rounds, sd, grades)
            trajs.append(res["traj"]); dstars.append(res["final_dstar"])
            print("  [%-6s seed%d] final_d*=%d  net_by_g=%s" % (arm, sd, res["final_dstar"], res["final_net_by_g"]), flush=True)
            print("            holdout_traj=%s" % res["traj"], flush=True)
            if res["kept_frac"]:
                print("            self_kept_frac=%s" % res["kept_frac"], flush=True)
        mt = np.mean(np.array(trajs), axis=0)
        agg[arm] = {"mean_traj": [round(x, 4) for x in mt.tolist()], "dstar_mean": float(np.mean(dstars))}
        print("  => %s mean_traj=%s d*=%.2f (%.1fs)\n" % (arm, agg[arm]["mean_traj"], agg[arm]["dstar_mean"], time.time() - t0), flush=True)

    print("=" * 80)
    print("VERDICT (kappa test, condition-gated, NO template):")
    st = agg["SELF"]["mean_traj"]; fr = agg["FROZEN"]["mean_traj"]; orc = agg["ORACLE"]["mean_traj"]
    half = len(st) // 2
    second_half_slope = slope(st[half:])
    full_slope = slope(st)
    rise_vs_r3 = st[-1] - (st[2] if len(st) > 2 else st[0])
    beats_frozen = (st[-1] - fr[-1]) >= 0.05
    persistent = (second_half_slope > 0.002) and (rise_vs_r3 >= 0.05)
    oracle_gap = orc[-1] - st[-1]
    print("  SELF   final=%.3f  full_slope=%+.4f  2nd_half_slope=%+.4f  rise_since_r3=%+.3f" % (st[-1], full_slope, second_half_slope, rise_vs_r3))
    print("  FROZEN final=%.3f   ORACLE final=%.3f (gap SELF->ORACLE=%.3f)" % (fr[-1], orc[-1], oracle_gap))
    print("  SELF d*=%.2f  ORACLE d*=%.2f  FROZEN d*=%.2f" % (agg["SELF"]["dstar_mean"], agg["ORACLE"]["dstar_mean"], agg["FROZEN"]["dstar_mean"]))
    print("  (1) beats FROZEN floor (>=0.05): %s" % beats_frozen)
    print("  (2) PERSISTENT non-saturating rise (2nd-half slope>0 & rise>=0.05): %s" % persistent)
    if persistent and beats_frozen:
        print("  [KAPPA>0] SELF-EVOLUTION REAL: self-labeled loop keeps improving on untouchable hold-out, no saturation.")
        print("            Existence proof of a small self-evolving model on this task.")
    elif beats_frozen and not persistent:
        print("  [SATURATING] self-loop improves then FLATTENS (kappa->0). Better but not unbounded self-evolution.")
    elif not beats_frozen:
        print("  [NO-LIFT] self-loop no better than frozen -> self-labels carry no usable signal here.")
    else:
        print("  [MIXED] see per-criterion flags above.")
    print("=" * 80, flush=True)
