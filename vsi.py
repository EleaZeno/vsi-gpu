# -*- coding: utf-8 -*-
"""
VSI NEXT-STEP: E6 self-legislation arm ON THE HARD CARRY-ADDITION world (PaddlePaddle).
Combines: carry-hard world (real multi-position interaction) + 5-arm self-leg experiment
(SELFLEG_COST / SELFLEG / FIXED1 / FIXED0 / RANDOM) + type-layered unseen-G hold-out
(anti-wireheading) + multi-seed + condition-gated NO-TEMPLATE verdict.
Self-contained. For Baidu AI Studio V100. Auto-uses GPU.
Main question: can the model self-decide supervision density (rho) per round, save
supervision (rho<1), still reach the FIXED1 ceiling on UNSEEN hold-out, beat RANDOM,
and NOT wirehead? On a HARD task (carry chain) adaptivity finally has room to matter.
"""
import os, copy, time
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from collections import deque, defaultdict, Counter

if paddle.device.cuda.device_count() > 0:
    paddle.set_device("gpu"); DEV = "gpu"
else:
    paddle.set_device("cpu"); DEV = "cpu"

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


def make_q(rng, g, rho):
    a = [int(rng.integers(0, B)) for _ in range(g)]
    b = [int(rng.integers(0, B)) for _ in range(g)]
    out, tr = add_with_trace(a, b)
    prompt = ["add"] + [str(d) for d in reversed(a)] + ["plus"] + [str(d) for d in reversed(b)] + ["eq"]
    inter = []
    for (d, c) in tr:
        if rng.random() < rho:
            inter += ["carry", str(c), str(d), "|"]
    final = ["sep"] + [str(d) for d in reversed(out)]
    ans = inter + final
    if len(prompt + ans) + 1 <= SEQ:
        return prompt, ans
    return None


def final_of(ans):
    if "sep" not in ans:
        return "<e>"
    i = ans.index("sep")
    digs = [t for t in ans[i + 1:] if t in stoi and t in DIGITS]
    return "".join(digs) if digs else "<e>"


def encode_pair(prompt, ans):
    seq = prompt + ans
    ids = [stoi[w] for w in seq]
    ids = (ids + [PAD] * (SEQ - len(ids)))[:SEQ]
    mask = [0] * SEQ
    for i in range(len(prompt) - 1, len(seq)):
        if i < SEQ:
            mask[i] = 1
    return ids, mask

class LM(nn.Layer):
    def __init__(self, d=128, nhead=8, nlayer=4):
        super().__init__()
        self.emb = nn.Embedding(Vt_full, d)
        self.pos = self.create_parameter([1, SEQ, d], default_initializer=nn.initializer.Constant(0.0))
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0, activation="relu")
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.head = nn.Linear(d, Vt_full)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos[:, :T]
        m = paddle.triu(paddle.full([T, T], float("-inf")), diagonal=1).unsqueeze([0, 1])
        return self.head(self.enc(h, src_mask=m))


def train_on(model, opt, pairs, steps, bs=256):
    if not pairs:
        return
    model.train()
    enc = [encode_pair(p, a) for p, a in pairs]
    toks = paddle.to_tensor([e[0] for e in enc], dtype="int64")
    masks = paddle.to_tensor([e[1] for e in enc], dtype="float32")
    n = toks.shape[0]
    for _ in range(steps):
        idx = paddle.randint(0, n, [min(bs, n)])
        x = paddle.gather(toks, idx); mk = paddle.gather(masks, idx)[:, :-1]
        logits = model(x[:, :-1]); tgt = x[:, 1:]
        ce = F.cross_entropy(logits.reshape([-1, Vt_full]), tgt.reshape([-1]), reduction="none").reshape(tgt.shape)
        loss = (ce * mk).sum() / paddle.clip(mk.sum(), min=1.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.clear_grad()


@paddle.no_grad()
def solve_batch(model, prompts):
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
        lg = model(paddle.to_tensor(batch))
        for i, c in enumerate(cur):
            if done[i]:
                continue
            nx = int(paddle.argmax(lg[i, len(c) - 1]))
            if nx == PAD:
                done[i] = True; continue
            outs[i].append(itos.get(nx, "?")); c.append(nx)
            if len(c) >= SEQ:
                done[i] = True
        if all(done):
            break
    return outs


def pass_and_baseline(model, rng, g, rho, n=200):
    prompts = []; gold = []
    for _ in range(n):
        qa = make_q(rng, g, rho)
        if not qa:
            continue
        prompts.append(qa[0]); gold.append(final_of(qa[1]))
    if not prompts:
        return 0.0, 0.0
    preds = solve_batch(model, prompts)
    ok = sum(1 for p, gd in zip(preds, gold) if final_of(p) == gd)
    maj = Counter(gold).most_common(1)[0][1] / len(gold)
    return ok / len(gold), maj


def build_unseen_G(dmax, per_d=200, gseed=99999):
    g = np.random.default_rng(gseed); G = {}
    for d in range(1, dmax + 1):
        qs = []
        for _ in range(per_d):
            qa = make_q(g, d, 0.0)
            if qa:
                qs.append(qa)
        G[d] = qs
    return G


def eval_unseen_G(model, G, dmax):
    res = {}
    for d in range(1, dmax + 1):
        qs = G[d]
        prompts = [p for p, a in qs]; gold = [final_of(a) for p, a in qs]
        if not prompts:
            res[d] = (0.0, 0.0, 0.0); continue
        preds = solve_batch(model, prompts)
        ok = sum(1 for p, gd in zip(preds, gold) if final_of(p) == gd)
        tot = len(qs); maj = Counter(gold).most_common(1)[0][1] / tot
        res[d] = (ok / tot, maj, ok / tot - maj)
    return res


def dstar_from(res, dmax):
    ds = 0
    for d in range(1, dmax + 1):
        pr, mj, net = res[d]
        if pr >= 0.6 and net >= 0.20 and ds == d - 1:
            ds = d
        else:
            break
    return ds


def pick_rho(mode, rho_t, front_acc, rng):
    if mode == "FIXED1":
        return 1.0
    if mode == "FIXED0":
        return 0.0
    if mode == "RANDOM":
        return float(rng.uniform(0.0, 1.0))
    if mode == "SELFLEG":
        if front_acc < 0.60:
            return min(1.0, rho_t + 0.20)
        elif front_acc > 0.88:
            return max(0.0, rho_t - 0.15)
        return rho_t
    if mode == "SELFLEG_COST":
        if front_acc < 0.55:
            return min(1.0, rho_t + 0.25)
        elif front_acc > 0.80:
            return max(0.0, rho_t - 0.30)
        return max(0.0, rho_t - 0.05)
    return rho_t

def run_arm(mode, rounds, steps, nq, seed, dmax, lr=2e-3, cost=0.15):
    rng = np.random.default_rng(seed); paddle.seed(seed)
    model = LM()
    sched = paddle.optimizer.lr.CosineAnnealingDecay(lr, T_max=max(rounds, 2), eta_min=lr * 0.15)
    opt = paddle.optimizer.Adam(learning_rate=sched, parameters=model.parameters())
    hist = defaultdict(lambda: deque(maxlen=2))
    G = build_unseen_G(dmax)
    best_score = -1e9; best_state = None
    rho_traj = []; self_traj = []
    rho_t = 0.5 if mode in ("SELFLEG", "SELFLEG_COST") else (1.0 if mode == "FIXED1" else 0.0)
    for r in range(rounds):
        d_front = 1
        for d in range(1, dmax + 1):
            if (hist[d][-1] if len(hist[d]) else 0.0) >= 0.6:
                d_front = min(d + 1, dmax)
            else:
                break
        front_acc = hist[d_front][-1] if len(hist[d_front]) else 0.0
        rho_t = pick_rho(mode, rho_t, front_acc, rng)
        rho_traj.append(round(rho_t, 3))
        w = {d: 0.04 for d in range(1, dmax + 1)}; w[1] += 0.25; w[d_front] += 0.40
        if d_front + 1 <= dmax:
            w[d_front + 1] += 0.18
        tot = sum(w.values()); dl = list(w); ps = np.array([w[d] / tot for d in dl]); ps /= ps.sum()
        pairs = []
        for _ in range(nq):
            d = int(rng.choice(dl, p=ps)); qa = make_q(rng, d, rho_t)
            if qa:
                pairs.append(qa)
        train_on(model, opt, pairs, steps=steps)
        self_acc = []
        for d in range(1, dmax + 1):
            pr, _ = pass_and_baseline(model, rng, d, rho_t, n=40); hist[d].append(pr); self_acc.append(pr)
        self_score = sum(self_acc) / len(self_acc); self_traj.append(round(self_score, 3))
        sel_metric = self_score - (cost * rho_t if mode == "SELFLEG_COST" else 0.0)
        if sel_metric > best_score:
            best_score = sel_metric; best_state = copy.deepcopy(model.state_dict())
        sched.step()
    if best_state is not None:
        model.set_state_dict(best_state)
    final_ug = eval_unseen_G(model, G, dmax)
    deep = sum(final_ug[d][2] for d in range(3, dmax + 1)) / max(dmax - 2, 1)
    return {"dG": dstar_from(final_ug, dmax),
            "self_last": self_traj[-1] if self_traj else 0.0,
            "mean_rho": float(np.mean(rho_traj)) if rho_traj else 0.0,
            "deep_net": deep}


if __name__ == "__main__":
    rounds = 30; steps = 300; nq = 480; seeds = [0, 1, 2]; dmax = 6; cost = 0.15
    arms = ["SELFLEG_COST", "SELFLEG", "FIXED1", "FIXED0", "RANDOM"]
    print("=" * 78)
    print("VSI E6 self-legislation ON HARD CARRY world (Paddle) device=%s paddle=%s" % (DEV, paddle.__version__))
    print("seeds=%s rounds=%d steps=%d nq=%d dmax=%d cost=%s" % (seeds, rounds, steps, nq, dmax, cost))
    print("=" * 78, flush=True)
    agg = {}
    for arm in arms:
        rows = []; t0 = time.time()
        for sd in seeds:
            res = run_arm(arm, rounds, steps, nq, sd, dmax, cost=cost); rows.append(res)
            print("  [%-12s seed%d] unseenG_d*=%d  mean_rho=%.3f  self=%.2f  deep_net=%+.2f"
                  % (arm, sd, res["dG"], res["mean_rho"], res["self_last"], res["deep_net"]), flush=True)
        dGs = [r["dG"] for r in rows]; rhos = [r["mean_rho"] for r in rows]
        selfs = [r["self_last"] for r in rows]; deeps = [r["deep_net"] for r in rows]
        agg[arm] = {"dG_mean": float(np.mean(dGs)), "dG_std": float(np.std(dGs)), "dG_all": dGs,
                    "rho_mean": float(np.mean(rhos)), "self_mean": float(np.mean(selfs)),
                    "deep_mean": float(np.mean(deeps))}
        print("  => %s: d*=%.2f+/-%.2f %s mean_rho=%.3f  (%.1fs)\n"
              % (arm, agg[arm]["dG_mean"], agg[arm]["dG_std"], dGs, agg[arm]["rho_mean"], time.time() - t0), flush=True)
    print("=" * 78)
    print("VERDICT (anti-self-deception, condition-gated, NO template):")
    need = ("SELFLEG_COST", "FIXED1", "FIXED0", "RANDOM")
    if all(k in agg for k in need):
        sc = agg["SELFLEG_COST"]; f1 = agg["FIXED1"]; f0 = agg["FIXED0"]; rd = agg["RANDOM"]
        print("  SELFLEG_COST d*=%.2f  FIXED1=%.2f  FIXED0=%.2f  RANDOM=%.2f"
              % (sc["dG_mean"], f1["dG_mean"], f0["dG_mean"], rd["dG_mean"]))
        print("  SELFLEG_COST mean_rho=%.3f (FIXED1=1.0); self=%.2f deep_net=%+.2f"
              % (sc["rho_mean"], sc["self_mean"], sc["deep_mean"]))
        beats_floor = (sc["dG_mean"] - f0["dG_mean"]) >= 1.0
        beats_random = (sc["dG_mean"] - rd["dG_mean"]) >= 0.5
        is_adaptive = sc["rho_mean"] <= 0.85
        near_ceiling = (f1["dG_mean"] - sc["dG_mean"]) <= 1.0
        wireheading = (sc["self_mean"] > 0.8 and sc["deep_mean"] < 0.15)
        print("  (A) beats FIXED0 floor (+>=1): %s" % beats_floor)
        print("  (B) beats RANDOM adversary (+>=0.5): %s" % beats_random)
        print("  (C) ADAPTIVE not couch (mean_rho<=0.85): %s" % is_adaptive)
        print("  (D) near FIXED1 ceiling (gap<=1): %s" % near_ceiling)
        print("  (E) no wireheading: %s" % (not wireheading))
        if beats_floor and beats_random and is_adaptive and near_ceiling and not wireheading:
            print("  [LOCKED] E6 self-legislation is REAL adaptive value on a HARD task:")
            print("           beats floor+random, saves supervision (rho<1), reaches ceiling, no self-deception.")
        elif beats_floor and near_ceiling and not is_adaptive:
            print("  [TRIVIAL] reaches ceiling but mean_rho~1.0 == just copied FIXED1 (still task too soft for adaptivity).")
        elif wireheading:
            print("  [CAUGHT] self high but unseen-G deep flat = wireheading caught.")
        elif not beats_random:
            print("  [NO-VALUE] cannot beat RANDOM rho -> legislation carries no info.")
        else:
            print("  [PARTIAL/NEG] mixed signals -> see per-criterion flags above.")
    else:
        print("  (arms incomplete)")