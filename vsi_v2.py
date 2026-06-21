# -*- coding: utf-8 -*-
"""
VSI E6 self-legislation v2 ON HARD CARRY world (PaddlePaddle, V100).
UPGRADES vs v1:
  (1) COST SWEEP: SELFLEG arm run at cost in {0.0,0.15,0.3,0.5,0.7} -> rho-vs-cost curve.
  (2) rho-by-difficulty: log rho applied per grade (g1/g3/g6) to expose LOCAL adaptivity
      that the global mean_rho hides.
  (3) Bigger model (d=192,nhead=8,nlayer=6) + rounds=40 (V100 can take it).
  (4) Keep 5-arm baseline (FIXED1/FIXED0/RANDOM) + unseen-G hold-out + condition-gated verdict.
Main question refined: does rho FALL as cost RISES (mechanism works) or stay pinned (mechanism dead)?
And: is there LOCAL adaptivity (save rho on easy g, spend on hard g) masked by global mean?
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
    def __init__(self, d=192, nhead=8, nlayer=6):
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

# ---- per-grade adaptive rho: the model self-legislates a DIFFERENT rho for each grade ----
def pick_rho_pergrade(mode, rho_g, acc_g, cost, rng):
    """rho_g, acc_g are dicts grade->value. Returns updated rho_g dict."""
    new = {}
    for g in rho_g:
        r = rho_g[g]; a = acc_g.get(g, 0.0)
        if mode == "FIXED1":
            new[g] = 1.0
        elif mode == "FIXED0":
            new[g] = 0.0
        elif mode == "RANDOM":
            new[g] = float(rng.uniform(0.0, 1.0))
        elif mode in ("SELFLEG", "SELFLEG_COST"):
            # if already solved this grade, drop rho (save supervision); cost makes drop more aggressive
            if a > 0.85:
                drop = 0.30 + (cost * 0.8 if mode == "SELFLEG_COST" else 0.10)
                new[g] = max(0.0, r - drop)
            elif a < 0.55:
                new[g] = min(1.0, r + 0.25)
            else:
                # cost arm trims even in the middle zone
                trim = (cost * 0.5) if mode == "SELFLEG_COST" else 0.05
                new[g] = max(0.0, r - trim)
        else:
            new[g] = r
    return new


def run_arm(mode, rounds, steps, nq, seed, dmax, lr=2e-3, cost=0.15):
    rng = np.random.default_rng(seed); paddle.seed(seed)
    model = LM()
    sched = paddle.optimizer.lr.CosineAnnealingDecay(lr, T_max=max(rounds, 2), eta_min=lr * 0.15)
    opt = paddle.optimizer.Adam(learning_rate=sched, parameters=model.parameters())
    hist = defaultdict(lambda: deque(maxlen=2))
    G = build_unseen_G(dmax)
    best_score = -1e9; best_state = None
    rho_g = {g: (0.5 if mode in ("SELFLEG", "SELFLEG_COST") else (1.0 if mode == "FIXED1" else 0.0)) for g in range(1, dmax + 1)}
    rho_g_traj = []  # list of dict snapshots
    self_traj = []
    for r in range(rounds):
        acc_g = {g: (hist[g][-1] if len(hist[g]) else 0.0) for g in range(1, dmax + 1)}
        rho_g = pick_rho_pergrade(mode, rho_g, acc_g, cost, rng)
        rho_g_traj.append({g: round(rho_g[g], 3) for g in rho_g})
        # curriculum frontier
        d_front = 1
        for d in range(1, dmax + 1):
            if (hist[d][-1] if len(hist[d]) else 0.0) >= 0.6:
                d_front = min(d + 1, dmax)
            else:
                break
        w = {d: 0.04 for d in range(1, dmax + 1)}; w[1] += 0.25; w[d_front] += 0.40
        if d_front + 1 <= dmax:
            w[d_front + 1] += 0.18
        tot = sum(w.values()); dl = list(w); ps = np.array([w[d] / tot for d in dl]); ps /= ps.sum()
        pairs = []
        for _ in range(nq):
            d = int(rng.choice(dl, p=ps)); qa = make_q(rng, d, rho_g[d])
            if qa:
                pairs.append(qa)
        train_on(model, opt, pairs, steps=steps)
        self_acc = []
        for d in range(1, dmax + 1):
            pr, _ = pass_and_baseline(model, rng, d, rho_g[d], n=40); hist[d].append(pr); self_acc.append(pr)
        self_score = sum(self_acc) / len(self_acc); self_traj.append(round(self_score, 3))
        mean_rho_now = float(np.mean([rho_g[g] for g in rho_g]))
        sel_metric = self_score - (cost * mean_rho_now if mode == "SELFLEG_COST" else 0.0)
        if sel_metric > best_score:
            best_score = sel_metric; best_state = copy.deepcopy(model.state_dict())
        sched.step()
    if best_state is not None:
        model.set_state_dict(best_state)
    final_ug = eval_unseen_G(model, G, dmax)
    deep = sum(final_ug[d][2] for d in range(3, dmax + 1)) / max(dmax - 2, 1)
    # last-third averaged per-grade rho (settled behaviour)
    tail = rho_g_traj[-(rounds // 3):] if rho_g_traj else []
    rho_by_g = {}
    for g in range(1, dmax + 1):
        vals = [snap[g] for snap in tail] if tail else [rho_g[g]]
        rho_by_g[g] = float(np.mean(vals))
    mean_rho = float(np.mean([rho_by_g[g] for g in rho_by_g]))
    return {"dG": dstar_from(final_ug, dmax),
            "self_last": self_traj[-1] if self_traj else 0.0,
            "mean_rho": mean_rho,
            "rho_easy": rho_by_g.get(1, 0.0),
            "rho_hard": rho_by_g.get(dmax, 0.0),
            "rho_by_g": rho_by_g,
            "deep_net": deep}

if __name__ == "__main__":
    rounds = 40; steps = 300; nq = 480; seeds = [0, 1, 2]; dmax = 6
    print("=" * 80)
    print("VSI E6 self-legislation v2 (COST SWEEP + per-grade rho) device=%s paddle=%s" % (DEV, paddle.__version__))
    print("seeds=%s rounds=%d steps=%d nq=%d dmax=%d  model=d192/h8/L6" % (seeds, rounds, steps, nq, dmax))
    print("=" * 80, flush=True)

    # ---- PART 1: cost sweep on SELFLEG_COST ----
    cost_grid = [0.0, 0.15, 0.3, 0.5, 0.7]
    print("\n--- COST SWEEP (SELFLEG_COST arm; does mean_rho FALL as cost RISES?) ---", flush=True)
    sweep = {}
    for c in cost_grid:
        rows = []; t0 = time.time()
        for sd in seeds:
            res = run_arm("SELFLEG_COST", rounds, steps, nq, sd, dmax, cost=c); rows.append(res)
            print("  [cost=%.2f seed%d] d*=%d mean_rho=%.3f rho_easy(g1)=%.2f rho_hard(g%d)=%.2f deep=%+.2f"
                  % (c, sd, res["dG"], res["mean_rho"], res["rho_easy"], dmax, res["rho_hard"], res["deep_net"]), flush=True)
        sweep[c] = {"dG": float(np.mean([r["dG"] for r in rows])),
                    "mean_rho": float(np.mean([r["mean_rho"] for r in rows])),
                    "rho_easy": float(np.mean([r["rho_easy"] for r in rows])),
                    "rho_hard": float(np.mean([r["rho_hard"] for r in rows])),
                    "deep": float(np.mean([r["deep_net"] for r in rows]))}
        print("  => cost=%.2f: d*=%.2f mean_rho=%.3f easy=%.2f hard=%.2f (%.1fs)\n"
              % (c, sweep[c]["dG"], sweep[c]["mean_rho"], sweep[c]["rho_easy"], sweep[c]["rho_hard"], time.time() - t0), flush=True)

    # ---- PART 2: baselines for the verdict (at cost=0.3 reference) ----
    print("\n--- BASELINES (FIXED1 / FIXED0 / RANDOM) ---", flush=True)
    base = {}
    for arm in ["FIXED1", "FIXED0", "RANDOM"]:
        rows = []; t0 = time.time()
        for sd in seeds:
            res = run_arm(arm, rounds, steps, nq, sd, dmax, cost=0.3); rows.append(res)
            print("  [%-7s seed%d] d*=%d mean_rho=%.3f deep=%+.2f"
                  % (arm, sd, res["dG"], res["mean_rho"], res["deep_net"]), flush=True)
        base[arm] = {"dG": float(np.mean([r["dG"] for r in rows])),
                     "mean_rho": float(np.mean([r["mean_rho"] for r in rows])),
                     "deep": float(np.mean([r["deep_net"] for r in rows]))}
        print("  => %s: d*=%.2f mean_rho=%.3f (%.1fs)\n" % (arm, base[arm]["dG"], base[arm]["mean_rho"], time.time() - t0), flush=True)

    # ---- VERDICT ----
    print("=" * 80)
    print("VERDICT v2 (NO template, condition-gated):")
    rhos = [sweep[c]["mean_rho"] for c in cost_grid]
    # Q1: does rho fall as cost rises? (monotone-ish: rho at max cost < rho at zero cost by >=0.15)
    rho_falls = (sweep[cost_grid[0]]["mean_rho"] - sweep[cost_grid[-1]]["mean_rho"]) >= 0.15
    # Spearman sign via simple pairwise
    import itertools
    conc = sum(1 for i, j in itertools.combinations(range(len(cost_grid)), 2) if rhos[j] < rhos[i])
    disc = sum(1 for i, j in itertools.combinations(range(len(cost_grid)), 2) if rhos[j] > rhos[i])
    mono = conc - disc
    # Q2: local adaptivity at some cost (rho_easy < rho_hard by >=0.15)
    local_adapt = any((sweep[c]["rho_hard"] - sweep[c]["rho_easy"]) >= 0.15 for c in cost_grid)
    best_gap_c = max(cost_grid, key=lambda c: sweep[c]["rho_hard"] - sweep[c]["rho_easy"])
    # Q3: capability kept while saving (some cost with d* within 1 of FIXED1 AND mean_rho<=0.85)
    f1 = base["FIXED1"]["dG"]
    saves_keeps = [c for c in cost_grid if (f1 - sweep[c]["dG"]) <= 1.0 and sweep[c]["mean_rho"] <= 0.85]
    print("  cost grid mean_rho: %s" % ["%.0f:%.2f" % (c * 100, sweep[c]["mean_rho"]) for c in cost_grid])
    print("  FIXED1 d*=%.2f FIXED0 d*=%.2f RANDOM d*=%.2f" % (f1, base["FIXED0"]["dG"], base["RANDOM"]["dG"]))
    print("  (Q1) rho FALLS with cost (drop>=0.15): %s  [pairwise mono=%d/%d]" % (rho_falls, mono, len(list(itertools.combinations(range(len(cost_grid)), 2)))))
    print("  (Q2) LOCAL adaptivity exists (rho_hard-rho_easy>=0.15 at some cost): %s  [best at cost=%.2f gap=%.2f]"
          % (local_adapt, best_gap_c, sweep[best_gap_c]["rho_hard"] - sweep[best_gap_c]["rho_easy"]))
    print("  (Q3) saves AND keeps capability (d* within1 of FIXED1 & mean_rho<=0.85): %s  [costs: %s]" % (bool(saves_keeps), saves_keeps))
    if rho_falls and saves_keeps:
        print("  [MECHANISM-WORKS] rho responds to cost AND a cost exists where model saves supervision yet keeps capability.")
        print("                    -> self-legislation is REAL; earlier TRIVIAL was just too-low cost penalty.")
    elif rho_falls and not saves_keeps:
        print("  [TRADEOFF] rho responds to cost, BUT saving supervision costs capability (no free lunch at this scale).")
    elif local_adapt:
        print("  [LOCAL-ONLY] global rho pinned, but model DOES save on easy grades & spend on hard = hidden local adaptivity.")
    else:
        print("  [PINNED] rho stays high regardless of cost & no local structure -> mechanism inert at this scale.")
    print("=" * 80, flush=True)
