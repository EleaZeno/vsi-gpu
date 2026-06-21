# -*- coding: utf-8 -*-
"""
VSI EXPERIMENT 4 (PyTorch): STV-BOOTSTRAP across the TWO-SIDED CLIFF.
Convergence of all prior experiments (vN-threshold -> carry-PHASE -> m13-TRIVIAL ->
exp2-POISON -> exp3-CLIFF). exp3 nailed it: kappa>0 needs pseudo-label
  kept(yield) x purity(correctness) BOTH above a floor.
Tuning verifier strictness only slides along the cliff (strict->starve, loose->poison).
STV (2605.30290) escapes by showing the verifier a REFERENCE SOLUTION -> purity high AND kept kept.
But STV needs the TRUE reference (=semi-supervised). 

THE ONE QUESTION (what this experiment exists to answer):
  Can we BOOTSTRAP a pseudo-reference (multi-path self-consistency + inverse-check, NEVER
  reading the true answer) to push kept x purity BOTH over the floor -> kappa NEG->POS,
  walking the narrow path between starvation and poisoning, with REAL hold-out gains
  (no wireheading)? Win = beat STV's dependence on true refs. Lose = clean confirmation of
  2601.05280 (no external grounding -> degeneration).

Arms:
  RAW        : majority vote, no verifier            -> poison floor (=exp3)
  VERIFY     : perfect inverse-check                 -> starvation control (=exp3)
  BOOTSTRAP* : multi-path self-consistency pseudo-ref -> inverse-check gate (NO true answer)
  STV_CHEAT  : verifier trained against TRUE ref      -> STV-style UPPER bound (peeks, only to gate)
  ORACLE     : true labels                            -> absolute ceiling
  FROZEN     : freeze after r0                         -> floor

Honest guards (inherit exp3): hold-out gseed 777777 NEVER touched; log purity x kept BOTH curves;
BOOTSTRAP must beat RAW (no poison) AND VERIFY (no starve) to count. Verdict condition-gated, no template.
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
    n = max(len(a), len(b)); a = a + [0]*(n-len(a)); b = b + [0]*(n-len(b))
    carry = 0; out = []; tr = []
    for i in range(n):
        s = a[i] + b[i] + carry; d = s % B; carry = s // B
        out.append(d); tr.append((d, carry))
    if carry: out.append(carry); tr.append((carry, 0))
    return out, tr

def true_sum_str(a, b):
    out, _ = add_with_trace(a, b); return "".join(str(d) for d in reversed(out))

def gen_problem(rng, g):
    a = [int(rng.integers(0, B)) for _ in range(g)]; b = [int(rng.integers(0, B)) for _ in range(g)]
    prompt = ["add"] + [str(d) for d in reversed(a)] + ["plus"] + [str(d) for d in reversed(b)] + ["eq"]
    return a, b, prompt

def str_to_intlist_le(s):
    if not s or any(c not in "0123456789" for c in s): return None
    return [int(c) for c in reversed(s)]

def verify_label(a, b, final_str):
    """INTERNAL verifier: claimed_sum - a == b ? Uses inverse-check only, never true label."""
    digs = str_to_intlist_le(final_str)
    if digs is None: return False
    val = sum(d*(B**i) for i, d in enumerate(digs))
    va = sum(d*(B**i) for i, d in enumerate(a)); vb = sum(d*(B**i) for i, d in enumerate(b))
    return (val - va) == vb

def final_of(ans):
    if "sep" not in ans: return "<e>"
    i = ans.index("sep"); digs = [t for t in ans[i+1:] if t in stoi and t in DIGITS]
    return "".join(digs) if digs else "<e>"

def build_answer(a, b, rho, rng):
    out, tr = add_with_trace(a, b); inter = []
    for (d, c) in tr:
        if rng.random() < rho: inter += ["carry", str(c), str(d), "|"]
    return inter + ["sep"] + [str(d) for d in reversed(out)]

def make_pseudo_answer(final_str):
    return ["sep"] + list(final_str)

def encode_pair(prompt, ans):
    seq = prompt + ans; ids = [stoi[w] for w in seq if w in stoi]
    ids = (ids + [PAD]*(SEQ-len(ids)))[:SEQ]; mask = [0]*SEQ
    for i in range(len(prompt)-1, len(prompt)+len(ans)):
        if i < SEQ: mask[i] = 1
    return ids, mask

class LM(nn.Module):
    def __init__(self, d=192, nhead=8, nlayer=6):
        super().__init__()
        self.emb = nn.Embedding(Vt_full, d); self.pos = nn.Parameter(torch.zeros(1, SEQ, d))
        layer = nn.TransformerEncoderLayer(d, nhead, 4*d, dropout=0.0, activation="relu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayer); self.head = nn.Linear(d, Vt_full)
    def forward(self, x):
        T = x.size(1); h = self.emb(x) + self.pos[:, :T]
        m = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        return self.head(self.enc(h, mask=m))

def train_on(model, opt, pairs, steps, bs=256):
    if not pairs: return
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
        loss = (ce*mk).sum()/mk.sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

@torch.no_grad()
def _decode(model, prompts, temp=0.0):
    model.eval()
    cur = [[stoi[w] for w in p] for p in prompts]
    done = [False]*len(cur); outs = [[] for _ in cur]
    for _ in range(SEQ-1):
        L = max(len(c) for c in cur)
        if L >= SEQ: break
        batch = np.full((len(cur), L), PAD, dtype="int64")
        for i, c in enumerate(cur):
            cc = c[-SEQ:]; batch[i, :len(cc)] = cc
        lg = model(torch.tensor(batch, device=DEV))
        pos = torch.tensor([len(c)-1 for c in cur], device=DEV)
        step_logits = lg[torch.arange(len(cur), device=DEV), pos]
        if temp <= 0.0: nxs = torch.argmax(step_logits, dim=-1)
        else:
            probs = F.softmax(step_logits/temp, dim=-1); nxs = torch.multinomial(probs, 1).squeeze(-1)
        nxs = nxs.cpu().tolist()
        for i, c in enumerate(cur):
            if done[i]: continue
            nx = nxs[i]
            if nx == PAD: done[i] = True; continue
            outs[i].append(itos.get(nx, "?")); c.append(nx)
            if len(c) >= SEQ: done[i] = True
        if all(done): break
    return outs

def greedy_finals(model, prompts):
    return [final_of(o) for o in _decode(model, prompts, temp=0.0)]

# ---------- core: bootstrap self-consistency pseudo-reference (the novel module) ----------
def self_label_bootstrap(model, prompts, metas, mode, rng, k=5, temp=0.7, agree=0.6,
                         n_paths=3, path_temps=(0.5, 0.8, 1.1), consensus=2):
    """Pseudo-labels with an escalating information source per mode:
      RAW       : majority vote over k samples, accept all (poison control)
      VERIFY    : majority vote, accept iff perfect inverse-check passes (starve control)
      BOOTSTRAP : MULTI-PATH self-consistency -> a label is promoted to 'pseudo-reference'
                  only if >=consensus independent decode paths (different temps) AGREE on it
                  AND it passes inverse-check. NO true answer is ever read. This manufactures
                  STV's information-asymmetry from multi-view self-agreement, not ground truth.
      STV_CHEAT : accept iff label == true_sum (peeks at true ref to GATE only) -> STV upper bound.
    Returns (pairs, kept_frac, purity). purity = frac accepted that are actually correct.
    """
    n = len(prompts)
    # base k-sample majority vote (batched single decode, k-fold)
    big = prompts * k
    finals_big = [final_of(o) for o in _decode(model, big, temp=temp)]
    votes = [Counter() for _ in prompts]
    for j in range(k):
        for i in range(n):
            f = finals_big[j*n + i]
            if f != "<e>": votes[i][f] += 1

    # for BOOTSTRAP: extra independent decode paths at different temps, collect per-prompt path-finals
    path_finals = None
    if mode == "BOOTSTRAP":
        path_finals = [[] for _ in prompts]
        for t in path_temps[:n_paths]:
            fb = [final_of(o) for o in _decode(model, prompts, temp=t)]
            for i in range(n):
                if fb[i] != "<e>": path_finals[i].append(fb[i])

    pairs = []; n_acc = 0; n_correct = 0
    for i, p in enumerate(prompts):
        if not votes[i]:
            continue
        lab, cnt = votes[i].most_common(1)[0]
        if cnt / k < agree:
            continue
        a, b = metas[i]
        if mode == "RAW":
            accept = True
        elif mode == "VERIFY":
            accept = verify_label(a, b, lab)
        elif mode == "STV_CHEAT":
            accept = (lab == true_sum_str(a, b))  # peeks at TRUE ref to gate (STV upper bound)
        else:  # BOOTSTRAP
            # promote to pseudo-reference iff multi-path consensus AND inverse-check both hold
            agree_cnt = sum(1 for f in path_finals[i] if f == lab)
            accept = (agree_cnt >= consensus) and verify_label(a, b, lab)
        if accept:
            pairs.append((p, make_pseudo_answer(lab))); n_acc += 1
            if lab == true_sum_str(a, b): n_correct += 1
    kf = n_acc / max(len(prompts), 1)
    purity = (n_correct / n_acc) if n_acc else 0.0
    return pairs, kf, purity


def build_holdout(grades, per_g=300, gseed=777777):
    g = np.random.default_rng(gseed); H = {}
    for gg in grades:
        items = []
        for _ in range(per_g):
            a, b, prompt = gen_problem(g, gg); items.append((prompt, true_sum_str(a, b)))
        H[gg] = items
    return H

def eval_holdout(model, H, grades):
    res = {}
    for gg in grades:
        items = H[gg]; prompts = [p for p, t in items]; gold = [t for p, t in items]
        preds = greedy_finals(model, prompts)
        ok = sum(1 for pr, gd in zip(preds, gold) if pr == gd); tot = len(items)
        maj = Counter(gold).most_common(1)[0][1] / tot
        res[gg] = (ok/tot, maj, ok/tot - maj)
    return res

def holdout_score(res, grades):
    return float(np.mean([max(0.0, res[g][2]) for g in grades]))

def dstar(res, grades):
    ds = 0
    for i, g in enumerate(grades):
        acc, mj, net = res[g]
        if acc >= 0.6 and net >= 0.20 and ds == i: ds = i + 1
        else: break
    return ds

def slope(ys):
    if len(ys) < 2: return 0.0
    return float(np.polyfit(np.arange(len(ys)), ys, 1)[0])


def run_arm(mode, rounds, seed, grades, nq=600, steps=200, lr=2e-3, k=5):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    model = LM().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(rounds, 2), eta_min=lr*0.2)
    H = build_holdout(grades)
    traj = []; kept = []; purity = []
    seed_pairs = []
    for _ in range(400):
        a, b, prompt = gen_problem(rng, 1); seed_pairs.append((prompt, build_answer(a, b, 1.0, rng)))
    train_on(model, opt, seed_pairs, steps=120)

    for r in range(rounds):
        cur = eval_holdout(model, H, grades); front = grades[0]
        for g in grades:
            if cur[g][0] >= 0.6: front = g
        gen_grades = [g for g in grades if g <= min(front + 1, grades[-1])]
        if mode == "FROZEN" and r >= 1:
            pass
        else:
            prompts = []; metas = []
            for _ in range(nq):
                gg = int(rng.choice(gen_grades)); a, b, prompt = gen_problem(rng, gg)
                prompts.append(prompt); metas.append((a, b))
            if mode == "ORACLE":
                pairs = [(p, build_answer(a, b, 1.0, rng)) for p, (a, b) in zip(prompts, metas)]
                kf, pur = 1.0, 1.0
            else:
                pairs, kf, pur = self_label_bootstrap(model, prompts, metas, mode, rng, k=k)
            train_on(model, opt, pairs, steps=steps)
            kept.append(round(kf, 3)); purity.append(round(pur, 3)); sched.step()
        res = eval_holdout(model, H, grades); traj.append(round(holdout_score(res, grades), 4))
    final = eval_holdout(model, H, grades)
    return {"traj": traj, "kept": kept, "purity": purity,
            "final_dstar": dstar(final, grades),
            "net_by_g": {g: round(final[g][2], 3) for g in grades}}


def kappa_sign(traj):
    half = len(traj)//2; sh = slope(traj[half:]); rise = traj[-1] - (traj[2] if len(traj) > 2 else traj[0])
    if sh > 0.002 and rise >= 0.05: return "kappa>0", sh, rise
    if traj[-1] <= traj[2] - 0.03: return "kappa<0", sh, rise
    return "kappa~0", sh, rise

if __name__ == "__main__":
    rounds = 24; seeds = [0, 1]; grades = [1, 2, 3, 4, 5, 6]
    ARMS = ["RAW", "VERIFY", "BOOTSTRAP", "STV_CHEAT", "ORACLE", "FROZEN"]
    print("=" * 88)
    print("VSI EXP4: STV-BOOTSTRAP across the two-sided cliff  device=%s  rounds=%d seeds=%s" % (DEV, rounds, seeds))
    print("BOOTSTRAP = multi-path self-consistency pseudo-ref + inverse-check, NEVER reads true answer.")
    print("Win = BOOTSTRAP beats RAW(no poison) AND VERIFY(no starve) -> kappa>0 on untouchable hold-out.")
    print("=" * 88, flush=True)

    S = {}
    for mode in ARMS:
        trajs = []; purs = []; kepts = []; ds = []; t0 = time.time()
        for sd in seeds:
            res = run_arm(mode, rounds, sd, grades)
            trajs.append(res["traj"]); ds.append(res["final_dstar"])
            if res["purity"]:
                purs.append(np.mean(res["purity"])); kepts.append(np.mean(res["kept"]))
            ks, sh, rise = kappa_sign(res["traj"])
            print("  [%-10s s%d] %-8s d*=%d final=%.3f 2hSlope=%+.4f rise=%+.3f" % (mode, sd, ks, res["final_dstar"], res["traj"][-1], sh, rise), flush=True)
            print("       traj=%s" % res["traj"], flush=True)
            if res["purity"]:
                print("       purity=%s kept=%s" % (res["purity"], res["kept"]), flush=True)
        mt = np.mean(np.array(trajs), axis=0).round(4).tolist()
        mks, msh, mrise = kappa_sign(mt)
        mp = round(float(np.mean(purs)), 3) if purs else None
        mk = round(float(np.mean(kepts)), 3) if kepts else None
        S[mode] = {"traj": mt, "kappa": mks, "purity": mp, "kept": mk, "dstar": float(np.mean(ds)), "final": mt[-1]}
        print("  => %-10s MEAN %-8s final=%.3f purity=%s kept=%s d*=%.2f (%.0fs)\n" % (mode, mks, mt[-1], mp, mk, float(np.mean(ds)), time.time() - t0), flush=True)

    print("=" * 88)
    print("TWO-SIDED CLIFF MAP (final / purity / kept / d* / kappa):")
    print("  %-11s %-8s %-7s %-7s %-6s %-9s" % ("arm", "final", "purity", "kept", "d*", "kappa"))
    for n in ["RAW", "VERIFY", "BOOTSTRAP", "STV_CHEAT", "ORACLE", "FROZEN"]:
        s = S[n]
        print("  %-11s %-8.3f %-7s %-7s %-6.2f %-9s" % (n, s["final"], s["purity"], s["kept"], s["dstar"], s["kappa"]))
    print("-" * 88)

    bs = S["BOOTSTRAP"]; raw = S["RAW"]; ver = S["VERIFY"]; stv = S["STV_CHEAT"]
    beats_raw = bs["final"] > raw["final"] + 0.03          # not poisoned
    beats_ver = bs["final"] > ver["final"] + 0.03          # not starved
    kpos = bs["kappa"] == "kappa>0"
    near_stv = (stv["final"] > 0) and (bs["final"] >= 0.7 * stv["final"])
    if kpos and beats_raw and beats_ver:
        if near_stv:
            print("  [BOOTSTRAP WINS] self-consistency pseudo-ref hits kappa>0 AND approaches STV upper bound")
            print("                   WITHOUT reading true answers -> escapes STV semi-supervised dependence.")
        else:
            print("  [NARROW PATH CONFIRMED] BOOTSTRAP survives the cliff (kappa>0, beats both poison & starve)")
            print("                   but trails STV_CHEAT -> partial escape; pseudo-ref weaker than true ref.")
    elif beats_raw and beats_ver and not kpos:
        print("  [EDGE OF PATH] BOOTSTRAP avoids both cliffs (beats poison & starve) but kappa not clearly >0;")
        print("                 sits on the saddle -- needs annealing/curriculum to climb. Inconclusive lift.")
    else:
        print("  [WINDOW CLOSED] BOOTSTRAP still falls into poison or starvation -> clean confirmation of")
        print("                  2601.05280: no external grounding -> degeneration. Pseudo-ref insufficient here.")
    print("=" * 88, flush=True)
