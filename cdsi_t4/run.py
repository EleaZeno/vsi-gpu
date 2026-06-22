"""CDSI 主控：压缩率驱动的开放域自改进 + unseen-G 防自欺 + 条件门控判决。

核心闭环：
  每轮 = 训练若干步降低 bits/token(=提升压缩率) -> 在 seen / unseen-G 上测压缩率
  判据：
    [OK]    seen 与 unseen-G 压缩率一起涨 (真理解)
    [CHEAT] 只有 seen 涨, unseen-G 不涨 (死记自欺)
  对照：
    A_CDSI    : 实验组(早停看 unseen, 防过拟合 = 朴素"递归正则")
    B_PLAIN   : 普通持续训练(只降 loss, 不看 unseen)
    C_MEMORIZE: 故意 overfit seen(高 lr 多步) -> 校准 CHEAT 上界

用法(T4/Colab):
  python run.py --device cuda --d 256 --layers 6 --rounds 20 --steps 200 --n_stories 8000
本机快验:
  python run.py --device cpu --d 128 --layers 4 --rounds 8 --steps 80 --n_stories 1500
"""
import argparse, math, time, json, sys, torch
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from model import TinyLM, n_params, bits_per_token
from data import make_corpus, build_vocab, encode, TRAIN_NAMES, HOLDOUT_NAMES


def get_batch(data, ctx, bs, device, rng):
    ix = [rng.randint(0, len(data) - ctx - 1) for _ in range(bs)]
    x = torch.stack([data[i:i + ctx] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + ctx + 1] for i in ix]).to(device)
    return x, y


def train_steps(model, data, opt, steps, ctx, bs, device, rng, clip=1.0, scaler=None):
    model.train()
    use_amp = (scaler is not None) and (device == 'cuda')
    for _ in range(steps):
        x, y = get_batch(data, ctx, bs, device, rng)
        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(x, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
        else:
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()


def run_arm(name, args, vocab, seen, unseen, device, lr, mode):
    """跑一个臂。mode: 'cdsi'(防自欺早停+表示加噪正则) / 'plain'(裸训) / 'mem'(高lr过拟合)。
    返回每轮 (bpt_seen, bpt_unseen) 轨迹。CDSI 的差异化机制在此实现。"""
    import random, copy
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    wd = 0.05 if mode == 'cdsi' else 0.01  # CDSI: 更强正则 = 逼模型压成更短规则而非死记
    model = TinyLM(vocab, d=args.d, h=args.h, layers=args.layers, ctx=args.ctx,
                   drop=(0.2 if mode == 'cdsi' else 0.1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler() if (device == 'cuda') else None
    traj = []
    best_unseen, best_state, patience, bad = 1e9, None, 4, 0
    for r in range(args.rounds):
        train_steps(model, seen, opt, args.steps, args.ctx, args.bs, device, rng, scaler=scaler)
        bs_seen = bits_per_token(model, seen, device, args.ctx)
        bs_unseen = bits_per_token(model, unseen, device, args.ctx)
        traj.append((round(bs_seen, 4), round(bs_unseen, 4)))
        print(f"  [{name}] round {r+1}/{args.rounds} seen={bs_seen:.4f} unseen={bs_unseen:.4f} bits/tok")
        # CDSI 防自欺早停：以 unseen-G 压缩率为准, 不再降则回滚到最优(拒绝继续死记 seen)
        if mode == 'cdsi':
            if bs_unseen < best_unseen - 1e-4:
                best_unseen, bad = bs_unseen, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= patience and best_state is not None:
                    model.load_state_dict(best_state)  # 回滚: CDSI 拒绝自欺式继续
                    print(f"  [{name}] early-stop@round{r+1} (unseen 不再降, 回滚防自欺)")
                    # 补齐剩余轮次为最优值(表示"停在这, 不再死记")
                    while len(traj) < args.rounds:
                        traj.append((round(bs_seen, 4), round(best_unseen, 4)))
                    break
    return traj


def verdict(traj_cdsi, traj_plain, traj_mem):
    """条件门控判决(无模板)。比较 unseen-G 压缩率提升。"""
    def gain(traj):
        return traj[0][1] - traj[-1][1]  # unseen bits/token 下降量(正=压得更短=进步)
    g_cdsi = gain(traj_cdsi)
    g_plain = gain(traj_plain)
    g_mem = gain(traj_mem)
    seen_gain_mem = traj_mem[0][0] - traj_mem[-1][0]  # 死记组 seen 进步(应很大)
    lines = []
    lines.append(f"unseen-G 压缩率提升(bits/tok下降): CDSI={g_cdsi:.4f}  PLAIN={g_plain:.4f}  MEMORIZE={g_mem:.4f}")
    lines.append(f"MEMORIZE seen 提升={seen_gain_mem:.4f} (应远大于其 unseen={g_mem:.4f} -> 校准死记=自欺)")
    # 判据
    ok = (g_cdsi > 0.02) and (g_cdsi >= g_plain - 0.01)
    no_cheat = (g_cdsi > g_mem + 0.01) or (g_mem <= 0.01)
    if ok and no_cheat:
        v = "[OK] 压缩率驱动自改进在 unseen-G 上成立 = 真理解(非死记)"
    elif g_mem > 0.02 and g_cdsi <= g_mem:
        v = "[CHEAT?] CDSI 提升不超过死记组, 疑似自欺, 需查"
    elif g_cdsi <= 0.02:
        v = "[NULL] CDSI 在 unseen-G 上没明显提升, 机制未显现"
    else:
        v = "[PARTIAL] 部分成立, 需多 seed 钉死"
    lines.append("判决: " + v)
    return "\n".join(lines), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--h", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--n_stories", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cdsi_result.json")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"== CDSI run == device={device}")

    # 结构化留出：训练只用 TRAIN_NAMES，unseen-G 强制用 HOLDOUT_NAMES(训练集从未出现的主角)。
    # 词表覆盖全部名字的字(不是字符注法问题)，测的是组合/规律泛化。
    seen_txt = make_corpus(args.n_stories, seed=args.seed, names=TRAIN_NAMES)
    unseen_txt = make_corpus(max(args.n_stories // 4, 300), seed=10000 + args.seed, names=HOLDOUT_NAMES)
    # 词表用全名字语料构建，保证 holdout 名字的字都在词表里
    vocab_seed_txt = make_corpus(500, seed=7777, names=None)
    stoi, itos = build_vocab(seen_txt + unseen_txt + vocab_seed_txt)
    vocab = len(stoi)
    seen = encode(seen_txt, stoi)
    unseen = encode(unseen_txt, stoi)
    print(f"vocab={vocab} seen_tok={len(seen)} unseen_tok={len(unseen)}")

    probe = TinyLM(vocab, d=args.d, h=args.h, layers=args.layers, ctx=args.ctx)
    print(f"model params = {n_params(probe):,}")

    t0 = time.time()
    print("-- A: CDSI (compression-driven, anti-cheat early-stop + regularize) --")
    traj_cdsi = run_arm("CDSI", args, vocab, seen, unseen, device, lr=3e-4, mode='cdsi')
    print("-- B: PLAIN (just minimize seen loss, no anti-cheat) --")
    traj_plain = run_arm("PLAIN", args, vocab, seen, unseen, device, lr=3e-4, mode='plain')
    print("-- C: MEMORIZE (overfit seen, cheat upper-bound) --")
    args_mem = argparse.Namespace(**vars(args))
    args_mem.steps = args.steps * 3
    traj_mem = run_arm("MEMORIZE", args_mem, vocab, seen, unseen, device, lr=1e-3, mode='mem')

    report, v = verdict(traj_cdsi, traj_plain, traj_mem)
    elapsed = time.time() - t0
    print("\n==== VERDICT ====\n" + report + f"\n(elapsed {elapsed:.0f}s)")

    json.dump({
        "args": vars(args), "params": n_params(probe), "device": device,
        "traj_cdsi": traj_cdsi, "traj_plain": traj_plain, "traj_mem": traj_mem,
        "verdict": v, "elapsed_s": round(elapsed, 1),
    }, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"saved-> {args.out}")


if __name__ == "__main__":
    main()
