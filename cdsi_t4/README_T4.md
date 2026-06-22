# CDSI on T4 — 一键运行指南

## 这是什么
"压缩率驱动的开放域自改进 (CDSI)" 最小可验证实验。
核心命题: 微型模型以"无损压缩率(bits/token)"为唯一内生奖励自我改进,
用 unseen-G(训练集从未出现的主角名)防自欺 —— 压缩率在没见过的数据上也涨=真理解, 否则=死记。

## 文件
- model.py : 微型 Transformer LM + bits_per_token 度量(=压缩后长度)
- data.py  : 中文 TinyStories 风格语料, 结构化留出(TRAIN_NAMES vs HOLDOUT_NAMES)
- run.py   : CDSI 闭环 + 3对照(CDSI/PLAIN/MEMORIZE) + 条件门控判决, 支持 cuda+混合精度

## Colab / T4 跑法

### 1. 上传 3 个 .py 到 Colab 工作目录(或 git clone)
### 2. 确认 GPU
```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
### 3. 跑 Small 档(10M, T4 甜点, ~20-40分钟)
```bash
!python run.py --device cuda --d 256 --h 8 --layers 6 --ctx 256 \
    --bs 64 --rounds 24 --steps 300 --n_stories 8000 --seed 0 --out cdsi_small_s0.json
```
### 4. 多 seed 钉死(铁律: 单seed不算坐实)
```bash
!python run.py --device cuda --d 256 --h 8 --layers 6 --ctx 256 --bs 64 --rounds 24 --steps 300 --n_stories 8000 --seed 1 --out cdsi_small_s1.json
!python run.py --device cuda --d 256 --h 8 --layers 6 --ctx 256 --bs 64 --rounds 24 --steps 300 --n_stories 8000 --seed 2 --out cdsi_small_s2.json
```
### 5. (可选)Mid 档冲碾压展示(30M, ~1.5-3h)
```bash
!python run.py --device cuda --d 384 --h 8 --layers 8 --ctx 384 --bs 48 --rounds 30 --steps 400 --n_stories 15000 --out cdsi_mid.json
```

## 怎么看结果(关键)
跑完看末尾 VERDICT:
- **MEMORIZE seen 提升 ≫ unseen 提升** -> 死记自欺判据校准成功(本机已验证, 死记 unseen 涨0.07 vs seen涨1.86)。
- **关键看点**: T4 大规模 + 24轮, PLAIN 会真正开始过拟合(unseen 先降后升=自欺),
  这时 **CDSI 若能稳住(unseen 不回升)且终值 < PLAIN = 命题成立 [OK]**。
  这是本机玩具规模(够不到过拟合临界点)验证不了、必须上 T4 才见分晓的核心。

## 本机已验证(无需在 T4 重复)
- 代码无 bug, 闭环跑通, bits/token 正常下降。
- 防自欺判据成立: 结构化留出后死记组在 unseen-G 涨不动。
- 待 T4 验证的唯一问题: 大规模过拟合时 CDSI 防自欺是否真的赢 PLAIN。

## 诚实边界
- 数据仍是模板中文故事(非真实语料)。若 T4 上 [OK], 下一步换真实中文 TinyStories 语料复现。
- 架构用标准 Transformer(认怂用最好现成件), 创新在 CDSI 训练循环不在网络结构。
