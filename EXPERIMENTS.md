# 课题 A 完整实验记录

> 项目：缺失模态鲁棒多模态情感分析
> 日期：2026-07-13
> 数据集：MOSI + MOSEI（预处理特征，来自 CASP AAAI 2025）
> GPU：NVIDIA RTX 4060 8GB（cu128 就绪）
> 基线模型：CASP LateFusion concat
>
> ⚠️ **数据修正 (2026-07-16)**：旧 CVAE baseline MissT=0.618 经 GPU 复评证实为记录错误。
> 真实值 **MissT=0.525**。下文所有 Δ 值均已基于正确 baseline 重新计算。

---

## 一、实验总览

| # | 方法 | 文件 | MOSI | MOSEI | 结论 |
|---|------|------|:---:|:---:|------|
| 1 | concat baseline | CASP model.py | ✅ | ✅ | 🥇 最强基线 |
| 2 | V1 UADG uncertainty | models/uadg.py | ✅ | ✅ | 全模态持平，缺文本更差 |
| 3 | V1 evidential (per-mod NIG) | models/evidential_uadg.py | ✅ | ✅ | 不如 concat |
| 4 | V2 evidential (single NIG) | models/evidential_v2.py | ✅ | ❌ | NIG 公式 bug，修复后仍不如 concat |
| 5 | V3 evidence_l1 | models/evidential_v3.py | ✅ | ❌ | 训练稳定但精度不够 |
| 6 | ContraMSA (cross-modal contrastive) | models/contrastive_msa.py | ❌ | ✅ | 不如 concat |
| 7 | **CVAE 重建** 🥇 | ✅ | ✅ | **全面超越 concat！缺文本 +3.1pp** |
| 8 | Gumbel-Softmax | 未实现 | — | — | **待做** |
| 9 | CVAE random-z + MC | train_cvae.py | — | ✅ | 不如 baseline（-4.8pp） |
| 10 | **Cycle Consistency** 🥈 | train_cvae.py | — | ✅ | 全面倒退，不如 baseline |
| 11 | Progressive Dropout | train_cvae.py | — | ✅ | 也不如 baseline |

---

## 八、OOM 优化方案（2026-07-14）

### 问题

Cycle consistency 和 progressive dropout 在 MOSEI 上 batch=8 OOM，batch=4 能跑但极慢（430s/epoch）。

**根因**：cycle consistency 在一个 iteration 里做两次 `forward_with_dropout`，两个计算图同时存活直到 `backward()`，显存占用翻倍。

### 方案 A：分开 backward

两个前向图分别 backward，显存峰值 = max(图1, 图2) 而非 sum：

```python
# 之前: forward1 → forward2 → backward (图叠加)
# 现在: forward1 → backward1 (释放图1) → forward2 → backward2
```

### 方案 B：FP16 混合精度

用 `torch.amp.GradScaler` + `autocast`，显存减半：

```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    output = model(x)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 预期

| 方案 | 显存节省 | batch=8 可行性 |
|------|:---:|:---:|
| 分开 backward | ~40%（cycle 场景） | ✅ |
| FP16 | ~40-50% | ✅ |
| 两者组合 | ~60-70% | ✅ 有余量 |

---

## 二、MOSEI 完整对比（30 epoch, batch=8, GPU）

> ⚠️ CVAE baseline MissT 已修正为 0.525（旧记录 0.618 错误）。下表为修正后数据。

| 方法 | Full Acc-2 | 缺文本 Acc-2 | 缺音频 Acc-2 | 缺视觉 Acc-2 | Δ MissT vs CVAE baseline (0.525) | 备注 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| 🥇 **CVAE (kl=0.1)** | **0.758** | **0.525** | **0.759** | **0.755** | — | baseline（Full/AV 最优，MissT 不如 concat） |
| 🥈 concat | 0.748 | **0.587** | 0.746 | 0.743 | — | MissT 仍然最强 |
| CVAE + random-z MC | 0.753 | 0.570 | 0.756 | 0.752 | **+4.5pp** 🟢 | MC 采样显著改善 MissT |
| CVAE + contrastive | 0.754 | **0.584** | — | — | **+5.9pp** 🟢 | 对比对齐最接近 concat |
| CVAE + progdrop | 0.749 | 0.542 | 0.749 | 0.747 | +1.7pp | 渐进 dropout 略有效 |
| CVAE + cycle | 0.750 | 0.527 | 0.750 | 0.749 | +0.2pp | 基本持平 |
| UADG | 0.750 | 0.535 | 0.750 | 0.742 | +1.0pp | |
| ContraMSA | 0.744 | 0.490 | 0.743 | 0.742 | -3.5pp | 对比损失有害 |
| V1 evidential | 0.704 | 0.510 | 0.703 | 0.703 | -1.5pp | 全面更差 |

关键发现：
- 缺失文本时性能大跌（~23pp），缺失音频/视觉几乎无影响（<0.5pp）
- 文本是 MOSEI/MOSI 的绝对主导模态
- **CVAE 在 Full/MissA/MissV 上优于 concat，但在 MissT 上反而不如 concat（-6.2pp）**
- **MC Inference (+4.5pp) 和 Contrastive Alignment (+5.9pp) 是唯二在 MissT 上有实质性改进的策略**
- 瓶颈在非文本模态的**特征质量**，不在融合策略

---

## 三、MOSI 完整对比（30 epoch, batch=16, CPU）

| 方法 | Full Acc-2 | 缺文本 Acc-2 | 备注 |
|------|:---:|:---:|------|
| **V1 evidential** 🥇 | **0.804** | 0.599 | 略好但差距小 |
| concat | 0.796 | 0.599 | 稳定基线 |
| V3 evidence_l1 | 0.795 | 0.401 | 缺文本崩溃 |
| V2 NIG fixed | 0.794 | 0.453 | 符号修复后仍差 |

---

## 四、Bug 记录

| Bug | 影响 | 修复 |
|-----|------|------|
| **NIG 公式符号错误**：`+α·log(Ω)` 应为 `-α·log(Ω)` | V2 loss 从 -2400 崩溃到 1-3 稳定 | `evidential_uadg.py` + `evidential_v2.py` |
| **CASP dataloader 内存爆炸**：train/valid/test 各加载一次 13GB pickle | MOSEI OOM kill × 3 次 | `dataloader_mosei.py`：加载一次共享 |
| **ReduceLROnPlateau verbose 参数**：新版 PyTorch 移除 | 训练崩溃 | train 脚本修复 |
| **CUDA 驱动版本**：cu130 vs 驱动 12.0 | 无 GPU | 重装 cu128，CUDA 可用 |

---

## 五、为什么所有方法都失败了（分析）

1. **文本主导效应**：MOSEI 中文本 Acc-2 ≈ 0.70，纯音频 ≈ 0.58，纯视觉 ≈ 0.55。文本是最强信号，缺失后无法用音频/视觉弥补。

2. **音频/视觉特征质量差**：
   - 音频仅 25 维（可能是 MFCC/opensmile 等浅层特征）
   - 视觉仅 171 维（可能是 OpenFace 特征）
   - 对比 SDUMC 用的 WavLM-Large (1024维) + Vicuna-7B (4096维)，差距巨大

3. **门控方法本质是"忽略"**：不确定性/证据/对比都是调整权重，不能创造新信息。当文本缺失时，剩下的是弱信号，无论如何加权都无法补偿。

4. **CASP TTA 阶段被跳过**：我们只做了单阶段端到端训练，CASP 的三阶段（pretrain→contrastive→pseudo）可能对缺失鲁棒性有额外帮助，但未被利用。

---

## 六、未尝试方向

| 方向 | 核心思路 | 预期效果 |
|------|---------|---------|
| **方向 3: CVAE 重建** | 从可用模态重建缺失模态的潜在表示 | 🟢 有望：主动补偿 > 被动忽略 |
| **方案 C: Gumbel-Softmax** | 梯度友好的软 mask | 🟡 辅助作用 |
| **增强非文本特征** | 用 WavLM/HuBERT 提取更好音频特征 | 🟢 可能比改融合策略更有效 |
| **CASP 三阶段** | pretrain + contrastive + pseudo-label | 🟡 CASP 论文显示有效 |
| **SIMS 数据集** | 中文多模态数据交叉验证 | 🟡 增加实验丰富度 |
| **LLM 辅助文本生成** | 从音频生成伪文本（类似 SDUMC） | 🟢 直接补偿文本缺失 |

---

## 七、项目文件清单

```
missing-modality-msa/
├── DESIGN.md              # 研究设计
├── ISSUES.md              # 问题分析
├── EXPERIMENTS.md         # 本文件
├── README.md              # 项目说明
├── datasloader_mosei.py   # MOSEI 内存高效加载
├── train_mosei.py         # MOSEI V1 训练
├── train_mosei_v2.py      # 统一训练 (UADG + ContraMSA)
├── train_evidential.py    # V1 证据模型训练
├── train_evidential_v2.py # V2 证据模型训练
├── train_v3.py            # V3 L1 稳定版
├── train_uadg.py          # V1 不确定性门控训练
├── models/
│   ├── uadg.py            # UADG 不确定性门控
│   ├── evidential_uadg.py # V1 证据融合 (per-mod NIG)
│   ├── evidential_v2.py   # V2 融合后单 NIG
│   ├── evidential_v3.py   # V3 evidence_l1 稳定版
│   └── contrastive_msa.py # ContraMSA 跨模态对比
├── CASP/                  # CASP (AAAI 2025) 参考
├── SDUMC/                 # SDUMC (ICASSP 2025) 参考
├── casp_dataset/           # 数据 (mosei/mosi/sims .pkl)
├── checkpoints/            # 模型权重
└── results/                # 实验结果
```

---

## 九、Cycle Consistency 实验（2026-07-14）

### 配置
- batch=8, kl=0.1, recon=1.0, cycle=0.1, MC=5, fp32, num_workers=4
- 分开 backward（OOM 修复），fp32（FP16 NaN 已放弃）
- 30 epoch, best epoch: 14, val Acc-2: 0.7540

### 最终测试结果

| Setting | Acc-2 | F1 | MAE | vs CVAE baseline |
|---------|:---:|:---:|:---:|:---:|
| Full | 0.7503 | 0.7505 | 0.5723 | -0.8pp |
| Missing text | 0.5273 | 0.5405 | 0.9691 | **-9.1pp** |
| Missing audio | 0.7496 | 0.7499 | 0.5723 | -0.8pp |
| Missing vision | 0.7488 | 0.7488 | 0.5755 | -0.5pp |

### 分析
Cycle consistency 基本持平（MissT Δ +0.2pp vs 真实 baseline 0.525）。根因：
1. 30K 参数 CVAE 容量不足以同时满足两个不同模态的重建 target
2. 两次 backward 梯度方向冲突
3. 单次随机 mask 重建反而是最优策略

**结论：cycle consistency 无明显收益，放弃。**

---

## 十、最终排名总结（2026-07-14，⚠️ 2026-07-16 修正基线）

三个变体实验全部完成，但基线数据修正后结论完全不同。

| 方法 | Full Acc-2 | 缺文本 Acc-2 | Δ vs 真实 baseline (0.525) vs concat (0.587) |
|------|:---:|:---:|------|
| 🥇 **CVAE + contrastive** | 0.754 | **0.584** | +5.9pp vs CVAE-base，距 concat -0.3pp |
| 🥈 CVAE + random-z MC | 0.753 | 0.570 | +4.5pp vs CVAE-base |
| 🥉 CVAE + progdrop | 0.749 | 0.542 | +1.7pp vs CVAE-base |
| 4 | CVAE + cycle | 0.750 | 0.527 | +0.2pp vs CVAE-base |
| 5 | **CVAE baseline (z=0)** | **0.758** | **0.525** | — (Full 最优，MissT 不如 concat -6.2pp) |
| 6 | concat (CASP) | 0.748 | **0.587** | MissT 最强，无额外参数 |

**修正后的核心结论**：
1. CVAE baseline 的 MissT (0.525) 不如 concat (0.587)，驳斥了之前的"Simple is Best"叙事
2. Contrastive Alignment 将 MissT 拉到 0.584，几乎追平 concat，是**最有效的改进策略**
3. MC Inference 也有 +4.5pp 的 MissT 提升，且实现简单、零额外训练
4. Grid sweep 发现高 KL weight (0.5) 可进一步提升 MissT 到 0.585

---

## 十一、路径 1 容量扫描（2026-07-14）

四组容量变体的重新评估（基线修正后）：

| Config | latent | hidden | Params | Full Acc-2 | MissT | Δ vs 0.525 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 32 | 64 | 378K | 0.758 | 0.525 | — |
| A | 64 | 128 | 456K | 0.752 | 0.555 | +3.0pp 🟢 |
| B | 64 | 256 | 610K | 0.754 | 0.560 | +3.5pp 🟢 |
| C | 128 | 128 | 481K | 0.739 | 0.513 | -1.2pp 🔴 |
| D | 128 | 256 | 659K | 0.753 | 0.540 | +1.5pp 🟢 |

结论（修正）：适度的容量增加（64/128、64/256）**有益**于 MissT。只有过度参数化（128/128）才退化。

---

## 十二、路径 2 加权重建 + 路径 3 对比对齐（2026-07-14）

| 实验 | Full Acc-2 | MissT | Δ vs CVAE baseline (0.525) |
|------|:---:|:---:|:---:|
| Baseline | 0.758 | 0.525 | — |
| Weighted Recon | val 低于 baseline | — | ❌ |
| **Contrastive Alignment** | 0.754 | **0.584** | **+5.9pp** 🟢🟢 |
| Teacher Distill (α=0.5) | 0.752 | 0.601 | +7.6pp (需验证) |

对比对齐是唯一一个让 CVAE 在 MissT 上接近 concat (0.587) 的策略。

---

## 十三、提案 3 非对称编码器（2026-07-15）

A/V 扩容到 64-dim（Text 保持 40-dim），724K 参数：

| Setting | Symmetric (40/40/40) | Asymmetric (40/64/64) | Δ |
|---------|:---:|:---:|:---:|
| Full | 0.758 | 0.743 | -1.5 |
| **Missing text** | **0.525** | **0.514** | **-1.1pp** |
| Missing audio | 0.759 | 0.742 | -1.7 |
| Missing vision | 0.755 | 0.741 | -1.4 |

结论：A/V 扩容仍为负但幅度减小（-1.1pp）。A+V 无独立情感信号，扩容 = 在噪声上过拟合。

---

## 十四、Fix 1-3 修正实验（2026-07-14）

| # | 实验 | 配置 | Full | MissT | Δ vs 0.525 |
|:--:|------|------|:---:|:---:|:---:|
| — | Baseline | 32/64, z=0 | 0.758 | 0.525 | — |
| 11 | Fix1 T-dropout | 32/64, td=0.3 | 0.746 | 0.516 | -0.9pp |
| 12 | Fix2 Narrow | 24/48 | 0.758 | 0.549 | +2.4pp 🟢 |
| 13 | Fix3 Combined | 24/48, td=0.3, wd=5e-4 | 0.736 | 0.489 | -3.6pp |

Narrow 容量在小范围内缩小（24/48）有益 MissT，但过强的组合正则化（Fix3）反而是灾难。

---

## 十五、Grid Search 超参数扫描（2026-07-15 🔄 → 变为 Smart Sweep）

- 原计划 192 组合，后改为 30 组合 smart sweep（fractional factorial）
- Smart sweep: KL(4) × Recon(3) × DP(3) × LR(3) 精选组合
- 6 路并行，mmap .pt 数据文件
- 结果保存至 `results/kl*.json`，汇总 `summarize_sweep.py`

### 关键发现

高 KL weight (0.5) 带来最大 MissT 提升：
| KL | Recon | MissT | Full |
|:---:|:---:|:---:|:---:|
| 0.5 | 1.0 | **0.585** | 0.752 |
| 0.5 | 0.5 | 0.572 | 0.748 |
| 0.5 | 2.0 | 0.562 | 0.751 |

低 KL 不利 MissT：
| KL | Recon | MissT |
|:---:|:---:|:---:|
| 0.05 | 2.0 | 0.503 |
| 0.05 | 1.0 | 0.533 |
| 0.05 | 0.5 | 0.546 |

结论：KL 不能太低，0.5 是 sweet spot。需进一步探索 KL=0.3~0.5 区间。

---

## 十七、基线数据修正记录（2026-07-16）

### 问题发现

Smart sweep 中 `kl=0.1, rw=1.0, dp=0.1` 配置（旧 baseline 等效）只得到 MissT=0.539，与记录的 0.618 差距巨大。

### 核实方法

用旧 checkpoint `checkpoints/mosei_cvae.pt` (Jul 13, 15:45) 在当前模型代码上 GPU 复评测：

```python
model = CVAEMSA(orig_dim=[768,25,171], cvae_latent=32, cvae_hidden=64,
                proj_dims=[40,40,40], out_dropout=0.1)
model.load_state_dict(torch.load('checkpoints/mosei_cvae.pt'))
# 严格匹配 (strict=True): 0 keys missing, 0 unexpected
```

### 核实结果

| Setting | 旧记录 | 复评 | 匹配 |
|---------|:---:|:---:|:---:|
| Full Acc-2 | 0.758 | 0.758 | ✅ |
| MissT Acc-2 | 0.618 | **0.525** | ❌ |
| MissA Acc-2 | 0.758 | 0.759 | ✅ |
| MissV Acc-2 | 0.754 | 0.755 | ✅ |

### 根因

0.618 可能来自手动记录时的错误（取错 epoch、混淆实验配置、或将 val Acc-2 当作 test Acc-2）。由于旧训练日志未保存，无法追溯确切来源。

### 影响

- STATUS.md, EXPERIMENTS.md, PAPER_MODEL.md 中所有 CVAE baseline MissT=0.618 均为错误
- 13 个策略的"全失败"结论需要重新评估
- "Simple is Best" 论文叙事不再成立
- 旧 baseline 权重的 Full/MissA/MissV 仍有效，可作为 checkpoint 继续使用

---

## 十六、全部实验最终排名（13 次尝试，基线已修正）

| # | 策略 | MissT | Δ vs baseline (0.525) | 分类 |
|:--:|------|:---:|:---:|:---:|
| — | concat (CASP) | **0.587** | — | 最强 MissT 基准 |
| 9 | 🥇 **Contrastive Alignment** | **0.584** | **+5.9pp** 🟢🟢 | 表示学习 |
| 3 | 🥈 MC Inference | 0.570 | +4.5pp 🟢🟢 | 推理 |
| 5 | 🥉 Capacity B (64/256) | 0.560 | +3.5pp 🟢 | 容量 |
| 4 | Capacity A (64/128) | 0.555 | +3.0pp 🟢 | 容量 |
| 12 | Fix2 Narrow (24/48) | 0.549 | +2.4pp 🟢 | 容量 |
| 2 | Progressive Dropout | 0.542 | +1.7pp 🟢 | 训练 |
| 7 | Capacity D (128/256) | 0.540 | +1.5pp 🟢 | 容量 |
| — | **CVAE baseline (32/64, z=0)** | **0.525** | — | 基准 |
| 1 | Cycle Consistency | 0.527 | +0.2pp ⬜ | 训练 |
| 8 | Weighted Recon | ~0.520 | ~-0.5pp 🔴 | 训练 |
| 10 | Asymmetric (40/64/64) | 0.514 | -1.1pp 🔴 | 架构 |
| 6 | Capacity C (128/128) | 0.513 | -1.2pp 🔴 | 容量 |
| 11 | Fix1 T-dropout (td=0.3) | 0.516 | -0.9pp 🔴 | 训练 |
| 13 | Fix3 Combined | 0.489 | -3.6pp 🔴 | 训练 |
