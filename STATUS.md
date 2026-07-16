# 课题 A 工作状态（截至 2026-07-16）

> ⚠️ **数据修正 (2026-07-16)**：旧 baseline MissT=0.618 经 GPU 复评测证实为记录错误。真实 CVAE baseline：
> Full=0.758, **MissT=0.525**（不如 concat 的 0.587）。详见[基线数据修正](#基线数据修正)。

## 当前运行中

| 任务 | 状态 | 配置 |
|------|:--:|------|
| **Smart Sweep** (30 combos) | 🔄 6路并行 | Block 3 LR sweep 进行中, ~18/30 完成 |
| **Smart Sweep 2** (KL=0.5 深度探索) | ⏳ | 基于 kl=0.5 系列 MissT 最高的发现 |

## 基线数据修正

### 旧 baseline 核实

旧 checkpoint `mosei_cvae.pt` (Jul 13) 用当前模型代码 + .pt 数据 GPU 复评：

| Setting | 旧记录 | **实际评测** |
|---------|:---:|:---:|
| Full | 0.758 | **0.758** ✓ |
| **Missing text** | **0.618** ❌ | **0.525** |
| Missing audio | 0.758 | 0.759 |
| Missing vision | 0.754 | 0.755 |

**根因**：0.618 是在过去对话中手动记录时出错（可能是取了不同 epoch 的 val 值，或与其他实验混淆）。此错误已传导至 STATUS.md, EXPERIMENTS.md, PAPER_MODEL.md。

### 正确基线对比

| 方法 | Full Acc-2 | MissT Acc-2 | MissA Acc-2 | MissV Acc-2 |
|------|:---:|:---:|:---:|:---:|
| concat (CASP) | 0.748 | **0.587** | 0.746 | 0.743 |
| **CVAE baseline (old)** | **0.758** | 0.525 | **0.759** | **0.755** |

**结论**：CVAE 在 Full/MissA/MissV 上优于 concat，但在 **MissT 上反而更差 (-6.2pp)**。这完全推翻了之前"Simple is Best"的叙事。

## Smart Sweep 关键发现

### Block 1-2 完成 (18/30)

| Config | Full | MissT | 备注 |
|--------|:---:|:---:|------|
| kl=0.5, rw=1.0, dp=0.2 | 0.752 | **0.585** ⭐ | 接近 concat (0.587) |
| kl=0.5, rw=0.5, dp=0.2 | 0.748 | 0.572 | |
| kl=0.5, rw=2.0, dp=0.2 | 0.751 | 0.562 | |
| kl=0.1, rw=0.5, dp=0.2 | 0.754 | 0.557 | |
| kl=0.2, rw=0.5, dp=0.2 | 0.750 | 0.554 | |
| kl=0.05, rw=0.5, dp=0.2 | 0.747 | 0.546 | |
| kl=0.1, rw=1.0, dp=0.1 | 0.744 | 0.539 | 旧 baseline 等效配置 |
| kl=0.1, rw=1.0, dp=0.2 | 0.748 | 0.523 | |
| kl=0.05, rw=1.0, dp=0.2 | 0.753 | 0.533 | |

**趋势**：**高 KL weight (0.5) + recon=1.0 是目前最优 MissT 配置**，首次让 CVAE 在 MissT 上接近 concat baseline。

## 13 种策略重新评估（基于真实 baseline 0.525）

| # | 策略 | MissT | Δ vs 0.525 | 重新评估 |
|:--:|------|:---:|:---:|------|
| 1 | Cycle Consistency | 0.527 | +0.2pp | ⬜ 持平，非失败 |
| 2 | Progressive Dropout | 0.542 | +1.7pp | 🟢 正向 |
| 3 | **MC Inference (random-z)** | **0.570** | **+4.5pp** | 🟢🟢 **显著正向！** |
| 4 | Capacity A (64/128) | 0.555 | +3.0pp | 🟢 正向 |
| 5 | **Capacity B (64/256)** | **0.560** | **+3.5pp** | 🟢🟢 正向 |
| 6 | Capacity C (128/128) | 0.513 | -1.2pp | 🔴 仍为负 |
| 7 | Capacity D (128/256) | 0.540 | +1.5pp | 🟢 正向 |
| 8 | Weighted Reconstruction | 0.520(est) | ~-0.5pp | 🔴 仍为负 |
| 9 | **Contrastive Alignment** | **0.584** | **+5.9pp** | 🟢🟢 **最有效！接近 concat 0.587** |
| 10 | Asymmetric (40/64/64) | 0.514 | -1.1pp | 🔴 仍为负 |
| 11 | Fix1 T-dropout (td=0.3) | 0.516 | -0.9pp | 🔴 仍为负 |
| 12 | Fix2 Narrow (24/48) | 0.549 | +2.4pp | 🟢 正向 |
| 13 | Fix3 Combined | 0.489 | -3.6pp | 🔴 仍为负 |

### 新核心结论

1. **弃用 "Simple is Best" 叙事** — 旧 baseline 0.618 并不存在，CVAE baseline MissT 不如 concat
2. **正向前 6 名**: Contrastive Alignment (+5.9) > MC Inference (+4.5) > Capacity B (+3.5) > Capacity A (+3.0) > Fix2 Narrow (+2.4) > Progressive Dropout (+1.7)
3. **Grid sweep 发现 KL=0.5 是更关键的超参** — 仅调整 KL weight 就能将 MissT 从 0.525 提升到 0.585
4. **未来方向**: KL=0.5 + Contrastive + MC Inference 组合可能是突破点
5. **论文策略需要重新制定** — 不再讲"简单最优"，而是讲"通过系统优化从 0.525 提升到 0.585+"的过程

## 模态分析

| 模态 | 标签 R² | 结论 |
|------|:---:|------|
| Text | 0.268 | 唯一独立情感信号 |
| Audio | -0.102 | 无独立信号 |
| Vision | -0.102 | 无独立信号 |
| A+V | -0.078 | 联合也无信号 |

## 论文写作状态

| 文档 | 路径 | 状态 |
|------|------|:--:|
| 最佳模型规格 | `paper/PAPER_MODEL.md` | ✅ 已修正基线 |
| **超参分析** | **`paper/HYPERPARAM_ANALYSIS.md`** | **✅ 完成 — 论文级分析** |
| 论文大纲 | `paper/PAPER_OUTLINE.md` | ⚠️ 需重新设计叙事 |
| 论文正文 | `paper/paper_draft.tex` | 🔄 填充中 |
| MOSI 实验 | — | ⏳ |
| SIMS 实验 | — | ⏳ |

## Smart Sweep 关键发现（超参分析）

| 发现 | 数据 |
|------|------|
| KL weight 主导性 | 解释 MissT 方差的 **63.3%** |
| KL=0.5 相变式跳升 | MissT +4.8pp vs KL=0.1 |
| KL×Recon 交互 | 低 KL→RW=0.5 最优；高 KL→RW=1.0 最优 |
| 理论解释 | 低 KL 导致 posterior collapse，高 KL 强制学到真正跨模态映射 |
| 论文策略 | 作为独立分析 Section 嵌入主论文 |

## 下一步工作计划

| 优先级 | 任务 | 状态 | 预计 |
|:--:|------|:--:|------|
| 🔄 | Smart Sweep 完成（30 combos） | 运行中 | 今晚 |
| 🔴 P0 | KL=0.5 深度探索（更多 KL/Recon 组合） | ⏳ | 1 天 |
| 🔴 P0 | 重新评估 MC Inference + Contrastive 组合 | ⏳ | 1 天 |
| 🔴 P0 | MOSI: concat baseline + CVAE-MSA 最佳配置 | ⏳ | 1-2 天 |
| 🟡 P1 | 更新论文文档（修正基线数据后） | ⏳ | 1 天 |
| 🟡 P1 | SIMS 中文数据集 | ⏳ | 3-4 天 |
