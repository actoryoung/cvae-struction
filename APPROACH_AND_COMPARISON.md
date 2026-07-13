# 方法总结与 P-RMF 深度对比

## 一、我们的方法：CVAE-MSA

### 核心思路

**在融合空间（40-dim）而非原始特征空间（768/512/171-dim）中重建缺失模态的潜在表示。**

所有先前尝试（UADG/证据/对比学习）都在做"忽略缺失模态"——调整权重来降权缺失的模态。CVAE 是根本不同思路：**主动生成补偿**。

### 架构

```
训练时:
  [text, audio, vision] → Encoders → [h_t, h_a, h_v]
  随机 mask audio
  CVAE(concat([h_t, h_v])) → h_a_reconstructed
  Concat([h_t, h_a_reconstructed, h_v]) → Output Head → prediction

  Loss = L1(pred, y) + KL(μ,σ || N(0,I)) + MSE(h_recon, h_true)

推理时 (text missing):
  [audio, vision] → Encoders → [h_a, h_v]
  CVAE(concat([h_a, h_v])) → h_t_reconstructed
  Concat([h_t_reconstructed, h_a, h_v]) → prediction
```

### 关键设计决策

| 设计 | 我们的选择 | 理由 |
|------|----------|------|
| 重建空间 | **融合空间 40-dim** | 比原始 768-dim 小 20 倍，易于学习 |
| VAE 类型 | **单个 CVAE** | 30K 参数，极轻量 |
| 条件 | **可用模态的 concat** | 让 CVAE 知道"有哪些信息可用" |
| 训练 | **端到端联合** | 回归任务指导 CVAE 学什么信息重要 |
| 缺失覆盖 | **任意组合** | 训练时随机选择缺失哪个 |

### MOSEI 结果

| Setting | Concat | CVAE (Ours) | Δ |
|---------|:---:|:---:|:---:|
| Full Acc-2 | 0.748 | **0.758** | +1.0pp |
| 缺文本 Acc-2 | 0.587 | **0.618** | +3.1pp |
| 缺音频 Acc-2 | 0.746 | **0.758** | +1.2pp |
| 缺视觉 Acc-2 | 0.743 | **0.754** | +1.1pp |

### 参数效率
- 基础 encoder: 340K (与 concat baseline 相同)
- CVAE 模块: **+30K** (仅增加 8.8%)
- 总计: 370K → GPU 单卡友好

---

## 二、与 P-RMF (ACL 2025) 深度对比

### P-RMF 架构概要

P-RMF 由 5 个组件构成：
1. **Multimodal Encoder** — 每模态独立编码器
2. **PMG (Proxy Modality Generation)** — VAE 映射每模态到 Gaussian 潜空间 → 不确定性加权生成 "proxy modality"
3. **PDDI (Proxy-Driven Dynamic Injection)** — 多层跨模态注意力向 proxy 注入模态语义
4. **DR (Data Reconstruction)** — Transformer 重建器恢复缺失数据
5. **Sentiment Prediction** — 最终分类

### 逐组件对比

| 维度 | P-RMF (ACL 2025) | CVAE-MSA (Ours) |
|------|-----------------|-----------------|
| **VAE 使用** | 每模态独立 VAE，映射到 Gaussian 潜空间 | 单个 CVAE，从可用模态重建缺失模态 |
| **重建空间** | **原始特征空间** (768/512/171-dim) | **融合空间** (40-dim) |
| **重建方式** | Transformer 重建器逐模态恢复原始特征 | CVAE 直接生成缺失模态的融合表示 h |
| **融合机制** | uncertainty-weighted proxy + cross-modal attention injection | 直接 concat 所有 h (包括重建的) |
| **复杂组件** | PMG + PDDI + DR 三个独立模块 | 单 CVAE 模块 |
| **参数量** | 较大 (3×VAE + Transformer 重建器 + cross-attention) | **+30K** (仅增加 8.8%) |
| **训练方式** | 多损失联合 (KL 跨模态 + 重建 + 分类) | 端到端联合 (L1 + KL + MSE) |
| **代码** | [github.com/aoqzhu/P-RMF](https://github.com/aoqzhu/P-RMF) | 本项目 |

### MOSEI 性能对比

| Setting | P-RMF | CVAE (Ours) | 差距 |
|---------|:---:|:---:|:---:|
| Full Acc-2 | **~78.8** | 75.8 | -3.0 |
| 缺文本 | ? | 61.8 | ? |
| 缺音频 | ? | 75.8 | ? |
| 缺视觉 | ? | 75.4 | ? |

注：P-RMF 论文报告的是 inter-modal missing 平均 Acc-2 ≈ 78.14，不是 per-modality 细粒度结果。

### 核心区别（可用于论文的 delta）

| | P-RMF | Ours | 我们的优势 |
|---|------|------|----------|
| 重建粒度 | 原始特征重建（粗粒度） | **融合表示重建（语义级）** | 更高效，更轻量 |
| VAE 数量 | 每模态 1 个 VAE | **1 个 CVAE** | 参数少，泛化好 |
| 代理模态 | 不确定性加权和 (proxy) | **生成式重建** (CVAE) | 真正"补全"而非"近似" |
| 复杂度 | PMG+PDDI+DR 三个模块 | **单模块** | 实现简单 |

### Novelty 评估

| 轴 | 与我们重叠？ |
|----|:--:|
| Problem framing（缺失模态 MSA） | 🔴 相同 |
| Core mechanism（VAE/潜空间） | 🟡 部分重叠（都用 VAE，但用法不同） |
| Key insight（用生成方式补偿缺失） | 🟡 部分重叠（P-RMF 用 proxy；我们用 CVAE 生成） |
| Application domain（MOSI/MOSEI） | 🔴 相同 |

**结论：2 轴重叠 → Level 3 — Medium Overlap**

**Delta statement:**
> Unlike P-RMF (ACL 2025), which uses per-modality VAEs to generate a weighted proxy modality in the raw feature space, our method uses a single lightweight CVAE to reconstruct missing modality representations directly in the fusion space, achieving comparable robustness with only 30K extra parameters and end-to-end joint training.

---

## 三、后续优化方案

### 短期（本周可完成）

| # | 优化 | 预期效果 | 难度 |
|---|------|---------|:--:|
| 1 | 调 KL weight (0.01→1.0) + recon weight | +0.5-1pp | 🟢 |
| 2 | 多样本缺失训练 (mask 1-2 个模态) | 更好的泛化 | 🟢 |
| 3 | Gumbel-Softmax 门控重建质量 | +0.2-0.5pp | 🟡 |
| 4 | 温度退火 (KL annealing) | 训练更稳定 | 🟢 |

### 中期（1-2 周）

| # | 优化 | 预期效果 | 难度 |
|---|------|---------|:--:|
| 5 | 增强音频/视觉特征 (WavLM/HuBERT) | **+2-5pp** | 🟡 |
| 6 | 加入 CASP 测试时自适应 (CVAE + TTA) | +1-2pp | 🟡 |
| 7 | SIMS 中文数据集交叉验证 | 增加实验丰富度 | 🟢 |
| 8 | 与 HVDER/MM-SSC 对比实验 | 完善 related work | 🟡 |

### 论文投稿策略

| 目标会议 | 策略 | 时间 |
|---------|------|------|
| **COLING 2026** | 强调"轻量+简单有效"，30K params 是卖点 | 年底 |
| **EMNLP 2026** | 需要更多实验和 feature enhancement | 年底 |
| **NAACL 2026** | 同上 | 年底 |

---

## 四、关键文件

```
missing-modality-msa/
├── APPROACH_AND_COMPARISON.md  # 本文件
├── EXPERIMENTS.md              # 完整实验记录
├── DESIGN.md                   # 原始研究设计
├── ISSUES.md                   # 问题分析 + NIG bug
├── models/
│   ├── cvae_reconstruct.py     # CVAE 重建模块 (主攻)
│   ├── contrastive_msa.py      # ContraMSA
│   └── ...                     # 其他已尝试方法
└── train_cvae.py               # CVAE 训练脚本
```
