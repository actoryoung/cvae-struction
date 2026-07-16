# CVAE-MSA: Best Model Specification

> **Paper-ready documentation — ⚠️ 2026-07-16 基线数据修正**
> Date: 2026-07-16 | Target venues: EMNLP / COLING / AAAI
>
> **修正说明**：旧 baseline MissT=0.618 经 GPU 复评证实为记录错误，真实值 **MissT=0.525**。
> 此修正推翻了 "Simple is Best" 叙事，CVAE baseline 在 MissT 上实际不如 concat (0.587)。

---

## 一、模型命名

**CVAE-MSA** — Conditional Variational Autoencoder for Missing-Modality Sentiment Analysis

---

## 二、核心思想

### 2.1 问题定义

多模态情感分析中，**文本模态缺失时性能崩溃**是现有方法的共同缺陷。此前所有方法（CASP, SDUMC, UADG等）对缺失模态均采用**零填充**（zero-filling）策略——不产生新信息，只能被动承受信息损失。

### 2.2 核心创新

**在融合空间（fusion space）中重建缺失模态的潜在表示**，而非在原始特征空间中重建。

| 设计选择 | 我们的做法 | 对比方法（P-RMF, HVDER） |
|---------|---------|------|
| 重建空间 | 融合空间 (40-dim) | 原始特征空间 (768/25/171-dim) |
| 重建器结构 | 单 CVAE | 多个 VAE / VAE+Diffusion |
| 参数开销 | +30K (8.8%) | 10×–50× |
| 推理方式 | 确定性 z=0 或 MC 采样 | 随机采样 |

**关键直觉**：原始特征维度差异巨大（text=768d, audio=25d, vision=171d），在原始空间重建需要处理异构维度且计算昂贵。我们将所有模态投影到统一 40-dim 融合空间后，CVAE 只需重建这个紧凑表示，大幅降低难度和参数。

### 2.3 推理策略演变

| 策略 | MissT | 说明 |
|------|:---:|------|
| **MC Inference (random-z)** | **0.570** (+4.5pp vs baseline) | 多样本采样平均提升鲁棒性 |
| **Contrastive + z=0** | **0.584** (+5.9pp) | 对比对齐 + CVAE 联合优化 |
| baseline z=0 | 0.525 | 确定性推理（先验均值） |

**结论（修正）**：MC 采样和对比对齐都显著改善 CVAE baseline 的 MissT，确定性 z=0 不是最优。

---

## 三、模型架构

### 3.1 整体结构

```
Input: [text(768d), audio(25d), vision(171d)]
  │
  ├── Proj_T: Conv1d(768→40) → TransformerEncoder(5层, 8头)
  ├── Proj_A: Conv1d(25→40)  → TransformerEncoder(5层, 8头)
  ├── Proj_V: Conv1d(171→40) → TransformerEncoder(5层, 8头)
  │
  ├── h_t(40d), h_a(40d), h_v(40d)
  │
  ├── CVAE Reconstructor ──────────────┐
  │   Encoder: [h_avail, h_target] → MLP(64) → μ(32), logvar(32)
  │   Decoder: [z(32), h_avail]   → MLP(64) → h_recon(40)
  │   Inference: z ~ N(0,I) MC sampling (k=5) or z=0
  │
  └── Concat[h_t, h_a, h_v] → Output Head
      Linear(120→80) → ReLU → Dropout(0.1)
      → Linear(80→40) → ReLU → Linear(40→1)
```

### 3.2 各组件参数

| 组件 | 描述 | 参数量 |
|------|------|:---:|
| **Text Projection** | Conv1d(768, 40) + Transformer(40d, 8头, 5层) | ~120K |
| **Audio Projection** | Conv1d(25, 40) + Transformer(40d, 8头, 5层) | ~108K |
| **Vision Projection** | Conv1d(171, 40) + Transformer(40d, 8头, 5层) | ~112K |
| **CVAE Encoder** | Linear(80→64) + ReLU + Linear(64→64) + ReLU + Linear(64→32)×2 | ~13K |
| **CVAE Decoder** | Linear(72→64) + ReLU + Linear(64→64) + ReLU + Linear(64→40) | ~13K |
| **Output Head** | Linear(120→80) + ReLU + Dropout + Linear(80→40) + ReLU + Linear(40→1) | ~11K |
| **Total** | | **~378K** |

CVAE 部分独立参数：~30K（仅占总参数 8.8%）

### 3.3 关键设计细节

1. **融合空间维度**：`proj_dim = 40`（对所有模态统一，对称设计）
2. **CVAE 潜变量维度**：`latent_dim = 32`。更大的 latent_dim 在某些容量配置下有正向效果
3. **CVAE 隐层维度**：`hidden_dim = 64`。容量 B (64/256) 的 MissT +3.5pp
4. **Transformer 编码器**：5层, 8头, 每模态独立。使用 CASP 的基础架构以保持公平对比
5. **输出头**：压缩金字塔结构 (120→80→40→1)，dropout=0.1 控制过拟合

---

## 四、训练配置

### 4.1 损失函数

$$L = L_{\text{reg}} + L_{\text{reg}}^{\text{drop}} + \beta \cdot KL(q(z|x_{avail}, x_{target}) \| p(z|x_{avail})) + \lambda \cdot MSE(h_{recon}, h_{target})$$

| 项 | 公式 | 权重 |
|----|------|:---:|
| 回归损失 (Full) | L1(pred_full, y) | 1.0 |
| 回归损失 (Drop) | L1(pred_drop, y) | 1.0 |
| KL 散度 | D_KL(q(z|x_avail, x_target) ‖ p(z|x_avail)) | **0.1–0.5** |
| 重建损失 | MSE(h_recon, h_target) | **0.5–1.0** |

### 4.2 关键超参数

| 超参数 | 旧最优值 | 新发现 | 说明 |
|--------|:--:|:--:|------|
| **KL weight (β)** | 0.1 | **0.5** ⭐ | 高 KL 显著提升 MissT (0.585) |
| **Recon weight (λ)** | 1.0 | 0.5–1.0 | 与 KL weight 平衡 |
| **Dropout prob** | 0.2 | 0.2 | |
| **Learning rate** | 1e-3 | 1e-3 | Adam 优化器 |
| **LR scheduler** | ReduceLROnPlateau (patience=10, factor=0.1) | | |
| **Gradient clip** | 0.8 | | |
| **Batch size** | 32 | | |
| **Epochs** | 30 | | |

---

## 五、实验结果

### 5.1 MOSEI 测试集（Acc-2，⚠️ 基线已修正）

| Setting | concat baseline | CVAE-MSA (旧 baseline) | CVAE-MSA (KL=0.5) |
|---------|:---:|:---:|:---:|
| **Full** | 0.748 | **0.758** | 0.752 |
| **Missing text** | **0.587** | 0.525 | **0.585** ⭐ |
| **Missing audio** | 0.746 | 0.759 | |
| **Missing vision** | 0.743 | 0.755 | |

**关键**：KL=0.5 将 CVAE 的 MissT 从 0.525 提升到 0.585，接近 concat 的 0.587。Full 轻微下降但仍优于 concat。

### 5.2 参数效率对比

| 方法 | 额外参数 | MissT Acc-2 |
|------|:---:|:---:|
| concat (CASP) | 0 | **0.587** |
| **CVAE-MSA (KL=0.5)** | +30K | **0.585** |
| CVAE + Contrastive | +80K | 0.584 |
| CVAE + MC Inference | +30K | 0.570 |
| CVAE baseline (KL=0.1) | +30K | 0.525 |
| UADG | +40K | 0.535 |
| Evidential NIG | +120K | 0.510 |

---

## 六、设计原则与经验教训（修正后）

### 6.1 什么有效（What Works）

1. **融合空间重建**：40-dim 紧凑空间是 CVAE 能工作的关键
2. **高 KL weight**：KL=0.5 是最关键的超参发现，单改它就能 +6pp MissT
3. **MC 多样本推理**：z ~ N(0,I) 采样平均提升 MissT +4.5pp
4. **对比对齐**：跨模态对比损失增强表示质量，MissT +5.9pp（最有效）
5. **适度容量增加**：64/128 或 64/256 配置有益 MissT（+3~3.5pp）

### 6.2 什么无效（What Doesn't Work）

| 策略 | Δ MissT vs 0.525 | 评估 |
|---------|:---:|------|
| Fix3 Combined（多重正则叠加） | -3.6pp | 🔴 过强正则化是灾难 |
| Capacity C (128/128) | -1.2pp | 🔴 等宽扩容有害 |
| Asymmetric A/V expansion | -1.1pp | 🔴 A+V 无信号，扩容=过拟合噪声 |
| Fix1 T-dropout | -0.9pp | 🔴 强制降 T 依赖 |
| Weighted Reconstruction | ~-0.5pp | 🔴 维度加权引入偏差 |
| Cycle Consistency | +0.2pp | ⬜ 无显著收益 |

### 6.3 修正后的核心结论

> 旧 "Simple is Best" 叙事基于错误的基线数据。正确的结论是：
>
> **CVAE 空间重建在缺失音频/视觉场景下确实优于 concat，但缺失文本场景下 baseline CVAE 反而不如 concat。**
> 通过系统性优化——高 KL weight (0.5)、MC 推理、对比对齐——可以将 MissT 从 0.525 提升到 0.585，基本追平 concat 的 0.587。
>
> **论文叙事方向（建议）**：不是 "简单就是最优"，而是 **"CVAE 重建是一种有潜力的框架，但需要精心调参才能发挥其效果——我们通过系统性超参搜索找到了关键杠杆（高 KL weight），并展示了多个改进路径（MC、对比、容量）"**。

---

## 七、与竞争方法的差异化

| 维度 | P-RMF (ACL 2025) | HVDER (2026) | **CVAE-MSA (Ours)** |
|------|------|------|------|
| 重建空间 | 原始特征空间 | 原始特征空间 | **融合空间 (40d)** |
| 重建器 | 多 VAE | VAE + Diffusion | **单 CVAE** |
| 参数量 | 大 | 很大 | **+30K (8.8%)** |
| 推理 | 随机采样 | 扩散去噪 | **MC 采样或 z=0** |
| 模态缺失处理 | 重建 | 重建+生成 | **重建（更轻量）** |
| MissT vs concat | 未知 | 未知 | 接近（0.585 vs 0.587） |

---

## 八、文件清单

| 文件 | 内容 |
|------|------|
| `models/cvae_reconstruct.py` | CVAE-MSA 完整模型定义 |
| `train_cvae.py` | 训练脚本（支持所有 CLI 参数） |
| `dataloader_mosei.py` | MOSEI 数据加载（pickle 版） |
| `dataloader_mosei_pt.py` | MOSEI 数据加载（.pt mmap 版） |
| `checkpoints/mosei_cvae.pt` | 旧 baseline 权重（Full 最优，MissT 弱） |
| `preprocess_mosei.py` | 数据预处理（pickle → .pt） |
| `record_results.py` | 实验结果自动记录 |
| `summarize_sweep.py` | Grid search 结果汇总 |
