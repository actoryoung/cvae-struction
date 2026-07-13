# 课题 A 研究设计文档

## 论文题目（暂定）

**Uncertainty-Aware Dynamic Gating for Robust Multimodal Sentiment Analysis under Missing Modalities**

## 1. 问题定义

### 1.1 研究问题

多模态情感分析（MSA）依赖文本、音频、视觉三种模态的协同。然而在真实场景中，经常出现**模态缺失**（如静音视频缺少音频、纯音频缺少文本转录、遮挡/低光照导致视觉特征不可靠）。

现有方法的局限：
- **CASP** (AAAI 2025): 只做测试时自适应（TTA），训练时不考虑模态缺失
- **SDUMC** (ICASSP 2025): 仅处理文本缺失，且需要 7B LLM 生成替代特征（太重）

**核心 gap**：缺少一个统一的、轻量级的、能处理**任意模态缺失组合**的鲁棒融合框架。

### 1.2 贡献声明

1. **Uncertainty-Aware Dynamic Gating (UADG)**：提出不确定性感知的动态门控机制，在模态缺失时自动调节可用模态的融合权重
2. **Modality Dropout Training**：训练时随机丢弃模态，使模型学习在各种缺失组合下自适应
3. **轻量级设计**：无需 LLM 辅助，参数量仅增加 <5%，单卡可训练
4. **全面实验**：在 CMU-MOSEI、CMU-MOSI、CH-SIMS 三个数据集上验证

## 2. 方法设计

### 2.1 整体架构

基于 CASP LateFusion 架构改进：

```
                    ┌──────────┐
   Text ───────────►│ Encoder  │──► h_t, σ_t
                    └──────────┘
                    ┌──────────┐
   Audio ──────────►│ Encoder  │──► h_a, σ_a
                    └──────────┘
                    ┌──────────┐
   Vision ─────────►│ Encoder  │──► h_v, σ_v
                    └──────────┘
                           │
                    ┌──────▼──────┐
                    │ Uncertainty │
                    │   Gating    │──► weights w_t, w_a, w_v
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Weighted    │
                    │   Fusion    │──► Regression Head
                    └─────────────┘
```

### 2.2 三个创新组件

#### 组件 1：不确定性估计分支 (Uncertainty Branch)

每个模态编码器输出两个值：
- `h_m`：模态表示（与 CASP 相同）
- `σ_m`：预测不确定性（通过额外的小 MLP 头）

```python
# 每个模态编码器后附加
self.uncertainty_head = nn.Sequential(
    nn.Linear(proj_dim, proj_dim // 2),
    nn.ReLU(),
    nn.Linear(proj_dim // 2, 1),
    nn.Softplus()  # 保证正值
)
```

#### 组件 2：动态门控融合 (Dynamic Gating)

基于不确定性自动计算融合权重：

```
w_m = exp(-σ_m / τ) / Σ_j exp(-σ_j / τ)
fused = Σ_m w_m · h_m
```

其中 τ 是温度参数。不确定性越低的模态获得越高权重。当模态缺失时（h_m = 0），其 σ_m 趋近于很大的值（或直接设 w_m = 0），自动被排除。

#### 组件 3：训练时模态 Dropout (Modality Dropout Training)

训练时随机丢弃模态（以概率 p_drop），让模型学会在各种缺失组合下自适应：

```python
# 7 种可能的缺失模式
# 完整: {T, A, V}
# 缺失单个: {A, V}, {T, V}, {T, A}  
# 缺失两个: {T}, {A}, {V}
# 全部缺失: {} (极端情况，可选)
```

### 2.3 损失函数

```
L_total = L_reg + λ_u * L_uncertainty + λ_c * L_consistency

L_reg: L1 regression loss（与 CASP 相同）
L_uncertainty: 鼓励不确定性估计的校准
  = (y - ŷ)² / (2σ²) + log(σ)  # Gaussian NLL
L_consistency: 同一模态在不同 dropout 下的表示一致性
  = MSE(h_full, h_dropped)
```

### 2.4 与 CASP 的对比

| 组件 | CASP | UADG (Ours) |
|------|------|-------------|
| 模态缺失处理 | 仅 TTA 阶段做 contrastive | 训练时 + 推理时统一处理 |
| 融合方式 | 静态拼接 | 不确定性感知动态加权 |
| 适应性 | 需要三阶段训练 | 端到端单阶段训练 |
| 参数增加 | - | < 5% |

## 3. 实验设计

### 3.1 数据集

- CMU-MOSEI（主实验）
- CMU-MOSI（跨数据集验证）
- CH-SIMS（中文多模态验证）

### 3.2 基线

- LateFusion baseline
- CASP (AAAI 2025)
- SDUMC (ICASSP 2025) — 仅 CMU-MOSEI
- MulT (AAAI 2020)
- Self-MM (AAAI 2021)

### 3.3 评估设置

- 完整模态（Text + Audio + Vision）
- 缺失文本（Audio + Vision only）
- 缺失音频（Text + Vision only）
- 缺失视觉（Text + Audio only）
- 缺失两个模态（single modality）

### 3.4 消融实验

- UADG w/o uncertainty branch（仅动态门控）
- UADG w/o modality dropout（仅不确定性估计）
- UADG w/o consistency loss
- 不同温度参数 τ 的影响
- 不同 dropout 概率的影响

### 3.5 指标

- MAE (Mean Absolute Error)
- Binary Accuracy (Acc-2)
- Multi-class Accuracy (Acc-5, Acc-7)
- F1 Score

## 4. 时间线

| 阶段 | 预计时间 | 内容 |
|------|---------|------|
| Week 1 | 当前 | 实现 UADG 核心代码 |
| Week 2 | | 在 MOSEI 上完成主实验 |
| Week 3 | | 消融实验 + MOSI/SIMS 实验 |
| Week 4 | | 论文撰写 |
| Week 5 | | 修改润色，投稿 |

## 5. 强化方向

### 🔥 方向 1：证据深度学习门控 (Evidential Deep Learning Gating) — 当前实现

**核心 insight**：当前不确定性门控是纯经验设计（MLP → softplus → softmax）。用证据深度学习（EDL, Amini et al. NeurIPS 2020）替换，给门控机制一个**理论解释**。

**方法**：
- 每个模态编码器输出 NIG (Normal-Inverse-Gamma) 分布的 4 个参数：(γ, ν, α, β)
  - γ_m: 模态特定的情感预测值
  - ν_m: 虚拟证据量（evidence）——越多证据 = 越可信
  - α_m, β_m: 控制预测方差的参数
- 融合权重：w_m = ν_m / Σ ν_j（证据越多，权重越大）
- 损失函数：NIG negative log-likelihood + 正则化项
  ```
  L_NIG = ½log(π/ν) - α·log(Ω) + (α+½)·log((y-γ)²ν + Ω) + log(Γ(α)/Γ(α+½))
  L_reg = |y - γ| · (2ν + α)    # 证据正则化：错误越大，惩罚越重
  ```
- **与原始 UADG 的关键区别**：ν_m 有严格的概率解释——它等价于"支持当前预测的虚拟观测数量"

**论文故事**：「多模态情感分析中各模态提供的证据质量不同→我们提出证据驱动的动态融合框架，首次将 Deep Evidential Regression 引入多模态融合」

### 🔥 方向 3：生成式 + 判别式联合框架 (Latent-Space Modality Reconstruction) — 备选

**核心 insight**：缺失模态直接补零+门控太粗糙。不如训练轻量级 CVAE 学习在融合空间中重建缺失模态。

**方法**：
- 训练一个条件 VAE（~100K 参数）从可用模态重建缺失模态的**潜在表示**（不是原始特征）
- 重建发生在**融合后的统一表示空间**中——比在原始特征空间中重建更容易
- 与 SDUMC 的关键区别：不需要 7B LLM，支持任意模态缺失
- 训练：完整模态 → 随机 mask 一个 → CVAE 重建 → KL + reconstruction loss
- 推理：可用模态 → Encoder → CVAE 重建缺失的 → Gating → 预测

**论文故事**：「轻量级潜在空间模态重建+证据门控联合框架，实现鲁棒且高效的多模态情感分析」

## 6. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 不确定性门控效果不显著 | 中 | 改用确定性门控（MLP 直接输出权重） |
| Modality dropout 训练不稳定 | 低 | 逐步增加 dropout 概率（curriculum） |
| 对比 CASP 提升不够 | 中 | 增加 TTA 组件作为补充（结合 CASP 第二阶段） |
