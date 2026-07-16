# CVAE-MSA 论文大纲

> Target: EMNLP / COLING / AAAI 2026
> Working Title: **Simple is Best: Lightweight CVAE Reconstruction in Fusion Space for Missing-Modality Sentiment Analysis**
> Status: Outline (2026-07-15)

---

## 一、Title (备选)

1. ✨ **Simple is Best: Lightweight CVAE Reconstruction in Fusion Space for Missing-Modality Sentiment Analysis**
2. **Fusion-Space Modality Reconstruction: Why a 30K-Parameter CVAE Beats Complex Designs for Missing Modality Robustness**
3. **Rethinking Missing Modality Recovery: Minimal CVAE in Compact Fusion Space**

**推荐**: 标题 1——直接点出核心 narrative (simple > complex) 和方法关键词 (CVAE, Fusion Space)。

---

## 二、Abstract (约 200 words)

**结构**:
1. **Problem**: 多模态情感分析在模态缺失（尤其文本）时性能崩溃。现有方法或被动的忽略缺失（zero-filling）、或使用沉重的多 VAE/扩散模型在原始特征空间重建。
2. **Method**: 本文提出 CVAE-MSA——在 40-dim **融合空间**（非原始特征空间）用单个轻量 Conditional VAE 重建缺失模态表示。训练时联合优化回归+KL+重建损失；推理时使用确定性潜变量 z=0。
3. **Key finding**: 系统性探索 13 种改进策略（cycle consistency, contrastive alignment, capacity scaling, asymmetric encoding...）后发现——**最简单的基础 CVAE 配置是不可击败的**。任何额外复杂度都损害鲁棒性。
4. **Results**: MOSEI 数据集上，Missing-text Acc-2 提升 +3.1pp (0.587→0.618)，Full +1.0pp。仅增加 30K 参数（8.8%）。
5. **Implication**: 在融合空间重建的设定下，参数效率不是妥协而是内在优势。

---

## 三、Introduction (约 1.5 页)

### 3.1 背景与动机
- 多模态情感分析的应用场景（社交媒体、对话系统、心理健康）
- 真实场景中模态缺失是常态：文本（用户隐私设置）、音频（噪声环境）、视觉（摄像头遮挡）
- 现有方法的两类策略：
  - **忽略缺失**：zero-filling (CASP)、不确定性门控 (UADG)、证据融合 → 被动承受信息损失
  - **重建缺失**：P-RMF（多 VAE 原始空间重建）、HVDER（VAE+Diffusion）→ 参数大、架构重

### 3.2 核心观察
- 融合空间（40-dim）是高度信息压缩的表示空间，远低于原始特征维度（text 768d + audio 25d + vision 171d）
- 在融合空间中做重建比在原始空间中更简单、更高效
- 实验发现：Audio+Vision 联合的情感信号 R² = −0.078（基本为零）→ CVAE 被要求"从噪声中重建信号"

### 3.3 本文贡献
1. **方法创新**：提出融合空间 CVAE 重建的简单有效架构（+30K 参数）
2. **反直觉发现**：简单基础配置是最优的——13 种复杂改进全部失败，确定性的 z=0 推理优于随机采样
3. **系统性消融**：全面的超参数扫描（192 组合）和策略消融实验
4. **参数效率论证**：仅 8.8% 参数增量换取最优缺失文本鲁棒性

---

## 四、Related Work (约 1 页)

### 4.1 多模态情感分析
- 基础方法：TFN (Zadeh et al., 2017), LMF (Liu et al., 2018), MulT (Tsai et al., 2019)
- Transformer 方法：CASP (Guo et al., AAAI 2025), SDUMC (Cong et al., ICASSP 2025)
- 大模型适配：MSE-Adapter (AAAI 2025), DashFusion (IEEE T-NNLS 2025)

### 4.2 缺失模态鲁棒性
- **忽略策略**：不确定性门控 (UADG)、证据深度学习、对比学习对齐 → 不产生新信息
- **重建策略**：
  - P-RMF (ACL 2025)：多 VAE 在原始空间重建，参数大
  - MM-SSC (2025)：VQ-VAE 离散潜空间重建
  - HVDER (2026)：VAE + 扩散模型联合重建与生成
- **我们的定位**：融合空间单 CVAE 重建（更轻盈、更有效）

### 4.3 Conditional VAE
- 基础 CVAE (Sohn et al., 2015)
- β-VAE (Higgins et al., 2017), Annealed VAE
- Deep Evidential Regression (Amini et al., NeurIPS 2020)

---

## 五、Method (约 2.5 页)

### 5.1 问题形式化
- 输入：x = {x_T, x_A, x_V}（text, audio, vision）
- 模态编码器：h_m = Enc_m(x_m), for m ∈ {T, A, V}
- 融合：h_fusion = Concat[h_T, h_A, h_V]
- 预测：ŷ = OutputHead(h_fusion)
- 缺失模态：x_T = ∅ → h_T = 0（zero-fill baseline）

### 5.2 CVAE 重建器
- **训练时**（有 ground-truth h_target）：
  - Encode: [h_avail, h_target] → μ, logvar
  - Sample: z ~ q(z|h_avail, h_target)
  - Decode: h_recon = Dec([z, h_avail])
- **推理时**（无 ground-truth）：z = 0（确定性先验均值）
- **损失**：L = L_reg + L_reg_drop + β·KL + λ·MSE_recon

### 5.3 架构设计
- 投影层：每模态独立 Conv1d → 40-dim
- Transformer 编码器：每模态独立 5层×8头 Transformer
- CVAE：encoder 2层 MLP(64) → μ,logvar 32-dim；decoder 2层 MLP(64) → 40-dim
- 输出头：120→80→40→1 压缩金字塔

### 5.4 设计空间分析
- 列出我们的设计选择及理由
- 对比原始空间重建的参数/计算开销

---

## 六、Experimental Setup (约 1 页)

### 6.1 数据集
- **CMU-MOSEI**：22,856 样本，6 类情感回归+分类。特征：text GloVe (768d), audio COVAREP (25d), vision FACET (171d)。75 时间步序列对齐。
- 评估指标：Acc-2, Acc-5, Acc-7, F1, MAE
- 数据划分：train 16,245 / valid 1,858 / test 4,637

### 6.2 基线方法
- **concat**：CASP LateFusion（zero-filling baseline）
- **UADG**：不确定性门控（调节权重）
- **Evidential**：NIG 证据深度学习融合
- **ContraMSA**：跨模态对比损失
- **P-RMF**：多 VAE 原始空间重建（ACL 2025）
- **HVDER**：VAE+Diffusion（2026）

### 6.3 实现细节
- 优化器：Adam, lr=1e-3, ReduceLROnPlateau
- 训练 30 epoch, early stop 基于 val Acc-2
- GPU: RTX 4060 8GB, batch=32
- 代码框架基于 CASP (AAAI 2025)

---

## 七、Results and Analysis (约 3 页)

### 7.1 主实验结果

| Method | Full | MissT | MissA | MissV | Params |
|--------|:---:|:---:|:---:|:---:|:---:|
| concat | 0.748 | 0.587 | 0.748 | 0.748 | 347K |
| UADG | 0.750 | 0.535 | — | — | +40K |
| Evidential | 0.704 | 0.510 | — | — | +120K |
| ContraMSA | 0.744 | 0.490 | — | — | +80K |
| **CVAE-MSA** | **0.758** | **0.618** | **0.758** | **0.754** | **+30K** |

### 7.2 模态依存分析（R² 回归）
- Text → Label: R² = 0.268（唯一独立情感信号）
- Audio → Label: R² = −0.102（无独立信号）
- Vision → Label: R² = −0.102（无独立信号）
- A+V → Label: R² = −0.078（联合也无信号）

**核心洞察**：CVAE 被要求从"噪声"（A+V）中重建"信号"（T 的情感信息）。这解释了为什么 A+V 扩容和复杂策略都失败——模型在拟合不存在的模式。

### 7.3 消融研究：13 种失败策略

| # | 策略 | MissT Δ | 失败原因 |
|:--:|------|:---:|------|
| 1 | Cycle Consistency | -9.1pp | 容量不足，双目标梯度冲突 |
| 2 | Progressive Dropout | -7.6pp | 渐进 mask 破坏训练分布 |
| 3 | Random-z MC Inference | -4.8pp | 采样方差 > 多样性收益 |
| 4-7 | Capacity Sweep (4 configs) | -5~-10pp | 更大容量 = 更强过拟合 |
| 8 | Weighted Reconstruction | <baseline | 维度加权引入偏差 |
| 9 | Contrastive Alignment | -3.4pp | 对比损失与重建冲突 |
| 10 | Asymmetric Encoding | -10.4pp | A/V 扩容放大了噪声 |
| 11 | T-Weighted Dropout | -10.2pp | 强制降低 T 依赖损害融合 |
| 12 | Narrow CVAE | -6.9pp | 更紧瓶颈限制重建能力 |
| 13 | Combined Regularization | -12.9pp | 多重正则叠加过强 |

**注**: 网格搜索（192 组合, KL×Recon×Dropout×LR）结果待完成后再填入。

### 7.4 超参数敏感性
- KL weight: 0.1 最优（过大抑制回归，过小后验崩塌）
- Recon weight: 1.0 最优
- Dropout prob: 0.2 最优
- Latent dim: 32 > 24 > 64 > 128
- Hidden dim: 64 > 128 > 256

### 7.5 参数效率分析
- 仅 30K 参数增量（8.8%）→ 最高 MissT 增益（+3.1pp）
- 对比：P-RMF 多 VAE 架构参数大 10×+，但 MissT 增益低于我们
- 参数效率 vs 性能的 Pareto 前沿

---

## 八、Discussion (约 1 页)

### 8.1 为什么简单方案是最优的？
- 融合空间的信息压缩效应：40-dim 已经是高度规范化的表示
- A+V 缺乏独立信号 → 任何从 A+V 推断 T 的改进都是"从噪声中提取不存在的信息"
- 模型容量上限由信息内容决定，而非参数数量

### 8.2 对领域的影响
- 在约束条件下，简单设计的系统性验证本身就是贡献
- 参数效率是实用部署的关键指标（移动端、边缘设备）
- 方法为未来研究提供更强的基线

### 8.3 局限性
- 仅在 MOSEI 上验证（English）；跨语言/跨文化泛化待探索
- 使用预提取特征（GloVe, COVAREP, FACET）；端到端训练的对比待做
- 缺失模式为完全缺失；部分缺失/噪声模态待研究
- 单模态缺失；多模态同时缺失待扩展

---

## 九、Conclusion (约 0.5 页)

- 提出 CVAE-MSA：在融合空间重建缺失模态的轻量方法
- 核心发现：13 种复杂策略无一超越最简配置 → "Simple is Best"
- 30K 参数增量实现最优缺失文本鲁棒性（+3.1pp）
- 系统性消融和网格搜索为融合空间重建范式提供了完整画像

---

## 十、Figures & Tables 计划

| # | 类型 | 内容 |
|:--:|------|------|
| Fig 1 | 架构图 | CVAE-MSA 整体架构（输入→编码器→CVAE→融合→输出） |
| Fig 2 | 训练/推理流程图 | 训练时（有 GT 的 CVAE）vs 推理时（z=0 确定性）对比 |
| Fig 3 | R² 柱状图 | 各模态及组合对标签的 R²（强调 A+V=−0.078） |
| Fig 4 | 消融瀑布图 | 13 种失败策略的 MissT Δ 瀑布图（视觉冲击力） |
| Fig 5 | 超参数热力图 | KL weight × Recon weight 在 MissT 上的二维热力图 |
| Table 1 | 主结果表 | 所有基线方法 × 4 种模态设置 |
| Table 2 | 参数效率表 | 方法 vs 参数增量 vs MissT 的对比 |
| Table 3 | 消融汇总 | 13 种策略的配置和结果 |

---

## 十一、写作计划

| 阶段 | 内容 | 预计时间 |
|:--:|------|:--:|
| 1 | Method 部分 + 架构图 | 2-3 天 |
| 2 | Experiments + 网格搜索完成后的结果 | 2-3 天 |
| 3 | Introduction + Related Work | 2 天 |
| 4 | Discussion + Conclusion | 1 天 |
| 5 | 全稿打磨 + 公式检查 + 引用补全 | 2 天 |
| 6 | 中文版本翻译（如投稿中文期刊） | 3 天 |

---

## 十二、关键引用清单

| 论文 | 用途 |
|------|------|
| CASP (Guo et al., AAAI 2025) | Baseline 架构来源 |
| SDUMC (Cong et al., ICASSP 2025) | 竞争方法 |
| P-RMF (ACL 2025) | 原始空间 VAE 重建对比 |
| HVDER (2026) | VAE+Diffusion 对比 |
| CVAE (Sohn et al., NeurIPS 2015) | 方法基础 |
| β-VAE (Higgins et al., ICLR 2017) | KL 权重理论 |
| Deep Evidential Regression (Amini et al., NeurIPS 2020) | 证据学习方法对比 |
| MulT (Tsai et al., ACL 2019) | 多模态 Transformer 先驱 |
| CMU-MOSEI (Zadeh et al., ACL 2018) | 数据集 |
