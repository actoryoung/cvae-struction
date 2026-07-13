# 课题 A：证据驱动的多模态情感分析鲁棒融合

> Evidential Deep Learning for Robust Multimodal Sentiment Analysis under Missing Modalities

## 快速开始

### 0. 环境

```bash
pip install torch numpy scikit-learn
```

### 1. 下载数据集

从 CASP Google Drive 下载预处理数据：
https://drive.google.com/file/d/1tQSw1S16ujHQ069W3QTi3BJ49Q8Gya8N/view

```bash
unzip data.zip -d data/
```

### 2. 模型检查

```bash
# UADG (不确定性门控 V1)
python models/uadg.py

# Evidential (证据深度学习门控 V2 — 主攻)
python models/evidential_uadg.py
```

### 3. 训练

```bash
# Baseline: concat fusion (CASP LateFusion)
python train_evidential.py --mode concat --dataset mosei --datapath data/mosei.pkl \
    --num_epochs 30 --name checkpoints/baseline.pt

# V2 主实验: Evidential fusion
python train_evidential.py --mode evidential --dataset mosei --datapath data/mosei.pkl \
    --num_epochs 30 --name checkpoints/evidential.pt

# V2 + evidence regularizer (推荐)
python train_evidential.py --mode evidential --reg_weight 1.0 --dataset mosei \
    --datapath data/mosei.pkl --num_epochs 30 --name checkpoints/evidential_reg.pt

# V2 + modality dropout
python train_evidential.py --mode evidential --use_dropout --reg_weight 0.1 \
    --dataset mosei --datapath data/mosei.pkl --num_epochs 30 --name checkpoints/evidential_drop.pt

# V1: Uncertainty gating
python train_uadg.py --gating_mode uncertainty --dataset mosei --datapath data/mosei.pkl \
    --num_epochs 30 --name checkpoints/uadg.pt
```

## 方法对比

| 方法 | 门控机制 | 理论支撑 | 缺失模态处理 | 参数量 |
|------|---------|---------|-------------|--------|
| concat (baseline) | 静态拼接 | 无 | 补零 | ~364K |
| UADG (V1) | exp(-σ) softmax | 经验性 | 权重归零 | ~375K |
| **Evidential (V2)** | **ν / Σν (证据比)** | **EDL 理论** | **证据为零** | **~365K** |

## 核心创新 (V2: Evidential Fusion)

基于 **Deep Evidential Regression** (Amini et al., NeurIPS 2020)：

- 每个模态编码器输出 NIG 分布参数：**(γ, ν, α, β)**
  - γ_m: 模态特定的情感预测
  - **ν_m: 虚拟证据量** — 核心创新点
- 融合权重：**w_m = ν_m / Σ ν_j**
  - 证据越多的模态权重越大
  - 缺失模态证据为 0 → 权重自动归零
- 损失函数：NIG negative log-likelihood + 证据正则化

**理论优势**：ν 有严格的概率解释——等价于"支持当前预测的虚拟观测数量"

## 目录结构

```
missing-modality-msa/
├── README.md              # 本文件
├── DESIGN.md              # 完整研究设计文档
├── train_uadg.py          # V1 训练脚本 (uncertainty gating)
├── train_evidential.py    # V2 训练脚本 (evidential fusion) ★ 主攻
├── models/
│   ├── uadg.py            # V1: 不确定性门控模型
│   └── evidential_uadg.py # V2: 证据深度学习门控模型 ★ 主攻
├── CASP/                  # CASP (AAAI 2025) 参考代码
├── SDUMC/                 # SDUMC (ICASSP 2025) 参考代码
├── data/                  # 数据集 (需下载)
├── checkpoints/           # 模型保存
└── results/               # 实验结果
```

## 实验矩阵

| 模型 | MAE (完整) | MAE (缺文本) | MAE (缺音频) | MAE (缺视觉) | Acc-2 |
|------|-----------|-------------|-------------|-------------|-------|
| concat (baseline) | - | - | - | - | - |
| UADG uncertainty (V1) | - | - | - | - | - |
| **Evidential (V2)** | **-** | **-** | **-** | **-** | **-** |
| + evidence reg | - | - | - | - | - |
| + modality dropout | - | - | - | - | - |

## 强化路线

| 方向 | 状态 | 描述 |
|------|------|------|
| **方向 1: 证据深度学习门控** | 🔥 主攻 | 基于 EDL，用 NIG 分布证据量做融合权重 |
| 方向 3: 生成式+判别式联合 | 💾 备选 | 轻量 CVAE 在融合空间中重建缺失模态 |
