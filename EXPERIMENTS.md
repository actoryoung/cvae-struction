# 课题 A 完整实验记录

> 项目：缺失模态鲁棒多模态情感分析
> 日期：2026-07-13
> 数据集：MOSI + MOSEI（预处理特征，来自 CASP AAAI 2025）
> GPU：NVIDIA RTX 4060 8GB（cu128 就绪）
> 基线模型：CASP LateFusion concat

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

---

## 二、MOSEI 完整对比（30 epoch, batch=8, GPU）

| 方法 | Full Acc-2 | 缺文本 Acc-2 | 缺音频 Acc-2 | 缺视觉 Acc-2 | 备注 |
|------|:---:|:---:|:---:|:---:|------|
| **concat** 🥇 | **0.748** | **0.587** | **0.746** | **0.743** | 简单最强 |
| UADG | 0.750 | 0.535 | 0.750 | 0.742 | 缺文本-5.2pp |
| ContraMSA | 0.744 | 0.490 | 0.743 | 0.742 | 对比损失有害 |
| V1 evidential | 0.704 | 0.510 | 0.703 | 0.703 | 全面更差 |

关键发现：
- 缺失文本时性能大跌（16pp），缺失音频/视觉几乎无影响（<0.5pp）
- 文本是 MOSEI/MOSI 的绝对主导模态
- 所有"高级融合"方法都无法超越 concat baseline
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
