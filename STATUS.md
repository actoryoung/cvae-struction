# 课题 A 工作状态（截至 2026-07-17）

## 实验完成总结

| 阶段 | 数据 | 实验数 | 结论 |
|------|:--:|:--:|------|
| Smart Sweep | MOSEI | 30 | KL 解释 63% 方差，KL=0.5 最优 |
| Combo Sweep | MOSEI | 21 | 策略叠加不及纯 KL 调参 |
| Fill-in | MOSEI | 6 | KL=0.4+CT0.7 为全局最优 |
| MOSI Test | MOSI | 9 | CVAE MissT 不及 concat（数据量瓶颈） |

**总计：66 次训练实验，覆盖 2 个数据集。**

---

## 最终最佳模型

| 属性 | 值 |
|------|-----|
| 配置 | KL=0.4, RW=1.0, Contrastive cw=0.7, DP=0.2, LR=0.001 |
| 架构 | CVAE-MSA (latent=32, hidden=64, proj_dim=40) |
| 参数 | 377,737 (+ projection head ~80K for contrastive) |

### MOSEI 最终结果

| Setting | concat (CASP) | **CVAE-MSA (Ours)** |
|---------|:---:|:---:|
| Full | 0.748 | **0.752** |
| **Missing text** | 0.587 | **0.597** (+1.0pp) |
| Missing audio | 0.746 | 0.751 |
| Missing vision | 0.743 | 0.744 |

### MOSI 结果

| Setting | concat | CVAE-MSA |
|---------|:---:|:---:|
| Full | 0.767 | **0.797** |
| Missing text | **0.599** | 0.565 |
| MissA/MissV | 0.767/0.766 | 0.797/0.797 |

---

## 核心发现

1. **KL weight 是主导超参**：解释 MissT 方差的 63%
2. **KL×策略交互**：MC/Contrastive 在低 KL 时增益大，高 KL 时趋于零
3. **β-U 曲线双峰**：KL=0.4 和 0.8 均为峰值，0.6 意外低
4. **策略叠加拮抗**：MC+Contrastive 合并比单独用更差
5. **数据集规模效应**：MOSEI (16K) 最优 KL=0.4-0.8；MOSI (1.3K) 需要 KL→0

## 论文写作状态

| 文档 | 路径 | 状态 |
|------|------|:--:|
| 最佳模型规格 | `paper/PAPER_MODEL.md` | ✅ 包含 MOSEI+MOSI 完整结果 |
| 超参分析 | `paper/HYPERPARAM_ANALYSIS.md` | ✅ 包含跨数据集分析 |
| 论文大纲 | `paper/PAPER_OUTLINE.md` | ⚠️ 需更新叙事 |
| 论文正文 | `paper/paper_draft.md` | 🔄 待填充实验数据 |

## 下一步

| 任务 | 状态 |
|------|:--:|
| 撰写论文正文 | ⏳ |
| CH-SIMS 中文数据集（可选） | ⏳ |
| 多 seed 验证最终配置 | ⏳ |
