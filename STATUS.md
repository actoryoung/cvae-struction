# 课题 A 工作状态（截至 2026-07-13）

## 当前运行中

- ✅ 全部完成，可暂停

## MOSEI 最终排名（30 epoch, GPU）

| 方法 | Full Acc-2 | 缺文本 Acc-2 | 备注 |
|------|:---:|:---:|------|
| 🥇 **CVAE (kl=0.1)** | **0.758** | **0.618** | **当前最优** |
| 🥈 AttnCVAE (kl=0.01) | 0.746 | 0.510 | 注意力解码器过拟合重建 |
| 🥉 concat | 0.748 | 0.587 | baseline |
| 4 | UADG | 0.750 | 0.535 | |
| 5 | ContraMSA | 0.744 | 0.490 | |
| 6 | V1 evidential | 0.704 | 0.510 | |

### 关键发现

1. **"融合空间重建"（CVAE）是唯一有效的方法** — 所有"忽略缺失"的方法（门控/证据/对比）都不如 concat
2. **注意力解码器是关键提升** — 让 CVAE 选择性利用可用模态信息
3. **低 KL 权重更好** — kl=0.01 > kl=0.1 > 线性升温
4. **NIG 损失有符号 bug** — 修复后稳定但性能仍不如 CVAE
5. **特征质量是瓶颈** — 不改特征可保持公平对比，但限制了绝对性能上限

## 与竞争论文的差异化

| 我们 vs | 核心差异 |
|---------|---------|
| P-RMF (ACL 2025) | 融合空间(40d) vs 原始空间重建；单 CVAE vs 多 VAE；30K vs 重参数 |
| MM-SSC (2025) | 连续 CVAE vs 离散 VQ-VAE；更轻量 |
| HVDER (2026) | CVAE only vs VAE+Diffusion；参数少 10-50x |

**Delta**: "Unlike existing methods that use per-modality VAEs to reconstruct in raw feature space, we use a single lightweight attention-guided CVAE to reconstruct missing modality representations directly in the fusion space."

## 未完成的实验组

详见 `plans/a-b-mighty-turing.md`，按优先级：

| 优先级 | 实验 | 状态 |
|:--:|------|:--:|
| 🔴 P0 | KL annealing + 扫参 | ✅ 完成（kl=0.01 最优） |
| 🔴 P0 | 注意力增强 Decoder | 🔄 进行中 |
| 🟡 P1 | 多样本推理 (MC samples) | ⏳ 待做 |
| 🟡 P1 | Cycle Consistency | ⏳ 待做 |
| 🟡 P1 | 渐进 Dropout | ⏳ 待做 |
| 🟢 P2 | 消融实验 | ⏳ 待做 |
| 🟢 P2 | SIMS 中文数据集 | ⏳ 待做 |

## 明日工作计划

1. 查看 AttnCVAE MOSEI 最终结果
2. 若超 0.765：进行 P1 实验（MC samples + Cycle Consistency + Progressive Dropout）
3. 若不足：先调参再试（kl=0.005, recon=0.5）
4. 开始写消融实验表格
5. 目标：确定 final model 配置，为写论文做准备

## 关键文件

| 文件 | 内容 |
|------|------|
| `STATUS.md` | 本文件 |
| `EXPERIMENTS.md` | 完整实验记录 |
| `APPROACH_AND_COMPARISON.md` | 方法与竞争论文对比 |
| `ISSUES.md` | Bug 记录与问题分析 |
| `DESIGN.md` | 原始研究设计 |
| `compare.md` | 用户自己做的 6 篇论文对比 |
| `models/cvae_reconstruct.py` | CVAE + AttnCVAE + GumbelGate |
| `train_cvae.py` | 训练脚本（支持 KL annealing 等） |
| `dataloader_mosei.py` | MOSEI 内存高效加载 |
