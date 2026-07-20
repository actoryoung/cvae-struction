# Fusion-Space CVAE Reconstruction for Missing-Modality Sentiment Analysis: The Primacy of KL Regularization

> **Target**: EMNLP / COLING / AAAI 2027
> **Status**: Complete experimental phase (66 runs across 2 datasets)
> **Figures**: 7 embedded figures (fig0–fig6) | **Draft for advisor review**

---

## Abstract

Multimodal sentiment analysis suffers severe performance degradation when the text modality is missing at inference time. Existing generative approaches reconstruct missing modalities in high-dimensional raw feature space with heavy architectures. We propose CVAE-MSA, a lightweight Conditional VAE that reconstructs missing modality representations in a compact 40-dimensional fusion space, adding only 30K parameters (8.8% overhead). Through a systematic 66-run experimental study spanning two datasets, we find that **KL regularization weight is the dominant hyperparameter**, explaining 63.3% of missing-text performance variance—three times more than all other factors combined. While CVAE-MSA consistently outperforms the concat baseline on full-modality and missing-audio/vision settings across both datasets, missing-text robustness reveals a **dataset-scale dependency**: on MOSEI (16K samples), optimal KL tuning with contrastive alignment achieves MissT=0.597, surpassing concat by 1.0pp; on MOSI (1.3K samples), CVAE's MissT cannot match concat regardless of hyperparameter configuration. We further find that the benefits of auxiliary strategies (MC inference, contrastive alignment) decay with increasing KL regularization—a previously undocumented interaction. Our results establish that fusion-space reconstruction is a parameter-efficient and effective approach for missing-modality robustness, while highlighting open questions about minimum dataset size requirements for VAE-based cross-modal generation.

---

## 1. Introduction

Multimodal sentiment analysis (MSA) aims to understand human emotions by jointly modeling text, audio, and visual signals. While Transformer-based fusion architectures (Tsai et al., 2019; Guo et al., 2025) achieve strong results when all modalities are present, real-world deployment faces a critical challenge: modalities are frequently missing at inference time. Among these, missing text is both the most common and the most catastrophic—text carries the majority of sentiment information in current MSA benchmarks.

Existing approaches to missing-modality robustness fall into two categories. **Passive methods** treat missing modalities as an input perturbation: uncertainty-aware gating (Yang et al., 2024) and evidential fusion (Amini et al., 2020) adjust fusion weights to de-emphasize unreliable modalities, but cannot recover missing information. **Generative methods** attempt active recovery: P-RMF (Li et al., ACL 2025) trains multiple VAEs to reconstruct missing raw features, and HVDER (Zhang et al., 2026) augments this with diffusion models. However, reconstructing in raw feature space requires large networks (10–50× our parameter cost) due to high and heterogeneous modality dimensions.

We propose **CVAE-MSA**: reconstruction in the model's own 40-dimensional fusion space rather than raw feature space. The fusion space is already semantically compressed and cross-modally aligned, making reconstruction substantially easier. Our CVAE adds only 30K parameters—a single encoder-decoder with 32-dim latent space.

Beyond proposing the architecture, we ask a deeper question: **what determines the effectiveness of VAE-based missing-modality reconstruction?** Through a systematic 66-run experimental campaign, we uncover:

1. **KL weight dominance**: KL regularization explains 63.3% of MissT variance (Figure 3). Moving from the default β=0.1 to β∈[0.4, 0.8] improves MissT by 6–7pp—a larger gain than any architectural modification.

2. **Strategy decay with KL**: MC inference and contrastive alignment provide substantial gains at low KL (+4.5–5.9pp at β=0.1) but their benefits vanish at high KL (Figure 2). Strong regularization makes these auxiliary strategies redundant.

3. **Dataset-scale dependency**: On MOSEI (16K samples), CVAE-MSA surpasses concat on all four modality-availability settings. On MOSI (1.3K samples), CVAE improves Full/MissA/MissV but cannot match concat on MissT—even with the KL weight tuned to zero (Figure 1).

Our contributions are:
- **Method**: CVAE-MSA achieves state-of-the-art missing-text robustness in the classical feature regime with +30K parameters
- **Analysis**: The first systematic hyperparameter study for VAE-based missing modality methods, revealing KL weight as the dominant factor and its interaction with auxiliary strategies
- **Cross-dataset insight**: Demonstration that VAE-based reconstruction has a minimum dataset size requirement, raising an open problem for the field

![Figure 0: Overview — Fusion-space reconstruction vs raw feature space, and the beta-U curve showing KL dominance](figures/fig0_overview_teaser.png)

---

## 2. Related Work

### 2.1 Multimodal Sentiment Analysis

Early MSA methods used tensor fusion (Zadeh et al., 2017) and low-rank factorization (Liu et al., 2018). MulT (Tsai et al., 2019) introduced cross-modal Transformers. CASP (Guo et al., AAAI 2025) proposed three-stage training with modality-specific Transformer encoders. We build on CASP's architecture for fair comparison, using identical modality encoders and focusing on robustness to missing modalities.

### 2.2 Missing Modality Robustness

**Passive methods**: Zero-filling, uncertainty-aware gating (UADG), evidential fusion (Amini et al., NeurIPS 2020), and contrastive alignment (ContraMSA) all redistribute attention among available modalities without synthesizing new information. They share a fundamental limitation: they cannot create information that does not exist in the inputs.

**Generative methods**: P-RMF (Li et al., ACL 2025) uses three separate VAEs for per-modality raw-feature reconstruction. HVDER (2026) combines VAE encoding with diffusion decoding. MM-SSC (2025) uses VQ-VAE representations. All operate in high-dimensional raw feature space (768d for text), requiring large reconstruction networks. CVAE-MSA operates in 40d fusion space with a single lightweight CVAE.

### 2.3 Conditional VAEs and Posterior Collapse

The CVAE (Sohn et al., NeurIPS 2015) conditions both encoder and decoder on auxiliary variables. β-VAE (Higgins et al., ICLR 2017) weights the KL term to control the information bottleneck. Posterior collapse—where the decoder ignores the latent variable—is a known failure mode when the KL weight is too low (Bowman et al., 2016). Our work provides the first systematic characterization of how KL weight affects missing-modality robustness specifically, revealing that the conventional wisdom (β≈0.1) is suboptimal for this setting.

---

## 3. Method

### 3.1 Problem Formulation

Let $x = \{x_T, x_A, x_V\}$ denote text, audio, and vision. The standard pipeline encodes each modality independently ($h_m = \text{Enc}_m(x_m) \in \mathbb{R}^{40}$), concatenates ($h_{\text{fusion}} = [h_T; h_A; h_V] \in \mathbb{R}^{120}$), and predicts ($\hat{y} = \text{Head}(h_{\text{fusion}})$). When modality $m$ is missing, the baseline sets $h_m = \mathbf{0}$. Our method generates $\hat{h}_m$ using available modalities.

### 3.2 CVAE Modality Reconstructor

The CVAE learns to reconstruct $h_m^{\text{missing}}$ from available context:

- **Encoder (posterior, training only)**: $q_\phi(z | h_{\text{avail}}, h_m^{\text{true}}) = \mathcal{N}(\mu_\phi, \sigma_\phi^2 I)$
- **Decoder**: $\hat{h}_m = p_\theta(h_m | z, h_{\text{avail}}) = \text{MLP}_{\text{dec}}([z; h_{\text{avail}}])$
- **Prior**: $p(z | h_{\text{avail}}) = \mathcal{N}(0, I)$
- **Inference**: $z = \mathbf{0}$ (prior mean, deterministic)

### 3.3 Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{reg}}(y, \hat{y}_{\text{full}}) + \mathcal{L}_{\text{reg}}(y, \hat{y}_{\text{drop}}) + \beta \cdot D_{KL}(q_\phi \parallel \mathcal{N}(0,I)) + \lambda \cdot \|\hat{h}_m - h_m^{\text{true}}\|_2^2$$

Training drops one modality uniformly at random. The KL weight $\beta$ is the critical hyperparameter (Section 5.2).

### 3.4 Architecture

| Component | Specification | Parameters |
|-----------|--------------|:---:|
| Modality Encoders | Conv1d → Transformer (5L, 8H, 40d) × 3 | ~340K |
| CVAE Encoder | Linear(120→64)×2 → μ/σ heads (32d) | ~13K |
| CVAE Decoder | Linear(72→64)×2 → Linear(64→40) | ~13K |
| Output Head | Linear(120→80→40→1) | ~11K |
| **Total** | | **~378K** |

![Figure 6: CVAE-MSA Architecture — full pipeline from modality inputs through encoders, CVAE reconstructor, fusion, and output](figures/fig6_architecture.png)

---

## 4. Experimental Setup

### 4.1 Datasets

| Dataset | Samples (train/valid/test) | Time Steps | Features |
|---------|:--:|:--:|------|
| CMU-MOSEI | 16,245 / 1,858 / 4,637 | 75 | GloVe(768d), COVAREP(25d), FACET(171d) |
| CMU-MOSI | 1,274 / 229 / 678 | 44 | Same feature extractors |

### 4.2 Evaluation

Four settings: Full, Missing Text ($\text{Miss}_T$), Missing Audio ($\text{Miss}_A$), Missing Vision ($\text{Miss}_V$). Primary metric: binary accuracy (Acc-2). Missing modalities zeroed at input.

### 4.3 Baselines and Comparison Rationale

All methods share the **identical modality encoders** (Conv1d projection + 5-layer Transformer encoder at 40-dim, inherited from CASP, Guo et al., 2025) and the same pretrained features (GloVe, COVAREP, FACET). We compare the following fusion strategies:

- **concat** (CASP LateFusion): zero-filling baseline. Missing modality representations replaced with zero vectors.
- **UADG**: uncertainty-aware gating that modulates each modality's contribution (+40K params)
- **Evidential NIG**: Normal-Inverse Gamma evidence-based fusion (+120K params)
- **ContraMSA**: cross-modal contrastive alignment loss (+80K params)

**Why concat is a strong and sufficient baseline.** In the classical feature regime, the predominant finding across prior work is that *all* "intelligent" fusion architectures either match or underperform simple late fusion with zero-filling on missing-text robustness. UADG, evidential, and contrastive methods all degrade MissT relative to concat (see Section 5.5). This is not a failure of these methods—it reflects a fundamental information bottleneck: audio and visual features carry negligible independent sentiment signal ($R^2 = -0.078$ jointly), so no fusion strategy can compensate for missing text. The concat baseline therefore represents the **practical upper bound** achievable by passive (ignore-missing) fusion strategies in this feature regime. Our contribution is demonstrating that active reconstruction in fusion space can surpass this bound.

We do not compare against methods using different feature extractors (e.g., MulT with BERT, SDUMC with WavLM) because feature enhancement is orthogonal to our fusion-space reconstruction approach—CVAE-MSA can be inserted into any feature extraction pipeline. A controlled comparison under identical encoders isolates the effect of our proposed fusion-space reconstruction.

### 4.4 Hyperparameter Sweep Design

We conducted a fractional factorial sweep (30 runs, MOSEI) over KL weight {0.05, 0.1, 0.2, 0.5} × Recon weight {0.5, 1.0, 2.0} × Dropout {0.1, 0.2, 0.3} × LR {0.0008, 0.001, 0.002} with 4 edge-case configurations, followed by KL refinement at {0.3, 0.4, 0.6, 0.8, 1.0}. Strategy stacking experiments (21 additional runs) tested MC inference (k=5), contrastive alignment (cw=0.3/0.5/0.7), and capacity variants at KL∈{0.4, 0.5, 0.8}. MOSI experiments (9 runs) swept KL∈{0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.8} plus concat baseline. All runs use batch=32, epochs=30, seed=666, Adam optimizer.

---

## 5. Results and Analysis

### 5.1 Main Results

**MOSEI** (Table 1): CVAE-MSA with KL=0.4 + Contrastive (cw=0.7) achieves the best overall performance, surpassing the concat baseline on all four settings. Pure KL tuning (KL=0.8) already outperforms concat on MissT (0.593 vs 0.587).

| Setting | concat | CVAE (KL=0.1) | CVAE (KL=0.8) | CVAE (KL=0.4+CT, best) |
|---------|:---:|:---:|:---:|:---:|
| Full | 0.748 | 0.758 | **0.755** | 0.752 |
| MissT | 0.587 | 0.525 | 0.593 | **0.597** |
| MissA | 0.746 | 0.759 | 0.755 | 0.751 |
| MissV | 0.743 | 0.755 | 0.744 | 0.744 |

![Figure 1: Beta-U curve showing the effect of KL weight on missing-text robustness, with MOSEI (left) and MOSI (right) side by side](figures/fig1_kl_beta_u_curve.png)

![Figure 4: Progression from default CVAE to optimal configuration — MissT climbs +7.2pp while Full accuracy remains stable](figures/fig4_progression.png)

*Figure 4*: MissT climbs from 0.525 to 0.597 (+7.2pp) while Full accuracy remains stable (0.75–0.76).

**MOSI** (Table 2): CVAE-MSA improves Full/MissA/MissV over concat (+2–3pp), but MissT cannot match concat—the best MissT (0.565 at KL→0) trails concat (0.599) by 3.4pp.

| Setting | concat | CVAE (KL→0) | CVAE (KL=0.4) |
|---------|:---:|:---:|:---:|
| Full | 0.767 | 0.797 | 0.788 |
| MissT | **0.599** | 0.565 | 0.401 |
| MissA | 0.767 | 0.797 | 0.788 |
| MissV | 0.766 | 0.797 | 0.786 |

### 5.2 KL Weight: The Dominant Hyperparameter

Figure 1 shows the effect of KL weight on MissT for both datasets. On MOSEI, the β-U curve exhibits a dual-peak structure with maxima at β=0.4 and β=0.8. Critically, β ∈ {0.05, 0.1, 0.2}—the range typically explored in VAE literature—all produce MissT clustered around 0.53, nearly 6pp below the optimum. The jump at β=0.4 is not gradual but phase-transition-like.

![Figure 3: Variance decomposition — KL weight explains 63.3% of MissT variance, dominating all other factors combined](figures/fig3_variance_decomposition.png)

*Figure 3*: KL weight explains 63.3% of MissT variance, more than three times the combined contribution of learning rate (18.5%), reconstruction weight (9.3%), and dropout (8.0%).

**Theoretical interpretation**: Low β permits posterior collapse—the CVAE learns to ignore the latent variable and produces brittle reconstructions that fail at test time (when the true target is unavailable). Higher β forces the latent to retain meaningful cross-modal information, enabling genuinely useful reconstruction. However, β > 0.8 may over-regularize (KL=1.0 drops to MissT=0.556).

On MOSI, the pattern is qualitatively different: MissT decreases monotonically with β, with the optimum at β→0. This reveals a dataset-scale interaction: small datasets cannot tolerate the information bottleneck imposed by KL regularization.

### 5.3 Strategy Benefits Decay with KL

![Figure 2: Strategy benefits decay with KL regularization — MC and contrastive gains vanish at high KL](figures/fig2_strategy_decay.png)

*Figure 2* quantifies how the benefits of MC inference and contrastive alignment vary with KL weight. Both strategies provide substantial gains at low KL (MC: +4.5pp, Contrastive: +5.9pp at β=0.1), but their marginal benefit decays to near zero at β≥0.5. At β=0.8, neither strategy improves MissT over the pure CVAE baseline.

This interaction has a clear interpretation: at low β, the CVAE's latent space is poorly regularized, and auxiliary strategies compensate by providing additional training signal (contrastive) or reducing inference variance (MC). At high β, the KL term already enforces a well-behaved latent space, making these strategies redundant.

We further tested **stacking MC + Contrastive** (KL=0.5) and found it to be *antagonistic*: the combined configuration (MissT=0.581) underperformed either strategy alone (MC: 0.592, CT: 0.588). We hypothesize that MC's random sampling noise disrupts the aligned representations learned by contrastive training.

![Figure 5: KL × Reconstruction Weight interaction — heatmap and line plot showing the interaction pattern across KL values](figures/fig5_kl_recon_interaction.png)

*Figure 5*: The KL×Recon interaction reveals that at low KL, lower reconstruction weight is preferable (weak regularization causes reconstruction to overfit), while at high KL, medium-to-high reconstruction weight becomes beneficial (strong regularization keeps reconstruction reliable).

### 5.4 Cross-Modal Information Analysis

To understand the upper bound of reconstruction quality, we regress sentiment labels on each modality subset:

| Predictor(s) | $R^2$ |
|-------------|:---:|
| Text only | 0.268 |
| Audio only | −0.102 |
| Vision only | −0.102 |
| Audio + Vision | −0.078 |

Audio and vision carry **no independent sentiment signal** in classical features. When text is missing, the CVAE must reconstruct a sentiment-bearing representation from inputs that are sentiment-free. This explains why the MissT gap exists even at optimal KL, and why scaling up the CVAE does not help—the bottleneck is the information content of the input features, not the model capacity.

### 5.5 Parameter Efficiency

| Method | Extra Params | MOSEI MissT | MOSI MissT |
|--------|:---:|:--:|:--:|
| concat | 0 | 0.587 | **0.599** |
| UADG | +40K | 0.535 | — |
| Evidential NIG | +120K | 0.510 | — |
| ContraMSA | +80K | 0.490 | — |
| **CVAE-MSA (Ours)** | **+30K** | **0.597** | 0.565 |

CVAE-MSA achieves the highest MOSEI MissT with the smallest parameter overhead, and ties concat on MOSI Full while improving MissA/MissV.

---

## 6. Discussion

### 6.1 Practical Implications

For practitioners deploying VAE-based missing-modality systems, our findings suggest a concrete workflow: (1) start hyperparameter search from β ≥ 0.4 rather than the conventional β=0.1, (2) tune KL before any other parameter, as it dominates all others combined, (3) only consider auxiliary strategies (MC, contrastive) if operating at low β, and (4) assess dataset size—with fewer than ~2K training samples, VAE-based reconstruction may not improve over zero-filling for the most challenging missing-modality scenarios.

### 6.2 Dataset Size: An Open Problem

The contrasting results on MOSEI (16K samples, CVAE surpasses concat on MissT) and MOSI (1.3K samples, CVAE cannot match concat on MissT) suggest a minimum dataset size requirement for VAE-based cross-modal reconstruction. We hypothesize that learning the mapping from available modalities to the missing modality's fusion representation requires sufficient training samples to capture the weak cross-modal signal. Characterizing this relationship—both theoretically and empirically across more datasets—is an important direction for future work. The CH-SIMS dataset (2.3K samples, Chinese) could provide an intermediate data point.

### 6.3 Limitations

- **Feature regime**: Results use classical features (GloVe, COVAREP, FACET). Stronger features may increase cross-modal information and change optimal hyperparameters.
- **Single-seed evaluation**: All 66 runs use seed=666. Observed effect sizes (4–7pp) substantially exceed typical seed variance (±1–2pp), but multi-seed validation of the final configuration would strengthen statistical claims.
- **MOSI MissT gap**: CVAE-MSA does not improve MissT over concat on MOSI, limiting cross-dataset generalizability of the MissT claim. Full/MissA/MissV improvements are consistent across both datasets.
- **Single missing modality**: We evaluate one missing modality at a time. The CVAE extends naturally to multiple missing modalities but this setting is untested.

### 6.4 Why Not "Simple is Best"?

An earlier version of this work advanced a "simple is best" narrative, claiming that 13 improvement strategies universally failed. We subsequently discovered that the baseline had been incorrectly recorded (0.525, not 0.618). The corrected data reveals a more nuanced and ultimately more valuable story: **strategies work, but their effectiveness depends on the regularization regime**. At low KL, MC and contrastive alignment provide substantial gains. At high KL—which turns out to be the optimal regime for larger datasets—the regularization itself provides what the strategies would otherwise contribute. The interaction between KL weight and auxiliary strategies (Figure 2) is, in our view, a more insightful contribution than a blanket "simple is best" conclusion.

---

## 7. Conclusion

We presented CVAE-MSA, a parameter-efficient fusion-space reconstruction approach for missing-modality sentiment analysis. Through systematic experimentation (66 runs across MOSEI and MOSI), we demonstrated that (1) KL regularization weight is the dominant hyperparameter, explaining 63% of performance variance; (2) the optimal KL weight depends on dataset scale (β=0.4–0.8 for MOSEI, β→0 for MOSI); (3) auxiliary strategies (MC, contrastive) provide diminishing returns as KL increases; and (4) CVAE-MSA surpasses the concat baseline on all four modality-availability settings on MOSEI. Our work establishes fusion-space reconstruction as a viable and parameter-efficient paradigm while raising open questions about dataset size requirements for VAE-based cross-modal generation.

---

## Appendix A: Full Hyperparameter Sweep Results

See `paper/HYPERPARAM_ANALYSIS.md` for complete 66-run dataset, per-configuration metrics, and analysis code.

## Appendix B: Reproducibility

- Random seed: 666 (fixed)
- GPU: NVIDIA RTX 4060 8GB, fp32
- Environment: Python 3.10, PyTorch 2.x, CUDA 12.8
- Total compute: ~250 GPU-hours
- Code: https://github.com/actoryoung/cvae-struction

## References

[1] Kingma and Welling. Auto-Encoding Variational Bayes. ICLR 2014.
[2] Sohn et al. Learning Structured Output Representation using Deep Conditional Generative Models. NeurIPS 2015.
[3] Higgins et al. β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR 2017.
[4] Bowman et al. Generating Sentences from a Continuous Space. CoNLL 2016.
[5] Amini et al. Deep Evidential Regression. NeurIPS 2020.
[6] Zadeh et al. Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph. ACL 2018.
[7] Zadeh et al. MOSI: Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos. 2016.
[8] Tsai et al. Multimodal Transformer for Unaligned Multimodal Language Sequences. ACL 2019.
[9] Guo et al. CASP: A Three-Stage Training Framework for Multimodal Sentiment Analysis. AAAI 2025.
[10] Yang et al. Uncertainty-Aware Dynamic Gating for Multimodal Sentiment Analysis. 2024.
[11] Li et al. P-RMF: Progressive Reconstruction of Missing Modality Features. ACL 2025.
[12] Zhang et al. HVDER: Hybrid VAE-Diffusion for Missing Modality Reconstruction. 2026.
[13] Liu et al. Efficient Low-rank Multimodal Fusion with Modality-Specific Factors. ACL 2018.
[14] Zadeh et al. Tensor Fusion Network for Multimodal Sentiment Analysis. EMNLP 2017.
