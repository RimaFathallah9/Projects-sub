# Report outline

**Facial Expression Recognition Using Deep Learning: A Comparative Study of CNN Architectures and Transfer Learning**

Write each section only from real outputs. Items marked (data) are filled from files, never from memory.

1. **Abstract** - problem, three models, protocol, headline numbers (data), main finding, limitation.
2. **Introduction** - FER as a pattern-recognition task; why FER-2013 is hard (noise, imbalance); contributions; explicit non-claim about emotion.
3. **Related work** - see references below (read each before citing and check details).
4. **Dataset & EDA** - sizes, class counts, duplicates/leakage removed, conflicting labels, quality outliers (`reports/eda.json`, figures).
5. **Methodology** - splits, leakage control, metrics, selection on val, multi-seed protocol, statistical caution.
6. **Model architectures** - Custom CNN diagram + rationale; ResNet18/MobileNetV2 adaptations; input protocols.
7. **Experimental setup** - hardware (RTX 4050 Laptop 6 GB), software versions (`summary.json`), hyper-parameters, sweep design.
8. **Results & discussion** - ablation tables (sweeps), final comparison, efficiency (params/MACs/latency), per-class results, imbalance analysis.
9. **Explainability & error analysis** - Grad-CAM galleries, randomisation sanity check, border mass, confident errors vs label noise.
10. **Limitations & ethics** - label noise, no subgroup data, construct validity of "emotion", dual-use risks, consent in the demo.
11. **Conclusion & future work** - e.g. FER+ relabelling, other datasets for cross-dataset testing, calibration, subgroup audits.
12. **References**

## Candidate references (real papers; verify bibliographic details before use)
- Goodfellow et al., "Challenges in Representation Learning: A report on three machine learning contests" (FER-2013 origin), arXiv:1307.0414.
- Barsoum et al., "Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution" (FER+), ICMI 2016, arXiv:1608.01041.
- Khaireddin & Chen, "Facial Emotion Recognition: State of the Art Performance on FER2013", arXiv:2105.03588.
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018. Howard et al., "MobileNets", arXiv:1704.04861.
- Ioffe & Szegedy, "Batch Normalization", ICML 2015. Srivastava et al., "Dropout", JMLR 2014.
- Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS 2014.
- Selvaraju et al., "Grad-CAM", ICCV 2017. Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018.
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019.
- Barrett et al., "Emotional Expressions Reconsidered: Challenges to Inferring Emotion from Human Facial Movements", Psychological Science in the Public Interest, 2019.
