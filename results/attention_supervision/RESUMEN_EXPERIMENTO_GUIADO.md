# Attention supervision control (ResNet-50, seed 42)

Generated: 2026-09-17 14:21

EP is CAM-based (layer4, 7x7, soft mask, predicted class) and is not comparable with the Grad-CAM audit.

| Run | Crop | Attention loss | λ | Val BACC | Test BACC | Δ vs reference | EP (CAM) |
|---|---|---|---|---|---|---|---|
| reference (unmodified) | – | – | – | – | 0.8135 | – | 0.462 |
| crop0_attn1_w0.5_seed42 | False | True | 0.5 | 0.7935 | 0.7502 | -0.0633 | 0.759 |
| crop1_attn0_w0_seed42 | True | False | - | 0.7803 | 0.7766 | -0.0369 | 0.474 |
| crop1_attn1_w0.2_seed42 | True | True | 0.2 | 0.7788 | 0.7478 | -0.0657 | 0.886 |
| crop1_attn1_w0.5_seed42 | True | True | 0.5 | 0.7921 | 0.7904 | -0.0231 | 0.911 |

| Run | Ensemble (no TTA) | TTA ensemble | Δ TTA vs reference |
|---|---|---|---|
| crop1_attn1_w0.2_seed42 | 0.8207 | - | - |
| crop1_attn1_w0.5_seed42 | 0.8319 | 0.8314 | -0.0172 |

Reference: ResNet-50 0.8135, ensemble 0.8315, TTA ensemble 0.8486.
