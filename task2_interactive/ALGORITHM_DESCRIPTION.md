# [Algorithm Description] Task 2 Team MIAGENT

**1. Task** — Task 2 (Interactive Fragment Segmentation)

**2. Team name** — MIAGENT

**3. Authors** — Yiheng Wang, Yufan He

**4. Affiliations** — NVIDIA

**5. Contact author and email address** — Yiheng Wang, vennw@nvidia.com

**6. Algorithm name or title** — Click-gated two-stage segmentation with a progressive
per-click upgrade

---

## 7. Method description

The authors formulated the task as a fast, complete answer that is then progressively replaced
by a slower and better one for as long as the time budget allows.

The first stage segments anatomy into five classes (background, sacrum, left hip, right hip,
femur) and applies a **click gate**: because every click carries the name of the bone it belongs
to, the anatomy prediction is restricted to exactly the classes the user actually clicked.

The second stage predicts the contact-surface (CSM) representation per bone and decodes it into
fragment instances by connected components, exactly as in the authors' Task 1 solution. Clicks
are used only to name the resulting instances. This produces a complete, valid answer quickly.

The third stage is a separate per-click network. It takes three
channels — the CT, a Gaussian heatmap of one fragment click marked as foreground, and a heatmap
of all remaining clicks marked as background — and segments that single fragment. Because each
fragment is segmented independently from its own click, this model **cannot merge two fragments
by construction**. After the contact-surface answer exists, the pipeline spends
whatever wall clock remains re-segmenting bones with this model, replacing a bone's labels the
moment that bone finishes. If time runs out, the CSM answer for the remaining bones is still
there, so the output is always complete.

The per-click masks neither tile the bone nor stay disjoint, so the upgrade trades a small
amount of precision and split accuracy for a large reduction in merge errors.

---

## 8. Main technical contributions / novel components

1. **A click gate.** The clicks state which bones are present, so the anatomy prediction is
   restricted to the clicked classes rather than inferred from image geometry.

2. **A progressive, anytime upgrade.** A complete answer is produced first and then improved
   bone by bone under a wall-clock budget, so the submission degrades gracefully instead of
   timing out, and a per-fragment model becomes usable inside the container's time limit.

3. **Merge-freedom by construction.** In the upgrade stage a merge cannot occur, because each
   fragment is segmented from its own click with all other clicks marked as background.

4. **An inference path built for the container's limits:** GPU trilinear resampling instead of
   CPU spline resampling, a fully in-memory per-fragment path with no intermediate volume
   written to disk, and Gaussian click heatmaps computed only inside a 7-sigma window.

---

## 9. Step-by-step pipeline

1. Read the CT and the click JSON.
2. Anatomy segmentation into five classes.
3. **Click gate** — keep only the anatomical classes that were actually clicked.
4. Per-bone contact-surface (CSM) prediction and connected-component decoding into fragment
   instances; clicks assign the instance labels. *A complete answer now exists.*
5. **Progressive upgrade** — for each bone, while wall clock remains: run the per-click
   3-channel model once per fragment of that bone, keep the component containing the click,
   and replace that bone's labels with the result.
6. Expand the instance map to fill the anatomical mask so no anatomy voxel is left unlabelled.
7. Write the result as a compressed uint8 `.mha`.

---

## 10. External data

None. Only the data distributed by the challenge was used.

## 11. Externally pretrained models

None external. The anatomy and CSM models are the authors' own Task 1 models; the per-click
fragment model was trained from scratch on the challenge data. No third-party pretrained
weights (for example TotalSegmentator) were used.

## 12. Preprocessing

nnU-Net's standard CT preprocessing: intensity clipping to the foreground 0.5–99.5 percentile
range, z-score normalisation with the dataset foreground statistics, and resampling to the
planned spacing. Click prompts are converted to Gaussian heatmaps (σ = 1 voxel) at the click
coordinates and appended as extra input channels. Resampling at inference is done on the GPU
with trilinear interpolation.

## 13. Data augmentation

nnU-Net v2's default 3D augmentation: random rotation, scaling, elastic deformation, Gaussian
noise, Gaussian blur, brightness and contrast changes, low-resolution simulation and gamma
correction. Mirroring is disabled, because a spatial mirror along the left/right axis moves the
left and right hips without relabelling them.

## 14. Training and validation strategy

All networks are nnU-Net v2 models, ResEnc XL planner, `3d_fullres` configuration.

* Anatomy and CSM models: trained on **all 340 patients** (`fold_all`).
* Per-click fragment model: trained on **fold 0** (~272 of 340 patients), one training sample
  per (fragment, CT) pair.

Method decisions were validated on held-out cases and scored with the official evaluator.

## 15. Loss functions

nnU-Net v2's default compound loss: Dice loss plus cross-entropy, with deep supervision at every
decoder resolution.

## 16. Base network architecture

nnU-Net v2 Residual Encoder U-Net, XL preset (`ResidualEncoderUNet`, six stages, features per
stage 32/64/128/256/320/320, instance normalisation, LeakyReLU), `3d_fullres`.

* Anatomy model: 5 input channels (CT + 4 anatomical click heatmaps), 5 output classes.
* CSM model: 1 input channel, 3 output classes (background / foreground / contact).
* Per-click fragment model: 3 input channels (CT + foreground click + background clicks),
  2 output classes.

## 17. Ensembling strategies used during inference

None. Every stage runs a single set of weights and test-time augmentation is disabled; the
wall-clock budget is committed to the progressive upgrade instead.

## 18. Code repository

<https://github.com/yiheng-wang-nv/pengwin_2026_solutions>

The repository is **currently private** and will be made **public after the challenge results
are announced**. Please contact the corresponding author if access is needed before then.

## 19. References

1. Sang, Y., Liu, Y., Yibulayimu, S., Wang, Y., Killeen, B. D., Liu, M., Ku, P.-C., Johannsen, O.,
   Gotkowski, K., Zenk, M., Maier-Hein, K., Isensee, F., Yue, P., Wang, Y., Yu, H., Pan, Z., He, Y.,
   Liang, X., Liu, D., Fan, F., Jurgas, A., Skalski, A., Ma, Y., Yang, J., Płotka, S., Litka, R.,
   Zhu, G., Song, Y., Unberath, M., Armand, M., Ruan, D., Zhou, S. K., Cao, Q., Zhao, C., Wu, X. &
   Wang, Y. *Benchmark of Segmentation Techniques for Pelvic Fracture in CT and X-Ray: Summary of
   the PENGWIN 2024 Challenge.* IEEE Transactions on Medical Imaging 45(5), 2212–2228 (2026).
   doi:10.1109/TMI.2025.3650126
2. Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J. & Maier-Hein, K. H. *nnU-Net: a
   self-configuring method for deep learning-based biomedical image segmentation.*
   Nature Methods 18, 203–211 (2021).
3. Isensee, F. et al. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image
   Segmentation.* MICCAI (2024). — the Residual Encoder XL presets used here.
4. PENGWIN 2026 Task 2 baseline repository:
   <https://github.com/Zrrr1997/PENGWIN2026_Task2_InteractiveSeg_Baseline>
5. PENGWIN 2026 Task 1 baseline repository (the anatomy and contact-surface stages of this
   pipeline build on it):
   <https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline>
