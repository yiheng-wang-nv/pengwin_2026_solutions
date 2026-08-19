# [Algorithm Description] Task 1 Team MIAGENT

**1. Task** — Task 1 (Automatic Fragment Segmentation)

**2. Team name** — MIAGENT

**3. Authors** — Yiheng Wang, Yufan He

**4. Affiliations** — NVIDIA

**5. Contact author and email address** — Yiheng Wang, vennw@nvidia.com

**6. Algorithm name or title** — Two-stage anatomy-then-contact-surface segmentation with a
geometry-gated instance decoder

---

## 7. Method description

The authors formulated the task as two semantic segmentation problems followed by a
deterministic instance decoder.

The first network segments the CT into five anatomical classes (background, sacrum, left hip,
right hip, femur). Its output is then passed through a **geometry gate**: the challenge defines a
deterministic rule, computed from the input CT's spacing and physical extent, that decides
whether a scan is a pelvic scan (sacrum / left hip / right hip) or a femur scan. A scan is never
both, so the classes belonging to the other region are set to background. The rule is evaluated
on the input geometry, independently of the network's output.

Each surviving bone is then cropped by masking the CT with its anatomical label, and the second
network predicts a three-class contact-surface map (CSM) for that bone: background, fragment
interior, and **contact** — the thin surface where two fragments touch. Predicting the contact
surface rather than the fragments themselves turns instance segmentation into a semantic problem,
because the contact class is exactly the separator that connected-component analysis needs.

The instance decoder then runs per bone: connected components of the interior class become
fragment cores, components smaller than a threshold are dropped as noise, and each contact
component is dilated by a cubic structuring element (kernel 7) and assigned to the core it
touches. Fragments are finally offset by a per-bone label base to produce the 0–200 instance
map.

---

## 8. Main technical contributions / novel components

1. **A label-swapping left/right mirror augmentation that makes bilateral symmetry usable.**
   The pelvis has *distinct* left-hip and right-hip classes, so a plain spatial mirror along the
   left/right axis moves the two hips without relabelling them. The authors mirror only that
   axis and swap labels 2 ↔ 3 whenever the flip is applied, which makes the flip a valid
   bilateral-symmetry augmentation and doubles the effective anatomical data. Mirror test-time
   augmentation is correspondingly disabled, because nnU-Net's built-in version averages
   predictions without swapping label channels and would be invalid on this axis: the symmetry
   is exploited at training time, where it holds, and not at inference time, where it does not.

2. **A gate evaluated on input geometry.** The pelvic/femur decision is made from the CT's
   spacing and physical extent rather than from the predicted voxel counts, so it remains
   correct independently of the anatomy network's output on that case.

3. **A contact-surface formulation that turns instance segmentation into semantic
   segmentation.** The second network predicts the thin surface where two fragments touch, and
   that class is precisely the separator connected-component analysis needs, so fragment
   instances follow from a deterministic decoder rather than from an instance head.

---

## 9. Step-by-step pipeline

1. Read the input CT.
2. **Anatomical segmentation** into five classes.
3. **Geometry gate** — classify the scan as pelvic or femur from the CT's spacing and physical
   extent, and zero the minority region of the anatomy prediction.
4. For each remaining bone, mask the CT with that bone's anatomical label.
5. **Contact-surface segmentation** per bone: background / fragment interior / contact.
6. **Instance decoding** per bone: connected components of the interior class form cores, small
   components smaller than 100 voxels are discarded, and each contact component is dilated with
   a 7x7x7 structuring element and merged into the single core it touches.
7. Offset each bone's instance labels by its label base and write the merged 0–200 map as a
   compressed uint8 `.mha`.

---

## 10. External data

None. Only the data distributed by the challenge was used.

## 11. Externally pretrained models

None. Both networks are trained from random initialisation on the challenge data. No third-party
pretrained weights (for example TotalSegmentator) were used.

## 12. Preprocessing

nnU-Net's standard CT preprocessing: intensity clipping to the foreground 0.5–99.5 percentile
range, z-score normalisation with the dataset foreground statistics, and resampling to the
planned spacing. The second stage additionally receives a CT masked by the anatomical label of
the bone being processed, so each bone is segmented in isolation. The contact-surface training
labels are derived from the ground-truth fragment labels by morphological analysis of the
inter-fragment interfaces.

## 13. Data augmentation

nnU-Net v2's default 3D augmentation: random rotation, scaling, elastic deformation, Gaussian
noise, Gaussian blur, brightness and contrast changes, low-resolution simulation and gamma
correction.

Mirroring is handled specially for the **anatomy network**. Plain mirroring is invalid here
because the left and right hips are separate classes and a spatial flip does not relabel them,
which is why the baseline disables mirroring. The authors instead use a custom trainer
(`nnUNetTrainer_LRMirror`) that mirrors only the left/right axis — axis 2 of the preprocessed
`(z, y, x)` volume — and swaps labels 2 ↔ 3 whenever that axis is flipped, making the flip a
correct bilateral-symmetry augmentation. Only that one axis is mirrored.

Mirror **test-time** augmentation is disabled for both stages. For the anatomy network this is
required for correctness: nnU-Net's built-in mirror TTA does not swap label channels, so
averaging over a left/right flip would be invalid on that axis.

## 14. Training and validation strategy

Both networks are nnU-Net v2 models, ResEnc XL planner, `3d_fullres` configuration, 1000 epochs.

The submitted weights are trained on **all 340 patients** (`fold_all`), so inference costs one
forward pass per stage.

Method decisions were validated out-of-fold: each of the 320 cross-validation patients was
predicted by the one fold that did not train on it, and the resulting predictions — never a
small in-training-set subset — were scored with the official evaluator.

## 15. Loss functions

nnU-Net v2's default compound loss: Dice loss plus cross-entropy, with deep supervision at every
decoder resolution.

## 16. Base network architecture

nnU-Net v2 Residual Encoder U-Net, XL preset (`ResidualEncoderUNet`, six stages, features per
stage 32/64/128/256/320/320, instance normalisation, LeakyReLU), `3d_fullres`, patch size
256×160×160.

* Anatomy network: 1 input channel, 5 output classes, trained with the custom
  `nnUNetTrainer_LRMirror` described in item 13.
* Contact-surface network: 1 input channel, 3 output classes (background / foreground / contact),
  stock `nnUNetTrainer`.

For deployment the anatomy checkpoint's `trainer_name` is relabelled to
`nnUNetTrainerNoMirroring` and its allowed mirror axes emptied. The custom trainer class is not
present inside the container and nnU-Net rebuilds the network from `trainer_name`; the
architecture the two names produce is identical, and the relabelling is what keeps mirror TTA
off at inference. The trained weights are unchanged.

## 17. Ensembling strategies used during inference

None. Both stages run a single all-data model and test-time augmentation is disabled.

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
4. PENGWIN 2026 Task 1 baseline repository (the two-stage design and the contact-surface
   representation this work builds on):
   <https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline>
