# [Algorithm Description] Task 3 Team MIAGENT

**1. Task** — Task 3 (Fracture Reduction)

**2. Team name** — MIAGENT

**3. Authors** — Yiheng Wang, Yufan He

**4. Affiliations** — NVIDIA

**5. Contact author and email address** — Yiheng Wang, vennw@nvidia.com

**6. Algorithm name or title** — Damped AssemblyNet with a geometry-conditioned residual pose
head and a five-member SE(3) ensemble

---

## 7. Method description

The authors treat fracture reduction as per-fragment SE(3) pose regression and address it in
three stages, each correcting a different failure mode of the previous one.

**Stage 1 — damped iterative assembly.** The backbone is AssemblyNet, the transformer-based
fragment-assembly network provided by the Task 3 baseline, trained **only on the simulated
fracture data**. No clinical data reaches the backbone at any point: all clinical adaptation is
delegated to stage 2. Pose is produced by iterative refinement, but each update is damped
by a factor of 0.3 before it is applied, and iteration stops when the largest per-point
displacement between two successive iterations falls below 2 mm (at most 20 iterations; 5,000
points sampled per bone and 1,000 per fragment). Damping keeps the refinement convergent on
fragments with a small contact surface, where an undamped update overshoots and the sequence
oscillates rather than settling.

**Stage 2 — a residual pose head that carries all of the clinical adaptation.** This is where
the simulation-trained backbone is adapted to clinical anatomy, by a head with 142,086
parameters — roughly three orders of magnitude smaller than the backbone. A small network sees the
predicted fragment point cloud and its local context (the nearest points of the surrounding
fragments) and predicts a correction expressed as a displacement of the fragment centroid
together with a left-multiplied SO(3) rotation. It deliberately never regresses the translation
column of the 4×4 matrix, because that column is coupled to rotation about the CT origin and is
not an independent degree of freedom. The head is initialised to output exactly the identity
correction, so an under-trained or heavily regularised head leaves the base prediction
unchanged.

**Stage 3 — a five-member ensemble in the physical parameterisation.** Five AssemblyNet models,
each with its own residual head, are combined. They come from three independent training runs
that differ in random seed: one model from the first, one from the second, and three
checkpoints from the third. A residual head is tied to the base whose predictions it was fitted
on, so the pairing is not interchangeable.

Poses are combined per fragment as **SO(3) rotation + fragment-centroid displacement**:

```
d = R c + t − c          (displacement of the fragment centroid)
t = c + d − R c          (rebuild the translation column)
```

Rotations are averaged on SO(3) through the matrix logarithm and re-projected with an SVD;
centroid displacements are averaged in ℝ³. This decomposition is what allows rotation and
translation to carry different ensemble weights, because those two quantities are physically
independent whereas the raw matrix entries are not.

This is also why the combination is not performed in se(3): there `log(T) = [[ω]×, v]` with
`v = V(ω)⁻¹ t`, so the translation part is a function of the rotation and cannot carry a weight
of its own. The same coupling applies, one level up, to the raw 4×4 translation column.

---

## 8. Main technical contributions / novel components

1. **Damped iterative inference with a max-point convergence criterion.** Each update is scaled
   by 0.3 before being applied and iteration stops on the largest per-point displacement, which
   makes the refinement loop converge rather than oscillate.

2. **A fragment-centric residual pose head.** It corrects an existing pose in the SO(3) +
   centroid-displacement parameterisation, never regressing the coupled translation column, and
   is zero-initialised so that it is an exact identity map before training.

3. **A pose ensemble in the same physical parameterisation, with separate rotation and
   translation weights.** Because rotation and centroid displacement are physically
   independent, each can carry its own weight vector across members; this is not expressible on
   the raw matrix or in se(3).

---

## 9. Step-by-step pipeline

1. Load the fragment meshes and sample point clouds (5,000 points per bone, 1,000 per fragment).
2. For each of the five members: run AssemblyNet iteratively with update damping 0.3, stopping
   when the maximum per-point change falls below 2 mm or after 20 iterations.
3. Normalise the transforms with respect to the first sacrum fragment.
4. For the same member: build the fragment and context point clouds from its own prediction and
   apply its own five-fold residual head (all five fold heads averaged).
5. Repeat 2–4 for each of the five members.
6. Combine the five members per fragment: weighted rotation mean on SO(3), weighted mean of the
   fragment-centroid displacement, then rebuild the 4×4 matrix.
7. Validate orthogonality and determinant of every output rotation and write
   `reduction-poses-matrices.json`.

---

## 10. External data

None. Only the data distributed by the challenge was used — the simulated fracture set for
pre-training and the clinical training set for fine-tuning.

## 11. Externally pretrained models

None. AssemblyNet is trained by the authors on the challenge's own simulated fracture data,
starting from random initialisation.

## 12. Preprocessing

The input is mesh geometry rather than images, so there is no intensity preprocessing. Vertices
are loaded with mesh processing disabled (`trimesh.load(..., process=False)`) so that vertices
are neither merged nor reordered; this keeps fragment centroids identical to those the official
evaluator computes. Point clouds are sampled with a fixed seed for reproducibility, and poses
are expressed relative to the first sacrum fragment.

## 13. Data augmentation

For AssemblyNet, the baseline's own simulation-to-real augmentation, unchanged. For the residual
head, the training targets are the per-fragment pose errors of its base model and the only
stochasticity is the point sampling itself.

## 14. Training and validation strategy

**AssemblyNet** — trained on the simulated fracture set only. The clinical training set is never
used to update backbone weights; all clinical adaptation happens in the residual head, which
keeps the five backbones independent of the clinical cross-validation split.

**Residual head** — five-fold cross-validation over the 170 clinical cases. Each fold's head is
trained only on that fold's training split, and every residual number the authors report is an
out-of-fold prediction, so no case is scored by a model that trained on it. The number of epochs
is chosen by an inner validation split *inside* each fold, so the outer fold stays untouched. At
inference all five fold heads are averaged, which makes the deployed model slightly stronger
than the reported out-of-fold figure.

**Ensemble weights** — selected on clinical cases 001–183 and confirmed on the locked cases
184–200.

## 15. Loss functions

**AssemblyNet** — the baseline's coordinate loss, unchanged.

**Residual head** — a weighted sum of smooth-L1 terms on the corrected fragment points, the
centroid displacement and the rotation vector, plus a small identity-regularisation term
penalising corrections away from the identity:

```
L = L_point + 0.25 · L_centroid + 2.0 · L_rotation + 0.01 · L_identity
```

with β = 1.0 mm for the point and centroid terms and β = 1° for the rotation term.

## 16. Base network architecture

**Stage 1** — AssemblyNet (transformer-based fragment assembly) as provided by the Task 3
baseline repository, coordinate output mode.

**Stage 2** — `ResidualPointNet`, a small permutation-invariant point encoder in the PointNet
style. Two encoders (three 1×1 convolutions with GELU, then max-pooling over points) embed the
fragment cloud and its context cloud into 128 dimensions each. The two clouds carry surface
points and normals; the context cloud additionally carries the identity of the neighbouring
fragment. These embeddings are concatenated with 14 global features describing bone identity,
the base model's motion for this fragment and the assembly context, and passed through a two-layer MLP (LayerNorm, GELU, dropout 0.10) to
six outputs. The outputs are squashed by `tanh` and scaled to a bounded correction of at most
20 mm in centroid displacement and 25° in rotation, the rotation being applied through a
differentiable Rodrigues formula. The final layer is zero-initialised, so the head starts as an
exact identity map. The head has 142,086 parameters, which is also why five of them cost almost
nothing at inference.

## 17. Ensembling strategies used during inference

Two levels.

**Within a member** — the five cross-validation residual heads are averaged.

**Across members** — the five models are combined per fragment as in item 7: a weighted rotation
mean on SO(3) through the matrix logarithm with SVD re-projection, and a weighted mean of the
fragment-centroid displacement. Rotation and translation carry separate weight vectors:

| | model A | model B | run-3 ckpt 1 | run-3 ckpt 2 | run-3 ckpt 3 |
|---|---:|---:|---:|---:|---:|
| rotation | **0.600** | 0.050 | 0.117 | 0.117 | 0.117 |
| translation | 0.280 | 0.280 | 0.147 | 0.147 | 0.147 |

The two vectors differ because the two halves of the pose are carried by different members.
Rotation is concentrated on the model with the lowest standalone rotation error and given only
a small share to the member with the highest, while translation is spread across the three
independent training runs in roughly equal shares (0.28 / 0.28 / 0.44) rather than equally
across the five models, since members drawn from the same run are not independent.

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
2. Yibulayimu, S., Liu, Y., Sang, Y., Qin, J., Shi, C., Liang, C., Zhu, G., Wang, Y., Zhao, C. &
   Wu, X. *FracFormer: Fracture Reduction Planning With Transformer-Based Shape Restoration and
   Fracture Data Simulation.* IEEE Transactions on Medical Imaging 44(8), 3270–3283 (2025).
   doi:10.1109/TMI.2025.3561030 — the transformer-based reduction-planning benchmark this task's
   baseline builds on.
3. Liu, Y., Yibulayimu, S., Sang, Y., Zhu, G., Shi, C., Liang, C., Cao, Q., Zhao, C., Wu, X. &
   Wang, Y. *Preoperative fracture reduction planning for image-guided pelvic trauma surgery: A
   comprehensive pipeline with learning.* Medical Image Analysis 102, 103506 (2025).
   doi:10.1016/j.media.2025.103506
4. PENGWIN 2026 Task 3 baseline repository (AssemblyNet, the undamped iterative inference and the
   coordinate loss this work builds on):
   <https://github.com/Sutuk/PENGWIN2026_Task3_Reduction_Baseline>
5. Qi, C. R., Su, H., Mo, K. & Guibas, L. J. *PointNet: Deep Learning on Point Sets for 3D
   Classification and Segmentation.* CVPR (2017). — the residual head's encoder design.
6. Moakher, M. *Means and averaging in the group of rotations.* SIAM Journal on Matrix Analysis
   and Applications 24(1), 1–16 (2002). — rotation averaging on SO(3) via the matrix logarithm,
   and why the result must be re-projected onto SO(3).
7. Pennec, X. *Intrinsic statistics on Riemannian manifolds: basic tools for geometric
   measurements.* Journal of Mathematical Imaging and Vision 25, 127–154 (2006). — Fréchet /
   Karcher means on Lie groups. The authors implemented the iterative Fréchet mean as an
   alternative to the one-shot log-Euclidean mean used here and measured the difference at
   0.0021° on average over 1,014 fragments, so the cheaper form was kept.
8. Murray, R. M., Li, Z. & Sastry, S. S. *A Mathematical Introduction to Robotic Manipulation.*
   CRC Press (1994). — the SE(3) exponential and the coupling `v = V(ω)⁻¹t` that motivates
   averaging in the SO(3) + centroid parameterisation rather than in se(3).
