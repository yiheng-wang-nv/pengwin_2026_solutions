"""Task 2 with a FLAT cost model: segment every fragment at once, let the clicks name them.

The shipped pipeline (`process.py`) runs ds457 once per fragment, so its runtime is linear
in fragment count. That is what times out on Grand Challenge: the 5-case dev phase tops out
at 8 fragments and passes, while a real case with 30 fragments needs ~30 full sliding
windows and blows the 600 s/case budget. Every constant-factor fix was tried and measured
(in-place heatmaps 2.52->0.18 s, cached preprocessing -2.0 s/fragment, window dedup, ROI
cropping, torch.compile, ONNX, larger tile step) and together they bought 1.64x -- not the
~7x the worst cases need. The cost model has to change, not its constant.

This routes Task 2 through Task 1's decoder, whose cost is FLAT:

    CT + clicks -> ds456 anatomy (clicks used here) -> pelvic/femur gate
                -> per-bone masked CT -> ds002 CSM -> frac_to_instance
                -> give each instance the label of the click inside it

4 inferences whether the case has 2 fragments or 30.

The clicks are demoted from "drive the segmentation" to "name the instances", which sounds
like a downgrade and measurably is not. On the 7 anchor cases with the official evaluator:

    ds457 per-fragment (shipped) : dice 0.8692  insF1 0.9143  merge 0.333  split 0.048
    this pipeline                : dice 0.8962  insF1 0.9508  merge 0.286  split 0.000

and that comparison is biased AGAINST this pipeline, because ds457 trained on all 340
patients including these 7 while ds002 held them out.

Click post-processing beyond plain label assignment was tried and rejected on measurements,
so it is deliberately absent here: splitting a component that holds two clicks (the clicks
being direct evidence of an under-segmentation) costs dice 0.8962 -> 0.8806 and drives
split_error 0.0000 -> 0.4286, and adaptively re-decoding each bone until every click owns a
distinct component changes the voxels but not a single official metric (dice +0.0002).
The remaining gap to the 0.9946 that a ground-truth CSM decodes to is in the CSM model, not
in the decoder.

`lowmem_resampling.install()` is NOT optional. nnU-Net's stock resampler is a scipy order-3
spline on a float64 copy, and it dominates everything else here: profiled at 69.6 s of
preprocessing against 3.9 s of actual GPU sliding window per fragment. The shipped
`process.py` installs the GPU (trilinear) replacement on line 47; this file was written
without it and therefore ran ~11x slower than the pipeline it was derived from -- long
enough that a per-fragment upgrade path looked infeasible and was nearly abandoned on that
false evidence. Trilinear differs from cubic on ~1% of fragment-foreground voxels, which the
instance metrics do not resolve.

Env:
  CSM_KERNEL / CSM_CCF  decoder knobs (5 / 100 = the Task 1 champion's, swept and null)
  TILE_STEP             sliding-window density (0.5 = what every score was measured with)
  UPGRADE_DS457         after the CSM result exists, re-segment bones with the per-click
                        model for as long as the wall clock allows (merge-free by design)
  CASE_BUDGET_S         the organisers' per-case limit, 600 s
"""

from __future__ import annotations

import time
# Start the case clock BEFORE the heavy imports. torch + nnU-Net + SimpleITK cost ~4.5 s
# locally and more on the Grand Challenge node, and that time is inside the organisers'
# 600 s just as much as the inference is. Measured: container wall clock 262.83 s vs
# 258.3 s reported from after the imports.
_T0 = time.time()

import gc
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

INF = os.environ.get("INF_DIR", "/opt/algorithm/inference")
INPUT = os.environ.get("INPUT_DIR", "/input")
OUT_SLUG = os.environ.get("OUTPUT_SLUG", "peripelvic-fracture-ct-segmentation")
OUTPUT = os.environ.get("OUTPUT_DIR", f"/output/images/{OUT_SLUG}")
MODEL = os.environ.get("nnUNet_results", "/opt/ml/model")
sys.path.insert(0, INF)

import lowmem_resampling  # noqa: E402
lowmem_resampling.install()   # GPU resampling — see below
import inmem  # noqa: E402
import fastheat  # noqa: E402
import fastpre  # noqa: E402
import click_channel  # noqa: E402
from nnunetv2.inference.export_prediction import (  # noqa: E402
    convert_predicted_logits_to_segmentation_with_correct_shape)
from frac_to_instance import merged_mask_to_instance  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402

# ds458's channel-1 encoding. Overwritten from the model pack's dataset.json when a ds458
# pack is loaded; the default is the legacy sparse Gaussian so a pack that predates the key
# behaves exactly as it always did.
CLICK_ENCODING, CLICK_SIGMA = (click_channel.LEGACY_PARAMS["encoding"],
                               click_channel.LEGACY_PARAMS["sigma"])

TILE_STEP = float(os.environ.get("TILE_STEP", "0.5"))
CSM_KERNEL = int(os.environ.get("CSM_KERNEL", "5"))
CSM_CCF = int(os.environ.get("CSM_CCF", "100"))
# Repair CSM merges with per-click ds457 inference, O(merges) not O(fragments). Needs
# Dataset457 in the model pack; silently degrades to the plain flat pipeline without it.
HYBRID = os.environ.get("HYBRID_DS457", "0") == "1"
# Soft threshold on the CSM contact class. None = argmax (shipped behaviour).
_ct = os.environ.get("CONTACT_TAU", "")
CONTACT_TAU = float(_ct) if _ct else None
# Progressive upgrade: after the flat CSM result exists, spend whatever wall clock is
# left re-segmenting bones with the per-click ds457 model (merge-free by construction).
# CASE_BUDGET_S is the organisers' 600 s/case; SAFETY leaves room for I/O and the write.
UPGRADE = os.environ.get("UPGRADE_DS457", "1") == "1"
CASE_BUDGET_S = float(os.environ.get("CASE_BUDGET_S", "600"))
SAFETY = float(os.environ.get("BUDGET_SAFETY", "0.90"))
# DEFAULTS ARE OFF ON PURPOSE. The four switches below (strict partition, hole fill,
# contested-voxel resolution, skip-clean-bones) were added after the evaluated configuration
# was fixed and are not part of it. Enabling them would change the segmentation at the same
# time as the weights, making the two effects unattributable. With these off the
# image reproduces v5 byte-for-byte in behaviour, so the ONLY difference in the next
# submission is the fold_all fix below plus whichever model pack is mounted. Turn them on
# one at a time, each against its own submission.
# Make each upgraded bone a strict partition (no holes, no contested voxels). See
# _partition_bone; this is the step the shipped process.py does as expand_fragments.
STRICT_PARTITION = os.environ.get("STRICT_PARTITION", "0") == "1"
# Max voxel distance a hole may be from a claimed region to be filled. 0 = unlimited,
# negative = do not fill at all. Filling is the half of the partition that costs: it
# buys dice +0.012 but drives split_error 0.0185 -> 0.0741 and precision 0.9880 ->
# 0.9671, because expanding into a region ds457 missed entirely lets several neighbours
# converge on a ground-truth fragment that none of them predicted.
FILL_MAX_DIST = float(os.environ.get("FILL_MAX_DIST", "-1"))
# Re-assign voxels claimed by two fragments to the nearest click. Measured HARMFUL:
# split_error 0.0185 -> 0.0741 with no compensating gain, and limiting the hole fill
# leaves it unchanged, which is what isolates this half as the cause. The cut it makes
# through an overlap can run across a ground-truth fragment, leaving that fragment
# covered by two predicted instances.
RESOLVE_CONTESTED = os.environ.get("RESOLVE_CONTESTED", "0") == "1"
# Never re-segment a bone whose clicks already each own a CSM component.
SKIP_CLEAN = os.environ.get("SKIP_CLEAN_BONES", "0") == "1"

# Official label ranges: sacrum 1-50, left hip 51-100, right hip 101-150, femur 151-200.
BONES = {1: ("sacrum", 0), 2: ("leftHip", 50), 3: ("rightHip", 100), 4: ("femur", 150)}
CLICK_KEYWORD = {"Sacrum": 1, "Left Hip": 2, "Right Hip": 3, "Femur": 4}

_LAST = [_T0]


def _t(label):
    now = time.time()
    print(f"  [{now - _T0:7.1f}s] +{now - _LAST[0]:6.1f}s  {label}", flush=True)
    _LAST[0] = now


def _to_np(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def find_model(ds_glob, tr_glob):
    hits = sorted(glob.glob(os.path.join(MODEL, ds_glob, tr_glob)))
    if not hits:
        sys.exit(f"model not found: {ds_glob}/{tr_glob} under {MODEL}")
    return hits[0]


def make_predictor(model_dir):
    """Load a fold, tolerating either checkpoint filename.

    The Task 1 packs ship checkpoint_best.pth for the champion single-fold weights and
    checkpoint_final.pth for the all-data ones, so which name exists depends on the pack
    rather than on this code.
    """
    numeric = sorted(int(Path(d).name.split("_")[1])
                     for d in glob.glob(os.path.join(model_dir, "fold_[0-9]")))
    if numeric:
        folds = tuple(numeric)
    elif glob.glob(os.path.join(model_dir, "fold_all")):
        # The all-data packs train fold_all, not fold_0. An earlier version defaulted folds to
        # (0,) before testing for fold_all, which made that branch unreachable and any
        # fold_all pack die on a missing fold_0/checkpoint_best.pth.
        folds = ("all",)
    else:
        folds = (0,)
    names = ["checkpoint_best.pth", "checkpoint_final.pth"]
    probe = os.path.join(model_dir, f"fold_{folds[0]}")
    chk = next((n for n in names if os.path.exists(os.path.join(probe, n))), names[0])
    p = nnUNetPredictor(tile_step_size=TILE_STEP, use_gaussian=True, use_mirroring=False,
                        device=torch.device("cuda"), verbose=False,
                        verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_dir, folds, checkpoint_name=chk)
    print(f"  loaded {Path(model_dir).parent.name} folds={folds} {chk}", flush=True)
    return p


def find(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(INPUT, pat), recursive=True))
        if hits:
            return hits[0]
    return None


def apply_gate(a, clicks=None):
    """Keep only the anatomy classes the user actually clicked; drop the rest.

    Task 1 has to infer whether a scan is pelvic or femur -- it uses the official geometry
    rule on the CT's spacing and physical extent -- and the shipped Task 2 pipeline guessed
    it from which region the model predicted more voxels of. Task 2 does not have to guess
    at all: the clicks name their bones ("Sacrum", "Left Hip", "Right Hip", "Femur"), which
    is user-supplied fact rather than inference.

    It matters. The geometry rule disagrees with the clicks on 45 of the 340 training cases
    (13.2%), and the voxel-majority gate has a failure mode of its own: one pelvic case with
    enough spurious femur prediction zeroes all three pelvic bones, and every click on them
    then lands on background and produces no output at all. Gating by clicks cannot
    misroute, and it subsumes the pelvic/femur split -- a case with only femur clicks keeps
    only femur.

    Falls back to the voxel-majority rule when no clicks parse, so the function still has
    defined behaviour on malformed input.
    """
    keep = {aid for aid in BONES if clicks and clicks.get(aid)}
    if keep:
        a[~np.isin(a, list(keep))] = 0
        return a
    pelvic = int(((a >= 1) & (a <= 3)).sum())
    femur = int((a == 4).sum())
    if femur > pelvic:
        a[(a >= 1) & (a <= 3)] = 0
    else:
        a[a == 4] = 0
    return a


def parse_clicks(json_path):
    """[(z, y, x), ...] per anatomy id.

    The clicks JSON stores points ALREADY in (z, y, x) despite every official parser
    commenting them '# [x, y, z]'. convert_fragments (which the winning ds457 pipeline
    uses) does not reverse them; convert_anatomy does. Case 084 settles it: shape
    (268, 512, 512) with a click at [192, 248, 272], which read as (x, y, z) puts z at 272,
    past the 268 slices. Reversing here silently drops each click outside its own fragment
    -- measured 0/10 matches on case 193 -- while everything downstream still looks
    plausible, so this is worth stating rather than inferring.
    """
    out = {1: [], 2: [], 3: [], 4: []}
    for p in json.load(open(json_path))["points"]:
        z, y, x = p["point"]
        for kw, aid in CLICK_KEYWORD.items():
            if kw in p["name"]:
                out[aid].append((int(z), int(y), int(x)))
                break
    return out


def label_instances_by_clicks(inst, clicks_zyx, offset):
    """Give each predicted instance the fragment id of the click that lands in it.

    Instances holding no click get nothing -- the user did not mark them. Two clicks in one
    instance means the CSM under-segmented; the first click wins and the second fragment is
    lost. Cutting the instance between them instead was measured and is worse (see module
    docstring), so the loss is taken deliberately.
    """
    out = np.zeros(inst.shape, dtype=np.int32)
    per_comp = {}
    for idx, (z, y, x) in enumerate(clicks_zyx):
        if not (0 <= z < inst.shape[0] and 0 <= y < inst.shape[1] and 0 <= x < inst.shape[2]):
            continue
        comp = int(inst[z, y, x])
        if comp:
            per_comp.setdefault(comp, []).append(idx)
    for comp, idxs in per_comp.items():
        out[inst == comp] = offset + idxs[0] + 1
    return out, len(per_comp), len(clicks_zyx), per_comp


def upgrade_bone_with_ds457(final, ana, aid, offset, clicks_aid, pred457, frag_pre,
                            ct_arr, props, all_coords, base_index, budget_left):
    """Re-segment one bone's fragments with the per-click model, replacing the CSM result.

    The CSM pipeline and ds457 fail in opposite ways: the CSM decoder splits very rarely
    (split error 0.017) and merges often (0.419), while the per-click model cannot merge at
    all but overlaps. A merge is a hole in the predicted contact surface, and lowering the
    threshold on that surface does nothing -- measured, tau 0.4 and 0.3 leave every
    official metric bit-identical -- because the model's contact probabilities are
    saturated. It is not that we failed to squeeze the output; the seam is not visible to
    the model there. ds457 never needs to see the seam: each fragment gets its own binary
    segmentation driven by its own click, so merges are structurally impossible.

    ds457 is not used for everything only because its cost is linear in fragment count
    (~7-15 s per fragment locally, ~2.3x that on the T4). So it is spent per BONE, cheapest
    first, for as long as the wall clock allows -- the CSM result is already in `final`, so
    running out of budget degrades to today's shipped behaviour rather than to nothing.

    Returns (final, seconds_used, n_done). Bone-level granularity is deliberate: upgrading
    half a bone would mix two labelings of the same region.
    """
    from keep_clicked_fragment import keep_component_by_heatmap

    mask = ana == aid
    if not mask.any():
        return final, 0.0, 0
    t0 = time.time()
    buf = np.empty((3,) + ct_arr.shape, dtype=np.float32)
    buf[0] = ct_arr
    fg, bg = buf[1], buf[2]
    out = np.zeros(ct_arr.shape, dtype=np.uint8)
    nclaim = np.zeros(ct_arr.shape, dtype=np.uint8)
    done = 0
    for i in range(len(clicks_aid)):
        fastheat.fill_fg_bg(fg, bg, all_coords, base_index + i, sigma=1.0)
        pdata, pprops = frag_pre.run(buf, props)
        logits = pred457.predict_logits_from_preprocessed_data(torch.from_numpy(pdata)).cpu()
        seg = _to_np(convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, pred457.plans_manager, pred457.configuration_manager,
            pred457.label_manager, pprops, False))
        del pdata, logits
        kept = keep_component_by_heatmap(seg, fg, threshold=0.5)
        m = (kept > 0) & mask
        if m.any():
            # Overlap policy. ds457 segments each fragment independently, so two fragments can
            # claim the same voxel. The evaluated configuration lets the LATER fragment
            # overwrite; first-come-first-served is part of the strict-partition work and is
            # not part of that configuration, so it stays behind the same switch. Leaving it
            # unconditional would change the segmentation in a submission whose stated purpose
            # is to change only the weights.
            if STRICT_PARTITION:
                out[m & (nclaim == 0)] = offset + i + 1
                nclaim[m] += 1
            else:
                out[m] = offset + i + 1
        done += 1
        # Abort mid-bone if the estimate was wrong: the CSM labels for this bone are still
        # intact in `final` because nothing has been written back yet.
        if time.time() - t0 > budget_left:
            return final, time.time() - t0, 0
    if STRICT_PARTITION:
        out = _partition_bone(out, nclaim, mask, clicks_aid, offset)
    final[mask] = 0                      # drop the CSM labeling for this bone
    m = out > 0
    final[m] = out[m]
    return final, time.time() - t0, done


def _partition_bone(out, nclaim, mask, clicks_aid, offset):
    """Make the upgraded bone a strict partition: every voxel in exactly one fragment.

    ds457 segments each fragment independently, so its masks neither tile the bone nor stay
    disjoint. Both failure modes cost real metrics:

      unclaimed (nclaim==0)  a voxel inside the bone that no fragment predicted becomes
                             background and belongs to no instance at all. This is the
                             hole-filling step the shipped `process.py` does at [9] and this
                             pipeline was missing -- it drives recall, dice, and especially
                             HD95/ASSD, which punish isolated uncovered regions hard.
      contested (nclaim>1)   two fragments claim the same voxel and the loop's write order
                             decides, arbitrarily. Adjacent fragments end up interleaved,
                             which is a plausible source of the split_error jump from
                             0.017 to 0.124 when the upgrade was switched on.

    Contested voxels go to the nearest click, which is the only evidence available about
    which fragment a boundary voxel belongs to. Unclaimed voxels go to the nearest already-
    assigned voxel via the EDT's index map -- nearer than the original's "give the whole
    hole to the largest fragment", and it respects fragment shape rather than volume.

    Work is confined to the bone's bounding box; on a 70M-voxel case the box is a small
    fraction of it and the whole step costs well under a second.
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree

    idx = np.argwhere(mask)
    if len(idx) == 0:
        return out
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    o, nc, mk = out[sl], nclaim[sl], mask[sl]

    contested = mk & (nc > 1)
    if RESOLVE_CONTESTED and contested.any() and len(clicks_aid) > 1:
        pts = np.array(clicks_aid, dtype=float) - lo
        labs = np.array([offset + i + 1 for i in range(len(clicks_aid))], dtype=np.uint8)
        vox = np.argwhere(contested)
        _, nn = cKDTree(pts).query(vox)
        o[tuple(vox.T)] = labs[nn]

    unclaimed = mk & (o == 0)
    if unclaimed.any() and (o > 0).any():
        dist, inds = distance_transform_edt(o == 0, return_distances=True,
                                            return_indices=True)
        # Only close gaps, do not annex. Filling every unclaimed voxel regardless of
        # distance was measured and is the harmful half of this step: dice +0.0187 and
        # merge halved, but precision 0.9880 -> 0.9610 and split_error 0.0185 -> 0.0741,
        # which costs more rank than the dice buys. A voxel far from any prediction is
        # not a seam between fragments -- it is a region the model did not claim, and
        # handing it to the nearest neighbour manufactures false positives and lets one
        # fragment sprawl across another's territory.
        if FILL_MAX_DIST < 0:
            unclaimed[:] = False
        elif FILL_MAX_DIST > 0:
            unclaimed &= dist <= FILL_MAX_DIST
        if unclaimed.any():
            o[unclaimed] = o[tuple(inds[:, unclaimed])]
    out[sl] = o
    return out


def refine_collisions(final, inst, per_comp, aid, offset, pred457, ct_img, ct_arr, props,
                      all_coords, base_index):
    """Re-segment ONLY the fragments the CSM merged, using the per-click ds457 model.

    This is the middle ground between the two pipelines, and it exists because the local
    measurement said the flat pipeline's one real weakness is merges. On the 20 held-out
    patients ds457 scores merge_error 0.0370 against this pipeline's 0.3889 -- a 10x gap
    that leakage cannot explain away, because ds457's design makes merges structurally
    impossible: every fragment gets its own forward pass driven by its own click, so two
    fragments are never asked to be separated by a predicted contact surface.

    The reason the whole pipeline is not just ds457 is cost: one full sliding window PER
    FRAGMENT is linear in fragment count and times out on Grand Challenge. But a merge is
    rare -- roughly one per case -- so paying ds457's price only where the CSM actually
    failed is O(merges), not O(fragments). A 30-fragment case that needed 30 inferences
    needs about 2 here.

    Detection needs no ground truth: two clicks inside one component IS the merge, and the
    clicks are given. `base_index` is where this bone's clicks start in `all_coords`, which
    is the global click order ds457's background-heatmap channel is defined over -- get it
    wrong and the background channel describes the wrong fragments.
    """
    import fastheat
    from keep_clicked_fragment import keep_component_by_heatmap

    merged = {c: i for c, i in per_comp.items() if len(i) > 1}
    if not merged:
        return final, 0

    buf = np.empty((3,) + ct_arr.shape, dtype=np.float32)
    buf[0] = ct_arr
    fg, bg = buf[1], buf[2]
    n_inf = 0
    for comp, idxs in merged.items():
        region = inst == comp
        # Clear the merged component's provisional label before re-filling it, so a click
        # whose ds457 prediction lands outside the region does not leave the old label.
        final[region & (final > 0)] = 0
        for i in idxs:
            _a = time.time()
            fastheat.fill_fg_bg(fg, bg, all_coords, base_index + i, sigma=1.0)
            _b = time.time()
            seg = _to_np(pred457.predict_single_npy_array(buf, props, None, None, False))
            _c = time.time()
            kept = keep_component_by_heatmap(seg, fg, threshold=0.5)
            _d = time.time()
            print(f"      [REPAIR] heat={_b-_a:.1f}s infer={_c-_b:.1f}s keepcc={_d-_c:.1f}s",
                  flush=True)
            n_inf += 1
            # confine the result to the merged component: ds457 sees the whole CT and may
            # pick up the neighbouring fragment, which the CSM already labelled correctly
            m = (kept > 0) & region
            if m.any():
                final[m] = offset + i + 1
    return final, n_inf


def main():
    ct = find(["**/*.mha", "**/*.nii.gz"])
    clk = find(["**/*clicks*.json", "**/*.json"])
    if ct is None or clk is None:
        sys.exit(f"Missing input: ct={ct} clicks={clk} under {INPUT}")
    print(f"CT     : {ct}\nclicks : {clk}", flush=True)

    # Anatomy: prefer Task 1's 1-channel ds001 over Task 2's 5-channel ds456 when the pack
    # ships it. The two models are identical in labels, patch size, target spacing and
    # architecture -- the ONLY difference is 5 input channels (CT + 4 click heatmaps) vs 1.
    #
    # That difference is what breaks Grand Challenge's 16 GiB DRAM cap. nnU-Net's
    # preprocessing resamples every channel to the target spacing and pads each to float64,
    # so ds456's anatomy pass costs ~5x ds001's, and the cost scales with the RESAMPLED
    # shape, not the raw voxel count. Measured under a 15 GiB cap with ds456: case 389
    # (resampled 131M, the largest of the 340) survived on 306 MiB of headroom, and case
    # 048 (111M) was OOM-killed inside the anatomy stage. A dev-phase submission died the
    # same way ("Unable to allocate 932 MiB" in resample_data_or_seg). Task 1 ships ds001
    # and passes GC, which is the strongest evidence available that 1 channel fits.
    #
    # What is given up: ds456 knows which bones the user clicked. That matters less here
    # than it sounds, because the gate already drops the pelvic/femur minority and only
    # bones carrying clicks are decoded at all -- but it is an accuracy question, so it is
    # measured on the anchor cases rather than assumed. ANATOMY_DS forces either one.
    want = os.environ.get("ANATOMY_DS", "")
    ana_ds1 = glob.glob(os.path.join(MODEL, "Dataset001*", "nnUNetTrainer*"))
    use_ds001 = (want == "001") or (not want and bool(ana_ds1))
    if use_ds001:
        pred_ana = make_predictor(find_model("Dataset001*", "nnUNetTrainer*ResEncUNetXLPlans__3d_fullres"))
    else:
        pred_ana = make_predictor(find_model(
            "Dataset456*", "nnUNetTrainerNoMirroring__nnUNetResEncUNetXLPlans__3d_fullres"))
    print(f"  anatomy = {'ds001 (1ch)' if use_ds001 else 'ds456 (5ch)'}", flush=True)
    # ds458 is the click-conditioned CSM: same task and same channel-0 input as ds002,
    # plus a per-bone click heatmap. Preferred when the pack ships it, so the two can be
    # compared by swapping the model pack alone.
    use_458 = bool(glob.glob(os.path.join(MODEL, "Dataset458*", "nnUNetTrainer*")))
    csm_dir = find_model("Dataset458*" if use_458 else "Dataset002*",
                         "nnUNetTrainer*ResEncUNetXLPlans__3d_fullres")
    pred_csm = make_predictor(csm_dir)
    global CLICK_ENCODING, CLICK_SIGMA
    if use_458:
        # Read the channel-1 encoding off the model itself rather than assuming one. A pack
        # with no "click_encoding" key predates the field and is the sparse sigma=1 Gaussian,
        # which is what LEGACY_PARAMS returns -- so old packs keep working unchanged.
        try:
            with open(os.path.join(csm_dir, "dataset.json")) as f:
                dsj = json.load(f)
        except Exception:
            dsj = {}
        CLICK_ENCODING, CLICK_SIGMA = click_channel.params_from_dataset_json(dsj)
        print(f"  CSM = ds458 (2ch, click-conditioned; "
              f"encoding={CLICK_ENCODING} sigma={CLICK_SIGMA})", flush=True)
    else:
        print("  CSM = ds002 (1ch)", flush=True)
    _t("load predictors")

    ct_img = sitk.ReadImage(ct)
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    props = inmem.build_props(ct_img)
    _t("read CT")

    # [1] anatomy -- the clicks are used here, and it is one inference either way.
    #
    # ct_arr is dropped across this call and re-read afterwards. It is a full-volume
    # float32 (530 MiB on the largest case) that anatomy_input has already copied into
    # channel 0, so holding it through the prediction is pure waste -- and the waste is
    # not academic: a dev-phase run died here with "Unable to allocate 932 MiB" inside
    # nnU-Net's resample, which pads every one of the 5 channels to float64. Re-reading
    # costs ~1.5 s against a ~200 s stage.
    clicks = parse_clicks(clk)
    # ORACLE_ANATOMY replaces stage 1 with the ground-truth anatomy mask. Diagnostic only,
    # default off: it answers "how much of the end-to-end error is the anatomy model's?"
    # without which the question can only be argued, not measured. The anatomy model's own
    # IoU-A is ~0.965, which SUGGESTS it is not the bottleneck, but suggesting is not
    # measuring.
    oracle = os.environ.get("ORACLE_ANATOMY", "")
    if oracle:
        stem0 = Path(ct).name.replace(".nii.gz", "").replace(".mha", "")
        gt_ana = sitk.ReadImage(str(Path(oracle) / f"{stem0}.mha"))
        ana = sitk.GetArrayFromImage(gt_ana).astype(np.uint8)
        print(f"  [ORACLE] anatomy from {oracle}", flush=True)
    else:
        data_ana = (ct_arr[None] if use_ds001
                    else inmem.anatomy_input(ct_img, ct_arr, clk))
        del ct_arr
        gc.collect()
        ana = _to_np(pred_ana.predict_single_npy_array(data_ana, props, None, None, False))
        del data_ana
        gc.collect()
    ana = apply_gate(ana, clicks)
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    _t(f"anatomy ({'ds001' if use_ds001 else 'ds456'}) + gate")

    # uint8, not int32: labels top out at 200, and on a 100M-voxel case the wider dtype
    # would cost 400 MB of the 16 GiB the organisers cap DRAM at. The organisers reported
    # a RAM failure on an earlier submission, so the accounting is not academic.
    final = np.zeros(ct_arr.shape, dtype=np.uint8)
    # ds457 is loaded only when HYBRID is on AND the pack ships it. `all_coords` and
    # `base_index` reproduce inmem.fragment_iter's global click ordering exactly (anatomy
    # 1,2,3,4 concatenated, JSON order within each), because ds457's background channel is
    # "every click except this one" and is defined over that ordering. The per-anatomy
    # count check is there because a silent ordering mismatch would feed the model the
    # wrong background and still produce a plausible-looking mask.
    pred457, all_coords, base_index, n_refined = None, [], {}, 0
    if HYBRID:
        import convert_fragments_to_nnunet_input as _frag
        frags = _frag.parse_points_by_fragment(clk)
        for a in (1, 2, 3, 4):
            base_index[a] = len(all_coords)
            all_coords += [p["coord"] for p in frags[a]]
        for a in (1, 2, 3, 4):
            if len(frags[a]) != len(clicks[a]):
                sys.exit(f"click parse mismatch anatomy {a}: {len(frags[a])} vs {len(clicks[a])}")
        if glob.glob(os.path.join(MODEL, "Dataset457*", "nnUNetTrainer*")):
            pred457 = make_predictor(find_model(
                "Dataset457*", "nnUNetTrainer*ResEncUNetXLPlans__3d_fullres"))
        else:
            print("  [HYBRID] Dataset457 不在模型包里 -- 合并修复关闭", flush=True)
    tmp_csm = Path(os.environ.get("TMPDIR", "/tmp")) / "_csm.nii.gz"
    stats = []
    csm_ok = {}
    for aid, (bone, offset) in BONES.items():
        mask = ana == aid
        if not mask.any() or not clicks[aid]:
            continue
        # [2] one CSM inference per bone -- FLAT in fragment count
        masked = np.where(mask, ct_arr, 0).astype(np.float32)[None]
        if use_458:
            # ds458 is the same CSM task with a second input channel telling it where the
            # fragments are; channel 0 is the identical per-bone masked CT, so the two models
            # differ only in that. The channel is built by click_channel.build -- the same
            # function the training data was generated with -- using the encoding recorded in
            # THIS model's dataset.json. It used to be reimplemented inline here with a
            # hardcoded sigma=1 Gaussian, which silently stopped matching the moment the
            # generator moved to the dense distance field.
            masked = np.stack([masked[0],
                               click_channel.build(ct_arr.shape, clicks[aid],
                                                   CLICK_ENCODING, CLICK_SIGMA)])
        if CONTACT_TAU is None:
            csm = _to_np(pred_csm.predict_single_npy_array(masked, props, None, None, False))
        else:
            # Lower the bar for calling a voxel "contact surface", WITHOUT moving the
            # foreground boundary: argmax decides bone-vs-background as before, and only
            # voxels already inside the bone can flip from 1 (solid) to 2 (contact).
            #
            # This is the one decoder lever never tried. Every earlier sweep (kernel_size,
            # ccf_threshold, reassign-small-cores, largest-CC) operated on the ARGMAX mask
            # and moved nothing. But merge errors are holes in the contact surface, and a
            # hole is a voxel whose contact probability lost to foreground -- 0.4 vs 0.6
            # still argmaxes to solid and the two fragments stay joined. Dropping the
            # threshold rebuilds the wall out of evidence the model already produced.
            #
            # Why the risk is worth taking here specifically: split error is 0.017 while merge
            # error is 0.419, so the two are two orders of magnitude apart. There is a lot of
            # split headroom to spend on merges,
            # and this is the knob that spends it.
            csm, probs = pred_csm.predict_single_npy_array(masked, props, None, None, True)
            csm = _to_np(csm)
            probs = _to_np(probs)
            csm[(csm > 0) & (probs[2] > CONTACT_TAU)] = 2
            del probs
        del masked
        img = sitk.GetImageFromArray(csm.astype(np.uint8))
        img.CopyInformation(ct_img)
        sitk.WriteImage(img, str(tmp_csm), True)
        inst, _ = merged_mask_to_instance(str(tmp_csm), kernel_size=CSM_KERNEL,
                                          ccf_threshold=CSM_CCF, device="cuda")
        lab, matched, total, per_comp = label_instances_by_clicks(inst, clicks[aid], offset)
        # matched < total means the CSM either merged two fragments into one component or
        # missed one entirely -- both are cases the per-click model can repair. matched ==
        # total means every click already owns its own component and there is nothing to fix.
        csm_ok[aid] = (matched == total)
        m = lab > 0
        final[m] = lab[m]
        extra = 0
        if pred457 is not None:
            final, extra = refine_collisions(final, inst, per_comp, aid, offset, pred457,
                                             ct_img, ct_arr, props, all_coords,
                                             base_index[aid])
            n_refined += extra
        stats.append(f"{bone}:{matched}/{total}" + (f"+{extra}" if extra else ""))
        _t(f"{bone} CSM + decode" + (f" + {extra} 次 ds457 修复" if extra else ""))
    if tmp_csm.exists():
        tmp_csm.unlink()

    # --- progressive upgrade: spend the remaining wall clock on per-click ds457 ----------
    # A complete, valid result already exists at this point, so this can be cut off at any
    # bone and still produce today's shipped output. Bones are taken cheapest-first (fewest
    # fragments) to maximise how many get upgraded within the budget.
    if UPGRADE and glob.glob(os.path.join(MODEL, "Dataset457*", "nnUNetTrainer*")):
        import convert_fragments_to_nnunet_input as _frag
        frags = _frag.parse_points_by_fragment(clk)
        all_coords, base_index = [], {}
        for a in (1, 2, 3, 4):
            base_index[a] = len(all_coords)
            all_coords += [p["coord"] for p in frags[a]]
        bad = [a for a in (1, 2, 3, 4) if len(frags[a]) != len(clicks[a])]
        if bad:
            print(f"  [UPGRADE] 点击解析不一致 anatomy={bad}, 跳过升级", flush=True)
        else:
            # Release the anatomy and CSM networks first. Holding all three resident made
            # each ds457 fragment take ~90 s against the ~8 s the same model, same case and
            # same GPU contention costs in the shipped per-fragment pipeline; nnU-Net does
            # not report a fallback, it just runs slowly when the device is tight.
            del pred_ana, pred_csm
            gc.collect()
            torch.cuda.empty_cache()
            pred457 = make_predictor(find_model(
                "Dataset457*", "nnUNetTrainer*ResEncUNetXLPlans__3d_fullres"))
            frag_pre = fastpre.FragmentPreprocessor(pred457, verify_first=0)
            deadline = CASE_BUDGET_S * SAFETY
            # Spend the budget where there is evidence the CSM got it wrong. Bones whose
            # every click already owns a component need no repair, and on the 20 held-out
            # cases 18 of 40 bones (45%) are in that state -- upgrading them consumed 19%
            # of the fragment inferences to re-derive an answer the CSM already had.
            #
            # Worse than wasteful, it is a chance to do harm: ds457 segments each fragment
            # independently, so it reintroduces the overlap and coverage problems the CSM
            # decomposition does not have, and that is a plausible share of the online
            # precision 0.972 -> 0.935 and split 0.017 -> 0.124 regressions.
            cand = [a for a in BONES if clicks[a] and (ana == a).any()]
            if SKIP_CLEAN:
                # Spend the budget where the CSM demonstrably failed: bones whose every click
                # already owns a component are skipped, and the rest go first. On the 20
                # held-out cases 18 of 40 bones are already clean, so this is 45% of the
                # inferences reclaimed.
                need = [a for a in cand if not csm_ok.get(a, False)]
                order = sorted(need, key=lambda a: len(clicks[a]))
            else:
                # v5 ordering, kept exactly: every candidate bone, fewest clicks first. Under
                # a wall clock the ORDER decides which bones get reached, so reordering is a
                # behaviour change even when the set is identical -- it does not belong in a
                # submission meant to vary only the weights.
                order = sorted(cand, key=lambda a: len(clicks[a]))
            up = []
            for aid in order:
                left = deadline - (time.time() - _T0)
                if left <= 0:
                    break
                final, used, n = upgrade_bone_with_ds457(
                    final, ana, aid, BONES[aid][1], clicks[aid], pred457, frag_pre,
                    ct_arr, props, all_coords, base_index[aid], left)
                up.append(f"{BONES[aid][0]}:{n}/{len(clicks[aid])}"
                          + ("" if n else "(超预算,保留CSM)"))
                _t(f"{BONES[aid][0]} ds457 升级 {n}/{len(clicks[aid])} 碎片 ({used:.1f}s)")
            print(f"  [UPGRADE] {' '.join(up)}", flush=True)

    os.makedirs(OUTPUT, exist_ok=True)
    stem = Path(ct).name.replace(".nii.gz", "").replace(".mha", "")
    out_mha = f"{OUTPUT}/{stem}.mha"
    out = sitk.GetImageFromArray(final)
    out.CopyInformation(ct_img)
    sitk.WriteImage(out, out_mha, True)
    # GC's output interface expects ONLY the segmentation image in this folder.
    for extra in glob.glob(os.path.join(OUTPUT, "*")):
        if not extra.endswith(".mha"):
            os.remove(extra)
    print(f"  [TIMING] total={time.time() - _T0:.1f}s  ds457修复={n_refined}次  "
          f"clicks matched: {' '.join(stats)}",
          flush=True)
    print(f"wrote {out_mha}", flush=True)


if __name__ == "__main__":
    main()
