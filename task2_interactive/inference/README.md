
# Inference

The inference is done in two phases:

- Phase 1: Anatomy prediction (sacrum, hips, femur) with the baseline trained on the 456 dataset

- Phase 2: Fragment prediction with the baseline trained on the 457 dataset

---

## Inference pipeline

```
Phase 1:
                ┌─────────────────────────────┐                      filter_left_right.py
       CT   ──► │Anatomical model (Dataset456)│ ──► 5-class anatomy ──────────────────► (improved) 5-class anatomy
 + Heatmaps for │   0=bg, 1=sacrum, 2=leftHip,│      mask per voxel                                 mask per voxel
 each anatomy   │   3=rightHip, 4=femur       │
(1 + 4 channels)└─────────────────────────────┘

-------------------------------------------------------------------------------------------------------------------

Phase 2:
                ┌─────────────────────────────┐                      
       CT   ──► │                             │ ──► 2-class fragment ─────────────────┐ 
 + Heatmap for  │  Fragment model (Dataset457)│       mask per voxel                  │
 one fragment + │       0=bg, 1=fragment      │                                       │
 Heatmap for    │                             │                                       │
 background     └─────────────────────────────┘                                       │
(1 + 2 channels)                                                                      │
          │                                                                           │ 
          └───────────────────────────────────────────────────────────────────────────┘
                       repeat with another forward pass for each fragment
                              num. forward passes = num. fragments
                    skip anatomies with only one component (copy them from Phase 1) 
                                                  │
                                                  │
                                                  │ for each fragment prediction
                                                  ▼
                                    ┌────────────────────────────┐
                                    │ Mask the 2-class prediction│
                                    │ with the anatomy prediction│
                                    │ from Phase 1, i.e., zero   │
                                    │ out voxels outside anatomy │
                                    └─────────────┬──────────────┘
                                                  ▼
                                    ┌────────────────────────────┐
                                    │Copy prediction for unbroken│ 
                                    │     bones from Phase 1     │      
                                    └─────────────┬──────────────┘
                                                  ▼
                                    ┌────────────────────────────┐
                                    │  Map all prediction IDs    │ ──► challenge submission
                                    │  into PENGWIN ranges       │      (sacrum 1-50, leftHip 51-100,
                                    │                            │       rightHip 101-150, femur 151-200)
                                    └────────────────────────────┘
```

`frac_to_instance.py` covers the boxed step. The final ID-range offset stage is dataset-specific and is left to the user.

---

## Inference in Phase 1
Convert the json file to heatmaps and rename the image to end on ```_0000.mha``` using this command:

```bash
python convert_anatomy_to_nnunet_input.py 
    --json input/perpelvic-fragment-clicks.json \
    --image input/peripelvic-fracture-ct/test_case.mha \
    --output temp/nnunet_input/ 
```

This results in the following **input:**
```
├── temp/nnunet_input/          <pid>_0000.mha (CT)
                                <pid>_0001.mha (sacrum clicks)
                                <pid>_0002.mha (left hip clicks)
                                <pid>_0003.mha (right hip clicks)
                                <pid>_0004.mha (femur clicks)
```

```bash
# 1) Anatomical predictions
nnUNetv2_predict \
    -i  temp/nnunet_input/ \
    -o  temp/predictions/anatomical \
    -d  456 -c 3d_fullres -f 0 
```

**Output**: A directory of anatomy predictions saved by `nnUNetv2_predict`. Each `.mha` is an integer volume with values:

| value | meaning |
|-------|---------|
| 0     | background |
| 1     | sacrum     |
| 2     | leftHip    |
| 3     | rightHip   |
| 4     | femur      | 

We also do a very simple post-processing of these predictions to make sure that the left hip is on the left and the right hip is on the right. This script computes the centroid of the sacrum and flips all connected components with centroids left of the sacrum to the leftHip class, and all components with centroid right of the sacrum to the rightHip class. The femur and sacrum predictions remain unchanged.
```bash
python inference/filter_left_right.py \
    --input  temp/predictions/anatomical/ 
```

## Inference in Phase 2

We first create the nnUNet inputs for the Phase 2 baseline model:

```bash
python convert_fragments_to_nnunet_input.py 
    --json input/perpelvic-fragment-clicks.json \
    --image input/peripelvic-fracture-ct/test_case.mha
    --output_base temp/nnunet_input/fragments/
```
This results into multiple cases you need to process with the nnUNet fragment model saved in `sacrum`, `left_hip`, `right_hip`, and `femur` subdirectories. If a directory for a certain anatomy does not exist, it means it has only one (or 0) fragments and we will just copy the prediction from the anatomy model.
```
temp/nnunet_input/fragments/
└── sacrum
    ├── 1
    │   ├── <pid>_0000.mha
    │   ├── <pid>_0001.mha
    │   └── <pid>_0002.mha
    ├── 2
    │   ├── <pid>_0000.mha
    │   ├── <pid>_0001.mha
    │   └── <pid>_0002.mha
    └── 3
        ├── <pid>_0000.mha
        ├── <pid>_0001.mha
        └── <pid>_0002.mha
└── left_hip
│   ├── 51
│   │   ├── <pid>_0000.mha
│   │   ├── <pid>_0001.mha
│   │   └── <pid>_0002.mha
│   └── 52
│       ├── <pid>_0000.mha
│       ├── <pid>_0001.mha
│       └── <pid>_0002.mha
...

```
All we need to do now is predict for each fragment like this (pseudocode):
```bash
# 2) Fragment predictions
nnUNetv2_predict \
    -i  temp/nnunet_input/fragments/{anatomy}/{fragment}/ \
    -o  temp/predictions/fragments/{anatomy}{fragment} \
    -d  457 -c 3d_fullres -f 0 
```

We can do this with this `bash` script that iterates over all fragment inputs:
```bash
for fragment_dir in temp/nnunet_input/fragments/*/*/; do
    fragment_dir=${fragment_dir%/}
    
    # Extract anatomy and fragment ID from path
    anatomy=$(basename $(dirname "$fragment_dir"))
    fragment=$(basename "$fragment_dir")
    
    # Set output directory
    output_dir="temp/predictions/fragments/${anatomy}/${fragment}"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run nnUNet prediction
    nnUNetv2_predict \
        -i "$fragment_dir" \
        -o "$output_dir" \
        -d 457 \
        -c 3d_fullres \
        -f 0
    
    echo "Completed: $anatomy/$fragment"
done
```
Then, we postprocess these predictions to only keep the connected component that is clicked by the heatmap:
```bash
for pred in temp/predictions/fragments/*/*/*.mha; do 
    pid=$(basename $(dirname $pred))
    anatomy=$(basename $(dirname $(dirname $pred)))
    heatmap=$(ls temp/nnunet_input/fragments/$anatomy/$pid/*_0001.mha 2>/dev/null)
    if [ -f "$heatmap" ]; then
        python keep_clicked_fragment.py --input_pred "$pred" --input_heatmap "$heatmap" --output "$pred" --quiet
    else
        echo "Warning: No heatmap found for $anatomy/$pid"
    fi
done
```


Then, for anatomies with only a single fragment, we simply copy their predictions into the same interface:
```bash
    python copy_single_fragments.py \
        --anatomy_pred temp/predictions/anatomical/ \
        --fragment_input temp/nnunet_input/fragments/ \
        --fragment_output temp/predictions/fragments/ \
        --pid <pid>
```

We merge all of these predictions into one final `.mha` file containing values between `0-200` in the PENGWIN format. We use this script:

```bash
    python merge_fragment_predictions.py \
        --input temp/predictions/fragments/ \
        --output output/pelvic-fracture-segmentation/final_prediction.mha \
        --pid <pid>
```

As a final step, we fill up all `background` voxels with predicted anatomy based on the fragment from that anatomy that has the largest volume. 
```bash
    python PENGWIN2026_expand_fragments_to_anatomy.py \
        --anat_pred temp/predictions/anatomical/<pid>.mha \
        --frag_pred output/pelvic-fracture-segmentation/final_prediction.mha \
        --output output/pelvic-fracture-segmentation/final_prediction.mha 
```