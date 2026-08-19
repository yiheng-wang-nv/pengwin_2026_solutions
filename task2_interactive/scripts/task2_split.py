"""Write splits_final.json for Task2 datasets (456 anatomy / 457 fragments)
using EXACTLY the same 20 val patients as Task1 (read from ds001's split).

Patient-level: all cases of a held-out patient go to val (456: one case/patient;
457: all that patient's fragment cases). Guarantees no leak + consistency with
Task1 (and with ds456 warm-started from ds001).

Run AFTER nnUNetv2_plan_and_preprocess (needs gt_segmentations/).
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path


def pid_of(case: str) -> str:
    """Extract 3-digit patient id from a case name.
    456: '001' -> '001'.  457: 'case_001_frag_Sacrum_000' -> '001'."""
    if case.startswith("case_"):
        return case.split("_")[1]
    m = re.match(r"(\d{3})", case)
    return m.group(1) if m else case


def task1_val_patients(prep: Path) -> set[str]:
    """Read the 20 val patient ids from Task1 ds001 splits_final.json."""
    f = prep / "Dataset001_PENGWIN_Anatomical" / "splits_final.json"
    s = json.load(open(f))[0]
    pids = set()
    for c in s["val"]:
        m = re.match(r"PENGWIN_(\d{3})", c)
        if m:
            pids.add(m.group(1))
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. Dataset456_PENGWIN")
    args = ap.parse_args()

    prep = Path(os.environ["nnUNet_preprocessed"])
    val_pids = task1_val_patients(prep)
    print(f"Task1 val patients ({len(val_pids)}): {sorted(val_pids)}")

    gt = prep / args.dataset / "gt_segmentations"
    if not gt.is_dir():
        raise SystemExit(f"{gt} not found (run plan_and_preprocess first)")
    cases = sorted(p.name.removesuffix(".nii.gz").removesuffix(".mha") for p in gt.glob("*"))
    cases = [c for c in cases if c]

    train = sorted(c for c in cases if pid_of(c) not in val_pids)
    val = sorted(c for c in cases if pid_of(c) in val_pids)

    out = prep / args.dataset / "splits_final.json"
    out.write_text(json.dumps([{"train": train, "val": val}], indent=2))

    tr_p = {pid_of(c) for c in train}
    va_p = {pid_of(c) for c in val}
    print(f"{args.dataset}:")
    print(f"  train: {len(train)} cases / {len(tr_p)} patients")
    print(f"  val  : {len(val)} cases / {len(va_p)} patients")
    print(f"  overlap patients (must be 0): {len(tr_p & va_p)}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
