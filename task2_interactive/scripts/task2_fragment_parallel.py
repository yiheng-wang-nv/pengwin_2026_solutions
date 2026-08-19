"""Parallel driver for Task2 fragment-dataset creation (Dataset457).

The baseline create_fragment_nnUNet_dataset.py processes 340 source cases in a
single-process loop -> ~12h. Each case is independent, so we reuse its
process_case() and map the 340 cases over a process pool.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

# Resolve relative to this file, not an absolute path: the original pointed at
# $HOME/code/miccai_pengwin26 (no `dev/`), the June checkout that has since
# been deleted, so the import died with ModuleNotFoundError on the machine it was written for.
BASELINE = (Path(__file__).resolve().parents[1] / "external"
            / "PENGWIN2026_Task2_InteractiveSeg_Baseline" / "preprocessing" / "fragments")
sys.path.insert(0, str(BASELINE))
from create_fragment_nnUNet_dataset import process_case  # noqa: E402


def _worker(job):
    ct, label_path, clicks_root, out_images, out_labels, case_id = job
    try:
        process_case(ct, label_path, clicks_root, out_images, out_labels, case_id)
        return case_id, "ok"
    except Exception as e:
        return case_id, f"ERR {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--clicks_root", required=True)
    ap.add_argument("--out_images", required=True)
    ap.add_argument("--out_labels", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()

    images = Path(args.images); labels = Path(args.labels)
    clicks_root = Path(args.clicks_root)
    out_images = Path(args.out_images); out_labels = Path(args.out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(images.glob("*_0000.mha"))
    jobs = []
    for ct in ct_files:
        cid = ct.name.replace("_0000.mha", "")
        lbl = labels / f"{cid}.mha"
        if not lbl.exists():
            print(f"[WARN] missing label {cid}", flush=True); continue
        jobs.append((ct, lbl, clicks_root, out_images, out_labels, cid))

    print(f"{len(jobs)} source cases on {args.jobs} workers", flush=True)
    done = 0
    with Pool(args.jobs) as pool:
        for cid, status in pool.imap_unordered(_worker, jobs):
            done += 1
            if status != "ok" or done % 20 == 0:
                print(f"  [{done}/{len(jobs)}] {cid}: {status}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
