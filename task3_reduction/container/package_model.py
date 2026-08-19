#!/usr/bin/env python3
"""Compact model tar for the N-member centroid ensemble image.

Generalises docker_task3_dual/package_model.py from two members to any number. Each member is
one AssemblyNet checkpoint plus the five-fold residual trained on THAT base's own predictions --
a residual head is tied to the base it was cached from, so the pairing is not interchangeable
and the manifest records it explicitly, along with the weights the score was validated at.

Checkpoints are stripped to their state_dict (optimizer state is dead weight in a submission)
and every file is sha256'd so the container can verify what it loaded.
"""
from __future__ import annotations
import argparse, hashlib, json, tarfile, tempfile
from pathlib import Path
import torch


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            d.update(b)
    return d.hexdigest()


def strip(src: Path, dst: Path) -> None:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if "state_dict" not in ck:
        raise SystemExit(f"no state_dict in {src}")
    torch.save({"state_dict": ck["state_dict"]}, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--member", action="append", required=True, metavar="TAG:CKPT:RESIDUAL_DIR",
                   help="repeat once per member, in the SAME order as --w-rot/--w-trans")
    p.add_argument("--w-rot", type=float, nargs="+", required=True)
    p.add_argument("--w-trans", type=float, nargs="+", required=True)
    p.add_argument("--name", default="task3_multi_base_centroid_ensemble")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    members = []
    for spec in args.member:
        tag, ckpt, resid = spec.split(":", 2)
        members.append((tag, Path(ckpt), Path(resid)))
    n = len(members)
    if len(args.w_rot) != n or len(args.w_trans) != n:
        raise SystemExit(f"{n} members but {len(args.w_rot)} rot / {len(args.w_trans)} trans weights")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task3-multi-") as tmp:
        root = Path(tmp)
        files, sources = {}, {}
        for tag, ckpt, resid in members:
            if not ckpt.exists():
                raise SystemExit(f"missing checkpoint {ckpt}")
            folds = sorted(resid.glob("fold*_last.pt"))
            if len(folds) != 5:
                raise SystemExit(f"expected five residual checkpoints in {resid}, got {len(folds)}")
            strip(ckpt, root / f"model_{tag}.ckpt")
            rdir = root / f"residual_{tag}"
            rdir.mkdir()
            for f in folds:
                (rdir / f.name).write_bytes(f.read_bytes())
            files[f"model_{tag}.ckpt"] = sha256(root / f"model_{tag}.ckpt")
            files.update({f"residual_{tag}/{f.name}": sha256(f) for f in folds})
            sources[tag] = {"assemblynet_source": str(ckpt),
                            "assemblynet_sha256": sha256(ckpt),
                            "residual_source": str(resid)}
        manifest = {"name": args.name,
                    "order": [t for t, _, _ in members],
                    "w_rot": args.w_rot, "w_trans": args.w_trans,
                    "members": sources, "files": files}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with tarfile.open(args.out, "w:gz") as tar:
            for tag, _, _ in members:
                tar.add(root / f"model_{tag}.ckpt", arcname=f"model_{tag}.ckpt")
                tar.add(root / f"residual_{tag}", arcname=f"residual_{tag}")
            tar.add(root / "manifest.json", arcname="manifest.json")
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")
    print(f"sha256 {sha256(args.out)}")
    print(f"members: {[t for t,_,_ in members]}")
    print(f"w_rot   {args.w_rot}\nw_trans {args.w_trans}")


if __name__ == "__main__":
    main()
