"""
train.py — reproduce the submitted model (TGR-36 Day 2, coffee bean instance segmentation, 7 classes)

Two stages with the exact settings used for weights/best.pt (E6):
  stage 1 (E3): yolo26n-seg.pt -> separated-bean piles  · imgsz 1024 · 100 epochs
  stage 2 (E6): stage-1 last.pt -> touching-bean piles · imgsz  640 ·  40 epochs  (= weights/best.pt)
Common: batch 8 · AdamW lr0 0.001 lrf 0.01 · seed 0 deterministic · patience 0 (no early stop) · workers 4
The validation split points at the training images (only 21 labelled photos exist), so the model is
taken from last.pt (fixed epoch count), not best.pt. Finally exports stage 2 to NCNN fp32 imgsz 640.

Training machine (not the Pi): ultralytics==8.4.208, torch 2.15.0, Apple M-series (device mps) or cuda/cpu.

  python train.py --data1 <stage1 data.yaml> --data2 <stage2 data.yaml> --out runs_train \
      --start yolo26n-seg.pt --device mps
  python train.py --data2 <stage2 data.yaml> --stage1-weights <stage1 last.pt> --out runs_train   # stage 2 only
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = dict(batch=8, optimizer="AdamW", lr0=0.001, lrf=0.01, seed=0, deterministic=True,
              patience=0, workers=4, exist_ok=False)


def train(start, data, epochs, imgsz, out, name, device):
    from ultralytics import YOLO
    model = YOLO(str(start))
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, device=device,
                project=str(out), name=name, **COMMON)
    last = out / name / "weights" / "last.pt"
    if not last.is_file():
        raise SystemExit(f"Error: training finished without {last}")
    return last


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data1", type=Path, help="stage-1 data.yaml (separated beans); not needed with --stage1-weights")
    p.add_argument("--data2", type=Path, required=True, help="stage-2 data.yaml (touching-bean piles)")
    p.add_argument("--out", type=Path, required=True, help="new folder for training runs")
    p.add_argument("--start", type=Path, default=HERE / "yolo26n-seg.pt",
                   help="pretrained start weights (ultralytics yolo26n-seg.pt; downloaded by name if missing)")
    p.add_argument("--stage1-weights", type=Path, help="skip stage 1 and fine-tune from this last.pt")
    p.add_argument("--epochs1", type=int, default=100)
    p.add_argument("--epochs2", type=int, default=40)
    p.add_argument("--imgsz1", type=int, default=1024)
    p.add_argument("--imgsz2", type=int, default=640)
    p.add_argument("--device", default="mps", help="mps / cuda index / cpu")
    p.add_argument("--export-to", type=Path, default=None,
                   help="copy last.pt + NCNN export here as best.pt / best_ncnn_model (e.g. weights)")
    a = p.parse_args()

    for f in [a.data2] + ([a.data1] if a.data1 else []):
        if not f.is_file():
            raise SystemExit(f"Error: data.yaml not found: {f}")
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"Error: --out must be new or empty: {a.out}")
    out = a.out.resolve()

    if a.stage1_weights:
        if not a.stage1_weights.is_file():
            raise SystemExit(f"Error: not found {a.stage1_weights}")
        s1 = a.stage1_weights
    else:
        if not a.data1:
            raise SystemExit("Error: --data1 is required unless --stage1-weights is given")
        start = a.start if a.start.is_file() else Path(a.start.name)  # name only -> ultralytics downloads it
        s1 = train(start, a.data1.resolve(), a.epochs1, a.imgsz1, out, "stage1_img1024", a.device)
    s2 = train(s1, a.data2.resolve(), a.epochs2, a.imgsz2, out, "stage2_ft_img640", a.device)

    from ultralytics import YOLO
    exp = Path(YOLO(str(s2)).export(format="ncnn", imgsz=a.imgsz2, half=False, device="cpu"))
    print(f"stage 2 last.pt: {s2}\nNCNN export: {exp}")
    if a.export_to:
        a.export_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s2, a.export_to / "best.pt")
        dst = a.export_to / "best_ncnn_model"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir()
        for n in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
            shutil.copy2(exp / n, dst / n)
        print(f"copied to {a.export_to}/best.pt and {dst}")


if __name__ == "__main__":
    sys.exit(main())
