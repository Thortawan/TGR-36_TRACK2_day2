"""
infer_seg_ncnn.py — ต้นแบบ infer Day 2 (YOLO-seg NCNN · ncnn + numpy + opencv + pillow · ไม่ใช้ torch/ultralytics)
ทำผลตามสเปก DAY2_INFER_SPEC.md: <out>/seg_mask/<stem>.png (uint16 ช่องเดียว ขนาดภาพหลัง EXIF) · predictions.json · vis/

ต่างจาก infer.py ของเพื่อน (เหตุผลใน LOG ข้อ 8):
  - หลังทายทำเฉพาะในกรอบของแต่ละเม็ด (ขยาย logit ด้วย bilinear เป็นความละเอียดเต็มในกรอบ แล้วตัดที่ 0) → เร็ว
  - NMS เลือกได้ว่าข้ามคลาส (--agnostic 1) · iou เริ่ม 0.7 เท่าท่อ .pt
  - เม็ดหนึ่งเก็บก้อนใหญ่สุด + เติมรู (เฉพาะพิกเซลที่ยังไม่มีเจ้าของ) · พิมพ์เตือนเมื่อทิ้งส่วน · ไม่หยุดทั้งรัน
  - --refine color: แก้แถบขอบ ±--band px ด้วยสีของภาพ (พิกเซลที่ "ไม่ใช่กระดาษ/เงา") · ค่าเริ่ม none
  - seg_mask uint16 PNG บีบอัดระดับ 1 · vis เป็น JPEG ย่อ --vis-scale
error ที่หยุดทั้งรัน (exit 1): input/weights ไม่พบ · ไม่มีภาพ · อ่านภาพไม่ได้ · ผลผิดรูปแบบ

ตัวส่ง: ค่าเริ่ม = refine color band 20 (distance transform) + เขียนไฟล์เบื้องหลัง + เขียนทับ --out ได้ → python infer.py --input test/ --weights weights/ --out outputs/
  python infer.py --input <ภาพ> --weights <โฟลเดอร์ NCNN> --out <โฟลเดอร์> \
      --conf 0.25 --iou 0.7 --agnostic 1 --refine none --threads 4
"""
import argparse
import json
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import ncnn
import numpy as np
import yaml
from PIL import Image, ImageOps

CLASSES = ("black", "broken", "good", "husk", "immature", "parchment", "shell")
EXT = {".jpg", ".jpeg", ".png"}


def warn(msg):
    print(f"  [เตือน] {msg}", file=sys.stderr)


def load_model(wdir, threads):
    params = sorted(wdir.rglob("*.ncnn.param")) or sorted(wdir.rglob("*.param"))
    if len(params) != 1:
        raise SystemExit(f"Error: ต้องมีไฟล์ .param 1 ไฟล์ใน {wdir} (พบ {len(params)})")
    param = params[0]
    binf = param.with_suffix(".bin")
    meta = yaml.safe_load((param.parent / "metadata.yaml").read_text())
    names = [meta["names"][i] for i in range(len(meta["names"]))]
    if tuple(n.lower() for n in names) != CLASSES:
        raise SystemExit(f"Error: ลำดับคลาสในโมเดล {names} ≠ {CLASSES}")
    imgsz = meta["imgsz"]
    imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = threads
    net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = False
    if net.load_param(str(param)) != 0 or net.load_model(str(binf)) != 0:
        raise SystemExit(f"Error: โหลด NCNN ไม่ได้ {param}")
    return net, imgsz


def letterbox(rgb, size):
    """เหมือน ultralytics LetterBox(auto=False, center=True) + cv2.INTER_LINEAR"""
    th, tw = size
    h, w = rgb.shape[:2]
    r = min(th / h, tw / w)
    nw, nh = round(w * r), round(h * r)
    left, top = round((tw - nw) / 2 - 0.1), round((th - nh) / 2 - 0.1)
    canvas = np.full((th, tw, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def run_net(net, canvas):
    x = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32) / 255.0
    mat = ncnn.Mat(x).clone()
    with net.create_extractor() as ex:
        ex.input("in0", mat)
        _, o0 = ex.extract("out0")
        _, o1 = ex.extract("out1")
    return np.array(o0, np.float32, copy=True), np.array(o1, np.float32, copy=True)


def nms(boxes, scores, cls, thr, agnostic, max_det):
    off = boxes + (0 if agnostic else cls[:, None] * 7680.0)  # แบบ ultralytics: เลื่อนกรอบตามคลาส
    order = np.argsort(-scores, kind="stable")
    area = (off[:, 2] - off[:, 0]) * (off[:, 3] - off[:, 1])
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        wh = np.clip(np.minimum(off[i, 2:], off[rest, 2:]) - np.maximum(off[i, :2], off[rest, :2]), 0, None)
        inter = wh[:, 0] * wh[:, 1]
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= thr]
    return np.array(keep, int)


def paper_lab(lab_small):
    """สีกระดาษ = ค่ากลาง Lab ของขอบภาพ (ภาพย่อ)"""
    b = np.concatenate([lab_small[:8].reshape(-1, 3), lab_small[-8:].reshape(-1, 3),
                        lab_small[:, :8].reshape(-1, 3), lab_small[:, -8:].reshape(-1, 3)])
    return np.median(b, axis=0)


def predict(net, imgsz, rgb, a):
    H, W = rgb.shape[:2]
    canvas, r, left, top = letterbox(rgb, imgsz)
    pred, proto = run_net(net, canvas)
    nc = len(CLASSES)
    nm, ph, pw = proto.shape
    sc = pred[4:4 + nc]
    cls, conf = sc.argmax(0), sc.max(0)
    idx = np.flatnonzero(conf > a.conf)
    if idx.size == 0:
        return np.zeros((H, W), np.uint16), []
    xywh = pred[:4, idx].T
    boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    keep = nms(boxes, conf[idx], cls[idx], a.iou, a.agnostic, a.max_det)
    idx, boxes = idx[keep], boxes[keep]
    coef = pred[4 + nc:, idx].T  # (K, nm)
    stride = imgsz[1] / pw  # 4
    ob = (boxes - [left, top, left, top]) / r  # กรอบในพิกัดภาพเต็ม
    ob = np.clip(ob, 0, [W, H, W, H])
    paper = None
    if a.refine == "color":
        small = cv2.resize(rgb, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        paper = paper_lab(cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32))
        kband = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.band + 1, 2 * a.band + 1))
    inst = np.zeros((H, W), np.uint16)
    objects = []
    for k in range(len(idx)):  # เรียง conf มาก→น้อย อยู่แล้ว (จาก nms)
        x1, y1 = int(np.floor(ob[k, 0])), int(np.floor(ob[k, 1]))
        x2, y2 = int(np.ceil(ob[k, 2])), int(np.ceil(ob[k, 3]))
        if a.refine == "color":  # เผื่อขอบให้แถบปรับขยายออกนอกกรอบได้
            x1, y1 = max(0, x1 - a.band), max(0, y1 - a.band)
            x2, y2 = min(W, x2 + a.band), min(H, y2 + a.band)
        cw, ch = x2 - x1, y2 - y1
        if cw < 2 or ch < 2:
            continue
        logit = (coef[k] @ proto.reshape(nm, -1)).reshape(ph, pw)
        # พิกเซล (u,v) ในครอป → จุดกึ่งกลางพิกเซลภาพเต็ม → พิกัด input → พิกัด proto
        sx = r / stride
        M = np.array([[sx, 0, ((x1 + 0.5) * r + left) / stride - 0.5],
                      [0, sx, ((y1 + 0.5) * r + top) / stride - 0.5]], np.float32)
        crop = cv2.warpAffine(logit, M, (cw, ch), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        m = (crop > 0).astype(np.uint8)
        # ตัดนอกกรอบ (แบบ ultralytics crop_mask) — ยกเว้นตอน refine ที่เผื่อขอบไว้
        bx1, by1 = int(np.floor(ob[k, 0])) - x1, int(np.floor(ob[k, 1])) - y1
        bx2, by2 = int(np.ceil(ob[k, 2])) - x1, int(np.ceil(ob[k, 3])) - y1
        box_m = np.zeros_like(m)
        box_m[max(0, by1):by2, max(0, bx1):bx2] = 1
        m &= box_m
        if a.refine == "color" and m.any():
            lab = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2LAB).astype(np.float32)
            dist = np.sqrt(((lab - paper) ** 2).sum(-1))
            chroma = np.sqrt(((lab[..., 1:] - paper[1:]) ** 2).sum(-1))
            fg = (dist > a.fg_dist) & ~((chroma < a.shadow_chroma) & (lab[..., 0] > paper[0] - 60))
            if a.band_method == "dt":  # ระยะถึงขอบด้วย distance transform (เร็วกว่าวงรี ~9 เท่า)
                din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
                dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
                band = (din <= a.band + 0.5) & (dout <= a.band + 0.5)
            else:
                band = cv2.dilate(m, kband).astype(bool) & ~cv2.erode(m, kband).astype(bool)
            m = np.where(band, fg, m.astype(bool)).astype(np.uint8)
        owned = inst[y1:y2, x1:x2] != 0
        m[owned] = 0
        n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            continue
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if n > 2:
            lost = int(st[1:, cv2.CC_STAT_AREA].sum() - st[big, cv2.CC_STAT_AREA])
            if lost > a.min_area:
                warn(f"เม็ด {CLASSES[cls[idx[k]]]} แตก {n - 1} ส่วน ทิ้งส่วนเล็ก {lost} px (เก็บก้อนใหญ่สุด)")
        m = (lbl == big).astype(np.uint8)
        if int(st[big, cv2.CC_STAT_AREA]) < a.min_area:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 3:
            continue
        filled = np.zeros_like(m)
        cv2.drawContours(filled, [c], -1, 1, -1)  # เติมรู
        if (filled.astype(bool) & owned).any():
            warn("รูในเม็ดมีเม็ดอื่นอยู่ — mask ไม่รวมพิกเซลนั้น (polygon ขอบนอกครอบอยู่)")
        filled[owned] = 0
        oid = len(objects) + 1
        inst[y1:y2, x1:x2][filled.astype(bool)] = oid
        pts = (c[:, 0, :] + [x1, y1]).astype(int)
        objects.append({"id": oid, "class": CLASSES[int(cls[idx[k]])], "polygon": pts.tolist()})
    return inst, objects


def save_vis(path, rgb, inst, scale):
    if scale != 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        inst = cv2.resize(inst, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    n = int(inst.max())
    pal = np.zeros((n + 1, 3), np.uint16)
    ids = np.arange(1, n + 1)
    pal[1:] = np.stack([(ids * 67) % 192 + 64, (ids * 131) % 192 + 64, (ids * 43) % 192 + 64], 1)
    out = rgb.copy()
    fg = inst != 0
    out[fg] = ((out[fg].astype(np.uint16) + pal[inst[fg]]) // 2).astype(np.uint8)
    edge = cv2.morphologyEx((inst > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    out[edge > 0] = (255, 0, 0)
    cv2.imwrite(str(path), out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--agnostic", type=int, choices=[0, 1], default=1)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--min-area", type=int, default=50, help="px ภาพเต็ม · ก้อนเล็กกว่านี้ไม่นับเป็นเม็ด")
    ap.add_argument("--refine", choices=["none", "color"], default="color")
    ap.add_argument("--band", type=int, default=20, help="(refine color) ความกว้างแถบขอบ px ภาพเต็ม")
    ap.add_argument("--band-method", choices=["morph", "dt"], default="dt",
                    help="(refine color) แถบขอบด้วย dilate/erode วงรี (morph) หรือ distance transform (dt เร็วกว่า)")
    ap.add_argument("--fg-dist", type=float, default=18.0, help="(refine color) ระยะ Lab จากกระดาษที่นับเป็นเม็ด")
    ap.add_argument("--shadow-chroma", type=float, default=6.0, help="(refine color) เทาใกล้กระดาษ = เงา")
    ap.add_argument("--vis-scale", type=float, default=0.5)
    ap.add_argument("--async-write", type=int, choices=[0, 1], default=1,
                    help="1 = เขียน seg_mask/vis ในเธรดเบื้องหลังขณะทายภาพถัดไป (ผลเหมือนเดิมทุกไบต์)")
    ap.add_argument("--allow-existing-out", type=int, choices=[0, 1], default=1,
                    help="1 = --out มีอยู่แล้วได้ (ลบ seg_mask/ vis/ predictions.json เดิมแล้วเขียนใหม่)")
    a = ap.parse_args()

    if not a.input.exists():
        raise SystemExit(f"Error: ไม่พบ input {a.input}")
    if not a.weights.is_dir():
        raise SystemExit(f"Error: ไม่พบโฟลเดอร์ weights {a.weights}")
    files = sorted([p for p in a.input.rglob("*") if p.is_file() and p.suffix.lower() in EXT]
                   if a.input.is_dir() else [a.input], key=lambda p: p.name.casefold())
    if not files:
        raise SystemExit(f"Error: ไม่มีภาพ .jpg/.jpeg/.png ใน {a.input}")
    stems = [p.stem.casefold() for p in files]
    if len(set(stems)) != len(stems):
        raise SystemExit("Error: ชื่อภาพ (stem) ซ้ำกัน")
    if a.out.exists() and not a.out.is_dir():
        raise SystemExit(f"Error: --out ไม่ใช่โฟลเดอร์: {a.out}")
    if a.out.exists() and any(a.out.iterdir()):
        if not a.allow_existing_out:
            raise SystemExit(f"Error: --out ต้องเป็นโฟลเดอร์ใหม่หรือว่าง: {a.out}")
        if a.input.resolve() == a.out.resolve() or a.out.resolve() in a.input.resolve().parents:
            raise SystemExit("Error: --out ซ้อนกับ --input")
        # ลบเฉพาะผลเดิมของสคริปต์นี้ (seg_mask/, vis/, predictions.json) ไม่แตะไฟล์อื่น
        for sub in ("seg_mask", "vis"):
            if (a.out / sub).is_dir():
                shutil.rmtree(a.out / sub)
        (a.out / "predictions.json").unlink(missing_ok=True)
        print(f"เขียนทับผลเดิมใน {a.out}", flush=True)
    cv2.setNumThreads(a.threads)
    net, imgsz = load_model(a.weights, a.threads)
    (a.out / "seg_mask").mkdir(parents=True, exist_ok=True)
    (a.out / "vis").mkdir(exist_ok=True)
    run_net(net, np.full((imgsz[0], imgsz[1], 3), 114, np.uint8))  # warm-up
    preds, times = [], []

    def write_outputs(stem, rgb, inst):
        if not cv2.imwrite(str(a.out / "seg_mask" / f"{stem}.png"), inst, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise RuntimeError(f"เขียน seg_mask ไม่ได้ {stem}")
        save_vis(a.out / "vis" / f"{stem}.jpg", rgb, inst, a.vis_scale)

    # --async-write 1: เขียนไฟล์ภาพก่อนหน้าในเธรดเบื้องหลัง ขณะอ่าน/ทายภาพถัดไป (cv2.imwrite ปล่อย GIL)
    # ไฟล์ผลเหมือนโหมดปกติทุกไบต์ · คิวไม่เกิน 2 ภาพ · เวลารายภาพ = ช่วงห่างระหว่างภาพ (รวมรอเขียนตอนท้าย)
    pool = ThreadPoolExecutor(max_workers=1) if a.async_write else None
    pending, starts = [], []

    def drain(keep):
        while len(pending) > keep:
            try:
                pending.pop(0).result()
            except Exception as e:  # noqa: BLE001
                pool.shutdown(wait=True, cancel_futures=True)
                raise SystemExit(f"Error: {e}")

    for i, f in enumerate(files, 1):
        t0 = time.perf_counter()
        starts.append(t0)
        try:
            with Image.open(f) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Error: อ่านภาพไม่ได้ {f.name}: {e}")
        inst, objects = predict(net, imgsz, rgb, a)
        if pool:
            drain(1)
            pending.append(pool.submit(write_outputs, f.stem, rgb, inst))
        else:
            try:
                write_outputs(f.stem, rgb, inst)
            except RuntimeError as e:
                raise SystemExit(f"Error: {e}")
        preds.append({"image_name": f.name, "objects": objects})
        dt = time.perf_counter() - t0
        if not pool:
            times.append(dt)
        print(f"[{i}/{len(files)}] {f.name}: {len(objects)} เม็ด {dt:.2f} s"
              + (" (ไม่รวมเขียนไฟล์ที่ทำเบื้องหลัง)" if pool else ""), flush=True)
    if pool:
        drain(0)
        pool.shutdown(wait=True)
        ends = starts[1:] + [time.perf_counter()]
        times = [e - s for s, e in zip(starts, ends)]
    target = a.out / "predictions.json"
    with target.open("w", encoding="utf-8") as fp:  # รูปแบบเดียวกับ infer.py เพื่อน: 1 ภาพ/1 เม็ด ต่อบรรทัด
        fp.write("[\n")
        for i, p in enumerate(preds):
            fp.write('  {\n    "image_name": ' + json.dumps(p["image_name"], ensure_ascii=False)
                     + ',\n    "objects": [')
            for j, o in enumerate(p["objects"]):
                fp.write(("\n" if j == 0 else ",\n") + "      " + json.dumps(o, ensure_ascii=False, allow_nan=False))
            fp.write(("\n    " if p["objects"] else "") + "]\n  }" + (",\n" if i + 1 < len(preds) else "\n"))
        fp.write("]\n")
    if json.loads(target.read_text(encoding="utf-8")) != preds:
        raise SystemExit("Error: predictions.json อ่านกลับไม่ตรงกับผล")
    print(f"บันทึก {len(files)} ภาพ → {a.out} · median {statistics.median(times):.2f} s/ภาพ "
          f"(รวมอ่าน/เขียน ไม่รวมโหลดโมเดล) · imgsz {imgsz} conf {a.conf} iou {a.iou} "
          f"agnostic {a.agnostic} refine {a.refine}" + (" · async-write" if pool else ""))


if __name__ == "__main__":
    main()
