"""Day 2 coffee-bean instance segmentation using a local YOLO NCNN export.

python infer.py --input test/ --weights weights/ --out outputs/
CPU only; no downloads, training, TTA, image voting or classifier fallback.
See README.md for export layout, preprocessing and configuration.
"""
import argparse
import io
import json
import math
import sys
import time
import warnings
from pathlib import Path

CLASSES = ("black", "broken", "good", "husk", "immature", "parchment", "shell")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CONFIG = {
    "imgsz": None,                # integer or [height, width]; defaults to export metadata
    "input_name": "in0",
    "output_name": "out0",
    "proto_name": "out1",
    "norm_scale": 1.0,            # optional training-data LANCZOS downscale
    "norm_quality": None,         # optional training-data JPEG quality
    "conf": 0.25,
    "iou": 0.5,                   # class-aware box NMS
    "max_det": 1000,
    "max_nms": 30000,
    "mask_threshold": 0.5,
    "min_area": 3,                # component pixels at ORIGINAL image size
    "hole_policy": "error",       # or "fill", an explicit, logged mask modification
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Image or folder, searched recursively")
    p.add_argument("--weights", required=True, type=Path, help="Folder with a YOLO NCNN segmenter")
    p.add_argument("--out", required=True, type=Path, help="New or empty output folder")
    p.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be positive")
    return args


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def discover_model(weights):
    """Select one NCNN text graph/binary pair, without guessing between exports."""
    if not weights.is_dir():
        raise ValueError(f"--weights must be an existing folder: {weights}")
    manifest = weights / "models.json"
    if manifest.is_file():
        data = read_json(manifest)
        if not isinstance(data, dict) or set(data) != {"segmenter"}:
            raise ValueError("models.json must contain exactly one 'segmenter' entry")
        entry = data["segmenter"]
        if not isinstance(entry, dict) or set(entry) != {"param", "bin"}:
            raise ValueError("models.json segmenter must contain 'param' and 'bin'")
        pair = []
        for key in ("param", "bin"):
            value = entry[key]
            if not isinstance(value, str):
                raise ValueError("Model paths must be strings relative to weights/")
            path = (weights / value).resolve()
            if weights.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"Model file must exist inside weights/: {value}")
            pair.append(path)
        if pair[0] == pair[1]:
            raise ValueError("NCNN graph and binary must be different files")
        return tuple(pair)
    files = sorted(p for p in weights.rglob("*") if p.is_file())
    graphs = []
    for path in files:
        with path.open("rb") as stream:
            if stream.read(32).splitlines()[:1] == [b"7767517"]:
                graphs.append(path)
    if len(graphs) != 1:
        raise ValueError("Expected one NCNN text graph; use models.json to select an export. "
                         "Export the trained YOLO segmenter to NCNN first.")
    param = graphs[0]
    conventional = param.with_suffix(".bin")
    if conventional != param and conventional.is_file():
        return param, conventional
    binaries = []
    for path in files:
        if path.parent != param.parent or path == param:
            continue
        with path.open("rb") as stream:
            sample = stream.read(4096)
        try:
            sample.decode("utf-8")
            binary = b"\x00" in sample
        except UnicodeDecodeError:
            binary = True
        if binary:
            binaries.append(path)
    if len(binaries) != 1:
        raise ValueError(f"Cannot select binary for {param}; provide models.json")
    return param, binaries[0]


def validate_config(cfg):
    size = cfg["imgsz"]
    if type(size) is int:
        size = [size, size]
    if not isinstance(size, list) or len(size) != 2 or any(
            type(v) is not int or v < 1 for v in size):
        raise ValueError("imgsz must be a positive integer or [height,width] matching the export")
    cfg["imgsz"] = size
    if cfg["hole_policy"] not in {"error", "fill"}:
        raise ValueError("hole_policy must be 'error' or 'fill'")
    for key in ("input_name", "output_name", "proto_name"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty NCNN blob name")
    if cfg["output_name"] == cfg["proto_name"]:
        raise ValueError("Predictions and prototypes need different output names")
    for key in ("conf", "iou", "mask_threshold"):
        v = cfg[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if not 0 < cfg["mask_threshold"] < 1:
        raise ValueError("mask_threshold must be strictly between 0 and 1")
    for key in ("max_det", "max_nms", "min_area"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["max_nms"] < cfg["max_det"]:
        raise ValueError("max_nms must be at least max_det")
    v = cfg["norm_scale"]
    if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
        raise ValueError("norm_scale must be positive and finite")
    q = cfg["norm_quality"]
    if q is not None and (type(q) is not int or not 1 <= q <= 100):
        raise ValueError("norm_quality must be null or an integer from 1 to 100")


def load_models(weights, threads):
    pair = discover_model(weights)
    try:
        import cv2
        import ncnn
        import numpy as np
        import yaml
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Install dependencies: python -m pip install -r requirements.txt") from exc
    cv2.setNumThreads(threads)
    metadata = {}
    meta_path = pair[0].parent / "metadata.yaml"
    if meta_path.is_file():
        metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected a mapping in {meta_path}")
    if metadata.get("task", "segment") != "segment":
        raise ValueError("Expected a YOLO instance segmentation export (task: segment)")
    cfg = dict(DEFAULT_CONFIG)
    cfg["imgsz"] = metadata.get("imgsz")
    config_path = weights / "config.json"
    if config_path.is_file():
        extra = read_json(config_path)
        if not isinstance(extra, dict):
            raise ValueError("config.json must be a JSON object")
        if set(extra) - set(cfg):
            raise ValueError(f"Unknown Day 2 config keys: {sorted(set(extra) - set(cfg))}")
        cfg.update(extra)
    validate_config(cfg)
    class_path = weights / "classes.txt"
    names = metadata.get("names")
    if class_path.is_file():
        names = [s.strip() for s in class_path.read_text(encoding="utf-8-sig").splitlines()
                 if s.strip()]
    if isinstance(names, dict):
        try:
            indexed = {int(k): v for k, v in names.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError("Metadata names need integer channel keys") from exc
        if len(indexed) != len(names) or set(indexed) != set(range(len(indexed))):
            raise ValueError("Metadata channel indices must be consecutive from 0")
        names = [indexed[i] for i in range(len(indexed))]
    if not isinstance(names, list) or not all(isinstance(s, str) for s in names):
        raise ValueError("Provide metadata names or weights/classes.txt in model channel order")
    names = [s.strip().lower() for s in names]
    if len(names) != len(CLASSES) or set(names) != set(CLASSES):
        raise ValueError(f"Expected {CLASSES} once each in model channel order; got {names}")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = threads
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_arithmetic = False
    if hasattr(net.opt, "use_bf16_storage"):
        net.opt.use_bf16_storage = False
    if net.load_param(str(pair[0])) != 0 or net.load_model(str(pair[1])) != 0:
        raise ValueError(f"NCNN could not load {pair}")
    inputs, outputs = set(net.input_names()), set(net.output_names())
    if cfg["input_name"] not in inputs or not {cfg["output_name"], cfg["proto_name"]} <= outputs:
        raise ValueError(f"Configured blobs do not match graph: inputs={sorted(inputs)}, "
                         f"outputs={sorted(outputs)}")
    print(f"Loaded YOLO segmentation NCNN model, input {cfg['imgsz']}, {threads} CPU threads",
          flush=True)
    return {"np": np, "cv2": cv2, "ncnn": ncnn, "Image": Image, "ImageOps": ImageOps,
            "net": net, "cfg": cfg, "names": names}


def run_net(m, chw):
    np, cfg = m["np"], m["cfg"]
    arr = np.ascontiguousarray(chw, dtype=np.float32)
    mat = m["ncnn"].Mat(arr).clone()  # keep storage alive independently of NumPy
    results = {}
    with m["net"].create_extractor() as ex:
        if ex.input(cfg["input_name"], mat) != 0:
            raise RuntimeError("NCNN input failed")
        for name in (cfg["output_name"], cfg["proto_name"]):
            ret, out = ex.extract(name)
            if ret != 0:
                raise RuntimeError(f"NCNN extraction failed: {name}")
            result = np.array(out, dtype=np.float32, copy=True)
            if not result.size or not np.isfinite(result).all():
                raise ValueError(f"Empty or nonfinite model output: {name}")
            results[name] = result
    return results


def load_image(m, path):
    with m["Image"].open(path) as image:
        return m["ImageOps"].exif_transpose(image).convert("RGB")


def preprocess(m, image):
    """EXIF-oriented RGB -> optional training normalization -> letterbox -> CHW /255."""
    np, cv2, cfg = m["np"], m["cv2"], m["cfg"]
    work = image
    if cfg["norm_scale"] != 1.0:
        size = tuple(max(1, round(v * cfg["norm_scale"])) for v in image.size)
        work = work.resize(size, m["Image"].Resampling.LANCZOS)
    if cfg["norm_quality"] is not None:
        with io.BytesIO() as buffer:
            work.save(buffer, "JPEG", quality=cfg["norm_quality"])
            buffer.seek(0)
            with m["Image"].open(buffer) as compressed:
                work = compressed.convert("RGB")
    target_h, target_w = cfg["imgsz"]
    w, h = work.size
    ratio = min(target_w / w, target_h / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = round((target_w - nw) / 2 - 0.1), round((target_h - nh) / 2 - 0.1)
    pixels = cv2.resize(np.asarray(work), (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = pixels
    g = {"left": left, "top": top, "width": nw, "height": nh,
         "input_w": target_w, "input_h": target_h,
         "original_w": image.width, "original_h": image.height}
    return np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32) / 255.0, g


def restore_plane(m, plane, g):
    """Undo padding/resize in one interpolation using pixel-center coordinates."""
    np, cv2 = m["np"], m["cv2"]
    ph, pw = plane.shape
    sx = g["width"] * pw / (g["original_w"] * g["input_w"])
    sy = g["height"] * ph / (g["original_h"] * g["input_h"])
    tx = (g["left"] + 0.5 * g["width"] / g["original_w"]) * pw / g["input_w"] - 0.5
    ty = (g["top"] + 0.5 * g["height"] / g["original_h"]) * ph / g["input_h"] - 0.5
    transform = np.asarray([[sx, 0, tx], [0, sy, ty]], dtype=np.float32)
    return cv2.warpAffine(plane, transform, (g["original_w"], g["original_h"]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def nms(np, boxes, scores, classes, threshold, max_det):
    order = np.argsort(-scores, kind="stable")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        intersection = np.maximum(0, np.minimum(boxes[i, 2:], boxes[rest, 2:]) -
                                  np.maximum(boxes[i, :2], boxes[rest, :2])).prod(axis=1)
        overlap = intersection / np.maximum(area[i] + area[rest] - intersection, 1e-9)
        order = rest[(classes[rest] != classes[i]) | (overlap <= threshold)]
    return np.asarray(keep, dtype=np.int64)


def yolo_candidates(m, outputs, g):
    """Yield class/mask pairs in descending confidence, one full-size mask at a time.

    Raw NCNN layout: predictions (4+nc+nm,N), prototypes (nm,Hm,Wm).
    No objectness column, embedded NMS or batch dimension.
    """
    np, cfg = m["np"], m["cfg"]
    pred, proto = outputs[cfg["output_name"]], outputs[cfg["proto_name"]]
    nc = len(m["names"])
    if proto.ndim != 3 or min(proto.shape) < 1:
        raise ValueError(f"Expected CHW mask prototypes, got {proto.shape}")
    nm, ph, pw = proto.shape
    channels = 4 + nc + nm
    if pred.ndim != 2 or pred.shape[0] != channels:
        raise ValueError(f"Expected YOLO raw output ({channels},N), got {pred.shape}; "
                         "detection-only and NMS-packed/end-to-end outputs are unsupported")
    class_scores = pred[4:4 + nc]
    if np.any(class_scores < 0) or np.any(class_scores > 1):
        raise ValueError("YOLO class scores must be probabilities in [0,1]")
    classes, scores = class_scores.argmax(axis=0), class_scores.max(axis=0)
    indices = np.flatnonzero((scores > cfg["conf"]) & (pred[2] > 0) & (pred[3] > 0))
    if len(indices) > cfg["max_nms"]:
        warnings.warn(f"Capping NMS candidates at {cfg['max_nms']}")
        indices = indices[np.argsort(-scores[indices], kind="stable")[:cfg["max_nms"]]]
    rows = pred[:, indices].T
    boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2,
                            rows[:, :2] + rows[:, 2:4] / 2), axis=1)
    keep = nms(np, boxes, scores[indices], classes[indices], cfg["iou"], cfg["max_det"])
    threshold = math.log(cfg["mask_threshold"] / (1 - cfg["mask_threshold"]))
    for k in keep:
        logits = (rows[k, 4 + nc:] @ proto.reshape(nm, -1)).reshape(ph, pw)
        plane = restore_plane(m, logits, g)
        box = boxes[k].copy()
        box[[0, 2]] = (box[[0, 2]] - g["left"]) * g["original_w"] / g["width"]
        box[[1, 3]] = (box[[1, 3]] - g["top"]) * g["original_h"] / g["height"]
        x1, x2 = np.clip(np.ceil(box[[0, 2]]), 0, g["original_w"]).astype(int)
        y1, y2 = np.clip(np.ceil(box[[1, 3]]), 0, g["original_h"]).astype(int)
        mask = np.zeros(plane.shape, dtype=bool)
        mask[y1:y2, x1:x2] = plane[y1:y2, x1:x2] > threshold
        yield m["names"][int(classes[indices[k]])], mask


def add_components(m, instances, objects, name, mask):
    """Make masks single-owner, split disconnected pieces, derive full polygons."""
    np, cv2, cfg = m["np"], m["cv2"], m["cfg"]
    mask = mask & (instances == 0)  # higher-confidence predictions own overlaps
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count > 2:
        warnings.warn(f"Splitting disconnected {name} mask into {count - 1} components")
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < cfg["min_area"]:
            warnings.warn(f"Dropping {name} component with {area} pixels "
                          f"(min_area={cfg['min_area']})")
            continue
        x, y, w, h = (int(v) for v in stats[component, :4])
        region = (labels[y:y + h, x:x + w] == component).astype(np.uint8)
        contours, hierarchy = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        exterior = [i for i, entry in enumerate(hierarchy[0]) if entry[3] == -1]
        if len(exterior) != 1:
            raise ValueError(f"{name} component has multiple exterior contours")
        contour = contours[exterior[0]]
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            warnings.warn(f"Dropping degenerate {name} component ({area} pixels): "
                          "cannot form a polygon with positive area")
            continue
        raster = np.zeros(region.shape, dtype=np.uint8)
        cv2.fillPoly(raster, [contour], 1)
        if not np.array_equal(raster, region):
            if cfg["hole_policy"] == "error":
                raise ValueError(f"{name} mask has holes that one polygon cannot represent. "
                                 "Set hole_policy='fill' in config.json only if intentional.")
            added = (raster != 0) & (region == 0)
            occupied = instances[y:y + h, x:x + w] != 0
            others = (labels[y:y + h, x:x + w] != 0) & (region == 0)
            if np.any(added & (occupied | others)):
                raise ValueError("Filling a mask hole would overwrite another bean")
            warnings.warn(f"Filling {int(added.sum())} hole pixels in {name} mask")
            region = raster
        instance_id = len(objects) + 1
        if instance_id > 65535:
            raise ValueError("Too many instances for a uint16 PNG")
        instances[y:y + h, x:x + w][region != 0] = instance_id
        points = contour[:, 0, :] + np.asarray([x, y], dtype=np.int32)
        objects.append({"id": instance_id, "class": name, "polygon": points.tolist()})


def validate_result(m, size, instances, objects):
    np, cv2 = m["np"], m["cv2"]
    w, h = size
    if instances.dtype != np.uint16 or instances.shape != (h, w):
        raise ValueError("Mask must be an original-size, single-channel uint16 array")
    ids = [o["id"] for o in objects]
    if ids != list(range(1, len(objects) + 1)):
        raise ValueError("Object IDs must be consecutive from 1")
    counts = np.bincount(instances.reshape(-1), minlength=len(objects) + 1)
    if set(int(i) for i in np.flatnonzero(counts)) - {0} != set(ids):
        raise ValueError("PNG IDs and JSON IDs do not agree")
    for obj in objects:
        if set(obj) != {"id", "class", "polygon"} or obj["class"] not in CLASSES:
            raise ValueError("Invalid object fields/class")
        points = np.asarray(obj["polygon"])
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2 \
                or not np.issubdtype(points.dtype, np.integer):
            raise ValueError("Polygon needs at least three integer [x,y] points")
        if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
            raise ValueError("Polygon coordinates are outside the original image")
        x, y, rw, rh = cv2.boundingRect(points.astype(np.int32))
        raster = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(raster, [(points - [x, y]).astype(np.int32)], 1)
        region = instances[y:y + rh, x:x + rw] == obj["id"]
        if not np.array_equal(raster != 0, region) or int(region.sum()) != counts[obj["id"]]:
            raise ValueError(f"Polygon and instance mask disagree for ID {obj['id']}")


def predict_image(m, path):
    image = load_image(m, path)
    chw, g = preprocess(m, image)
    outputs = run_net(m, chw)
    instances = m["np"].zeros((image.height, image.width), dtype=m["np"].uint16)
    objects = []
    for name, mask in yolo_candidates(m, outputs, g):
        add_components(m, instances, objects, name, mask)
    validate_result(m, image.size, instances, objects)
    return image, instances, objects


def save_outputs(m, output, path, image, instances, objects):
    np, Image = m["np"], m["Image"]
    target = output / "seg_mask" / f"{path.stem}.png"
    Image.fromarray(instances).save(target)
    with Image.open(target) as saved:
        decoded = np.asarray(saved)
        if decoded.ndim != 2 or not np.array_equal(decoded, instances):
            raise ValueError(f"Written PNG failed ID validation: {target}")
    palette = np.zeros((len(objects) + 1, 3), dtype=np.uint8)
    for obj in objects:
        i = obj["id"]
        palette[i] = [(i * 67) % 192 + 64, (i * 131) % 192 + 64, (i * 43) % 192 + 64]
    rgb = np.array(image, copy=True)
    foreground = instances != 0
    rgb[foreground] = (rgb[foreground].astype(np.uint16) +
                       palette[instances[foreground]].astype(np.uint16)) // 2
    Image.fromarray(rgb).save(output / "vis" / f"{path.stem}.png")


def collect_images(source):
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    images = list(p for p in source.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) \
        if source.is_dir() else [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    images.sort(key=lambda p: (p.name.casefold(), str(p)))
    if not images:
        raise ValueError(f"No supported images (.jpg/.jpeg/.png) in {source}")
    for label, values in (("filenames", [p.name.casefold() for p in images]),
                          ("stems", [p.stem.casefold() for p in images])):
        if len(values) != len(set(values)):
            raise ValueError(f"Input images must have unique {label}, including subfolders")
    return images


def check_output(source, weights, output, images):
    if not weights.is_dir():
        raise ValueError(f"--weights must be an existing folder: {weights}")
    for folder in ([source] if source.is_dir() else []) + [weights]:
        if output == folder or output in folder.parents or folder in output.parents:
            raise ValueError("Output must be separate from input and weights; "
                             "neither folder may contain the other")
    for path in images:
        if output == path.resolve() or output in path.resolve().parents:
            raise ValueError(f"Output would contain input image: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty folder: {output}")


def run(args):
    source, weights, output = (p.expanduser().resolve()
                               for p in (args.input, args.weights, args.out))
    images = collect_images(source)
    check_output(source, weights, output, images)
    model = load_models(weights, args.threads)
    output.mkdir(parents=True, exist_ok=True)
    (output / "seg_mask").mkdir()
    (output / "vis").mkdir()
    predictions, elapsed = [], 0.0
    for index, path in enumerate(images, 1):
        start = time.perf_counter()
        try:
            image, instances, objects = predict_image(model, path)
            save_outputs(model, output, path, image, instances, objects)
        except Exception as exc:
            raise RuntimeError(f"Failed on {path.name}: {exc}") from exc
        predictions.append({"image_name": path.name, "objects": objects})
        seconds = time.perf_counter() - start
        elapsed += seconds
        print(f"[{index}/{len(images)}] {path.name}: {len(objects)} beans, {seconds:.2f} s",
              flush=True)
    target = output / "predictions.json"
    with target.open("w", encoding="utf-8") as stream:
        json.dump(predictions, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    if read_json(target) != predictions or len(predictions) != len(images):
        raise RuntimeError("predictions.json failed validation")
    print(f"Saved {len(images)} images to {output}; mean {elapsed / len(images):.2f} s/image "
          "(image I/O and validation included; model loading excluded)")


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
