#! python3
import os
import glob
import json
import pickle as pkl
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2


DEFAULT_KEY = "side_stage_classifier"


def _ensure_dir(d: str) -> None:
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)


def _extract_image_from_obs(obs: Dict[str, Any], key: str) -> np.ndarray:
    """
    Robustly fetch image array from obs.
    Supports:
      - obs[key]
      - obs["images"][key]
      - obs["image"][key]
    """
    if key in obs:
        img = obs[key]
    elif "images" in obs and isinstance(obs["images"], dict) and key in obs["images"]:
        img = obs["images"][key]
    elif "image" in obs and isinstance(obs["image"], dict) and key in obs["image"]:
        img = obs["image"][key]
    else:
        raise KeyError(f"Cannot find image key='{key}' in obs. keys={list(obs.keys())}")

    img = np.asarray(img)

    # Handle ChunkingWrapper: image often is (T,H,W,C) with T=1
    # Also handle accidental extra singleton dims.
    while img.ndim > 3 and img.shape[0] == 1:
        img = np.squeeze(img, axis=0)

    if img.ndim != 3:
        raise ValueError(f"Expected image ndim=3 after squeeze, got shape={img.shape}")

    # Normalize dtype to uint8 RGB
    if img.dtype != np.uint8:
        # sometimes float in [0,1]
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    return img


def cmd_extract(args: argparse.Namespace) -> None:
    _ensure_dir(args.out_dir)

    # Load all pkl from classifier_data
    pkl_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl files found in {args.input_dir}")

    images: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    for path in pkl_paths:
        try:
            with open(path, "rb") as f:
                data = pkl.load(f)
        except EOFError:
            print(f"[!] Skip empty/broken file: {path}")
            continue

        # record_success_fail 保存的是 transition list :contentReference[oaicite:8]{index=8}
        for i, trans in enumerate(data):
            obs = trans["observations"] if not args.use_next_obs else trans["next_observations"]
            try:
                img = _extract_image_from_obs(obs, args.key)
            except Exception as e:
                if args.skip_bad:
                    continue
                raise e

            images.append(img)
            meta.append(
                {
                    "src_file": os.path.basename(path),
                    "transition_index": i,
                    "from": "next_observations" if args.use_next_obs else "observations",
                }
            )

            if args.limit > 0 and len(images) >= args.limit:
                break
        if args.limit > 0 and len(images) >= args.limit:
            break

    images_arr = np.stack(images, axis=0)  # (N,H,W,C)
    if args.shuffle:
        perm = np.random.permutation(len(images_arr))
        images_arr = images_arr[perm]
        meta = [meta[i] for i in perm]

    out_images = os.path.join(args.out_dir, "images.npy")
    out_meta = os.path.join(args.out_dir, "meta.json")
    np.save(out_images, images_arr)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "key": args.key,
                "num_samples": int(images_arr.shape[0]),
                "image_shape": list(images_arr.shape[1:]),
                "items": meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # init labels with -1 (unlabeled)
    out_labels = os.path.join(args.out_dir, "labels.npy")
    if not os.path.exists(out_labels):
        labels = -np.ones((images_arr.shape[0],), dtype=np.int32)
        np.save(out_labels, labels)

    print(f"[+] extracted: {images_arr.shape} -> {out_images}")
    print(f"[+] meta saved -> {out_meta}")
    print(f"[+] labels init/resume -> {out_labels}")


def _bgr(img_rgb_uint8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)


def _draw_hud(
    img_bgr: np.ndarray,
    idx: int,
    n: int,
    label: int,
    counts: Dict[int, int],
    num_classes: int,
    meta_item: Dict[str, Any] | None,
) -> np.ndarray:
    canvas = img_bgr.copy()

    def put(line: str, y: int) -> None:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    put(f"Index: {idx+1}/{n}", 25)
    put(f"Label: {label}", 50)

    # meta
    if meta_item is not None:
        put(f"Src: {meta_item.get('src_file', '')}", 75)
        put(f"TransIdx: {meta_item.get('transition_index', '')}", 100)

    # counts
    base_y = 130
    for c in range(num_classes):
        put(f"Count[{c}] = {counts.get(c, 0)}", base_y + 22 * c)

    put("Keys: [0-9]=set label, a=prev, d=next, s=save, q=quit", base_y + 22 * num_classes + 30)
    return canvas


def cmd_label(args: argparse.Namespace) -> None:
    images_path = os.path.join(args.dataset_dir, "images.npy")
    labels_path = os.path.join(args.dataset_dir, "labels.npy")
    meta_path = os.path.join(args.dataset_dir, "meta.json")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Missing {images_path}. Run extract first.")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing {labels_path}. Run extract first.")

    images = np.load(images_path)  # (N,H,W,C)
    labels = np.load(labels_path).astype(np.int32)  # (N,)
    meta_items = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_items = json.load(f).get("items", None)

    n = images.shape[0]
    if labels.shape[0] != n:
        raise ValueError(f"labels length {labels.shape[0]} != images length {n}")

    # resume: find first unlabeled
    if args.start_index >= 0:
        idx = args.start_index
    else:
        unlabeled = np.where(labels < 0)[0]
        idx = int(unlabeled[0]) if len(unlabeled) else 0

    def compute_counts() -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for c in range(args.num_classes):
            counts[c] = int(np.sum(labels == c))
        return counts

    def save() -> None:
        np.save(labels_path, labels)
        # also export a compact npz for training convenience
        out_npz = os.path.join(args.dataset_dir, "labeled_stage_dataset.npz")
        keep = labels >= 0
        np.savez_compressed(
            out_npz,
            images=images[keep],
            labels=labels[keep].astype(np.int32),
        )
        print(f"[+] saved labels -> {labels_path}")
        print(f"[+] exported npz   -> {out_npz} (kept {int(np.sum(keep))}/{n})")

    cv2.namedWindow("Stage Labeler", cv2.WINDOW_AUTOSIZE)
    counts = compute_counts()

    while True:
        img_rgb = images[idx]
        h, w = img_rgb.shape[:2]
        scale = max(512 / h, 512 / w, 1.0)
        if scale > 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img_rgb_disp = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_rgb_disp = img_rgb
        
        img_bgr = _bgr(img_rgb_disp)
        meta_item = meta_items[idx] if (meta_items is not None and idx < len(meta_items)) else None
        hud = _draw_hud(
            img_bgr=img_bgr,
            idx=idx,
            n=n,
            label=int(labels[idx]),
            counts=counts,
            num_classes=args.num_classes,
            meta_item=meta_item,
        )
        cv2.imshow("Stage Labeler", hud)

        k = cv2.waitKey(0) & 0xFF

        # quit
        if k in (ord("q"), 27):
            save()
            break

        # save
        if k == ord("s"):
            save()
            continue

        # prev / next
        if k == ord("a"):
            idx = max(0, idx - 1)
            continue
        if k == ord("d"):
            idx = min(n - 1, idx + 1)
            continue

        # numeric label
        if ord("0") <= k <= ord("9"):
            new_label = int(chr(k))
            if new_label >= args.num_classes:
                print(f"[!] label {new_label} >= num_classes={args.num_classes}, ignored.")
                continue

            old = int(labels[idx])
            labels[idx] = new_label
            counts = compute_counts()

            # auto-next
            if args.auto_next:
                idx = min(n - 1, idx + 1)
            continue

    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("stage_data_tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="Extract side_stage_classifier images from classifier_data/*.pkl")
    p_ext.add_argument("--input_dir", type=str, default="./classifier_data")
    p_ext.add_argument("--out_dir", type=str, default="./stage_dataset")
    p_ext.add_argument("--key", type=str, default=DEFAULT_KEY)
    p_ext.add_argument("--use_next_obs", action="store_true", help="Extract from next_observations instead of observations")
    p_ext.add_argument("--shuffle", action="store_true")
    p_ext.add_argument("--limit", type=int, default=-1)
    p_ext.add_argument("--skip_bad", action="store_true", help="Skip transitions missing key/invalid shape")
    p_ext.set_defaults(func=cmd_extract)

    p_lab = sub.add_parser("label", help="Interactive labeling UI")
    p_lab.add_argument("--dataset_dir", type=str, default="./stage_dataset")
    p_lab.add_argument("--num_classes", type=int, default=4)
    p_lab.add_argument("--start_index", type=int, default=-1, help="-1 means resume from first unlabeled")
    p_lab.add_argument("--auto_next", action="store_true", help="Go to next sample after labeling")
    p_lab.set_defaults(func=cmd_label)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
