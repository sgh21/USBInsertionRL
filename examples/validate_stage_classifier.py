#! python3
import os
import argparse
import time
import csv
import numpy as np
import cv2
import jax
import jax.numpy as jnp
from flax.training import checkpoints

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.networks.reward_classifier import create_classifier


def _make_batch(img: np.ndarray) -> dict:
    arr = np.asarray(img)
    if arr.ndim == 3:          # (H,W,C)
        arr = arr[None, None, ...]
    elif arr.ndim == 4:        # (T,H,W,C) with T=1
        arr = arr[None, ...]
    else:
        raise ValueError(f"Unexpected image shape {arr.shape}")
    return {"side_stage_classifier": arr}


def load_classifier(env, ckpt_dir: str, num_classes: int, encoder_type: str, seed: int):
    sample = env.observation_space.sample()
    dummy = _make_batch(sample["side_stage_classifier"])
    key = jax.random.PRNGKey(seed)
    state = create_classifier(
        key=key,
        sample=dummy,
        image_keys=["side_stage_classifier"],
        n_way=num_classes,
        encoder_type=encoder_type,
    )
    state = checkpoints.restore_checkpoint(os.path.abspath(ckpt_dir), state)
    print(f"[+] loaded stage classifier from {os.path.abspath(ckpt_dir)}")
    return state


def infer(classifier_state, img: np.ndarray):
    obs = _make_batch(img)
    logits = classifier_state.apply_fn({"params": classifier_state.params}, obs, train=False)  # (1, K)
    probs = jax.nn.softmax(logits, axis=-1)
    probs = np.asarray(probs)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred, conf, probs.tolist()


def save_image_with_overlay(img_rgb: np.ndarray, pred: int, conf: float, save_path: str):
    """在图像上叠加预测结果并保存。输入为 RGB，cv2 需 BGR。"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    text = f"pred: {pred}  conf: {conf:.3f}"
    cv2.putText(img_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, img_bgr)


def main():
    parser = argparse.ArgumentParser("validate_stage_classifier")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="./stage_classifier_ckpt")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--encoder_type", type=str, default="resnet18", choices=["resnet18", "resnet-pretrained"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="stage_classifier_preds.csv")
    parser.add_argument("--output_dir", type=str, default="stage_classifier_images")
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()

    assert args.exp_name in CONFIG_MAPPING, "Unknown exp_name"
    config = CONFIG_MAPPING[args.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False, stage_classifier=False)

    classifier_state = load_classifier(env, args.ckpt_dir, args.num_classes, args.encoder_type, args.seed)

    obs, _ = env.reset()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    print(f"[+] saving images to {out_dir}")
    print(f"[+] writing predictions to {out_csv}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "pred", "conf", "probs", "image_path"])  # header

        try:
            for step in range(args.max_steps):
                actions = np.zeros(env.action_space.sample().shape)
                next_obs, rew, done, truncated, info = env.step(actions)
                if "intervene_action" in info:
                    actions = info["intervene_action"]

                img = next_obs["side_stage_classifier"]
                if img.ndim == 4 and img.shape[0] == 1:
                    img = img[0]
                img_rgb = np.asarray(img)

                pred, conf, probs = infer(classifier_state, img_rgb)

                img_path = os.path.join(out_dir, f"step_{step:05d}_pred{pred}_conf{conf:.3f}.png")
                save_image_with_overlay(img_rgb, pred, conf, img_path)

                writer.writerow([step, pred, f"{conf:.6f}", probs, img_path])

                if done or truncated:
                    obs, _ = env.reset()
                else:
                    obs = next_obs

                time.sleep(0.01)  # 轻微节流，避免占满 CPU
        finally:
            env.close()

    print("[+] done.")


if __name__ == "__main__":
    main()