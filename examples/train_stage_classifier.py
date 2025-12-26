#! python3
import os
import json
import argparse
import numpy as np

import jax
from jax import numpy as jnp
import optax
from flax.training import checkpoints
from tqdm import tqdm

from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.vision.data_augmentations import batched_random_crop


def _ensure_dir(d: str) -> None:
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)


def _load_dataset(dataset_dir: str):
    npz_path = os.path.join(dataset_dir, "labeled_stage_dataset.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}. Run stage_data_tool.py label first.")
    data = np.load(npz_path)
    images = data["images"].astype(np.uint8)  # (N,H,W,C)
    labels = data["labels"].astype(np.int32)  # (N,)
    return images, labels


def _split_train_val(images, labels, val_ratio: float, seed: int):
    n = images.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    images = images[perm]
    labels = labels[perm]
    n_val = int(n * val_ratio)
    x_val, y_val = images[:n_val], labels[:n_val]
    x_tr, y_tr = images[n_val:], labels[n_val:]
    return (x_tr, y_tr), (x_val, y_val)


def _make_obs_batch(images_uint8: np.ndarray) -> dict:
    """
    将图像批次转换为分类器期望的输入格式。
    
    create_classifier 使用 EncodingWrapper(enable_stacking=True)，
    期望输入形状为 (B, T, H, W, C)，其中 T=1。
    """
    # images_uint8: (B, H, W, C) -> (B, 1, H, W, C)
    return {"side_stage_classifier": images_uint8[:, None, ...]}


def main():
    p = argparse.ArgumentParser("train_stage_classifier")
    p.add_argument("--dataset_dir", type=str, default="./stage_dataset")
    p.add_argument("--exp_name", type=str, default="usb_pickup_insertion", help="Only used for bookkeeping/log naming")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--encoder_type", type=str, default="resnet18", choices=["resnet18", "resnet-pretrained"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--augment_crop_padding", type=int, default=0, help="0 disables random crop; e.g. 4")
    p.add_argument("--ckpt_dir", type=str, default="./stage_classifier_ckpt")
    p.add_argument("--save_best", action="store_true", help="Save best checkpoint based on val accuracy")
    args = p.parse_args()

    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    _ensure_dir(os.path.abspath(args.ckpt_dir))

    images, labels = _load_dataset(args.dataset_dir)
    if images.shape[0] < args.batch_size:
        print(f"[!] Warning: dataset N={images.shape[0]} < batch_size={args.batch_size}")

    # 过滤掉未标注的样本 (label == -1)
    valid_mask = labels >= 0
    images = images[valid_mask]
    labels = labels[valid_mask]
    
    if images.shape[0] == 0:
        raise ValueError("No labeled samples found! Please label your data first.")

    # basic stats
    counts = {c: int(np.sum(labels == c)) for c in range(args.num_classes)}
    print("[+] dataset size:", images.shape[0], "image_shape:", images.shape[1:], "counts:", counts)

    (x_tr, y_tr), (x_val, y_val) = _split_train_val(images, labels, args.val_ratio, args.seed)
    print("[+] train:", x_tr.shape[0], "val:", x_val.shape[0])

    # init model (multi-class)
    key = jax.random.PRNGKey(args.seed)

    # build a sample for init - 确保至少有 1 个样本
    n_sample = min(len(x_tr), max(1, args.batch_size))
    sample_obs = _make_obs_batch(x_tr[:n_sample])
    
    classifier = create_classifier(
        key=key,
        sample=sample_obs,
        image_keys=["side_stage_classifier"],
        n_way=args.num_classes,
        encoder_type=args.encoder_type,
    )

    # override optimizer if needed (create_classifier 默认 adam(1e-4))
    if args.lr != 1e-4:
        tx = optax.adam(args.lr)
        classifier = classifier.replace(tx=tx, opt_state=tx.init(classifier.params))

    @jax.jit
    def train_step(state, obs_batch, y_batch, rng):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                obs_batch,
                train=True,
                rngs={"dropout": rng},
            )  # (B, K)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        pred = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(pred == y_batch)
        return state, loss, acc

    @jax.jit
    def eval_step(state, obs_batch, y_batch):
        logits = state.apply_fn({"params": state.params}, obs_batch, train=False)  # (B,K)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()
        pred = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(pred == y_batch)
        return loss, acc, pred

    def maybe_augment(rng, obs_batch):
        if args.augment_crop_padding <= 0:
            return obs_batch
        img = obs_batch["side_stage_classifier"]  # (B,T,H,W,C)
        img = batched_random_crop(img, rng, padding=args.augment_crop_padding, num_batch_dims=2)
        return {"side_stage_classifier": img}

    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    rng = jax.random.PRNGKey(args.seed + 123)

    steps_per_epoch = max(1, int(np.ceil(x_tr.shape[0] / args.batch_size)))

    for epoch in tqdm(range(args.num_epochs), desc="Training"):
        # shuffle each epoch
        perm = np.random.default_rng(args.seed + epoch).permutation(x_tr.shape[0])
        x_tr_e = x_tr[perm]
        y_tr_e = y_tr[perm]

        # train loop
        tr_losses = []
        tr_accs = []
        for s in range(steps_per_epoch):
            b0 = s * args.batch_size
            b1 = min((s + 1) * args.batch_size, x_tr_e.shape[0])
            xb = x_tr_e[b0:b1]
            yb = y_tr_e[b0:b1]

            obs_b = _make_obs_batch(xb)
            rng, aug_key, step_key = jax.random.split(rng, 3)
            obs_b = maybe_augment(aug_key, obs_b)

            yb_j = jnp.asarray(yb)
            classifier, loss, acc = train_step(classifier, obs_b, yb_j, step_key)
            tr_losses.append(float(loss))
            tr_accs.append(float(acc))

        # eval in batches
        val_losses = []
        val_accs = []
        if x_val.shape[0] > 0:
            val_steps = max(1, int(np.ceil(x_val.shape[0] / args.batch_size)))
            for s in range(val_steps):
                b0 = s * args.batch_size
                b1 = min((s + 1) * args.batch_size, x_val.shape[0])
                xb = x_val[b0:b1]
                yb = y_val[b0:b1]
                if xb.shape[0] == 0:
                    continue
                obs_b = _make_obs_batch(xb)
                loss, acc, _ = eval_step(classifier, obs_b, jnp.asarray(yb))
                val_losses.append(float(loss))
                val_accs.append(float(acc))

        tr_loss = float(np.mean(tr_losses))
        tr_acc = float(np.mean(tr_accs))
        va_loss = float(np.mean(val_losses)) if val_losses else 0.0
        va_acc = float(np.mean(val_accs)) if val_accs else 0.0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch+1:03d}/{args.num_epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        # checkpoint
        if args.save_best:
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                checkpoints.save_checkpoint(args.ckpt_dir, classifier, step=epoch + 1, overwrite=True)
                print(f"  [*] New best val_acc={va_acc:.4f}, saved checkpoint")
        else:
            # save last
            checkpoints.save_checkpoint(args.ckpt_dir, classifier, step=epoch + 1, overwrite=True)

    # save training log
    log_path = os.path.join(args.ckpt_dir, f"train_log_{args.exp_name}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_name": args.exp_name,
                "num_classes": args.num_classes,
                "encoder_type": args.encoder_type,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "val_ratio": args.val_ratio,
                "augment_crop_padding": args.augment_crop_padding,
                "best_val_acc": best_val_acc,
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[+] training log saved -> {log_path}")
    print(f"[+] best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()