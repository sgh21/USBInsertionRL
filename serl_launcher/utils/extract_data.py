#! python3

import os
import pickle
from typing import List

import numpy as np

KEYS = ["side_policy", "wrist_1", "wrist_2"]


def load_observations(chunks, key):
    arr = np.array([entry[key] for entry in chunks], dtype=np.uint8)
    if arr.ndim == 5:
        arr = np.squeeze(arr, axis=1)
    return arr


def main():
    mixed_chunks: List[np.ndarray] = []

    try:
        for idx in range(1, 100000):
            filename = f"buffer/transitions_{idx}000.pkl"
            if not os.path.exists(filename):
                print(f"[+] No more files after {filename}")
                break

            print(f"[+] Load from {filename}")
            with open(filename, "rb") as file:
                data = pickle.load(file)

            observations = [obs["observations"] for obs in data]
            per_file_arrays = []
            for key in KEYS:
                per_file_arrays.append(load_observations(observations, key))

            mixed = np.concatenate(per_file_arrays, axis=0)
            mixed_chunks.append(mixed)
    except Exception as exc:
        print(f"[!] Unwanted exception: {exc}")
    finally:
        if mixed_chunks:
            merged = np.concatenate(mixed_chunks, axis=0)
            np.random.shuffle(merged)
            with open("dataset/data.pkl", "wb") as file:
                pickle.dump(merged[256:], file)
            print(f"[+] Saved {len(merged[256:])} mixed samples, shape: {merged[256:].shape}")
            with open("dataset/valid.pkl", "wb") as file:
                pickle.dump(merged[:256], file)
            print(f"[+] Saved {len(merged[:256])} mixed samples, shape: {merged[:256].shape}")
        else:
            print("[!] No data collected.")


if __name__ == "__main__":
    main()
