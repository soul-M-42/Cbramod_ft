from pathlib import Path
import sys
import argparse
from collections import Counter

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def to_float_array(x):
    """Best-effort conversion of tensor/list/scalar to float numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(float)
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        return np.array([], dtype=float)


def main(dataset_dir: Path, max_check: int = 100, max_nan_check: int = 200, save_plot: bool = True, show_plot: bool = False):
    # 1) Add project data folder to path and import EEGDataset
    project_root = Path(__file__).resolve().parents[1]
    proj_data_dir = project_root / "data"
    if str(proj_data_dir) not in sys.path:
        sys.path.append(str(proj_data_dir))

    from dataset_hdf5 import EEGDataset

    # 2) Resolve and validate dataset path
    dataset_dir = dataset_dir.expanduser().resolve()
    print("=" * 80)
    print("Dataset dir:", dataset_dir)
    print("Exists:", dataset_dir.exists())
    h5_preview = sorted([p.name for p in dataset_dir.glob("*.h5")])[:10] if dataset_dir.exists() else []
    print("Example .h5 files:", h5_preview)
    print("=" * 80)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    # 3) Instantiate dataset
    print("Instantiating EEGDataset...")
    dataset = EEGDataset(dataset_dir)
    print("Done instantiating.")

    # 4) Inspect metadata and preview
    print("\n[Metadata]")
    print("Number of samples:", len(dataset))
    print("Channel names (ch_names):", dataset.ch_names)
    print("Dataset name:", dataset.dataset_name)
    print("\nPreview of first 3 samples (file, path, meta):")
    for s in dataset.samples[:3]:
        print({"file": s.get("file"), "path": s.get("path"), "meta": s.get("meta")})

    if len(dataset) == 0:
        print("Dataset is empty; exiting.")
        return

    # 5) Load one sample and print shapes
    print("\n[Single sample check]")
    data, class_label, rating_label, sub, trial = dataset[0]
    print("data.shape:", tuple(data.shape))
    print("data.dtype:", data.dtype)
    print("data.min(), data.max():", float(data.min()), float(data.max()))
    print("class_label.shape:", tuple(class_label.shape) if torch.is_tensor(class_label) else type(class_label))
    print("rating_label.shape:", tuple(rating_label.shape) if torch.is_tensor(rating_label) else type(rating_label))
    print("sub:", sub, "trial:", trial)
    print("Note: data layout is [n_channels, n_segments, 200].")

    # 6) Batch-check labels and distributions
    print("\n[Label distribution check]")
    n_check = min(max_check, len(dataset))
    class_tuples = []
    rating_tuples = []
    nan_count = 0

    for i in tqdm(range(n_check), desc="Checking labels", unit="sample"):
        _, class_i, rating_i, _, _ = dataset[i]
        class_arr = to_float_array(class_i).reshape(-1)
        rating_arr = to_float_array(rating_i).reshape(-1)

        class_t = tuple(class_arr.tolist()) if class_arr.size > 0 else (str(class_i),)
        rating_t = tuple(rating_arr.tolist()) if rating_arr.size > 0 else (str(rating_i),)
        class_tuples.append(class_t)
        rating_tuples.append(rating_t)

        if (class_arr.size > 0 and np.isnan(class_arr).any()) or (rating_arr.size > 0 and np.isnan(rating_arr).any()):
            nan_count += 1

    class_counts = Counter(class_tuples)
    rating_counts = Counter(rating_tuples)

    print("Checked samples:", n_check)
    print("Samples with NaNs in labels:", nan_count)
    print("Unique class-label vectors:", len(class_counts))
    print("Unique rating-label vectors:", len(rating_counts))
    print("Top 5 class-label patterns (count, vector_len):")
    for i, (k, v) in enumerate(class_counts.most_common(5), start=1):
        print(i, v, len(k))

    # 7) Visualize one sample (first 4 channels, first segment)
    print("\n[Plot]")
    data0, _, _, _, _ = dataset[0]
    seg0 = data0[:, 0, :].cpu().numpy()  # [n_channels, 200]
    t = np.arange(seg0.shape[1])
    plt.figure(figsize=(10, 6))
    offset = 0.0
    for ch in range(min(4, seg0.shape[0])):
        y = seg0[ch] + offset
        ch_name = dataset.ch_names[ch] if dataset.ch_names and ch < len(dataset.ch_names) else f"ch{ch}"
        plt.plot(t, y, label=ch_name)
        offset += float(np.max(np.abs(seg0[ch])) * 2 + 1e-6)

    plt.xlabel("Sample index")
    plt.ylabel("Amplitude (offset for clarity)")
    plt.title("First sample — first 4 channels (segment 0)")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        out_path = Path(__file__).resolve().parent / "faced_sample_plot.png"
        plt.savefig(out_path, dpi=150)
        print("Saved plot to:", out_path)

    if show_plot:
        plt.show()
    else:
        plt.close()

    # 8) Sanity checks: per-subject ranges + NaN/global min/max
    print("\n[Sanity checks]")
    print("Per-subject ranges (average min/max):")
    try:
        dataset.calculate_sub_ranges()
    except Exception as e:
        print("Error while calculating sub ranges:", e)

    check_n = min(max_nan_check, len(dataset))
    global_min = float("inf")
    global_max = float("-inf")
    nan_found = False

    for i in tqdm(range(check_n), desc="Checking NaNs/minmax", unit="sample"):
        data_i, _, _, _, _ = dataset[i]
        arr = data_i.detach().cpu().numpy()
        if np.isnan(arr).any():
            nan_found = True
            break
        global_min = min(global_min, float(arr.min()))
        global_max = max(global_max, float(arr.max()))

    print("Checked samples for NaNs:", check_n)
    print("NaN found in data subset:", nan_found)
    print("Global min/max in subset:", global_min, global_max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load FACED dataset and inspect shape/labels in terminal.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/mnt/dataset2/Processed_datasets/EEG_Bench/FACED_emo_label_smooth_window_2s",
        help="Path to FACED processed h5 dataset directory.",
    )
    parser.add_argument("--max-check", type=int, default=100, help="Max samples for label distribution check.")
    parser.add_argument("--max-nan-check", type=int, default=200, help="Max samples for NaN/min-max data check.")
    parser.add_argument("--no-save-plot", action="store_true", help="Do not save the sample plot.")
    parser.add_argument("--show-plot", action="store_true", help="Show plot window (if GUI is available).")

    args = parser.parse_args()
    main(
        dataset_dir=Path(args.dataset_dir),
        max_check=args.max_check,
        max_nan_check=args.max_nan_check,
        save_plot=not args.no_save_plot,
        show_plot=args.show_plot,
    )
