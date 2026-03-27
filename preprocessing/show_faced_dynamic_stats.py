import argparse
import os
import re

import h5py


def _as_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _numeric_key(x):
    s = _as_str(x)
    nums = re.findall(r"\d+", s)
    return int(nums[-1]) if nums else float("inf")


def main():
    parser = argparse.ArgumentParser(
        description="Show per-subject and per-trial sample counts/shapes for FACED dynamic H5 files"
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        default="/mnt/dataset2/Processed_datasets/EEG_Bench/FACED_emo_label_smooth_window_2s",
        help="Directory containing sub_*.h5 files",
    )
    args = parser.parse_args()

    h5_files = sorted([f for f in os.listdir(args.h5_dir) if f.endswith(".h5")], key=_numeric_key)

    print(f"Found {len(h5_files)} subject files in: {args.h5_dir}\n")

    for file_name in h5_files:
        file_path = os.path.join(args.h5_dir, file_name)

        with h5py.File(file_path, "r") as f:
            trial_keys = sorted(list(f.keys()), key=_numeric_key)
            print(f"Subject: {file_name} | n_trial={len(trial_keys)}")

            for tk in trial_keys:
                trial_grp = f[tk]
                sample_keys = sorted(list(trial_grp.keys()), key=_numeric_key)
                print(f"  Trial {tk}: n_sample={len(sample_keys)}")

                for sk in sample_keys:
                    sample = trial_grp[sk]["eeg"]
                    print(f"    Sample {sk}: shape={tuple(sample.shape)}")

            print()


if __name__ == "__main__":
    main()
