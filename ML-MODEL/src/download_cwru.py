import os
import urllib.request
import numpy as np
from scipy.io import loadmat

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cwru")

DATASET_MAP = {
    "normal": {
        "0hp": 97, "1hp": 98, "2hp": 99, "3hp": 100
    },
    "inner_race_007": {
        "0hp": 105, "1hp": 106, "2hp": 107, "3hp": 108
    },
    "ball_007": {
        "0hp": 118, "1hp": 119, "2hp": 120, "3hp": 121
    },
    "outer_race_007": {
        "0hp": 130, "1hp": 131, "2hp": 132, "3hp": 133
    },
    "inner_race_014": {
        "0hp": 169, "1hp": 170, "2hp": 171, "3hp": 172
    },
    "ball_014": {
        "0hp": 185, "1hp": 186, "2hp": 187, "3hp": 188
    },
    "outer_race_014": {
        "0hp": 197, "1hp": 198, "2hp": 199, "3hp": 200
    },
    "inner_race_021": {
        "0hp": 209, "1hp": 210, "2hp": 211, "3hp": 212
    },
    "ball_021": {
        "0hp": 222, "1hp": 223, "2hp": 224, "3hp": 225
    },
    "outer_race_021": {
        "0hp": 234, "1hp": 235, "2hp": 236, "3hp": 237
    },
}

SAMPLING_RATE = 12000  # 12 kHz drive end


def _find_de_key(mat_data, file_num):
    for key in mat_data.keys():
        if "DE_time" in key:
            return key
    padded = f"X{file_num:03d}_DE_time"
    if padded in mat_data:
        return padded
    unpadded = f"X{file_num}_DE_time"
    if unpadded in mat_data:
        return unpadded
    non_meta = [k for k in mat_data.keys() if not k.startswith("__")]
    if non_meta:
        return non_meta[0]
    raise KeyError(f"No drive-end key found in file {file_num}. Keys: {list(mat_data.keys())}")


def download_file(file_num, dest_dir):
    dest_path = os.path.join(dest_dir, f"{file_num}.mat")
    if os.path.exists(dest_path):
        return dest_path

    url = BASE_URL.format(file_num)
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def load_signal(mat_path, file_num):
    mat = loadmat(mat_path)
    key = _find_de_key(mat, file_num)
    signal = mat[key].flatten().astype(np.float64)
    return signal


def download_and_prepare(fault_types=None, loads=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if fault_types is None:
        fault_types = list(DATASET_MAP.keys())
    if loads is None:
        loads = ["0hp", "1hp", "2hp", "3hp"]

    signals = {}
    total = sum(1 for ft in fault_types for ld in loads if ld in DATASET_MAP.get(ft, {}))
    count = 0

    print(f"Downloading CWRU bearing dataset ({total} files)...")
    print(f"  Fault types: {fault_types}")
    print(f"  Load conditions: {loads}")
    print()

    for fault_type in fault_types:
        if fault_type not in DATASET_MAP:
            print(f"  [WARN] Unknown fault type: {fault_type}, skipping")
            continue

        for load in loads:
            if load not in DATASET_MAP[fault_type]:
                continue

            file_num = DATASET_MAP[fault_type][load]
            count += 1
            print(f"  [{count}/{total}] {fault_type} @ {load} (file {file_num})")

            mat_path = download_file(file_num, raw_dir)
            signal = load_signal(mat_path, file_num)

            key = f"{fault_type}_{load}"
            signals[key] = {
                "signal": signal,
                "fault_type": fault_type,
                "load": load,
                "file_num": file_num,
                "n_samples": len(signal),
            }
            print(f"    -> {len(signal)} samples ({len(signal)/SAMPLING_RATE:.1f}s)")

    np.savez_compressed(
        os.path.join(DATA_DIR, "cwru_signals.npz"),
        **{k: v["signal"] for k, v in signals.items()}
    )

    metadata = {k: {mk: mv for mk, mv in v.items() if mk != "signal"} for k, v in signals.items()}
    np.save(os.path.join(DATA_DIR, "cwru_metadata.npy"), metadata, allow_pickle=True)

    print(f"\nDone. {count} signals saved to {DATA_DIR}/")
    print(f"  cwru_signals.npz  ({count} arrays)")
    print(f"  cwru_metadata.npy (metadata dict)")

    return signals


if __name__ == "__main__":
    download_and_prepare()
