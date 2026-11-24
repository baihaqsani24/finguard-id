"""
========================================================
Final Project — FinGuard ID (Batch HCK-45)

Skrip utilitas untuk menyalin features.py dan artefak model
dari folder root/model ke deployment/src/ dan pipeline/dags/
setelah modeling.ipynb dijalankan ulang.
========================================================
"""

import hashlib
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COPY_PAIRS = [
    ("features.py", "deployment/src/features.py"),
    ("features.py", "pipeline/dags/features.py"),
    ("model/model_kelayakan.pkl", "deployment/src/model_kelayakan.pkl"),
    ("model/model_restrukturisasi.pkl", "deployment/src/model_restrukturisasi.pkl"),
    ("model/model_metadata.json", "deployment/src/model_metadata.json"),
    ("model/umr_reference.json", "deployment/src/umr_reference.json"),
    ("pipeline/data/finguard_raw.csv", "deployment/src/finguard_raw.csv"),
    ("assets/finguard_icon.png", "deployment/src/finguard_icon.png"),
    ("assets/finguard_logo.png", "deployment/src/finguard_logo.png"),
    ("model/model_kelayakan.pkl", "pipeline/dags/model/model_kelayakan.pkl"),
    ("model/model_restrukturisasi.pkl", "pipeline/dags/model/model_restrukturisasi.pkl"),
    ("model/model_metadata.json", "pipeline/dags/model/model_metadata.json"),
    ("model/test_ids.csv", "pipeline/dags/model/test_ids.csv"),
]


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    for src_rel, dst_rel in COPY_PAIRS:
        src = os.path.join(BASE_DIR, src_rel)
        dst = os.path.join(BASE_DIR, dst_rel)
        if not os.path.exists(src):
            print(f"[SKIP] Sumber tidak ditemukan: {src_rel}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        assert md5_file(src) == md5_file(dst), f"Hash tidak cocok: {dst_rel}"
        print(f"[OK] {src_rel} -> {dst_rel}")
    print("Sinkronisasi artefak selesai dan seluruh file terverifikasi identik.")


if __name__ == "__main__":
    main()
