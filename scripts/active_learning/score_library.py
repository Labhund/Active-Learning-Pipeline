"""
score_library.py — Score the full ~300M compound library with the surrogate.

Streams HDF5 fingerprints in chunks to keep memory bounded (~3 GB peak).
Output: scores/predicted_{target}_round{N}.h5 with dataset 'predicted_scores'.

Usage:
    python scripts/active_learning/score_library.py \
        --target trpv1_5irz \
        --round 1 \
        --fp-file data/fingerprints/compounds.h5 \
        --model models/surrogate_trpv1_5irz_round0.json \
        --out scores/ \
        --chunk-size 100000
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import xgboost as xgb


def main(args=None):
    parser = argparse.ArgumentParser(description="Score full library with surrogate")
    parser.add_argument("--target", default="trpv1_5irz")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--fp-file", default="data/fingerprints/compounds.h5")
    parser.add_argument("--model", required=True, help="Path to XGBoost model JSON")
    parser.add_argument("--out", default="scores/")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    cfg = parser.parse_args(args)

    al_round = cfg.round
    exp_id = cfg.experiment_id
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"al_round{al_round}_{exp_id}.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predicted_{cfg.target}_{exp_id}_round{al_round}.h5"

    if out_path.exists():
        logging.info("Score file already exists: %s — skipping.", out_path)
        return

    model_path = Path(cfg.model)
    if not model_path.exists():
        logging.error("Model not found: %s", model_path)
        sys.exit(1)

    fp_file = Path(cfg.fp_file)
    if not fp_file.exists():
        logging.error("FP file not found: %s", fp_file)
        sys.exit(1)

    # Load model
    logging.info("Loading model: %s", model_path)
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    # Open HDF5 and determine total size
    with h5py.File(fp_file, "r") as f:
        N = f["fps"].shape[0]
    logging.info("Total HDF5 rows: %d", N)

    # Pre-allocate output array
    all_scores = np.full(N, np.nan, dtype=np.float32)

    chunk_size = cfg.chunk_size
    n_chunks = (N + chunk_size - 1) // chunk_size
    t0 = time.time()

    with h5py.File(fp_file, "r") as f:
        fp_ds = f["fps"]
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, N)

            packed = fp_ds[start:end]  # (chunk, 512) uint8
            fps = np.unpackbits(packed, axis=1).astype(np.float32)  # (chunk, 4096)

            all_scores[start:end] = model.predict(fps)

            if (chunk_idx + 1) % 100 == 0 or chunk_idx == n_chunks - 1:
                elapsed = time.time() - t0
                rate = (end) / elapsed
                eta = (N - end) / rate if rate > 0 else 0
                logging.info(
                    "Progress: %d / %d (%.1f%%)  %.0f rows/s  ETA %.0f s",
                    end,
                    N,
                    100.0 * end / N,
                    rate,
                    eta,
                )

    elapsed = time.time() - t0
    logging.info("Scoring complete in %.1f s (%.0f rows/s)", elapsed, N / elapsed)

    # Write output HDF5
    with h5py.File(out_path, "w") as f:
        ds = f.create_dataset("predicted_scores", data=all_scores, compression="lzf")
        ds.attrs["target"] = cfg.target
        ds.attrs["experiment_id"] = exp_id
        ds.attrs["round"] = al_round
        ds.attrs["n_compounds"] = N
        ds.attrs["timestamp"] = datetime.utcnow().isoformat()
        ds.attrs["model"] = str(model_path)

    logging.info("Saved: %s (%.2f GB)", out_path, out_path.stat().st_size / 1e9)


if __name__ == "__main__":
    main()
