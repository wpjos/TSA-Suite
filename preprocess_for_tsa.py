#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing pipeline extracted from train/train_new.py.

This script produces the exact input expected by TSA-Suite's forecasting
operators:

    from tsas.engine.operator.forecasting.itransformer import ITransformerForecaster
    forecaster = ITransformerForecaster(config=...)
    forecaster.set_chunk_ids(chunk_ids)
    forecaster.fit(x, y)

where x has shape (timesteps, num_features), y has shape (timesteps, 1),
and chunk_ids has shape (timesteps,). Both x and y are **NOT** normalized:
forecasting operators normalize them internally as needed.

Outputs (all aligned by row):
    - x.npy
    - y.npy
    - chunk_ids.npy
    - meta.joblib

Each preprocessing step is encapsulated in its own function and chained
through a single ``PreprocessPipeline`` class.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Preprocessing hyper-parameters."""

    data_file: str
    time_col: str = "datatime"
    max_gap: float = 5.0
    target_col: str = "diya_qibao_shuiwei_youxuanzhi"
    chunk_size: int = 100000
    ema_alpha: float = 0.3
    output_dir: str = "preprocessed_tsa"
    # 用户显式指定的特征列；如果为空，则回退到 expert_features
    feature_cols: List[str] = field(default_factory=list, repr=False)
    # 向后兼容：旧的默认专家特征列表
    expert_features: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self.feature_cols and not self.expert_features:
            self.expert_features = [
                "ningjie_shuiliuliang_youxuanzhi",
                "gaoya_geishuiliuliang_youxuanzhi",
                "chuyangqi_tiaojiefa_weizhifankui",
                "ranji_fuhe",
                "qiji_fuhe",
                "chuyangqi_rukou_yali",
                "diya_qibao_yali_youxuanzhi",
                "diya_zhuqiliuliang_youxuanzhi",
                "gaoya_jianwenshui_liuliang",
                "ranji_paiqi_wendu",
                self.target_col,
            ]


# ---------------------------------------------------------------------------
# Step 1: resolve feature columns
# ---------------------------------------------------------------------------

def resolve_feature_columns(
    data_path: str,
    target_col: str,
    feature_cols: List[str] | None = None,
    expert_features: List[str] | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Read the CSV header and return the feature columns that actually exist.

    Parameters
    ----------
    data_path : str
        Path to CSV file.
    target_col : str
        Target column name.
    feature_cols : list[str] | None
        User-specified feature columns. Takes precedence if non-empty.
    expert_features : list[str] | None
        Fallback feature list (backward compatibility).

    Returns
    -------
    feature_cols : List[str]
        Columns used for model input (guarantees target_col is included).
    cols_to_read : List[str]
        Columns to read from CSV (same as feature_cols unless time_col is
        present but not in feature_cols).
    """
    header_df = pd.read_csv(data_path, nrows=100)
    all_columns = header_df.columns.tolist()

    if target_col not in all_columns:
        raise ValueError(f"Missing target column '{target_col}' in {data_path}")

    source_cols = feature_cols if feature_cols else (expert_features or [target_col])
    selected_cols = [col for col in source_cols if col in all_columns]

    if target_col not in selected_cols:
        selected_cols.append(target_col)

    if not selected_cols:
        raise ValueError(
            f"No valid feature columns found in {data_path}. "
            f"Please check feature_cols / expert_features."
        )

    missing = [col for col in source_cols if col not in all_columns]
    if missing:
        print(f"Warning: requested columns not found in CSV and will be skipped: {missing}")

    return selected_cols, selected_cols.copy()


# ---------------------------------------------------------------------------
# Step 2: load CSV in chunks + time-gap chunking
# ---------------------------------------------------------------------------

def load_raw_chunks(
    data_path: str,
    cols_to_read: List[str],
    time_col: str,
    max_gap: float,
    chunk_size: int = 100000,
) -> List[Tuple[pd.DataFrame, np.ndarray]]:
    """
    Stream-read the CSV and split it into continuous segments based on
    ``max_gap`` seconds.

    Returns
    -------
    List[Tuple[DataFrame, ndarray]]
        Each tuple contains one continuous segment and its chunk_id array.
    """
    result: List[Tuple[pd.DataFrame, np.ndarray]] = []
    last_timestamp = pd.NaT
    current_chunk_id = 0

    for chunk in pd.read_csv(data_path, chunksize=chunk_size, usecols=cols_to_read):
        if time_col in chunk.columns:
            time_series = pd.to_datetime(chunk[time_col], errors="coerce")
            valid_mask = time_series.notna()
            chunk = chunk[valid_mask].copy()
            time_series = time_series[valid_mask]

            if len(chunk) == 0:
                continue

            time_diff = time_series.diff().dt.total_seconds().values
            if not pd.isna(last_timestamp):
                time_diff[0] = (time_series.iloc[0] - last_timestamp).total_seconds()
            else:
                time_diff[0] = 0.0

            gap_mask = time_diff > max_gap
            local_chunk_ids = gap_mask.cumsum() + current_chunk_id
            current_chunk_id = int(local_chunk_ids[-1])
            last_timestamp = time_series.iloc[-1]

            chunk = chunk.drop(columns=[time_col])
        else:
            local_chunk_ids = np.zeros(len(chunk), dtype=int)

        result.append((chunk, local_chunk_ids.astype(int)))

    return result


def concatenate_chunks(
    chunks: List[Tuple[pd.DataFrame, np.ndarray]]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Concatenate all loaded chunks into a single DataFrame + chunk_ids."""
    if not chunks:
        raise ValueError("No data chunks were loaded.")

    dfs = [c[0] for c in chunks]
    ids = [c[1] for c in chunks]

    df = pd.concat(dfs, ignore_index=True)
    chunk_ids = np.concatenate(ids)
    return df, chunk_ids


# ---------------------------------------------------------------------------
# Step 3: numeric conversion
# ---------------------------------------------------------------------------

def to_numeric_array(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Convert selected columns to a float32 numpy array."""
    safe_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return safe_df.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: zero-filling via linear interpolation per chunk
# ---------------------------------------------------------------------------

def fill_zeros_linear_interpolation(
    X: np.ndarray, chunk_ids: np.ndarray
) -> np.ndarray:
    """
    Replace zeros with NaN and interpolate linearly within each continuous
    chunk. Any NaNs that remain after interpolation (e.g. an all-zero chunk)
    are forward/backward filled within the chunk, then filled with the global
    median as a final safeguard.

    The operation is performed on a copy.
    """
    X = X.copy()
    df = pd.DataFrame(X)
    df.replace(0, np.nan, inplace=True)

    # Linear interpolation inside each continuous chunk
    df = df.groupby(chunk_ids, group_keys=False).apply(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    # Forward/backward fill for NaNs at chunk boundaries or all-NaN short chunks
    df = df.groupby(chunk_ids, group_keys=False).apply(
        lambda x: x.ffill().bfill()
    )

    # Final safeguard: fill any remaining NaNs with the global median of each column
    df.fillna(df.median(), inplace=True)

    X[:, :] = df.values
    return X


# ---------------------------------------------------------------------------
# Step 5: double EMA smoothing on target column per chunk
# ---------------------------------------------------------------------------

def apply_double_ema(
    X: np.ndarray,
    chunk_ids: np.ndarray,
    target_idx: int,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Apply double exponential moving average smoothing to the target column
    within each continuous chunk.
    """
    X = X.copy()
    unique_chunks = np.unique(chunk_ids)
    for cid in unique_chunks:
        mask = chunk_ids == cid
        series = X[mask, target_idx]
        if len(series) == 0:
            continue
        s1 = pd.Series(series).ewm(alpha=alpha, adjust=False).mean().values
        s2 = pd.Series(s1).ewm(alpha=alpha, adjust=False).mean().values
        X[mask, target_idx] = 2 * s1 - s2
    return X


# ---------------------------------------------------------------------------
# Step 6: extract target as y
# ---------------------------------------------------------------------------

def extract_target(
    X: np.ndarray, target_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep X unchanged and extract y = X[:, [target_idx]].

    Returns
    -------
    X : np.ndarray, shape (timesteps, num_features)
    y : np.ndarray, shape (timesteps, 1)
    """
    y = X[:, [target_idx]].copy()
    return X, y


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreprocessPipeline:
    """End-to-end preprocessing pipeline for TSA-Suite forecasters.

    Outputs (all aligned by row):
        - x.npy: input features, shape (timesteps, num_features)
        - y.npy: target column, shape (timesteps, 1)
        - chunk_ids.npy: continuous-segment ids, shape (timesteps,)
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.feature_cols: List[str] = []
        self.target_idx: int = -1
        self.chunk_ids: np.ndarray | None = None

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the full pipeline and return (x, y, chunk_ids) ready for TSA-Suite."""
        cfg = self.config

        # Step 1
        self.feature_cols, cols_to_read = resolve_feature_columns(
            cfg.data_file,
            cfg.target_col,
            feature_cols=cfg.feature_cols if cfg.feature_cols else None,
            expert_features=cfg.expert_features if cfg.expert_features else None,
        )

        # If time_col is not part of features, still read it for gap detection
        if cfg.time_col not in self.feature_cols:
            header_df = pd.read_csv(cfg.data_file, nrows=0)
            if cfg.time_col in header_df.columns and cfg.time_col not in cols_to_read:
                cols_to_read = cols_to_read + [cfg.time_col]

        self.target_idx = self.feature_cols.index(cfg.target_col)

        # Step 2
        raw_chunks = load_raw_chunks(
            cfg.data_file,
            cols_to_read,
            cfg.time_col,
            cfg.max_gap,
            cfg.chunk_size,
        )
        df, self.chunk_ids = concatenate_chunks(raw_chunks)

        # Step 3
        X = to_numeric_array(df, self.feature_cols)

        # Step 4
        X = fill_zeros_linear_interpolation(X, self.chunk_ids)

        # Step 5
        X = apply_double_ema(X, self.chunk_ids, self.target_idx, cfg.ema_alpha)

        # Step 6
        x, y = extract_target(X, self.target_idx)

        return x, y, self.chunk_ids

    def run_and_save(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the pipeline and persist the outputs."""
        x, y, chunk_ids = self.run()
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        x_path = os.path.join(cfg.output_dir, "x.npy")
        y_path = os.path.join(cfg.output_dir, "y.npy")
        chunk_ids_path = os.path.join(cfg.output_dir, "chunk_ids.npy")
        meta_path = os.path.join(cfg.output_dir, "meta.joblib")

        np.save(x_path, x)
        np.save(y_path, y)
        np.save(chunk_ids_path, chunk_ids)

        meta = {
            "feature_cols": self.feature_cols,
            "target_col": cfg.target_col,
            "target_idx": self.target_idx,
            "time_col": cfg.time_col,
            "n_samples": x.shape[0],
            "n_features": x.shape[1],
            "chunk_ids_path": chunk_ids_path,
        }
        joblib.dump(meta, meta_path)

        print(f"Saved preprocessed arrays:")
        print(f"  x: {x_path}  shape={x.shape}")
        print(f"  y: {y_path}  shape={y.shape}")
        print(f"  chunk_ids: {chunk_ids_path}  shape={chunk_ids.shape}")
        print(f"  meta: {meta_path}")
        return x, y, chunk_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_comma_list(value: str) -> List[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw CSV for TSA-Suite forecasting operators."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/total_final_0430/total_final_0430_balanced.csv",
        help="Path to raw CSV file.",
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default="datatime",
        help="Timestamp column name used for gap detection.",
    )
    parser.add_argument(
        "--max_gap",
        type=float,
        default=5.0,
        help="Maximum allowed gap (seconds) within one continuous chunk.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="diya_qibao_shuiwei_youxuanzhi",
        help="Target column name.",
    )
    parser.add_argument(
        "--feature_cols",
        type=str,
        default="",
        help="Comma-separated list of feature columns to use. If empty, falls back to expert_features defaults.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Rows per chunk when streaming the CSV.",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.3,
        help="Alpha for double EMA smoothing on the target.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="preprocessed_tsa",
        help="Directory to save x.npy, y.npy and meta.joblib.",
    )

    args = parser.parse_args()

    cfg = PreprocessConfig(
        data_file=args.data_file,
        time_col=args.time_col,
        max_gap=args.max_gap,
        target_col=args.target_col,
        feature_cols=_parse_comma_list(args.feature_cols),
        chunk_size=args.chunk_size,
        ema_alpha=args.ema_alpha,
        output_dir=args.output_dir,
    )

    pipeline = PreprocessPipeline(cfg)
    x, y, chunk_ids = pipeline.run_and_save()
    print("\nPreprocessing complete. Output is ready for TSA-Suite fit(x, y) with chunk_ids.")


if __name__ == "__main__":
    main()
