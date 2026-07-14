"""
TSAS-CLI 时序预测数据预处理脚本。
基于 .claude/skills/ts-preprocessing-for-tsas-cli.md 实现。
"""
import os
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# 用户可配置参数
# ---------------------------------------------------------------------------
config = {
    "data_path": "data/total_final_0430/total_final_0430_balanced.csv",
    "time_col": "datatime",
    "feature_cols": [
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
    ],
    "target_col": "diya_qibao_shuiwei_youxuanzhi",
    "output_dir": "preprocessed_for_tsas_10k",
    "n_rows": 10000,          # 仅取前 10000 条连续数据
    "chunksize": 100000,
    "max_gap": 5.0,
    "time_unit": "s",
    "impute_method": "linear",
    "treat_zero_as_missing": True,
    "fill_remaining_na": False,
    "smooth_target": True,
    "smooth_alpha": 0.3,
    "seq_len": 100,
    "pred_len": 30,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}


def detect_time_gaps(timestamps, max_gap, time_unit='s'):
    unit_map = {'s': 1, 'm': 60, 'h': 3600}
    factor = unit_map.get(time_unit, 1)
    time_diff = timestamps.diff().dt.total_seconds().values / factor
    time_diff[0] = 0
    gap_mask = time_diff > max_gap
    chunk_ids = gap_mask.cumsum()
    return chunk_ids


def impute_missing_values(data_array, chunk_ids, method='linear',
                          treat_zero_as_missing=True, fill_remaining_na=False):
    df = pd.DataFrame(data_array)

    if treat_zero_as_missing:
        df = df.replace(0, np.nan)

    if method == 'linear':
        df = df.groupby(chunk_ids, group_keys=False).apply(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
    elif method == 'ffill':
        df = df.groupby(chunk_ids, group_keys=False).apply(lambda x: x.ffill().bfill())
    elif method == 'bfill':
        df = df.groupby(chunk_ids, group_keys=False).apply(lambda x: x.bfill().ffill())
    elif method == 'median':
        df = df.groupby(chunk_ids, group_keys=False).transform(lambda x: x.fillna(x.median()))

    if fill_remaining_na:
        df = df.fillna(0)

    return df.values


def double_ema_smooth(data_array, chunk_ids, target_idx, alpha=0.3):
    for cid in np.unique(chunk_ids):
        mask = (chunk_ids == cid)
        series = data_array[mask, target_idx]
        if len(series) == 0:
            continue
        s1 = pd.Series(series).ewm(alpha=alpha, adjust=False).mean().values
        s2 = pd.Series(s1).ewm(alpha=alpha, adjust=False).mean().values
        data_array[mask, target_idx] = 2 * s1 - s2
    return data_array


def temporal_split(n_total, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"数据量不足以按 {train_ratio}:{val_ratio}:{test_ratio} 划分: "
            f"n_total={n_total}, n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )
    return n_train, n_val, n_test


def save_tsas_outputs(
    df_full,
    time_col,
    feature_cols,
    target_col,
    chunk_ids,
    n_train,
    n_val,
    seq_len,
    pred_len,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = feature_cols + [target_col]
    df_train = df_full.iloc[:n_train + n_val].copy()
    df_test = df_full.iloc[n_train + n_val:].copy()

    # 1. train.csv
    train_path = os.path.join(output_dir, "train.csv")
    df_train[[time_col] + numeric_cols].to_csv(train_path, index=False)

    # 2. chunk_ids.csv（与 train.csv 行对齐，无表头）
    chunk_ids_path = os.path.join(output_dir, "chunk_ids.csv")
    pd.DataFrame(chunk_ids[:n_train + n_val]).to_csv(
        chunk_ids_path, index=False, header=False
    )

    # 3. test_window.csv
    if len(df_test) < seq_len + pred_len:
        raise ValueError(
            f"测试集长度 {len(df_test)} 不足以构造 seq_len={seq_len} + pred_len={pred_len}"
        )
    test_window = df_test.iloc[:seq_len]
    test_window_path = os.path.join(output_dir, "test_window.csv")
    test_window[[time_col] + numeric_cols].to_csv(test_window_path, index=False)

    # 4. test_truth.csv
    test_truth = df_test.iloc[seq_len:seq_len + pred_len][[target_col]]
    test_truth_path = os.path.join(output_dir, "test_truth.csv")
    test_truth.to_csv(test_truth_path, index=False)

    # 5. meta.yaml
    meta = {
        "dataset": {
            "time_col": time_col,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "input_columns_itransformer": feature_cols + [target_col],
            "input_columns_tree": feature_cols,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": len(df_test),
            "seq_len": seq_len,
            "pred_len": pred_len,
            "n_chunks": int(len(np.unique(chunk_ids[:n_train + n_val]))),
        },
        "output_files": {
            "train": "train.csv",
            "chunk_ids": "chunk_ids.csv",
            "test_window": "test_window.csv",
            "test_truth": "test_truth.csv",
        },
    }
    meta_path = os.path.join(output_dir, "meta.yaml")
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    print(f"Saved outputs to {output_dir}/")


def main():
    # -----------------------------------------------------------------------
    # 1. 读取数据：仅取前 n_rows 条连续记录
    # -----------------------------------------------------------------------
    cols = [config["time_col"]] + config["feature_cols"] + [config["target_col"]]
    print(f"Reading first {config['n_rows']} rows from {config['data_path']} ...")
    df = pd.read_csv(config["data_path"], usecols=cols, nrows=config["n_rows"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # -----------------------------------------------------------------------
    # 2. 时间列处理
    # -----------------------------------------------------------------------
    df[config["time_col"]] = pd.to_datetime(
        df[config["time_col"]], errors='coerce'
    )
    df = df.dropna(subset=[config["time_col"]])
    print(f"After datetime parsing: {len(df)} rows.")

    # -----------------------------------------------------------------------
    # 3. 时间断点检测
    # -----------------------------------------------------------------------
    chunk_ids = detect_time_gaps(
        df[config["time_col"]], config["max_gap"], config["time_unit"]
    )
    n_chunks = len(np.unique(chunk_ids))
    print(f"Detected {n_chunks} time chunk(s) with max_gap={config['max_gap']}{config['time_unit']}.")

    # -----------------------------------------------------------------------
    # 4. 提取数值矩阵
    # -----------------------------------------------------------------------
    numeric_cols = config["feature_cols"] + [config["target_col"]]
    target_idx = len(config["feature_cols"])
    X_data = df[numeric_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)

    na_before = np.isnan(X_data).sum()
    print(f"NaN count before imputation: {na_before}")

    # -----------------------------------------------------------------------
    # 5. 缺失值插补
    # -----------------------------------------------------------------------
    X_data = impute_missing_values(
        X_data,
        chunk_ids,
        method=config["impute_method"],
        treat_zero_as_missing=config["treat_zero_as_missing"],
        fill_remaining_na=config["fill_remaining_na"],
    )
    print(f"NaN count after imputation: {np.isnan(X_data).sum()}")

    # -----------------------------------------------------------------------
    # 6. 目标变量平滑
    # -----------------------------------------------------------------------
    if config["smooth_target"]:
        X_data = double_ema_smooth(
            X_data, chunk_ids, target_idx, config["smooth_alpha"]
        )
        print(f"Target column '{config['target_col']}' smoothed (alpha={config['smooth_alpha']}).")

    # -----------------------------------------------------------------------
    # 7. 写回 DataFrame
    # -----------------------------------------------------------------------
    df[numeric_cols] = X_data

    # -----------------------------------------------------------------------
    # 8. 时序划分
    # -----------------------------------------------------------------------
    n_total = len(df)
    n_train, n_val, n_test = temporal_split(
        n_total,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
    )
    print(
        f"Temporal split: train={n_train}, val={n_val}, test={n_test} "
        f"(ratios {config['train_ratio']}/{config['val_ratio']}/{config['test_ratio']})"
    )

    # -----------------------------------------------------------------------
    # 9. 保存输出
    # -----------------------------------------------------------------------
    save_tsas_outputs(
        df,
        time_col=config["time_col"],
        feature_cols=config["feature_cols"],
        target_col=config["target_col"],
        chunk_ids=chunk_ids,
        n_train=n_train,
        n_val=n_val,
        seq_len=config["seq_len"],
        pred_len=config["pred_len"],
        output_dir=config["output_dir"],
    )

    print("\n预处理完成。输出文件：")
    for f in ["train.csv", "chunk_ids.csv", "test_window.csv", "test_truth.csv", "meta.yaml"]:
        p = os.path.join(config["output_dir"], f)
        size = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"  {f}: {size} bytes")


if __name__ == "__main__":
    main()
