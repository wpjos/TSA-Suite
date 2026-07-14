---
name: ts-preprocessing-for-tsas-cli
description: >
  为 TSAS-CLI 时序预测算子生成可直接消费的 CSV 文件。
  将原始时间序列 CSV 预处理为 train.csv、chunk_ids.csv、test_window_indices.csv、
  test_window.csv、test_truth.csv 和 meta.yaml，供 itransformer_forecaster、
  lightgbm_forecaster、xgboost_forecaster 等算子通过 forecasting fit / run
  子命令使用。
  不做特征缩放、滑动窗口和残差标签（TSAS 算子内部处理），但保留时间断点检测
  和 chunk_ids 输出，避免 iTransformer 训练窗口跨越时间断层。
  划分逻辑与 train_new.py 对齐：先计算完整数据上的合法窗口起始索引，再按
  70/15/15 切分窗口索引。
version: 1.0.0
triggers:
  - tsas 预处理
  - tsas 数据准备
  - TSAS CLI 预处理
  - tsas forecasting 预处理
  - 时序预测数据准备
  - forecasting 数据预处理
  - train.csv 生成
  - chunk_ids
  - tsas-num-forecasting-model 预处理
  - tsas num forecasting 数据
  - TSAS forecasting CSV
---

# TSAS-CLI 时序预测数据预处理技能

---

## 一、与通用预处理 skill 的区别

| 特性 | 通用版 `ts-preprocessing-pipeline.md` | 本技能 `ts-preprocessing-for-tsas-cli` |
|---|---|---|
| 输出 | `TimeSeriesDataset` / `DataLoader` | CSV 文件 + `meta.yaml` |
| 特征缩放 | 外部做（StandardScaler/MinMaxScaler） | **不做**，TSAS 算子内部处理 |
| 滑动窗口 | 外部做 | **不做**，TSAS 算子内部切窗 |
| 残差标签 | 外部构造 | **不做**，TSAS 算子内部构造 |
| 时间断点检测 | 可选 | 保留，输出 `chunk_ids.csv` |
| 下游使用 | 自定义 PyTorch 训练代码 | `tsas.engine.operator.cli forecasting fit/run` |

---

## 二、输出文件总览

| 文件 | 用途 | 形状/格式 |
|---|---|---|
| `train.csv` | `forecasting fit --input` | `(n_total_rows, 1 + num_features)`，**完整预处理后的数据**，含 `time_col` + feature_cols + target_col |
| `chunk_ids.csv` | `forecasting fit --chunk-ids` | `(n_total_rows, 1)`，单列表，无表头，每行一个整数 chunk 编号 |
| `test_window_indices.csv` | 评估时构造测试窗口 | `(n_test_windows, 1)`，测试窗口起始索引 |
| `test_window.csv` | `forecasting run --input` 示例 | `(seq_len, 1 + num_features)`，**第一个测试窗口**的输入序列 |
| `test_truth.csv` | `evaluation run` 真实值 | `(pred_len, 1)`，第一个测试窗口对应的 target_col |
| `meta.yaml` | 辅助配置生成 | 列名、split 窗口数、`seq_len`、`pred_len`、输出文件清单 |

**切分语义**（与 `train_new.py` 对齐）：

- 先在**完整数据**上计算不跨越 chunk 边界的合法窗口起始索引 `valid_indices`。
- 再对 `valid_indices` 按时间顺序切分为 train / val / test（默认 70% / 15% / 15%，`shuffle=False`，`random_state=42`）。
- `train.csv` / `chunk_ids.csv` 保存完整数据，供 `ITransformerForecaster` 内部复现同一套窗口切分。
- `test_window_indices.csv` 保存测试窗口起始索引，评估阶段直接用它构造 `x_test / y_true`。
- `test_window.csv` / `test_truth.csv` 仅作为单窗口推理示例。

---

## 三、完整流水线

```
原始 CSV
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 0: Schema Mapping                  │
│ 统一列名（可选）                        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 1: Feature Selection               │
│ 选择输入特征与目标变量                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 2: Chunked Loading                 │
│ 流式分块读取大 CSV                      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 3: Gap Detection                   │
│ 按时间间隔切分连续片段，生成 chunk_ids  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 4: Missing Value Imputation        │
│ 缺失值 / 异常值插补                     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 5: Target Smoothing                │
│ 目标变量降噪平滑（可选）                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 6: Window-Index Split              │
│ 先算完整数据的合法窗口起始索引，        │
│ 再按 70/15/15 切分窗口索引              │
└─────────────────────────────────────────┘
    │
    ▼
输出：train.csv, chunk_ids.csv,
      test_window_indices.csv,
      test_window.csv, test_truth.csv, meta.yaml
```

---

## 四、配置参数 Schema

```python
config = {
    # 数据路径与列名
    "data_path": "your_data.csv",
    "time_col": "timestamp",              # 时间列名
    "feature_cols": ["feat1", "feat2"],   # 输入特征列名列表
    "target_col": "target",               # 目标变量列名

    # 输出目录
    "output_dir": "preprocessed_for_tsas",

    # 流式读取
    "chunksize": 100000,                  # 分块读取行数

    # 时间断点检测
    "max_gap": 5.0,                       # 超过该秒数视为断点
    "time_unit": "s",                     # 时间单位：s / m / h

    # 目标变量平滑
    "smooth_target": True,                # 是否对目标变量做平滑
    "smooth_alpha": 0.3,                  # EMA 平滑系数 (0~1)

    # 窗口与预测（仅用于生成 test_window / test_truth，不训练）
    "seq_len": 100,                       # 历史输入步长
    "pred_len": 30,                       # 未来预测步长

    # 数据划分
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}
```

---

## 五、模块化函数实现

### 5.1 Schema Mapping（列名映射）

```python
def apply_schema_mapping(df, mapping_dict):
    """
    Args:
        df: pd.DataFrame
        mapping_dict: dict, {原始列名: 标准列名}
    Returns:
        pd.DataFrame
    """
    df = df.rename(columns=mapping_dict)
    return df
```

### 5.2 Chunked Loading（流式分块读取）

```python
import pandas as pd
import numpy as np

def load_data_chunks(data_path, cols_to_read, chunksize=100000, dtype=np.float32):
    """
    流式分块读取 CSV，返回 np.ndarray。
    """
    data_list = []
    for chunk in pd.read_csv(data_path, usecols=cols_to_read, chunksize=chunksize):
        chunk = chunk.apply(pd.to_numeric, errors='coerce')
        data_list.append(chunk.values.astype(dtype))
    return np.vstack(data_list)
```

### 5.3 Gap Detection（时间断点检测）

```python
def detect_time_gaps(timestamps, max_gap, time_unit='s'):
    """
    Args:
        timestamps: pd.Series of datetime
        max_gap: float, 断点阈值
        time_unit: 's' | 'm' | 'h'
    Returns:
        np.ndarray: chunk_ids
    """
    unit_map = {'s': 1, 'm': 60, 'h': 3600}
    factor = unit_map.get(time_unit, 1)

    time_diff = timestamps.diff().dt.total_seconds().values / factor
    time_diff[0] = 0
    gap_mask = time_diff > max_gap
    chunk_ids = gap_mask.cumsum()
    return chunk_ids
```

### 5.4 Missing Value Imputation（缺失值插补）

```python
def impute_missing_values(data_array, chunk_ids):
    """
    Args:
        data_array: np.ndarray, shape (N, num_features)
        chunk_ids: np.ndarray, shape (N,)
    Returns:
        np.ndarray
    """
    df = pd.DataFrame(data_array)
    df.replace(0, np.nan, inplace=True)
    df = df.groupby(chunk_ids, group_keys=False).apply(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    #df.fillna(0, inplace=True)
    return df.values
```

### 5.5 Target Smoothing（目标变量平滑）

```python
def double_ema_smooth(data_array, chunk_ids, target_idx, alpha=0.3):
    """
    Brown's Double EMA / Zero-Lag EMA，仅对目标变量列平滑。
    """
    for cid in np.unique(chunk_ids):
        mask = (chunk_ids == cid)
        series = data_array[mask, target_idx]
        if len(series) == 0:
            continue
        s1 = pd.Series(series).ewm(alpha=alpha, adjust=False).mean().values
        s2 = pd.Series(s1).ewm(alpha=alpha, adjust=False).mean().values
        data_array[mask, target_idx] = 2 * s1 - s2
    return data_array
```

### 5.6 Window-Index Split（窗口索引切分）

与 `train_new.py` 对齐：先计算完整数据上的合法窗口起始索引，再对索引做 70/15/15 切分。

```python
from sklearn.model_selection import train_test_split

def _get_valid_window_indices(chunk_ids, seq_len, pred_len):
    """
    返回不跨越 chunk 边界的合法窗口起始索引。
    """
    valid_indices = []
    for cid in np.unique(chunk_ids):
        mask = chunk_ids == cid
        chunk_row_indices = np.where(mask)[0]
        num_samples = len(chunk_row_indices) - seq_len - pred_len + 1
        if num_samples > 0:
            valid_indices.extend(chunk_row_indices[:num_samples].tolist())
    return np.array(valid_indices, dtype=int)


def split_window_indices(valid_indices, train_ratio=0.70, val_ratio=0.15):
    """
    按时间顺序切分窗口起始索引，返回 idx_train, idx_val, idx_test。
    """
    if len(valid_indices) == 0:
        raise ValueError("valid_indices 为空")

    idx_train, idx_temp = train_test_split(
        valid_indices,
        test_size=1 - train_ratio,
        random_state=42,
        shuffle=False,
    )
    val_size = val_ratio / (1 - train_ratio)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1 - val_size,
        random_state=42,
        shuffle=False,
    )

    if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
        raise ValueError(
            f"有效窗口数 {len(valid_indices)} 不足以按 "
            f"{train_ratio}/{val_ratio} 划分: "
            f"train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}"
        )

    return idx_train, idx_val, idx_test
```

### 5.7 Save Outputs（保存 TSAS-CLI 输入文件）

```python
import os
import yaml

def save_tsas_outputs(
    df_full,
    time_col,
    feature_cols,
    target_col,
    chunk_ids,
    idx_train,
    idx_val,
    idx_test,
    seq_len,
    pred_len,
    output_dir,
):
    """
    保存 train.csv / chunk_ids.csv / test_window_indices.csv /
    test_window.csv / test_truth.csv / meta.yaml。
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = feature_cols + [target_col]

    # 1. train.csv：完整预处理数据（ITransformerForecaster 内部复现同一套窗口切分）
    train_path = os.path.join(output_dir, "train.csv")
    df_full[[time_col] + numeric_cols].to_csv(train_path, index=False)

    # 2. chunk_ids.csv（与 train.csv 行对齐，无表头）
    chunk_ids_path = os.path.join(output_dir, "chunk_ids.csv")
    pd.DataFrame(chunk_ids).to_csv(chunk_ids_path, index=False, header=False)

    # 3. test_window_indices.csv：评估时直接读取
    test_window_indices_path = os.path.join(output_dir, "test_window_indices.csv")
    pd.DataFrame(idx_test).to_csv(test_window_indices_path, index=False, header=False)

    # 4. test_window.csv：第一个测试窗口的输入序列
    if len(idx_test) == 0:
        raise ValueError("测试窗口数为 0，无法生成 test_window/test_truth")
    first_test_start = int(idx_test[0])
    test_window = df_full.iloc[first_test_start : first_test_start + seq_len]
    test_window_path = os.path.join(output_dir, "test_window.csv")
    test_window[[time_col] + numeric_cols].to_csv(test_window_path, index=False)

    # 5. test_truth.csv：第一个测试窗口对应的目标值
    test_truth = df_full.iloc[
        first_test_start + seq_len : first_test_start + seq_len + pred_len
    ][[target_col]]
    test_truth_path = os.path.join(output_dir, "test_truth.csv")
    test_truth.to_csv(test_truth_path, index=False)

    # 6. meta.yaml
    max_train_row = int(idx_train[-1]) + seq_len + pred_len
    meta = {
        "dataset": {
            "time_col": time_col,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "input_columns_itransformer": feature_cols + [target_col],
            "input_columns_tree": feature_cols,
            "n_train": len(idx_train),
            "n_val": len(idx_val),
            "n_test": len(idx_test),
            "seq_len": seq_len,
            "pred_len": pred_len,
            "n_train_chunks": int(len(np.unique(chunk_ids[:max_train_row]))),
            "n_test_chunks": int(len(np.unique(chunk_ids))),
        },
        "output_files": {
            "train": "train.csv",
            "chunk_ids": "chunk_ids.csv",
            "test_window_indices": "test_window_indices.csv",
            "test_window": "test_window.csv",
            "test_truth": "test_truth.csv",
        },
    }
    meta_path = os.path.join(output_dir, "meta.yaml")
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    print(f"Saved outputs to {output_dir}/")
```

---

## 六、完整使用示例

```python
import pandas as pd
import numpy as np
import os

# 1. 配置
config = {
    "data_path": "data/your_data.csv",
    "time_col": "timestamp",
    "feature_cols": ["flow", "pressure", "load", "temperature"],
    "target_col": "water_level",
    "output_dir": "preprocessed_for_tsas",
    "chunksize": 100000,
    "max_gap": 5.0,
    "time_unit": "s",
    "smooth_target": True,
    "smooth_alpha": 0.3,
    "seq_len": 100,
    "pred_len": 30,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    # test_ratio 隐式为 1 - train_ratio - val_ratio
}

# 2. 读取数据
cols = [config["time_col"]] + config["feature_cols"] + [config["target_col"]]
df = pd.read_csv(config["data_path"], usecols=cols)

# 3. 时间列处理
df[config["time_col"]] = pd.to_datetime(df[config["time_col"]], errors='coerce')
df = df.dropna(subset=[config["time_col"]])

# 4. 时间断点检测
chunk_ids = detect_time_gaps(df[config["time_col"]], config["max_gap"], config["time_unit"])

# 5. 提取数值矩阵（不带时间列）
numeric_cols = config["feature_cols"] + [config["target_col"]]
target_idx = len(config["feature_cols"])
X_data = df[numeric_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)

# 6. 缺失值插补
X_data = impute_missing_values(X_data, chunk_ids)

# 7. 目标变量平滑
if config["smooth_target"]:
    X_data = double_ema_smooth(X_data, chunk_ids, target_idx, config["smooth_alpha"])

# 8. 把处理后的数值列写回 DataFrame（保留时间列）
df[numeric_cols] = X_data

# 9. 窗口索引切分（与 train_new.py 对齐）
valid_indices = _get_valid_window_indices(chunk_ids, config["seq_len"], config["pred_len"])
idx_train, idx_val, idx_test = split_window_indices(
    valid_indices,
    train_ratio=config["train_ratio"],
    val_ratio=config["val_ratio"],
)

# 10. 保存输出
save_tsas_outputs(
    df,
    time_col=config["time_col"],
    feature_cols=config["feature_cols"],
    target_col=config["target_col"],
    chunk_ids=chunk_ids,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    seq_len=config["seq_len"],
    pred_len=config["pred_len"],
    output_dir=config["output_dir"],
)
```

---

