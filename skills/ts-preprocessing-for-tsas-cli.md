---
name: ts-preprocessing-for-tsas-cli
description: >
  为 TSAS-CLI 时序预测算子生成可直接消费的 CSV 文件。
  将原始时间序列 CSV 预处理为 train.csv、chunk_ids.csv、test_window.csv、
  test_truth.csv 和 meta.yaml，供 itransformer_forecaster、lightgbm_forecaster、
  xgboost_forecaster 等算子通过 forecasting fit / run 子命令使用。
  不做特征缩放、滑动窗口和残差标签（TSAS 算子内部处理），但保留时间断点检测
  和 chunk_ids 输出，避免 iTransformer 训练窗口跨越时间断层。
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
| `train.csv` | `forecasting fit --input` | `(n_train + n_val, 1 + num_features)`，含 `time_col` + feature_cols + target_col |
| `chunk_ids.csv` | `forecasting fit --chunk-ids` | `(n_train + n_val, 1)`，单列表，无表头，每行一个整数 |
| `test_window.csv` | `forecasting run --input` | `(seq_len, 1 + num_features)`，测试集前 `seq_len` 行 |
| `test_truth.csv` | `evaluation run` 真实值 | `(pred_len, 1)`，测试集第 `[seq_len : seq_len + pred_len]` 行的 target_col |
| `meta.yaml` | 辅助配置生成 | 列名、split 大小、`seq_len`、`pred_len`、输出文件清单 |

**切分语义**：

- `train.csv` = 外部训练集 + 外部验证集。TSAS 算子内部会再按 `train_ratio` / `val_ratio` 切分。
- `test_window.csv` / `test_truth.csv` 来自外部测试集，不参与训练。

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
│ Step 6: Temporal Split                  │
│ 按时间顺序划分 train / val / test       │
└─────────────────────────────────────────┘
    │
    ▼
输出：train.csv, chunk_ids.csv,
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

    # 缺失值插补
    "impute_method": "linear",            # linear / ffill / bfill / median
    "treat_zero_as_missing": True,        # 是否把 0 视为缺失值
    "fill_remaining_na": False,           # 插值后剩余 NaN 是否填 0

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
def impute_missing_values(data_array, chunk_ids, method='linear',
                          treat_zero_as_missing=True, fill_remaining_na=False):
    """
    Args:
        data_array: np.ndarray, shape (N, num_features)
        chunk_ids: np.ndarray, shape (N,)
        method: 'linear' | 'ffill' | 'bfill' | 'median'
        treat_zero_as_missing: bool
        fill_remaining_na: bool
    Returns:
        np.ndarray
    """
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

### 5.6 Temporal Split（时序划分）

```python
def temporal_split(n_total, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    按时间顺序划分训练/验证/测试集，返回三个分区的行数。
    """
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
    n_train,
    n_val,
    seq_len,
    pred_len,
    output_dir,
):
    """
    保存 train.csv / chunk_ids.csv / test_window.csv / test_truth.csv / meta.yaml。
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = feature_cols + [target_col]
    df_train = df_full.iloc[:n_train + n_val].copy()
    df_test = df_full.iloc[n_train + n_val:].copy()

    # 1. train.csv
    train_path = os.path.join(output_dir, "train.csv")
    df_train[[time_col] + numeric_cols].to_csv(train_path, index=False)

    # 2. chunk_ids.csv（与 train.csv 行对齐，无表头）
    chunk_ids_path = os.path.join(output_dir, "chunk_ids.csv")
    pd.DataFrame(chunk_ids[:n_train + n_val]).to_csv(chunk_ids_path, index=False, header=False)

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
X_data = impute_missing_values(
    X_data, chunk_ids,
    method=config["impute_method"],
    treat_zero_as_missing=config["treat_zero_as_missing"],
    fill_remaining_na=config["fill_remaining_na"]
)

# 7. 目标变量平滑
if config["smooth_target"]:
    X_data = double_ema_smooth(X_data, chunk_ids, target_idx, config["smooth_alpha"])

# 8. 把处理后的数值列写回 DataFrame（保留时间列）
df[numeric_cols] = X_data

# 9. 时序划分
n_total = len(df)
n_train, n_val, n_test = temporal_split(
    n_total,
    train_ratio=config["train_ratio"],
    val_ratio=config["val_ratio"],
    test_ratio=config["test_ratio"]
)

# 10. 保存输出
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
```

---

## 七、与 TSAS-CLI 对接

### 7.1 生成 `forecasting_config.yaml`

根据 `meta.yaml` 中的列名生成配置。

**iTransformer 示例**：

```yaml
operator:
  name: "itransformer_forecaster"
  input_columns:
    - "flow"
    - "pressure"
    - "load"
    - "temperature"
    - "water_level"    # target 放在最后一列
  target_column: "water_level"
  config:
    seq_len: 100
    pred_len: 30
    d_model: 128
    nhead: 4
    num_layers: 2
    epochs: 30
    batch_size: 128
```

**LightGBM / XGBoost 示例**：

```yaml
operator:
  name: "lightgbm_forecaster"
  input_columns:
    - "flow"
    - "pressure"
    - "load"
    - "temperature"    # 不包含 target
  target_column: "water_level"
  config:
    seq_len: 100
    pred_len: 30
    num_leaves: 31
    n_estimators: 200
```

### 7.2 训练

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting fit \
  --input preprocessed_for_tsas/train.csv \
  --target water_level \
  --config forecasting_config.yaml \
  --chunk-ids preprocessed_for_tsas/chunk_ids.csv \
  --save model_dir/
```

### 7.3 推理

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting run \
  --input preprocessed_for_tsas/test_window.csv \
  --config forecasting_config.yaml \
  --load model_dir/ \
  --output forecasting_result.csv
```

### 7.4 评价

将 `forecasting_result.csv` 与 `test_truth.csv` 对齐后：

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli evaluation run \
  --input eval_aligned.csv \
  --config eval_config.yaml \
  --output eval_result.json
```

---

## 八、算子特化说明

| 算子 | `input_columns` 是否含 target | `chunk_ids` 行为 |
|---|---|---|
| `itransformer_forecaster` | **必须包含**，放在最后一列（`target_idx=-1`） | 有效使用，窗口不跨断层 |
| `lightgbm_forecaster` | **不包含**，只放特征列 | 不支持，传入时忽略并 warning |
| `xgboost_forecaster` | **不包含**，只放特征列 | 不支持，传入时忽略并 warning |
| `*_mimo_forecaster` | 同对应基算子 | 同对应基算子 |

---

## 九、常见问题

### Q1: 为什么 CLI 模式下不做特征缩放？

- `itransformer_forecaster` 内部会在训练样本范围内 fit `StandardScaler`。
- `lightgbm_forecaster` / `xgboost_forecaster` 是树模型，对特征缩放不敏感。
外部再做缩放会造成重复处理或数值不稳定。

### Q2: 为什么 CLI 模式下不做滑动窗口和残差标签？

TSAS 算子内部会自己完成：
- `forecasting fit` 从完整时间序列滑窗构造训练样本。
- `forecasting run` 接收 `(seq_len, num_features)` 窗口直接预测。
- `itransformer_forecaster` 内部预测残差并还原为物理量。

### Q3: `chunk_ids.csv` 为什么无表头？

CLI 通过 `pd.read_csv(path, header=None)` 读取，确保每行一个整数即可。

### Q4: 如果数据没有明显时间断层怎么办？

把 `max_gap` 设得很大，或直接令 `chunk_ids = np.zeros(len(df), dtype=int)`。此时 `itransformer_forecaster` 的行为与不传 `--chunk-ids` 一致。

### Q5: `test_window.csv` 为什么还要包含 target 列？

`itransformer_forecaster` 的输入窗口是完整的多变量时序窗口，target 列作为历史观测的一部分参与预测。树模型则只使用 `input_columns` 中的特征列。

---

## 十、可扩展方向

| 改进点 | 说明 |
|---|---|
| 多目标预测 | 将 `target_col` 改为列表，`test_truth.csv` 变为多列 |
| 多测试窗口 | 从 test 集中滑出多个 `(window, truth)` 对 |
| 验证集输出 | 额外输出 `val_window.csv` / `val_truth.csv` 用于调参 |
| 在线预处理 | 支持流式更新 scaler 和 chunk_ids |

---
