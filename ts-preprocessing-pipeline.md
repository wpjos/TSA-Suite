---
name: ts-preprocessing-pipeline
description: >
  通用多变量时序预测预处理与特征工程技能。
  提供从原始时间序列 CSV 到模型可输入样本的完整流水线，包含列名映射、特征选择、
  流式读取、时间断点检测、缺失值插补、目标变量平滑、标准化、滑动窗口、残差标签构造等。
  所有步骤均已参数化，不依赖特定项目或特定字段名。
triggers:
  - 时序预处理
  - 时序特征工程
  - 时间序列预处理
  - 时间序列特征工程
  - 通用预处理流水线
  - 滑动窗口
  - 缺失值插补
  - 标准化
  - 残差标签
  - 时序划分
  - 目标变量平滑
  - 数据清洗
  - ts preprocessing
  - time series preprocessing
  - feature engineering pipeline
---

# 通用时序预测预处理与特征工程流水线

> 本技能提供一个**不依赖具体项目**的多变量时序预测数据预处理与特征工程方案。
> 用户只需传入自己的配置参数（列名、窗口大小、采样间隔等），即可生成模型可用的 `(X, y)` 样本。

---

## 一、设计目标

- **参数化**：所有关键配置（`seq_len`、`horizon`、`max_gap`、`alpha` 等）均可配置。
- **去项目化**：不绑定任何具体业务字段、模型架构或文件路径。
- **模块化**：每个步骤封装为独立函数，可单独调用或组合使用。
- **可扩展**：用户可根据自己的数据特点替换或新增处理步骤。

---

## 二、适用场景

- 多变量时间序列预测（MVTSP）
- 工业传感器时序预测
- 能源、电力、水利、制造等领域的时序回归任务
- 使用 Transformer、KAN、MLP、LSTM 等模型前的数据准备

---

## 三、完整流水线总览

```
原始 CSV / DataFrame
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
│ 按时间间隔切分连续片段                  │
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
│ 目标变量降噪平滑                        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 6: Feature Scaling                 │
│ StandardScaler / MinMaxScaler           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 7: Temporal Split                  │
│ 按时间顺序划分训练/验证/测试集          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 8: Sliding Window                  │
│ 构造 (input, output) 监督样本           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 9: Target Engineering (Optional)   │
│ 残差标签构造                            │
└─────────────────────────────────────────┘
    │
    ▼
模型输入 (X, y)
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

    # 特征缩放
    "scaler_type": "standard",            # standard / minmax / none
    "scaler_path": "scaler.pkl",          # scaler 保存路径

    # 窗口与预测
    "seq_len": 100,                       # 历史输入步长
    "horizon": 30,                        # 未来预测步长

    # 标签工程
    "use_residual_label": True,           # 是否使用残差标签

    # 数据划分
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,

    # DataLoader
    "batch_size": 128,
    "num_workers": 0,
}
```

---

## 五、模块化函数实现

### 5.1 Schema Mapping（列名映射）

把不同来源的列名统一成标准名称。

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

**示例**：

```python
mapping = {
    "Time": "timestamp",
    "water_level": "target",
    "flow_rate": "feat1",
    "pressure": "feat2",
}
df = apply_schema_mapping(df, mapping)
```

---

### 5.2 Chunked Loading（流式分块读取）

```python
import pandas as pd
import numpy as np

def load_data_chunks(data_path, cols_to_read, chunksize=100000, dtype=np.float32):
    """
    流式分块读取 CSV，返回 np.ndarray 和 chunk_ids（可选在外部生成）。
    """
    data_list = []
    for chunk in pd.read_csv(data_path, usecols=cols_to_read, chunksize=chunksize):
        chunk = chunk.apply(pd.to_numeric, errors='coerce')
        data_list.append(chunk.values.astype(dtype))
    return np.vstack(data_list)
```

---

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

---

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

---

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

---

### 5.6 Feature Scaling（特征缩放）

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

def fit_scaler(X_train, scaler_type='standard'):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        return None
    scaler.fit(X_train)
    return scaler

def scale_features(X, scaler):
    if scaler is None:
        return X
    return scaler.transform(X)
```

**注意**：`scaler` 必须只在训练集上 fit，验证集和测试集只做 transform。

---

### 5.7 Temporal Split（时序划分）

```python
from sklearn.model_selection import train_test_split

def temporal_split(valid_indices, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    按时间顺序划分训练/验证/测试集。
    valid_indices: 所有合法样本起始索引
    """
    test_size = val_ratio + test_ratio
    idx_train, idx_temp = train_test_split(
        valid_indices, test_size=test_size, random_state=random_state, shuffle=False
    )
    val_size = val_ratio / test_size
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=1 - val_size, random_state=random_state, shuffle=False
    )
    return idx_train, idx_val, idx_test
```

---

### 5.8 Sliding Window + Target Engineering（滑动窗口与标签工程）

```python
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, indices, seq_len, horizon, use_residual_label=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.indices = indices
        self.seq_len = seq_len
        self.horizon = horizon
        self.use_residual_label = use_residual_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_x = start_idx + self.seq_len
        end_y = end_x + self.horizon

        X_seq = self.X[start_idx:end_x]
        Y_future = self.y[end_x:end_y]

        if self.use_residual_label:
            Y_base = self.y[end_x - 1]
            Y_residual = Y_future - Y_base
            return X_seq, Y_residual, Y_base, Y_future
        else:
            return X_seq, Y_future
```

---

### 5.9 构造合法样本索引

```python
def get_valid_indices(chunk_ids, seq_len, horizon):
    """
    保证每个样本的输入和输出都在同一个 chunk 内。
    """
    valid_indices = []
    for cid in np.unique(chunk_ids):
        mask = (chunk_ids == cid)
        rows = np.where(mask)[0]
        n_samples = len(rows) - seq_len - horizon + 1
        if n_samples > 0:
            valid_indices.extend(rows[:n_samples])
    return np.array(valid_indices)
```

---

## 六、完整使用示例

```python
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader

# 1. 配置
config = {
    "data_path": "data/your_data.csv",
    "time_col": "timestamp",
    "feature_cols": ["flow", "pressure", "load", "temperature"],
    "target_col": "water_level",
    "chunksize": 100000,
    "max_gap": 5.0,
    "time_unit": "s",
    "impute_method": "linear",
    "treat_zero_as_missing": True,
    "fill_remaining_na": False,
    "smooth_target": True,
    "smooth_alpha": 0.3,
    "scaler_type": "standard",
    "scaler_path": "saved_models/scaler.pkl",
    "seq_len": 600,
    "horizon": 20,
    "use_residual_label": True,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "batch_size": 128,
}

# 2. 读取数据
cols = [config["time_col"]] + config["feature_cols"] + [config["target_col"]]
df = pd.read_csv(config["data_path"], usecols=cols)

# 3. 时间列处理
df[config["time_col"]] = pd.to_datetime(df[config["time_col"]], errors='coerce')
df = df.dropna(subset=[config["time_col"]])

# 4. 时间断点检测
chunk_ids = detect_time_gaps(df[config["time_col"]], config["max_gap"], config["time_unit"])

# 5. 提取数值矩阵
all_cols = config["feature_cols"] + [config["target_col"]]
target_idx = len(config["feature_cols"])
X_data = df[all_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)

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

# 8. 时序划分
valid_indices = get_valid_indices(chunk_ids, config["seq_len"], config["horizon"])
idx_train, idx_val, idx_test = temporal_split(
    valid_indices,
    val_ratio=config["val_ratio"],
    test_ratio=config["test_ratio"],
    random_state=config["random_state"]
)

# 9. 标准化（只在训练集 fit）
max_train_row = idx_train[-1] + config["seq_len"] + config["horizon"]
scaler = fit_scaler(X_data[:max_train_row], config["scaler_type"])
X_data = scale_features(X_data, scaler)
joblib.dump(scaler, config["scaler_path"])

# 10. 构造 Dataset / DataLoader
y_data = X_data[:, [target_idx]]
train_ds = TimeSeriesDataset(X_data, y_data, idx_train, config["seq_len"], config["horizon"], config["use_residual_label"])
val_ds = TimeSeriesDataset(X_data, y_data, idx_val, config["seq_len"], config["horizon"], config["use_residual_label"])
test_ds = TimeSeriesDataset(X_data, y_data, idx_test, config["seq_len"], config["horizon"], config["use_residual_label"])

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
```

---

## 七、不同任务的适配指南

### 7.1 如果你的数据没有明显时间断点

把 `max_gap` 设得很大，或跳过 `detect_time_gaps`，直接令 `chunk_ids = np.zeros(len(df))`。

### 7.2 如果你的 0 值是合法值

设置 `treat_zero_as_missing=False`，只对真正的 `NaN` 做插补。

### 7.3 如果你不想用残差标签

设置 `use_residual_label=False`，Dataset 只返回 `(X_seq, Y_future)`。

此时模型应直接预测未来值，而不是残差。

### 7.4 如果你的目标变量不需要平滑

设置 `smooth_target=False`。

### 7.5 如果你用 MinMax 缩放

设置 `scaler_type='minmax'`。

### 7.6 如果你的序列很短

减小 `seq_len`，但要保证 `seq_len + horizon < chunk 长度`。

### 7.7 如果你要做多步预测多变量

把 `target_col` 改成多个列，`y` 的维度从 `(N, 1)` 变成 `(N, num_targets)`。

---

## 八、与模型输出的配合

| 标签工程方式 | 模型 forward 应该输出 | 验证/推理时如何处理 |
|---|---|---|
| `use_residual_label=True` | 残差 `Δy` | `pred_abs = last_known_value + pred_residual`，再反归一化 |
| `use_residual_label=False` | 绝对值 `y` | 直接反归一化 |

---

## 九、常见问题

### Q1: 为什么 scaler 要在训练集上 fit？

防止 **数据泄露（Data Leakage）**。如果在全量数据上 fit scaler，验证集和测试集的统计信息会泄露到训练过程中，导致指标虚高。

### Q2: 残差标签有什么好处？

让模型学习「变化量」而非「绝对值」，对非平稳时序更稳定，能缓解分布漂移问题。

### Q3: Double EMA 会不会造成未来信息泄露？

不会。Double EMA 是在历史数据上做指数平滑，没有使用未来值。但它会引入一定的相位滞后，零滞后校正（`2*s1 - s2`）可以部分抵消。

### Q4: 流式读取和一次性读取怎么选？

- CSV < 1GB：可以一次性读取。
- CSV > 1GB 或内存有限：建议用 `chunksize` 流式读取。

---

## 十、可扩展的改进方向

| 改进点 | 说明 |
|---|---|
| 异常值检测 | 加入 3-sigma / IQR / Isolation Forest 异常值处理 |
| 特征构造 | 加入滞后特征、差分特征、滚动统计特征 |
| 多尺度窗口 | 构造不同时间尺度的输入窗口 |
| 类别特征 | 加入工况、设备状态等类别特征的编码 |
| 样本权重 | 对稀有工况或近期样本加权 |
| 在线标准化 | 推理时支持流式更新 scaler |

---

## 十一、参考术语表

| 操作 | 中文术语 | 英文术语 |
|---|---|---|
| 列名映射 | 模式映射 | Schema Mapping |
| 特征选择 | 特征选择 | Feature Selection |
| 分块读取 | 流式加载 | Chunked Loading / Streaming ETL |
| 时间断点切分 | 时间序列分段 | Time Series Segmentation |
| 缺失值插补 | 缺失值插补 | Missing Value Imputation |
| 目标变量平滑 | 双重指数平滑 | Double Exponential Smoothing |
| 标准化 | 标准化 | Standardization |
| 滑动窗口 | 滑动窗口 | Sliding Window |
| 残差标签 | 标签工程 / 残差化 | Target Engineering / Residualization |
| 时序划分 | 时序划分 | Temporal Split |

---

## 十二、与项目特化版的关系

- **通用版**：`.claude/skills/ts-preprocessing-pipeline.md`（本文件）
- **华电项目特化版**：`.claude/skills/hbhd-predict-preprocessing.md`

通用版提供方法论和可复用代码模板；项目特化版则把通用模板应用到了华电的具体数据、字段和模型上。
