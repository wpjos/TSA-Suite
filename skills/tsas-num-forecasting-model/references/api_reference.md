# TSAS-CLI 时序预测算子使用参考

本文档介绍如何使用 `tsas.engine.operator.cli` 提供的命令行工具来执行时序预测训练、推理和评价指标计算。

## 1. 命令行调用入口

CLI 工具既支持通过 `tsas.engine.operator.cli` 作为统一入口（然后通过指定子模块名分发），也支持直接调用具体的模块入口。

**方式一：统一调用入口**

```bash
python -m tsas.engine.operator.cli <模块名> <子命令> [参数]
```

**方式二：指定模块调用入口**

```bash
python -m tsas.engine.operator.cli.<模块名> <子命令> [参数]
```

支持的模块名（`<模块名>`）：

- `forecasting`：时序预测模块
- `evaluation`：评价指标模块

支持的子命令：

- `help`：查看可用算子列表或指定算子的详细参数文档
- `run`：执行算子的推理/计算
- `fit`：执行算子的训练（仅支持有状态的可学习算子）

全局参数 `--encoding` 可放在模块名之前，用于指定终端输出编码：

```bash
python -m tsas.engine.operator.cli --encoding utf-8 forecasting help
```

## 2. `forecasting` 模块

### 2.1 配置文件格式

预测算子配置文件使用单个 `operator` 块，示例 `forecasting_config.yaml`：

```yaml
operator:
  name: "itransformer_forecaster"
  input_columns:
    - "feat_0"
    - "feat_1"
    - "target"
  target_column: "target"
  config:
    seq_len: 100
    pred_len: 30
    d_model: 128
    nhead: 4
    num_layers: 2
    epochs: 30
    batch_size: 128
```

字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `operator.name` | 是 | 算子名称，如 `itransformer_forecaster` |
| `operator.input_columns` | 否 | 输入特征列列表；为空或不写则使用全部列 |
| `operator.target_column` | 否 | 训练时的目标列名；`fit` 子命令的 `--target` 参数优先级更高 |
| `operator.config` | 否 | 算子实例参数字典；未指定字段使用算子默认值 |

配置文件支持 YAML（`.yaml` / `.yml`）、JSON（`.json`）、JSON5（`.json5`）。

### 2.2 获取帮助（help）

查看所有已注册的预测算子：

```bash
python -m tsas.engine.operator.cli forecasting help
```

查看指定算子的详细参数文档：

```bash
python -m tsas.engine.operator.cli forecasting help itransformer_forecaster
```

`help` 输出包含：算子说明、实例参数表（参数名、类型、必填、默认值、值域、说明）。

> 注意：`help` 列表通常不显示 `*_mimo_forecaster` 别名，但配置文件中可以直接使用这些名称。

### 2.3 训练（fit）

```bash
python -m tsas.engine.operator.cli forecasting fit \
  --input train.csv \
  --target target \
  --config forecasting_config.yaml \
  --save model_dir/ \
  --chunk-ids chunk_ids.csv
```

参数说明：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--input` / `-i` | 是 | 训练数据文件路径，CSV 格式，形状 `(timesteps, num_features)` |
| `--target` / `-t` | 是 | 目标变量列名 |
| `--config` / `-c` | 是 | 算子配置文件路径 |
| `--save` | 否 | 保存训练后模型的目录路径 |
| `--chunk-ids` | 否 | chunk_ids 文件路径，CSV 单列表，无表头；`itransformer_forecaster` 使用以避免窗口跨断层，树模型忽略并警告 |

训练完成后，`--save` 指定的目录下会保存：

- `config.json`：算子配置与元信息
- `_scaler.npz`（iTransformer）：标准化参数
- `_model_weights.pt`（iTransformer）：模型权重
- `_forecaster_state.npz`：预测器状态（特征数、目标数、目标索引等）

树模型（LightGBM / XGBoost）会保存对应的 booster 文件与状态文件。

### 2.4 推理（run）

```bash
python -m tsas.engine.operator.cli forecasting run \
  --input window.csv \
  --config forecasting_config.yaml \
  --load model_dir/ \
  --output pred.csv
```

参数说明：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--input` / `-i` | 是 | 推理窗口文件路径，CSV 格式，形状 `(seq_len, num_features)` |
| `--config` / `-c` | 是 | 算子配置文件路径 |
| `--load` | 否 | 已训练模型目录路径；可训练算子未训练时必须提供 |
| `--output` / `-o` | 是 | 输出预测结果文件路径 |
| `--keep-input` | 否 | 将原始输入列拼接到输出结果中 |
| `--auto-suffix` | 否 | 自动对冲突列名追加后缀（需与 `--keep-input` 配合使用） |

输出文件 `pred.csv` 的形状为 `(pred_len, num_targets)`，默认列名为 `forecast_0`、`forecast_1`、……

### 2.5 算子特殊说明

#### `itransformer_forecaster`

- 来源：`src/tsas/engine/operator/forecasting/itransformer.py`
- 基于 NPU-compatible iTransformer 模型：Dense Transformer Encoder + KAN 预测头 + Lag-Aware Refiner + 残差预测策略
- 需要 `torch`
- **仅支持单目标预测**，即 `y` 的形状必须为 `(timesteps, 1)`
- `target_idx` 默认 `-1`，表示目标列为 `input_columns` 的最后一列
- `device` 支持 `auto`、`cpu`、`cuda`、`npu`
- 训练时使用 `StandardScaler` 对训练样本拟合，避免数据泄漏

关键配置参数：

| 参数 | 默认值 | 值域 | 说明 |
|------|--------|------|------|
| `seq_len` | 100 | [100, 300] | 输入历史窗口长度 |
| `pred_len` | 30 | [30, 80] | 预测未来步长 |
| `d_model` | 256 | [128, 1024] | 模型嵌入维度 |
| `nhead` | 4 | [2, 16] | 注意力头数 |
| `num_layers` | 2 | [1, 8] | Encoder 层数 |
| `dim_feedforward` | 512 | [256, 2048] | FFN 隐藏层维度 |
| `dropout` | 0.2 | [0.1, 0.8] | Dropout 比率 |
| `lag_aware` | true | bool | 是否启用 Lag-Aware Refiner |
| `lag_max` | 16 | [8, 64] | 最大滞后步数 |
| `lag_bias_scale` | 2.0 | [1.0, 8.0] | 互相关先验偏置缩放 |
| `lag_dropout` | 0.2 | [0.1, 0.8] | Lag Refiner Dropout |
| `kan_grid_size` | 5 | [2, 20] | KAN 网格大小 |
| `target_idx` | -1 | [-1, -1] | 目标变量在特征中的列索引，-1 表示最后一列 |
| `epochs` | 30 | [15, 120] | 最大训练轮数 |
| `batch_size` | 128 | [64, 512] | 训练批次大小 |
| `lr` | 0.0002 | [0.0001, 0.0008] | 学习率 |
| `weight_decay` | 1e-5 | [5e-6, 4e-5] | 权重衰减 |
| `early_stop_patience` | 12 | [6, 48] | 早停耐心轮数 |
| `train_ratio` | 0.7 | [0.7, 0.7] | 训练集占比 |
| `val_ratio` | 0.15 | [0.15, 0.15] | 验证集占剩余数据比例 |
| `trend_weight` | 1.0 | [0.5, 4.0] | 趋势损失权重 |
| `time_weight_start` | 0.1 | [0.05, 0.4] | 时间加权损失起始权重 |
| `time_weight_end` | 1.0 | [0.5, 4.0] | 时间加权损失结束权重 |
| `max_grad_norm` | 1.0 | [0.5, 4.0] | 梯度裁剪范数 |
| `scheduler_factor` | 0.5 | [0.25, 2.0] | 学习率衰减因子 |
| `scheduler_patience` | 3 | [1, 12] | 学习率衰减耐心轮数 |
| `device` | auto | enum(auto, cpu, cuda, npu) | 计算设备 |

#### `lightgbm_forecaster` / `lightgbm_mimo_forecaster`

- 来源：`src/tsas/engine/operator/forecasting/lightgbm.py`
- 需要 `lightgbm`
- `strategy=None`（默认）为 Direct 多步策略：每个 `(horizon, target)` 组合训练一个 booster
- `strategy='MIMO'`：每个目标训练一个 booster，将步长索引作为额外特征
- `lightgbm_mimo_forecaster` 是 `lightgbm_forecaster(strategy='MIMO')` 的别名，默认 `num_leaves=63`

关键配置参数：

| 参数 | 默认值 | 值域 | 说明 |
|------|--------|------|------|
| `seq_len` | 96 | [1, 4096] | 输入历史窗口长度 |
| `pred_len` | 24 | [1, 500] | 预测未来步长 |
| `strategy` | None | None / "MIMO" | 多步预测策略 |
| `num_leaves` | 31 | [1, 1024] | 树叶子数 |
| `learning_rate` | 0.05 | [1e-6, 1.0] | 学习率 |
| `n_estimators` | 200 | [1, 10000] |  boosting 轮数 |
| `min_child_samples` | 20 | [1, 10000] | 叶子最小样本数 |
| `reg_alpha` | 0.1 | [0.0, 10.0] | L1 正则化 |
| `reg_lambda` | 0.1 | [0.0, 10.0] | L2 正则化 |
| `device` | cpu | enum(cpu, gpu) | 计算设备 |
| `n_jobs` | -1 | [-1, 64] | 并行线程数 |

#### `xgboost_forecaster` / `xgboost_mimo_forecaster`

- 来源：`src/tsas/engine/operator/forecasting/xgboost.py`
- 需要 `xgboost`
- 策略设计与 LightGBM 相同
- `xgboost_mimo_forecaster` 是 `xgboost_forecaster(strategy='MIMO')` 的别名，默认 `max_depth=6`

关键配置参数：

| 参数 | 默认值 | 值域 | 说明 |
|------|--------|------|------|
| `seq_len` | 96 | [1, 4096] | 输入历史窗口长度 |
| `pred_len` | 24 | [1, 500] | 预测未来步长 |
| `strategy` | None | None / "MIMO" | 多步预测策略 |
| `max_depth` | 4 | [1, 16] | 树最大深度 |
| `learning_rate` | 0.05 | [1e-6, 1.0] | 学习率 |
| `n_estimators` | 200 | [1, 10000] | boosting 轮数 |
| `min_child_weight` | 1.0 | [0.0, 100.0] | 最小子节点权重 |
| `reg_alpha` | 0.1 | [0.0, 10.0] | L1 正则化 |
| `reg_lambda` | 0.1 | [0.0, 10.0] | L2 正则化 |
| `device` | cpu | enum(cpu, gpu) | 计算设备 |
| `n_jobs` | -1 | [-1, 64] | 并行线程数 |

## 3. `evaluation` 模块

### 3.1 `forecasting_metrics`

- 来源：`src/tsas/engine/operator/evaluation/forecasting_metrics.py`
- 算子名称：`forecasting_metrics`
- 输入：`(y_true, y_pred)` 元组，支持 `np.ndarray` 与 `pd.DataFrame`，内部统一拉平为 1-D
- 输出：`ForecastingMetricResult`，包含 MAE、RMSE、MAPE、DTW

配置文件示例 `eval_config.yaml`：

```yaml
operators:
  - name: "forecasting_metrics"
    alias: "forecast_metrics"
    truth_columns: [ "y_true" ]
    predict_columns: [ "y_pred" ]
    config:
      epsilon: 1e-8
      max_dtw_len: 2000
```

字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 算子名称，固定为 `forecasting_metrics` |
| `alias` | 否 | 结果输出中的别名 |
| `truth_columns` | 是 | 真实值列名列表 |
| `predict_columns` | 是 | 预测值列名列表 |
| `config.main_scores` | 否 | 默认暴露全部 4 项指标；可覆写以选择 HPO 优化目标 |
| `config.epsilon` | 否 | 零值保护常数，默认 `1e-8` |
| `config.max_dtw_len` | 否 | DTW 最大采样长度，默认 `2000` |

执行评价：

```bash
python -m tsas.engine.operator.cli evaluation run \
  --input eval_aligned.csv \
  --config eval_config.yaml \
  --output eval_result.json
```

输出 JSON 结构示例：

```json
{
  "results": {
    "forecast_metrics": {
      "result": {
        "mae": 0.25,
        "rmse": 0.316,
        "mape": 5.2,
        "dtw": 0.3
      },
      "main_scores": {
        "mae": 0.25,
        "rmse": 0.316,
        "mape": 5.2,
        "dtw": 0.3
      }
    }
  }
}
```

## 4. 数据与配置文件格式

### 数据文件

- 首选 CSV 格式，第一行为列名。
- 训练数据：`fit` 需要完整时间序列，形状 `(timesteps, num_features)`。
- 推理窗口：`run` 需要一个历史窗口，形状 `(seq_len, num_features)`，其中 `seq_len` 与算子配置一致。
- 评价数据：需要包含与 `truth_columns` / `predict_columns` 对齐的真实值和预测值列。

### 配置文件

- 预测算子配置：`operator` 单算子块。
- 评价算子配置：`operators` 多算子列表（当前仅使用 `forecasting_metrics`）。
- 支持 `.yaml`、`.yml`、`.json`、`.json5`。

### 依赖说明

- `itransformer_forecaster` 需要 `torch`。
- `lightgbm_forecaster` 需要 `lightgbm`。
- `xgboost_forecaster` 需要 `xgboost`；macOS 上若加载失败，通常需要安装 `libomp`。
- `forecasting_metrics` 的 DTW 指标需要可选依赖 `fastdtw`；未安装时自动回退为 MAE。
