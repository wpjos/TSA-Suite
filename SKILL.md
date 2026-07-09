---
name: real-industrial-forecasting-pipeline
description: 基于 TSA-Suite 的真实工业时序预测端到端流程（预处理、训练、推理、评估）
version: 1.0.0
---

# 真实工业时序预测端到端流程

本 skill 用于在 TSA-Suite 项目中执行**真实工业数据**的时序预测训练与验证。

与现有 `industrial-forecasting-skill` 的区别：

- `industrial-forecasting-skill`：面向合成/demo 数据，使用单一 CSV + `split` 列。
- `real-industrial-forecasting-skill`：面向真实工业 CSV，使用 `preprocess_for_tsa.py` 进行时间分片与预处理，生成 `x.npy` / `y.npy` / `chunk_ids.npy`。

## 触发条件

当用户请求以下任务时触发：

- "在真实工业数据上训练时序预测模型"
- "运行 TSA-Suite 真实数据预测流程"
- "预处理并训练 ITransformer"
- "用 preprocess_for_tsa 的结果训练模型"
- "训练并验证工业时序预测模型"

## 功能范围

1. **数据预处理（可选）**：调用 `preprocess_for_tsa.py` 生成 `x.npy` / `y.npy` / `chunk_ids.npy`。
2. **模型训练**：使用 `ITransformerForecaster` 在真实工业数据上训练。
3. **内部验证**：利用 `ITransformerForecaster` 内置的 `val_ratio` 和早停机制。
4. **测试集推理评估（可选）**：调用 `inference_tsa.py` 计算 RMSE / MAE / MAPE / MASE / DTW。
5. **HPO 参数化入口**：提供 JSON 参数化脚本，供超参数优化循环调用。

## 执行步骤

### 一键完整流程（推荐）

```bash
python skills/real-industrial-forecasting-skill/run_all_pipelines.py \
  --run_preprocessing \
  --data_file data/total_final_0430/total_final_0430_balanced.csv \
  --run_inference
```

该命令会自动：
1. 调用 `preprocess_for_tsa.py` 生成预处理数组
2. 训练 `ITransformerForecaster`
3. 保存模型到 `saved_models/tsa_itransformer/`
4. 在测试集上推理并输出指标

### 分步执行

```bash
# 1. 预处理（只需执行一次）
python preprocess_for_tsa.py \
  --data_file data/total_final_0430/total_final_0430_balanced.csv \
  --output_dir preprocessed_tsa

# 2. 训练 + 测试集评估
python skills/real-industrial-forecasting-skill/run_train_pipeline.py \
  --run_inference
```

### 使用指定超参数训练

```bash
python skills/real-industrial-forecasting-skill/run_train_pipeline.py \
  --config skills/real-industrial-forecasting-skill/configs/real_forecast_config.yaml \
  --d_model 256 \
  --nhead 4 \
  --num_layers 2 \
  --epochs 20 \
  --run_inference
```

### HPO 参数化入口

```bash
python skills/real-industrial-forecasting-skill/run_train_with_params.py \
  --params '{"d_model": 256, "nhead": 4, "num_layers": 2, "epochs": 30}' \
  --data_file data/total_final_0430/total_final_0430_balanced.csv \
  --model_tag trial_001 \
  --output -
```

输出为 JSON：

```json
{
  "params": {...},
  "metrics": {
    "RMSE": ...,
    "MAE": ...,
    "MAPE(%)": ...,
    "MASE": ...,
    "DTW": ...
  },
  "model_dir": "..."
}
```

## 输出物

| 路径 | 说明 |
|------|------|
| `preprocessed_tsa/x.npy` | 输入特征数组 `(timesteps, num_features)` |
| `preprocessed_tsa/y.npy` | 目标数组 `(timesteps, 1)` |
| `preprocessed_tsa/chunk_ids.npy` | 连续片段 ID `(timesteps,)` |
| `saved_models/tsa_itransformer/` | 训练好的模型 |
| `saved_models/tsa_itransformer_<model_tag>/` | HPO  trial 模型 |

## 关键指标

- RMSE
- MAE
- MAPE(%)
- MASE
- DTW（需 `fastdtw`，缺失时回退到 MAE）

## 环境要求

- Python >= 3.11
- torch
- numpy、pandas、scikit-learn、joblib、yaml
- TSA-Suite 已安装或在 `PYTHONPATH` 中

## 注意事项

- 首次运行建议带 `--run_preprocessing` 生成预处理数组。
- 默认使用 CPU，可在配置中改为 `cuda` 或 `npu`。
- `chunk_ids` 不会随模型一起保存，推理时会重新从 `chunk_ids.npy` 加载并 `set_chunk_ids()`。
- 预处理后的目标列默认是 `x` 的最后一列，`target_idx=-1` 通常不需要修改。