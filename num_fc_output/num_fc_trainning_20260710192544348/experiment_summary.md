# 工业时序预测实验总结报告

## 1. 实验信息

| 项目 | 内容 |
| --- | --- |
| 实验时间戳 | 20260710192544348 |
| 输出目录 | `num_fc_output/num_fc_trainning_20260710192544348` |
| 使用算子 | `itransformer_forecaster` |
| 实验模式 | 训练 + 推理 + 评估 |
| 目标变量 | `diya_qibao_shuiwei_youxuanzhi` |
| 历史窗口长度 seq_len | 100 |
| 预测步长 pred_len | 30 |

## 2. 数据信息

### 2.1 原始数据

- 原始数据路径：`data/total_final_0430/total_final_0430_balanced.csv`
- 本次实验截取：前 **10000 条连续记录**
- 时间列：`datatime`

### 2.2 特征列（输入）

- `ningjie_shuiliuliang_youxuanzhi`
- `gaoya_geishuiliuliang_youxuanzhi`
- `chuyangqi_tiaojiefa_weizhifankui`
- `ranji_fuhe`
- `qiji_fuhe`
- `chuyangqi_rukou_yali`
- `diya_qibao_yali_youxuanzhi`
- `diya_zhuqiliuliang_youxuanzhi`
- `gaoya_jianwenshui_liuliang`
- `ranji_paiqi_wendu`
- `diya_qibao_shuiwei_youxuanzhi`（目标列，作为 iTransformer 历史观测输入，置于最后一列）

### 2.3 预处理参数

| 参数 | 值 |
| --- | --- |
| max_gap | 5.0 s |
| time_unit | s |
| impute_method | linear |
| treat_zero_as_missing | True |
| fill_remaining_na | False |
| smooth_target | True |
| smooth_alpha | 0.3 |
| train_ratio | 0.70 |
| val_ratio | 0.15 |
| test_ratio | 0.15 |

### 2.4 输出数据文件

| 文件 | 路径 | 说明 |
| --- | --- | --- |
| 训练集 | `preprocessed_for_tsas_10k/train.csv` | 训练 + 验证，共 8500 行 |
| chunk_ids | `preprocessed_for_tsas_10k/chunk_ids.csv` | 时间断点 chunk 编号，无表头 |
| 测试窗口 | `preprocessed_for_tsas_10k/test_window.csv` | 测试集前 seq_len=100 行 |
| 测试真值 | `preprocessed_for_tsas_10k/test_truth.csv` | 测试集第 [100:130] 行的目标值 |

## 3. 模型参数

```yaml
operator:
  name: itransformer_forecaster
  input_columns:
  - ningjie_shuiliuliang_youxuanzhi
  - gaoya_geishuiliuliang_youxuanzhi
  - chuyangqi_tiaojiefa_weizhifankui
  - ranji_fuhe
  - qiji_fuhe
  - chuyangqi_rukou_yali
  - diya_qibao_yali_youxuanzhi
  - diya_zhuqiliuliang_youxuanzhi
  - gaoya_jianwenshui_liuliang
  - ranji_paiqi_wendu
  - diya_qibao_shuiwei_youxuanzhi
  target_column: diya_qibao_shuiwei_youxuanzhi
  config:
    seq_len: 100
    pred_len: 30
    d_model: 256
    nhead: 4
    num_layers: 2
    dim_feedforward: 512
    dropout: 0.2
    step_cond_head: false
    lag_aware: true
    lag_max: 16
    lag_bias_scale: 2.0
    lag_dropout: 0.2
    kan_grid_size: 5
    target_idx: -1
    epochs: 30
    batch_size: 128
    lr: 0.0002
    weight_decay: 1.0e-05
    early_stop_patience: 12
    train_ratio: 0.7
    val_ratio: 0.15
    trend_weight: 1.0
    time_weight_start: 0.1
    time_weight_end: 1.0
    max_grad_norm: 1.0
    scheduler_factor: 0.5
    scheduler_patience: 3
    device: auto

```

训练过程：共 30 个 epoch，在第 19 个 epoch 触发早停（early_stop_patience=12）。
模型保存路径：`num_fc_output/num_fc_trainning_20260710192544348/model/`

## 4. 推理与评价

### 4.1 推理设置

- 推理输入：`preprocessed_for_tsas_10k/test_window.csv`（第一个窗口，100 行）
- 推理输出：`num_fc_output/num_fc_trainning_20260710192544348/forecasting_result.csv`（30 步预测）
- 对齐后评价输入：`num_fc_output/num_fc_trainning_20260710192544348/eval_input.csv`

### 4.2 评价指标配置

```yaml
operators:
- name: forecasting_metrics
  alias: forecast_metrics
  truth_columns:
  - y_true
  predict_columns:
  - y_pred
  config:
    naive_error: 1.0
    epsilon: 1e-8
    max_dtw_len: 2000

```

### 4.3 评价结果

| 指标 | 值 |
| --- | --- |
| MSE | 67.161761 |
| RMSE | 8.195228 |
| MAE | 6.263479 |
| MAPE | 1.498662 |
| SMAPE | 1.514210 |
| MASE | 6.263479 |
| DTW | 4.816052 |
| R² | -2.788708 |

**结果解读**：
- MAE/RMSE 绝对误差在 6~8 左右，相对于目标变量约 -400 的取值尺度，相对误差约 1.5%。
- MAPE/SMAPE 约为 1.5%，说明逐点相对误差较低。
- MASE 等于 MAE（因 naive_error=1.0），表示相对随机游走基线的误差倍数。
- R² 为负值，说明在此测试窗口上，模型整体拟合优度弱于简单均值基线；可能与测试窗口仅 30 步、目标值波动范围小、或模型未充分捕捉短期趋势有关。

## 5. 产出文件清单

| 文件 | 路径 | 说明 |
| --- | --- | --- |
| 预测配置文件 | `num_fc_output/num_fc_trainning_20260710192544348/forecasting_config.yaml` | itransformer_forecaster 算子配置 |
| 训练模型目录 | `num_fc_output/num_fc_trainning_20260710192544348/model/` | 保存的模型权重与状态 |
| 推理结果 | `num_fc_output/num_fc_trainning_20260710192544348/forecasting_result.csv` | 30 步预测值 |
| 评价输入 | `num_fc_output/num_fc_trainning_20260710192544348/eval_input.csv` | y_true / y_pred 对齐数据 |
| 评价配置 | `num_fc_output/num_fc_trainning_20260710192544348/eval_config.yaml` | forecasting_metrics 配置 |
| 评价结果 | `num_fc_output/num_fc_trainning_20260710192544348/eval_result.json` | 8 项时序预测指标 |
| 实验总结 | `num_fc_output/num_fc_trainning_20260710192544348/experiment_summary.md` | 本报告 |

---
*报告生成时间：2026-07-10 19:37:36*
