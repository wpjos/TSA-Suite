# 工业时序预测实验总结报告

## 一、实验信息

| 项目 | 内容 |
|---|---|
| 实验时间 | 2026-07-10 |
| 算子名称 | `itransformer_forecaster` |
| 实验模式 | 训练 + 推理 + 评价 |
| 输出目录 | `num_fc_output/` |

---

## 二、数据信息

### 2.1 原始数据

| 项目 | 内容 |
|---|---|
| 原始数据路径 | `data/total_final_0430/total_final_0430_balanced.csv` |
| 采样策略 | 取前 10000 条连续记录 |
| 时间列 | `datatime` |
| 目标列 | `diya_qibao_shuiwei_youxuanzhi` |
| 特征列 | `ningjie_shuiliuliang_youxuanzhi`, `gaoya_geishuiliuliang_youxuanzhi`, `chuyangqi_tiaojiefa_weizhifankui`, `ranji_fuhe`, `qiji_fuhe`, `chuyangqi_rukou_yali`, `diya_qibao_yali_youxuanzhi`, `diya_zhuqiliuliang_youxuanzhi`, `gaoya_jianwenshui_liuliang`, `ranji_paiqi_wendu` |

### 2.2 数据集划分

| 数据集 | 行数 | 说明 |
|---|---|---|
| 训练集 | 7000 | 外部训练集 |
| 验证集 | 1500 | 外部验证集（TSAS 算子内部再做 train/val 切分） |
| 测试集 | 1500 | 用于生成 test_window 和 test_truth |
| **合计** | **10000** | 采样后的总记录数 |

### 2.3 预处理输出

| 文件 | 路径 | 形状 |
|---|---|---|
| `train.csv` | `preprocessed_for_tsas_10k/train.csv` | `(8500, 12)` |
| `chunk_ids.csv` | `preprocessed_for_tsas_10k/chunk_ids.csv` | `(8500, 1)` |
| `test_window.csv` | `preprocessed_for_tsas_10k/test_window.csv` | `(100, 12)` |
| `test_truth.csv` | `preprocessed_for_tsas_10k/test_truth.csv` | `(30, 1)` |
| `meta.yaml` | `preprocessed_for_tsas_10k/meta.yaml` | — |

预处理阶段检测到 **8 个连续时间片段**，并通过 `--chunk-ids` 传入训练算子。

---

## 三、预处理参数

| 参数 | 值 |
|---|---|
| `seq_len` | 100 |
| `pred_len` | 30 |
| `max_gap` | 5.0 |
| `time_unit` | `"s"` |
| `impute_method` | `"linear"` |
| `treat_zero_as_missing` | True |
| `fill_remaining_na` | False |
| `smooth_target` | True |
| `smooth_alpha` | 0.3 |
| `train_ratio` | 0.70 |
| `val_ratio` | 0.15 |
| `test_ratio` | 0.15 |

---

## 四、模型参数

| 参数 | 值 |
|---|---|
| 算子 | `itransformer_forecaster` |
| `seq_len` | 100 |
| `pred_len` | 30 |
| `device` | `cpu` |
| 其他参数 | 使用算子默认值 |

---

## 五、训练过程

- 训练成功完成。
- 训练过程中验证损失持续下降，触发早停于 **第 25 个 epoch**。
- 模型已保存至 `num_fc_output/model/`。

---

## 六、评价结果

评价输入：`num_fc_output/eval_input.csv`（30 个样本点）

| 指标 | 值 |
|---|---|
| MSE | 133.5993 |
| RMSE | 11.5585 |
| MAE | 9.0959 |
| MAPE | 2.1886% |
| SMAPE | 2.1988% |
| MASE | 909.5947 |
| DTW | 7.5616 |
| R² | -6.5366 |

### 结果解读

- **MAPE / SMAPE** 在 2.2% 左右，说明预测值与真实值的相对误差较小。
- **R² 为负数**，说明模型预测效果差于简单的均值基线，可能存在一定的系统偏差或趋势估计不足。
- **MASE 较高**，结合 `naive_error=0.01` 的设置，该指标主要受分母影响，参考价值有限。
- 建议后续尝试：调整模型超参数、增加训练 epoch、使用更大样本、或尝试 LightGBM / XGBoost 基线对比。

---

## 七、产出文件清单

```text
num_fc_output/
├── forecasting_config.yaml      # 预测算子配置
├── model/                       # 训练保存的模型
├── forecasting_result.csv       # 推理预测结果 (30, 1)
├── eval_input.csv               # 对齐后的评价数据 (30, 2)
├── eval_config.yaml             # 评价指标配置
├── eval_result.json             # 评价指标计算结果
└── experiment_summary.md        # 本实验总结报告
```

---

## 八、使用过的命令

### 预处理

```bash
python run_preprocessing_for_tsas_10k.py
```

### 训练

```bash
python -m tsas.engine.operator.cli forecasting fit \
  --input preprocessed_for_tsas_10k/train.csv \
  --target diya_qibao_shuiwei_youxuanzhi \
  --config num_fc_output/forecasting_config.yaml \
  --chunk-ids preprocessed_for_tsas_10k/chunk_ids.csv \
  --save num_fc_output/model
```

### 推理

```bash
python -m tsas.engine.operator.cli forecasting run \
  --input preprocessed_for_tsas_10k/test_window.csv \
  --config num_fc_output/forecasting_config.yaml \
  --load num_fc_output/model \
  --output num_fc_output/forecasting_result.csv
```

### 评价

```bash
python -m tsas.engine.operator.cli evaluation run \
  --input num_fc_output/eval_input.csv \
  --config num_fc_output/eval_config.yaml \
  --output num_fc_output/eval_result.json
```
