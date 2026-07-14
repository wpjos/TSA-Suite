请基于以下三个 skill，完成一次 TSAS-CLI 时序预测实验：测试集 stride=1 滑窗评估 + 基于 `bo-for-experiment` 的 10 轮超参数闭环寻优。

- `.claude/skills/ts-preprocessing-for-tsas-cli.md`
- `.claude/skills/tsas-num-forecasting-model/SKILL.md`
- `.claude/skills/bo-for-experiment/SKILL.md`

## 一、实验配置

| 配置项 | 值 |
|---|---|
| 原始数据路径 | `<data_path>` |
| 时间列 | `<time_col>` |
| 输入特征列 | `<feature_cols>` |
| 目标列 | `<target_col>` |
| `seq_len` / `pred_len` | `<seq_len>` / `<pred_len>` |
| 划分比例 | `0.70 / 0.15 / 0.15` |
| 预测算子 | `<operator>` |
| Python 环境 | TSAS: `<tsa-suite-env-python>`, BO: `<bo-env-python>` |
| 输出根目录 | `<output_dir>` |
| 推理 batch size | `<eval_batch_size>`（默认 2000；一次性 OOM 时改小） |

### BO 配置

- 任务 ID：`<bo_task_id>`
- 参数空间：`<bo_params_config>`（示例见下）
- 优化目标：`<bo_objectives>`，示例 `[{"name":"overall_mae","direction":"min"}]`
- 默认参数：`<default_params>`
- 迭代：共 **10 轮**，trial 0 用默认参数，trial 1~9 由 BO 推荐

参数空间示例：

```json
[
  {"name":"d_model","type":"int","lb":128,"ub":512},
  {"name":"nhead","type":"int","lb":2,"ub":8},
  {"name":"num_layers","type":"int","lb":1,"ub":4},
  {"name":"lr","type":"pow","lb":1e-4,"ub":1e-2,"base":10},
  {"name":"batch_size","type":"int","lb":32,"ub":256}
]
```

## 二、数据预处理

按 `ts-preprocessing-for-tsas-cli.md` 预处理一次，输出：

- `train.csv`、`chunk_ids.csv`、`test_window.csv`、`test_truth.csv`、`meta.yaml`
- **额外**：`test.csv`（完整测试集）和 `test_chunk_ids.csv`（单列表、无表头），用于滑窗评估

## 三、单轮训练与评估

### 训练

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting fit \
  --input <output_dir>/train.csv --target <target_col> \
  --config <output_dir>/trial_<i>/forecasting_config.yaml \
  --save <output_dir>/trial_<i>/model \
  --chunk-ids <output_dir>/chunk_ids.csv
```

`forecasting_config.yaml` 模板：

```yaml
operator:
  name: "<operator>"
  input_columns: [<feature_cols>, "<target_col>"]
  target_column: "<target_col>"
  config:
    seq_len: <seq_len>
    pred_len: <pred_len>
    epochs: <epochs>
    d_model: <d_model>
    nhead: <nhead>
    num_layers: <num_layers>
    lr: <lr>
    batch_size: <batch_size>
```

### 滑窗评估

生成 `evaluate_sliding_window.py`，要求：

1. 读取 `test.csv` 和 `test_chunk_ids.csv`，按 chunk 分组
2. 每 chunk 内 `stride=1` 滑窗；chunk 长度 `< seq_len + pred_len` 则跳过
3. 把所有有效窗口拼接成 `x_test`，形状 `[N_windows, seq_len, num_features]`；对应的真实值 `y_true` 形状 `[N_windows, pred_len, num_targets]`
4. **批量推理**：用算子 Python API 加载已训练模型并推理
   ```python
   forecaster = <OperatorClass>.load(<output_dir>/trial_<i>/model)
   y_pred = forecaster.run(x_test)  # 形状 [N_windows, pred_len, num_targets]
   ```
   - **默认分批推理**：把所有窗口拼成 `x_test` 后，按 `eval_batch_size=2000` 分批调用 `forecaster.run(batch)`，再拼接成完整 `y_pred`
   - 若显存/内存充足且窗口数不大，可增大 `eval_batch_size` 甚至设为窗口总数进行一次性推理
   - 若 OOM，改小 `eval_batch_size`
5. 对每个预测步构造 `eval_step_{h}.csv`，调用 `evaluation run` 得到该步指标
6. 构造 `eval_all.csv`，调用 `evaluation run` 得到整体指标

`eval_config.yaml`：

```yaml
operators:
  - name: "forecasting_metrics"
    alias: "forecast_metrics"
    truth_columns: ["y_true"]
    predict_columns: ["y_pred"]
    config:
      epsilon: 1e-8
      max_dtw_len: 2000
```

### 单轮输出

- 总窗口数 `N_windows`
- `per_step_metrics.csv`：`step,mae,rmse,mape,dtw`
- `overall_metrics.json`
- 终端打印 trial 编号、参数、窗口数、per-step 表、overall 指标

## 四、BO 闭环优化

生成 `bo_sliding_window_loop.py`，流程如下：

### 1. 初始化 BO 任务

```bash
<bo-env-python> .claude/skills/bo-for-experiment/main.py --mode init \
  --non_interactive \
  --params_config '<bo_params_config>' \
  --objectives '<bo_objectives>' \
  --task_id <bo_task_id> \
  --data_dir <output_dir>/bo
```

### 2. Trial 0：默认参数

- 用 `<default_params>` 生成 `trial_0/forecasting_config.yaml`
- 训练 → 滑窗评估 → 从 `overall_metrics.json` 提取目标指标
- 记录 `(params, metric)`

### 3. Trial 1~9：BO 推荐

```bash
<bo-env-python> .claude/skills/bo-for-experiment/main.py --mode iterate \
  --task_id <bo_task_id> \
  --x_new '<cumulative_params_list>' \
  --y_new '<cumulative_metrics_list>' \
  --n_suggest 1 \
  --data_dir <output_dir>/bo \
  --format json
```

- 解析 stdout 最后一行 JSON，取 `suggestions[0]`
- 生成 `trial_i/forecasting_config.yaml` → 训练 → 评估 → 追加观测

### 4. 输出最优参数

10 轮后读取 `<output_dir>/bo/<bo_task_id>_history.json`，按目标方向选出最优 trial，保存：

```json
{
  "best_trial": 3,
  "best_params": {...},
  "best_metric": {"overall_mae": ...},
  "all_trials": [...]
}
```

并打印 trial 对比表。

## 五、约束

- **训练**必须通过 `tsas.engine.operator.cli forecasting fit` 调用算子完成
- **指标计算**必须通过 `tsas.engine.operator.cli evaluation run` 调用算子完成
- **批量推理**可通过算子 Python API 完成（加载已保存模型后调用 `forecaster.run(x_test)`），禁止自行实现模型前向传播逻辑
- BO 推荐必须通过 `bo-for-experiment/main.py` 完成
- 禁止自行编写 Python 代码替代模型训练或指标计算核心逻辑
- 脚本只负责数据切分、批量推理 orchestration、循环调用 CLI、读取结果、汇总

## 六、交付物

1. `forecasting_config.yaml`（模板）
2. `eval_config.yaml`
3. `evaluate_sliding_window.py`
4. `bo_sliding_window_loop.py`
5. `run_experiment.sh`（预处理 → BO 初始化 → 10 轮闭环 → 输出最优参数）
