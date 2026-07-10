---
name: tsas-num-forecasting-model
description: >
  针对数值类时序数据的预测训练实验技能。使用 TSAS-CLI 工具中的时序预测算子
  （itransformer_forecaster、lightgbm_forecaster、xgboost_forecaster）
  对给定数据进行训练、推理和时序预测评价指标计算的完整实验流程。
  当用户提及"时序预测"、"时间序列预测"、"forecasting"、"数值预测"、"预测模型"等关键词时触发此技能。
version: 1.0.0
---

# 数值时序预测训练和验证实验

## 概述

本技能封装了使用 TSAS-CLI 工具进行时序预测（Time Series Forecasting）实验的完整工作流。核心流程包括：

1. 确定预测模型方案（算子选择与参数配置）
2. 参数校验
3. 使用训练数据执行训练（可训练算子）
4. 使用推理窗口执行预测
5. 计算时序预测评价指标
6. 收集关键结果和所有产出文件

---

## 强制遵守的约定

### Python 环境

所有需要执行 TSAS-CLI 命令以及 Python 命令或脚本时，**必须**使用已安装 TSA-Suite 的 Python 环境。由于不同机器的环境路径不同，本技能使用占位符表示：

```bash
<tsa-suite-env-python>
```

在单次执行命令时使用完整路径，例如：

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting help
```

在已知环境路径的交互式命令行中，也可以先激活该环境，例如：

```bash
source <tsa-suite-env>/bin/activate
# Windows 下:
# <tsa-suite-env>\Scripts\activate
```

> 执行任何命令前，先验证环境可用：
> ```bash
> <tsa-suite-env-python> -c "import tsas; print(tsas.__file__)"
> ```

### 禁止自行实现计算逻辑

**实验的所有核心流程必须且只能通过调用 TSAS-CLI 算子完成，绝对禁止自行编写 Python 或其他代码来替代。**

具体来说，以下操作**必须**通过 `python -m tsas.engine.operator.cli` 调用对应算子执行：

| 操作 | 必须使用的 CLI 模块 | 禁止行为 |
|------|------------------|---------|
| 模型训练 | `forecasting fit` | 禁止自行编写训练代码 |
| 时序预测/推理 | `forecasting run` | 禁止自行编写预测代码 |
| 评价指标计算 | `evaluation run` | 禁止自行编写 MSE/MAE/RMSE/MAPE/DTW/R² 等指标计算代码 |

**唯一的例外**：生成 YAML/JSON 配置文件、读取和展示 CLI 输出结果、生成实验总结报告等辅助性操作，可以正常使用文件读写工具。

如果任何原因导致 CLI 算子调用失败，应向用户报告错误并停止，**绝不允许**降级为自行实现代码来绕过。

### 输出目录

如果用户指定了输出的根目录，则应严格遵循；否则，应以当前工作目录（CWD）作为输出的根目录。

实验产生的所有文件统一存放于输出根目录的子文件夹中，命名格式：

```
num_fc_trainning_<毫秒级时间戳>
```

时间戳取实验开始执行时的精确时间，格式为 `YYYYMMDDHHmmssfff`（年月日时分秒毫秒）。

**输出目录结构示意**：

```
num_fc_trainning_<时间戳>/
├── forecasting_config.yaml      # 预测算子配置文件
├── model/                       # 训练保存的模型（如有训练步骤）
├── forecasting_result.csv       # 推理/预测结果数据
├── eval_config.yaml             # 评价指标配置文件
├── eval_result.json             # 评价指标计算结果
└── experiment_summary.md        # 实验总结报告
```

将该目录作为输出目录，并确保本技能生成的所有文件都保存在该目录下。

---

## 工作流程

### 第 1 步：确定预测模型方案

#### 可用预测算子

TSAS 时序预测模块（`tsas.engine.operator.forecasting`）已注册以下算子：

| 算子名称 | 类型 | 说明 |
|---------|------|------|
| `itransformer_forecaster` | 深度学习（iTransformer + KAN 头） | 适合复杂非线性时序关系，单目标预测；需要 `torch` |
| `lightgbm_forecaster` | 树模型（LightGBM） | Direct 多步策略，快速基线；需要 `lightgbm` |
| `lightgbm_mimo_forecaster` | LightGBM MIMO 变体 | `lightgbm_forecaster` 的 `strategy='MIMO'` 别名 |
| `xgboost_forecaster` | 树模型（XGBoost） | Direct 多步策略；需要 `xgboost` |
| `xgboost_mimo_forecaster` | XGBoost MIMO 变体 | `xgboost_forecaster` 的 `strategy='MIMO'` 别名 |

> 注：`help` 列表可能不显示 `*_mimo_forecaster` 别名，但配置文件中可以直接使用。

#### 算子选择规则

1. 对于**用户明确指定使用的算子**：
   - 必须使用用户指定的算子，**不允许自行移除或替换**。
   - 如果指定的算子名称不存在于注册中心，则直接报错。

2. 对于**用户未指定**或**用户未完全指定**的情况：
   - 若用户只说了模型家族（如 "LightGBM"、"XGBoost"、"iTransformer"），使用对应默认算子：
     - LightGBM → `lightgbm_forecaster`
     - XGBoost → `xgboost_forecaster`
     - iTransformer → `itransformer_forecaster`
   - 若用户完全没有指定，推荐先用 `lightgbm_forecaster` 作为快速基线；如果用户追求更高建模容量且数据量充足，再推荐 `itransformer_forecaster`。

### 第 2 步：获取算子参数信息

执行 CLI help 命令获取算子的完整参数定义：

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting help <算子名称>
```

解析输出中的参数表，提取每个参数的：

- 参数名与类型
- 默认值
- 值域/候选（如 `[100, 300]`、`enum(auto, cpu, cuda, npu)` 等）
- 是否必填
- 说明

### 第 3 步：参数校验与配置

根据用户指定的参数和 help 中的定义进行参数校验：

**校验规则**：

1. 用户指定的参数名必须存在于 help 文档的参数表中，否则报错。
2. 参数值必须符合类型要求。
3. 参数值必须在值域/候选范围内（如 `[100, 300]`、`enum(auto, cpu, cuda, npu)`、`(0.0, 1.0)` 等）。
4. 若校验失败，**立即报错并返回详细错误信息**，包括：参数名、用户设定值、合法值域/类型。

**参数填充规则**：

- 用户明确指定且校验通过 → 使用用户指定值
- 用户未指定 → 使用 help 文档中的默认值
- 必填参数无默认值且用户未指定 → 报错提示用户

**生成预测配置文件** `forecasting_config.yaml`：

```yaml
operator:
  name: "<算子名称>"
  input_columns:
    - "feat_0"
    - "feat_1"
    - "target"
  target_column: "target"
  config:
    <参数1>: <值1>
    <参数2>: <值2>
    ...
```

其中：
- `input_columns` 根据用户指定或数据特征列自动确定，作为 `fit`/`run` 的输入 `x`。
- `target_column` 指定训练时的目标列，作为 `fit` 的输入 `y`。
- `operator.config` 中放置算子实例参数。

> 对于 `itransformer_forecaster`，`target_idx` 默认 `-1`，表示目标列为 `input_columns` 的最后一列。因此通常需要把目标列放在 `input_columns` 末尾。

### 第 4 步：确定输入数据与实验模式

分析用户提供的 CSV 数据：

1. **读取数据头部**（最多只能读取前 5 行）了解列结构和数据格式。

2. **判断数据集划分情况**：
   - 若输入数据明确划分了训练文件和推理/验证文件 → 视为已划分
   - 否则 → 视为未划分

3. **确定实验模式**：

当前所有预测算子均为可训练算子（继承 `LearnableOperatorMixin`），因此：

| 是否提供训练数据 | 是否提供推理窗口 | 实验模式 |
|----------------|----------------|---------|
| ✅ | ✅ | 先 `fit` 训练，再 `run --load` 推理 |
| ✅ | ❌ | 仅执行 `fit` 训练并保存模型 |
| ❌ | ✅ | 直接 `run` 会失败，必须提供已训练模型目录 `--load`；否则报错提示用户先训练 |
| ❌ | ❌ | 报错，缺少输入数据 |

> 预测算子的 `run` 需要输入一个形状为 `(seq_len, num_features)` 的历史窗口；`fit` 需要输入完整时间序列 `(timesteps, num_features)`。两者数据文件可以不同。

### 第 5 步：创建输出目录

创建输出目录：

```bash
mkdir -p "<输出根目录>/num_fc_trainning_<时间戳>"
```

时间戳格式为 `YYYYMMDDHHmmssfff`（毫秒级）。

### 第 6 步：执行训练（如需要）

当实验模式要求训练时，**通过 TSAS-CLI 的 `forecasting fit` 命令执行训练**：

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting fit \
  --input <训练数据.csv> \
  --target <目标列名> \
  --config <输出目录>/forecasting_config.yaml \
  --save <输出目录>/model
```

如果前置预处理技能（如 `ts-preprocessing-for-tsas-cli`）提供了 `chunk_ids.csv`，可以追加 `--chunk-ids` 参数，避免 `itransformer_forecaster` 的训练窗口跨越时间断层：

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting fit \
  --input <训练数据.csv> \
  --target <目标列名> \
  --config <输出目录>/forecasting_config.yaml \
  --save <输出目录>/model \
  --chunk-ids <chunk_ids.csv>
```

> `chunk_ids.csv` 为单列表、无表头，每行一个整数，行数必须与 `<训练数据.csv>` 完全一致。树模型（LightGBM / XGBoost）会忽略该参数并打印警告。

训练成功后，`<输出目录>/model/` 目录将包含保存的模型参数。

> ⚠️ **禁止**自行编写 Python 代码来实现模型训练逻辑。必须且只能通过上述 CLI 命令调用。

### 第 7 步：执行推理

**通过 TSAS-CLI 的 `forecasting run` 命令执行推理**。根据是否进行了训练，命令不同：

**有训练步骤**：

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting run \
  --input <推理窗口.csv> \
  --config <输出目录>/forecasting_config.yaml \
  --load <输出目录>/model \
  --output <输出目录>/forecasting_result.csv
```

**无训练步骤（已提供预训练模型）**：

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting run \
  --input <推理窗口.csv> \
  --config <输出目录>/forecasting_config.yaml \
  --load <预训练模型目录> \
  --output <输出目录>/forecasting_result.csv
```

> ⚠️ **禁止**自行编写 Python 代码来实现时序预测逻辑。必须且只能通过上述 CLI 命令调用。

#### 输出列的保留策略

`forecasting run` 默认只输出预测结果列（如 `forecast_0`、`forecast_1` 等），不保留原始输入列。

- 若需保留原始列，使用 `--keep-input` 标志
- 若同时保留原始列可能引发**列名冲突**，需追加 `--auto-suffix` 自动加后缀

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli forecasting run \
  --input <推理窗口.csv> \
  --config <输出目录>/forecasting_config.yaml \
  --load <输出目录>/model \
  --keep-input --auto-suffix \
  --output <输出目录>/forecasting_result.csv
```

### 第 8 步：配置并执行评价指标

#### 8.1 确定评价策略

时序预测评价统一使用 `forecasting_metrics` 算子，计算 MSE、RMSE、MAE、MAPE、SMAPE、MASE、DTW、R² 八项指标。

评价输入必须是一个包含真实值和预测值列的 CSV 文件。该文件需要**预先对齐**：真实值与预测值的元素数量必须相同。

#### 8.2 生成评价配置文件

`eval_config.yaml` 示例：

```yaml
operators:
  - name: "forecasting_metrics"
    alias: "forecast_metrics"
    truth_columns: [ "y_true" ]
    predict_columns: [ "y_pred" ]
    config:
      naive_error: 0.01
      epsilon: 1e-8
      max_dtw_len: 2000
```

其中：
- `truth_columns`：真实值列名
- `predict_columns`：预测值列名
- `naive_error`：训练集随机游走基线误差，用于计算 MASE；未提供时 MASE 返回 `nan`
- `epsilon`：零值保护常数
- `max_dtw_len`：DTW 最大采样长度

#### 8.3 通过 TSAS-CLI 执行评价

**评价指标计算必须通过 TSAS-CLI 的 `evaluation run` 命令执行**：

```bash
<tsa-suite-env-python> -m tsas.engine.operator.cli evaluation run \
  --input <对齐后的评价数据.csv> \
  --config <输出目录>/eval_config.yaml \
  --output <输出目录>/eval_result.json
```

> **绝对禁止**自行编写 Python 代码来计算 MSE、RMSE、MAE、MAPE、SMAPE、MASE、DTW、R² 或任何其他评价指标。必须且只能通过上述 CLI 命令调用 `evaluation` 模块的算子来完成。如果 CLI 调用失败，应报告错误并停止，不得降级为自行实现。

### 第 9 步：生成实验总结

读取 `eval_result.json` 中的评价结果，结合实验配置，在输出目录中生成 `experiment_summary.md`，内容包括：

1. **实验信息**：时间戳、算子名称、实验模式（训练+推理 / 仅训练 / 仅推理）
2. **数据信息**：输入文件路径、数据行数/列数、特征列、目标列
3. **模型参数**：使用的完整参数配置
4. **评价结果**：关键指标摘要
5. **产出文件清单**：输出目录中所有文件的路径和说明

---

## 错误处理

### 参数校验失败

当用户指定了不合法的参数时，直接返回错误，格式如下：

```
❌ 参数校验失败：
- 参数 "<参数名>": 值 <用户值> 不合法。
  合法范围：<值域描述>
  说明：<参数说明>
```

多个错误同时列出。遇到校验错误时**不继续执行**实验。

### CLI 执行错误

若 CLI 命令执行返回非零退出码：

1. 捕获 stderr 输出
2. 向用户报告完整的错误信息
3. 不继续后续步骤
4. **禁止**在 CLI 失败后自行编写代码替代执行

### 未训练模型直接推理

若用户对可训练算子直接调用 `run` 而未提供 `--load`：

- CLI 会提示需要先训练
- 应向用户报告该提示，并停止执行
- **禁止**绕过训练自行生成预测结果

---

## 注意事项

1. 所有配置文件优先使用 YAML 格式（`.yaml`）。
2. `input_columns` 必须根据数据实际列名配置，不能凭猜测。
3. `target_column` 必须存在于输入数据中。
4. 对于 `itransformer_forecaster`，建议将目标列放在 `input_columns` 最后一位（与默认 `target_idx=-1` 保持一致）。
5. 推理前确认数据文件路径正确且可读；推理窗口的行数应等于算子配置的 `seq_len`。
6. 评价算子的 `truth_columns` 和 `predict_columns` 必须与评价数据中的实际列名完全匹配。
7. `fastdtw` 为可选依赖；未安装时 DTW 指标会自动回退为 MAE。
8. 若前置预处理技能 `ts-preprocessing-for-tsas-cli` 生成了 `chunk_ids.csv`，训练时建议通过 `--chunk-ids` 传入，使 `itransformer_forecaster` 避免窗口跨越时间断层。
9. 详细 TSAS-CLI API 参考请参阅 [references/api_reference.md](references/api_reference.md)。
