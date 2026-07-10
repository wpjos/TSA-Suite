# -*- coding: utf-8 -*-

"""
XGBoost 时序预测算子

支持两种多步预测策略：

- ``strategy=None``（默认，Direct）：为每个预测步长（以及每个目标维度）
  独立训练一个 XGBoost booster。
- ``strategy='MIMO'``：每个目标维度训练一个 XGBoost booster，以
  “窗口特征 + 步长索引”为输入，同时输出所有未来步长的预测值。

输入输出约定遵循 ``BaseForecaster``：

    fit(x, y):
        x: (timesteps, num_features)  DataFrame / ndarray
        y: (timesteps, num_targets)   DataFrame / ndarray

    run(x):
        x: (seq_len, num_features) 或 (batch, seq_len, num_features)
        返回: (pred_len, num_targets) 或 (batch, pred_len, num_targets)

特征工程采用通用窗口特征：将每个历史窗口 ``x[t-seq_len+1:t+1]``
展平为特征向量，不依赖 EPF 领域特征，从而适配任意数值时序数据。
"""

from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "无法加载 xgboost。请安装 xgboost 并确保 OpenMP 运行时可用 "
        "（macOS 用户可运行 `brew install libomp`）。"
    ) from exc

from tsas.engine.operator.forecasting.base import BaseForecaster, ForecastExtraOutput

__all__ = [
    'XGBoostForecasterConfig',
    'XGBoostForecaster',
    'XGBoostMIMOForecasterConfig',
    'XGBoostMIMOForecaster',
]


class XGBoostForecasterConfig(BaseModel):
    """XGBoost 预测算子实例参数。

    所有带 ``ge`` / ``le`` 约束的数值字段均可被 HPO 自动搜索。
    """

    # ---- 窗口 / 问题结构 ----
    seq_len: int = Field(default=96, ge=1, le=4096, description="输入历史窗口长度")
    pred_len: int = Field(default=24, ge=1, le=500, description="预测未来步长")

    # ---- 多步预测策略 ----
    strategy: Literal['MIMO'] | None = Field(
        default=None,
        description="多步预测策略，None 表示 Direct（默认），'MIMO' 表示 MIMO",
    )

    # ---- XGBoost booster 超参数 ----
    max_depth: int = Field(default=4, ge=1, le=16, description="树最大深度")
    learning_rate: float = Field(default=0.05, ge=1e-6, le=1.0, description="学习率")
    n_estimators: int = Field(default=200, ge=1, le=10000, description="提升轮数")
    min_child_weight: float = Field(default=1.0, ge=0.0, le=100.0, description="子节点最小权重和")
    reg_alpha: float = Field(default=0.1, ge=0.0, le=10.0, description="L1 正则化")
    reg_lambda: float = Field(default=0.1, ge=0.0, le=10.0, description="L2 正则化")

    # ---- 训练 / 调参 ----
    skip_tune: bool = Field(default=True, description="是否跳过内部网格调参")
    train_ratio: float = Field(default=0.8, ge=0.1, le=0.95, description="训练集占比")
    val_ratio: float = Field(default=0.1, ge=0.0, le=0.45, description="验证集占剩余数据比例")

    # ---- 运行参数 ----
    device: Literal['cpu', 'gpu'] = Field(default='cpu', description="计算设备")
    n_jobs: int = Field(default=-1, ge=-1, le=64, description="XGBoost 线程数，-1 表示全部")

    class Config:
        extra = 'forbid'


class XGBoostForecaster(BaseForecaster[ForecastExtraOutput,
                                       XGBoostForecasterConfig,
                                       None,
                                       None]):
    """XGBoost 时序预测算子。

    通过 ``config.strategy`` 选择多步预测策略：

    - ``None``（默认）：Direct 策略，为每个预测步长（以及每个目标维度）训练独立的
      XGBoost booster。
    - ``'MIMO'``：MIMO 策略，每个目标维度使用一个 XGBoost booster，以
      “窗口特征 + 步长索引”为输入，同时预测所有未来步长。

    特征由输入历史窗口展平得到，适用于通用数值时序。

    输入输出约定::

        fit(x, y):
            x: (timesteps, num_features)  DataFrame / ndarray
            y: (timesteps, num_targets)   DataFrame / ndarray

        run(x):
            x: (seq_len, num_features) 或 (batch, seq_len, num_features)
            返回: (pred_len, num_targets) 或 (batch, pred_len, num_targets)
    """

    _MODEL_DIR = '_models'
    _STATE_FILE = '_forecaster_state.npz'

    @classmethod
    def name(cls) -> str:
        return "xgboost_forecaster"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号。

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: XGBoostForecasterConfig | None = None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._models: dict = {}
        self._num_features: int | None = None
        self._num_targets: int | None = None

    def set_chunk_ids(self, chunk_ids: np.ndarray | None) -> None:
        """XGBoost 算子当前不支持 chunk_ids，传入时忽略并警告。"""
        if chunk_ids is not None:
            logger.warning(
                f"[{self.oid}] xgboost_forecaster 不支持 chunk_ids，"
                "传入的 chunk_ids 将被忽略。"
            )

    def _is_mimo(self) -> bool:
        """当前是否使用 MIMO 策略。"""
        return self.config.strategy == 'MIMO'

    def _base_params(self) -> dict:
        """构建 XGBoost 基础参数字典。"""
        cfg = self.config
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "min_child_weight": cfg.min_child_weight,
            "reg_alpha": cfg.reg_alpha,
            "reg_lambda": cfg.reg_lambda,
            "verbosity": 0,
            "seed": 42,
            "nthread": cfg.n_jobs,
        }
        if cfg.device != 'cpu':
            params["device"] = cfg.device
        return params

    def _build_samples(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """从完整时间序列构造监督学习样本。

        Direct 策略：
            - X_samples: ``(n_samples, seq_len * num_features)``
            - Y_samples: ``(n_samples, pred_len * num_targets)``

        MIMO 策略：
            - X_samples: ``(n_samples * pred_len, seq_len * num_features + 1)``
            - Y_samples: ``(n_samples * pred_len, num_targets)``

        Args:
            x: 形状 ``(timesteps, num_features)``
            y: 形状 ``(timesteps, num_targets)``

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(X_samples, Y_samples)``
        """
        cfg = self.config
        n_total = len(x)
        n_windows = n_total - cfg.seq_len - cfg.pred_len + 1
        if n_windows <= 0:
            raise ValueError(
                f"时间序列长度 {n_total} 不足以构造窗口 "
                f"(seq_len={cfg.seq_len}, pred_len={cfg.pred_len})"
            )

        num_features = x.shape[1]
        num_targets = y.shape[1]

        if self._is_mimo():
            total_samples = n_windows * cfg.pred_len
            X_samples = np.empty((total_samples, cfg.seq_len * num_features + 1), dtype=x.dtype)
            Y_samples = np.empty((total_samples, num_targets), dtype=y.dtype)

            sample_idx = 0
            for i in range(n_windows):
                start_x = i
                end_x = start_x + cfg.seq_len
                window_features = x[start_x:end_x].ravel()
                for k in range(cfg.pred_len):
                    target_idx = end_x + k
                    X_samples[sample_idx, :-1] = window_features
                    X_samples[sample_idx, -1] = k
                    Y_samples[sample_idx] = y[target_idx]
                    sample_idx += 1
        else:
            X_samples = np.empty((n_windows, cfg.seq_len * num_features), dtype=x.dtype)
            Y_samples = np.empty((n_windows, cfg.pred_len * num_targets), dtype=y.dtype)

            for i in range(n_windows):
                start_x = i
                end_x = start_x + cfg.seq_len
                end_y = end_x + cfg.pred_len
                X_samples[i] = x[start_x:end_x].ravel()
                Y_samples[i] = y[end_x:end_y].ravel()

        return X_samples, Y_samples

    def _extract_window_features(self, x: np.ndarray) -> np.ndarray:
        """将单个或多个窗口展平为特征向量。

        Args:
            x: 形状 ``(seq_len, num_features)`` 或 ``(batch, seq_len, num_features)``

        Returns:
            np.ndarray: 形状 ``(seq_len * num_features,)`` 或
            ``(batch, seq_len * num_features)``
        """
        if x.ndim == 2:
            return x.ravel()
        if x.ndim == 3:
            return x.reshape(x.shape[0], -1)
        raise ValueError(f"窗口维度必须是 2-D 或 3-D，当前为 {x.ndim}")

    def _build_predict_features(self, x: np.ndarray) -> np.ndarray:
        """MIMO 策略下为推理构造包含步长索引的特征矩阵。

        Args:
            x: 形状 ``(batch, seq_len, num_features)``

        Returns:
            np.ndarray: 形状 ``(batch * pred_len, seq_len * num_features + 1)``
        """
        cfg = self.config
        batch_size = x.shape[0]
        window_features = self._extract_window_features(x)  # (batch, seq_len * num_features)

        # 为每个 batch 项构造 pred_len 行，并在末尾拼接步长索引
        features = np.repeat(window_features, cfg.pred_len, axis=0)
        step_indices = np.tile(np.arange(cfg.pred_len, dtype=x.dtype), batch_size).reshape(-1, 1)
        features = np.concatenate([features, step_indices], axis=1)

        return features

    def _split_train_val(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """按时间顺序划分训练集和验证集。"""
        cfg = self.config
        n_total = len(X)
        n_train = int(n_total * cfg.train_ratio)

        if n_train <= 0 or n_train >= n_total:
            raise ValueError(
                f"训练集划分结果无效: n_total={n_total}, n_train={n_train}, "
                f"train_ratio={cfg.train_ratio}"
            )

        X_train, Y_train = X[:n_train], Y[:n_train]

        if cfg.val_ratio <= 0:
            return X_train, Y_train, None, None

        n_val = int((n_total - n_train) * cfg.val_ratio)
        if n_val <= 0:
            return X_train, Y_train, None, None

        X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
        return X_train, Y_train, X_val, Y_val

    def _fit_data(self, x: np.ndarray, y: np.ndarray, *, params: None) -> None:
        cfg = self.config
        self._num_features = x.shape[1]
        self._num_targets = y.shape[1]

        X_samples, Y_samples = self._build_samples(x, y)
        X_train, Y_train, X_val, Y_val = self._split_train_val(X_samples, Y_samples)

        base_params = self._base_params()

        if self._is_mimo():
            for j in range(self._num_targets):
                y_train_j = Y_train[:, j]
                dtrain = xgb.DMatrix(X_train, y_train_j)

                evals = [(dtrain, "train")]
                if X_val is not None and Y_val is not None:
                    y_val_j = Y_val[:, j]
                    dval = xgb.DMatrix(X_val, y_val_j)
                    evals.append((dval, "val"))

                logger.debug(
                    f"[{self.oid}] 训练 XGBoost MIMO booster: target={j}, "
                    f"n_train={len(X_train)}"
                )
                self._models[j] = xgb.train(
                    base_params,
                    dtrain,
                    num_boost_round=cfg.n_estimators,
                    evals=evals,
                    verbose_eval=False,
                )
        else:
            for k in range(cfg.pred_len):
                for j in range(self._num_targets):
                    target_col = k * self._num_targets + j
                    y_train_kj = Y_train[:, target_col]
                    dtrain = xgb.DMatrix(X_train, y_train_kj)

                    evals = [(dtrain, "train")]
                    if X_val is not None and Y_val is not None:
                        y_val_kj = Y_val[:, target_col]
                        dval = xgb.DMatrix(X_val, y_val_kj)
                        evals.append((dval, "val"))

                    logger.debug(
                        f"[{self.oid}] 训练 XGBoost booster: horizon={k}, target={j}, "
                        f"n_train={len(X_train)}"
                    )
                    self._models[(k, j)] = xgb.train(
                        base_params,
                        dtrain,
                        num_boost_round=cfg.n_estimators,
                        evals=evals,
                        verbose_eval=False,
                    )

    def _run_data(self, x: np.ndarray, *, params: None) -> np.ndarray:
        if not self._models:
            raise RuntimeError("模型尚未训练，无法执行推理")

        cfg = self.config
        if self._num_features is None or self._num_targets is None:
            raise RuntimeError("算子内部状态缺失，请先训练或加载模型")

        batched = x.ndim == 3
        if not batched:
            if x.shape != (cfg.seq_len, self._num_features):
                raise ValueError(
                    f"推理输入形状应为 (seq_len={cfg.seq_len}, num_features={self._num_features}), "
                    f"但当前为 {x.shape}"
                )
            x = x[np.newaxis, ...]  # (1, seq_len, num_features)
        else:
            if x.shape[1:] != (cfg.seq_len, self._num_features):
                raise ValueError(
                    f"批量推理输入形状应为 (batch, seq_len={cfg.seq_len}, num_features={self._num_features}), "
                    f"但当前为 {x.shape}"
                )

        batch_size = x.shape[0]

        if self._is_mimo():
            X_features = self._build_predict_features(x)
            dtest = xgb.DMatrix(X_features)

            preds = np.empty((batch_size, cfg.pred_len, self._num_targets), dtype=np.float64)
            for j in range(self._num_targets):
                model = self._models[j]
                pred_j = model.predict(dtest)
                pred_j = pred_j.reshape(batch_size, cfg.pred_len)
                preds[:, :, j] = pred_j
        else:
            X_features = self._extract_window_features(x)  # (batch, seq_len * num_features)
            dtest = xgb.DMatrix(X_features)

            preds = np.empty((batch_size, cfg.pred_len, self._num_targets), dtype=np.float64)
            for k in range(cfg.pred_len):
                for j in range(self._num_targets):
                    model = self._models[(k, j)]
                    preds[:, k, j] = model.predict(dtest)

        return preds if batched else preds[0]

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        model_dir = path / self._MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        if self._is_mimo():
            for j, model in self._models.items():
                model_path = model_dir / f"model_t{j}.json"
                model.save_model(str(model_path))
        else:
            for (k, j), model in self._models.items():
                model_path = model_dir / f"model_h{k}_t{j}.json"
                model.save_model(str(model_path))

        np.savez(
            path / self._STATE_FILE,
            num_features=self._num_features,
            num_targets=self._num_targets,
            pred_len=self.config.pred_len,
            seq_len=self.config.seq_len,
        )

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)

        state = np.load(path / self._STATE_FILE)
        self._num_features = int(state["num_features"])
        self._num_targets = int(state["num_targets"])
        pred_len = int(state["pred_len"])

        model_dir = path / self._MODEL_DIR
        self._models = {}
        if self._is_mimo():
            for j in range(self._num_targets):
                model_path = model_dir / f"model_t{j}.json"
                model = xgb.Booster()
                model.load_model(str(model_path))
                self._models[j] = model
        else:
            for k in range(pred_len):
                for j in range(self._num_targets):
                    model_path = model_dir / f"model_h{k}_t{j}.json"
                    model = xgb.Booster()
                    model.load_model(str(model_path))
                    self._models[(k, j)] = model

        self._fitted = True


class XGBoostMIMOForecasterConfig(XGBoostForecasterConfig):
    """XGBoost MIMO 预测算子实例参数（兼容别名）。

    等价于 ``XGBoostForecasterConfig(strategy='MIMO')``，并保留 MIMO 默认超参。
    """

    strategy: Literal['MIMO'] = Field(
        default='MIMO',
        description="MIMO 多步预测策略",
    )
    max_depth: int = Field(default=6, ge=1, le=16, description="树最大深度")

    class Config:
        extra = 'forbid'


class XGBoostMIMOForecaster(XGBoostForecaster):
    """XGBoost MIMO 时序预测算子（兼容别名）。

    等价于 ``XGBoostForecaster(config=XGBoostForecasterConfig(strategy='MIMO'))``。
    """

    _is_operator_alias: ClassVar[bool] = True
    _config_type: ClassVar[type[XGBoostMIMOForecasterConfig] | None] = XGBoostMIMOForecasterConfig

    @classmethod
    def name(cls) -> str:
        return "xgboost_mimo_forecaster"
