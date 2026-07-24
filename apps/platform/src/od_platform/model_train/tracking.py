#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : tracking.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_train/tracking.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-23 15:00:00
# @Version    : v1.0.0
# @Description:训练实验追踪——MLflow 集成 + 训练生命周期钩子
# @ChangeLog:
#   2026-07-23:00:00 | Matri | v1.0.0 | 初始化创建
"""训练实验追踪: TrainTrackingHooks + MlflowTracker.

设计纪律:
  - TrainTrackingHooks 对标 InferHooks, 4 个 fire_* 全部 try/except 包裹
  - MlflowTracker 每个 mlflow 调用独立 try/except, 追踪失败不炸训练
  - mlflow 是可选依赖, 未安装时自动降级为 no-op
  - build_tracking_hooks_from_config 是唯一入口, 外部不需要知道 MlflowTracker 存在
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 事件数据 (frozen, 纯数据, 对标 FrameEvent / ProgressEvent)
# ============================================================

@dataclass(frozen=True)
class TrainStartEvent:
    """训练开始事件——在 model.train() 调用前触发."""
    run_id:           str
    experiment_name:  Optional[str]
    config_snapshot:  Dict[str, Any]
    data_yaml:        str
    model:            str


@dataclass(frozen=True)
class FitEpochEndEvent:
    """每轮结束事件——ultralytics on_fit_epoch_end 回调触发."""
    epoch:   int
    metrics: Dict[str, float]


@dataclass(frozen=True)
class TrainEndEvent:
    """训练结束事件——model.train() 成功返回后触发."""
    run_id:      str
    save_dir:    Path
    best_weight: Optional[Path]
    metrics:     Any                  # TrainMetrics


@dataclass(frozen=True)
class TrainErrorEvent:
    """训练异常事件——model.train() 抛异常时触发."""
    run_id:    str
    exception: Exception


# ============================================================
# 钩子容器 (对标 InferHooks)
# ============================================================

@dataclass
class TrainTrackingHooks:
    """训练实验追踪钩子.

    CLI 默认行为: TrainTrackingHooks() 全部 None → 触发时全 short-circuit, 零开销.
    """

    on_train_start:    Optional[Callable[[TrainStartEvent],    None]] = None
    on_fit_epoch_end:  Optional[Callable[[FitEpochEndEvent],  None]] = None
    on_train_end:      Optional[Callable[[TrainEndEvent],      None]] = None
    on_train_error:    Optional[Callable[[TrainErrorEvent],    None]] = None

    # ------------------------------------------------------------------
    # 触发 helper —— 全部 try/except 包住, 业务回调抛异常不能炸训练
    # ------------------------------------------------------------------

    def fire_train_start(self, evt: TrainStartEvent) -> None:
        if self.on_train_start is not None:
            try:
                self.on_train_start(evt)
            except Exception as e:
                logger.warning(f"on_train_start 回调异常 (已吞): {e}")

    def fire_fit_epoch_end(self, evt: FitEpochEndEvent) -> None:
        if self.on_fit_epoch_end is not None:
            try:
                self.on_fit_epoch_end(evt)
            except Exception as e:
                logger.warning(f"on_fit_epoch_end 回调异常 (已吞): {e}")

    def fire_train_end(self, evt: TrainEndEvent) -> None:
        if self.on_train_end is not None:
            try:
                self.on_train_end(evt)
            except Exception as e:
                logger.warning(f"on_train_end 回调异常 (已吞): {e}")

    def fire_train_error(self, evt: TrainErrorEvent) -> None:
        if self.on_train_error is not None:
            try:
                self.on_train_error(evt)
            except Exception as e:
                logger.warning(f"on_train_error 回调异常 (已吞): {e}")


# ============================================================
# MLflow 追踪器
# ============================================================

class MlflowTracker:
    """MLflow 实验追踪器——轻量封装, 不改变训练流程.

    设计纪律:
      - 每个 mlflow 调用独立 try/except + logger.debug, 追踪失败不炸训练
      - mlflow 未安装 → _mlflow_available()=False → 静默 no-op
      - 外部不要直接实例化, 走 build_tracking_hooks_from_config
    """

    def __init__(self, tracking_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None):
        self._tracking_uri = tracking_uri or "sqlite:///mlflow.db"
        self._experiment_name = experiment_name or "od-platform"
        self._run: Any = None
        self._available = self._mlflow_available()

    # ------------------------------------------------------------------
    # 核心操作
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, tags: Optional[Dict[str, str]] = None) -> None:
        if not self._available:
            return
        try:
            import mlflow
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            self._run = mlflow.start_run(run_name=run_id, tags=tags)
            logger.info(f"MLflow run 已启动: {run_id} (实验: {self._experiment_name})")
        except Exception as e:
            logger.debug(f"MLflow start_run 失败 (不影响训练): {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self._available:
            return
        try:
            import mlflow
            # mlflow 限制单次最多 100 个参数, 分批写
            safe = {str(k): str(v) for k, v in params.items() if v is not None}
            batch: Dict[str, str] = {}
            for k, v in safe.items():
                batch[k] = v
                if len(batch) >= 100:
                    mlflow.log_params(batch)
                    batch.clear()
            if batch:
                mlflow.log_params(batch)
        except Exception as e:
            logger.debug(f"MLflow log_params 失败 (不影响训练): {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if not self._available:
            return
        try:
            import mlflow
            safe = {self._sanitize_key(k): v for k, v in metrics.items()}
            mlflow.log_metrics(safe, step=step)
        except Exception as e:
            logger.debug(f"MLflow log_metrics 失败 (不影响训练): {e}")

    def log_artifacts(self, paths: List[Path]) -> None:
        if not self._available:
            return
        for path in paths:
            try:
                if path and Path(path).exists():
                    import mlflow
                    mlflow.log_artifact(str(path))
            except Exception as e:
                logger.debug(f"MLflow log_artifact 失败 ({path}): {e}")

    def end_run(self) -> None:
        if not self._available:
            return
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            logger.debug(f"MLflow end_run 失败 (不影响训练): {e}")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """移除 mlflow 不允许的字符 (括号等), 保留 / _ - . 空格和字母数字."""
        return key.replace("(", "").replace(")", "")

    @staticmethod
    def _mlflow_available() -> bool:
        try:
            import mlflow  # noqa: F401
            return True
        except ImportError:
            return False


# ============================================================
# 工厂 + 桥接
# ============================================================

def build_tracking_hooks_from_config(
    config: Any, run: Any,
) -> TrainTrackingHooks:
    """从配置构造带 MLflow 回调的 TrainTrackingHooks.

    若 mlflow 未安装或 mlflow_enabled=False → 返回全空 hooks (零开销).
    """
    mlflow_enabled = getattr(config, "mlflow_enabled", False)
    if not mlflow_enabled:
        return TrainTrackingHooks()

    if not MlflowTracker._mlflow_available():
        logger.info("MLflow 未安装, 实验追踪已禁用. 安装: pip install mlflow")
        return TrainTrackingHooks()

    tracking_uri = getattr(config, "mlflow_tracking_uri", None) or None
    experiment_name = (
        getattr(config, "mlflow_experiment_name", None)
        or getattr(config, "experiment_name", None)
    )
    tracker = MlflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    def _on_train_start(evt: TrainStartEvent) -> None:
        tracker.start_run(
            evt.run_id,
            tags={"dataset": Path(evt.data_yaml).stem if evt.data_yaml else "unknown",
                  "model": evt.model},
        )
        tracker.log_params(evt.config_snapshot.get("values", {}))

    def _on_fit_epoch_end(evt: FitEpochEndEvent) -> None:
        # 只保留含 "/" 的 ultralytics 标准指标键 (过滤 epoch 等元数据)
        clean = {k: v for k, v in evt.metrics.items() if "/" in k}
        if clean:
            tracker.log_metrics(clean, step=evt.epoch)

    def _on_train_end(evt: TrainEndEvent) -> None:
        if evt.metrics is not None:
            clean = {k: v for k, v in evt.metrics.overall.items()
                     if isinstance(v, (int, float))}
            tracker.log_metrics(clean, step=0)
        artifacts: List[Path] = []
        if evt.best_weight and evt.best_weight.exists():
            artifacts.append(evt.best_weight)
        audit_path = run.artifact_path("odp_audit.json")
        if audit_path.exists():
            artifacts.append(audit_path)
        tracker.log_artifacts(artifacts)
        tracker.end_run()

    def _on_train_error(evt: TrainErrorEvent) -> None:
        tracker.end_run()

    return TrainTrackingHooks(
        on_train_start=_on_train_start,
        on_fit_epoch_end=_on_fit_epoch_end,
        on_train_end=_on_train_end,
        on_train_error=_on_train_error,
    )


def _make_ultralytics_callback(hooks: TrainTrackingHooks) -> Dict[str, Callable]:
    """把 TrainTrackingHooks 翻译为 ultralytics model.add_callback() 接受的字典.

    返回空 dict 时 ultralytics 不注册任何回调, 零开销.
    """
    callbacks: Dict[str, Callable] = {}

    if hooks.on_fit_epoch_end is not None:
        def _on_fit_epoch_end(trainer: Any) -> None:
            epoch = getattr(trainer, "epoch", -1) + 1
            raw = dict(getattr(trainer, "metrics", {}) or {})
            metrics: Dict[str, float] = {}
            for k, v in raw.items():
                try:
                    metrics[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
            hooks.fire_fit_epoch_end(FitEpochEndEvent(epoch=epoch, metrics=metrics))
        callbacks["on_fit_epoch_end"] = _on_fit_epoch_end

    return callbacks
