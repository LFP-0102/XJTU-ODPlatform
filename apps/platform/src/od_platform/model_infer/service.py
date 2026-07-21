#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : InferService — 编排 D5 配置 + 帧源捕获 + ultralytics 推理 + 美化绘制
"""推理服务编排器.

★ 核心纪律 (跟 D6 TrainService 完全同构): 不重新发明 D5 / 帧源 / 美化已有的轮子.

★ 接缝 (向后 100% 兼容):
  - predict() 3 个新参数: output_sink / hooks / cancel_token, 全部 keyword-only Optional[None]
  - 不传 = CLI 默认行为 (LocalFileSink / 空 hooks / 无 cancel)
  - 传了 = 桌面 / Web / Celery 业务方能完全定制输出 + 事件 + 取消

★ 跟训练的两点关键差异:
  1. 推理不调 model.predict(source=...) —— 而是 frame_source 逐帧喂 + 自己画
  2. 逐帧 model() 只传 predict 计算参数白名单, 不盲传 to_ultralytics_kwargs()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO


from od_platform.common import refs
from od_platform.common.paths import TRAINED_MODELS_DIR, PRETRAINED_MODELS_DIR, RUNS_DIR
from od_platform.common.naming import run_stem, model_stem
from od_platform.common.run_context import RunContext
from od_platform.common.logging_utils import ROOT_LOGGER_NAME

from od_platform.runtime_config import build_infer_config

from od_platform.frame_source import (
    create_frame_source, SourceType, IMAGE_EXTENSIONS, SourceConfig,
)
from od_platform.visualization import BeautifyVisualizer, DrawStyle

from .cancel import CancelToken
from .hooks import InferHooks
from .pipeline_config import PipelineConfig, load_pipeline_config
from .sinks import DetectionRecord, LocalFileSink, NullSink, OutputSink

logger = logging.getLogger(__name__)


# ============================================================================
# 逐帧 model() 的 predict 计算参数白名单
# ----------------------------------------------------------------------------
# 为什么不盲传 config.to_ultralytics_kwargs()? YOLOInferConfig 继承 BaseConfig,
# 带进来一堆训练向字段 (batch/workers/cache/rect/amp/seed/...), 这些传给逐帧
# model() 要么报错要么被忽略. 显式列出真正影响"单帧检测计算"的参数, 只传这些.
# ============================================================================
_PREDICT_KEYS: tuple[str, ...] = (
    "conf", "iou", "imgsz", "max_det", "classes",
    "agnostic_nms", "augment", "device", "retina_masks",
)


def _find_project_log_path() -> Path | None:
    """从项目根 logger 找 FileHandler 的实际文件路径 (只读, 给 audit 用).

    ★ 根 logger 名是 ROOT_LOGGER_NAME('od_platform'). 早先这里误写成 'odp_platform'
      (多了个 p), 于是永远找不到 FileHandler、audit 里 log_path 恒为 None. 已修.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            return Path(h.baseFilename)
    return None


def _resolve_model_arg(model: str) -> str:
    """推理模型解析 —— 跟训练侧 model_train.service._resolve_model_arg 同构, 但语义相反.

    - 训练: 从 models/pretrained/ 找; 找不到就返回裸名, 交给 ultralytics 联网自动下载.
    - 推理: 从 models/trained/ 找(用户自己训出来的权重); 找不到【直接报错】——
            推理不该悄悄下一个不相干的预训练权重来跑, 那样结果是错的还不自知.

    refs.resolve_model 已负责: 绝对路径/带目录 → 原样; 多格式(.pt/.onnx/.engine/...)
    不再把后缀写死成 .pt.
    """
    p = refs.resolve_model(model)
    if not p.exists():
        raise FileNotFoundError(
            f"未找到推理模型: {p}\n"
            f"  · 训练好的权重请放在 {TRAINED_MODELS_DIR} 下, 再用 --model <名字> 指向它;\n"
            f"  · 或直接传模型的绝对路径 (--model /abs/path/to/xxx.pt);\n"
            f"  · 支持 .pt / .onnx / .engine 等格式."
        )
    logger.info("模型解析: %s", p)
    return str(p)


def _resolve_infer_source(raw_source: str, pipe: PipelineConfig):
    """把推理输入源字符串解析成 create_frame_source 认的 str | SourceConfig.

    - 裸数字("0")→ 相机: 用 infer_pipeline.yaml 的相机块 + 该设备号造 CameraConfig,
      让分辨率/帧率/后端/编码真正生效(修复"相机自定义配置被整段丢弃"的问题);
    - 其余(视频/图片/目录/URL)→ 原样返回字符串, 交给 frame_source 的字符串规则识别.
    """
    s = str(raw_source)
    if s.isdigit():
        cam = pipe.build_camera_config(camera_id=int(s))
        if cam is not None:
            return cam
    return s


# ============================================================================
# 推理统计 —— 推理侧没有 mAP, 取而代之的是 帧数/检测数/每类计数/FPS
# ============================================================================
@dataclass
class InferStats:
    """一次推理跑完的统计快照."""
    frames: int = 0
    detections: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    infer_time_sec: float = 0.0

    # 多维帧率 (引擎跑完填入)
    capture_fps: float = 0.0
    infer_fps: float = 0.0
    render_fps: float = 0.0
    loop_fps: float = 0.0
    current_fps: float = 0.0
    speed_ms: dict[str, float] = field(default_factory=dict)

    @property
    def avg_fps(self) -> float:
        return self.frames / self.infer_time_sec if self.infer_time_sec > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (self.infer_time_sec / self.frames * 1000.0) if self.frames else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "per_class": dict(self.per_class),
            "infer_time_sec": round(self.infer_time_sec, 4),
            "avg_fps": round(self.avg_fps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "fps": {
                "capture": self.capture_fps,
                "infer": self.infer_fps,
                "render": self.render_fps,
                "loop": self.loop_fps,
                "current": self.current_fps,
            },
            "speed_ms": dict(self.speed_ms),
        }


def log_infer_stats(stats: InferStats, *, logger: logging.Logger = logger) -> None:
    """漂亮打印推理统计 (含多维帧率)."""
    logger.info(f"处理帧数:   {stats.frames}")
    logger.info(f"检测总数:   {stats.detections}")
    logger.info(f"平均延迟:   {stats.avg_latency_ms:.2f} ms/帧")
    logger.info("帧率(FPS):  捕获 %.1f | 推理 %.1f | 渲染 %.1f | loop %.1f | 当前 %.1f" % (
        stats.capture_fps, stats.infer_fps, stats.render_fps,
        stats.loop_fps, stats.current_fps,
    ))
    if stats.speed_ms:
        logger.info("模型 speed(ms): 预处理 %.2f | 推理 %.2f | 后处理 %.2f" % (
            stats.speed_ms.get("preprocess", 0.0),
            stats.speed_ms.get("inference", 0.0),
            stats.speed_ms.get("postprocess", 0.0),
        ))
    if stats.per_class:
        logger.info("各类别检测数:")
        for name, cnt in sorted(stats.per_class.items(), key=lambda kv: -kv[1]):
            logger.info(f"    {name:<20} {cnt}")


@dataclass(frozen=True)
class InferResult:
    """推理结果一次性快照 (跟 TrainResult 平行)."""
    success:    bool
    output_dir: Path
    stats:      dict[str, Any] = field(default_factory=dict)
    infer_time: float | None = None
    saved:      bool = False
    error:      str | None = None
    audit_path: Path | None = None
    log_path:   Path | None = None


# ============================================================================
# InferService 主类
# ============================================================================
class InferService:
    """YOLO 推理流程编排."""

    def __init__(self) -> None:
        """__init__ 不接任何参数 —— 配置都通过 predict() 传."""
        pass

    def predict(
        self,
        *,
        # ---- 由调用方(CLI / infer_yolo)预先建好后传入, 跟训练侧 train_yolo 一致 ----
        config: Any,
        merger: Any,
        run: RunContext,
        pipeline_yaml: str | Path | None = None,
        # ---- CLI 默认行为参数 ----
        beautify: bool = True,
        warmup_frames: int = 0,
        window_name: str = "odp-infer",
        show_info: bool = True,
        # ---- ★ 接缝参数 (业务定制), keyword-only + 默认 None 让 CLI 行为不变 ----
        output_sink: OutputSink | None = None,
        hooks: InferHooks | None = None,
        cancel_token: CancelToken | None = None,
    ) -> InferResult:
        """跑一次完整推理.

        ★ 跟训练侧对齐: config / merger / run 由调用方(CLI 或 infer_yolo)先建好再传进来.
          run(RunContext)决定输出根目录(runs/inference/<run_id>)与产物命名; 日志名由 CLI
          用同一个 run_id 预先设好 —— 不再"先用临时名跑、事后重命名".

        Args:
            config:        已构建并校验过的 YOLOInferConfig.
            merger:        配置溯源器(写 audit 用, 允许 None).
            run:           RunContext, 提供 run_id / run_dir(= 输出根目录).
            pipeline_yaml: 帧源+美化 infer_pipeline.yaml 路径. None 走默认.
            beautify:      是否美化. False → 退回 YOLO 原生 plot().
            warmup_frames: 启动丢弃前 N 帧 (摄像头帧率不稳).
            window_name:   显示窗口标题 (--show 时).
            show_info:     是否画 HUD 信息面板.
            output_sink:   自定义输出适配器 (默认按 want_save 选 LocalFileSink / NullSink).
            hooks:         生命周期回调 (默认全空回调, 零开销).
            cancel_token:  程序化取消信号.

        Returns:
            InferResult. ★ 永不抛 —— 任何异常打包进 InferResult.error.
        """
        if hooks is None:
            hooks = InferHooks()

        start = datetime.now()
        output_dir: Path = run.run_dir       # RunContext 在 __enter__ 里已建好并去重

        try:
            # ================================================================
            # 阶段 1: 读帧源+美化配置 (D5 的 infer.yaml 已由调用方解析进 config)
            # ================================================================
            pipe: PipelineConfig = load_pipeline_config(pipeline_yaml)

            # ================================================================
            # 阶段 2: 上下文日志
            # ================================================================
            logger.info("=" * 60)
            logger.info(f"开始 YOLO 推理 (task={config.task})".center(60))
            logger.info("=" * 60)

            raw_model = config.model or "yolo11n.pt"
            raw_source = config.source
            logger.info(f"运行 ID:     {run.run_id}")
            logger.info(f"任务类型:    {config.task}")
            logger.info(f"输入源(声明): {raw_source!r}")
            logger.info(f"模型(声明):  {raw_model}")

            # ================================================================
            # 阶段 3: 源 + 模型解析
            # ================================================================
            if raw_source is None:
                raise RuntimeError(
                    "未指定推理输入源. 请在 infer.yaml 写 source, 或用 "
                    "`odp-infer --source <图/视频/目录/摄像头号>` 传入."
                )
            # 模型: models/trained/ 下找, 找不到直接报错; 支持绝对路径 + 多格式(.pt/.onnx/.engine)
            model_arg = _resolve_model_arg(raw_model)
            logger.info(f"模型(解析):  {model_arg}")

            # 帧源: 解析成 create_frame_source 认的 str | SourceConfig.
            #   相机源(裸数字)会带上 infer_pipeline.yaml 的相机配置 → 自定义配置真正生效.
            eff_source = _resolve_infer_source(str(raw_source), pipe)
            if isinstance(eff_source, SourceConfig):
                logger.info(f"帧源配置:    {eff_source!r}")

            # ================================================================
            # 阶段 4: 加载模型 + 建美化器 + 决定 sink
            # ================================================================
            model = YOLO(model_arg)
            class_names: list[str] = list(model.names.values())

            do_beautify = beautify and pipe.viz_enabled
            visualizer: BeautifyVisualizer | None = None
            if do_beautify:
                visualizer = BeautifyVisualizer(
                    labels=class_names,
                    label_mapping=pipe.label_mapping or None,
                    color_mapping=pipe.color_mapping or None,
                    default_color=pipe.default_color,
                    font_path=pipe.font_path,
                )
            else:
                logger.info("美化已关闭, 使用 YOLO 原生 plot() 绘制.")

            # 输出根 = run.run_dir(runs/inference/<run_id>), 跟训练 runs/training/<run_id> 对齐.
            # 产物名 stem 与日志名同源(都由 run_stem(run_id, model) 生成), 天然一致.
            stem = run_stem(stage="infer", run_id=run.run_id, model=model_stem(raw_model))
            logger.info(f"输出目录:    {output_dir}")
            logger.info(f"产物名称:    {stem}")

            # 逐帧 predict 计算参数 (白名单, 不盲传整个 config)
            predict_kwargs = {
                k: getattr(config, k)
                for k in _PREDICT_KEYS
                if getattr(config, k, None) is not None
            }
            predict_kwargs["verbose"] = False

            want_save = bool(getattr(config, "save", True))
            want_show = bool(getattr(config, "show", False))
            save_txt = bool(getattr(config, "save_txt", False))
            save_crop = bool(getattr(config, "save_crop", False))
            save_conf = bool(getattr(config, "save_conf", False))
            # 只要"存美化图 / 存标签 / 存切片"任一开着, 就要落盘(且不能丢帧)
            want_any_save = want_save or save_txt or save_crop

            # ★ 决定 sink: 调用方没传 → 按配置自动选; 传了 → 用调用方的.
            #   即便 save=False, 只要 save_txt / save_crop 开着也落盘(只是不写美化图).
            if output_sink is None:
                if want_any_save:
                    output_sink = LocalFileSink(
                        save_images=want_save,
                        save_txt=save_txt,
                        save_conf=save_conf,
                        save_crop=save_crop,
                    )
                else:
                    output_sink = NullSink()
            else:
                logger.info(f"使用调用方提供的 sink: {output_sink.__class__.__name__}")

            # ================================================================
            # 阶段 5: 执行推理
            # ================================================================
            logger.info("=" * 60)
            logger.info("启动推理".center(60))
            logger.info("=" * 60)

            stats = InferStats()
            processor = _FrameProcessor(
                model=model,
                predict_kwargs=predict_kwargs,
                do_beautify=do_beautify,
                visualizer=visualizer,
                use_label_mapping=pipe.use_label_mapping,
                style_overrides=pipe.style_overrides,
                names=model.names,
            )

            raw_batch = getattr(config, "batch", 16)
            batch_size = raw_batch if isinstance(raw_batch, int) and raw_batch >= 1 else 16

            from .pipeline import ThreadedPipeline
            logger.info(f"执行引擎: 多线程流水线 (batch={batch_size}, 显示与主循环解耦)")
            pipeline = ThreadedPipeline(
                processor=processor,
                source=eff_source,
                output_dir=output_dir,
                output_sink=output_sink,
                batch_size=batch_size,
                save=want_any_save,
                show=want_show,
                show_info=show_info,
                window_name=window_name,
                warmup_frames=warmup_frames,
                hooks=hooks,
                cancel_token=cancel_token,
            )
            interrupted = pipeline.run(stats)

            if interrupted:
                logger.warning("推理被用户提前结束.")

            # ================================================================
            # 阶段 6: 推理统计
            # ================================================================
            logger.info("=" * 60)
            logger.info("推理完成".center(60))
            logger.info("=" * 60)
            log_infer_stats(stats, logger=logger)

            # ================================================================
            # 阶段 7: 审计快照 (写进 run_dir, 与训练 odp_audit.json 一致)
            # ================================================================
            audit_path: Path | None = run.artifact_path("odp_audit.json")
            log_path = _find_project_log_path()
            try:
                audit_payload = {
                    "mode": "infer",
                    "run_id": run.run_id,
                    "stem": stem,
                    "config": config.to_audit_snapshot(),
                    "merger": merger.to_audit_log() if merger is not None else None,
                    "pipeline": pipe.to_audit(),
                    "stats": stats.to_dict(),
                    "result_summary": {
                        "output_dir": str(output_dir),
                        "saved": want_save,
                        "beautified": do_beautify,
                        "infer_time_sec": (datetime.now() - start).total_seconds(),
                        "log_path": str(log_path) if log_path else None,
                    },
                }
                audit_path.write_text(
                    json.dumps(audit_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info(f"审计快照:   {audit_path}")
            except OSError as e:
                logger.warning(f"写审计快照失败 (不影响推理结果): {e}")
                audit_path = None

            # ---- 收尾 ----
            infer_time = (datetime.now() - start).total_seconds()
            logger.info("=" * 60)
            logger.info(f"推理总耗时: {infer_time:.2f} 秒")
            logger.info(f"输出目录:   {output_dir}")
            if want_save:
                logger.info("结果已保存到上面的目录.")
            if log_path:
                logger.info(f"本次日志:   {log_path}")
            logger.info("=" * 60)

            result = InferResult(
                success=True,
                output_dir=output_dir,
                stats=stats.to_dict(),
                infer_time=infer_time,
                saved=want_save,
                audit_path=audit_path,
                log_path=log_path,
            )
            hooks.fire_complete(result)
            return result

        # ====================================================================
        # 顶层异常拦截 —— 永不抛, 打包成 InferResult.error
        # ====================================================================
        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            infer_time = (datetime.now() - start).total_seconds()
            hooks.fire_error(e)
            return InferResult(
                success=False,
                output_dir=output_dir,
                stats={},
                infer_time=infer_time,
                error=str(e),
                log_path=_find_project_log_path(),
            )


# ============================================================================
# 帧处理器 —— 把"推理"和"绘制"拆成两半, pipeline 共用
# ============================================================================
@dataclass
class _FrameProcessor:
    model: Any
    predict_kwargs: dict[str, Any]
    do_beautify: bool
    visualizer: BeautifyVisualizer | None
    use_label_mapping: bool
    style_overrides: dict[str, Any]
    names: dict[int, str]
    _style: DrawStyle | None = None

    def infer_batch(self, images: list):
        """主线程: 批量推理. 返回 (results, labels_list, n_list, batch_dt)."""
        t0 = time.perf_counter()
        results = self.model(images, **self.predict_kwargs)
        batch_dt = time.perf_counter() - t0
        labels_list: list[list[str]] = []
        n_list: list[int] = []
        for result in results:
            boxes = result.boxes
            n = 0 if boxes is None else len(boxes)
            n_list.append(n)
            labels_list.append(
                [self.names[i] for i in boxes.cls.int().cpu().tolist()] if n else []
            )
        return results, labels_list, n_list, batch_dt

    def draw(self, image, result, labels, n):
        """绘制单帧 → annotated(BGR). 美化关时退回 YOLO 原生 plot()."""
        if self.do_beautify and self.visualizer is not None:
            if self._style is None:
                h, w = image.shape[:2]
                self._style = DrawStyle.from_image_size(h, w, **self.style_overrides)
            boxes = result.boxes
            dets = BeautifyVisualizer.from_yolo_results(
                boxes=(boxes.xyxy.cpu().numpy() if n else _empty_boxes()),
                confidences=(boxes.conf.cpu().numpy() if n else _empty_conf()),
                labels=labels,
            )
            return self.visualizer.draw(
                image, dets, style=self._style, use_label_mapping=self.use_label_mapping,
            )
        return result.plot()

    def extract_records(self, result, labels, n) -> list[DetectionRecord]:
        """从 YOLO 结果抽出存标签/切片所需的检测记录(与美化解耦).

        只有 sink.wants_detections=True 时 pipeline 才会调这个 —— 不存标签/切片就零开销.
        """
        if not n:
            return []
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        xywhn = boxes.xywhn.cpu().numpy()      # 归一化 cx,cy,w,h (YOLO txt 格式)
        conf = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.int().cpu().tolist()
        recs: list[DetectionRecord] = []
        for i in range(n):
            x1, y1, x2, y2 = xyxy[i]
            cx, cy, bw, bh = xywhn[i]
            recs.append(DetectionRecord(
                cls_id=int(cls_ids[i]),
                name=labels[i] if i < len(labels) else str(cls_ids[i]),
                conf=float(conf[i]),
                xyxy=(int(x1), int(y1), int(x2), int(y2)),
                xywhn=(float(cx), float(cy), float(bw), float(bh)),
            ))
        return recs


def _empty_boxes():
    import numpy as np
    return np.zeros((0, 4), dtype=float)


def _empty_conf():
    import numpy as np
    return np.zeros((0,), dtype=float)


def infer_yolo(
    yaml_path: str | Path | None = None,
    pipeline_yaml: str | Path | None = None,
    cli_args: dict[str, Any] | None = None,
    *,
    run: RunContext | None = None,
    beautify: bool = True,
    warmup_frames: int = 0,
    window_name: str = "odp-infer",
    show_info: bool = True,
    output_sink: OutputSink | None = None,
    hooks: InferHooks | None = None,
    cancel_token: CancelToken | None = None,
) -> InferResult:
    """一行启动推理 —— 编程式调用的便捷入口(风格跟 train_yolo 一致).

    自己负责建 config + RunContext, 再调 InferService.predict; 没传 run 就临时开一个
    `RunContext("inference")`. CLI 走的是"CLI 建 config/RunContext/日志、再调 predict"的
    路径(见 cli/infer_model.py), 好让日志名用同一个 run_id 预先设死; 这里给没有 CLI 的
    业务方兜底.

    ★ 依然遵守 never-throws: 连配置构建阶段的异常也打包成 InferResult 返回, 不外抛.
    """
    service = InferService()
    try:
        config, merger = build_infer_config(
            yaml_path=yaml_path or "infer.yaml",
            cli_args=cli_args,
        )
    except Exception as e:
        logger.error(f"推理配置构建失败: {e}", exc_info=True)
        if hooks is not None:
            hooks.fire_error(e)
        return InferResult(
            success=False,
            output_dir=Path("unknown"),
            error=str(e),
            log_path=_find_project_log_path(),
        )

    def _run(r: RunContext) -> InferResult:
        return service.predict(
            config=config,
            merger=merger,
            run=r,
            pipeline_yaml=pipeline_yaml,
            beautify=beautify,
            warmup_frames=warmup_frames,
            window_name=window_name,
            show_info=show_info,
            output_sink=output_sink,
            hooks=hooks,
            cancel_token=cancel_token,
        )

    if run is not None:
        return _run(run)
    with RunContext("inference") as r:
        return _run(r)

