#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : sinks.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_infer/sinks.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-21 10:01:48
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-21:01:48 | Matri | v1.0.0 | 初始化创建
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : sinks.py
# @Project   : ODPlatform
# @Function  : 推理结果输出适配器 (OutputSink 抽象 + LocalFileSink + NullSink)
"""推理结果输出适配器.

把"画好的 annotated 帧"送到不同目的地的抽象层:

  - OutputSink     : 抽象基类 (open/write/close 三个方法)
  - LocalFileSink  : 本地 mp4 / jpg (★ CLI 默认行为, 等价于老式 cv2.imwrite/VideoWriter)
  - NullSink       : 不写任何东西 (web 流推 / 只跑统计 / 不存盘 时用)

业务端在自己仓库继承 OutputSink 实现自定义 sink:
  - S3Sink         : 写对象存储
  - WebSocketSink  : 实时推流
  - QtSignalSink   : 桥接到 Qt 信号 (本仓库 apps/desktop 里有实现, 见阶段 11)

实现纪律 (与 D8 InferService 的"永不抛"一脉相承):
  - write() 内部 try/except 包住, 单帧写入失败 logger.warning + 跳过, 不抛
  - close() 幂等 (允许多次调用)
  - open() 由 pipeline 在首批帧到达后调用 (此时 source_type 已知)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from od_platform.frame_source import SourceType

logger = logging.getLogger(__name__)


# ============================================================================
# 检测记录 (与 ultralytics 解耦的中间结构; sink 不依赖 YOLO 的 Result 类型)
# ============================================================================
@dataclass(frozen=True)
class DetectionRecord:
    """一条检测结果的最小快照 —— 够 sink 存标签(txt)和切片(crop)用.

    xyxy   : 像素坐标 (x1, y1, x2, y2), 存切片时按它从原图裁.
    xywhn  : 归一化 (cx, cy, w, h), 存 YOLO txt 时用 (与 ultralytics save_txt 同格式).
    """
    cls_id: int
    name: str
    conf: float
    xyxy: tuple[int, int, int, int]
    xywhn: tuple[float, float, float, float]


# ============================================================================
# 抽象基类
# ============================================================================
class OutputSink(ABC):
    """推理结果输出适配器.

    生命周期: open() → write() × N → close().

    实现要求:
      - write 永不抛 (出错 logger.warning + 跳过这一帧)
      - close 幂等 (多次调不出错)
      - 线程亲缘: write 在 _Renderer 线程被调, open/close 在主线程
    """

    @abstractmethod
    def open(self, output_dir: Path, source_type: SourceType) -> None:
        """初始化. 由 ThreadedPipeline 在首批帧到达后调用.

        Args:
            output_dir:  推理输出根目录 (部分 sink 可能不用)
            source_type: 帧源类型 (决定按视频流还是单图处理)
        """

    @property
    def wants_detections(self) -> bool:
        """是否需要 pipeline 额外传入逐帧检测记录(存标签/切片才需要).

        默认 False —— 只关心 annotated 帧的 sink(NullSink / 业务推流 sink)无需改动,
        pipeline 也就不会为它们做多余的检测记录提取, 零开销.
        """
        return False

    @abstractmethod
    def write(self, frame, annotated: np.ndarray,
              detections: Sequence[DetectionRecord] | None = None) -> None:
        """写一帧结果.

        Args:
            frame:      frame_source 给出的 Frame 对象 (含 image / info 元数据)
            annotated:  已画过框的 BGR 图像
            detections: 该帧的检测记录列表; 仅当 wants_detections=True 时 pipeline 才会传,
                        否则为 None. 只存 annotated 的 sink 忽略它即可.
        """

    @abstractmethod
    def close(self) -> None:
        """收尾. 由 ThreadedPipeline 在退出时调用. 必须幂等."""


# ============================================================================
# 内置实现 1: LocalFileSink (CLI 默认, 视频写 mp4 / 图片写 jpg)
# ============================================================================
class LocalFileSink(OutputSink):
    """本地文件 sink — 把结果落到 <run_dir>/<result_subdir>/ 下, 三类产物各占一个子目录:

        <run_dir>/result/
            images/<名字>.jpg  或  images/output.mp4   ← 渲染好的美化图 / 视频
            labels/<名字>.txt                          ← save_txt: YOLO 格式标签 (每图一份)
            crops/<类别>/<名字>_<i>.jpg                ← save_crop: 按检测框从"原图"裁的切片

    (images / labels / crops 三者平级, 结构统一, 便于下游按目录取用.)

    开关:
      - save_images: 是否写美化图/视频 (默认 True; 关掉可只出标签/切片)
      - save_txt   : 每帧一份 YOLO txt —— 行格式 `cls cx cy w h`(归一化)
      - save_conf  : txt 行末追加置信度 `... conf` (需 save_txt=True 配套)
      - save_crop  : 每个检测框裁一张切片, 按类别分子目录

    命名:
      - 视频/相机源: 美化写成单个 output.mp4; 标签/切片按帧号 frame_000123 命名;
      - 图片/目录源: 每张各写各的, 按源文件名(去扩展名)命名.
    """

    def __init__(
        self,
        *,
        save_images: bool = True,
        save_txt: bool = False,
        save_conf: bool = False,
        save_crop: bool = False,
        result_subdir: str = "result",
        images_subdir: str = "images",
    ) -> None:
        self.save_images = save_images
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.result_subdir = result_subdir
        self.images_subdir = images_subdir

        self.result_dir: Path | None = None
        self.images_dir: Path | None = None
        self.labels_dir: Path | None = None
        self.crops_dir: Path | None = None
        self._is_stream: bool = False
        self._video = None      # cv2.VideoWriter, lazy
        self._count: int = 0

    @property
    def wants_detections(self) -> bool:
        # 只有要存标签或切片时, 才需要 pipeline 把逐帧检测记录传进来
        return self.save_txt or self.save_crop

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        self._is_stream = source_type in (SourceType.VIDEO, SourceType.CAMERA)
        self.result_dir = Path(output_dir) / self.result_subdir
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            self.images_dir = self.result_dir / self.images_subdir
            self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.save_txt:
            self.labels_dir = self.result_dir / "labels"
            self.labels_dir.mkdir(parents=True, exist_ok=True)
        if self.save_crop:
            self.crops_dir = self.result_dir / "crops"
            self.crops_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "输出落盘目录: %s (图像=%s, 标签=%s, 切片=%s)",
            self.result_dir, self.save_images, self.save_txt, self.save_crop,
        )

    def _frame_stem(self, frame) -> str:
        """本帧产物的基名: 视频按帧号, 图片按源文件名(去扩展名)."""
        idx = getattr(frame.info, "frame_index", self._count)
        if self._is_stream:
            return f"frame_{idx:06d}"
        fname = getattr(frame.info, "filename", None) or f"frame_{idx:06d}"
        return Path(fname).stem

    def write(self, frame, annotated, detections=None) -> None:
        import cv2
        stem = self._frame_stem(frame)

        # 1) 美化图 / 视频 → result/images/
        if self.save_images:
            try:
                if self._is_stream:
                    if self._video is None:
                        h, w = annotated.shape[:2]
                        fps = float(frame.info.fps) if getattr(frame.info, "fps", None) else 30.0
                        out = self.images_dir / "output.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self._video = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
                    self._video.write(annotated)
                else:
                    cv2.imwrite(str(self.images_dir / f"{stem}.jpg"), annotated)
            except Exception as e:
                logger.warning(f"LocalFileSink 写图像失败, 跳过: {e}")

        # 2) 标签 txt
        if self.save_txt and detections is not None:
            self._write_labels(stem, detections)

        # 3) 切片 crop
        if self.save_crop and detections is not None:
            self._write_crops(stem, frame, detections)

        self._count += 1

    def _write_labels(self, stem: str, detections) -> None:
        try:
            lines = []
            for d in detections:
                cx, cy, w, h = d.xywhn
                if self.save_conf:
                    lines.append(f"{d.cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {d.conf:.6f}")
                else:
                    lines.append(f"{d.cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            # 没检测也写个空 txt, 方便下游按"每图一份"对齐 (与 ultralytics 行为一致)
            (self.labels_dir / f"{stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"LocalFileSink 写标签失败, 跳过: {e}")

    def _write_crops(self, stem: str, frame, detections) -> None:
        import cv2
        img = frame.image
        h_img, w_img = img.shape[:2]
        for i, d in enumerate(detections):
            try:
                x1, y1, x2, y2 = d.xyxy
                x1 = max(0, min(int(x1), w_img)); x2 = max(0, min(int(x2), w_img))
                y1 = max(0, min(int(y1), h_img)); y2 = max(0, min(int(y2), h_img))
                if x2 <= x1 or y2 <= y1:
                    continue
                safe_cls = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(d.name))
                cls_dir = self.crops_dir / safe_cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cls_dir / f"{stem}_{i}.jpg"), img[y1:y2, x1:x2])
            except Exception as e:
                logger.warning(f"LocalFileSink 写切片失败, 跳过: {e}")

    def close(self) -> None:
        if self._video is not None:
            try:
                self._video.release()
            except Exception as e:
                logger.warning(f"LocalFileSink.close release 失败 (已吞): {e}")
            finally:
                self._video = None


# ============================================================================
# 内置实现 2: NullSink (不写任何东西)
# ============================================================================
class NullSink(OutputSink):
    """什么也不写.

    用途:
      - web 流推: annotated 已通过 WebSocketSink 推走, 后端无需落盘
      - desktop: annotated 已通过 QtSignalSink 给 UI, 不需要本地文件
      - 性能基准测试: 排除 IO 干扰
      - --no-save 模式下的 CLI
    """

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        pass

    def write(self, frame, annotated, detections=None) -> None:
        pass

    def close(self) -> None:
        pass