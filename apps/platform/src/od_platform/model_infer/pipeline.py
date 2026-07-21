#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : pipeline.py
# @Project   : ODPlatform
# @Function  : 多线程推理流水线 — 采集/推理/渲染/显示 四级解耦, 主线程不碰 GUI
"""多线程推理流水线 (含接缝: OutputSink / InferHooks / CancelToken).

★ 速度的关键: 主线程【只做 predict + 非阻塞派发 + 读按键】, 从不调
  cv2.imshow / cv2.waitKey. 显示在【独立线程】里用 cv2.pollKey (非阻塞) 刷新.

四级流水线:
    reader 线程  读帧 (相机→只留最新; 视频/图片→有界阻塞)
      → 主线程   批量 predict + 派发 + 周期 fire_progress + 4 个 cancel 查询点
      → render 线程  美化绘制 → sink.write → fire_frame → 叠 HUD
      → display 线程 imshow + pollKey (仅 show=True 时)

接缝 (向后 100% 兼容, CLI 不传等价老式行为):
  - output_sink: 任意 OutputSink 实现, 默认 LocalFileSink (CLI) 或 NullSink (no-save)
  - hooks      : InferHooks dataclass, 默认全空回调零开销
  - cancel_token: CancelToken, 默认 None → 主循环查询零开销

loop FPS 测量纪律 (见 7.9.3):
  - 批级测量 + 均摊到帧, 不在批内 for 逐帧 update
  - 不然滑动窗口会被几微秒的污染样本灌满
"""
from __future__ import annotations

import logging
import time
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread

from od_platform.frame_source import (
    create_frame_source, CameraConfig, SourceConfig, is_stream_url,
)

from .cancel import CancelToken
from .hooks import FrameEvent, InferHooks, ProgressEvent
from .overlay import Metrics, draw_hud, draw_pause
from .sinks import NullSink, OutputSink

logger = logging.getLogger(__name__)

_SENTINEL = object()


def _put_latest(q: Queue, item) -> None:
    """满则丢最旧再放 (latest-wins, 低延迟)."""
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Full:
            pass


def _put_block(q: Queue, item) -> None:
    """满则阻塞 (不丢, 存盘完整)."""
    while True:
        try:
            q.put(item, timeout=1.0)
            return
        except Full:
            continue


def _is_live_source(source) -> bool:
    """是否实时源 —— 实时源用 latest-wins(只留最新帧, 低延迟),
    非实时源用有界阻塞(全处理不丢帧).

    - 相机配置(CameraConfig)      → 实时;
    - 网络流 URL(rtsp/rtmp/http(s))→ 实时;
    - 裸数字字符串("0")           → 实时(相机);
    - 视频文件 / 图片 / 目录        → 非实时.

    注: 早先这里只认 isdigit + rtsp/rtmp, 漏了 http(s) 直播流(会被当成非实时,
        走批处理阻塞导致延迟越堆越高); 现在统一用 frame_source 的 is_stream_url.
    """
    if isinstance(source, CameraConfig):
        return True
    if isinstance(source, SourceConfig):
        path = getattr(source, "path", None)
        return bool(path) and is_stream_url(str(path))
    s = str(source)
    return s.isdigit() or is_stream_url(s)


class _Controller:
    """暂停控制 (线程共享)."""
    def __init__(self) -> None:
        self._paused = Event()
    def toggle(self) -> None:
        self._paused.clear() if self._paused.is_set() else self._paused.set()
    def is_paused(self) -> bool:
        return self._paused.is_set()


# ============================================================================
# reader 线程
# ============================================================================
class _Reader(Thread):
    """读帧线程. 相机: 只留最新 (低延迟); 视频/图片: 有界阻塞 (全处理不丢)."""

    def __init__(self, source, *, live: bool, capacity: int, capture_fps) -> None:
        super().__init__(daemon=True)
        # source 现在是 create_frame_source 直接认的 str | SourceConfig.
        # 相机若带自定义配置, 这里就是一个 CameraConfig 实例 —— 配置会真正生效.
        self._source = source
        self._live = live
        self._capture_fps = capture_fps
        self.q: Queue = Queue(maxsize=1 if live else capacity)
        self._stop_evt = Event()
        self.source_type = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            with create_frame_source(self._source) as src:
                self.source_type = src.get_source_type()
                t_prev = time.perf_counter()
                for frame in src:
                    if self._stop_evt.is_set():
                        break
                    now = time.perf_counter()
                    self._capture_fps.update((now - t_prev) * 1000)
                    t_prev = now
                    if self._live:
                        _put_latest(self.q, frame)
                    else:
                        _put_block(self.q, frame)
        except Exception as e:
            self.error = e
        finally:
            _put_block(self.q, _SENTINEL)

    def get(self, timeout: float):
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None

    def get_nowait(self):
        try:
            return self.q.get_nowait()
        except Empty:
            return None

    def stop(self) -> None:
        self._stop_evt.set()


# ============================================================================
# render 线程
# ============================================================================
class _Renderer(Thread):
    """渲染线程: 美化绘制 → sink.write → fire_frame → 叠 HUD → 丢给显示队列."""

    def __init__(
        self,
        processor,
        in_q: Queue,
        out_q: Queue,
        *,
        drop: bool,
        output_sink: OutputSink,
        show: bool,
        show_info: bool,
        recording: bool,
        metrics: Metrics,
        hooks: InferHooks,
    ) -> None:
        super().__init__(daemon=True)
        self._proc = processor
        self._in = in_q
        self._out = out_q
        self._drop = drop
        self._sink = output_sink
        self._show = show
        self._show_info = show_info
        self._recording = recording
        self._m = metrics
        self._hooks = hooks
        self._stop_evt = Event()
        self._frame_idx = 0

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                item = self._in.get(timeout=0.1)
            except Empty:
                continue
            if item is _SENTINEL:
                _put_block(self._out, _SENTINEL)
                break
            frame, result, labels, n = item
            t0 = time.perf_counter()
            try:
                annotated = self._proc.draw(frame.image, result, labels, n)
            except Exception as e:
                logger.warning(f"渲染单帧失败, 跳过: {e}")
                continue
            self._m.render.update((time.perf_counter() - t0) * 1000)

            # 存标签/切片才需要逐帧检测记录; 只关心 annotated 的 sink(NullSink 等)零开销
            records = None
            if self._sink.wants_detections:
                try:
                    records = self._proc.extract_records(result, labels, n)
                except Exception as e:
                    logger.warning(f"提取检测记录失败 (不影响图像存储): {e}")
                    records = None

            # ★ 接缝 1: sink.write (NullSink no-op, LocalFileSink 写本地, QtSignalSink 发 Qt 信号)
            self._sink.write(frame, annotated, records)

            # ★ 接缝 2: fire_frame
            if self._hooks.on_frame is not None:
                self._hooks.fire_frame(FrameEvent(
                    frame_idx=self._frame_idx,
                    image=frame.image,
                    annotated=annotated,
                    n_detections=n,
                    detections=None,
                ))
            self._frame_idx += 1

            if self._show:
                # 永远 copy 给 HUD (sink 可能持有 annotated 引用, 不能就地改)
                disp = annotated.copy()
                draw_hud(disp, self._m, n_dets=n, recording=self._recording,
                         show_info=self._show_info)
                _put_latest(self._out, disp) if self._drop else _put_block(self._out, disp)


# ============================================================================
# display 线程 (★ 唯一碰 cv2.imshow/pollKey 的地方)
# ============================================================================
class _Display(Thread):
    """显示线程: imshow + pollKey (非阻塞) 抓键 + 暂停层."""

    def __init__(self, out_q: Queue, window_name: str, controller: _Controller) -> None:
        super().__init__(daemon=True)
        self._out = out_q
        self._win = window_name
        self._ctrl = controller
        self._stop_evt = Event()
        self._key_lock = Lock()
        self._key = -1
        self._last = None

    def stop(self) -> None:
        self._stop_evt.set()

    def get_key(self) -> int:
        with self._key_lock:
            k, self._key = self._key, -1
            return k

    def run(self) -> None:
        import cv2
        poll = cv2.pollKey if hasattr(cv2, "pollKey") else (lambda: cv2.waitKey(1))
        while not self._stop_evt.is_set():
            frame = None
            try:
                item = self._out.get(timeout=0.03)
                if item is _SENTINEL:
                    pass
                else:
                    frame = item
                    self._last = frame
            except Empty:
                if self._ctrl.is_paused() and self._last is not None:
                    frame = self._last.copy()
                    draw_pause(frame)
            if frame is not None:
                cv2.imshow(self._win, frame)
            key = poll() & 0xFF
            if key != 255:
                with self._key_lock:
                    self._key = key
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ============================================================================
# 主编排
# ============================================================================
class ThreadedPipeline:
    """多线程流水线. run() 阻塞跑完, 填充 stats, 返回是否被用户中断."""

    def __init__(
        self,
        *,
        processor,
        source,
        output_dir,
        output_sink: OutputSink,
        batch_size,
        save,
        show,
        show_info,
        window_name,
        warmup_frames,
        hooks: InferHooks | None = None,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.proc = processor
        # source: str | SourceConfig, 原样透传给 frame_source(不再 str() 掉, 否则
        # CameraConfig 会被转成 repr, 相机自定义配置又丢了).
        self.source = source
        self.source_label = source.identity if isinstance(source, SourceConfig) else str(source)
        self.output_dir = output_dir
        self.sink = output_sink
        self.batch_size = max(1, batch_size)
        self.save = save
        self.show = show
        self.show_info = show_info
        self.window_name = window_name
        self.warmup_frames = warmup_frames
        self.hooks = hooks if hooks is not None else InferHooks()
        self.cancel_token = cancel_token

    def _is_cancelled(self) -> bool:
        """统一查询入口 (cancel_token 为 None 时永远 False, 零开销)."""
        return self.cancel_token is not None and self.cancel_token.is_cancelled()

    def run(self, stats) -> bool:
        m = Metrics()
        live = _is_live_source(self.source)
        eff_batch = 1 if live else self.batch_size
        render_drop = not self.save

        reader = _Reader(self.source, live=live,
                         capacity=max(eff_batch * 2, 8), capture_fps=m.capture)
        in_q: Queue = Queue(maxsize=max(eff_batch * 2, 4))
        out_q: Queue = Queue(maxsize=2)
        controller = _Controller()

        renderer = None
        display = None
        interrupted = False
        warmed = 0
        last_batch_end_t = None
        start_time = time.perf_counter()
        sink_opened = False

        reader.start()

        try:
            while True:
                # 暂停: 不 predict, 只查取消 + 读键
                if controller.is_paused():
                    if self._is_cancelled():           # ★ 查询点 1
                        logger.info("收到取消信号 (暂停状态), 退出.")
                        interrupted = True
                        break
                    if self._handle_key(display, controller):
                        interrupted = True
                        break
                    time.sleep(0.02)
                    continue

                # ★ 查询点 2: 取批之前查
                if self._is_cancelled():
                    logger.info("收到取消信号, 退出主循环.")
                    interrupted = True
                    break

                # 取一批
                first = reader.get(timeout=2.0)
                if first is None:
                    if reader.error:
                        raise reader.error
                    continue
                if first is _SENTINEL:
                    break
                batch = [first]
                ended = False
                for _ in range(eff_batch - 1):
                    nxt = reader.get_nowait()
                    if nxt is None:
                        break
                    if nxt is _SENTINEL:
                        ended = True
                        break
                    batch.append(nxt)

                # warmup: 丢弃前 N 帧
                if warmed < self.warmup_frames:
                    warmed += len(batch)
                    if ended:
                        break
                    continue

                # 首批时拉起 renderer/display + 开 sink
                if renderer is None:
                    try:
                        self.sink.open(self.output_dir, reader.source_type)
                        sink_opened = True
                    except Exception as e:
                        logger.error(f"sink.open 失败, 退化用 NullSink: {e}")
                        self.sink = NullSink()
                        self.sink.open(self.output_dir, reader.source_type)
                        sink_opened = True

                    renderer = _Renderer(
                        self.proc, in_q, out_q,
                        drop=render_drop,
                        output_sink=self.sink,
                        show=self.show,
                        show_info=self.show_info,
                        recording=self.save,
                        metrics=m,
                        hooks=self.hooks,
                    )
                    renderer.start()
                    if self.show:
                        display = _Display(out_q, self.window_name, controller)
                        display.start()

                # --- 主线程: 批量 predict ---
                images = [f.image for f in batch]
                results, labels_list, n_list, batch_dt = self.proc.infer_batch(images)
                stats.infer_time_sec += batch_dt
                for frame, result, labels, n in zip(batch, results, labels_list, n_list):
                    stats.frames += 1
                    stats.detections += n
                    for name in labels:
                        stats.per_class[name] = stats.per_class.get(name, 0) + 1
                    m.add_speed(getattr(result, "speed", None))
                    # 不在这里 update m.loop, 批级测量在循环外做
                    if render_drop:
                        _put_latest(in_q, (frame, result, labels, n))
                    else:
                        _put_block(in_q, (frame, result, labels, n))

                    # ★ 周期 fire_progress
                    if (self.hooks.on_progress is not None
                            and stats.frames % self.hooks.progress_interval_frames == 0):
                        self.hooks.fire_progress(ProgressEvent(
                            frame_idx=stats.frames,
                            total_frames=None,
                            elapsed_sec=time.perf_counter() - start_time,
                            fps_loop=m.loop.fps,
                            fps_infer=m.infer.fps,
                            detections_total=stats.detections,
                        ))

                # ★ 批级测量 loop FPS, 均摊到每帧填窗口
                batch_end_t = time.perf_counter()
                if last_batch_end_t is not None:
                    per_frame_loop_ms = (batch_end_t - last_batch_end_t) * 1000 / len(batch)
                    for _ in batch:
                        m.loop.update(per_frame_loop_ms)
                last_batch_end_t = batch_end_t

                # ★ 查询点 3: 派发完查
                if self._is_cancelled():
                    logger.info("收到取消信号 (派发后), 退出.")
                    interrupted = True
                    break

                # ★ 查询点 4: 读键
                if self._handle_key(display, controller):
                    interrupted = True
                    break
                if ended:
                    break
        finally:
            reader.stop()
            _put_block(in_q, _SENTINEL)
            if renderer is not None:
                renderer.join(timeout=3.0)
                renderer.stop()
            if display is not None:
                time.sleep(0.05)
                display.stop()
                display.join(timeout=1.0)
            if sink_opened:
                try:
                    self.sink.close()
                except Exception as e:
                    logger.warning(f"sink.close 异常 (已吞): {e}")

        _write_fps(stats, m)
        logger.info("流水线收尾: 捕获 %.1f | 推理 %.1f | 渲染 %.1f | loop %.1f FPS"
                    % (m.capture.fps, m.infer.fps, m.render.fps, m.loop.fps))
        return interrupted

    def _handle_key(self, display, controller) -> bool:
        """读显示线程抓到的键. 返回是否要退出."""
        if display is None:
            return False
        key = display.get_key()
        if key in (ord("q"), 27):
            logger.info("用户请求退出 (q/Esc).")
            return True
        if key == ord(" "):
            controller.toggle()
            logger.info("已暂停 (空格恢复)" if controller.is_paused() else "已恢复")
        return False


def _write_fps(stats, m: Metrics) -> None:
    snap = m.snapshot()
    stats.capture_fps = snap["capture_fps"]
    stats.infer_fps = snap["infer_fps"]
    stats.render_fps = snap["render_fps"]
    stats.loop_fps = snap["loop_fps"]
    stats.current_fps = snap["current_fps"]
    stats.speed_ms = snap["speed_ms"]