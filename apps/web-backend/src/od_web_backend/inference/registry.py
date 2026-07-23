"""模型注册表:扫描 models/trained、LRU 缓存已加载模型。

- 真实(yolo)模式:用引擎 refs.resolve_model 解析路径,ultralytics.YOLO 加载,LRU 缓存;
  可选 sidecar `<权重>.meta.json` 提供 {task, classes, metrics}(sync 时可自动回填 classes)。
- 演示(demo)模式:直接返回 settings.DEMO_MODELS,不加载任何权重。
"""
from __future__ import annotations
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)

_MODEL_SUFFIXES = ('.pt', '.onnx', '.engine', '.torchscript', '.mlpackage', '.tflite')

# name -> loaded model(仅 yolo 模式使用)
_MODEL_CACHE: 'OrderedDict[str, Any]' = OrderedDict()

_DEMO_SIZES = {'yolo11n': 6_240_000, 'yolo11s': 22_413_000, 'yolo11m': 49_720_000}


def _trained_dir() -> Path:
    from od_platform.common.paths import TRAINED_MODELS_DIR
    return TRAINED_MODELS_DIR


def _sidecar(path: Path) -> dict:
    meta = path.with_suffix(path.suffix + '.meta.json')
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding='utf-8'))
        except Exception:
            logger.warning('读取 sidecar 失败: %s', meta)
    return {}


def _demo_models() -> list[dict]:
    from datetime import datetime
    out = []
    for m in settings.DEMO_MODELS:
        size = next((v for k, v in _DEMO_SIZES.items() if k in m['name']), 20_000_000)
        out.append({
            'name': m['name'],
            'task': m.get('task', 'detect'),
            'classes': m['classes'],
            'num_classes': len(m['classes']),
            'size_bytes': size,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'metrics': m.get('metrics'),
        })
    return out


def list_trained_models() -> list[dict]:
    """列出可用模型(结构对齐前端 DetectionModel)。"""
    if settings.INFER_BACKEND == 'demo':
        return _demo_models()

    directory = _trained_dir()
    if not directory.exists():
        return []
    models: list[dict] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in _MODEL_SUFFIXES:
            continue
        stat = path.stat()
        meta = _sidecar(path)
        classes = meta.get('classes', [])
        models.append({
            'name': path.name,
            'task': meta.get('task', 'detect'),
            'classes': classes,
            'num_classes': len(classes),
            'size_bytes': stat.st_size,
            'updated_at': _iso(stat.st_mtime),
            'metrics': meta.get('metrics'),
        })
    return models


def sync_models() -> list[dict]:
    """重新扫描;真实模式下为缺 sidecar 的权重加载一次以回填类别名。"""
    if settings.INFER_BACKEND == 'demo':
        return _demo_models()

    directory = _trained_dir()
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in _MODEL_SUFFIXES:
                continue
            meta_path = path.with_suffix(path.suffix + '.meta.json')
            if meta_path.exists():
                continue
            try:
                model = get_model(path.name)
                names = list(model.names.values())
                meta_path.write_text(
                    json.dumps({'task': 'detect', 'classes': names}, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )
                logger.info('已回填模型类别 sidecar: %s (%d 类)', path.name, len(names))
            except Exception as exc:
                logger.warning('加载模型失败,跳过 sidecar 回填: %s (%s)', path.name, exc)
    return list_trained_models()


def get_model(name: str):
    """加载(或从 LRU 缓存取)一个 ultralytics 模型。仅 yolo 模式调用。"""
    if name in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(name)
        return _MODEL_CACHE[name]

    from od_platform.common.refs import resolve_model
    path = resolve_model(name)
    if not path.exists():
        raise FileNotFoundError(f'模型不存在: {path}(请把权重放到 models/trained/ 下)')

    from ultralytics import YOLO
    logger.info('加载模型: %s', path)
    model = YOLO(str(path))

    _MODEL_CACHE[name] = model
    _MODEL_CACHE.move_to_end(name)
    while len(_MODEL_CACHE) > settings.MODEL_CACHE_SIZE:
        evicted, _ = _MODEL_CACHE.popitem(last=False)
        logger.info('LRU 淘汰模型缓存: %s', evicted)
    return model


def model_classes(name: str) -> list[str]:
    """取某模型的类别列表(demo 用配置,yolo 用 sidecar/加载)。"""
    if settings.INFER_BACKEND == 'demo':
        for m in settings.DEMO_MODELS:
            if m['name'] == name:
                return m['classes']
        return list(settings.VIZ_LABEL_MAPPING.keys())
    # yolo
    for m in list_trained_models():
        if m['name'] == name and m['classes']:
            return m['classes']
    try:
        return list(get_model(name).names.values())
    except Exception:
        return list(settings.VIZ_LABEL_MAPPING.keys())


def _iso(ts: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(ts).isoformat(timespec='seconds')
