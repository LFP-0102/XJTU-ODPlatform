#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : test_model_eval.py
# @Path       : XJTU-ODPlatfrom/apps/platform/tests/test_model_eval.py
# @Project    : XJTU-ODPlatfrom
# @Author     : ODPlatform Team
# @Function   : model_eval 模块单元测试
"""model_eval 模块测试.

测试范围:
  · EvalMetrics 构造 / from_yolo_results / to_dict
  · EvalReport 构造 / 序列化 / Markdown 渲染
  · ComparisonReport 构造 / best_model / overall_winner / 序列化 / Markdown 渲染
  · rank_classes / diagnose_problem_classes / compare_per_class
  · EvalRecord / EvalHistory 序列化
  · CLI 参数解析
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# ---- 辅助: 构造 mock EvalMetrics ----

def _mock_metrics(model_name: str = "test_model", **overrides) -> Any:
    """构造一个带典型值的 EvalMetrics, 用于测试."""
    from od_platform.model_eval.metrics import EvalMetrics

    defaults: Dict[str, Any] = {
        "run_id": "20260723-120000",
        "model_name": model_name,
        "model_path": f"models/trained/{model_name}.pt",
        "task": "detect",
        "split": "val",
        "precision": 0.8521,
        "recall": 0.7812,
        "mAP50": 0.8356,
        "mAP50_95": 0.6123,
        "f1": 0.8151,
        "accuracy": 0.7234,
        "fitness": 0.6543,
        "speed_ms": {"preprocess": 0.5, "inference": 12.3, "loss": 0.1, "postprocess": 0.8, "total": 13.7},
        "per_class": {
            "helmet": {"precision": 0.90, "recall": 0.85, "f1": 0.874, "mAP50": 0.88, "mAP50_95": 0.65},
            "person": {"precision": 0.78, "recall": 0.72, "f1": 0.749, "mAP50": 0.76, "mAP50_95": 0.55},
            "vest":   {"precision": 0.88, "recall": 0.77, "f1": 0.821, "mAP50": 0.87, "mAP50_95": 0.64},
        },
    }
    defaults.update(overrides)
    return EvalMetrics(**defaults)


def _mock_metrics_b(model_name: str = "baseline") -> Any:
    """构造一个略差的 baseline 模型指标."""
    return _mock_metrics(
        model_name=model_name,
        precision=0.7812, recall=0.7015, mAP50=0.7543, mAP50_95=0.5234,
        f1=0.7392, accuracy=0.6543, fitness=0.5891,
        per_class={
            "helmet": {"precision": 0.82, "recall": 0.78, "f1": 0.799, "mAP50": 0.81, "mAP50_95": 0.58},
            "person": {"precision": 0.71, "recall": 0.65, "f1": 0.679, "mAP50": 0.70, "mAP50_95": 0.48},
            "vest":   {"precision": 0.81, "recall": 0.68, "f1": 0.739, "mAP50": 0.79, "mAP50_95": 0.51},
        },
    )


# ============================================================
# EvalMetrics 测试
# ============================================================

class TestEvalMetrics:
    """EvalMetrics 数据结构测试."""

    def test_construction(self):
        m = _mock_metrics()
        assert m.model_name == "test_model"
        assert m.precision == 0.8521
        assert m.recall == 0.7812
        assert m.mAP50 == 0.8356
        assert m.mAP50_95 == 0.6123
        assert m.task == "detect"
        assert m.split == "val"

    def test_frozen(self):
        m = _mock_metrics()
        try:
            m.precision = 0.99  # type: ignore[misc]
            assert False, "frozen dataclass 应该拒绝赋值"
        except Exception:
            pass  # 预期行为

    def test_f1_derived(self):
        m = _mock_metrics()
        expected = 2 * 0.8521 * 0.7812 / (0.8521 + 0.7812)
        assert abs(m.f1 - expected) < 1e-6

    def test_to_dict(self):
        m = _mock_metrics()
        d = m.to_dict()
        assert d["model_name"] == "test_model"
        assert d["precision"] == 0.8521
        assert isinstance(d["per_class"], dict)
        assert "helmet" in d["per_class"]
        # NaN -> None
        m_nan = _mock_metrics(precision=math.nan)
        d_nan = m_nan.to_dict()
        assert d_nan["precision"] is None

    def test_to_dict_json_serializable(self):
        d = _mock_metrics().to_dict()
        s = json.dumps(d, ensure_ascii=False)
        assert len(s) > 0

    def test_per_class_structure(self):
        m = _mock_metrics()
        assert len(m.per_class) == 3
        h = m.per_class["helmet"]
        assert h["precision"] == 0.90
        assert h["mAP50_95"] == 0.65

    def test_nan_fields(self):
        m = _mock_metrics(accuracy=math.nan, fitness=math.nan)
        d = m.to_dict()
        assert d["accuracy"] is None
        assert d["fitness"] is None


# ============================================================
# EvalReport 测试
# ============================================================

class TestEvalReport:
    """EvalReport 报告测试."""

    def _report(self) -> Any:
        from od_platform.model_eval.report import EvalReport
        return EvalReport(
            run_id="20260723-120000",
            model_name="test_model",
            model_path="models/trained/test_model.pt",
            data_yaml="datasets/helmet.yaml",
            split="val",
            created_at="2026-07-23T12:00:00",
            metrics=_mock_metrics(),
        )

    def test_construction(self):
        r = self._report()
        assert r.run_id == "20260723-120000"
        assert r.model_name == "test_model"

    def test_to_dict(self):
        d = self._report().to_dict()
        assert d["metrics"]["precision"] == 0.8521
        assert d["data_yaml"] == "datasets/helmet.yaml"

    def test_write_json(self):
        r = self._report()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.json"
            r.write_json(p)
            assert p.exists()
            data = json.loads(p.read_text(encoding="utf-8"))
            assert data["model_name"] == "test_model"

    def test_write_csv(self):
        r = self._report()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "result.csv"
            r.write_csv(p)
            assert p.exists()
            content = p.read_text(encoding="utf-8-sig")
            assert "Precision" in content
            assert "test_model" in content

    def test_render_markdown(self):
        md = self._report().render_markdown()
        assert "# 模型评估报告" in md
        assert "test_model" in md
        assert "Precision" in md
        assert "0.8521" in md
        assert "helmet" in md  # 类别级指标

    def test_render_to_logger(self):
        import logging
        r = self._report()
        logger = logging.getLogger("test_eval_report")
        logger.setLevel(logging.INFO)
        # 不应抛异常
        r.render_to_logger(logger)


# ============================================================
# ComparisonReport 测试
# ============================================================

class TestComparisonReport:
    """ComparisonReport 多模型对比测试."""

    def _report(self) -> Any:
        from od_platform.model_eval.report import ComparisonReport
        return ComparisonReport(
            run_id="20260723-120000",
            data_yaml="datasets/helmet.yaml",
            split="val",
            created_at="2026-07-23T12:00:00",
            models=[_mock_metrics("yolo11n"), _mock_metrics_b("yolo11s")],
        )

    def test_construction(self):
        r = self._report()
        assert len(r.models) == 2

    def test_best_model(self):
        r = self._report()
        assert r.best_model("mAP50_95") == "yolo11n"
        assert r.best_model("precision") == "yolo11n"

    def test_fastest_model(self):
        r = self._report()
        # 两个 model 速度一样(都用 mock 默认值), 谁先谁胜
        assert r.fastest_model() is not None

    def test_overall_winner(self):
        r = self._report()
        assert r.overall_winner() == "yolo11n"

    def test_to_dict(self):
        d = self._report().to_dict()
        assert "analysis" in d
        assert d["analysis"]["overall_winner"] == "yolo11n"

    def test_write_json(self):
        r = self._report()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "comparison.json"
            r.write_json(p)
            assert p.exists()
            data = json.loads(p.read_text(encoding="utf-8"))
            assert len(data["models"]) == 2

    def test_write_csv(self):
        r = self._report()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "comparison.csv"
            r.write_csv(p)
            assert p.exists()
            content = p.read_text(encoding="utf-8-sig")
            assert "yolo11n" in content
            assert "yolo11s" in content

    def test_render_markdown(self):
        md = self._report().render_markdown()
        assert "多模型对比评估报告" in md
        assert "yolo11n" in md
        assert "yolo11s" in md
        assert "差异分析" in md
        assert "总体最优" in md or "**yolo11n**" in md

    def test_single_model_comparison(self):
        """单模型也可以进对比(退化情况)."""
        from od_platform.model_eval.report import ComparisonReport
        r = ComparisonReport(
            run_id="t", data_yaml="d.yaml", split="val",
            created_at="2026-01-01T00:00:00",
            models=[_mock_metrics("only_model")],
        )
        assert r.overall_winner() == "only_model"
        assert len(r.models) == 1

    def test_empty_comparison(self):
        from od_platform.model_eval.report import ComparisonReport
        r = ComparisonReport(
            run_id="t", data_yaml="d.yaml", split="val",
            created_at="2026-01-01T00:00:00", models=[],
        )
        assert r.overall_winner() is None
        md = r.render_markdown()
        assert "无可用模型结果" in md


# ============================================================
# analyzer 测试
# ============================================================

class TestAnalyzer:
    """深度分析工具测试."""

    def test_rank_classes(self):
        from od_platform.model_eval.analyzer import rank_classes
        m = _mock_metrics()
        rankings = rank_classes(m, sort_by="mAP50_95")
        assert len(rankings) == 3
        assert rankings[0].class_name == "helmet"  # mAP50_95=0.65 最高
        assert rankings[0].rank == 1
        assert rankings[-1].class_name == "person"  # mAP50_95=0.55 最低

    def test_rank_classes_sort_by_f1(self):
        from od_platform.model_eval.analyzer import rank_classes
        m = _mock_metrics()
        rankings = rank_classes(m, sort_by="f1")
        assert rankings[0].class_name == "helmet"  # f1=0.874

    def test_rank_classes_worst(self):
        from od_platform.model_eval.analyzer import rank_classes
        m = _mock_metrics()
        rankings = rank_classes(m, sort_by="mAP50_95", worst=True)
        assert rankings[0].class_name == "person"  # 最差

    def test_rank_classes_top_n(self):
        from od_platform.model_eval.analyzer import rank_classes
        m = _mock_metrics()
        rankings = rank_classes(m, sort_by="mAP50_95", top_n=2)
        assert len(rankings) == 2

    def test_rank_classes_empty(self):
        from od_platform.model_eval.analyzer import rank_classes
        m = _mock_metrics(per_class={})
        assert rank_classes(m) == []

    def test_diagnose_problem_classes(self):
        from od_platform.model_eval.analyzer import diagnose_problem_classes
        m = _mock_metrics()
        diag = diagnose_problem_classes(m, threshold=0.85)
        # helmet f1=0.874 > 0.85 good
        # person f1=0.749 < 0.85 → warning, > 0.425 → not critical
        # vest f1=0.821 < 0.85 → warning
        assert "person" in diag["warning"]
        assert "helmet" in diag["good"]

    def test_compare_per_class(self):
        from od_platform.model_eval.analyzer import compare_per_class
        ma = _mock_metrics("model_a")
        mb = _mock_metrics_b("model_b")
        diffs = compare_per_class(ma, mb, metric="mAP50_95")
        assert len(diffs) == 3
        # model_a 在所有类别上都更好
        assert all(d.winner == "A" for d in diffs)

    def test_compare_per_class_tie(self):
        from od_platform.model_eval.analyzer import compare_per_class
        ma = _mock_metrics("model_a")
        mb = _mock_metrics("model_b")  # 同指标
        diffs = compare_per_class(ma, mb, metric="mAP50_95", min_diff=10.0)
        assert all(d.winner == "tie" for d in diffs)

    def test_render_confusion_markdown_empty(self):
        from od_platform.model_eval.analyzer import render_confusion_markdown
        md = render_confusion_markdown([])
        assert "不可用" in md

    def test_render_ranking_markdown(self):
        from od_platform.model_eval.analyzer import rank_classes, render_ranking_markdown
        rankings = rank_classes(_mock_metrics(), sort_by="mAP50_95")
        md = render_ranking_markdown(rankings)
        assert "类别性能排序" in md
        assert "helmet" in md
        assert "person" in md

    def test_ranking_to_dict(self):
        from od_platform.model_eval.analyzer import rank_classes
        rankings = rank_classes(_mock_metrics(), sort_by="mAP50_95")
        d = rankings[0].to_dict()
        assert d["rank"] == 1
        assert d["class"] == "helmet"

    def test_per_class_diff_to_dict(self):
        from od_platform.model_eval.analyzer import compare_per_class
        diffs = compare_per_class(_mock_metrics("A"), _mock_metrics_b("B"))
        d = diffs[0].to_dict()
        assert "class" in d
        assert "winner" in d


# ============================================================
# history 测试
# ============================================================

class TestEvalHistory:
    """评估历史追踪测试."""

    def _report(self, model_name: str = "test_model", run_id: str = "20260723-120000") -> Any:
        from od_platform.model_eval.report import EvalReport
        return EvalReport(
            run_id=run_id, model_name=model_name,
            model_path=f"models/trained/{model_name}.pt",
            data_yaml="datasets/helmet.yaml",
            split="val", created_at="2026-07-23T12:00:00",
            metrics=_mock_metrics(model_name),
        )

    def test_record_from_report(self):
        from od_platform.model_eval.history import EvalRecord
        record = EvalRecord.from_report(self._report())
        assert record.model_name == "test_model"
        assert record.mAP50 == 0.8356

    def test_record_to_dict(self):
        from od_platform.model_eval.history import EvalRecord
        record = EvalRecord.from_report(self._report())
        d = record.to_dict()
        assert d["model_name"] == "test_model"
        # NaN -> None
        r_nan = EvalRecord.from_report(
            self._report().__class__(
                run_id="t", model_name="nan_model",
                model_path="p", data_yaml="d", split="val",
                created_at="2026-01-01T00:00:00",
                metrics=_mock_metrics(precision=math.nan),
            )
        )
        assert r_nan.to_dict()["precision"] is None

    def test_record_roundtrip(self):
        from od_platform.model_eval.history import EvalRecord
        original = EvalRecord.from_report(self._report())
        d = original.to_dict()
        restored = EvalRecord.from_dict(d)
        assert restored.model_name == original.model_name
        assert restored.mAP50 == original.mAP50

    def test_history_add_and_save(self):
        from od_platform.model_eval.history import EvalHistory
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "test_history.json"
            history = EvalHistory(data_yaml="datasets/helmet.yaml", _file_path=fp)
            history.add_record(self._report("model_v1", "run_001"))
            history.add_record(self._report("model_v2", "run_002"))
            assert len(history.records) == 2
            saved = history.save(fp)
            assert saved.exists()

    def test_history_load(self):
        from od_platform.model_eval.history import EvalHistory
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "test_history.json"
            h1 = EvalHistory(data_yaml="datasets/helmet.yaml", _file_path=fp)
            h1.add_record(self._report("model_v1", "run_001"))
            h1.save(fp)

            h2 = EvalHistory.load(fp)
            assert len(h2.records) == 1
            assert h2.records[0].model_name == "model_v1"

    def test_history_duplicate_run_id(self):
        from od_platform.model_eval.history import EvalHistory
        history = EvalHistory(data_yaml="d.yaml")
        history.add_record(self._report("m1", "run_001"))
        history.add_record(self._report("m2", "run_001"))  # 重复 run_id
        assert len(history.records) == 1  # 应跳过

    def test_history_best_record(self):
        from od_platform.model_eval.history import EvalHistory
        history = EvalHistory(data_yaml="d.yaml")
        history.add_record(self._report("m1", "run_001"))
        # m2 在 mAP50_95 上更好
        history.add_record(
            self._report("m2", "run_002").__class__(
                run_id="run_002", model_name="m2",
                model_path="p", data_yaml="d", split="val",
                created_at="2026-07-24T12:00:00",
                metrics=_mock_metrics("m2", mAP50_95=0.75),
            )
        )
        best = history.best_record(metric="mAP50_95")
        assert best is not None
        assert best.model_name == "m2"

    def test_trend_report(self):
        from od_platform.model_eval.history import EvalHistory
        history = EvalHistory(data_yaml="d.yaml")
        history.add_record(self._report("my_model", "run_001"))
        history.add_record(
            self._report("my_model", "run_002").__class__(
                run_id="run_002", model_name="my_model",
                model_path="p", data_yaml="d", split="val",
                created_at="2026-07-24T12:00:00",
                metrics=_mock_metrics("my_model", mAP50_95=0.70, f1=0.85),
            )
        )
        trend = history.trend("my_model")
        assert len(trend.records) == 2
        assert trend.improvement("mAP50_95") is not None
        assert trend.improvement("mAP50_95") > 0  # 改进了

    def test_trend_markdown(self):
        from od_platform.model_eval.history import EvalHistory
        history = EvalHistory(data_yaml="d.yaml")
        history.add_record(self._report("my_model", "run_001"))
        md = history.trend("my_model").render_markdown()
        assert "评估趋势" in md
        assert "my_model" in md

    def test_load_or_create(self):
        from od_platform.model_eval.history import EvalHistory
        with tempfile.TemporaryDirectory() as td:
            # 需要 patch _HISTORY_DIR 才能用临时目录, 这里只测创建路径
            history = EvalHistory(data_yaml="nonexistent_data.yaml")
            assert len(history.records) == 0


# ============================================================
# CLI 参数解析测试
# ============================================================

class TestCLI:
    """CLI 入口测试."""

    def test_parser_model_flag(self):
        from od_platform.cli.evaluate_model import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--model", "best.pt", "--data", "helmet"])
        assert args.model == "best.pt"
        assert args.models is None
        assert args.data == "helmet"

    def test_parser_models_flag(self):
        from od_platform.cli.evaluate_model import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--models", "a.pt", "b.pt", "--data", "helmet"])
        assert args.models == ["a.pt", "b.pt"]
        assert args.model is None

    def test_parser_mutually_exclusive(self):
        from od_platform.cli.evaluate_model import _build_parser
        parser = _build_parser()
        try:
            parser.parse_args(["--model", "a.pt", "--models", "b.pt", "c.pt"])
            assert False, "应拒绝同时指定 --model 和 --models"
        except SystemExit:
            pass  # argparse 互斥组拒绝 → SystemExit

    def test_parser_split_default(self):
        from od_platform.cli.evaluate_model import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--model", "best.pt"])
        assert args.split is None  # 走 val.yaml 默认

    def test_parser_split_override(self):
        from od_platform.cli.evaluate_model import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--model", "best.pt", "--split", "test"])
        assert args.split == "test"

    def test_collect_cli_overrides(self):
        from od_platform.cli.evaluate_model import _build_parser, _collect_cli_overrides
        parser = _build_parser()
        args = parser.parse_args(["--model", "best.pt", "--conf", "0.5", "--iou", "0.7"])
        overrides = _collect_cli_overrides(args)
        assert "model" not in overrides  # CLI 行为开关, 不进配置
        assert overrides["conf"] == 0.5
        assert overrides["iou"] == 0.7


# ============================================================
# 服务层测试 (不需要实际 GPU)
# ============================================================

class TestServiceHelpers:
    """Service 内部辅助函数测试."""

    def test_display_name(self):
        from od_platform.model_eval.service import _display_name
        assert _display_name("best.pt") == "best"
        assert _display_name("models/trained/my_model.pt") == "my_model"
        assert _display_name("yolo11n") == "yolo11n"

    def test_resolve_model_trained(self):
        # 这依赖实际文件系统, 只测函数存在
        from od_platform.model_eval.service import _resolve_model_arg
        # 不带路径分隔符的名字会被当作 trained 模型名
        result = _resolve_model_arg("nonexistent_model_for_test.pt")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_s_formatter(self):
        from od_platform.model_eval.service import _s
        assert _s(0.1234) == "0.1234"
        assert _s(math.nan) == "N/A"


# ============================================================
# _fmt 测试 (report.py 内部)
# ============================================================

class TestFormatting:
    """格式化辅助函数测试."""

    def test_fmt_normal(self):
        from od_platform.model_eval.report import _fmt
        assert _fmt(0.12345678) == "0.1235"
        assert _fmt(1.0) == "1.0000"

    def test_fmt_nan(self):
        from od_platform.model_eval.report import _fmt
        assert _fmt(math.nan) == "N/A"
        assert _fmt(None) == "N/A"

    def test_fmt_ms(self):
        from od_platform.model_eval.report import _fmt_ms
        assert "ms" in _fmt_ms(12.345)
        assert _fmt_ms(math.nan) == "N/A"


# ============================================================
# 极差分析测试
# ============================================================

class TestRangeAnalysis:
    """ComparisonReport 极差分析测试."""

    def test_metric_values(self):
        from od_platform.model_eval.report import ComparisonReport
        r = ComparisonReport(
            run_id="t", data_yaml="d.yaml", split="val",
            created_at="2026-01-01T00:00:00",
            models=[_mock_metrics("m1"), _mock_metrics_b("m2")],
        )
        vals = r._metric_values("mAP50_95")
        assert len(vals) == 2
        assert vals[0][1] > vals[1][1]  # m1 > m2

    def test_best_per_metric_in_dict(self):
        d = _mock_metrics().__class__.__name__  # just check it doesn't crash
        from od_platform.model_eval.report import ComparisonReport
        r = ComparisonReport(
            run_id="t", data_yaml="d.yaml", split="val",
            created_at="2026-01-01T00:00:00",
            models=[_mock_metrics("m1"), _mock_metrics_b("m2")],
        )
        analysis = r.to_dict()["analysis"]
        assert analysis["best_per_metric"]["mAP50_95"] == "m1"


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
