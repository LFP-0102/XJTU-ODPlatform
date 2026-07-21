#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :lineage.py
# @Time      :2026/7/17 14:17:20
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :划分血缘契约
# apps/platform/src/od_platform/common/lineage.py
"""划分血缘契约:把一次数据划分冻结成可复现、可审计、可逐样本核对的产物。

它是"生产一次划分"和"核验一次划分"两方共用的契约,所以住在公共层 common/——
生产方写 manifest 用它、验证方读 manifest 核对也用它,两边都只向下依赖这里,谁也不 import 谁。

一份 manifest 回答四个问题:
  1. 每张图去了哪一组?              → 每条 SampleLineage.split
  2. 每张图的内容有没有被动过?      → 每条 SampleLineage.sha256   ← 逐样本指纹
  3. 整份数据集是不是同一版?        → contract_fingerprint         ← 划分契约指纹
  4. 用哪个版本的流水线切的?        → tool_version
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# 工具版本号。半年后有人问"这数据是用哪个版本的流水线切的",看这里。
TOOL_VERSION = "odp-transform/0.4"


def _sha256_bytes(data: bytes) -> str:
    """计算单段字节流的 sha256。用于给每个样本的标签内容算独立指纹。"""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class SampleLineage:
    """一个样本的血缘原子: 它在哪一组、它的内容指纹是什么。

    把'去了哪组(split)'和'内容指纹(sha256)'钉在同一条记录上——核对时对每个文件
    一次查表就能同时问两句:
        · 你在你该在的那一组吗?   (split)
        · 你的内容还是当初那份吗?  (sha256)

    frozen=True: 一条血缘一旦装配就不许改——它是"当时的事实",不是可变状态。
    """

    stem: str      # 样本名(不含扩展名,如 'c_001')。路径是环境的奴隶,stem 是数据的灵魂。
    split: str     # "train" / "val" / "test" —— 它被划到哪一组。
    sha256: str    # 该样本标签内容的 sha256。


def compute_contract_fingerprint(
    dataset: str, strategy: str, seed: int, rations: Tuple[float, float, float],
    names: List[str], samples: List[SampleLineage], tool_version: str = TOOL_VERSION,
) -> str:
    """对'定义了这次划分的一切'做规范化哈希 = 划分契约指纹。

    进指纹的东西分两类,少一类都能被绕过:
    · '怎么切的'(recipe): dataset / strategy / seed / rations / names / tool_version
    · '切成了什么'(结果): 每个样本的 (stem, split, sha256) 三元组

    规范化是可复现的前提(否则字典/列表顺序一变,指纹就变):
    1. 每个样本压成 (stem, split, sha256) 元组,整份 samples 按元组排序;
    2. json.dumps 开 sort_keys=True + 紧凑 separators。
    把一切不确定性拍平,指纹才真正确定。

    tool_version 进指纹、created_at 不进 —— 判据只有一句:'它会不会改变数据本身'。
    同一份 raw、同一个 seed,被 v0.3 和 v0.4 的流水线切,切法可能已经变了,
    所以版本号必须进;而 created_at 只是'什么时候切的',同输入两次跑时间戳必不同、
    内容却一模一样,它进了指纹就会让'同输入同指纹'当场失效。
    """
    canonical = {
        "dataset": dataset,
        "strategy": strategy,
        "seed": seed,
        "rations": list(rations),
        "names": list(names),
        # 三元组整体排序: 顺序无关,内容(在哪组 + 什么哈希)决定指纹
        "samples": sorted((s.stem, s.split, s.sha256) for s in samples),
        "tool_version": tool_version,
    }
    blob = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class SplitManifest:
    """划分契约: 记录'谁在哪一组、内容是什么'以及'当初是怎么切的'。"""

    # —— recipe: 当初是怎么切的 ——
    dataset: str                        # 数据集名(如 "demo")
    strategy: str                       # 划分策略(如 "random" / "stratified_multilabel")
    seed: int                           # 随机种子(可复现的命门)
    rations: Tuple[float, float, float]  # 切分比例 (train, val, test)
    names: List[str]                    # 类名清单(对应 yaml 里的 names)

    # —— 血缘: 逐样本 {去哪组 + 内容指纹} ——
    samples: List[SampleLineage]

    # —— 划分契约指纹: 整份划分的身份 ——
    contract_fingerprint: str = ""

    # —— 时间与版本 ——
    created_at: str = ""                # ISO 格式时间戳(不进指纹)
    tool_version: str = TOOL_VERSION

    @classmethod
    def build(
        cls, dataset: str, strategy: str, seed: int, rations: Tuple[float, float, float],
        names: List[str], splits: Dict[str, List[str]], label_bytes: Dict[str, bytes],
        created_at: str = "",           # 由编排器注入(与 run_id / 现场目录名同一时间戳);留空则本地 now()
    ) -> "SplitManifest":
        """唯一的装配入口: 从'谁在哪组'(splits) + '标签原始字节'(label_bytes) 装配 manifest。

        指纹不是外面算好塞进来的,是 build() 自己对着原始字节算出来的——这就杜绝了
        '伪造一份 manifest 说指纹是对的'这种作弊: 指纹从内容里长出来,不是贴上去的。
        """
        # 1. 逐样本装配血缘: 把'在哪组'和'内容哈希'钉在同一条记录上
        samples: List[SampleLineage] = []
        for split, stems in splits.items():
            for stem in stems:
                samples.append(
                    SampleLineage(stem=stem, split=split, sha256=_sha256_bytes(label_bytes[stem]))
                )
        samples.sort(key=lambda s: s.stem)  # 复现: 落盘顺序确定

        # 2. 算划分契约指纹(recipe + 全体逐样本三元组)
        fp = compute_contract_fingerprint(dataset, strategy, seed, rations, names, samples)

        # 3. 组装并返回
        return cls(
            dataset=dataset, strategy=strategy, seed=seed, rations=tuple(rations),
            names=list(names), samples=samples, contract_fingerprint=fp,
            created_at=created_at or datetime.now().isoformat(timespec="seconds"),
        )

    # ---------- 便捷视图: 让两类消费者各取所需,而底层只有一份真相 ----------

    def stems_of(self, split: str) -> List[str]:
        """某一组里有哪些 stem —— 给落盘方用(它只关心'该复制哪些文件到哪个组')。

        splits 这个'按组分桶'的视图,从逐样本血缘里派生,不再单独存一份(派生属性永不存值)。
        """
        return [s.stem for s in self.samples if s.split == split]

    def by_stem(self) -> Dict[str, SampleLineage]:
        """按 stem 建索引 —— 给核验方用(扫盘时对每个文件一次查到它的 {split, sha256})。"""
        return {s.stem: s for s in self.samples}

    # ---------- 落盘与读回 ----------

    def to_json(self) -> str:
        """序列化为 JSON 字符串(带缩进,方便人类审查)。asdict 会把 samples 递归成 list[dict]。"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def write(self, path: Path) -> None:
        """物理冻结: 写入磁盘。逐样本血缘(含 split)全量落盘。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "SplitManifest":
        """从磁盘解冻: 读回内存。samples 的每条 dict 还原成 SampleLineage。"""
        d = json.loads(path.read_text(encoding="utf-8"))
        d["rations"] = tuple(d["rations"])                       # JSON 把 tuple 读成 list,扳回来
        d["samples"] = [SampleLineage(**s) for s in d["samples"]]
        return cls(**d)
