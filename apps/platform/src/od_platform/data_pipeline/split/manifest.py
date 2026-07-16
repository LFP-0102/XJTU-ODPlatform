from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime


TOOL_VERSION = "odp-datapipeline-0.1.0"

def _sha256_bytes(data: bytes) -> str:
    # 计算单字节流的sha256，用于给每个样本标签内容算独立指纹
    return hashlib.sha256(data).hexdigest()

@dataclass(frozen=True)
class SampleLineage:
    """一个样本的血缘"""
    stem: str  # 样本名
    split: str  # 划分名
    sha256: str  # 样本内容的sha256


def compute_contract_fingerprint(
    dataset: str,
    strategy: str,
    seed: int,
    ratios: Tuple[float, float, float ],
    names: List[str],
    samples: List[SampleLineage],
    tool_version: str = TOOL_VERSION,
) -> str:
    """计算合同/划分契约的指纹"""
    canonical = {
        "dataset": dataset,
        "strategy": strategy,
        "seed": seed,
        "ratios": ratios,
        "names": names,
        "samples": sorted((s.stem, s.split, s.sha256) for s in samples),
        "tool_version": tool_version,
    }
    blob = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass()
class SplitManifest:
    """划分契约：记录谁在哪一组，内容是什么，当初是怎么切的"""
    dataset: str  # 数据集名字
    strategy: str  # 划分策略 random,
    seed: int  # 随机种子
    rations: Tuple[float , float , float ]  # 划分比例
    names: List[str]  # 划分的类名清单
    samples: List[SampleLineage]  # 样本血缘清单
    contract_fingerprint: str = ""  # 划分契约的指纹
    created_at: str = ""  # 创建时间
    tool_version: str = TOOL_VERSION

    @classmethod
    def build(cls,
              dataset: str,
              strategy: str,
              seed: int,
              rations: Tuple[float , float , float],
              names: List[str],
              splits: Dict[str, List[str]],
              label_bytes: Dict[str, bytes],
              created_at: str = "",
              ) -> SplitManifest:
        # 需要逐个样本装配血缘，把在哪组，和内容哈希 订在同一个记录上
        samples: List[SampleLineage] = []
        for split, stems in splits.items():
            for stem in stems:
                samples.append(
                    SampleLineage(stem=stem, split=split, sha256=_sha256_bytes(label_bytes[stem]))
                )
        samples.sort(key=lambda s: s.stem)

        # 算划分契约的指纹
        fp = compute_contract_fingerprint(dataset, strategy, seed, rations, names, samples)

        return cls(
            dataset=dataset,
            strategy=strategy,
            seed=seed,
            rations=tuple(rations),
            names=list(names),
            samples=samples,
            contract_fingerprint=fp,
            created_at=created_at or datetime.now().isoformat(timespec="seconds")
        )

    # 便捷视图
    def stems_of(self, split: str) -> List[str]:
        return [s.stem for s in self.samples if s.split == split]

    def by_stem(self) -> Dict[str, SampleLineage]:
        return {s.stem: s for s in self.samples}

    # 落盘与读回
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding='utf-8')

    @classmethod
    def read(cls, path: Path) -> "SplitManifest":
        d = json.loads(path.read_text(encoding="utf-8"))
        d['rations'] = tuple(d['rations'])
        d['samples'] = [SampleLineage(**s) for s in d['samples']]
        return cls(**d)












