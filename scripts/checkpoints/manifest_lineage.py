# scripts/checkpoints/manifest_lineage.py
"""阶段 5 Checkpoint:冻结 / 读回 / 确定性 / 逐样本核对(位置 + 内容)。"""
import tempfile, time
from pathlib import Path
from od_platform.data_pipeline.split.manifest import SplitManifest

# —— 模拟一次划分的结果(谁在哪组)与标签原始字节 ——
splits = {"train": ["c_001", "c_002", "rare_000"], "val": ["c_003"], "test": ["c_004", "rare_001"]}
label_bytes = {
    "c_001": b"0 0.5 0.5 0.2 0.2\n", "c_002": b"0 0.6 0.6 0.1 0.1\n",
    "rare_000": b"0 0.1 0.1 0.05 0.05\n1 0.8 0.8 0.1 0.1\n", "c_003": b"0 0.2 0.2 0.1 0.1\n",
    "c_004": b"0 0.9 0.9 0.05 0.05\n", "rare_001": b"0 0.3 0.3 0.1 0.1\n1 0.4 0.4 0.2 0.2\n",
}
COMMON = dict(dataset="demo", strategy="random", seed=42, rations=(0.7, 0.15, 0.15), names=["head", "反光衣"])

m = SplitManifest.build(**COMMON, splits=splits, label_bytes=label_bytes)
idx = m.by_stem()

print("契约指纹:", m.contract_fingerprint)
for s in m.samples:
    print(f"  {s.stem:<9} split={s.split:<5} sha256={s.sha256[:16]}…")

# 落盘→读回 + 确定性
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "manifest.json"; m.write(p); back = SplitManifest.read(p)
    print("读回相等?", m.contract_fingerprint == back.contract_fingerprint, m.samples == back.samples)
time.sleep(1); m2 = SplitManifest.build(**COMMON, splits=splits, label_bytes=label_bytes)
print("时间戳变/指纹不变?", m.created_at != m2.created_at, m.contract_fingerprint == m2.contract_fingerprint)

# 作案A:改内容
la = dict(label_bytes); la["c_001"] = b"0 0.5 0.7 0.2 0.3\n"
m_a = SplitManifest.build(**COMMON, splits=splits, label_bytes=la)
print("改内容→逐样本变/契约变?", idx["c_001"].sha256 == m_a.by_stem()["c_001"].sha256,
      m.contract_fingerprint != m_a.contract_fingerprint)
# 作案B:挪组(内容一字未动)
sb = {k: list(v) for k, v in splits.items()}; sb["train"].remove("rare_000"); sb["test"].append("rare_000")
m_b = SplitManifest.build(**COMMON, splits=sb, label_bytes=label_bytes); rb = m_b.by_stem()["rare_000"]
print("挪组→内容哈希不变、split 变、契约变?",
      idx["rare_000"].sha256 == rb.sha256, f"{idx['rare_000'].split}->{rb.split}",
      m.contract_fingerprint != m_b.contract_fingerprint)