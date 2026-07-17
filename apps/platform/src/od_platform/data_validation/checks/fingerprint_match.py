"""check: fingerprint_match —— 审计第一问'验的是不是同一份数据',用契约指纹回答。"""
from __future__ import annotations

# 契约算法住在 common/lineage(生产/核验两方共用的公共设施),不 import data_pipeline。
from od_platform.common.lineage import compute_contract_fingerprint
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)
from od_platform.data_validation.checks._lineage import load_manifest_from_yaml

_NAME = "fingerprint_match"


@check(_NAME)
def validate_fingerprint_match(ctx: CheckContext) -> CheckResult:
    manifest, err, odp = load_manifest_from_yaml(ctx.snapshot.yaml_data)
    if err:   # 账本读不回来:连"验的是哪份"都答不了
        return CheckResult(_NAME, CheckSeverity.ERROR,
                        f"拿不到划分契约,无法确认验的是不是同一份数据 —— {err}",
                        {"reason": err})

    # 第一问 · 自洽:从 samples 重算,应等于账本自己记的
    recomputed = compute_contract_fingerprint(
        manifest.dataset, manifest.strategy, manifest.seed, manifest.rations,
        manifest.names, manifest.samples, manifest.tool_version,
    )
    if recomputed != manifest.contract_fingerprint:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                        "manifest 自身指纹对不上 —— 文件被手改过或已损坏",
                        {"recorded": manifest.contract_fingerprint, "recomputed": recomputed})

    # 第二问 · 同源:yaml 盖的指纹,应等于账本的指纹
    yaml_fp = odp.get("contract_fingerprint")
    if yaml_fp != manifest.contract_fingerprint:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                        "yaml 指纹与 manifest 不一致 —— 这份 yaml 指向的不是这份划分",
                        {"yaml_fingerprint": yaml_fp,
                            "manifest_fingerprint": manifest.contract_fingerprint})

    return CheckResult(_NAME, CheckSeverity.PASS,
                    f"指纹一致,确系同一份划分（{manifest.contract_fingerprint[:12]}…）",
                    {"contract_fingerprint": manifest.contract_fingerprint,
                        "n_samples": len(manifest.samples),
                        "strategy": manifest.strategy, "seed": manifest.seed})