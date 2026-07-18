from od_platform.runtime_config.base_config import BaseConfig

# 1. 默认配置
c = BaseConfig()
print(c.model_dump_json(indent=2))

# 2. 验证错误
try:
    BaseConfig(batch=0)
except Exception as e:
    print(f"✓ 拒绝 batch=0: {e}")

try:
    BaseConfig(batch=True)
except Exception as e:
    print(f"✓ 拒绝 batch=bool: {e}")

try:
    BaseConfig(cache="foobar")
except Exception as e:
    print(f"✓ 拒绝 cache='foobar': {e}")

# 3. 警告
import warnings
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    BaseConfig(imgsz=649)
    assert any("不是 32 的倍数" in str(w.message) for w in ws)
    print("✓ imgsz=649 触发警告")

# 4. 字段元数据
batch_meta = c.get_field_metadata("batch")
print(f"✓ batch group: {batch_meta['group']}")
print(f"✓ batch tips[0]: {batch_meta['tips'][0]}")

# 5. 字段分组
groups = c.get_field_groups()
for g, fields in groups.items():
    print(f"  {g}: {len(fields)} 个字段")

# 6. 转 ultralytics kwargs
kwargs = c.to_ultralytics_kwargs()
assert "verbose" not in kwargs
print(f"✓ kwargs 不含 verbose, 共 {len(kwargs)} 个参数")

# 7. extra="forbid": 拒绝拼错的字段名
try:
    BaseConfig(epchs=300)          # epochs 拼错成 epchs
except Exception as e:
    print(f"✓ 拒绝未知字段 epchs: {type(e).__name__}")

# 8. mode="before" 让 bool 守卫真正生效
try:
    BaseConfig(batch=True)         # 必须报 batch 不能是 bool, 而不是静默变成 1
except Exception as e:
    assert "bool" in str(e)
    print("✓ batch=True 被 bool 守卫拦下(未静默变成 1)")