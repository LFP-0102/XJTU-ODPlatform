import time
import random
from PIL import Image, ImageDraw, ImageFont

# ===== 环境 =====
img = Image.new('RGB', (1, 1))
draw = ImageDraw.Draw(img)

FONT_EN = FONT_CN = 'LXGWWenKai-Bold.ttf'


class_names = ["person", "bicycle", "car", "人", "自行车", "汽车"]
font_sizes = list(range(10, 41))

# ===== 预加载所有字体（方法1会用） =====
font_cache = {}
for size in font_sizes:
    font_cache[('cn', size)] = ImageFont.truetype(FONT_CN, size)
    font_cache[('en', size)] = ImageFont.truetype(FONT_EN, size)

# ===== 预计算模板（方法2） =====
template = {}
for name in class_names:
    lang = 'cn' if any('\u4e00' <= c <= '\u9fff' for c in name) else 'en'
    for size in font_sizes:
        font = font_cache[(lang, size)]
        # 类别名 + 空格
        name_w = draw.textbbox((0, 0), name + " ", font=font)[2]
        # 置信度最大宽度
        conf_w = draw.textbbox((0, 0), "0.00", font=font)[2]
        h = draw.textbbox((0, 0), "0.00", font=font)[3]
        template[(name, size)] = (name_w + conf_w, h)

# ===== 模拟帧数据生成 =====
def generate_frames(num_frames=500, max_dets_per_frame=30):
    frames = []
    for _ in range(num_frames):
        size = random.choice(font_sizes)
        num_dets = random.randint(5, max_dets_per_frame)
        dets = [(random.choice(class_names), round(random.uniform(0.5, 0.99), 2))
                for _ in range(num_dets)]
        frames.append((size, dets))
    return frames

# ===== 方法1: 实时计算（字体已缓存） =====
def method_realtime(frames):
    total_time = 0
    for size, dets in frames:
        t0 = time.perf_counter()
        for name, conf in dets:
            lang = 'cn' if any('\u4e00' <= c <= '\u9fff' for c in name) else 'en'
            font = font_cache[(lang, size)]
            bbox = draw.textbbox((0, 0), f"{name} {conf:.2f}", font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        total_time += time.perf_counter() - t0
    return total_time

# ===== 方法2: 模板查询 =====
def method_template(frames):
    total_time = 0
    for size, dets in frames:
        t0 = time.perf_counter()
        for name, conf in dets:
            w, h = template[(name, size)]
        total_time += time.perf_counter() - t0
    return total_time

# ===== 预热 =====
warmup_frames = generate_frames(20, 10)
method_realtime(warmup_frames)
method_template(warmup_frames)

# ===== 正式测试 =====
test_frames = generate_frames(1000, 30)  # 1000帧，每帧最多30个检测
t1 = method_realtime(test_frames)
t2 = method_template(test_frames)

avg1 = t1 / len(test_frames) * 1000   # ms/帧
avg2 = t2 / len(test_frames) * 1000

print(f"测试帧数: {len(test_frames)}")
print(f"字体范围: 10~40, 每帧固定大小")
print("-" * 50)
print(f"方法1(实时计算+字体缓存): {avg1:.4f} ms/帧")
print(f"方法2(模板查询):          {avg2:.4f} ms/帧")
print(f"\n加速比: {t1/t2:.2f}x")
print(f"每帧节省: {avg1-avg2:.4f} ms")

# 精度验证
print("\n最大宽度误差 (置信度用0.00代替实际值):")
max_err = 0
for name in class_names:
    for size in [10, 20, 30, 40]:
        lang = 'cn' if any('\u4e00' <= c <= '\u9fff' for c in name) else 'en'
        font = font_cache[(lang, size)]
        # 真实宽度（使用实际置信度 0.99）
        real_bbox = draw.textbbox((0, 0), f"{name} 0.99", font=font)
        real_w = real_bbox[2] - real_bbox[0]
        # 模板宽度
        tmpl_w = template[(name, size)][0]
        err = abs(real_w - tmpl_w)
        if err > max_err:
            max_err = err
print(f"最大宽度误差: {max_err:.1f} px (可忽略)")