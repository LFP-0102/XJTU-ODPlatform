from PIL import Image, ImageDraw, ImageFont


def calculate_text_size(text, font_path=None, font_size=40):
    """
    计算文字宽高

    参数:
        text: 文字内容
        font_path: 字体文件路径（可选，不提供则用默认字体）
        font_size: 字号大小
    """
    # 1. 准备文字内容
    text = str(text)

    # 2. 加载字体
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # 3. 创建绘图对象（需要临时画布）
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)

    # 4. 计算文字边界框
    bbox = draw.textbbox((0, 0), text, font=font)

    # 5. 提取宽高
    width = bbox[2] - bbox[0]  # 右边界 - 左边界
    height = bbox[3] - bbox[1]  # 下边界 - 上边界

    return width, height


# ============ 使用示例 ============

# 示例1: 使用系统字体
width, height = calculate_text_size(
    text="Hello 世界！",
    font_path="./LXGWWenKai-Bold.ttf",  # macOS示例
    font_size=48
)
print(f"文字尺寸: {width:.1f} x {height:.1f} 像素")

# 示例2: 使用默认字体
width, height = calculate_text_size(
    text="Test 123",
    font_size=40
)
print(f"文字尺寸: {width:.1f} x {height:.1f} 像素")