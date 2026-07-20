import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ==================== 配置区（只需修改这里）====================
# 1. 转换格式选择：'coco' 或 'voc' (不区分大小写)
FORMAT = "voc"

# 2. 路径配置
# 输入数据集根目录（必须包含 annotations 和 images 文件夹）
INPUT_ROOT = r"C:\Users\Matri\Desktop\XJTU-ODPlatfrom\data\raw\MRI_PASCAL"
# 输出 YOLO 数据集根目录
OUTPUT_ROOT = r"C:\Users\Matri\Desktop\XJTU-ODPlatfrom\data\processed\voc"

# 3. 类别名称列表（顺序决定 class_id，从 0 开始）
# 提示：COCO 格式如果留空 []，脚本会自动从 JSON 提取并过滤背景类；VOC 格式建议手动指定。
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]

# 4. 数据集划分比例 (train : val : test)
SPLIT_RATIO = (0.7, 0.2, 0.1)
# 5. 随机种子
RANDOM_SEED = 42


# ================================================================


# ==================== 公共工具函数 ====================
def generate_yaml(output_root, class_names):
    """生成 YOLO 的 data.yaml 配置文件"""
    yaml_path = Path(output_root) / "data.yaml"
    yaml_content = f"""# YOLO 数据集配置文件
# 自动生成于统一转换脚本 (Format: {FORMAT.upper()})
path: {Path(output_root).resolve().as_posix()}   # 数据集根目录
train: train/images
val: val/images
test: test/images   # 如果没有测试集，可删除这一行
nc: {len(class_names)}
names: {class_names}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"📄 配置文件已生成：{yaml_path}")


# ==================== VOC 转换逻辑 ====================
def parse_voc_xml(xml_path, class_to_id):
    """解析单个 VOC XML 文件，返回 YOLO 格式标注行"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)
    lines = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in class_to_id:
            continue
        cls_id = class_to_id[name]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # 转为 YOLO 归一化坐标
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # 防止越界
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        width = min(max(width, 0), 1)
        height = min(max(height, 0), 1)

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


def convert_voc_to_yolo(input_root, output_root, class_names, split_ratio, seed):
    """VOC 转 YOLO 主逻辑"""
    random.seed(seed)
    class_to_id = {name: i for i, name in enumerate(class_names)}

    ann_dir = Path(input_root) / "annotations"
    img_dir = Path(input_root) / "images"

    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 {ann_dir} 或 {img_dir}，请检查路径")

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"❌ {ann_dir} 中没有 .xml 文件")

    print(f"🔍 [VOC] 找到 {len(xml_files)} 个标注文件，开始处理...")

    pairs = []
    for xml_path in xml_files:
        # 自动匹配常见图片后缀
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
            img_path = img_dir / (xml_path.stem + ext)
            if img_path.exists():
                pairs.append((xml_path, img_path))
                break

    if not pairs:
        raise ValueError("❌ 没有可用的标注-图片对，请检查文件名是否一致")

    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    splits = {"train": pairs[:train_end], "val": pairs[train_end:val_end], "test": pairs[val_end:]}

    for split_name, split_pairs in splits.items():
        if not split_pairs: continue
        img_out = Path(output_root) / split_name / "images"
        lbl_out = Path(output_root) / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for xml_path, img_path in split_pairs:
            dest_img = img_out / img_path.name
            if img_path.resolve() != dest_img.resolve():
                shutil.copy2(img_path, dest_img)

            yolo_lines = parse_voc_xml(xml_path, class_to_id)
            dest_label = lbl_out / (xml_path.stem + ".txt")
            with open(dest_label, "w") as f:
                f.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

        print(f"✅ [VOC] {split_name}: 成功处理 {len(split_pairs)} 张图片")

    generate_yaml(output_root, class_names)


# ==================== COCO 转换逻辑 ====================
def load_coco_data(ann_dir):
    """读取并合并 annotations 目录下的所有 COCO JSON 文件（防 ID 冲突）"""
    all_images = {}
    all_annotations = {}
    coco_categories = {}

    json_files = list(ann_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"❌ 在 {ann_dir} 中未找到任何 .json 文件")

    print(f"🔍 [COCO] 发现 {len(json_files)} 个 JSON 文件，正在解析合并...")

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for cat in data.get("categories", []):
            coco_categories[cat["id"]] = cat["name"]

        # 使用 file_name 作为唯一键，防止不同 JSON 中 image_id 重复覆盖
        local_id_to_name = {}
        for img in data.get("images", []):
            file_name = img["file_name"]
            local_id_to_name[img["id"]] = file_name
            all_images[file_name] = img

        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in local_id_to_name:
                file_name = local_id_to_name[img_id]
                if file_name not in all_annotations:
                    all_annotations[file_name] = []
                all_annotations[file_name].append(ann)

    return all_images, all_annotations, coco_categories


def convert_coco_to_yolo(input_root, output_root, class_names, split_ratio, seed):
    """COCO 转 YOLO 主逻辑"""
    random.seed(seed)

    ann_dir = Path(input_root) / "annotations"
    img_dir = Path(input_root) / "images"

    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 {ann_dir} 或 {img_dir}，请检查路径")

    all_images, all_annotations, coco_categories = load_coco_data(ann_dir)

    # 自动推断类别（如果 CLASS_NAMES 为空）
    if not class_names:
        exclude_names = {"background", "objects", "_background", "none"}
        names = set()
        for cid, cname in coco_categories.items():
            if cname.lower() not in exclude_names:
                names.add(cname)
        final_classes = sorted(list(names))
    else:
        final_classes = class_names

    print(f"✅ [COCO] 目标类别映射: { {name: i for i, name in enumerate(final_classes)} }")

    # 建立 COCO ID 到 YOLO ID 的映射
    coco_to_yolo = {}
    for coco_id, coco_name in coco_categories.items():
        if coco_name in final_classes:
            coco_to_yolo[coco_id] = final_classes.index(coco_name)

    pairs = []
    for file_name, img_info in all_images.items():
        img_path = img_dir / file_name
        if not img_path.exists():
            img_path = img_dir / Path(file_name).name  # 兼容去子目录
            if not img_path.exists():
                continue

        annotations = all_annotations.get(file_name, [])
        pairs.append((img_info, annotations, img_path))

    if not pairs:
        raise ValueError("❌ 没有可用的图片数据，请检查路径和 JSON 内容")

    print(f"📊 [COCO] 共找到 {len(pairs)} 张有效图片，开始划分数据集...")
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    splits = {"train": pairs[:train_end], "val": pairs[train_end:val_end], "test": pairs[val_end:]}

    for split_name, split_pairs in splits.items():
        if not split_pairs: continue
        img_out = Path(output_root) / split_name / "images"
        lbl_out = Path(output_root) / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_info, annotations, img_path in split_pairs:
            dest_img = img_out / img_path.name
            if img_path.resolve() != dest_img.resolve():
                shutil.copy2(img_path, dest_img)

            img_w = float(img_info.get("width", 1))
            img_h = float(img_info.get("height", 1))

            yolo_lines = []
            for ann in annotations:
                coco_cat_id = ann["category_id"]
                if coco_cat_id not in coco_to_yolo:
                    continue

                yolo_cls_id = coco_to_yolo[coco_cat_id]
                bbox = ann["bbox"]  # [x_min, y_min, width, height]
                x_min, y_min, bw, bh = bbox

                # 转为 YOLO 归一化坐标
                x_center = (x_min + bw / 2) / img_w
                y_center = (y_min + bh / 2) / img_h
                w = bw / img_w
                h = bh / img_h

                # 防止越界
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                yolo_lines.append(f"{yolo_cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            dest_lbl = lbl_out / (img_path.stem + ".txt")
            with open(dest_lbl, "w", encoding="utf-8") as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines) + "\n")

        print(f"✅ [COCO] {split_name}: 成功处理 {len(split_pairs)} 张图片")

    generate_yaml(output_root, final_classes)


# ==================== 主函数入口 ====================
def main():
    print(f"🚀 开始执行 [{FORMAT.upper()}] 格式到 YOLO 的转换...")
    print("=" * 50)

    try:
        if FORMAT.lower() == "voc":
            convert_voc_to_yolo(INPUT_ROOT, OUTPUT_ROOT, CLASS_NAMES, SPLIT_RATIO, RANDOM_SEED)
        elif FORMAT.lower() == "coco":
            convert_coco_to_yolo(INPUT_ROOT, OUTPUT_ROOT, CLASS_NAMES, SPLIT_RATIO, RANDOM_SEED)
        else:
            print(f"❌ 错误：不支持的格式 '{FORMAT}'。请在配置区选择 'voc' 或 'coco'。")
            return
    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")
        return

    print("=" * 50)
    print(f"🎉 转换完成！YOLO 数据集已保存至：{Path(OUTPUT_ROOT).resolve()}")


if __name__ == "__main__":
    main()