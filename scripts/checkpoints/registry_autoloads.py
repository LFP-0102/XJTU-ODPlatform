"""阶段 2 Checkpoint:走 service(自动发现生效)+ 三态白名单。"""
from pathlib import Path


from od_platform.data_pipeline.convert.service import convert_data_to_yolo

names = convert_data_to_yolo(
    input_dir=Path(r"C:\Users\刘赋平\Desktop\XJTU-ODPlatform\data\raw\MRI_PASCAL\annotations"),
    output_labels_dir=Path(r"C:\Users\刘赋平\Desktop\XJTU-ODPlatform\data\processed\pascal_voc"),
    annotation_format = "pascal_voc"
)
print(names)