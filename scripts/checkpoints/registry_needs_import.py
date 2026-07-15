from pathlib import Path
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
from od_platform.data_pipeline.convert.registry import ConvertOptions, get_converter, list_converters
from od_platform.common.constants import Task


print(list_converters())
for f in list_converters():
    print(get_converter(f).supported_tasks)

names = convert_data_to_yolo(
    input_dir=Path(r"C:\Users\刘赋平\Desktop\XJTU-ODPlatfrom\data\raw\MRI\annotations"),
    output_labels_dir=Path(r"C:\Users\刘赋平\Desktop\XJTU-ODPlatfrom\data\processed\coco"),
    annotation_format = "coco",
    options=ConvertOptions(task=Task.SEGMENT,classes=[ 'meningioma_tumor', 'pituitary_tumor'])
)
print(names)