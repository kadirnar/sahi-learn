from sahi_lite.model import Yolov5DetectionModel
from sahi_lite.predict import get_prediction, get_sliced_prediction
from sahi_lite.utils.cv import read_image
from sahi_lite.utils.yolov5 import download_yolov5s6_model

yolov5_model_path = 'models/yolov5s6.pt'
download_yolov5s6_model(destination_path=yolov5_model_path)

detection_model = Yolov5DetectionModel(
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)

result = get_prediction(read_image("demo_data/highway.jpg"), detection_model)
# result.export_visuals(export_dir="demo_data/")

result = get_sliced_prediction(
    "demo_data/highway.jpg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

result.export_visuals(export_dir="demo_data/")
