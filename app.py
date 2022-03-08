from sahi.model import Yolov5DetectionModel
from sahi.predict import get_prediction
from sahi.utils.cv import read_image
from sahi.utils.yolov5 import download_yolov5s6_model

yolov5_model_path = 'models/yolov5s6.pt'
download_yolov5s6_model(destination_path=yolov5_model_path)

detection_model = Yolov5DetectionModel(
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)

result = get_prediction(read_image("demo_data/highway.jpg"), detection_model)
result.export_visuals(export_dir="demo_data/")

# TODO: Opencv kütüphanesini kullanarak show fonksiyonu yazılacak.
