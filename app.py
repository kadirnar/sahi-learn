import gradio as gr
from sahi_lite.utils.cv import visualize_object_predictions
from sahi_lite.model import Yolov5DetectionModel
from sahi_lite.predict import get_prediction, get_sliced_prediction
from sahi_lite.slicing import get_slice_bboxes
from PIL import Image
import numpy

IMAGE_SIZE = 640

# Model
model = Yolov5DetectionModel(
    model_path="yolov5s6.pt", device="cpu", confidence_threshold=0.5, image_size=IMAGE_SIZE
)


def sahi_yolo_inference(
        image,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_match_threshold=0.5,
):
    image_width, image_height = image.size
    sliced_bboxes = get_slice_bboxes(
        image_height,
        image_width,
        slice_height,
        slice_width,
        overlap_height_ratio,
        overlap_width_ratio,
    )
    if len(sliced_bboxes) > 60:
        raise ValueError(
            f"{len(sliced_bboxes)} slices are too much for huggingface spaces, try smaller slice size."
        )

    # standard inference
    prediction_result_1 = get_prediction(
        image=image, detection_model=model
    )
    visual_result_1 = visualize_object_predictions(
        image=numpy.array(image),
        object_prediction_list=prediction_result_1.object_prediction_list,
    )
    output_1 = Image.fromarray(visual_result_1["image"])

    # sliced inference
    prediction_result_2 = get_sliced_prediction(
        image=image,
        detection_model=model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        postprocess_match_threshold=postprocess_match_threshold,
    )
    visual_result_2 = visualize_object_predictions(
        image=numpy.array(image),
        object_prediction_list=prediction_result_2.object_prediction_list,
    )

    output_2 = Image.fromarray(visual_result_2["image"])

    return output_1, output_2


inputs = [
    gr.inputs.Image(type="pil", label="Original Image"),
    gr.inputs.Number(default=512, label="slice_height"),
    gr.inputs.Number(default=512, label="slice_width"),
    gr.inputs.Number(default=0.2, label="overlap_height_ratio"),
    gr.inputs.Number(default=0.2, label="overlap_width_ratio"),
    gr.inputs.Number(default=0.5, label="postprocess_match_threshold"),
]

outputs = [
    gr.outputs.Image(type="pil", label="YOLOv5s"),
    gr.outputs.Image(type="pil", label="YOLOv5s + SAHI"),
]

title = "Small Object Detection with SAHI + YOLOv5"
article = "<p style='text-align: center'>SAHI is a lightweight vision library for performing large scale object " \
          "detection/ instance segmentation.. <a href='https://github.com/obss/sahi'>SAHI Github</a> | <a " \
          "href='https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images" \
          "-small-objects-c8b086af3b80'>SAHI Blog</a> | <a href='https://github.com/fcakyon/yolov5-pip'>YOLOv5 " \
          "Github</a> </p> "

gr.Interface(
    sahi_yolo_inference,
    inputs,
    outputs,
    title=title,
    article=article,
    theme="huggingface",
).launch(debug=True, enable_queue=True)
