<div align="center">
<h1>
  SAHI-LiTE: SAHI'den Beraber Kodlamak İster Misiniz
</h1>
<h4>
    <img width="700" alt="teaser" src="obss.png">
</h4>

</div>



Herkese merhabalar ben Kadir Nar. SAHI kütüphanesine gönül vermiş bir geliştiriciyim. 
Bu repo da sizlere model.py dosyasını anlatacağım. Hadi başlayalım :) 


### 1. Class Yapısı Oluştur
Class ismini oluştururkan model isminin yanına DetectionModel(Detection) yazıyoruz.

##### Örnekler:
1.1 Mmdet:

```
class MmdetDetectionModel(DetectionModel)
```
1.2 Yolov5:
```
class Yolov5DetectionModel(DetectionModel):
```
1.3 Detectron2:
```
class Detectron2DetectionModel(DetectionModel)
```
1.4 TorchVision:
```
class TorchVisionDetectionModel(DetectionModel)
```

2. load_model(): Bu fonksyion 3 aşamadan oluşmaktadır.

a. Kütüphaneyi yüklüyoruz. PYPI desteği olmayan kütüphanelerin kurulumunu desteklemiyoruz.

b. Modele girecek resimlerin image_size değerlerini güncelliyoruz.

c. category_mapping değişkenini {"1": "pedestrian"} bu formatta olacak şekilde yazıyoruz.

##### Örnekler:

2.1 Mmdet:

```
def load_model(self):
    """
    Detection model is initialized and set to self.model.
    """
    try:
        import mmdet
    except ImportError:
        raise ImportError(
            'Please run "pip install -U mmcv mmdet" ' "to install MMDetection first for MMDetection inference."
        )

    from mmdet.apis import init_detector

    # create model
    model = init_detector(
        config=self.config_path,
        checkpoint=self.model_path,
        device=self.device,
    )

    # update model image size
    if self.image_size is not None:
        model.cfg.data.test.pipeline[1]["img_scale"] = (self.image_size, self.image_size)

    # set self.model
    self.model = model

    # set category_mapping
    if not self.category_mapping:
        category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
        self.category_mapping = category_mapping
```

2.2 Yolov5:
```
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
            import yolov5
        except ImportError:
            raise ImportError('Please run "pip install -U yolov5" ' "to install YOLOv5 first for YOLOv5 inference.")

        # set model
        try:
            model = yolov5.load(self.model_path, device=self.device)
            model.conf = self.confidence_threshold
            self.model = model
        except Exception as e:
            TypeError("model_path is not a valid yolov5 model path: ", e)

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping
```
2.3 Detectron2:


```
def load_model(self):
    try:
        import detectron2
    except ImportError:
        raise ImportError(
            "Please install detectron2. Check "
            "`https://detectron2.readthedocs.io/en/latest/tutorials/install.html` "
            "for instalattion details."
        )

    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.model_zoo import model_zoo

    cfg = get_cfg()
    cfg.MODEL.DEVICE = self.device

    try:  # try to load from model zoo
        config_file = model_zoo.get_config_file(self.config_path)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config_path)
    except Exception as e:  # try to load from local
        print(e)
        if self.config_path is not None:
            cfg.merge_from_file(self.config_path)
        cfg.MODEL.WEIGHTS = self.model_path
    # set input image size
    if self.image_size is not None:
        cfg.INPUT.MIN_SIZE_TEST = self.image_size
        cfg.INPUT.MAX_SIZE_TEST = self.image_size
    # init predictor
    model = DefaultPredictor(cfg)

    self.model = model

    # detectron2 category mapping
    if self.category_mapping is None:
        try:  # try to parse category names from metadata
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            category_names = metadata.thing_classes
            self.category_names = category_names
            self.category_mapping = {
                str(ind): category_name for ind, category_name in enumerate(self.category_names)
            }
        except Exception as e:
            logger.warning(e)
            # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
            if cfg.MODEL.META_ARCHITECTURE == "RetinaNet":
                num_categories = cfg.MODEL.RETINANET.NUM_CLASSES
            else:  # fasterrcnn/maskrcnn etc
                num_categories = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            self.category_names = [str(category_id) for category_id in range(num_categories)]
            self.category_mapping = {
                str(ind): category_name for ind, category_name in enumerate(self.category_names)
            }
    else:
        self.category_names = list(self.category_mapping.values())
```
2.4 TorchVision:


```
def load_model(self):
    try:
        import torchvision
    except ImportError:
        raise ImportError(
            "torchvision is not installed. Please run 'pip install -U torchvision to use this "
            "torchvision models'"
        )

    # set model
    try:
        from sahi.utils.torch import torch

        model = self.config_path
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        model = model.to(self.device)
        self.model = model
    except Exception as e:
        raise Exception(f"Failed to load model from {self.model_path}. {e}")

    # set category_mapping
    from sahi.utils.torchvision import COCO_CLASSES

    if self.category_mapping is None:
        category_names = {str(i): COCO_CLASSES[i] for i in range(len(COCO_CLASSES))}
        self.category_mapping = category_names
```