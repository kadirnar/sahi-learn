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

Eklemek istediğimz kütüphanesinin ismi NarDet olsun :) 

```
class NarDetDetectionModel(DetectionModel):
```
load_model(): NarDet modelinin PYPI desteği olması gerekmektedir.
```
def load_model(self):
    try:
        import NarDet
    except ImportError:
        raise ImportError(
            "NarDet is not installed. Please run 'pip install -U NarDet to use this "
            "NarDet models'"
        )
```
set model: NarDet model dosyasını oluşturuyoruz. 
Daha sonra model.to(self.device) benzeri yapı kullanmalıyız.
```

# set model
try:
    # model dosyasını oluşturuyoruz fakat self.device yapısı kullanmamız gerekiyor.
    self.model = narmodel.NarModel(self.device) # Bunun benzeri bir şey.
except Exception as e:
    raise Exception(f"Failed to load model from {self.model_path}. {e}")

```
Sınıfları {"1": "pedestrian"} bu şekilde olacak şekilde kodlamasını yapıyoruz.
```
    # set category_mapping
    self.category_mapping = {
        str(i): category for i, category in enumerate(self.model.classes)
    }
    # Bunun gibi bir şey kodlamanız gerekecektir.
```
perform_inference(): NarDet modelinin resimlerin giriş değerlerini veriyoruz.
```
def perform_inference(self, image: np.ndarray, image_size: int = None):
    # image değerine resize işlemi uygulamanız gerekiyor.
    
    if image_size is not None:
        image = cv2.resize(image, (self.image_size, self.image_size))
        prediction_result = self.model.(image)
    else:
        prediction_result = self.model.(image)
       
    self._original_predictions = prediction_result
    
```
num_categories(): NarDet modelinin kategorilerinin sayısını veriyoruz.
```
@property
def num_categories(self):
    """
    Returns number of categories
    """
    return len(self.category_mapping)

```
has_mask(): NarDet modelinin masklerinin olup olmadığını kontrol ediyoruz.
```
@property
def has_mask(self):
    """
    Returns if model output contains segmentation mask
    """
    return self.model.with_mask
```
category_names(): NarDet modelinin kategorilerinin isimlerini veriyoruz.
```
@property
def category_names(self):
    return self.category_mapping


```
_create_object_prediction_list_from_original_predictions():
    NarDet modelinin tahminlerini döndürüyoruz.
```
def _create_object_prediction_list_from_original_predictions(
    self,
    shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
    full_shape_list: Optional[List[List[int]]] = None,
    
):
        # SAHI yapısı standart yapıları kullanıyoruz.
        
        original_predictions = self._original_predictions
        category_mapping = self.category_mapping

        # compatilibty for sahi v0.8.20
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]
            
        # boxes, masks, scores, category_ids predict değerlerini for döngüsüne döndürüyoruz.
        boxes = original_predictions["boxes"]
        masks = original_predictions["masks"]
        scores = original_predictions["scores"]
        category_ids = original_predictions["category_ids"]
        
        # check if predictions contain mask
        try: 
            masks = original_predictions["masks"]
        except KeyError:
            masks = None
            
        # create object_prediction_list
        object_prediction_list_per_image = []
        object_prediction_list = []  
        
        # boxes değişkeninde for döngüsü oluşturuyoruz.
        for ind in range(len(boxes)):
            score = scores[ind]
            score = score if score > self.self.confidence_threshold 
            category_id = category_ids[ind]
            category = category_mapping[str(category_id)]
            box = boxes[ind]
            mask = masks[ind] if masks is not None else None
            
            object_prediction = ObjectPrediction(
                bbox=bbox,
                bool_mask=mask,
                category_id=category_id,
                category_name=self.category_mapping[str(category_id)],
                shift_amount=shift_amount,
                score=score,
                full_shape=full_shape,
            )
            object_prediction_list.append(object_prediction)

        object_prediction_list_per_image = [object_prediction_list]

        self._object_prediction_list_per_image = object_prediction_list_per_image            
            
```
