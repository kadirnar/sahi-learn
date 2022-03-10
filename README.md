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

Devamı gelecek :)
