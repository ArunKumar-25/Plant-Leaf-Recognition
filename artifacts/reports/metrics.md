# Evaluation results

Backbone: MobileNetV2-alpha1, augmented, top 35% fine-tuned.

Model trained on 15 classes: Ulmus carpinifolia, Sorbus aucuparia, Salix cinerea, Populus, Tilia, Sorbus intermedia, Fagus silvatica, Acer, Salix aurita, Quercus, Alnus incana, Betula pubescens, Salix alba 'Sericea, Populus tremula, Ulmus glabra

- Train / val / test: 765 / 135 / 225 images
- **Test accuracy: 100.0%**

```
                     precision    recall  f1-score   support

 Ulmus carpinifolia      1.000     1.000     1.000        15
   Sorbus aucuparia      1.000     1.000     1.000        15
      Salix cinerea      1.000     1.000     1.000        15
            Populus      1.000     1.000     1.000        15
              Tilia      1.000     1.000     1.000        15
  Sorbus intermedia      1.000     1.000     1.000        15
    Fagus silvatica      1.000     1.000     1.000        15
               Acer      1.000     1.000     1.000        15
       Salix aurita      1.000     1.000     1.000        15
            Quercus      1.000     1.000     1.000        15
       Alnus incana      1.000     1.000     1.000        15
   Betula pubescens      1.000     1.000     1.000        15
Salix alba 'Sericea      1.000     1.000     1.000        15
    Populus tremula      1.000     1.000     1.000        15
       Ulmus glabra      1.000     1.000     1.000        15

           accuracy                          1.000       225
          macro avg      1.000     1.000     1.000       225
       weighted avg      1.000     1.000     1.000       225

```
