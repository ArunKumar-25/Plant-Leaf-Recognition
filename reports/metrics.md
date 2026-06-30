# Evaluation results

Backbone: MobileNetV2 (ImageNet, alpha=0.5), frozen + trainable head.

Model trained on 15 classes: Ulmus carpinifolia, Sorbus aucuparia, Salix sinerea, Populus, Tilia, Sorbus intermedia, Fagus silvatica, Acer, Salix aurita, Quercus, Alnus incana, Betula pubescens, Salix alba 'Sericea, Populus tremula, Ulmus glabra

- Train / val / test: 765 / 135 / 225 images
- **Test accuracy: 98.7%**

```
                     precision    recall  f1-score   support

 Ulmus carpinifolia      0.938     1.000     0.968        15
   Sorbus aucuparia      1.000     1.000     1.000        15
      Salix sinerea      1.000     1.000     1.000        15
            Populus      1.000     1.000     1.000        15
              Tilia      1.000     1.000     1.000        15
  Sorbus intermedia      1.000     1.000     1.000        15
    Fagus silvatica      1.000     1.000     1.000        15
               Acer      1.000     1.000     1.000        15
       Salix aurita      1.000     0.933     0.966        15
            Quercus      1.000     1.000     1.000        15
       Alnus incana      1.000     0.933     0.966        15
   Betula pubescens      1.000     0.933     0.966        15
Salix alba 'Sericea      0.938     1.000     0.968        15
    Populus tremula      0.938     1.000     0.968        15
       Ulmus glabra      1.000     1.000     1.000        15

           accuracy                          0.987       225
          macro avg      0.988     0.987     0.987       225
       weighted avg      0.988     0.987     0.987       225

```
