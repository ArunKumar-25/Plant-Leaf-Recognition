# Evaluation results

Backbone: MobileNetV2-alpha1, augmented, top 35% fine-tuned.

Model trained on 15 classes: Ulmus carpinifolia, Sorbus aucuparia, Salix cinerea, Populus, Tilia, Sorbus intermedia, Fagus silvatica, Acer, Salix aurita, Quercus, Alnus incana, Betula pubescens, Salix alba 'Sericea, Populus tremula, Ulmus glabra

- Train / val / test: 768 / 136 / 226 images
- **Test accuracy: 99.1%**

```
                     precision    recall  f1-score   support

 Ulmus carpinifolia      1.000     0.933     0.966        15
   Sorbus aucuparia      1.000     1.000     1.000        15
      Salix cinerea      1.000     1.000     1.000        15
            Populus      1.000     1.000     1.000        15
              Tilia      1.000     1.000     1.000        15
  Sorbus intermedia      0.938     1.000     0.968        15
    Fagus silvatica      1.000     1.000     1.000        15
               Acer      1.000     1.000     1.000        15
       Salix aurita      0.938     1.000     0.968        15
            Quercus      1.000     1.000     1.000        15
       Alnus incana      1.000     1.000     1.000        15
   Betula pubescens      1.000     0.933     0.966        15
Salix alba 'Sericea      1.000     1.000     1.000        15
    Populus tremula      1.000     1.000     1.000        15
       Ulmus glabra      1.000     1.000     1.000        16

           accuracy                          0.991       226
          macro avg      0.992     0.991     0.991       226
       weighted avg      0.992     0.991     0.991       226

```
