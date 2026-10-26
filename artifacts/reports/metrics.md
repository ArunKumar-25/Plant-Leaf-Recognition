# Evaluation results

Backbone: MobileNetV2-alpha1, augmented, top 35% fine-tuned.

Model trained on 15 classes: Ulmus carpinifolia, Sorbus aucuparia, Salix cinerea, Populus, Tilia, Sorbus intermedia, Fagus silvatica, Acer, Salix aurita, Quercus, Alnus incana, Betula pubescens, Salix alba 'Sericea, Populus tremula, Ulmus glabra

- Train / val / test: 812 / 143 / 237 images
- **Test accuracy: 97.9%**

```
                     precision    recall  f1-score   support

 Ulmus carpinifolia      1.000     1.000     1.000        15
   Sorbus aucuparia      1.000     1.000     1.000        16
      Salix cinerea      0.889     1.000     0.941        16
            Populus      1.000     0.938     0.968        16
              Tilia      1.000     1.000     1.000        17
  Sorbus intermedia      0.938     1.000     0.968        15
    Fagus silvatica      1.000     1.000     1.000        17
               Acer      1.000     0.933     0.966        15
       Salix aurita      1.000     0.882     0.938        17
            Quercus      0.933     0.933     0.933        15
       Alnus incana      1.000     1.000     1.000        16
   Betula pubescens      0.938     1.000     0.968        15
Salix alba 'Sericea      1.000     1.000     1.000        15
    Populus tremula      1.000     1.000     1.000        16
       Ulmus glabra      1.000     1.000     1.000        16

           accuracy                          0.979       237
          macro avg      0.980     0.979     0.979       237
       weighted avg      0.980     0.979     0.979       237

```
