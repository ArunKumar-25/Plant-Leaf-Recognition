# Evaluation results

Backbone: MobileNetV2-alpha1, augmented, top 35% fine-tuned.

Model trained on 15 classes: Ulmus carpinifolia, Sorbus aucuparia, Salix cinerea, Populus, Tilia, Sorbus intermedia, Fagus silvatica, Acer, Salix aurita, Quercus, Alnus incana, Betula pubescens, Salix alba 'Sericea, Populus tremula, Ulmus glabra

- Train / val / test: 795 / 141 / 233 images
- **Test accuracy: 95.7%**

```
                     precision    recall  f1-score   support

 Ulmus carpinifolia      1.000     0.933     0.966        15
   Sorbus aucuparia      0.941     1.000     0.970        16
      Salix cinerea      1.000     1.000     1.000        16
            Populus      1.000     0.875     0.933        16
              Tilia      0.875     0.875     0.875        16
  Sorbus intermedia      1.000     0.933     0.966        15
    Fagus silvatica      0.842     1.000     0.914        16
               Acer      1.000     1.000     1.000        15
       Salix aurita      0.842     1.000     0.914        16
            Quercus      1.000     1.000     1.000        15
       Alnus incana      0.938     0.938     0.938        16
   Betula pubescens      1.000     1.000     1.000        15
Salix alba 'Sericea      1.000     1.000     1.000        15
    Populus tremula      1.000     0.812     0.897        16
       Ulmus glabra      1.000     1.000     1.000        15

           accuracy                          0.957       233
          macro avg      0.963     0.958     0.958       233
       weighted avg      0.961     0.957     0.957       233

```
