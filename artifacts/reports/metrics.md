# Evaluation results

Backbone: MobileNetV2-alpha1, augmented, top 35% fine-tuned.

Model trained on 15 classes: Ulmus carpinifolia, Sorbus aucuparia, Salix cinerea, Populus, Tilia, Sorbus intermedia, Fagus silvatica, Acer, Salix aurita, Quercus, Alnus incana, Betula pubescens, Salix alba 'Sericea, Populus tremula, Ulmus glabra

- Train / val / test: 802 / 146 / 237 images
- **Test accuracy: 95.8%**

```
                     precision    recall  f1-score   support

 Ulmus carpinifolia      1.000     0.933     0.966        15
   Sorbus aucuparia      1.000     1.000     1.000        15
      Salix cinerea      0.938     0.938     0.938        16
            Populus      1.000     0.938     0.968        16
              Tilia      1.000     0.938     0.968        16
  Sorbus intermedia      1.000     1.000     1.000        16
    Fagus silvatica      0.941     1.000     0.970        16
               Acer      1.000     0.875     0.933        16
       Salix aurita      0.938     0.938     0.938        16
            Quercus      0.938     0.938     0.938        16
       Alnus incana      0.762     1.000     0.865        16
   Betula pubescens      0.938     0.938     0.938        16
Salix alba 'Sericea      1.000     1.000     1.000        15
    Populus tremula      1.000     0.941     0.970        17
       Ulmus glabra      1.000     1.000     1.000        15

           accuracy                          0.958       237
          macro avg      0.964     0.958     0.959       237
       weighted avg      0.963     0.958     0.959       237

```
