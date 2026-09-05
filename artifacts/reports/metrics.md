# Evaluation results

Backbone: MobileNetV2-alpha1, augmented, top 35% fine-tuned.

Model trained on 15 classes: Ulmus carpinifolia, Sorbus aucuparia, Salix cinerea, Populus, Tilia, Sorbus intermedia, Fagus silvatica, Acer, Salix aurita, Quercus, Alnus incana, Betula pubescens, Salix alba 'Sericea, Populus tremula, Ulmus glabra

- Train / val / test: 800 / 143 / 234 images
- **Test accuracy: 97.4%**

```
                     precision    recall  f1-score   support

 Ulmus carpinifolia      0.933     0.933     0.933        15
   Sorbus aucuparia      1.000     1.000     1.000        16
      Salix cinerea      1.000     1.000     1.000        16
            Populus      1.000     0.750     0.857        16
              Tilia      0.889     1.000     0.941        16
  Sorbus intermedia      1.000     1.000     1.000        15
    Fagus silvatica      0.941     1.000     0.970        16
               Acer      0.938     1.000     0.968        15
       Salix aurita      0.938     0.938     0.938        16
            Quercus      1.000     1.000     1.000        15
       Alnus incana      1.000     1.000     1.000        16
   Betula pubescens      1.000     1.000     1.000        15
Salix alba 'Sericea      1.000     1.000     1.000        15
    Populus tremula      1.000     1.000     1.000        16
       Ulmus glabra      1.000     1.000     1.000        16

           accuracy                          0.974       234
          macro avg      0.976     0.975     0.974       234
       weighted avg      0.976     0.974     0.974       234

```
