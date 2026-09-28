# Evaluation results

Backbone: MobileNetV2-alpha1, augmented, top 35% fine-tuned.

Model trained on 15 classes: Ulmus carpinifolia, Sorbus aucuparia, Salix cinerea, Populus, Tilia, Sorbus intermedia, Fagus silvatica, Acer, Salix aurita, Quercus, Alnus incana, Betula pubescens, Salix alba 'Sericea, Populus tremula, Ulmus glabra

- Train / val / test: 808 / 144 / 236 images
- **Test accuracy: 97.0%**

```
                     precision    recall  f1-score   support

 Ulmus carpinifolia      0.938     1.000     0.968        15
   Sorbus aucuparia      1.000     0.938     0.968        16
      Salix cinerea      1.000     0.938     0.968        16
            Populus      1.000     0.938     0.968        16
              Tilia      1.000     1.000     1.000        16
  Sorbus intermedia      1.000     1.000     1.000        15
    Fagus silvatica      0.941     1.000     0.970        16
               Acer      1.000     0.933     0.966        15
       Salix aurita      0.938     0.938     0.938        16
            Quercus      0.882     1.000     0.938        15
       Alnus incana      1.000     1.000     1.000        16
   Betula pubescens      0.938     1.000     0.968        15
Salix alba 'Sericea      1.000     1.000     1.000        15
    Populus tremula      0.941     0.941     0.941        17
       Ulmus glabra      1.000     0.941     0.970        17

           accuracy                          0.970       236
          macro avg      0.972     0.971     0.971       236
       weighted avg      0.972     0.970     0.970       236

```
