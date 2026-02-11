# Analiză erori – context industrial (Etapa 6)

## Confuzii principale (true → predicted)
- **iarna → mixt**: 5 cazuri
- **mixt → vara**: 1 cazuri

## Interpretare
- Erorile apar frecvent când iluminarea este neuniformă, există blur (vibrații) sau ROI nu surprinde complet banda de rulare.
- În special, clasele **mixt** și **iarnă** pot fi similare vizual în anumite condiții.

## Recomandări
- Creșterea numărului de exemple originale pentru clasa sub-reprezentată.
- Augmentări specifice domeniului (brightness/contrast, blur, perspective) – deja incluse la train.
- Folosirea unui prag de încredere (softmax) în UI pentru cazurile incerte.

## Classification report (test)
```
precision    recall  f1-score   support

       iarna     1.0000    0.5833    0.7368        12
        mixt     0.3750    0.7500    0.5000         4
        vara     0.9231    1.0000    0.9600        12

    accuracy                         0.7857        28
   macro avg     0.7660    0.7778    0.7323        28
weighted avg     0.8777    0.7857    0.7986        28
```
