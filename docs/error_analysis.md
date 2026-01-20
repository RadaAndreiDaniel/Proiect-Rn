# Analiză erori – context industrial (Nivel 2)

## Observații cheie
Cele mai frecvente confuzii (true → predicted):
- **mixt → iarna**: 2 cazuri
- **mixt → vara**: 2 cazuri
- **vara → iarna**: 2 cazuri
- **iarna → mixt**: 1 cazuri

## Interpretare industrială
- În medii industriale, **iluminarea variabilă**, reflexiile și **vibrațiile** produc blur/noise care schimbă textura.
- Variațiile de **perspectivă** (unghi camera / poziționare) pot altera pattern-ul perceput.
- Augmentările folosite simulează aceste efecte (perspective + lighting + blur + noise) pentru a crește robustețea.
## Recomandări tehnice
- Creșteți setul de **test** (19 imagini e foarte puțin) pentru scoruri stabile.
- Mențineți echilibrarea claselor (sampler ponderat inclus).
- Pentru aplicație reală, utilizați prag de încredere (softmax) și tratați cazurile incerte.

## Raport clasificare (test)
```
precision    recall  f1-score   support

       iarna     0.6364    0.8750    0.7368         8
        mixt     0.0000    0.0000    0.0000         4
        vara     0.7143    0.7143    0.7143         7

    accuracy                         0.6316        19
   macro avg     0.4502    0.5298    0.4837        19
weighted avg     0.5311    0.6316    0.5734        19
```
