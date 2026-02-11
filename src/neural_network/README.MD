# 🧠 Neural Network – Tire Tread Classification

## Scopul modulului

Acest modul implementează **antrenarea, validarea și testarea** unei rețele
neuronale convoluționale pentru **clasificarea benzii de rulare a anvelopelor**
în clasele:
- vară
- iarnă
- mixt

Modulul operează pe seturile de date `train / validation / test` și produce
modele salvate, metrici de evaluare și materiale de analiză a erorilor.

---

## Structura datelor de intrare

Datele sunt citite din:
data/
├── train/
├── validation/
└── test/


## Arhitectura rețelei

- Tip rețea: **CNN**
- Arhitectură: **ResNet18**
- Framework: **PyTorch**
- Număr clase: determinat automat din date


## Preprocesare și augmentări

### Train
- Resize la **224 × 224**
- Perspective ușoare (simulare poziționare cameră)
- Variații de iluminare (ColorJitter)
- Blur Gaussian ușor (vibrații)
- Zgomot Gaussian (noise industrial)
- Normalizare ImageNet

### Validation / Test
- Resize
- Normalizare (fără augmentări)

Augmentările sunt alese pentru a simula **condiții reale de utilizare**.

---

## Strategie de antrenare

- Optimizator: **AdamW**
- Loss: **CrossEntropyLoss**
- Sampler ponderat (`WeightedRandomSampler`) pentru echilibrarea claselor
- Early Stopping pe `val_loss`
- Scheduler `ReduceLROnPlateau` pe `val_loss`

Cel mai bun model este selectat în funcție de **loss-ul de validare**.

---

## Evaluare și rezultate

Pe setul de test sunt calculate:
- Accuracy
- F1-score (macro)
- Classification Report
- Confusion Matrix

Rezultatele sunt salvate automat în:
results/test_metrics.json
docs/confusion_matrix.png
docs/loss_curve.png
docs/error_analysis.md
docs/errors/ (imagini clasificate greșit)



## Salvare model

Cel mai bun model este salvat în:
models/trained_model.pt


Checkpoint-ul conține:
- starea modelului
- epoca optimă
- loss-ul de validare
- lista claselor
- parametri de normalizare

---

## Rulare

Scriptul principal este:
train_and_test_V2.py

Rulare:
```bash
python train_and_test_V2.py
Scriptul gestionează automat:

detectarea dispozitivului (CPU / CUDA)

încărcarea datelor

antrenarea

evaluarea

salvarea rezultatelor

Structura fișierelor

src/neural_network/
 ├── train_and_test_V2.py
 └── README.md