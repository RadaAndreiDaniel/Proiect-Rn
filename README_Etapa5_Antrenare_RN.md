# README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Rada Andrei Daniel
**Repository GitHub:** https://github.com/RadaAndreiDaniel/Proiect-Rn
**Data predării:** 16 decembrie 2025

---

## 1. Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista oficială de etape ale proiectului.

Obiectivul principal al Etapei 5 este **antrenarea efectivă a modelului de Rețea Neuronală Convoluțională (CNN)** definit în Etapa 4, evaluarea performanței acestuia pe un set de test separat și integrarea modelului antrenat în aplicația finală de clasificare a anvelopelor.

Modelul este utilizat pentru clasificarea benzii de rulare a anvelopelor în următoarele clase:

* vară
* iarnă
* mixt

---

## 2. Continuitate cu Etapa 4 (Prerequisite)

Înainte de realizarea Etapei 5, toate cerințele Etapei 4 au fost îndeplinite:

* arhitectura modelului CNN (ResNet18) este definită și documentată
* state machine-ul aplicației este implementat și descris (docs/state_machine.png)
* minimum 40% din datele utilizate sunt originale, obținute prin adnotare manuală
* modulul de Data Logging generează fișiere CSV cu datele adnotate
* interfața aplicației (UI) este funcțională

Aceste condiții permit trecerea la etapa de antrenare și evaluare a modelului RN.

---

## 3. Date utilizate pentru antrenare

Datele utilizate în această etapă provin din:

* imagini originale colectate și adnotate manual
* extragerea regiunii de interes (ROI) corespunzătoare benzii de rulare
* imagini procesate și salvate local

Dataset-ul final este împărțit în:

* 70% set de antrenare (train)
* 15% set de validare (validation)
* 15% set de test (test)

Structura dataset-ului este organizată în directoarele:

* data/train/
* data/validation/
* data/test/

---

## 4. Pipeline de procesare a datelor

Aplicația utilizează un pipeline secvențial de procesare a imaginilor, care include următoarele etape:

1. încărcarea imaginii brute
2. preprocesare (redimensionare, normalizare)
3. extragerea ROI-ului corespunzător benzii de rulare
4. inferența folosind modelul CNN antrenat
5. afișarea rezultatului de clasificare în interfața aplicației

Pipeline-ul descrie fluxul de procesare a datelor și este utilizat atât în timpul antrenării, cât și în faza de inferență.

---

## 5. State Machine și relația cu pipeline-ul

Execuția pipeline-ului este controlată de un **state machine**, care gestionează logica aplicației și tranzițiile dintre stările funcționale.

State machine-ul asigură:

* rularea etapelor în ordinea corectă
* imposibilitatea lansării inferenței fără existența unei adnotări valide
* utilizarea coerentă a aplicației din interfața grafică

Pipeline-ul este responsabil de procesarea datelor, iar state machine-ul controlează momentul și condițiile în care fiecare etapă este executată.

---

## 6. Configurarea antrenării modelului

Antrenarea modelului a fost realizată utilizând framework-ul **PyTorch**, folosind arhitectura **ResNet18**.

Configurația principală de antrenare este:

* număr epoci: 10
* batch size: 16
* optimizer: Adam
* funcție de pierdere: CrossEntropyLoss

---

## 7. Tabel de hiperparametri

| Hiperparametru      | Valoare          | Justificare                                                      |
| ------------------- | ---------------- | ---------------------------------------------------------------- |
| Learning rate       | 0.001            | Valoare standard pentru Adam, asigură convergență stabilă        |
| Batch size          | 16               | Compromis între stabilitatea gradientului și consumul de memorie |
| Număr epoci         | 10               | Suficient pentru demonstrarea procesului de antrenare            |
| Optimizer           | Adam             | Adaptive learning rate, potrivit pentru CNN                      |
| Loss function       | CrossEntropyLoss | Potrivită pentru clasificare multi-clasă                         |
| Funcții de activare | ReLU / Softmax   | ReLU pentru non-linearitate, Softmax pentru probabilități        |

---

## 8. Procesul de antrenare

Modelul a fost antrenat pe setul de date de antrenare, iar performanța a fost monitorizată pe setul de validare.

Fișiere generate în urma antrenării:

* model antrenat: models/trained_model.pth
* istoric antrenare: results/training_history.csv

Fișierul training_history.csv conține, pentru fiecare epocă:

* train loss
* train accuracy
* validation loss
* validation accuracy

---

## 9. Evaluarea pe setul de test

Evaluarea modelului a fost realizată pe un set de test separat, care conține exclusiv imagini nevăzute în procesul de antrenare și validare.

Metricile obținute pe test set sunt:

* **Test Accuracy:** 87.5%
* **Test F1-score (macro):** 0.619%

Rezultatele indică o acuratețe ridicată a clasificării. Valoarea mai redusă a scorului F1 macro este explicată de distribuția inegală a claselor și de dimensiunea relativ redusă a dataset-ului, fiind un comportament așteptat în acest context.

Acuratețea ridicată obținută pe setul de antrenare este justificată de dimensiunea redusă a dataset-ului și nu este utilizată ca indicator principal al performanței reale.

---

## 10. Integrarea modelului în aplicația finală

Modelul antrenat este integrat în aplicația finală, care permite:

* încărcarea unei imagini
* adnotarea manuală a benzii de rulare
* clasificarea automată a anvelopei utilizând modelul CNN antrenat

Un screenshot demonstrativ al inferenței reale este disponibil în:

* docs/screenshots/inference_real.png

---

## 11. Analiza erorilor (Nivel 2)

### Clase confundate frecvent

Modelul poate confunda ocazional clasele **mixt** și **iarnă**, din cauza similarităților vizuale ale profilului benzii de rulare.

### Cauze ale erorilor

* iluminare neuniformă
* rezoluție redusă a imaginilor
* ROI incomplet

### Implicații practice

* clasificările false negative (iarnă → vară) sunt critice din punct de vedere al siguranței
* clasificările false positive sunt acceptabile și pot fi reinspectate

### Măsuri de îmbunătățire

* creșterea numărului de imagini pentru clasa „mixt”
* augmentări de date (iluminare, contrast)
* utilizarea de class weights
* creșterea rezoluției imaginilor de intrare

---

## 12. Structura repository-ului

proiect-rn-Rada_Andrei_Daniel/
│
├── config/
│       README.txt
│
├── data/
│   │   README.txt
│   │
│   ├── raw/
│   │   ├── vara/
│   │   ├── iarna/
│   │   └── mixt/
│   │
│   ├── processed/
│   │   ├── vara/
│   │   ├── iarna/
│   │   └── mixt/
│   │
│   ├── generated/
│   │       annotation_log.csv
│   │
│   ├── train/
│   ├── validation/
│   └── test/
│
├── docs/
│   │   state_machine.png
│   │   state_machine.txt
│   │
│   └── screenshots/
│           interfata.jpg
│           inference_real.png
│
├── models/
│       trained_model.pth
│
├── results/
│       training_history.csv
│       test_metrics.json
│
└── src/
    ├── app/
    │       finalApp.py
    │       README.txt
    │
    ├── data_acquisition/
    │       annotator.py
    │       decupareBandaDeRulareAntrenament.py
    │       model_annotation.py
    │       README.txt
    │
    ├── neural_network/
    │       train_and_test.py
    │       README.md
    │
    └── preprocessing/
│
│    
│   README_ETAPA3.md
│   README_ETAPA4.md
│   README_ETAPA5.md
│   requirements.txt


---

## 13. Concluzie

În cadrul Etapei 5 a fost realizată prima versiune complet funcțională a sistemului, incluzând:

* antrenarea unui model CNN pe date reale
* evaluarea obiectivă a performanței
* integrarea modelului într-o aplicație utilizabilă

Modelul obținut este funcțional și reprezintă o bază solidă pentru îmbunătățiri ulterioare în etapele următoare ale proiectului.
