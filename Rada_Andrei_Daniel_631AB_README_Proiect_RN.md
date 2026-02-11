## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | [Rada Andrei-Daniel] |
| **Grupa / Specializare** | [ex: 631AB / Informatică Industrială] |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | [https://github.com/RadaAndreiDaniel/Proiect-Rn] |
| **Acces Repository** | [Public]
| **Stack Tehnologic** | [Python ]
| **Domeniul Industrial de Interes (DII)** | [Automotive]
| **Tip Rețea Neuronală** | [CNN (Rețea Neuronală Convoluțională)] |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | [68%] | [78%] | [14.7%] | [✓] |
| F1-Score (Macro) | ≥0.65 | [0.6] | [0.73] | [+21.7] | [✓] |
| Latență Inferență | ≤50 ms | ~40 ms | ~35 ms | -5 ms | [✓] |
| Contribuție Date Originale | ≥40% | [60%] | [✓] |
| Nr. Experimente Optimizare | ≥4 | 4 experimente optimizate documentate in readMe [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**. Modelul de rețea neuronală a fost antrenat de la zero, iar minimum 40% din date reprezintă contribuție originală (generate/etichetate de mine).

Instrumentele AI (precum ChatGPT si Gemini) au fost utilizate doar ca suport pentru explicații, structurare, debugging și îmbunătățirea documentației, nu pentru preluarea integrală a codului, arhitecturii sau dataset-ului.

Pot explica și justifica fiecare decizie tehnică implementată în cadrul proiectului.


**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [✓] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [✓] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [✓] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [✓] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [✓] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.



## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz


[ În prezent, verificarea conformității anvelopelor se face manual, prin oprirea vehiculului și inspecție vizuală directă de către autorități. Acest proces este limitat și nu permite monitorizare eficientă la scară largă, în special în sezonul rece când utilizarea anvelopelor de iarnă este obligatorie.

  Proiectul propune automatizarea acestui proces printr-un sistem bazat pe cameră fixă și rețea neuronală convoluțională optimizată, capabilă să detecteze și să clasifice anvelopele (iarnă / vară / mixt). Soluția oferă suport decizional autorităților și contribuie la creșterea siguranței rutiere.]

### 2.2 Beneficii Măsurabile Urmărite


1. [Automatizarea verificării conformității anvelopelor cu reducerea intervenției manuale cu peste 70%.]
2. [Clasificarea tipului de anvelopă cu o acuratețe pe test set ≥75%.]
3. [Obținerea unui F1-Score ≥0.70 pentru echilibru între precision și recall.]
4. [Latență de inferență sub 50 ms per imagine (potrivit pentru utilizare aproape real-time).]
5. [Reducerea riscului de vehicule neconforme nedetectate prin îmbunătățirea recall-ului față de modelul baseline.]

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| 
| [Nevoie reală concretă] | [Detectare + clasificare automată a anvelopei din imagine] | [Neural Network (CNN)] | [Accuracy ≥75%, F1 ≥0.70] |
| [Verificarea manuală și limitată a anvelopelor în trafic] | [Cameră fixă + procesare automată a imaginilor] | |    [Monitorizare la scară largă fără creșterea personalului] | [Latență <50 ms / imagine] |
[Reducerea erorilor umane în evaluare] | [Standardizare decizie prin model antrenat și optimizat] [Creștere F1 cu >50% față de baseline] |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | [Dataset public / Senzori proprii / Simulare / Mixt] |
| **Sursa concretă** | [https://www.pirelli.com/global/en-ww/homepage/
https://www.continental.com/ro-ro/
https://www.michelin.ro/
] |
| **Tipuri de date** | [Imagini] |
| **Format fișiere** | [CSV / PNG / JSON / WEBP] |
| **Perioada colectării/generării** | [ Noiembrie 2025 - Ianuarie 2026] |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | [120] |
| **Observații originale (M)** | [227] |
| **Procent contribuție originală** | [60%] |
| **Tip contribuție** | [ Etichetare manuală / Date sintetice] |
| **Locație cod generare** | `src/data_acquisition/[model_annotation.py]` |
| **Locație date originale** | `data/generated/` |

**Descriere metodă generare/achiziție:**

Datele originale au fost colectate manual de pe platformele oficiale ale producătorilor de anvelope (ex: Pirelli, Continental etc.), selectând imagini relevante pentru categoriile iarnă, vară și mixt. Imaginile au fost descărcate, verificate și etichetate manual pentru a asigura corectitudinea clasei.

Ulterior, acestea au fost prelucrate prin scripturi proprii de procesare (redimensionare, normalizare, eliminare fundal nerelevant, ajustare contrast/luminozitate), pentru a standardiza formatul și a îmbunătăți relevanța vizuală a profilului anvelopei. Aceste date sunt relevante deoarece surprind clar diferențele de textură și profil dintre tipurile de anvelope, esențiale pentru clasificarea corectă în contextul aplicației propuse.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | [140] |
| Validation | 15% | [30] |
| Test | 15% | [30] |

**Preprocesări aplicate:**
Redimensionare imagini la dimensiune fixă (ex: 224x224 pixeli) pentru uniformizarea inputului în CNN

Normalizare valori pixeli în intervalul [0,1]

Conversie format imagine (RGB standardizat)

Eliminare imagini neclare sau nerelevante

Etichetare manuală și verificare consistență clase

Împărțire stratificată în train / validation / test (70% / 15% / 15%)





## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software


Modul	Tehnologie	Funcționalitate Principală	Locație în Repo

Data Logging / Acquisition	Python	Colectare imagini anvelope, organizare dataset și pregătire pentru preprocesare	src/data_acquisition/

Neural Network	PyTorch	Detectare și clasificare multi-clasă a anvelopelor (iarnă / mixt / vară) folosind CNN	src/neural_network/

Web Service / UI	Streamlit	Interfață pentru încărcare imagine și afișare predicție + scor confidence	src/app/

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png` *(sau `state_machine_v2.png` dacă actualizată în Etapa 6)*

**Stări principale și descriere:**

| Stare                      | Descriere                                                                            | Condiție Intrare                           | Condiție Ieșire                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `IDLE`                     | Așteaptă acțiunea utilizatorului (selectare imagine/video)                           | Pornire aplicație / revenire după o rulare | Utilizator selectează fișier                                                             |
| `LOAD_IMAGE/VIDEO`         | Încărcare fișier de intrare și validare format (imagine/video)                       | Fișier selectat din `IDLE`                 | Fișier valid → `AUTOMATIC_IDENTIFICATION` / Fișier invalid → `ERROR`                     |
| `AUTOMATIC_IDENTIFICATION` | Detectare automată a benzii de rulare / ROI (zona relevantă a anvelopei)             | Fișier valid încărcat                      | ROI detectat → `PREPROCESS` / Detecție eșuată → `MANUAL_ANNOTATION` / Excepție → `ERROR` |
| `MANUAL_ANNOTATION`        | Delimitare manuală ROI de către utilizator (fallback dacă detecția automată eșuează) | Detecția automată a eșuat                  | ROI valid definit → `PREPROCESS` / Anotare eșuată/anulată → `ERROR`                      |
| `PREPROCESS`               | Preprocesare ROI: redimensionare, normalizare, pregătire tensor pentru CNN           | ROI disponibil (auto sau manual)           | Input gata → `RN_INFERENCE`                                                              |
| `RN_INFERENCE`             | Inferență CNN: predicție clasă (iarna/vara/mixt) + scoruri                           | Input preprocesat                          | Predicție finalizată → `DISPLAY_RESULT`                                                  |
| `DISPLAY_RESULT`           | Afișare rezultat (verdict + confidence) și opțiune de rulare pe alt input            | Predicție disponibilă                      | Utilizator alege input nou → `IDLE` / Închidere aplicație                                |
| `ERROR`                    | Gestionare erori (fișier invalid, detecție/annotare eșuată, excepții)                | Orice stare care produce o eroare          | Reset / revenire aplicație → `IDLE`                                                      |


**Justificare alegere arhitectură State Machine:**

Structura State Machine reflectă pașii reali ai aplicației, de la încărcarea imaginii până la afișarea verdictului final. Separarea clară a etapelor (identificare, preprocesare, inferență, afișare) permite gestionarea eficientă a erorilor și oferă posibilitatea intervenției manuale dacă detecția automată eșuează, asigurând stabilitate și modularitate sistemului.


### 4.3 Actualizări State Machine în Etapa 6 (dacă este cazul)

| Componentă Modificată  | Valoare Etapa 5                   | Valoare Etapa 6                                        | Justificare Modificare                                       |
| ---------------------- | --------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| Detectare bandă rulare | Doar `MANUAL_ANNOTATION`          | `AUTOMATIC_IDENTIFICATION` + fallback manual           | Automatizare proces și reducerea intervenției utilizatorului |
| Flux State Machine     | Identificare manuală obligatorie  | Detectare automată + manuală doar dacă eșuează         | Creștere eficiență și reducere timp procesare                |
| Stabilitate procesare  | Fără gestionare detecție automată | Tranziții dedicate către `ERROR` și fallback controlat | Creștere robustețe sistem                                    |




## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
Input (shape: [3, 224, 224])
  → Conv2D(32, 3x3) → ReLU → MaxPool(2x2)
  → Conv2D(64, 3x3) → ReLU → MaxPool(2x2)
  → Conv2D(128, 3x3) → ReLU → MaxPool(2x2)
  → Flatten
  → Dense(128) → ReLU → Dropout(0.5)
  → Dense(3) → Softmax
Output: 3 clase (iarna / mixt / vara)

**Justificare alegere arhitectură:**

Arhitectura CNN a fost aleasă deoarece problema este una de clasificare de imagini, unde extragerea automată a caracteristicilor vizuale (textură, profil, pattern) este esențială. S-a optat pentru o rețea de complexitate medie pentru a obține un echilibru între acuratețe și latență, evitând modele foarte adânci care ar fi crescut timpul de inferență fără beneficii semnificative pentru dimensiunea dataset-ului disponibil.



### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală        | Justificare Alegere                                                                      |
| -------------- | --------------------- | ---------------------------------------------------------------------------------------- |
| Learning Rate  | 0.0003                | Convergență stabilă observată în experimente, fără oscilații majore ale val_loss         |
| Batch Size     | 16                    | Potrivit pentru dimensiunea redusă a dataset-ului și stabilitate mai bună a gradientului |
| Epochs         | 6 (best epoch = 1)    | Early stopping implicit prin selectarea celui mai bun model pe validation                |
| Optimizer      | Adam                  | Optimizator adaptiv, potrivit pentru clasificare imagini                                 |
| Loss Function  | CrossEntropyLoss      | Clasificare multi-clasă (3 clase)                                                        |
| Regularizare   | Dropout 0.5           | Reducere overfitting observat în modelul baseline                                        |
| Early Stopping | Monitorizare val_loss | Selectare automată a modelului cu cea mai mică val_loss                                  |


### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| Baseline: Accuracy 63.16% | F1 0.4837
Final: Accuracy 78.57% | F1 0.7323


| Exp#         | Modificare față de Baseline                                | Accuracy   | F1-Score   | Timp Antrenare | Observații                                |
| ------------ | ---------------------------------------------------------- | ---------- | ---------- | -------------- | ----------------------------------------- |
| **Baseline** | Configurația din Etapa 5                                   | 63.16%     | 0.4837     | ~3 min         | Model instabil, F1 scăzut pe clasa „mixt” |
| Exp 1        | Learning Rate ajustat 0.001 → 0.0003                       | 68.42%     | 0.55       | ~4 min         | Convergență mai stabilă                   |
| Exp 2        | Adăugare strat Conv suplimentar                            | 71.42%     | 0.61       | ~5 min         | Creștere capacitate extragere features    |
| Exp 3        | Dropout 0.3 → 0.5                                          | 74.00%     | 0.68       | ~5 min         | Reducere overfitting                      |
| Exp 4        | Batch size 32 → 16                                         | 76.19%     | 0.70       | ~6 min         | Stabilitate gradient îmbunătățită         |
| Exp 5        | Detectare automată ROI + preprocesare îmbunătățită         | 78.57%     | 0.7323     | ~6 min         | Generalizare mai bună                     |
| **FINAL**    | Configurația optimizată (LR=0.0003, Dropout=0.5, Batch=16) | **78.57%** | **0.7323** | ~6 min         | **Model folosit în aplicația finală**     |


**Justificare alegere model final:**

Modelul final a fost ales deoarece oferă cel mai bun echilibru între performanță și complexitate. Deși adăugarea de straturi a crescut ușor timpul de antrenare, îmbunătățirea F1-Score (+51% față de baseline) justifică alegerea. Configurația finală reduce overfitting-ul și îmbunătățește generalizarea, menținând în același timp o latență potrivită pentru utilizare aproape real-time.

**Referințe fișiere:** `results/optimization_experiments_final.csv`, `models/optimized_model_final.pt`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)


Accuracy: 0.7857

F1 macro: 0.7323

Precision macro: 0.7660

Recall macro: 0.7778

| Metric                | Valoare | Target Minim | Status |
| --------------------- | ------- | ------------ | ------ |
| **Accuracy**          | 78.57%  | ≥70%         | ✓      |
| **F1-Score (Macro)**  | 0.7323  | ≥0.65        | ✓      |
| **Precision (Macro)** | 0.7660  | -            | -      |
| **Recall (Macro)**    | 0.7778  | -            | -      |


**Îmbunătățire față de Baseline (Etapa 5):**

| Metric   | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
| -------- | ------------------ | ------------------- | ------------ |
| Accuracy | 87.5%              | 78.57%              | -8.93%       |
| F1-Score | 0.619              | 0.7323              | +0.1133      |

Observație importantă pentru interpretare:

Accuracy a scăzut (-8.93%)

F1-score a crescut semnificativ (+0.1133, adică ~+18.3%)

Aceasta confirmă că optimizarea a fost orientată pe echilibrarea claselor și îmbunătățirea performanței globale reale, nu pe maximizarea accuracy brute.


**Referință fișier:** `results/final_metrics_final.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**
| Aspect                                 | Observatie                                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Clasa cu cea mai bună performanță**  | **vara** – Precision 92.31%, Recall 100%                                                                                        |
| **Clasa cu cea mai slabă performanță** | **mixt** – Precision 37.50%, Recall 75.00%                                                                                      |
| **Confuzii frecvente**                 | Clasa **iarna** confundată frecvent cu **mixt** (5 din 12 cazuri – 41.67%), din cauza similarității vizuale a profilului        |
| **Dezechilibru clase**                 | Clasa **mixt** are doar 4 exemple în test set (vs 12 pentru iarna și 12 pentru vara), ceea ce explică instabilitatea metricilor |




---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă                 | Stare Etapa 5                      | Modificare Etapa 6                                     | Justificare                                                            |
| -------------------------- | ---------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------- |
| **Model încărcat**         | `trained_model.pth` (baseline)     | `optimized_model_final.pt`                             | Creștere F1-score (0.619 → 0.7323) și echilibrare mai bună între clase |
| **Detectare bandă rulare** | Doar manuală (`MANUAL_ANNOTATION`) | Detectare automată + fallback manual                   | Automatizare proces și reducerea intervenției utilizatorului           |
| **Threshold decizie**      | Implicit (Softmax max class)       | Ajustare indirectă prin echilibrare + sampler ponderat | Reducerea supra-clasificării în clasa „mixt”                           |
| **UI - feedback vizual**   | Afișare clasă simplă               | Afișare clasă + scoruri de încredere (confidence %)    | Transparanță și suport decizional pentru utilizator                    |
| **Training pipeline**      | Fără early stopping                | Early stopping activ (best epoch = 1, stop la epoca 6) | Prevenire overfitting și stabilitate generalizare                      |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`
În screenshot-ul inference_optimized.png este prezentată interfața aplicației în momentul realizării unei inferențe cu modelul optimizat. Se observă imaginea încărcată de utilizator, zona benzii de rulare identificată (automat sau manual) și rezultatul clasificării afișat sub forma clasei prezise (iarna / mixt / vara), împreună cu scorurile de încredere (confidence %) pentru fiecare categorie.

Screenshot-ul demonstrează integrarea completă a modelului optimizat în aplicația software și funcționarea end-to-end a pipeline-ului: încărcare imagine → preprocesare → inferență → afișare rezultat.


### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/` *(GIF / Video / Secvență screenshots)*



| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Input | [ex: Upload imagine nouă (NU din train/test)] |
| 2 | Procesare | [ex: Bară de progres + preprocesare vizibilă] |
| 3 | Inferență | [ex: Predicție afișată: "Clasa: Vara,Iarna,Mixt, Confidence: 87%"] |
| 4 | Decizie |  |

**Latență măsurată end-to-end:** [X] ms  
**Data și ora demonstrației:** [11.02.2026, 20:00]

---

## 8. Structura Repository-ului Final

```
proiect-rn-[nume-prenume]/
│
├── README.md                               # ← ACEST FIȘIER (Overview Final Proiect - Pe moodle la Evaluare Finala RN > Upload Livrabil 1 - Proiect RN (Aplicatie Sofware) - trebuie incarcat cu numele: NUME_Prenume_Grupa_README_Proiect_RN.md)
│
├── docs/
│   ├── etapa3_analiza_date.md              # Documentație Etapa 3
│   ├── etapa4_arhitectura_SIA.md           # Documentație Etapa 4
│   ├── etapa5_antrenare_model.md           # Documentație Etapa 5
│   ├── etapa6_optimizare_concluzii.md      # Documentație Etapa 6
│   │
│   ├── state_machine.png                   # Diagrama State Machine inițială
│   ├── state_machine_v2.png                # (opțional) Versiune actualizată Etapa 6
│   ├── confusion_matrix_optimized.png      # Confusion matrix model final
│   │
│   ├── screenshots/
│   │   ├── ui_demo.png                     # Screenshot UI schelet (Etapa 4)
│   │   ├── inference_real.png              # Inferență model antrenat (Etapa 5)
│   │   └── inference_optimized.png         # Inferență model optimizat (Etapa 6)
│   │
│   ├── demo/                               # Demonstrație funcțională end-to-end
│   │   └── demo_end_to_end.gif             # (sau .mp4 / secvență screenshots)
│   │
│   ├── results/                            # Vizualizări finale
│   │   ├── loss_curve.png                  # Grafic loss/val_loss (Etapa 5)
│   │   ├── metrics_evolution.png           # Evoluție metrici (Etapa 6)
│   │   └── learning_curves_final.png       # Curbe învățare finale
│   │
│   └── optimization/                       # Grafice comparative optimizare
│       ├── accuracy_comparison.png         # Comparație accuracy experimente
│       └── f1_comparison.png               # Comparație F1 experimente
│
├── data/
│   ├── README.md                           # Descriere detaliată dataset
│   ├── raw/                                # Date brute originale
│   ├── processed/                          # Date curățate și transformate
│   ├── generated/                          # Date originale (contribuția ≥40%)
│   ├── train/                              # Set antrenare (70%)
│   ├── validation/                         # Set validare (15%)
│   └── test/                               # Set testare (15%)
│
├── src/
│   ├── data_acquisition/                   # MODUL 1: Generare/Achiziție date
│   │   ├── README.md                       # Documentație modul
│   │   ├── generate.py                     # Script generare date originale
│   │   └── [alte scripturi achiziție]
│   │
│   ├── preprocessing/                      # Preprocesare date (Etapa 3+)
│   │   ├── data_cleaner.py                 # Curățare date
│   │   ├── feature_engineering.py          # Extragere/transformare features
│   │   ├── data_splitter.py                # Împărțire train/val/test
│   │   └── combine_datasets.py             # Combinare date originale + externe
│   │
│   ├── neural_network/                     # MODUL 2: Model RN
│   │   ├── README.md                       # Documentație arhitectură RN
│   │   ├── model.py                        # Definire arhitectură (Etapa 4)
│   │   ├── train.py                        # Script antrenare (Etapa 5)
│   │   ├── evaluate.py                     # Script evaluare metrici (Etapa 5)
│   │   ├── optimize.py                     # Script experimente optimizare (Etapa 6)
│   │   └── visualize.py                    # Generare grafice și vizualizări
│   │
│   └── app/                                # MODUL 3: UI/Web Service
│       ├── README.md                       # Instrucțiuni lansare aplicație
│       └── main.py                         # Aplicație principală
│
├── models/
│   ├── untrained_model.h5                  # Model schelet neantrenat (Etapa 4)
│   ├── trained_model.h5                    # Model antrenat baseline (Etapa 5)
│   ├── optimized_model.h5                  # Model FINAL optimizat (Etapa 6) ← FOLOSIT
│   └── final_model.onnx                    # (opțional) Export ONNX pentru deployment
│
├── results/
│   ├── training_history.csv                # Istoric antrenare - toate epocile (Etapa 5)
│   ├── test_metrics.json                   # Metrici baseline test set (Etapa 5)
│   ├── optimization_experiments.csv        # Toate experimentele optimizare (Etapa 6)
│   ├── final_metrics.json                  # Metrici finale model optimizat (Etapa 6)
│   └── error_analysis.json                 # Analiza detaliată erori (Etapa 6)
│
├── config/
│   ├── preprocessing_params.pkl            # Parametri preprocesare salvați (Etapa 3)
│   └── optimized_config.yaml               # Configurație finală model (Etapa 6)
│
├── requirements.txt                        # Dependențe Python (actualizat la fiecare etapă)
└── .gitignore                              # Fișiere excluse din versionare
```

### Legendă Progresie pe Etape

| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| `data/raw/`, `processed/`, `train/`, `val/`, `test/` | ✓ Creat | - | Actualizat* | - |
| `data/generated/` | - | ✓ Creat | - | - |
| `src/preprocessing/` | ✓ Creat | - | Actualizat* | - |
| `src/data_acquisition/` | - | ✓ Creat | - | - |
| `src/neural_network/model.py` | - | ✓ Creat | - | - |
| `src/neural_network/train.py`, `evaluate.py` | - | - | ✓ Creat | - |
| `src/neural_network/optimize.py`, `visualize.py` | - | - | - | ✓ Creat |
| `src/app/` | - | ✓ Creat | Actualizat | Actualizat |
| `models/untrained_model.*` | - | ✓ Creat | - | - |
| `models/trained_model.*` | - | - | ✓ Creat | - |
| `models/optimized_model.*` | - | - | - | ✓ Creat |
| `docs/state_machine.*` | - | ✓ Creat | - | (v2 opțional) |
| `docs/etapa3_analiza_date.md` | ✓ Creat | - | - | - |
| `docs/etapa4_arhitectura_SIA.md` | - | ✓ Creat | - | - |
| `docs/etapa5_antrenare_model.md` | - | - | ✓ Creat | - |
| `docs/etapa6_optimizare_concluzii.md` | - | - | - | ✓ Creat |
| `docs/confusion_matrix_optimized.png` | - | - | - | ✓ Creat |
| `docs/screenshots/` | - | ✓ Creat | Actualizat | Actualizat |
| `results/training_history.csv` | - | - | ✓ Creat | - |
| `results/optimization_experiments.csv` | - | - | - | ✓ Creat |
| `results/final_metrics.json` | - | - | - | ✓ Creat |
| **README.md** (acest fișier) | Draft | Actualizat | Actualizat | **FINAL** |

*\* Actualizat dacă s-au adăugat date noi în Etapa 4*

### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - Accuracy=X.XX, F1=X.XX (optimizat)" |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

```
Python >= 3.8 (recomandat 3.10+)


### 9.2 Instalare

# 1. Clonare repository
git clone https://github.com/RadaAndreiDaniel/Proiect-Rn
cd Proiect-Rn

# 2. Creare mediu virtual (recomandat)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# sau: venv\Scripts\activate    # Windows

# 3. Instalare dependențe
pip install -r requirements.txt

### 9.3 Rulare Pipeline Complet

```bash
# Pasul 1: Evaluare model salvat
# (generează metrici și confusion matrix)
python src/neural_network/train_and_test.py --evaluate

# Pasul 3: Lansare aplicație UI
python src/app/finalApp.py


### 9.4 Verificare Rapidă 

```bash

# Verificare încărcare model optimizat
python -c "import torch; model=torch.load('models/optimized_model_final.pt'); print('✓ Model încărcat cu succes')"


 9.5 Structură Comenzi LabVIEW (dacă aplicabil)

10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|--------------------------------|--------|----------|--------|
| [Automatizarea verificării anvelopelo] | [Reducere intervenție manuală >70%] | [Detectare automată + fallback manual implementat] | [✓] |
| Accuracy pe test set | ≥70% | [78.57%] | [✓] |
| F1-Score pe test set | ≥0.65 | [0.7323] | [✓] |
| [Latență inferență] | [~30–40 ms] | [realizat] | [✓/✗] |



 10.2 Ce NU Funcționează – Limitări Cunoscute


Limitare 1: Modelul poate confunda clasele „iarna” și „mixt” în cazurile de tranziție sezonieră, unde diferențele vizuale sunt mai puțin evidente.Acest lucru nu constituie un dezavantaj deoarce in sezonul rece sunt acceptate si cele mixte deci nu ar rezulta o sanctiune falsa.

Limitare 2: În condiții de iluminare slabă sau unghi nefavorabil al imaginii, performanța poate scădea.

Limitare 3: Sistemul nu este încă validat în condiții reale de trafic (video live, mișcare, ploaie, noapte).

Funcționalități planificate dar neimplementate:

Optimizare pentru rulare pe hardware dedicat

10.3 Lecții Învățate (Top 5)

1 Importanța F1-score-ului în locul accuracy brute: Un accuracy mare nu garantează performanță echilibrată între clase; F1-score (macro) oferă o imagine mai realistă în cazul claselor inegale.

2 Echilibrarea claselor influențează decisiv rezultatul: Modul de distribuție a datelor și utilizarea sampler-ului ponderat au avut impact mai mare decât simpla modificare a learning rate-ului.

3 Early stopping previne supra-antrenarea: Oprirea antrenării la momentul optim a îmbunătățit capacitatea de generalizare și a redus overfitting-ul.

4 Analiza confusion matrix-ului este esențială: Metricile globale pot ascunde probleme specifice; analiza erorilor pe clasă oferă informații mult mai valoroase pentru optimizare.

5 Structurarea pe etape a proiectului ajută la integrare: Implementarea modulară (Data Acquisition – RN – UI) și actualizarea progresivă a documentației au făcut posibilă integrarea finală fără blocaje majore.

 10.4 Retrospectivă

 Ce ați schimba dacă ați reîncepe proiectul?

 Dacă aș relua proiectul, aș acorda mai multă atenție organizării și urmăririi progresului. As documenta clar, după fiecare sesiune de lucru, stadiul exact al dezvoltării și următorii pași, pentru a evita pierderea timpului recitind și reanalizând codul deja scris. O planificare mai riguroasă a etapelor și o evidență clară a modificărilor ar fi redus timpul necesar reluării muncii după pauze.

 De asemenea, aș implementa interfața web într-o fază mai timpurie a proiectului. Integrarea rapidă a UI-ului ar fi permis testarea continuă a pipeline-ului end-to-end, evitând rulări repetate din foldere separate și simplificând procesul de validare a fiecărei modificări aduse modelului.



10.5 Direcții de Dezvoltare Ulterioară

Short-term (1-2 săptămâni):

Reproiectarea structurii experimentelor și definirea clară a unui plan de optimizare încă de la început (metrică principală, criterii de selecție model, strategie de echilibrare), pentru a reduce timpul pierdut pe testări necontrolate.

Medium-term (1-2 luni):

Optimizarea procesării video și reducerea latenței pentru funcționare stabilă în flux continuu de trafic.

Long-term:

Integrarea sistemului pe un dispozitiv dedicat montat fix într-un punct de monitorizare (ex: cameră inteligentă instalată pe stâlp rutier), pentru implementare reală în teren și monitorizare automată continuă.

## 11. Bibliografie


1. Abaza Felician Bogdan, Cursul Retele Neuronale, 2026, URL: https://curs.upb.ro/2025/course/view.php?id=1338

2. Bohundan, Treadscan, 2022 , https://github.com/bohundan/treadscan3

3. Surse suplimentare
ResNet: https://www.comet.com/site/blog/resnet-how-one-paper-changed-deep-learning-forever/

Roboflow: https://roboflow.com/

PyTorchPaper: https://programming-ocean.com/articles/pytorch-paper-summary.php

F1 Score in Imbalanced Data: https://oboe.com/learn/mastering-the-f1-score-for-machine-learning-evaluation-1ln5iez/f1-score-in-imbalanced-data-16cskzn

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [ ] **Accuracy ≥70%** pe test set (verificat în `results/final_metrics.json`)
- [ ] **F1-Score ≥0.65** pe test set
- [ ] **Contribuție ≥40% date originale** (verificabil în `data/generated/`)
- [ ] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [ ] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [ ] **Confusion matrix** generată și interpretată (Secțiunea 6.2)
- [ ] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [ ] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [ ] **Demonstrație end-to-end** disponibilă în `docs/demo/`

### Repository și Documentație

- [ ] **README.md** complet (toate secțiunile completate cu date reale)
- [ ] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [ ] **Screenshots** prezente în `docs/screenshots/`
- [ ] **Structura repository** conformă cu Secțiunea 8
- [ ] **requirements.txt** actualizat și funcțional
- [ ] **Cod comentat** (minim 15% linii comentarii relevante)
- [ ] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare

- [ ] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [ ] **Tag `v0.6-optimized-final`** creat și pushed
- [ ] **Commit-uri incrementale** vizibile în `git log` (nu 1 commit gigantic)
- [ ] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

### Verificare Anti-Plagiat

- [ ] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [ ] **Minimum 40% date originale** (nu doar subset din dataset public)
- [ ] Cod propriu sau clar atribuit (surse citate în Bibliografie)

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** [11.02.2026]  
**Tag Git:** `v0.6-optimized-final`

---
