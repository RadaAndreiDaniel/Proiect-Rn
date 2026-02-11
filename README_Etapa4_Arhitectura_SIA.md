README – Etapa 4
Arhitectura Aplicației SIA bazată pe Rețele Neuronale

Disciplina: Rețele Neuronale
Instituție: POLITEHNICA București – FIIR
Student: Rada Andrei Daniel
Repository: https://github.com/RadaAndreiDaniel/Proiect-Rn

1. Scopul aplicației și nevoia reală

Nevoia reală avută în vedere este verificarea automată a conformității anvelopelor în trafic, în special în perioadele sezoniere.

În cadrul controalelor rutiere (ex. baraje organizate de poliție), identificarea tipului de anvelopă se face în prezent manual, fiind un proces lent și dependent de operator. Aplicația propusă urmărește analiza imaginilor sau secvențelor video capturate în trafic pentru a determina dacă un vehicul este echipat corect cu anvelope de vară, iarnă sau mixt.

Soluția este gândită ca un instrument de suport decizional, care asistă operatorul uman și nu îl înlocuiește complet.

2. Nevoie reală → Soluție SIA → Modul software
Nevoie reală	Soluție SIA	Modul
Identificarea vehiculelor cu anvelope neconforme sezonului	Clasificarea benzii de rulare din imagini / video	Neural Network (CNN)
Reducerea influenței fundalului și poziției anvelopei	Extragerea regiunii de interes (ROI)	Data Acquisition
Asistarea operatorului	Interfață de analiză și afișare rezultate	App / UI

Metrici urmărite:

clasificare în 3 clase (vară / iarnă / mixt);

inferență sub 1 secundă pentru o imagine;

pipeline complet funcțional end-to-end.

3. Contribuția originală la setul de date

Contribuția originală constă în adnotarea manuală a imaginilor reale cu anvelope, folosind un instrument software dezvoltat în cadrul proiectului.

Utilizatorul selectează manual puncte cheie ale anvelopei, pe baza cărora este extrasă banda de rulare (ROI). Fiecare imagine astfel obținută reprezintă o observație originală, generată prin intervenție umană directă.

Cod: src/data_acquisition/annotor.py

Date generate: data/generated/vara | iarna | mixt

Dovezile includ capturi din timpul adnotării, imagini ROI generate și structura dataset-ului organizată pe clase.

4. State Machine al aplicației

Locație diagramă: docs/state_machine.png

Fluxul aplicației este:

IDLE
  ↓
LOAD_IMAGE
  ↓
AUTOMATIC_IDENTIFICATION
  ├─ succes → PREPROCESS
  └─ eșec → MANUAL_ANNOTATION → PREPROCESS
                           ↓
                     RN_INFERENCE
                           ↓
                     DISPLAY_RESULT
                           ↓
                          IDLE

ERROR → IDLE


Identificarea automată este utilizată implicit, iar adnotarea manuală este folosită ca mecanism de fallback atunci când detecția automată nu este posibilă sau este nesigură.

5. Modulele aplicației
Modul 1 – Data Acquisition

adnotare manuală a benzii de rulare;

generare imagini ROI utilizabile de rețeaua neuronală;

implementat în Python + OpenCV.

Modul 2 – Neural Network

rețea CNN bazată pe ResNet18;

antrenare, validare și testare implementate;

model salvabil și reutilizabil.

Modul 3 – App / UI

aplicație cu interfață grafică;

upload imagine sau video;

alegere metodă (automat / manual);

afișare verdict și scoruri.

6. Structura repository-ului
Proiect_Rada_Andrei_Daniel_Rn
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── neural_network/
│   │   ├── train_and_test_V2.py
│   │   └── README.md
│   └── app/
│       ├── finalApp.py
│       └── README.md
├── docs/
│   ├── state_machine.png
│   └── screenshots/interfata.png
├── models/trained_model.pt
├── README_Etapa3.md
└── README_Etapa4_Arhitectura_SIA.md

7. Concluzie

În Etapa 4 a fost realizat un schelet complet și funcțional al aplicației SIA, demonstrând integrarea modulelor de achiziție date, procesare, inferență și interfață utilizator.