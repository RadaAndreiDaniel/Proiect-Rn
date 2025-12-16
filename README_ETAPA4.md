README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

Disciplina: Rețele Neuronale
Instituție: POLITEHNICA București – FIIR
Student: Rada Andrei Daniel
Link Repository GitHub: https://github.com/RadaAndreiDaniel/Proiect-Rn
Data: 9 Decembrie 2025

Scopul Etapei 4

Această etapă corespunde punctului 5. Dezvoltarea arhitecturii aplicației software bazată pe RN din lista de 9 etape prezentate în documentul RN Specificații proiect.pdf.

Scopul etapei este realizarea unui schelet complet și funcțional al unui Sistem cu Inteligență Artificială (SIA) pentru clasificarea tipului de anvelopă (vară, iarnă, mixtă) pe baza benzii de rulare extrase din imagini.

În această etapă, accentul este pus pe arhitectura aplicației și integrarea modulelor, nu pe performanța finală a modelului.

1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

Identificarea tipului de anvelopă pe baza benzii de rulare	
Clasificarea imaginii benzii de rulare în vară / iarnă / mixt	RN (CNN) + Aplicație
Reducerea influenței fundalului și poziției anvelopei	
Adnotare manuală pentru extragerea benzii de rulare ---- Data Acquisition (Annotator)
Asistarea utilizatorului în analiza anvelopei	
Interfață care primește imagine, permite adnotare și afișează verdictul	App / UI

Metrici măsurabile:

timp de inferență < 1 secundă pentru o imagine

clasificare în 3 clase distincte

pipeline complet funcțional end-to-end


2. Contribuția Originală la Setul de Date
Contribuția originală la setul de date:

Total observații finale: ~30 imagini
Observații originale: ~20 imagini

Tipul contribuției:

 [X] Date generate prin simulare fizică

 [X] Etichetare/adnotare manuală

 [X] Date sintetice prin metode avansate

Descriere detaliată

Contribuția originală constă în adnotarea manuală a imaginilor cu anvelope, folosind un instrument software dezvoltat în cadrul proiectului. Utilizatorul indică manual reperele necesare pentru delimitarea benzii de rulare, iar aplicația extrage regiunea de interes (ROI).

Această metodă permite obținerea unor imagini curate, relevante pentru clasificare, reducând influența fundalului, iluminării și poziționării anvelopei. Fiecare imagine adnotată reprezintă o observație originală, creată prin intervenție umană directă.

Locația codului: src/data_acquisition/annotator.py
Locația datelor: data/processed/

Dovezi:

capturi de ecran din timpul adnotării (GUI)

imagini preview generate (256×256)

structura dataset-ului organizată pe clase

3. Diagrama State Machine a Întregului Sistem

Locație: docs/state_machine.png

State Machine propus pentru aplicația de clasificare a anvelopelor:
IDLE → LOAD_IMAGE → MANUAL_ANNOTATION → PREPROCESS →
RN_INFERENCE → DISPLAY_RESULT → IDLE
                    ↓
                  ERROR

Justificarea State Machine-ului ales

Aplicația este una de clasificare asistată de utilizator, unde intervenția umană este necesară pentru extragerea corectă a benzii de rulare. Din acest motiv, State Machine-ul include explicit o stare de MANUAL_ANNOTATION.

Stările principale sunt:

### Justificarea State Machine-ului ales:

Am ales o arhitectură de tip **clasificare asistată de utilizator (human-in-the-loop image classification)**, deoarece proiectul urmărește identificarea tipului de anvelopă (vară, iarnă sau mixtă) pe baza benzii de rulare, iar delimitarea corectă a regiunii de interes necesită intervenție umană.

Aplicația nu funcționează în regim de monitorizare continuă, ci procesează imagini individuale furnizate de utilizator, motiv pentru care State Machine-ul este organizat ca un flux secvențial clar, cu gestionarea explicită a erorilor.

#### Stările principale sunt:

1. **IDLE**  
   Sistemul este în stare de repaus și așteaptă input-ul utilizatorului (selectarea unei imagini pentru analiză).

2. **LOAD_IMAGE**  
   Imaginea selectată de utilizator este încărcată din sistemul de fișiere și validată (format, lizibilitate).

3. **MANUAL_ANNOTATION**  
   Utilizatorul indică manual reperele necesare delimitării benzii de rulare, folosind interfața grafică de adnotare. Această stare este esențială pentru eliminarea influenței fundalului și poziționării anvelopei.

4. **PREPROCESS**  
   Banda de rulare extrasă este redimensionată la o dimensiune standard (256×256 px) și pregătită pentru inferența rețelei neuronale.

5. **RN_INFERENCE**  
   Modelul de rețea neuronală convoluțională, definit și compilat anterior, realizează inferența și produce probabilități pentru cele trei clase (vară, iarnă, mixt).

6. **DISPLAY_RESULT**  
   Rezultatul clasificării este afișat utilizatorului sub forma unui verdict și a scorurilor de încredere asociate fiecărei clase.

#### Tranzițiile critice sunt:

- **IDLE → LOAD_IMAGE**: utilizatorul selectează o imagine de analizat  
- **LOAD_IMAGE → MANUAL_ANNOTATION**: imaginea este validă și poate fi procesată  
- **MANUAL_ANNOTATION → PREPROCESS**: utilizatorul finalizează adnotarea (confirmare prin acțiune explicită)  
- **PREPROCESS → RN_INFERENCE**: imaginea este pregătită în format compatibil cu modelul RN  
- **RN_INFERENCE → DISPLAY_RESULT**: inferența este finalizată cu succes  

- **LOAD_IMAGE → ERROR**: imaginea nu poate fi citită sau are format invalid  
- **MANUAL_ANNOTATION → ERROR**: adnotarea este anulată sau incompletă  

#### Starea ERROR:

Starea **ERROR** este necesară pentru gestionarea situațiilor neprevăzute, precum:
- încărcarea unei imagini invalide sau corupte,
- anularea adnotării de către utilizator,
- imposibilitatea generării benzii de rulare.

În aceste cazuri, aplicația poate reveni controlat în starea **IDLE**, fără a afecta stabilitatea sistemului.


4. Scheletul Complet al celor 3 Module Cerute la Curs
Modul	Tehnologie	Locație	Cerință minimă funcțională
Data Acquisition	Python + OpenCV	src/data_acquisition/	Permite adnotare manuală și generează ROI
Neural Network	PyTorch	src/neural_network/test_and_train.py	Model definit, compilat, salvabil
App / UI	CLI + GUI OpenCV	src/app/finalApp.py	Primește input și afișează verdict
Modul 1: Data Acquisition

✔ Codul rulează fără erori
✔ Permite adnotare manuală
✔ Generează imagini ROI utilizabile de RN

Modul 2: Neural Network

✔ Arhitectură CNN definită (ResNet-based)
✔ Model compilat și salvabil
✔ Model reutilizabil în aplicație

⚠️ Modelul NU este optimizat în această etapă

Modul 3: App / UI

✔ Aplicația pornește fără erori
✔ Primește imagine de la utilizator
✔ Integrează adnotarea manuală și inferența RN

Structura Repository-ului la Finalul Etapei 4
Proiect_Rada_Andrei_Daniel_Rn
├── data/
│   ├── raw/
│   ├── processed/ vara/iarna/mixt
│   ├── generated/ 
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/  # Din Etapa 3
│   ├── neural_network/
│		├──train_and_test.py
│               └──README.md
│   └── app/
│          ├──finalApp.py
│          └──README.md  
├── docs/
│   ├── state_machine.png   
│   │ 
│   │	state_machine.txt 
│   └── screenshots/
│		└──interfata.png
├── models/ 
│        └──trained_model.pth
├── config/
├── README.md
├── README_Etapa3.md              
├── README_Etapa4_Arhitectura_SIA.md             
└── requirements.txt

Concluzie

În Etapa 4 a fost realizat un schelet complet și funcțional al aplicației SIA, demonstrând integrarea modulelor de achiziție date, procesare și inferență cu rețele neuronale. Pipeline-ul este complet funcțional și pregătit pentru etapa de antrenare și optimizare.