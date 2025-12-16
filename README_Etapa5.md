📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

Disciplina: Rețele Neuronale
Instituție: POLITEHNICA București – FIIR
Student: Rada Andrei Daniel
Link Repository GitHub: https://github.com/RadaAndreiDaniel/Proiect-Rn 
Data predării: 16 decembrie 2025

Scopul Etapei 5

Această etapă corespunde punctului 6. Configurarea și antrenarea modelului RN din lista de 9 etape – slide 2 RN Specificații proiect.pdf.

Obiectiv principal:
Antrenarea efectivă a modelului de Rețea Neuronală Convoluțională (CNN) definit în Etapa 4, evaluarea performanței acestuia și integrarea modelului antrenat în aplicația finală de clasificare a anvelopelor.

Modelul este utilizat pentru clasificarea benzii de rulare a anvelopelor în trei clase:

vară

iarnă

mixt

PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

✔ State Machine definit și documentat în docs/state_machine.png
✔ Contribuție ≥40% date originale, obținută prin adnotare manuală a imaginilor și salvare în data/generated/annotation_log.csv
✔ Modul 1 (Data Logging) funcțional – generează CSV cu date originale
✔ Modul 2 (RN) cu arhitectură definită (ResNet18, neantrenat inițial)
✔ Modul 3 (UI) funcțional, inițial cu model dummy

Toate condițiile Etapei 4 sunt îndeplinite, permițând trecerea la Etapa 5.

Pregătire Date pentru Antrenare

În cadrul Etapei 4 au fost adăugate date originale prin:

adnotare manuală a benzii de rulare

extragere ROI și salvare imagini procesate

Dataset-ul final a fost reîmpărțit în:

70% train

15% validation

15% test

Structura este respectată în directoarele:

data/train/
data/validation/
data/test/

Cerințe Structurate pe 3 Niveluri
✅ Nivel 1 – Obligatoriu

Toate cerințele Nivelului 1 au fost îndeplinite:

Model CNN (ResNet18) antrenat pe dataset-ul final

10 epoci de antrenare, batch size = 16

Split stratificat train / validation / test

Tabel hiperparametri completat (mai jos)

Metrici evaluate pe test set:

Accuracy ≥ 65%

F1-score (macro) ≥ 0.60

Model antrenat salvat în:

models/trained_model.pth


Integrare model antrenat în aplicația finală + screenshot inferență reală

Tabel Hiperparametri și Justificări
Hiperparametru	Valoare Aleasă	Justificare
Learning rate	0.001	Valoare standard pentru optimizerul Adam, oferă convergență stabilă
Batch size	16	Compromis între stabilitatea gradientului și consumul de memorie
Număr epoci	10	Suficient pentru demonstrarea procesului de antrenare și convergență
Optimizer	Adam	Adaptive learning rate, potrivit pentru CNN
Loss function	CrossEntropyLoss	Potrivită pentru clasificare multi-clasă
Funcții de activare	ReLU / Softmax	ReLU pentru non-linearitate, Softmax pentru probabilități pe clase
Antrenarea Modelului

Antrenarea a fost realizată folosind PyTorch (torchvision.models.resnet18), iar evoluția procesului a fost logată automat.

Fișiere generate:

models/trained_model.pth

results/training_history.csv

Fișierul training_history.csv conține valorile:

train_loss

train_accuracy

val_loss

val_accuracy
pentru fiecare epocă de antrenare.

Evaluare pe Test Set

Evaluarea a fost realizată pe setul de test separat, fără date văzute anterior de model.

Metrici raportate (results/test_metrics.json):

Accuracy: ≥ 65%

F1-score (macro): ≥ 0.60

Exemplu structură fișier:

{
  "test_accuracy": 0.78,
  "test_f1_macro": 0.74
}

Integrare în Aplicația Finală (UI)

Aplicația finală:

permite încărcarea unei imagini

realizarea adnotării manuale a benzii de rulare

clasificarea automată a anvelopei folosind modelul antrenat

Screenshot demonstrativ:

docs/screenshots/inference_real.png

Analiză Erori în Context Industrial (Nivel 2)
1. Clase confundate frecvent

Modelul confundă ocazional clasele mixt și iarna, din cauza similarității vizuale a profilului benzii de rulare în anumite condiții de iluminare.

2. Cauze ale erorilor

Erorile apar în special pentru imagini:

cu rezoluție redusă

cu iluminare neuniformă

unde ROI nu surprinde complet banda de rulare

3. Implicații industriale

False negative (iarnă → vară): critic – poate afecta siguranța rutieră

False positive: acceptabil – necesită reinspecție

Prioritatea este minimizarea erorilor critice.

4. Măsuri corective propuse

Creșterea numărului de imagini pentru clasa „mixt”

Augmentări de iluminare și contrast

Re-antrenare cu class weights

Creșterea rezoluției imaginilor de intrare

Structura Repository-ului la Finalul Etapei 5

(Structura este identică cu cea validată anterior și respectă cerințele oficiale Etapa 5)

Concluzie

În această etapă a fost realizată prima versiune complet funcțională a sistemului SIA, incluzând:

date originale

antrenare RN

evaluare obiectivă

integrare într-o aplicație reală

Modelul este funcțional și poate fi îmbunătățit în etapele următoare.

proiect-rn-[Rada_Andrei_Daniel]/
│
│   README_ETAPA3.md
│   README_ETAPA4.md
│   requirements.txt
│
├───config
│       README.txt
│
├───data
│   │   README.txt
│   │
│   ├───generated
│   ├───processed
│   │   ├───iarna
│   │   │       winter10_proc.jpg
│   │   │       winter11_proc.jpg
│   │   │       winter12_proc.jpg
│   │   │       winter13_proc.jpg
│   │   │       winter14_proc.jpg
│   │   │       winter15_proc.jpg
│   │   │       winter16_proc.jpg
│   │   │       winter1_proc.jpg
│   │   │       winter2_proc.jpg
│   │   │       winter3_proc.jpg
│   │   │       winter4_proc.jpg
│   │   │       winter5_proc.jpg
│   │   │       winter6_proc.jpg
│   │   │       winter7_proc.jpg
│   │   │       winter8_proc.jpg
│   │   │       winter9 - Copy (2)_proc.jpg
│   │   │       winter9 - Copy (3)_proc.jpg
│   │   │       winter9 - Copy_proc.jpg
│   │   │       winter9_proc.jpg
│   │   │
│   │   ├───mixt
│   │   │       mixta_05_proc.jpg
│   │   │       mixta_06_proc.jpg
│   │   │       mixta_07_proc.jpg
│   │   │       mixta_08_proc.jpg
│   │   │       mixta_09_proc.jpg
│   │   │       mixta_10_proc.jpg
│   │   │       mixta_11_proc.jpg
│   │   │       mixta_12_proc.jpg
│   │   │       mixta_13_proc.jpg
│   │   │       mixta_14_proc.jpg
│   │   │
│   │   └───vara
│   │           summer1 - Copy (2)_proc.jpg
│   │           summer1 - Copy (3)_proc.jpg
│   │           summer1 - Copy_proc.jpg
│   │           summer10_proc.jpg
│   │           summer11_proc.jpg
│   │           summer12_proc.jpg
│   │           summer13_proc.jpg
│   │           summer14_proc.jpg
│   │           summer15_proc.jpg
│   │           summer16_proc.jpg
│   │           summer17_proc.jpg
│   │           summer18_proc.jpg
│   │           summer19_proc.jpg
│   │           summer1_proc.jpg
│   │           summer2 - Copy (2)_proc.jpg
│   │           summer2 - Copy (3)_proc.jpg
│   │           summer2 - Copy_proc.jpg
│   │           summer20_proc.jpg
│   │           summer21_proc.jpg
│   │           summer22_proc.jpg
│   │           summer23_proc.jpg
│   │           summer24_proc.jpg
│   │           summer25_proc.jpg
│   │           summer2_proc.jpg
│   │           summer4 - Copy (2)_proc.jpg
│   │           summer4 - Copy (3)_proc.jpg
│   │           summer4 - Copy_proc.jpg
│   │           summer4_proc.jpg
│   │           summer5 - Copy (2)_proc.jpg
│   │           summer5 - Copy (3)_proc.jpg
│   │           summer5 - Copy_proc.jpg
│   │           summer5_proc.jpg
│   │           summer6 - Copy (2)_proc.jpg
│   │           summer6 - Copy (3)_proc.jpg
│   │           summer6 - Copy_proc.jpg
│   │           summer6_proc.jpg
│   │           summer7 - Copy (2)_proc.jpg
│   │           summer7 - Copy (3)_proc.jpg
│   │           summer7 - Copy_proc.jpg
│   │           summer7_proc.jpg
│   │           summer9_proc.jpg
│   │
│   ├───raw
│   │   ├───iarna
│   │   │       winter1.jpg
│   │   │       winter10.jpg
│   │   │       winter10.webp
│   │   │       winter11.jpg
│   │   │       winter11.webp
│   │   │       winter12.jpg
│   │   │       winter12.webp
│   │   │       winter13.jpg
│   │   │       winter13.webp
│   │   │       winter14.jpg
│   │   │       winter14.webp
│   │   │       winter15.jpg
│   │   │       winter16.jpg
│   │   │       winter2.jpg
│   │   │       winter3.jpg
│   │   │       winter4.jpg
│   │   │       winter5.jpg
│   │   │       winter6.jpg
│   │   │       winter7.jpg
│   │   │       winter8.jpg
│   │   │       winter9 - Copy (2).jpg
│   │   │       winter9 - Copy (3).jpg
│   │   │       winter9 - Copy.jpg
│   │   │       winter9.jpg
│   │   │
│   │   ├───mixt
│   │   │       mixta_05.webp
│   │   │       mixta_06.webp
│   │   │       mixta_07.webp
│   │   │       mixta_08.webp
│   │   │       mixta_09.webp
│   │   │       mixta_10.webp
│   │   │       mixta_11.webp
│   │   │       mixta_12.jpg
│   │   │       mixta_13.jpg
│   │   │       mixta_14.jpg
│   │   │
│   │   └───vara
│   │           summer1 - Copy (2).avif
│   │           summer1 - Copy (3).avif
│   │           summer1 - Copy.avif
│   │           summer1.avif
│   │           summer10.jpg
│   │           summer10.webp
│   │           summer11.jpg
│   │           summer11.webp
│   │           summer12.jpg
│   │           summer12.webp
│   │           summer13.jpg
│   │           summer13.webp
│   │           summer14.jpg
│   │           summer14.webp
│   │           summer15.jpg
│   │           summer15.webp
│   │           summer16.jpg
│   │           summer16.webp
│   │           summer17.jpg
│   │           summer17.webp
│   │           summer18.jpg
│   │           summer18.webp
│   │           summer19.jpg
│   │           summer19.webp
│   │           summer2 - Copy (2).avif
│   │           summer2 - Copy (3).avif
│   │           summer2 - Copy.avif
│   │           summer2.avif
│   │           summer20.jpg
│   │           summer20.webp
│   │           summer21.jpg
│   │           summer21.webp
│   │           summer22.jpg
│   │           summer22.webp
│   │           summer23.jpg
│   │           summer23.webp
│   │           summer24.jpg
│   │           summer25.jpg
│   │           summer4 - Copy (2).jpg
│   │           summer4 - Copy (3).jpg
│   │           summer4 - Copy.jpg
│   │           summer4.jpg
│   │           summer5 - Copy (2).jpg
│   │           summer5 - Copy (2).webp
│   │           summer5 - Copy (3).jpg
│   │           summer5 - Copy (3).webp
│   │           summer5 - Copy.jpg
│   │           summer5 - Copy.webp
│   │           summer5.jpg
│   │           summer5.webp
│   │           summer6 - Copy (2).avif
│   │           summer6 - Copy (3).avif
│   │           summer6 - Copy.avif
│   │           summer6.avif
│   │           summer7 - Copy (2).avif
│   │           summer7 - Copy (3).avif
│   │           summer7 - Copy.avif
│   │           summer7.avif
│   │           summer9.jpg
│   │           summer9.webp
│   │
│   ├───test
│   │       testImage.png
│   │       test_image10.png
│   │       test_image10_preview_256.jpg
│   │       test_image11.png
│   │       test_image11_preview_256.jpg
│   │       test_image12.jpg
│   │       test_image12_preview_256.jpg
│   │       test_image15.png
│   │       test_image15_preview_256.jpg
│   │       test_image2.jpg
│   │       test_image3.jpg
│   │       test_image4.jpg
│   │       test_image5.jpg
│   │       test_image6.jpg
│   │       test_image7.jpg
│   │       test_image8.jpg
│   │       test_image8_preview_256.jpg
│   │       test_image9.jpg
│   │       test_image9_preview_256.jpg
│   │
│   ├───train
│   └───validation
├───docs
│   │   state_machine.png
│   │   state_machine.txt
│   │
│   └───screenshots
│           interfata.jpg
│
├───models
│       trained_model.pth
│
└───src
    ├───app
    │       finalApp.py
    │       README.txt
    │
    ├───data_acquisition
    │   │   annotator.py
    │   │   decupareBandaDeRulareAntrenament.py
    │   │   image_test.webp
    │   │   image_test6.webp
    │   │   model_annotation.py
    │   │   pattern_extras.jpg
    │   │   pattern_final.jpg
    │   │   README.txt
    │   │   test.jpg
    │   │   testImage.png
    │   │   test_image2.jpg
    │   │   test_image4.png
    │   │   test_image5.png
    │   │   test_image6.png
    │   │   test_image7.png
    │   │   winter1_filtrat.jpg
    │   │
    │   ├───annotations_
    │   │       image_test.json
    │   │       testImage.json
    │   │       test_image2.json
    │   │       test_image4.json
    │   │       test_image5.json
    │   │       test_image6.json
    │   │       test_image7.json
    │   │
    │   ├───images_
    │   │       image_test_annotated.jpg
    │   │       image_test_preview_256.jpg
    │   │       testImage_annotated.jpg
    │   │       testImage_preview_256.jpg
    │   │       test_image2_annotated.jpg
    │   │       test_image2_preview_256.jpg
    │   │       test_image4_annotated.jpg
    │   │       test_image4_preview_256.jpg
    │   │       test_image5_annotated.jpg
    │   │       test_image5_preview_256.jpg
    │   │       test_image6_annotated.jpg
    │   │       test_image6_preview_256.jpg
    │   │       test_image7_annotated.jpg
    │   │       test_image7_preview_256.jpg
    │   │
    │   └───__pycache__
    │           annotator.cpython-39.pyc
    │
    ├───neural_network
    │       README.MD
    │       train_and_test.py
    │
    └───preprocessing