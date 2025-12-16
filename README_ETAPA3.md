📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

Disciplina: Rețele Neuronale
Instituție: POLITEHNICA București – FIIR
Student: Rada Andrei Daniel
Data: 25 Noiembrie 2025

Introducere

Acest document descrie activitățile realizate în Etapa 3 a proiectului la disciplina Rețele Neuronale, având ca scop analiza și pregătirea setului de date necesar antrenării unui model de rețea neuronală.

Problema abordată constă în clasificarea tipului de anvelopă (vară, iarnă, mixtă) pe baza benzii de rulare, informație extrasă din imagini reale. În această etapă accentul este pus exclusiv pe date: calitatea lor, structura și modul de preprocesare.

1. Structura Repository-ului GitHub (versiunea Etapei 3)
Proiect_Rada_Andrei_Daniel_Rn/
├── README.md
├── docs/
│   └── datasets/              # descriere seturi de date și surse
├── data/
│   ├── raw/                   # imagini brute cu anvelope
│   ├── processed/             # imagini preprocesate (ROI)
│   ├── train/                 # set de instruire
│   ├── validation/            # set de validare
│   └── test/                  # set de testare
├── src/
│   ├── preprocessing/         
│   ├── data_acquisition/  
│   │ 		 │ annotator.py
│   │ 		 └── model_annotation.py    
│   └── neural_network/        
├── config/
└── requirements.txt

2. Descrierea Setului de Date
2.1 Sursa datelor

Origine: imagini reale cu anvelope auto

Modul de achiziție: ☑ Fișier extern
		    ☑ Poze proprii

Perioada colectării: 1 Noiembrie 2024 –  25 Noiembrie 2025

Condiții de colectare: iluminare variabilă, poziționare diferită a anvelopei

Imaginile brute sunt stocate în directorul data/raw/.

2.2 Caracteristicile dataset-ului

Număr total de observații: ~30 imagini

Număr de caracteristici: datele sunt de tip imagine (fără features numerice explicite)

Tipuri de date: ☑ Imagini

Format fișiere: ☑ JPG / ☑ PNG

Clasele definite în dataset:

anvelope de vară

anvelope de iarnă

anvelope mixte (all-season)

2.3 Descrierea caracteristicilor

În cadrul acestui proiect, caracteristicile sunt reprezentate de informația vizuală conținută în banda de rulare a anvelopei.

Caracteristică	Tip	Descriere
Bandă de rulare	Imagine	Regiune de interes extrasă din imaginea originală
Pattern caneluri	Vizual	Dispunerea și orientarea canalelor
Textură profil	Vizual	Indicator al tipului de anvelopă
Adâncime relativă	Vizual	Diferențe între clase

Fișier recomandat: data/README.md

3. Analiza Exploratorie a Datelor (EDA) – Sintetic
3.1 Analize realizate

Pentru dataset-ul de tip imagine au fost realizate:

analiza distribuției imaginilor pe clase

verificarea rezoluțiilor

analiză vizuală a variațiilor de textură

identificarea imaginilor neclare sau necorespunzătoare

3.2 Analiza calității datelor

Valori lipsă: nu se aplică (imagini)

Imagini invalide: eliminate în etapa de curățare

Dezechilibru de clasă: ușor prezent, acceptabil pentru această etapă

3.3 Probleme identificate

variații mari de iluminare

fundal diferit între imagini

poziționare neuniformă a anvelopei

Aceste probleme justifică necesitatea preprocesării și extragerii regiunii de interes.

4. Preprocesarea Datelor
4.1 Curățarea datelor

eliminarea imaginilor neclare sau incomplete

eliminarea duplicatelor

selecția manuală a imaginilor valide

4.2 Transformarea caracteristicilor

extragerea benzii de rulare (ROI)

conversia imaginilor la grayscale (unde este necesar)

redimensionarea imaginilor la dimensiune standard (256×256 px)

4.3 Structurarea seturilor de date

Împărțire utilizată:

~80% – train

~10% – validation

~10% – test

Principii respectate:

separarea strictă a setului de test

fără scurgere de informație între seturi

organizarea pe clase

4.4 Salvarea rezultatelor preprocesării

imaginile preprocesate sunt salvate în data/processed/

seturile de date sunt organizate în data/train/, data/validation/, data/test/

codul de preprocesare se află în src/preprocessing/

5. Fișiere Generate în Această Etapă

data/raw/ – imagini brute

data/processed/ – imagini preprocesate (ROI)

data/train/, data/validation/, data/test/

src/preprocessing/ – scripturi de preprocesare

data/README.md – descrierea dataset-ului



Concluzie

Etapa 3 a permis obținerea unui set de date curat, structurat și pregătit pentru etapa următoare, în care va fi definită și antrenată rețeaua neuronală pentru clasificarea tipului de anvelopă.