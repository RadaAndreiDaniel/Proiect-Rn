"""
Neural Network Module – Tire Tread Classification
-------------------------------------------------

Acest modul definește și utilizează o rețea neuronală convoluțională (CNN)
pentru clasificarea tipului de anvelopă (vară, iarnă, mixtă) pe baza
imaginii benzii de rulare.

Scopul acestui fișier, în Etapa 4 a proiectului, este de a demonstra
arhitectura rețelei neuronale și integrarea acesteia în pipeline-ul
aplicației SIA, nu de a obține performanță maximă.

Caracteristici principale:
- Rețea neuronală de tip CNN
- Arhitectură bazată pe ResNet18
- Clasificare în 3 clase
- Model definit, compilat și reutilizabil
- Posibilitatea de salvare și reîncărcare a modelului

Input:
- Imagini RGB ale benzii de rulare
- Dimensiune standard: 224×224 px

Output:
- Probabilități pentru clasele: vară, iarnă, mixt

Notă:
În această etapă, modelul poate avea ponderi inițializate aleator sau
pre-antrenate, fără a fi optimizat. Performanța nu reprezintă un criteriu
de evaluare în Etapa 4.

Autor: Rada Andrei Daniel
"""
