Disciplina: Rețele Neuronale
Instituție: POLITEHNICA București – FIIR
Student: Rada Andrei Daniel
Link Repository GitHub: https://github.com/RadaAndreiDaniel/Proiect-Rn 
Data predării: 20 ianuarie 2026


## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN, analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă.

**CONTEXT IMPORTANT:** 
- Etapa 6 **ÎNCHEIE ciclul formal de dezvoltare** al proiectului
- Aceasta este **ULTIMA VERSIUNE înainte de examen** pentru care se oferă **FEEDBACK**
- Pe baza feedback-ului primit, componentele din **TOATE etapele anterioare** pot fi actualizate iterativ

**Pornire obligatorie:** Modelul antrenat și aplicația funcțională din Etapa 5:
- Model antrenat cu metrici baseline (Accuracy ≥65%, F1 ≥0.60)
- Cele 3 module integrate și funcționale
- State Machine implementat și testat

---

## MESAJ CHEIE – ÎNCHEIEREA CICLULUI DE DEZVOLTARE ȘI ITERATIVITATE

**ATENȚIE: Etapa 6 ÎNCHEIE ciclul de dezvoltare al aplicației software!**

**CE ÎNSEAMNĂ ACEST LUCRU:**
- Aceasta este **ULTIMA VERSIUNE a proiectului înainte de examen** pentru care se mai poate primi **FEEDBACK** de la cadrul didactic
- După Etapa 6, proiectul trebuie să fie **COMPLET și FUNCȚIONAL**
- Orice îmbunătățiri ulterioare (post-feedback) vor fi implementate până la examen

**PROCES ITERATIV – CE RĂMÂNE VALABIL:**
Deși Etapa 6 încheie ciclul formal de dezvoltare, **procesul iterativ continuă**:
- Pe baza feedback-ului primit, **TOATE componentele anterioare pot și trebuie actualizate**
- Îmbunătățirile la model pot necesita modificări în Etapa 3 (date), Etapa 4 (arhitectură) sau Etapa 5 (antrenare)
- README-urile etapelor anterioare trebuie actualizate pentru a reflecta starea finală

**CERINȚĂ CENTRALĂ Etapa 6:** Finalizarea și maturizarea **ÎNTREGII APLICAȚII SOFTWARE**:

1. **Actualizarea State Machine-ului** (threshold-uri noi, stări adăugate/modificate, latențe recalculate)
2. **Re-testarea pipeline-ului complet** (achiziție → preprocesare → inferență → decizie → UI/alertă)
3. **Modificări concrete în cele 3 module** (Data Logging, RN, Web Service/UI)
4. **Sincronizarea documentației** din toate etapele anterioare

**DIFERENȚIATOR FAȚĂ DE ETAPA 5:**
- Etapa 5 = Model antrenat care funcționează
- Etapa 6 = Model OPTIMIZAT + Aplicație MATURIZATĂ + Concluzii industriale + **VERSIUNE FINALĂ PRE-EXAMEN**


**IMPORTANT:** Aceasta este ultima oportunitate de a primi feedback înainte de evaluarea finală. Profitați de ea!

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

**Înainte de a începe Etapa 6, verificați că aveți din Etapa 5:**

- [ ] **Model antrenat** salvat în `models/trained_model.h5` (sau `.pt`, `.lvmodel`)
- [ ] **Metrici baseline** raportate: Accuracy ≥65%, F1-score ≥0.60
- [ ] **Tabel hiperparametri** cu justificări completat
- [ ] **`results/training_history.csv`** cu toate epoch-urile
- [ ] **UI funcțional** care încarcă modelul antrenat și face inferență reală
- [ ] **Screenshot inferență** în `docs/screenshots/inference_real.png`
- [ ] **State Machine** implementat conform definiției din Etapa 4

**Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 5 înainte de a continua.**

---

## Cerințe


1. Implementarea a minimum 4 experimente de optimizare

Au fost realizate patru experimente distincte de antrenare, prin variația controlată a modului de echilibrare a claselor și a strategiilor de optimizare. Experimentele au urmărit impactul distribuției datelor asupra performanței finale:

Experiment 1: egalizarea tuturor celor trei clase (iarna, mixt, vara);

Experiment 2: antrenare fără echilibrare a claselor;

Experiment 3: echilibrare parțială, fără optimizări suplimentare (augmentări / early stopping);

Experiment 4 (final): echilibru doar între iarna și vara, cu clasa mixt păstrată ca intermediară, augmentări vizuale, sampler ponderat și early stopping.

Această abordare a permis identificarea configurației optime și a limitărilor fiecărei strategii.



2.Tabel comparativ experimente** cu metrici și observații

| **Exp#**            | **Modificare față de Baseline (Etapa 5)**                                                          | **Accuracy** | **F1-score (macro)** | **Timp antrenare** | **Observații**                                                              |
| ------------------- | -------------------------------------------------------------------------------------------------- | ------------ | -------------------- | ------------------ | --------------------------------------------------------------------------- |
| **Baseline (Et.5)** | Model inițial (setări Etapa 5, înainte de optimizarea dataset-ului)                                | **0.875**    | **0.619**            | n/a                | Acuratețe mare, dar modelul tinde să supra-clasifice *mixt*                 |
| **Exp 1**           | Echilibrare completă pe 3 clase (*iarna*, *mixt*, *vara*)                                          | 0.875        | 0.619                | 10 Epoch     | Comportament dezechilibrat: *mixt* „atrage” multe predicții                 |
| **Exp 2**           | Fără echilibrare a claselor (fără tratament class imbalance)                                       | ~0.47        | ~0.38                | 15 Epoch         | Modelul favorizează clasa dominantă; performanță slabă                      |
| **Exp 3**           | Echilibrare parțială + dataset ajustat (fără optimizări avansate)                                  | ~0.63        | ~0.61                | 15 Epoch         | Îmbunătățire față de Exp 2, dar instabil pe clasa *mixt*                    |
| **Exp 4 (FINAL)**   | Echilibru iarnă–vară, *mixt* păstrată intermediar + augmentări + sampler ponderat + early stopping | **0.7857**   | **0.7323**           | 50 Epoch        | **BEST** – cel mai bun echilibru global; model ales pentru aplicația finală |


Configurația finală a fost aleasă deoarece oferă rezultate bune per total și se comportă mai echilibrat în cazul claselor inegale, fără să favorizeze mereu aceeași clasă. (mixt)



Am ales Exp 4 ca model final pentru că:
1. A obținut pe setul de test Accuracy = 0.7857 și F1-score (macro) = 0.7323, valori care reflectă mai bine performanța în contextul claselor inegale
2. Modelul obține rezultate echilibrate între clasele „iarna”, „mixt” și „vara”, cu F1-score-uri de 0.7368, 0.5000 și 0.9600, fără a favoriza constant o singură clasă
3. Early stopping-ul a oprit antrenarea la 6 epoci, cu best epoch = 1 și val_loss minim = 0.5275, indicând stabilitate și evitarea overfitting-ului
4. Diferența mică dintre val_loss (0.5275) și test_loss (0.5260) confirmă o bună capacitate de generalizare pe date nevăzute


Resurse învățare rapidă - Optimizare:**
- roboflow(https://www.youtube.com/watch?v=xfu44phj9d0) 
- score Macro: https://staff.fmi.uvt.ro/~daniela.zaharie/am2016/curs/curs12/am2016_slides12_RN.pdf




## 2. Analiza Detaliată a Performanței

### Interpretare Confusion Matrix:

**Clasa cu cea mai bună performanță:** vara
- Precision: 92.31%
- Recall: 100.00%
- Explicație: Toate cele 12 exemple din clasa „vara” au fost clasificate corect, fără confuzii cu celelalte clase. Aceasta sugerează că această clasă are caracteristici mai distincte față de „iarna” și „mixt”.

**Clasa cu cea mai slabă performanță:** mixt
- Precision: 37.50%
- Recall: 75.00%
- Explicație: Deși 3 din 4 exemple „mixt” au fost identificate corect, modelul confundă frecvent alte clase cu „mixt”, ceea ce duce la o precizie scăzută. De asemenea, setul are doar 4 exemple pentru această clasă, ceea ce influențează stabilitatea rezultatelor.

**Confuzii principale:**
1. Clasa iarna confundată cu clasa mixt în 41.67% din cazuri (5 din 12)
   - Cauză: Posibilă similaritate între caracteristicile celor două clase, mai ales în situații de tranziție sezonieră.
   - Impact industrial: Poate duce la aplicarea unor decizii sau procese specifice clasei „mixt” în loc de „iarna”.

2. Clasa mixt confundată cu clasa vara în 25.00% din cazuri (1 din 4)
   - Cauză: Overlap între caracteristicile celor două clase și număr redus de exemple „mixt”.
   - Impact industrial: Posibile clasificări eronate în condiții intermediare, afectând acuratețea deciziilor automate.




2.2 Analiza Detaliată a 5 Exemple Greșite

### Exemplu #3 – iarna clasificată ca mixt

**Context:** Imagine cu caracteristici specifice sezonului rece, dar cu elemente de tranziție
**Output RN:** Predicție: mixt

**Analiză:**
Modelul confundă frecvent „iarna” cu „mixt” (5 din 12 cazuri – 41.67%).
Este probabil ca imaginea să conțină trăsături comune ambelor clase, iar
modelul să fi detectat caracteristici intermediare.

**Implicație:**
Confuzia reduce recall-ul pentru „iarna” (58.33%), afectând consistența clasificării.

**Soluție:**
1. Mai multe exemple clare pentru clasa „iarna”
2. Augmentări care să evidențieze trăsăturile distinctive ale sezonului rece

### Exemplu #7 – iarna clasificată ca mixt

**Context:** Mostră cu distribuție neuniformă a caracteristicilor
**Output RN:** Predicție: mixt

**Analiză:**
Modelul pare sensibil la cazurile de graniță între clase.
Caracteristicile dominante nu sunt suficient de puternice pentru a susține
decizia „iarna”.

**Implicație:**
Scade precizia globală și contribuie la eroarea sistematică iarna → mixt.

**Soluție:**
1. Regularizare suplimentară
2. Feature engineering pentru a separa mai clar cele două clase

### Exemplu #11 – iarna clasificată ca mixt

**Context:** Date apropiate de media setului „mixt”
**Output RN:** Predicție: mixt

**Analiză:**
Modelul a învățat o reprezentare mai largă pentru „mixt”,
ceea ce duce la absorbirea unor exemple „iarna” în această clasă.

**Implicație:**
Dezechilibru între clase (12–4–12) poate influența frontiera de decizie.

**Soluție:**
1. Ponderare a claselor în funcția de pierdere
2. Creșterea numărului de exemple „mixt” pentru stabilizare

### Exemplu #15 – mixt clasificată ca vara

**Context:** Exemplu mixt cu caracteristici dominante de tip „vara”
**Output RN:** Predicție: vara

**Analiză:**
1 din 4 exemple „mixt” a fost clasificat ca „vara” (25%).
Recall pentru „mixt” este 75%, dar precizia este scăzută (37.5%),
ceea ce indică instabilitate din cauza numărului mic de exemple.

**Implicație:**
Clasa „mixt” este cea mai vulnerabilă și instabilă statistic.

**Soluție:**
1. Creșterea dataset-ului pentru „mixt”
2. Aplicarea de augmentări specifice acestei clase

### Exemplu #18 – iarna clasificată ca mixt

**Context:** Caz limită între sezon rece și tranziție
**Output RN:** Predicție: mixt

**Analiză:**
Modelul tinde să fie conservator și să aleagă „mixt” când
nivelul de certitudine este scăzut.

**Implicație:**
Această tendință explică precision scăzut pentru „mixt”
și recall redus pentru „iarna”.

**Soluție:**
1. Ajustarea pragului de decizie
2. Analiză a distribuției probabilităților pentru optimizarea threshold-ului
3. Nu reprezinta o problema mare, cea mai mare problema ar fi fost daca iarna ar fi fost identificata ca si vara ceea ce ar fi dus la o eroare mare a programului


3. Optimizarea Parametrilor și Experimentare
3.1 Strategia de Optimizare
### Strategie de optimizare adoptată:

**Abordare:** Manuală (experimente controlate succesive)

**Axe de optimizare explorate:**

1. **Arhitectură:**
   - Configurație baseline (Etapa 5)
   - Adăugare +1 strat ascuns (128 neuroni) → cea mai bună performanță

2. **Regularizare:**
   - Dropout 0.3 → 0.5
   - Early stopping (oprit la epoca 6)

3. **Learning rate:**
   - 0.0001 → 0.001
   - Scheduler cu reducere la 0.00015 după stagnare

4. **Augmentări:**
   - Nu s-au aplicat augmentări complexe
   - Accent pe stabilizarea clasificării clasei „mixt”

5. **Batch size:**
   - 10 → 50 (a redus stabilitatea și F1-score)

**Criteriu de selecție model final:**
Maximizarea F1-score (macro) pentru a evita favorizarea unei clase,
în special pentru a preveni supra-clasificarea în „mixt”.

**Buget computațional:**
~6 experimente principale
Antrenare medie: 15 minute /experiment (CPU)

3.3 Raport Final Optimizare
### Raport Final Optimizare

**Model baseline (Etapa 5):**
- Accuracy: 0.65
- F1-score: 0.63

**Model optimizat (Etapa 6):**
- Accuracy: 0.7857
- F1-score (macro): 0.7323

**Configurație finală aleasă:**
- Arhitectură: MLP cu +1 strat ascuns (128 neuroni)
- Learning rate: 0.0003 → 0.00015 (scheduler activ)
- Batch size: 32
- Regularizare: Dropout + Early Stopping
- Epoci: 6 (early stopping, best epoch = 1)

**Îmbunătățiri cheie:**
1. Adăugarea stratului ascuns suplimentar → creștere F1-score
2. Ajustarea learning rate-ului cu scheduler → stabilitate mai bună
3. Optimizarea configurației pentru a reduce tendința de clasificare excesivă in clasa "mixt"



## 4. Agregarea Rezultatelor și Vizualizări

**`results/final_metrics.json`** - metrici finale:
{
  "model": "optimized_model_final.pt",
  "test_accuracy": 0.7857,
  "test_f1_macro": 0.7323,
  "test_precision_macro": 0.7660,
  "test_recall_macro": 0.7778,
  "false_negative_rate": 0.2143,
  "false_positive_rate": 0.1223,
  "inference_latency_ms": null,
  "improvement_vs_baseline": {
    "accuracy": "+6.57%",
    "f1_score": "+5.23%",
    "latency": "N/A"
  }
}


### 4.2 Vizualizări Obligatorii

Salvați în `docs/results/`:

- [ ] `confusion_matrix_optimized.png` - Confusion matrix model final
- [ ] `learning_curves_final.png` - Loss și accuracy vs. epochs



### Evaluare sintetică a proiectului

**Obiective atinse:**
- [x] Model RN funcțional cu accuracy 78.57% pe test set
- [x] Integrare completă în aplicație software (3 module)
- [x] State Machine implementat și actualizat
- [x] Pipeline end-to-end testat și documentat
- [x] UI demonstrativ cu inferență reală
- [x] Documentație completă pe toate etapele

**Obiective parțial atinse:**
- [x] Performanță dezechilibrată între clase (F1 „mixt” = 0.50)
- [x] Stabilitate redusă pentru clasa „mixt” din cauza numărului mic de exemple (4 în test set)

**Obiective neatinse:**
- [ ] Lipseste precizia mare asupra clasei mixt deoarce exita riscul sa fie fals pozitiva si in cazul calselor vara si iarna datorita faptului ca inglobeaza caracteristici din ambele clase.

### Limitări tehnice ale sistemului

1. **Limitări date:**
   - Dataset relativ mic (171 train / 28 val / 28 test), ceea ce limitează capacitatea de generalizare
   - Dezechilibru între clase, în special număr redus pentru „mixt” (doar 4 exemple în test set)
   - Posibil overlap natural între clasele „iarna” și „mixt”, ceea ce îngreunează separarea clară

2. **Limitări model:**
   - Tendință de a clasifica exemple ambigue ca „mixt”, afectând precision (37.5%)
   - Recall scăzut pentru „iarna” (58.33%), din cauza confuziei frecvente cu „mixt”
   - Modelul este sensibil la cazurile de graniță între clase

3. **Limitări infrastructură:**
   - Antrenarea s-a realizat pe CPU, limitând posibilitatea explorării unui număr mai mare de experimente
   - Nu s-a realizat optimizare pentru deployment pe hardware dedicat sau aplicații în timp real

4. **Limitări validare:**
   - Test set redus (28 exemple), ceea ce poate influența stabilitatea metricilor
   - Lipsa validării pe un set extern complet independent


### Direcții viitoare de dezvoltare

**Pe termen scurt (1-3 luni):**
1. Colectare a minimum 50–100 exemple suplimentare pentru clasa „mixt” pentru echilibrarea dataset-ului
2. Implementare class weighting în funcția de pierdere pentru a reduce biasul către clasele dominante
3. Ajustarea pragurilor de decizie (threshold tuning) pentru a stabiliza predicțiile în cazurile ambigue
4. Extinderea test set-ului pentru evaluare mai robustă a performanței reale

**Pe termen mediu (3-6 luni):**
1. Testarea unor arhitecturi mai complexe (ex: rețele mai adânci sau modele pre-antrenate)
2. Implementarea unui sistem de monitorizare a performanței (detecție drift distribuție date)
3. Deployment pe hardware dedicat și analiză latență în condiții reale
4. Automatizarea procesului de re-antrenare pe măsură ce se colectează date noi


### Direcții viitoare de dezvoltare

**Pe termen scurt (1-3 luni):**
1. Colectare date adiționale pentru clasa „mixt” (ex: +50–100 exemple) și re-echilibrare (class weights / sampling)
2. Stabilizare predicții pentru cazurile ambigue prin threshold tuning și calibrare probabilități
3. Extindere pipeline pentru input video: captură live (RTSP/USB), preprocesare cadre și inferență în timp real
4. Optimizare latență prin batch=1, model export (TorchScript/ONNX) și reducerea dimensiunii modelului

**Pe termen mediu (3-6 luni):**
1. Implementare detecție obiecte în trafic (ex: vehicule) + tracking (ID-uri persistente pe cadre)
2. Integrare modul recunoaștere plăcuță: detecție plăcuță + OCR pentru număr (ANPR)
3. Validare în condiții reale (noapte/zi, ploaie, unghiuri diferite) și augmentări specifice (motion blur, low-light)
4. Implementare logging + monitoring (drift detection, alertare când scade performanța)


### Lecții învățate pe parcursul proiectului

**Tehnice:**
1. F1-score (macro) este mai relevant decât accuracy în cazul claselor inegale.
2. Dezechilibrul dataset-ului influențează puternic frontiera de decizie (ex: tendința de a clasifica excesiv în „mixt”).
3. Creșterea capacității modelului (strat suplimentar) a avut impact mai mare decât simpla modificare a learning rate-ului.
4. Early stopping ajută la prevenirea overfitting-ului chiar și pe dataset-uri mici.

**Proces:**
1. Analiza confusion matrix-ului a oferit informații mai valoroase decât metricile globale.
2. Experimentele controlate, schimbând un singur parametru o dată, au fost mai eficiente decât modificările simultane.
3. Iterațiile succesive și compararea clară a experimentelor au ajutat la alegerea unei configurații stabile.

**Colaborare / Organizare:**
1. Structurarea pe etape (model → evaluare → optimizare → integrare) a făcut proiectul mai ușor de gestionat.
2. Documentarea rezultatelor imediat după fiecare experiment a simplificat analiza finală.
