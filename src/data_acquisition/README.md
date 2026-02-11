#  Modul Data Acquisition – Adnotare Manuală Bandă de Rulare

## Descriere generală

Modulul **Data Acquisition** este responsabil de obținerea datelor originale
necesare antrenării și evaluării rețelei neuronale. În cadrul acestui proiect,
achiziția datelor se realizează prin **adnotare manuală asistată**, folosind
imagini reale cu anvelope auto.

Scopul principal al acestui modul este extragerea **benzii de rulare
(regiunea de interes – ROI)** din imaginea originală, reducând influența
fundalului, iluminării neuniforme și poziționării variabile a anvelopei.
Datele generate prin acest proces reprezintă contribuția originală a proiectului.

---

## Metoda de generare / achiziție a datelor

Achiziția datelor se realizează printr-un instrument software dezvoltat în Python,
implementat în fișierul `annotor.py`, care permite utilizatorului să:

1. Încarce o imagine cu o anvelopă auto
2. Indice manual reperele necesare delimitării benzii de rulare
3. Ajusteze selecția regiunii de interes (ROI)
4. Confirme adnotarea printr-o acțiune explicită (ex: apăsarea tastei ENTER)
5. Obțină automat o imagine decupată corespunzătoare benzii de rulare

Procesul este de tip **human-in-the-loop**, ceea ce asigură o delimitare precisă
și controlată a zonei relevante pentru clasificare, în special în situațiile
în care identificarea automată poate eșua sau produce rezultate incerte.

---

## Locația datelor generate

Imaginile rezultate în urma procesului de adnotare manuală sunt salvate în
directorul `data/generated/`, organizate pe clasele:

- `data/generated/vara/`
- `data/generated/iarna/`
- `data/generated/mixt/`

Această structură este compatibilă cu etapele ulterioare de preprocesare,
împărțire a setului de date și antrenare a rețelei neuronale.

---

## Parametri principali

- Format imagini: JPG / PNG  
- Conversie grayscale: aplicată după extragerea ROI  
- Redimensionare: realizată ulterior în etapa de preprocesare (256 × 256 px)  
- Metodă de selecție ROI: adnotare manuală asistată  
- Număr minim de observații generate: ≥ 100 imagini  

---

## Rulare modul

Modulul de achiziție a datelor poate fi rulat folosind comanda:

```bash
python annotor.py
