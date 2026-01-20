# 📥 Modul Data Acquisition – Adnotare Manuală Bandă de Rulare

## Descriere generală

Modulul **Data Acquisition** este responsabil de obținerea datelor necesare antrenării și testării rețelei neuronale. În cadrul acestui proiect, achiziția datelor se realizează prin **adnotare manuală asistată**, folosind imagini reale cu anvelope auto.

Scopul acestui modul este extragerea **benzii de rulare (regiunea de interes – ROI)** din imaginea originală, reducând influența fundalului, iluminării și poziționării anvelopei.

---

## Metoda de generare / achiziție a datelor

Achiziția datelor se realizează printr-un instrument software dezvoltat în Python, care permite utilizatorului să:

1. Încarce o imagine cu o anvelopă
2. Indice manual reperele necesare delimitării benzii de rulare
3. Confirme selecția printr-o acțiune explicită (ex: apăsarea tastei ENTER)
4. Obțină automat o imagine decupată a benzii de rulare (ROI)

Procesul este **human-in-the-loop**, ceea ce asigură o delimitare precisă a zonei relevante pentru clasificare.

Codul principal al acestui proces se află în:
