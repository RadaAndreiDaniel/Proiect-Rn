# 🖥️ UI / Aplicație – Clasificare Anvelopă

## Scop
Interfața oferă utilizatorului o aplicație simplă pentru:
- încărcarea unei **imagini** sau a unui **video** cu anvelopa;
- alegerea metodei de identificare a zonei (ROI): **YOLO** (automat) sau **MANUAL**;
- afișarea rezultatului de clasificare: **vară / iarnă / mixt** + scoruri (procente);
- afișarea imaginilor intermediare (bbox preview + crop/pattern prelucrat).

UI-ul este partea vizibilă a proiectului și rulează end-to-end împreună cu backend-ul Python.

---

## Ce face interfața (conform codului HTML/JS)
1. La pornire cere selectarea sezonului curent: `iarna` sau `vara` (vară/primăvară/toamnă).
2. Utilizatorul alege metoda:
   - `YOLO` = identificare automată ROI
   - `MANUAL` = selecție manuală ROI
3. Utilizatorul încarcă fișier (`image/*` sau `video/*`) și apasă **Analizează**.
4. UI trimite request către backend:
   - `POST /predict`
   - `multipart/form-data` cu:
     - `file` = imagine/video
     - `method` = `yolo` sau `manual`
5. UI afișează:
   - verdict (label): `vara`, `iarna`, `mixt`
   - bare de progres pentru cele 3 clase (scoruri)
   - JSON-ul complet al răspunsului (toggle “Detalii tehnice”)
   - imagini generate de backend:
     - `bbox_url` (preview detecție)
     - `crop_url` (pattern prelucrat)

---

## Contract API (backend → UI)

### Request
`POST /predict` (multipart)
- `file`: imagine sau video
- `method`: `yolo` | `manual`

### Response (JSON)
UI-ul acceptă aceste câmpuri (minim):
- `label`: `vara` | `iarna` | `mixt`
- scoruri (oricare dintre cheile de mai jos e acceptată):
  - `scores` / `probabilities` / `percentages` / `probs` / `confidences`
  - cu chei `vara/iarna/mixt` (sau `summer/winter/mixed`)
- opțional:
  - `bbox_url`: URL către imaginea cu bounding box
  - `crop_url`: URL către imaginea “pattern prelucrat”

Notă: UI detectează automat dacă scorurile sunt în [0..1] sau [0..100] și le normalizează pentru afișare.

---

## Output vizual
- Verdict colorat:
  - vară = roșu
  - iarnă = albastru
  - mixt = galben
- Alertă “NECONFORM” dacă sezonul selectat este `iarna` și verdictul este `vara`.
- Preview fișier încărcat:
  - pentru video se redă local imediat
  - pentru imagini se afișează `bbox_url` dacă e furnizat

---

Pipeline

UI-ul este interfața peste pipeline-ul complet:

LOAD (upload fișier)

AUTOMATIC_IDENTIFICATION (YOLO) sau fallback MANUAL

PREPROCESS (ROI → normalizare)

RN_INFERENCE (clasificare)

DISPLAY_RESULT (UI)

