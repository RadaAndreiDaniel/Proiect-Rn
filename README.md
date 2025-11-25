# Proiect-Rn

# Proiect Clasificare Anvelope – Banda de Rulare

## 1. Descriere Proiect

Scopul proiectului este identificarea automata a tipului de anvelopa (vara, iarna, mixta) pe baza imaginii benzii de rulare. Utilizatorul poate introduce o fotografie a unei anvelope, care este prelucrata pentru a extrage doar banda de rulare si a elimina janta sau fundalul. Modelul este antrenat pe imagini cropate ale benzii de rulare pentru a obtine o precizie ridicata.

---

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor
* **Origine:** Fotografii cu anvelope de diferite sezoane (vara, iarna, mixta), colectate manual sau din surse web.
* **Modul de achizitie:** ☑ Fișier extern / ☐ Senzori reali / ☐ Simulare / ☐ Generare programatica
* **Perioada / conditii colectare:** 2024–2025, diverse unghiuri si iluminari, atat roti montate cat si detașate.

### 2.2 Caracteristicile dataset-ului
* **Numar total de observatii:** momentan ~12–50 imagini per clasa (prototip), urmeaza extindere.
* **Numar de caracteristici (features):** 1 caracteristica principala – imaginea cropata a benzii de rulare.
* **Tipuri de date:** ☑ Imagini
* **Format fisiere:** ☑ PNG / ☑ JPG / `.webp`, `.avif` (convertite automat in `.jpg`)

### 2.3 Descrierea fiecarui feature

| Caracteristica     | Tip       | Unitate | Descriere                                       | Domeniu valori             |
|-------------------|-----------|---------|------------------------------------------------|----------------------------|
| imagine_cropata    | imagine   | pixel   | Imaginea benzii de rulare a anvelopei         | Rezolutie uniformizata     |
| clasa              | categorial| –       | Tipul anvelopei / sezon                        | {vara, iarna, mixta}      |

---

## 3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive
* Medie, mediana, deviatia standard pentru dimensiunea imaginilor
* Min–max si quartile pentru rezolutie
* Distributii pe clase (histograme)
* Identificarea outlierilor (imaginile corupte sau prea mici)

### 3.2 Analiza calitatii datelor
* Detectarea valorilor lipsa (% pe clasa / imagine)
* Detectarea imaginilor corupte sau cu extensii neacceptate
* Identificarea redundantei sau clasei foarte dezechilibrate

### 3.3 Probleme identificate
* Set mic de date initial (~12–50 imagini/clasa)
* Poze cu unghiuri variabile care afecteaza crop-ul benzii de rulare

---

## 4. Preprocesarea Datelor

### 4.1 Curatarea datelor
* Eliminare duplicate si imagini corupte
* Convertirea tuturor imaginilor la `.jpg`
* Decuparea automata a benzii de rulare si scalarea la rezolutie uniforma

### 4.2 Transformarea caracteristicilor
* Normalizare pixelilor intre 0–1
* Encoding pentru clasa (vara, iarna, mixta)
* Ajustarea dezechilibrului de clase prin augmentare imagini (rotiri, flip, crop)

### 4.3 Structurarea seturilor de date
* 70–80% – train
* 10–15% – validation
* 10–15% – test
* Stratificare pe clasa si fara data leakage

### 4.4 Salvarea rezultatelor preprocesarii
* Date preprocesate in `data/processed/`
* Seturi train/val/test in foldere dedicate
* Parametrii de preprocesare in `config/preprocessing_config.json` (optional)

---

## 5. Fisiere Generate in Aceasta Etapa

* `data/raw/` – date brute (toate imaginile initiale)
* `data/processed/` – date curatate si cropate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul pentru preprocesare imagini
* `data/README.md` – descrierea dataset-ului

---

## 6. Stare Etapa

- [x] Structura repository configurata
- [x] Dataset analizat (EDA realizata)
- [x] Date preprocesate (crop banda de rulare)
- [ ] Seturi train/val/test extinse
- [ ] Documentatie actualizata in README + `data/README.md`
- [ ] Antrenare model final pe dataset complet

---

## 7. Urmatorii Pasi

1. Colectarea unui volum mai mare de imagini per clasa (minim 100–200 pentru fiecare tip de anvelopa)
2. Testarea crop-ului automat pentru imagini cu unghiuri variabile
3. Antrenarea modelului CNN pentru clasificare

