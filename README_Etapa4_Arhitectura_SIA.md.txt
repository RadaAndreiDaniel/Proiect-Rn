# 🧠 ETAPA 4 – Arhitectura Completă a Aplicației SIA  
**Proiect:** Identificarea tipului de anvelopă (vară / iarnă / mixt)

---

## 1. Nevoie reală → Soluție SIA → Modul software

| Nevoie concretă | Cum o rezolvă SIA-ul | Modul responsabil |
|-----------------|----------------------|-------------------|
| Identificarea benzii de rulare din imagini video | Preprocesare cu YOLOv8 + TreadScan pentru extragerea tălpii | `src/preprocessing/` |
| Clasificarea tipului de anvelopă | Rețea neuronală CNN aplicată pe imaginea benzii | `src/neural_network/` |
| Afișarea rezultatului într-o manieră accesibilă | Interfață Streamlit pentru upload și afișare rezultat | `src/app/` |

---

## 2. Contribuția originală la dataset (regula 40%)

O parte din date a fost produsă prin extragerea cadrelor din videoclipuri proprii, folosind scriptul `extract_frames.py`. Acestea reprezintă contribuția originală necesară.

- **Total observații finale:** N  
- **Observații originale:** M  
- **Procent:** `M / N ≥ 40%`  

Datele originale sunt plasate în:

```
data/generated/
```

---

## 3. Diagrama State Machine

Fișierul se află în `docs/state_machine.png`.

Stările utilizate descriu fluxul normal al aplicației:
- IDLE  
- ACQUIRE_DATA  
- PREPROCESS  
- INFERENCE  
- DISPLAY_RESULT  
- ERROR  

Arhitectura aleasă permite gestionarea clară a fluxurilor succesive și separarea responsabilităților fiecărui modul.

---

## 4. Modul 1 – Data Acquisition

`src/data_acquisition/extract_frames.py` permite extragerea cadrelor din videoclipuri și generarea datasetului original. Acesta asigură procesarea elementară a inputului brut și generarea de imagini utilizabile în etapele următoare.

---

## 5. Modul 2 – Neural Network

În `src/neural_network/` se definește modelul de rețea neuronală (CNN simplificat), împreună cu procedurile de compilare și salvare. În această etapă accentul cade pe structurarea modelului și pregătirea lui pentru antrenare ulterioară.

---

## 6. Modul 3 – Interfața (Streamlit)

`src/app/main.py` oferă interfața în care utilizatorul poate încărca o imagine a benzii de rulare. Imaginea este preprocesată, trimisă către model, iar rezultatul este afișat sub forma tipului de anvelopă.

---

## 7. Structura proiectului

Structura exactă a proiectului, extrasă automat:

```
proiect/
  yolov8n.pt
  .vscode/
    settings.json
  config/
  data/
    generated/
    processed/
      frames_raw/
      pattern_crops/
      train_patterns/
        iarna/
          winter9 - Copy (2)_pattern.png
          winter9 - Copy (3)_pattern.png
          winter9 - Copy_pattern.png
          winter9_pattern.png
        mixt/
          mixta_01_pattern.png
          mixta_02_pattern.png
          mixta_03_pattern.png
          mixta_04_pattern.png
        vara/
          summer4 - Copy (2)_pattern.png
          summer4 - Copy (3)_pattern.png
          summer4 - Copy_pattern.png
          summer4_pattern.png
      wheel_crops/
    raw/
    test/
      roata1.mp4
      roata2.mp4
    train/
      iarna/
        winter1 - Copy (2).avif
        winter1 - Copy (3).avif
        winter1 - Copy.avif
        winter1.avif
        winter2 - Copy (2).avif
        winter2 - Copy (3).avif
        winter2 - Copy.avif
        winter2.avif
        winter3 - Copy (2).avif
        winter3 - Copy (3).avif
        winter3 - Copy.avif
        winter3.avif
        winter4 - Copy (2).avif
        winter4 - Copy (3).avif
        winter4 - Copy.avif
        winter4.avif
        winter5 - Copy (2).avif
        winter5 - Copy (3).avif
        winter5 - Copy.avif
        winter5.avif
        winter6 - Copy (2).avif
        winter6 - Copy (3).avif
        winter6 - Copy.avif
        winter6.avif
        winter7 - Copy (2).avif
        winter7 - Copy (3).avif
        winter7 - Copy.avif
        winter7.avif
        winter8 - Copy (2).avif
        winter8 - Copy (3).avif
        winter8 - Copy.avif
        winter8.avif
        winter9 - Copy (2).jpg
        winter9 - Copy (3).jpg
        winter9 - Copy.jpg
        winter9.jpg
      mixt/
        mixta_01.avif
        mixta_01.jpg
        mixta_02.jpg
        mixta_02.webp
        mixta_03.jpg
        mixta_03.webp
        mixta_04.jpg
        mixta_04.webp
      vara/
        summe8.avif
        summer1 - Copy (2).avif
        summer1 - Copy (3).avif
        summer1 - Copy.avif
        summer1.avif
        summer2 - Copy (2).avif
        summer2 - Copy (3).avif
        summer2 - Copy.avif
        summer2.avif
        summer4 - Copy (2).jpg
        summer4 - Copy (3).jpg
        summer4 - Copy.jpg
        summer4.jpg
        summer5 - Copy (2).webp
        summer5 - Copy (3).webp
        summer5 - Copy.webp
        summer5.webp
        summer6 - Copy (2).avif
        summer6 - Copy (3).avif
        summer6 - Copy.avif
        summer6.avif
        summer7 - Copy (2).avif
        summer7 - Copy (3).avif
        summer7 - Copy.avif
        summer7.avif
        vaRA.avif
        varaaa.jpg
        varaaaa.jpg
        varatest.webp
    validation/
  docs/
    datasets/
  models/
    saved_model.pth
  output/
  RCNN_model/
    coco_eval.py
    coco_utils.py
    engine.py
    group_by_aspect_ratio.py
    KeypointRCNN_training.ipynb
    presets.py
    README.md
    saved_model.pth
    train.py
    transforms.py
    utils.py
    dataset/
      info.txt
      annotations/
        1_000.json
        1_001.json
        1_002.json
        1_003.json
        1_004.json
        1_006.json
        1_007.json
        1_008.json
        1_009.json
        1_010.json
        1_011.json
        1_012.json
        1_013.json
        1_014.json
        1_016.json
        1_017.json
        1_018.json
        1_020.json
        1_021.json
        1_023.json
        1_024.json
        1_025.json
        1_027.json
        1_028.json
        1_029.json
        1_031.json
        1_032.json
        1_033.json
        1_034.json
        1_038.json
        1_039.json
        1_040.json
        1_041.json
        1_043.json
        1_044.json
        1_045.json
        1_047.json
        1_048.json
        1_049.json
        1_050.json
        1_051.json
        1_052.json
        1_053.json
        1_054.json
        1_055.json
        1_056.json
        1_058.json
        1_059.json
        1_060.json
        1_061.json
        1_062.json
        1_063.json
        1_064.json
        1_066.json
        1_067.json
        1_068.json
        1_069.json
        1_070.json
        1_072.json
        1_074.json
        1_076.json
        2_000.json
        2_001.json
        2_002.json
        2_003.json
        2_004.json
        2_005.json
        2_006.json
        2_007.json
        2_008.json
        2_009.json
        2_010.json
        2_011.json
        2_012.json
        2_013.json
        2_014.json
        2_016.json
        2_017.json
        2_018.json
        2_019.json
        2_020.json
        2_021.json
        2_022.json
        2_023.json
        2_024.json
        2_025.json
        2_026.json
        2_027.json
        2_028.json
        2_029.json
        2_031.json
        2_032.json
        2_033.json
        2_034.json
        2_035.json
        2_036.json
        2_037.json
        2_038.json
        2_039.json
        2_040.json
        2_041.json
        2_042.json
        2_043.json
        2_044.json
        2_045.json
        2_046.json
        2_047.json
        2_048.json
        2_049.json
        2_050.json
        2_051.json
        2_052.json
        2_053.json
        2_054.json
        2_055.json
        2_056.json
        2_057.json
        2_058.json
        2_059.json
        2_060.json
        2_061.json
        2_062.json
        2_063.json
        2_064.json
        2_065.json
        2_066.json
        2_067.json
        2_068.json
        2_069.json
        2_070.json
        2_071.json
        2_072.json
        2_073.json
        2_074.json
        2_075.json
        2_076.json
        2_077.json
        2_078.json
        2_079.json
        2_080.json
        2_081.json
        2_082.json
        2_083.json
        2_085.json
        2_086.json
        2_087.json
        2_088.json
        2_089.json
        2_090.json
        2_091.json
        2_092.json
      images/
        1_000.jpg
        1_001.jpg
        1_002.jpg
        1_003.jpg
        1_004.jpg
        1_006.jpg
        1_007.jpg
        1_008.jpg
        1_009.jpg
        1_010.jpg
        1_011.jpg
        1_012.jpg
        1_013.jpg
        1_014.jpg
        1_016.jpg
        1_017.jpg
        1_018.jpg
        1_020.jpg
        1_021.jpg
        1_023.jpg
        1_024.jpg
        1_025.jpg
        1_027.jpg
        1_028.jpg
        1_029.jpg
        1_031.jpg
        1_032.jpg
        1_033.jpg
        1_034.jpg
        1_038.jpg
        1_039.jpg
        1_040.jpg
        1_041.jpg
        1_043.jpg
        1_044.jpg
        1_045.jpg
        1_047.jpg
        1_048.jpg
        1_049.jpg
        1_050.jpg
        1_051.jpg
        1_052.jpg
        1_053.jpg
        1_054.jpg
        1_055.jpg
        1_056.jpg
        1_058.jpg
        1_059.jpg
        1_060.jpg
        1_061.jpg
        1_062.jpg
        1_063.jpg
        1_064.jpg
        1_066.jpg
        1_067.jpg
        1_068.jpg
        1_069.jpg
        1_070.jpg
        1_072.jpg
        1_074.jpg
        1_076.jpg
        2_000.jpg
        2_001.jpg
        2_002.jpg
        2_003.jpg
        2_004.jpg
        2_005.jpg
        2_006.jpg
        2_007.jpg
        2_008.jpg
        2_009.jpg
        2_010.jpg
        2_011.jpg
        2_012.jpg
        2_013.jpg
        2_014.jpg
        2_016.jpg
        2_017.jpg
        2_018.jpg
        2_019.jpg
        2_020.jpg
        2_021.jpg
        2_022.jpg
        2_023.jpg
        2_024.jpg
        2_025.jpg
        2_026.jpg
        2_027.jpg
        2_028.jpg
        2_029.jpg
        2_031.jpg
        2_032.jpg
        2_033.jpg
        2_034.jpg
        2_035.jpg
        2_036.jpg
        2_037.jpg
        2_038.jpg
        2_039.jpg
        2_040.jpg
        2_041.jpg
        2_042.jpg
        2_043.jpg
        2_044.jpg
        2_045.jpg
        2_046.jpg
        2_047.jpg
        2_048.jpg
        2_049.jpg
        2_050.jpg
        2_051.jpg
        2_052.jpg
        2_053.jpg
        2_054.jpg
        2_055.jpg
        2_056.jpg
        2_057.jpg
        2_058.jpg
        2_059.jpg
        2_060.jpg
        2_061.jpg
        2_062.jpg
        2_063.jpg
        2_064.jpg
        2_065.jpg
        2_066.jpg
        2_067.jpg
        2_068.jpg
        2_069.jpg
        2_070.jpg
        2_071.jpg
        2_072.jpg
        2_073.jpg
        2_074.jpg
        2_075.jpg
        2_076.jpg
        2_077.jpg
        2_078.jpg
        2_079.jpg
        2_080.jpg
        2_081.jpg
        2_082.jpg
        2_083.jpg
        2_085.jpg
        2_086.jpg
        2_087.jpg
        2_088.jpg
        2_089.jpg
        2_090.jpg
        2_091.jpg
        2_092.jpg
  src/
    app/
    data_acquisition/
      extract_frames.py
    neural_network/
      train_classifier.py
    preprocessing/
      detect_wheel_yolo.py
      extract_tire_pattern.py
      prepare_training_patterns.py
      run_treadscan_video.py
      test_extract.py
      tread_scan_IMAGE.py
      tread_scan_image2.py
      treadscan_inspect.py
      treadscan_test.py
      yolov8n.pt
      __pycache__/
        extract_tire_pattern.cpython-310.pyc
    utils/
```

---

## 8. Observații finale

Arhitectura realizată pune bazele sistemului complet de analiză a benzii de rulare, separând clar componentele de achiziție, preprocesare, clasificare și interfață. Acest schelet permite extinderea ulterioară prin antrenarea modelului, optimizarea performanței și rafinarea pipeline‑ului.
