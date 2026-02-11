

Setul de date utilizat în cadrul acestui proiect este compus din imagini cu anvelope, colectate din surse publice și imagini reale, și este organizat în mai multe etape de procesare pentru a putea fi utilizat eficient în antrenarea și evaluarea rețelelor neuronale.

Surse de date

Datele neprelucrate (raw) provin de pe diferite site-uri care comercializează anvelope, precum:

eMAG

Altex

Pirelli

Continental

Imaginile sunt preluate în forma originală, exact așa cum sunt disponibile pe aceste platforme, fără modificări inițiale.

Structura folderului data

raw/
Conține imagini originale cu anvelope, neprelucrate, provenite din sursele menționate mai sus. Aceste imagini reprezintă punctul de plecare pentru procesul de preprocesare.

processed/
Conține imaginile obținute în urma procesului de preprocesare realizat cu ajutorul scripturilor din src/preprocessing.

Preprocesarea include:

redimensionarea imaginilor la 256 × 256 pixeli;

conversia imaginilor în grayscale;

decuparea zonei relevante, respectiv doar amprenta (pattern-ul) benzii de rulare a anvelopei.

train/
Conține imaginile utilizate pentru antrenarea modelelor de rețele neuronale. Aceste imagini sunt obținute exclusiv din datele preprocesate provenite din surse online.

validation/
Conține imagini utilizate pentru validarea modelului în timpul procesului de antrenare.
Aceste imagini sunt diferite de cele din setul de antrenare și provin din imagini reale, capturate de pe anvelope montate pe autovehicule, din care este decupată doar zona cu pattern-ul cauciucului.

test/
Conține imagini utilizate exclusiv pentru testarea finală a performanței modelului.
Similar setului de validare, imaginile sunt decupate din fotografii reale ale anvelopelor montate pe autovehicule și nu sunt incluse sub nicio formă în procesul de antrenare sau validare.

Împărțirea setului de date (Data Split)

Setul de date este împărțit după cum urmează:

70% – set de antrenare (train)

15% – set de validare (validation)

15% – set de test (test)

Distribuția setului de date nu este complet echilibrată între toate clasele.
Există o balanță uniformă între clasele anvelopă de vară și anvelopă de iarnă, atât în setul de antrenare, cât și în seturile de validare și testare.

Clasa anvelopă mixtă este reprezentată de un număr semnificativ mai mic de imagini în toate subseturile (train, validation și test). Această decizie a fost luată în mod intenționat, în urma mai multor experimente și optimizări, care au arătat că anvelopele mixte reprezintă o combinație de caracteristici specifice anvelopelor de vară și de iarnă.

În urma testelor efectuate, s-a constatat că un set de date extins pentru clasa mixtă poate introduce ambiguități în procesul de învățare al modelului. Prin urmare, dimensiunea acestei clase a fost redusă în mod controlat, pentru a permite modelului să învețe mai clar diferențele fundamentale dintre clase și pentru a obține rezultate mai stabile și mai corecte în procesul de clasificare.

Seturile de validare și test sunt complet separate de setul de antrenare, pentru a evita fenomenul de data leakage și pentru a asigura o evaluare corectă a performanței modelelor.