# Audit Complet al Sistemului de Estimare și Predicție a Precipitațiilor (H-SAF)

## 1. Executive Summary
Acest raport prezintă un audit exhaustiv al repository-ului de estimare și predicție (nowcasting) a volumului de precipitații bazat pe date satelitare. Analiza a fost realizată de o echipă multidisciplinară (Arhitectură, Machine Learning, Matematică Numerică, Remote Sensing, Kalman Filtering, Performanță, Calitate, Testare).

**Concluzie generală:** Codebase-ul demonstrează o maturitate algoritmică remarcabilă, combinând tehnici hibride (Optical Flow DIS + Filtre Kalman cu model Singer + Advecție Semi-Lagrangiană). Utilizarea modelării log-normale pentru aria celulelor și update-ul covarianței în forma Joseph denotă cunoștințe avansate de matematică. Statisticile independente confirmă o poveste coerentă: algoritmul excelează la detecția furtunilor, tracking și conservarea formei (foarte competitiv la 30 min - 1h). **Totuși, sistemul supraestimează masiv durata de viață, aria și volumul la orizonturi mari (2h), generând numeroase alarme false.** Acest tipar indică lipsa unui model termodinamic de disipare, pe lângă suferințele de "fragilitate numerică" (metode de forțare PSD incorecte), decizii sub-optime de performanță (conversii O(N) în calculul IoU) și o proliferare a "magic numbers".

Arhitectura respectă principiile Clean Architecture la un nivel înalt (separarea UI, Core, IO), dar detaliile de implementare prezintă cuplare strânsă între modelul de domeniu (`StormCell`) și motoarele de calcul. Îmbunătățirile propuse pot crește semnificativ acuratețea și stabilitatea sistemului, fără a altera paradigma de bază (bazată pe centroizi).

---

## 2. Architecture Review
**Aspecte pozitive:** Separarea modulelor (`core`, `geo`, `io`, `dashboard`) este clară. Utilizarea Fațadei (`Orchestrator`) protejează starea aplicației de concurența interfeței grafice (Dash/Streamlit).

**Probleme identificate:**
- **Coupling (Cuplare):** Entitatea `StormCell` (din `domain.py`) este direct manipulată și mutată (state mutation) de `AdvectionEngine`, `StormTracker` și `KinematicUpdater`. Acest aspect contrazice Domain-Driven Design (DDD), unde entitățile ar trebui să își gestioneze propria stare.
- **God Classes & Monoliți:** `AdvectionEngine` și `StormTracker` sunt clase masive (monolitice). `AdvectionEngine.extrapolate` gestionează simularea cinematică, aplicarea măștilor de creștere, warp-ul spațial și filtrarea S-PROG într-o singură trecere. 
- **Magic Numbers:** Există o multitudine de constante hardcodate (`gamma=0.8`, `tau_growth=3.0`, praguri de distanță, kernel-uri pentru GaussianBlur). Acestea aruncă în aer principiul Open-Closed (OCP).

**Soluții:**
- Extragerea logicii de creștere spațială (`_create_spatial_growth_mask`) și a logicii S-PROG în strategii distincte (`SpatialGrowthStrategy`, `DiffusionStrategy`).
- Parametrizarea tuturor constantelor (din `config.py` sau fișiere JSON specifice regiunii climatice).
- Refactorizarea `StormCell` pentru a folosi imuabilitate (ex: `dataclasses.replace`) sau metode de encriptare a stării, în loc de mutație directă.

---

## 3. Mathematical Review
**Aspecte pozitive:** 
- Derivarea vitezei și accelerației pentru arie folosind spațiul logaritmic: `d(exp(L))/dt = exp(L) * L'` și `dd_area = exp(L) * (L'^2 + L'')`. Aceste ecuații sunt corecte analitic și previn apariția ariilor negative, o problemă clasică în filtrele standard.
- Utilizarea ecuației Singer pentru accelerație (amortizare) previne explozia cinematică (overshooting) la orizonturi mari (1-2 ore).

**Probleme identificate:**
- **Condiționarea Matricei și Catastrophic Cancellation:** Deși se folosește Joseph form `P = (I - KH)P(I - KH)' + KRK'` (excelent pentru stabilitate), matricea de zgomot a procesului `Q` poate deveni instabilă numeric la evaluarea exponențialelor `exp(-2*a)` în `add_singer_q_block` pentru limite de `dt` mic (a apropiat de 0). Taylor expansion ar fi necesar pentru limite (când `a < 1e-4`).
- **Probleme Floating Point:** Calculul `cumulative_factor = min(pred_area / area, max_growth)` poate da `NaN` sau instabilitate dacă `area` este extrem de aproape de zero. Deși s-a pus `max(1.0, area)`, ar fi mai robust matematic să lucrăm exclusiv în domeniu logaritmic până în punctul aplicării.

**Soluții:**
- Utilizarea expansiunii Taylor (sau Pade approximations) în `add_singer_q_block` pentru `alpha * dt` valori mici, evitând anularea catastrofală (`1 - exp(-x) ~ x - x^2/2`).

---

## 4. Kalman Review
Sistemul utilizează un filtru 8D (care tehnic e definit ca 9D cu `dim_x=9` și starea `dd_area` ca variabilă de tranziție) Constant Acceleration (CA) adaptat cu model Singer.

**Probleme identificate:**
- **Forțarea PSD (Positive Semi-Definite):** Linia `self._kf.P = (self._kf.P + self._kf.P.T) / 2.0` asigură doar simetria matricei, **NU** și certitudinea că toate valorile proprii sunt pozitive. Dacă `P` își pierde calitatea de PSD, filtrul Kalman poate exploda iremediabil.
- **R (Measurement Noise) Static:** `self._kf.R = diag(5.0, 5.0, 0.2)`. Aceasta asertează că incertitudinea măsurătorii centroizilor este mereu de 5 pixeli (pătrați). O furtună foarte mare și neregulată va avea o variație mult mai mare a centroidului față de o celulă mică și compactă.
- **Q (Process Noise) Ne-adaptiv:** Valorile variance-urilor (`0.05`, `0.01`) sunt presupuneri universale care nu se potrivesc la fel și pentru celule orografice (mici, scurte) și pentru sisteme frontale (mari, de lungă durată).

**Soluții:**
- **Robust PSD Forcing:** Calcularea Eigen-Decomposition și tăierea valorilor negative: 
  `eigval, eigvec = np.linalg.eigh(P); P = eigvec @ np.diag(np.maximum(eigval, 1e-6)) @ eigvec.T` (sau factorizare UDU/Cholesky).
- **Adaptive R:** R trebuie calculat ca funcție de dispersia spațială a pixelilor furtunii (`R_x = k * area_variance_x`).
- Modificarea modelului pentru a folosi Interacting Multiple Model (IMM) (ex: tranziție între viteză constantă și accelerație constantă în funcție de stadiul de viață al celulei).

---

## 5. Prediction Algorithm Review
Predicția folosește advecție semi-Lagrangiană (`cv2.remap`) combinată cu un blending de kinematică (Optical Flow global + Kalman Flow local).

**Probleme identificate:**
- **Lipsa modelării ciclului de viață (Lifecycle Modeling):** Deoarece un model pur cinematic are mișcare bună dar nicio constrângere pe evoluția intensității, furtunile tind să persiste prea mult. Acest defect explică direct creșterea majoră a ratei de alarme false (FAR) și bias-ul volumetric puternic pozitiv la orizonturi >1h.
- **Vector Field Tearing (Ruperea câmpului):** În `AdvectionEngine._blend_kinematics`, se forțează vectorii Kalman peste flow-ul optic folosind un multiplicator Gaussian (`weight = np.exp(...) * kalman_confidence`). Acest cross-fading direct modifică local câmpul de viteze într-un mod nerealist, generând divergență/convergență artificială (divergența câmpului `div V != 0`), ceea ce va "rupe" (smudge/tear) forma precipitațiilor la advecție.
- **Propagarea Erorii (S-PROG):** Implementarea S-PROG (blurring) este o aproximare (`cv2.GaussianBlur` scalat cu numărul pasului). Totuși, ea difuzează doar imaginea, fără a scădea din intensitatea de vârf proporțional pentru a păstra masa (Mass Conservation).

**Soluții:**
- Păstrarea paradigmei (centroizi), dar în loc de cross-fading pe *câmpul de vectori*, aplicarea interpolării tip "Kriging" sau un Smoothing Spline (TPS - Thin Plate Spline) asupra câmpului vectorilor de mișcare pentru a asigura derivate netede.
- La aplicarea S-PROG, utilizarea filtrelor care respectă conservarea masei (integrarea energiei pre/post blur și renormalizarea matricei).

---

## 6. Satellite Data Review (Remote Sensing)
**Probleme identificate:**
- **Bias de conversie dBZ:** În `flow_estimator.py`, s-a hardcodat formula `dbz = 23.0 + 16.0 * np.log10(r)` pentru a spori contrastul pentru DIS Optical Flow. Aceasta este relația clasică Marshall-Palmer, dar produsele satelitare H-SAF (ex: MW/IR blend) au propriile calibrații neliniare.
- **Proiecții Geospațiale (Parallax):** Nu se observă o corecție de paralaxă. Nori convectivi înalți văzuți de satelit geostaționar (Meteosat la 0 grade longitudine) suferă o deplasare aparentă masivă la latitudini ca cele ale României (45°N). Centroizii estimați pot fi decalați spațial cu 10-15 km față de precipitațiile de la sol.
- **Pixeli lipsă (Cloud masks/No data):** Folosirea `np.nan_to_num(..., nan=0.0)` forțează pixelii satelitari invalizi sau lipsă la ploaie zero. Acest lucru induce erori majore de formă a ariei, modificând artificial centroidul și perturbând filtrul Kalman.

**Soluții:**
- Utilizarea unei grile de validitate. Doar celulele cu >95% din arie cu date valide se updatează în Kalman (pentru restul aplicăm predict-only).
- Introducerea corecției de paralaxă în `projection.py` utilizând o altitudine asumată a topului norilor (din modele NWP sau setări).

---

## 7. Performance Review
**Probleme identificate:**
- **O(N) Set Conversion în IoU:** Funcția `Matcher._coords_iou` transformă array-uri `Nx2` în tuple, iar apoi le introduce în seturi Python, iterativ. O(N) la nivel Python pentru array-uri de mii de pixeli paralizează sistemul. E un bottleneck extrem.
  *Soluție:* Folosirea boolean masking pe un grid fix (deja existent în logica tracker-ului), sau funcția `numpy.intersect1d` / `numpy.isin(..., assume_unique=True)`.
- **Labeling repetat:** `ndi.find_objects(labeled_mask)` face o trecere eficientă, dar în bucla for, `ndi.sum_labels` etc., recalculează peste întreaga matrice (overhead ineficient).
- **Paralelizare:** Algoritmii sunt puternic secvențiali (CPU-bound) și rulează cu GIL blocat.

**Soluții:**
- Optimizarea funcției IoU cu operații bitwise numpy (`np.logical_and`).
- Înlocuirea logicii din `StormCellDetector` cu `skimage.measure.regionprops`, care compilează toate metricile (arie, bbox, centroid) într-o singură trecere extrem de rapidă bazată pe cod C/Cython.

---

## 8. Reliability Review
**Probleme identificate:**
- **Lipsă de izolare a erorilor Kalman:** Dacă o matrice singulară apare din date corupte și filtrul Kalman aruncă `LinAlgError`, metoda `update()` crapă și distruge întregul ciclu de predicție. Nu există blocuri de `try/except` robuste în jurul metodelor matematice.
- **Leak-uri în State:** `StormTracker.track` curăță filtrele inactive cu `cleanup_inactive`, dar obiectele `StormCell` ar putea reține memorie dacă există referințe circulare (cache_mask).

**Soluții:**
- Catch blocuri (`try / except np.linalg.LinAlgError`) care să treacă tracker-ul în starea de `fallback` (doar `predict`, fără `update` pentru pasul curent) sau re-inițializarea tracker-ului specific.

---

## 9. Code Quality Review
- **Docstrings & Typing:** Acurate, se folosește extensiv type hinting (`list[StormCell]`, `np.ndarray`). Este excelent.
- **Magic Strings / Numbers:** Sistemul e infestat de parametri hardcodați (ex: `cost < 500.0`, `gamma=0.8`, `base_ksize = int(base_sigma * 3) | 1`, multiplicatorii la IOU).
- **Inconsistențe:** `StormCell` folosește o combinație de metode tip dataclass/atribute, dar codul ocazional acesează atribute care ar putea să nu fie definite (ex: în `StormTracker`, `getattr(c, "is_tracked", False)` arată o lipsă de rigoare pe schema obiectului).

---

## 10. Security Review
- **Severitate: Low/Medium**.
- `config.py` încarcă `FTP_USER` via `dotenv`, ceea ce este sigur.
- `FrameProcessor` citește probabil fișiere, însă manipularea căilor (`file_path`) dinspre UI către motor implică un risc de Path Traversal dacă dashboard-ul permite utilizatorilor să trimită string-uri arbitrare (nevalidat).
- *Soluție:* Validarea absolută a fișierelor și sanitizarea la granița `Orchestrator`-ului.

---

## 11. Test Coverage Review
Codebase-ul conține `test_math_stability.py`, `test_advection.py`, dar acoperirea e minimală (doar 4 fișiere de test).
- **Ce lipsește:** Nu există teste pentru `StormTracker` (comportamentul split / merge este extrem de volatil, trebuie testat izolat).
- Teste care să injecteze valori de zgomot masive / `NaN`-uri în matricea de intrare pentru a asigura că `Kalman` și `AdvectionEngine` nu generează predicții `NaN` sau matrici care să cauzeze out-of-bounds în memorie.

---

## 12. Refactoring Plan
1. **Separarea logicii matematice (Math Layer):** Mutarea filtrelor Kalman și a metricilor într-un modul izolat și pur, ușor testabil.
2. **Încapsularea Mutațiilor (Domain Layer):** Modificarea `StormCell` într-un Data Class imuabil. O entitate trece printr-un engine de transformare (Tracker) care returnează o copie modificată (functional programming pattern, util pentru stabilitate temporală și replay-uri).
3. **Decuplarea Magic Numbers:** Extragerea parametrilor algoritmi într-un `TrackerConfig` (Thresholds, Sigmas, Gammas).

---

## 13. Accuracy Improvement Opportunities (Păstrând Paradigma Centroid)
- **Model de Disipare și Evoluție a Celulelor:** Modelarea explicită a ciclului de viață (naștere, maturitate, disipare). Intensitatea și aria trebuie să poată scădea în timp pe baza vârstei și istoricului celulei, tăind masiv din alarmele false (FAR) la orizonturi mari.
- **Local / Adaptive Covariance (Q și R-Matrix):** Filtre Kalman cu matrici adaptive care să răspundă diferit pentru celulele stabile față de cele în dezvoltare explozivă sau disipare.
- **Estimarea Incertitudinii Fiecărui Centroid:** Predicțiile cu incertitudine mare trebuie ponderate mai conservator. Un centroid format dintr-un roi de celule fragmentate va avea o influență redusă, lăsând Advection Engine-ul să folosească prioritar Optical Flow-ul global.
- **Corecție pentru Paralaxă:** Indispensabilă pentru imagini geostaționare, estimând înălțimea norilor pentru a corecta proiecția pe sol.
- **Neighbor Distance Metrics:** Pentru `Matcher`, în loc de distanța Euclidiană la centru, folosiți Distanța Mahalanobis care ține cont de forma (elipsa) furtunii (folosind covarianța spațială), eliminând greșelile frecvente de "furt-de-celulă" când două fronturi trec apropiate, dar au orientări diferite.

---

## 14. Bug List
| Bug / Fișier | Funcție | Severitate | Explicație | Soluție |
|--------------|---------|------------|------------|---------|
| `storm_filter.py` | `update` | **High** | `(P + P.T) / 2.0` nu garantează PSD. Poate rezulta în varianțe negative (instabilitate catastrofală a matricelor pe parcursul rulărilor lungi). | Reînlocuirea cu calculul Eigen Decomposition sau factorizare Cholesky. |
| `matcher.py` | `_coords_iou` | **High** (Perf) | Transformă matrice Nx2 în tuple, apoi Set. O(N) alocări per celulă per frame blochează complet performanța sistemului. | Implementare boolean numpy (O(1) operații de memorie masivă). |
| `advection_engine.py` | `extrapolate` | **Medium** | Dacă ploaia conține `NaN`, conversia `nan_to_num(0)` maschează datele lipsă, deformând masiv centroizii extrași din satelit. | Integrarea unui mask de invaliditate propagat către `StormTracker` pentru a opri update-ul pe celulele afectate (predicție bazată pur pe priors). |
| `advection_engine.py` | `_blend_kinematics` | **Medium** | Filtrarea hibridă direct pe un flux de viteză cauzează divergențe non-zero ("tearing"), ruinând corectitudinea curgerii masei de aer. | Trecerea la Spatial Interpolation (Thin Plate Spline) care asigură continuitate (C1/C2 limit). |

---

## 15. Prioritized TODO
**Critical:**
- [ ] Fix IOU Tuple-Set Memory Bottleneck (`matcher.py`).
- [ ] Fix PSD Covariance Forcing (`storm_filter.py`).
- [ ] Implement robust `NaN` & Missing Data Handling (`storm_tracker.py`, `advection_engine.py`).

**High:**
- [ ] Extragerea parametrilor Magic Numbers în fișiere de configurație (ex. `tracker_config.json`).
- [ ] Implementarea Mahalanobis distance în `Matcher` pentru asocieri corecte inter-frame.
- [ ] Refactorizarea codului S-PROG pentru Mass-Conservation limit.

**Medium:**
- [ ] Migrarea dinamică a matricii de zgomot al observației (R-matrix) pe baza formei furtunii.
- [ ] Integrarea regiunilor de validitate satelitară (`projection.py`).

**Low:**
- [ ] Teste automate extinse (Kalman convergence, KDTree merge-split edge cases).
- [ ] Paralelizarea (sau rescrierea) procesării cadrelor prin vectorizare NumPy pură pe axa timpului, unde este posibil.

---

## 16. Overall Score (Inclusiv Analiza Statistică Independentă)
| Categorie | Scora | Justificare sumară |
|---|---|---|
| Acuratețe volumetrică (30 min) | **9/10** | Extrem de precis pe termen scurt, formă și volum excelent conservate. |
| Acuratețe volumetrică (2 h) | **7/10** | Scădere generată de incapacitatea algoritmului de a stinge furtunile disipate. |
| Localizare spațială | **8.5/10** | Peste medie datorită blending-ului hibrid DIS + Kalman Cinematic. |
| Tracking al celulelor | **8.5/10** | Foarte bun. Nu pierde celule, chiar le menține excesiv de mult. |
| Controlul alarmelor false (FAR) | **5.5/10** | Supraestimează masiv durata de viață și generează alarme false la orizonturi mari. |
| Bias sistematic | **5/10** | Bias volumetric pronunțat de supraestimare pe termen mediu/lung. |
| Architecture & Maintainability | **6.5/10** | Clean, dar cu "magic numbers" și domain models cuplate în engine-uri. |
| Mathematical Correctness | **8/10** | Derivări Singer și modelări de arie logaritmică excelente. Lipsuri la matricea PSD. |
| Performance & Scalability | **6/10** | Bottleneck fatal în `_coords_iou` (Set conversion); algoritm greu paralelizabil. |

---

## Plan de Implementare Etapizat (Phased Rollout)

### Phase 1: Stabilitate și Performanță Numerică (Ziua 1-3)
- **T1:** Refactorizare `_coords_iou` pentru a utiliza matrici booleene și `numpy` intersect/bitwise, mărind viteza de ~10-50x.
- **T2:** Implementarea corecției veritabile PSD pentru matricea de covarianță P folosind Eigen-Decomposition.
- **T3:** Învelirea update-urilor Kalman în blocuri reziliente, fallback la `predict-only` la erori `LinAlg`.
- **Risc:** Minimal. Câștig uriaș în viteză și stabilitate la rulări cu volum mare de date.

### Phase 2: Consolidare Domeniu și Mentenanță (Săptămâna 2)
- **T1:** Centralizarea tuturor "magic numbers" (`gamma`, `dt`, `tau_growth`, limite maxime hibride, dimensiuni kernel) într-un modul izolat (sau injectat prin container) `AlgorithmsConfig`.
- **T2:** Trecerea calculului statistic (` StormCellDetector`) la `skimage.measure.regionprops` pentru robustețe la analiză topologică, cu focus pe extragerea orientării.
- **Risc:** Scăzut. Asigură mentenabilitatea codului pe termen lung, fără modificări teoretice.

### Phase 3: Calitatea Predicției (Algoritmice și Matematice) (Săptămâna 3)
- **T1:** (Incert/Context dependent): Validarea relației Marshall-Palmer. Verificarea calibrației de conversie cu specificațiile oficiale H-SAF.
- **T2:** Implementarea covarianței dinamice `R` proporțională cu dimensiunea celulei / varianța pixelilor pe cele două axe.
- **T3:** Eliminarea blending-ului direct pe vector-field și mutarea pe un Thin-Plate Spline smoothing (sau alt interpolator neted) pentru a menține un flow curat, non-divergent al maselor de nori.
- **Risc:** Mediu. Implică recalibrarea întregii curbe de acuratețe a motorului predictiv.

### Phase 4: Validare și Data Asimilare Extremă (Săptămâna 4)
- **T1:** Corectarea erorilor introduse de paralaxă și integrarea măștilor de pixeli lipsă în algoritmul de `StormTracker`.
- **T2:** Construirea de unit-teste pentru capabilitățile Kalman specifice: Split-Merge behaviour, NaN injection test.
- **Risc:** Ridicat, necesită experimente cu arhiva de date (eventual date cu unghi vizual extrem din EUMETSAT). Merită validare științifică aprofundată înainte de introducerea în modul operațional național.
