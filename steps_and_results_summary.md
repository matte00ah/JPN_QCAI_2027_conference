# Riepilogo degli step e dei risultati - esperimento proxy QCAI 2027

Questo documento racconta, in parole semplici, che cosa è stato implementato finora nell'esperimento descritto in `JPN_QCAI_conference_plan.md` e che cosa ne è risultato. Segue i capitoli del piano, ma copre solo gli step segnati lì come completati; gli step ancora aperti (per esempio la regressione dei parametri dei crateri QI7, i passi opzionali e i lavori futuri) sono lasciati fuori. Gli id degli step tra parentesi (es. QB2.3) rimandano alla voce corrispondente del piano, dove sono indicati il comando esatto e la sezione del report da cui proviene ogni numero.

In una frase, l'obiettivo dell'esperimento è provare, su una regione della Luna, un rilevatore di crateri che combina immagini ottiche e dati di elevazione tramite foundation model congelati e uno strato di fusione addestrato, e che poi riclassifica le posizioni candidate con una SVM (support vector machine, macchina a vettori di supporto) a kernel quantistico eseguita su simulatore. Quanto descritto qui sotto copre l'intera catena: i dati, l'estrazione delle feature, l'addestramento dello strato di fusione e la valutazione delle rappresentazioni, lo stage 1 classico, lo stadio a kernel quantistico su simulatore con le sue diagnostiche, i confronti con i metodi classici, e infine lo studio del rumore dei dispositivi reali e delle risorse hardware.

**Glossario** (le abbreviazioni usate nel documento; ognuna è spiegata anche la prima volta che compare)
- *Identificativi del piano*: QA-QL = le fasi (QA ambiente, QB regione ed etichette, QC pre-processing, QD analisi dei dati, QE backbone, QF fusione, QG valutazione delle rappresentazioni, QH stage 1, QI stadio quantistico, QJ confronti classici, QK rumore e risorse, QL presentazione); QA1.1, QI4.3 ... = singoli step; QDn = decisioni; QD-An = figure dell'analisi dei dati; RQn = domande di ricerca; FWn = lavori futuri.
- *Regione e insiemi*: R = la regione di studio (30°S-30°N, 0-90°E); S1-train / S1-val = blocchi di addestramento / validazione dello stage 1 (e della fusione e delle probe); QK-pool = blocchi da cui si estraggono i campioni di addestramento delle SVM a kernel ("campioni QK"); Test = blocchi di test.
- *Rappresentazioni (QF1)*: M-W = solo WAC; M-D = solo DEM; F-E = fusione precoce (un solo backbone su [WAC, rilievo, pendenza]); F-C = concatenazione dei token WAC e DEM; F-CM = concatenazione + MLP per token addestrato come F-X (controllo senza attenzione); F-X = la fusione a cross-attention proposta.
- *Feature map quantistiche (QI2)*: QK-A = codifica ad angoli (una rotazione per qubit, senza entanglement); QK-Z = feature map "ZZ" (rotazioni più porte che accoppiano i qubit vicini con interazioni Z⊗Z, cioè con entanglement); QK-P = kernel proiettato (confronta le misure dei singoli qubit dello stato di QK-Z).
- *Classificatori*: SVM = support vector machine (macchina a vettori di supporto); QSVC = SVM sul kernel quantistico; RBF = kernel gaussiano (radial basis function); RBF-wide = RBF con griglia di larghezze più ampia (controllo di QJ1); LIN-full = SVM lineare sui vettori completi, senza PCA (QJ2); MLP = multilayer perceptron, piccola rete neurale a strati; GBT = alberi di decisione con boosting; VQC / VQR = classificatore / regressore quantistico variazionale.
- *Metriche*: AP = average precision (qualità dell'ordinamento dei candidati; il livello casuale è la quota di positivi); AUROC = area sotto la curva ROC; F1 = media armonica di precisione e recall (0 pessimo, 1 perfetto); IC = intervallo di confidenza; KTA = allineamento kernel-etichette; PCA = analisi delle componenti principali; PCA-16 = le prime 16 componenti principali.
- *Reti*: ViT / ViT-S/14 = Vision Transformer / la sua versione piccola con patch di 14 pixel; DINOv2 = il ViT di Meta AI addestrato senza etichette, usato come backbone congelato; AdamW = ottimizzatore Adam con decadimento dei pesi.
- *Dati e missioni*: LROC = la fotocamera del Lunar Reconnaissance Orbiter; WAC = la sua Wide Angle Camera (immagini ottiche a 643 nm); LOLA = Lunar Orbiter Laser Altimeter; DEM = modello digitale di elevazione; SLDEM2015 = il DEM del 2015 ottenuto da LOLA e dalle immagini stereo della missione giapponese Kaguya; PDS = Planetary Data System della NASA; USGS = United States Geological Survey; RGB = i tre canali di colore rosso, verde, blu.
- *Calcolo*: GPU / CPU = scheda grafica di calcolo / processore; CUDA = piattaforma NVIDIA per il calcolo su GPU; H100 = la GPU NVIDIA del cluster; Aer = il simulatore quantistico di Qiskit; MD5 = impronta digitale del contenuto di un file; QCAI = il workshop della conferenza (Quantum Computing - Artificial Intelligence).
- *Simboli*: d = numero di componenti principali = numero di qubit; n = numero di esempi di addestramento della SVM a kernel; c = larghezza di banda (scala degli angoli) delle feature map; γ = larghezza del kernel gaussiano; C = regolarizzazione della SVM.

---

## 3 - Fase QA - Ambiente e acquisizione dei dati

### Software (QA1.1)
Tutto il lavoro gira nell'ambiente conda `mbulgarelli`. Contiene PyTorch con CUDA (la piattaforma NVIDIA per il calcolo su GPU), `timm` (che fornisce il modello DINOv2 (un Vision Transformer di Meta AI addestrato senza etichette) pre-addestrato), scikit-learn, Qiskit con il suo simulatore Aer, `qiskit-ibm-runtime` (servirà più avanti solo per i modelli di rumore simulati dei dispositivi IBM, senza bisogno di un account), pandas e matplotlib, più i pacchetti opzionali. Abbiamo verificato che un semplice circuito a 2 qubit (stato di Bell) gira sul simulatore Aer e che CUDA è disponibile sui nodi GPU (schede grafiche di calcolo) H100 del cluster (il nodo di login non ha GPU). La versione di Aer installata è quella per CPU (il processore); va bene così, perché i kernel quantistici verranno calcolati direttamente dagli statevector con NumPy/PyTorch su GPU, il che è esatto e abbastanza veloce.

### Dati già presenti su disco (QA2.1, QA2.2)
L'esperimento usa due tipi di dati grezzi, entrambi presi dal dataset del progetto completo sulla stessa macchina:

- **Ottico**: due tile del mosaico globale LROC WAC (Wide Angle Camera della Lunar Reconnaissance Orbiter Camera) a 643 nm, circa 100 m per pixel, che coprono 0-90°E, una a nord e una a sud dell'equatore.
- **Elevazione**: quattro tile di SLDEM2015, un modello digitale di elevazione ottenuto dall'altimetria laser LOLA (Lunar Orbiter Laser Altimeter) e dalle immagini stereo di Kaguya, a 512 pixel per grado (circa 59 m per pixel).

Questi 6 prodotti (14 file, circa 9,6 GB) sono stati copiati in `Dataset/Moon/` con esattamente la stessa struttura di cartelle e gli stessi nomi di file del dataset originale, così che tutti i lettori già esistenti funzionino senza modifiche. Un controllo ha confermato che ogni file copiato è identico al suo originale (stessa dimensione e stesso checksum MD5, l'impronta digitale del contenuto del file), e la tabella di copertura del dataset mostra che tutti e 6 i prodotti sono interamente validi sulla regione scelta, senza buchi.

### Dati scaricati (QA3.1, QA3.3)
Sono state scaricate due cose:

- **Il database dei crateri lunari di Robbins** (USGS, United States Geological Survey, release del 2018), che è la nostra fonte di etichette dei crateri. Elenca circa 1,3 milioni di crateri, ciascuno con latitudine/longitudine del centro e diametro di un cerchio adattato sulle immagini. Abbiamo tenuto solo la tabella, la sua label e il documento descrittivo, e scritto un `SOURCE.md` con provenienza e citazione. L'MD5 della tabella coincide con quello pubblicato; contiene 1.296.796 crateri, 21 colonne, nessun valore mancante e id univoci, e i valori sono coerenti con le unità attese (diametri in km da 1,0 a circa 2.492 km, longitudini 0-360°E).
- **I pesi di DINOv2 ViT-S/14** (Vision Transformer di taglia piccola con patch di 14 pixel) (circa 85 MB), salvati nella cache condivisa di Hugging Face così che i nodi di calcolo possano usarli. Abbiamo verificato le proprietà su cui si basa il resto della pipeline: 21,6 milioni di parametri, patch di 14 pixel, token di 384 valori e, per un input di 224 pixel, una griglia 16 x 16 di token di patch più un token di classe. Il modello si aspetta la normalizzazione standard di ImageNet.

---

## 4 - Fase QB - Regione, etichette e split geografico

### La regione (QB1.1)
La regione di studio R è la fascia **30°S-30°N, 0-90°E**. Questa scelta (decisione QD1) è stata fatta perché la regione è interamente coperta dai dati su disco, contiene sia mari scuri (Tranquillitatis, Serenitatis, Fecunditatis, Nectaris, Crisium) sia altopiani chiari e craterizzati, il Sole illumina il mosaico WAC da ovest su tutta l'area, e la distorsione geometrica della proiezione equirettangolare resta piccola: a ±30° di latitudine un pixel è al massimo circa il 15 % più stretto in direzione est-ovest che in direzione nord-sud. Le alternative considerate erano una fascia più stretta (20°S-20°N, meno distorsione ma meno altopiani) e due regioni separate per addestramento e test (un test di generalizzazione più forte, ma con mescolanze di terreni che potrebbero differire e confondere i risultati).

La **griglia WAC** di R è la griglia di riferimento di tutto l'esperimento: 18.194 x 27.291 pixel a 100 m, ottenuta impilando le due tile WAC all'equatore. La **griglia SLDEM** di R è fatta delle quattro tile DEM (digital elevation model, modello digitale di elevazione) affiancate (30.720 x 46.080 pixel). Abbiamo letto la geometria esatta dalle label PDS (Planetary Data System della NASA) originali e verificato che i prodotti si incastrano senza buchi né sovrapposizioni, e che le immagini sono continue attraverso le giunture.

Qui è emerso un punto sottile. Le tile WAC richiedono una correzione nota di georeferenziazione di un pixel (chiamata D15 nel progetto del dataset). Applicandola a ciascuna tile separatamente resta una riga vuota proprio all'equatore, perché quella riga è memorizzata come prima riga della tile meridionale. Applicandola invece al mosaico già unito, quella riga resta valida e continua con le vicine. La correzione viene quindi applicata al mosaico; di conseguenza solo l'ultimissima colonna di R (che entra per lo spostamento da fuori dei prodotti) è vuota, e si trova fuori da tutte le tile usate in seguito.

### Etichette dei crateri (QB2.1 - QB2.4)
Per prima cosa abbiamo tenuto i crateri del catalogo con il centro dentro R, più un margine di una tile attorno, e convertito le loro coordinate in posizioni riga/colonna sulla griglia WAC. R contiene **131.182 crateri**, per lo più piccoli (circa 89.500 tra 1 e 2 km). Abbiamo controllato la conversione visivamente disegnando i crateri del catalogo su ritagli WAC, per esempio attorno al cratere Theophilus (100 km) e attraverso la giuntura all'equatore: i contorni cadono sui bordi dei crateri, senza spostamenti visibili.

Poi R è stata suddivisa in **sotto-tile (celle) di 16 x 16 pixel**, cioè 1,6 km. Questa dimensione non è arbitraria: dopo che il backbone ridimensiona una tile di 256 pixel a 224 pixel, ogni cella di 16 pixel diventa esattamente una patch di 14 pixel, quindi una cella corrisponde a un token del backbone. Per ogni cella abbiamo memorizzato quanti centri di cratere cadono al suo interno e, per il più grande, il diametro, lo scostamento del centro dal centro della cella e il logaritmo del raggio (questi ultimi valori servono per un futuro passo di regressione).

Ogni cella riceve poi una classe (decisione QD5):
- **positiva**: almeno un centro di cratere con diametro tra 1 e 8 km;
- **negativa**: nessun centro di cratere di 1 km o più;
- **ignorata**: solo centri di crateri più grandi di 8 km, che sono fuori dall'intervallo di dimensioni di questo livello della pipeline.

I risultati sono riportati anche in tre bande di diametro, 1-2, 2-4 e 4-8 km. L'alternativa di usare solo i crateri sopra i 2 km (dove il catalogo è completo) è stata scartata perché eliminerebbe proprio la banda che motiva il progetto. Su R si ottengono 124.812 celle positive (6,4 %), circa 1,8 milioni di negative e 2.681 celle ignorate.

Una decisione progettuale importante (QD10) riguarda i crateri grandi che coprono più celle. Abbiamo scelto **etichette basate sul centro** (opzione A): un cratere appartiene solo alla cella che contiene il suo centro, che è la cella che deve rilevarlo; le altre celle che copre sono negative. È una scelta semplice, coerente con la pipeline principale, che dà una valutazione pulita (un cratere = un rilevamento) e non richiede una non-maximum suppression. Il suo svantaggio è che le celle vicine al centro di un cratere di 3-8 km vedono lo stesso cratere ma sono etichettate come negative. Un'alternativa più morbida (mappe di calore gaussiane attorno ai centri, come in CenterNet) è stata scartata perché non si adatta a una SVM binaria.

Il report sulle etichette mostra che solo il **2,76 %** delle celle positive contiene due o più centri di cratere (mai più di 3), come atteso. Poiché ogni cella rappresenta solo il suo cratere più grande, il 2,7 % dei crateri target non può essere rappresentato, quasi tutti nella banda 1-2 km; questo è un limite superiore del recall perso per costruzione e verrà dichiarato. Le classi sono fortemente sbilanciate: 14,5 negative per ogni positiva in totale, e fino a 248 per positiva nella banda 4-8 km. È per questo che l'average precision è la metrica principale e che i classificatori usano loss bilanciate per classe.

### Split geografico (QB3.1 - QB3.3)
Per evitare data leakage tra addestramento e test, lo split è fatto per area geografica, non per celle casuali. R è divisa in **216 blocchi di 5° x 5°**. I blocchi sono ordinati per densità di crateri di 1-8 km e tagliati in tre strati uguali (densità bassa, media, alta); dentro ogni strato vengono mescolati con un seed fisso e assegnati a quattro insiemi:

- **S1-train** (108 blocchi): addestramento dello strato di fusione, delle probe e in seguito dello stage 1;
- **S1-val** (24 blocchi): validazione, anch'essa fatta di blocchi interi così da essere separata spazialmente;
- **QK-pool** (42 blocchi): la fonte dei campioni per le SVM a kernel, mai vista dallo stage 1;
- **Test** (42 blocchi): la valutazione finale.

Gli insiemi sono risultati molto simili: densità media dei crateri 26,6-27,7 ogni 1.000 km² e 6,3-6,6 % di celle positive in ogni insieme.

Poiché le tile si sovrappongono (vedi QC7), una tile vicina al bordo di un blocco potrebbe mescolare pixel di due insiemi. Quindi ogni tile prende l'insieme del blocco che contiene il suo centro e viene **scartata se la sua impronta tocca un blocco di un altro insieme**. Delle 15.708 tile di R, **ne vengono tenute 12.790**. Questo buffer rimuove circa la stessa quota di positive e di celle, quindi il bilanciamento delle classi negli insiemi è preservato; abbiamo verificato che nessuna coppia di tile tenute e sovrapposte appartiene a insiemi diversi. Lo split è salvato come tabella e disegnato come mappa (poi Fig. D1), che mostra come ogni insieme abbia blocchi sia nei mari settentrionali sia negli altopiani meridionali.

---

## 5 - Fase QC - Pre-processing

Il pre-processing è volutamente una versione ridotta di quello del progetto completo ("opzione 1 clean"). I suoi output, i raster di R, sono salvati in una cartella di stage dedicata, `Dataset/Moon/preprocessing/clean_dataset/qcai2027_R/`, e non vengono mai modificati dagli step successivi; i prodotti originali non vengono mai toccati. È stata seguita una regola generale: un'operazione leggera viene applicata al volo in lettura, mentre un raster viene scritto solo dopo uno step pesante o un cambio di geometria.

### Lettura (QC1.1, QC1.2)
Il **mosaico WAC** di R è stato scritto come un unico array float32 memory-mapped (2 GB), con la correzione di georeferenziazione applicata al mosaico come spiegato sopra. Finestre casuali del raster scritto sono identiche a una lettura diretta dei prodotti, e gli unici pixel no-data sono quelli dell'ultima colonna. Alcune centinaia di valori di riflettanza leggermente negativi (un pixel su un milione) sono stati lasciati così, dato che la normalizzazione li taglia.

Le tile del **DEM** memorizzano il raggio lunare; vengono convertite al volo in elevazione in metri sopra la sfera di riferimento di 1.737,4 km (errore 0 m rispetto ai valori memorizzati). A questo punto non viene scritto alcun raster, perché il DEM viene comunque ricampionato nello step successivo; si evita così una copia da 5,7 GB. Le quote in R vanno da circa -5.900 m a +4.800 m.

### Ricampionamento del DEM sulla griglia WAC (QC2.1, QC2.2)
Per confrontare le due modalità pixel per pixel, il DEM deve stare sulla griglia WAC. Prima viene mediato su blocchi 2 x 2 (il che lo porta a circa 118 m e agisce da filtro anti-aliasing, attenuando anche le righe fini di SLDEM), poi interpolato bilinearmente nei centri dei pixel WAC. Rispetto al DEM nativo, la differenza ha media -0,008 m e deviazione standard 3,4 m: lo step leviga soltanto, senza bias né spostamenti.

Abbiamo poi controllato la **co-registrazione** delle due modalità, cioè che le strutture WAC e DEM cadano nello stesso punto. Usando le finestre diagnostiche già esistenti del progetto del dataset, e rimisurando 600 finestre direttamente sui nostri raster, lo spostamento residuo è di circa 0,05 pixel su entrambi gli assi. Essendo ben al di sotto della soglia di 0,5 pixel, non viene applicata alcuna correzione aggiuntiva.

### Maschere (QC3.1)
Un pixel è valido se sia la WAC sia il DEM sono validi lì, e una tile viene usata solo se almeno il 95 % dei suoi pixel è valido (i pochi pixel non validi verrebbero riempiti con la mediana della tile e segnalati). In pratica R è pulita: gli unici pixel non validi sono quelli dell'ultima colonna WAC, che sta fuori da ogni tile, quindi tutte le tile sono interamente valide e il riempimento non serve mai. La maschera non viene scritta come raster; viene ricalcolata in lettura.

### Il DEM come immagine a tre canali (QC4.1 - QC4.3)
Il backbone si aspetta immagini di tipo RGB (tre canali di colore: rosso, verde, blu), quindi il DEM viene trasformato in tre canali derivati (decisione QD3): **rilievo locale, pendenza e hillshade**. Usare l'elevazione grezza ripetuta su tre canali è stato scartato, perché la topografia a scala chilometrica dominerebbe la dinamica e i crateri piccoli diventerebbero invisibili dopo la normalizzazione.

- **Rilievo locale** (QC4.1): l'elevazione meno una sua copia levigata con un filtro gaussiano (σ = 3,2 km). Questo rimuove i grandi bacini e le pendenze regionali e conserva le forme alla scala dei crateri: la deviazione standard scende da 1.311 m (quota grezza) a 149 m. Al bordo di R il DEM viene specchiato; ciò riguarda solo circa l'1 % delle celle usate.
- **Pendenza** (QC4.2): calcolata in gradi con differenze centrali, tenendo conto che la spaziatura est-ovest dei pixel si riduce con la latitudine. La pendenza mediana è 4,3° e i valori più ripidi (fino a 66°) sono sulle pareti dei crateri.
- **Hillshade** (QC4.3): un'illuminazione sintetica del DEM, **con luce da nord** (azimut 0°, elevazione del Sole 30°).

L'hillshade merita una parola, perché qui il piano è stato cambiato. Inizialmente avevamo previsto un hillshade "multi-direzionale", media di quattro direzioni di luce. Tuttavia, per questo tipo di ombreggiatura, la media su quattro direzioni opposte cancella la parte direzionale e lascia solo una funzione della pendenza: abbiamo misurato che è uguale a 0,5 · cos(pendenza) sul 99,85 % dei pixel, quindi copierebbe semplicemente il canale della pendenza. Abbiamo allora scelto un'unica luce da nord, perpendicolare al Sole delle immagini WAC (che arriva da ovest). Le correlazioni misurate confermano il ragionamento: un hillshade illuminato da ovest ha correlazione +0,61 con la WAC, mentre quello illuminato da nord solo -0,04. In questo modo il ramo DEM mostra le pendenze nord-sud che l'ombreggiatura della WAC nasconde, invece di duplicare l'immagine WAC. Un hillshade illuminato come la WAC avrebbe vanificato lo scopo di avere una seconda modalità.

Gli input finali del backbone sono: **WAC** = la riflettanza ripetuta su tre canali; **DEM** = [rilievo, pendenza, hillshade]; e, per la baseline di early fusion, [WAC, rilievo, pendenza].

### Normalizzazione (QC5.1)
Ogni layer viene riportato in [0, 1] con un min-max robusto, usando il suo 1° e 99° percentile, e poi normalizzato con media e deviazione standard di ImageNet attese da DINOv2. I percentili sono calcolati **solo sui pixel delle tile di S1-train**, così che nessuna informazione di validazione o test entri nel pre-processing. I parametri sono salvati in `normalisation.json` insieme alle formule inverse, quindi la trasformazione può sempre essere annullata. Gli altri insiemi hanno distribuzioni molto simili (0,7-1,2 % di pixel tagliati per lato, contro l'1 % in S1-train).

### Ciò che non viene fatto (QC6.1)
Di proposito, gli input sono usati **così come sono forniti**: nessun destriping, nessun denoising, nessuna correzione fotometrica. Il mosaico WAC conserva quindi le sue deboli strisce est-ovest (distanti circa 1,7 km, all'incirca la dimensione di una sotto-tile) e la sua ombreggiatura topografica, e il DEM viene solo mediato sulla griglia a 100 m. Questo è dichiarato come limite nel paper. I controlli degli step precedenti dimostrano che i raster salvati non contengono correzioni nascoste.

### Tiling e memorizzazione (QC7.1 - QC7.3)
R viene tagliata in **tile di 256 x 256 pixel con passo 176** (80 pixel di sovrapposizione), come nella pipeline principale: 102 x 154 = 15.708 tile. Le origini delle tile stanno su una griglia regolare che parte dall'angolo nord-ovest, senza tile aggiuntive allineate ai bordi sud ed est; di conseguenza circa il 2 % delle celle, lungo quei bordi, non viene mai usato. Si è preferita una griglia regolare che dia lo stesso contesto a ogni cella piuttosto che recuperarle.

Poiché le tile si sovrappongono, una cella potrebbe essere vista da fino a quattro tile. Per valutare ogni cella esattamente una volta, ogni tile **possiede solo le sue 11 x 11 celle centrali** (decisione QD8). Siccome il passo è esattamente di 11 celle, i blocchi centrali delle tile vicine si incastrano senza buchi né sovrapposizioni, e ogni cella posseduta ha almeno 2 celle (3,2 km) di contesto attorno a sé dentro la sua tile. Lo abbiamo verificato sulla griglia reale: ognuna delle 15.708 tile possiede 121 celle, e nessuna cella è posseduta due volte.

Non viene scritto alcun file di tile: una tile è solo una finestra dei raster, letta quando serve. Un'unica classe, `TileReader`, esegue l'intera catena di lettura (finestra, maschera di validità, normalizzazione, composizione dei tre input del backbone e lista delle celle possedute). Come controllo, tutte le tile sono state lette una volta: ogni input è finito e nell'intervallo atteso, e la lettura non carica mai l'intera regione in memoria (al massimo 7,6 MB allocati).

### Controlli (QC8.1, QC8.2)
Abbiamo osservato 10 tile casuali in tutti e quattro i layer con i crateri del catalogo disegnati sopra: i contorni cadono sui bordi in ogni layer. La WAC mostra le pareti interne est illuminate (Sole da ovest), l'hillshade le pareti nord e sud, la pendenza gli anelli delle pareti. Un controllo numerico su 39 crateri più grandi di 2 km ha confermato che il fondo è più basso del bordo in tutti e che la parete interna est è più luminosa di quella ovest in tutti.

I conteggi finali del dataset sono:

| Insieme | Tile usate | Sotto-tile | Positive | Negative | Ignorate | % positive |
|---|---|---|---|---|---|---|
| S1-train | 6.731 | 814.451 | 52.114 | 761.221 | 1.116 | 6,41 |
| S1-val | 1.318 | 159.478 | 9.740 | 149.492 | 246 | 6,12 |
| QK-pool | 2.359 | 285.439 | 18.167 | 266.927 | 345 | 6,37 |
| Test | 2.382 | 288.222 | 18.911 | 268.884 | 427 | 6,57 |
| Totale | 12.790 | 1.547.590 | 98.932 | 1.446.524 | 2.134 | 6,40 |

Il dataset conserva 101.690 dei 128.290 crateri target (79 %); gli altri stanno in tile scartate dal buffer o lungo il bordo non usato di R.

### Differenze rispetto al pre-processing completo (sezione 6.1)
Rispetto al pre-processing completo del progetto principale, questo proxy usa la griglia equirettangolare nativa (nessuna riproiezione, distorsione est-ovest fino al 15 %), solo il livello a 100 m (nessuna piramide, quindi i crateri sopra gli 8 km sono fuori scope), statistiche di normalizzazione calcolate solo su R (non trasferibili ad altre regioni), nessun layer di destriping e un insieme ridotto di maschere e controlli.

---

## 6 - Fase QD - Analisi dei dati

Queste figure e tabelle descrivono i dati al lettore e giustificano le scelte progettuali.

### Mappa della regione, Fig. D1 (QD-A1)
La mappa mostra il mosaico WAC di R con i 216 blocchi colorati per insieme e un punto per blocco la cui dimensione è la densità di crateri. Ogni insieme ha blocchi sia nei mari scuri del nord e del centro sia negli altopiani densi del sud, e la dispersione delle densità dei blocchi è circa la stessa in tutti gli insiemi (mediana 25,5-27,2 crateri ogni 1.000 km²). In altre parole, addestramento e test coprono terreni simili.

### Distribuzione dimensione-frequenza dei crateri, Fig. D2 (QD-A2)
Questo grafico mostra il numero cumulativo di crateri per km² in funzione del diametro, in scala log-log. Tra 2 e 8 km il catalogo segue una legge di potenza con pendenza -1,96 entro il 5 %. Sotto circa 1,6 km la curva si appiattisce, e a 1 km ci sono circa il 18 % di crateri in meno di quanti la legge di potenza ne prevedrebbe. Questo è coerente con l'incompletezza del catalogo per i crateri più piccoli, e spiega parte dei falsi positivi attesi nella banda 1-2 km: alcuni rilevamenti "falsi" potrebbero essere crateri reali assenti dal catalogo. Poiché l'estrapolazione è un'assunzione, il valore dell'82 % è un'indicazione della completezza, non una misura.

### Tile di esempio, Fig. D3 (QD-A3)
Tre tile sono state scelte automaticamente come densa (una tile di altopiano), media e sparsa (una tile di mare), e mostrate in tutti e quattro i layer più la griglia delle etichette. La figura mostra che i crateri sono visibili in ogni layer, ciascuno con la propria illuminazione, che le etichette sono basate sul centro (un cratere grande rende positiva solo la sua cella centrale), e che molti crateri sotto 1 km sono visibili ma etichettati come negativi perché sono sotto l'intervallo delle etichette.

### Tabella del dataset, Table D1 (QD-A4)
Una tabella LaTeX riassume, per insieme, blocchi, area, tile, sotto-tile, positive per banda, negative e celle ignorate (gli stessi numeri visti sopra) e, per modalità, il prodotto, la risoluzione nativa ed effettiva e il punteggio di qualità. La risoluzione effettiva, stimata dallo spettro dell'immagine, è di circa 164 m per la WAC e 97 m per il DEM, anche se i loro pixel nominali sono 100 m e 59 m.

### Complementarità delle modalità, Fig. D4 (QD-A5)
Questa analisi si chiede se le due modalità portino informazioni diverse; se fossero completamente ridondanti, fonderle non potrebbe aiutare. Sono state fatte due misure.

La prima è la correlazione, cella per cella, tra la WAC e un hillshade illuminato come la WAC (da ovest). La correlazione mediana è **0,80**, quindi le due modalità sono **per lo più ridondanti**: gran parte di ciò che la WAC mostra è ombreggiatura della topografia. La ridondanza è più bassa sui terreni pianeggianti: mediana 0,65 nei blocchi a bassa densità, che sono per lo più mari, dove la WAC mostra anche contrasti di albedo (materiali scuri e chiari) che il DEM non ha. Con l'hillshade illuminato da nord effettivamente usato come input la correlazione è circa zero, a conferma che questo canale aggiunge le pendenze nord-sud che la WAC non mostra.

La seconda: per ogni cratere del catalogo di 1-8 km abbiamo misurato quanto chiaramente il suo bordo si distingue dallo sfondo in ciascuna modalità. Con una soglia severa, il 25 % è visibile in entrambe, il 25 % solo nella WAC, il 3,5 % solo nel DEM e il 47 % in nessuna delle due. La WAC porta gran parte del segnale dei crateri di 1-2 km, mentre il DEM guadagna peso al crescere della dimensione: nella banda 4-8 km la quota visibile solo nel DEM (12 %) supera quella visibile solo nella WAC (9 %).

La conclusione per la prima domanda di ricerca è che le due modalità sono in gran parte, ma non del tutto, ridondanti, quindi un **piccolo guadagno dalla fusione** è l'aspettativa realistica.

---

## 7 - Fase QE - Backbone ed estrazione dei token

### Il backbone (QE1.1, QE1.2)
Ogni modalità passa attraverso un **DINOv2 ViT-S/14** congelato, un vision transformer pre-addestrato in modo auto-supervisionato su immagini naturali; gli stessi pesi sono usati per entrambe le modalità, su input separati. Una tile di 256 pixel viene ridimensionata in memoria a 224 pixel, così che le patch di 14 pixel formino una **griglia 16 x 16 di token**, ciascuno un vettore di 384 numeri che descrive una cella di 1,6 km. Il modello è congelato (nessun parametro addestrabile) e dà token identici quando viene rieseguito, o quando le tile vengono processate una per una invece che in batch.

Sono stati fatti due controlli sulla geometria. Il positional embedding del modello, nativamente pensato per una griglia 37 x 37, viene ricampionato una volta sola al caricamento a 16 x 16, e questo dà gli stessi token che si otterrebbero ricampionandolo a ogni passaggio. Poi abbiamo verificato che **il token (i, j) "veda" davvero la cella (i, j)** con un test di occlusione: nascondere una cella (sostituendola con la media della tile) cambia il token corrispondente molto più di qualunque altro, per il 100 % delle celle testate negli input DEM ed early fusion e per il 99 % nell'input WAC (le due eccezioni erano sbagliate di una sola cella). Quindi le etichette stanno sui token che le vedono, senza spostamenti. Questo test ha sostituito il controllo visivo sulle norme dei token previsto inizialmente, perché le norme dei token variano troppo poco per mostrare uno spostamento di una cella.

### Cache dei token (QE2.1, QE2.2)
Eseguire il backbone è la parte costosa, quindi il suo output viene calcolato una volta sola e salvato. Su una GPU H100, in circa 9 minuti, abbiamo estratto i token delle **12.790 tile usate** (quelle tenute dal buffer; quelle scartate non devono mai essere usate) per tre input: WAC, DEM e l'immagine di early fusion [WAC, rilievo, pendenza]. Ciascuno è un array float16 memory-mapped da 2,5 GB. L'estrazione è riprendibile: se il job si interrompe, riparte da dove si era fermato. I controlli mostrano che la memorizzazione in float16 modifica i token in modo trascurabile (similarità coseno 1,000000 rispetto ai valori float32) e che GPU e CPU danno gli stessi token. Da qui in poi tutti gli strati di fusione e le probe vengono addestrati su questi token in cache, in pochi minuti, senza rieseguire il backbone.

---

## 8 - Fase QF - Fusione multimodale

### Le varianti confrontate (QF1)
Si confrontano sei rappresentazioni, ognuna delle quali dà un vettore per cella:

| Id | Variante | Parametri addestrati |
|---|---|---|
| M-W | Solo token WAC (384) | nessuno |
| M-D | Solo token DEM (384) | nessuno |
| F-E | Early fusion: un passaggio del backbone su [WAC, rilievo, pendenza] (384) | nessuno |
| F-C | Concatenazione dei token WAC e DEM (768) | nessuno |
| F-CM | Concatenazione + MLP (multilayer perceptron, una piccola rete neurale a strati) per token a 512, addestrata come F-X (la fusione a cross-attention, riga seguente) | 0,66 M |
| **F-X** | **Fusione a cross-attention bidirezionale** a 512 (proposta) | 4,68 M |

F-CM è il controllo chiave: ha lo stesso addestramento di F-X ma nessuna attention, quindi se F-X batte F-CM il guadagno viene dall'attention (guardare i vicini e l'altra modalità), non semplicemente dall'aver addestrato un proiettore. Tutte e sei le varianti sono lette attraverso un'unica interfaccia comune, che fornisce i vettori e le etichette delle celle possedute di qualunque insieme; abbiamo verificato che i campioni restituiti corrispondono esattamente ai conteggi del dataset.

### La fusione a cross-attention F-X (QF2.1 - QF2.3)
F-X prende le due mappe 16 x 16 di token da 384 valori e proietta ciascuna a 256 valori, aggiungendo un embedding appreso che indica "quale modalità" e un positional encoding 2-D fisso. Seguono **due blocchi di fusione** (decisione QD4). In ogni blocco i token WAC guardano i token DEM e i token DEM guardano i token WAC (cross-attention, 4 head); ogni flusso poi guarda la propria mappa (self-attention) e passa per una piccola rete feed-forward. L'intuizione: un token WAC che vede una mezzaluna scura può "chiedere" ai token DEM vicini se lì c'è una depressione con un bordo, e la self-attention aggiunge il vicinato, dato che i bordi si estendono su più token. Alla fine i due token aggiornati di ogni posizione vengono concatenati e mappati a **512 valori**, ottenendo una mappa fusa 16 x 16 x 512, come nella pipeline principale.

Il modello ha 4,68 milioni di parametri, più dei ~3 milioni stimati inizialmente, perché è stata scelta una larghezza standard della feed-forward (4 volte la dimensione del token). I controlli sul modello non addestrato hanno confermato il comportamento atteso: cambiare il DEM cambia il flusso WAC, una modalità completamente mascherata non ha alcuna influenza sull'output, la posizione dei token conta e ogni parametro riceve un gradiente.

### Addestramento auto-supervisionato (QF3.1 - QF3.5)
La fusione viene addestrata **senza etichette dei crateri**, così da restare una rappresentazione di uso generale (addestrarla con le etichette dei crateri è stato scartato proprio per questo motivo). Le etichette ci sono (vengono dal catalogo Robbins e le usano lo stage 1 e lo stage 2), ma tenerle fuori dalla fusione è una scelta di progetto con quattro motivi:
- **rappresentazioni generali**: la stessa fusione deve poter servire anche ad altri compiti del progetto completo (per esempio il ramo delle anomalie di ghiaccio) e ad altri corpi celesti, senza essere "piegata" sui crateri;
- **nessun bisogno di etichette**: per la maggior parte dei dati planetari le etichette non esistono;
- **nessun bias del catalogo**: il Robbins è incompleto proprio sui crateri di 1-2 km, e una fusione addestrata su di esso ne erediterebbe le lacune;
- **confronto neutro**: le rappresentazioni si possono confrontare in modo equo con le probe (fase QG).

L'informazione sui crateri entra solo dopo, nelle "teste" del compito (stage 1 e stage 2). Una compressione o riduzione addestrata con le etichette resta quindi possibile, ma appartiene al ramo dei crateri, non al blocco di fusione (vedi la discussione in QI4.4).

Il compito di addestramento è **ricostruire token nascosti**:

- **Mascheramento** (QF3.1): in ogni tile viene nascosto un 40-75 % casuale dei token di ciascuna modalità, in modo indipendente per le due modalità, e sostituito con un "mask token" appreso. Una posizione può quindi essere nascosta in una modalità e visibile nell'altra, il che spinge il modello a usare l'altra modalità per colmare il vuoto.
- **Modality drop-out** (QF3.2): nel 20 % delle tile un'intera modalità viene nascosta e l'altra è completamente visibile, così il modello deve ricostruire una modalità interamente a partire dall'altra.
- **Decoder e loss** (QF3.3): un decoder leggero (un layer transformer e due teste lineari, 1,1 M di parametri, scartato dopo l'addestramento) legge la mappa fusa e predice i token originali di entrambe le modalità. La loss è l'errore quadratico medio sulle sole posizioni nascoste, mediato sulle due modalità. I target sono normalizzati in modo che predire zero dia una loss di 1,0. Un breve fit di prova ha mostrato che il compito è apprendibile.

L'**addestramento** (QF3.4) usa le tile di S1-train, l'ottimizzatore AdamW (variante di Adam con decadimento dei pesi) con warm-up e decadimento coseno, batch di 128 tile ed early stopping sulla loss di validazione; F-CM è addestrata con esattamente la stessa ricetta e lo stesso decoder. Per ogni modello il learning rate viene scelto sul set di validazione, e il modello viene poi addestrato con tre seed per misurarne la variabilità. Il modello usato a valle è sempre quello con seed 2027, fissato in anticipo (non il migliore dei tre).

Con la prima ricetta (learning rate fino a 3e-4, 50 epoche) la migliore epoca era sempre l'ultima, segno che i modelli non avevano finito di imparare. Abbiamo quindi eseguito uno sweep esplorativo (QF3.5) con learning rate più alti e 100 epoche, con una regola fissata **prima** di lanciarlo: la nuova ricetta avrebbe sostituito la vecchia solo se avesse abbassato la loss di validazione di F-X di almeno il 2 % e fosse stata stabile. L'ha abbassata del 19,5 %, quindi la nuova ricetta (learning rate 1e-3, 100 epoche, 5 epoche di warm-up) è stata adottata, tutti i seed riaddestrati e i controlli rifatti. Abbiamo deciso di non spingere oltre il learning rate: la curva è piatta alla fine, il divario tra addestramento e validazione comincia a crescere e vogliamo evitare di ottimizzare troppo sul set di validazione.

Risultati con la ricetta adottata (migliore loss di validazione, media ± deviazione standard su 3 seed):

- **F-X 0,1627 ± 0,0008**
- **F-CM 0,1865 ± 0,0001**

F-X ricostruisce il **12,8 % meglio** di F-CM, una differenza di circa 30 deviazioni standard.

### Controlli della fusione (QF4.1 - QF4.3)
**Ricostruzione cross-modale** (QF4.1): abbiamo nascosto completamente una modalità e chiesto a ciascun modello di ricostruirla dall'altra. F-X fa meglio in entrambe le direzioni: WAC dal DEM 0,197 contro 0,208 (-5,3 %), DEM dalla WAC 0,256 contro 0,267 (-4,1 %). Quindi l'attention cattura più informazione cross-modale del proiettore per token. La loss sale solo moderatamente quando una modalità è nascosta del tutto, il che mostra di nuovo che gran parte di ciascuna modalità è prevedibile dall'altra, in linea con la ridondanza trovata in QD-A5.

**Controlli di sanità** (QF4.2): le mappe fuse non sono costanti (ogni canale varia, nessuna tile è piatta). Un'analisi delle componenti principali mostra che le prime 16 componenti contengono il 51 % della varianza per F-X e il 35 % per F-CM; F-X è la più comprimibile delle due, il che conta perché lo stage quantistico vedrà solo 16 componenti. Guardando l'attention, è per lo più **locale**: un token presta attenzione soprattutto alla stessa posizione e ai suoi vicini nell'altra modalità (circa il 54-66 % del peso nell'intorno 3 x 3 nel secondo blocco, contro il 3,5 % se fosse uniforme). Nel primo blocco, i token WAC danno circa il 40 % di peso in più della media alle celle di cratere del DEM, una moderata preferenza per i crateri; questa preferenza non si vede nel secondo blocco. Nel paper verrà descritta come "attention cross-modale locale con una moderata preferenza per i crateri", senza affermare che l'attention vada a cercare i crateri.

**Cache delle mappe fuse** (QF4.3): le mappe fuse dei modelli F-X e F-CM usati a valle sono state calcolate per tutte le tile usate e salvate come i token del backbone (3,35 GB ciascuna), così che tutte e sei le varianti vengano lette allo stesso modo dalle probe e dagli stage successivi.

---

## 9 - Fase QG - Valutazione delle rappresentazioni

### Linear probe (QG1)
Per confrontare in modo equo le sei rappresentazioni, le teniamo congelate e addestriamo sopra solo un classificatore molto semplice: una **regressione logistica bilanciata per classe** (la "linear probe"), addestrata sulle celle di S1-train e valutata una sola volta sulle celle di test. La sua forza di regolarizzazione viene scelta su S1-val, e il fit viene fatto su GPU con un metodo che dà lo stesso risultato di scikit-learn. La metrica principale è l'**average precision (AP)**, che riassume la curva precision-recall ed è significativa quando i positivi sono rari (qui il 6,6 % delle celle di test; un classificatore casuale otterrebbe circa 0,066). Le incertezze sono intervalli di confidenza al 95 % ottenuti ricampionando 1.000 volte i 42 blocchi di test (block bootstrap, per rispettare la correlazione spaziale tra celle vicine).

Risultati sul test, AP [IC 95 %, intervallo di confidenza] (AUROC, area sotto la curva ROC):

| Variante | AP | AUROC |
|---|---|---|
| M-W (solo WAC) | 0,460 [0,441, 0,479] | 0,901 |
| M-D (solo DEM) | 0,498 [0,478, 0,517] | 0,900 |
| F-E (early fusion) | 0,490 [0,471, 0,510] | 0,905 |
| F-C (concatenazione) | 0,560 [0,542, 0,578] | 0,923 |
| F-CM (proiettore addestrato) | 0,562 [0,544, 0,581] | 0,921 |
| **F-X (cross-attention)** | **0,574** [0,557, 0,592] | **0,927** |

Che cosa dicono questi numeri:

1. **Due modalità sono molto meglio di una.** Ogni late fusion guadagna 0,06-0,10 di AP sulla migliore modalità singola. Quindi, per il classificatore, i token WAC e DEM sono complementari, anche se le immagini in sé sono in gran parte ridondanti.
2. **L'early fusion non funziona bene.** Dare [WAC, rilievo, pendenza] a un unico backbone pre-addestrato su foto RGB va peggio del solo DEM: mescolare le modalità all'ingresso fa perdere informazione.
3. **La cross-attention aggiunge un guadagno piccolo ma significativo.** F-X batte la concatenazione di +0,014 di AP e il controllo addestrato F-CM di +0,012, con intervalli di confidenza che escludono lo zero (F-X vince nel 100 % dei ricampionamenti bootstrap contro ogni altra variante). Questo sostiene la prima domanda di ricerca, con l'entità modesta attesa vista la ridondanza delle modalità.
4. **La banda 4-8 km è molto difficile per ogni variante** (AP 0,01-0,04, con una prevalenza di appena lo 0,44 %). I crateri grandi si estendono su più celle e la cella centrale è difficile da distinguere dalle celle del bordo; per banda, il guadagno di F-X su F-C e F-CM è significativo per 1-2 e 2-4 km ma non per 4-8 km. Questo verrà discusso insieme allo stage 1 e tra i limiti dello studio.

### MLP probe (QG2)
La linear probe misura quanta informazione sui crateri si può leggere con un semplice "taglio" lineare nello spazio della rappresentazione. Per capire se c'è altra informazione che un classificatore lineare non riesce a sfruttare, ripetiamo la stessa prova con una **piccola rete neurale**: uno strato nascosto da 256 neuroni (MLP). Il protocollo è identico a quello della linear probe: stesse celle, stessa perdita bilanciata per classe, addestramento su S1-train, scelta su S1-val e una sola valutazione sul test. La regolarizzazione (il *weight decay*) viene scelta su S1-val, e l'addestramento si ferma quando l'AP su S1-val smette di migliorare (*early stopping*). Siccome una rete neurale dipende dall'inizializzazione casuale, la addestriamo con **tre seed** e riportiamo la media; le differenze tra seed sono piccole (deviazione standard 0,001-0,003 di AP).

Risultati sul test, AP [IC 95 %] (AUROC), con a destra il guadagno rispetto alla linear probe:

| Variante | AP MLP | AUROC | Guadagno sul lineare |
|---|---|---|---|
| M-W (solo WAC) | 0,507 [0,488, 0,527] | 0,909 | +0,048 |
| M-D (solo DEM) | 0,559 [0,542, 0,579] | 0,913 | +0,062 |
| F-E (early fusion) | 0,550 [0,530, 0,569] | 0,918 | +0,059 |
| F-C (concatenazione) | 0,603 [0,586, 0,622] | 0,931 | +0,043 |
| F-CM (proiettore addestrato) | 0,601 [0,584, 0,619] | 0,929 | +0,038 |
| **F-X (cross-attention)** | **0,605** [0,588, 0,622] | **0,933** | +0,031 |

Che cosa dicono questi numeri:

1. **Una testa non lineare estrae di più da tutte le rappresentazioni** (da +0,03 a +0,06 di AP): una parte dell'informazione sui crateri c'è, ma non è separabile con un taglio lineare.
2. **Con il MLP la concatenazione raggiunge F-X.** La differenza F-X meno F-C è di appena +0,0015, con un intervallo di confidenza che include lo zero; contro F-CM resta un vantaggio significativo ma minimo (+0,004). Quindi F-X e la semplice concatenazione contengono più o meno la stessa informazione sui crateri: il vantaggio di F-X visto con la linear probe (+0,014) non viene da informazione in più, ma dal fatto che F-X la **organizza in modo più leggibile linearmente**. Lo conferma anche il fatto che F-X è la rappresentazione che guadagna meno passando dal lineare al MLP: era già la più "lineare".
3. **Le fusioni tardive restano nettamente sopra le singole modalità e la early fusion** (F-X supera il solo DEM di 0,046 e il solo WAC di 0,098).

Per la prima domanda di ricerca questo rende la conclusione più precisa: la fusione a cross-attention non aggiunge informazione rispetto alla concatenazione, ma la rende accessibile a classificatori semplici. Questo conta per il resto della pipeline, perché lo stadio quantistico vedrà solo poche componenti principali (prova QG3). Due dettagli tecnici: il valore di *weight decay* scelto cambia pochissimo (l'AP su S1-val varia di meno di 0,001 tra i valori provati), e l'epoca migliore arriva presto (tra la 3 e la 11 su un massimo di 30), segno che con 800.000 esempi la rete tende a sovra-adattarsi rapidamente.

### Probe compressa con PCA, analisi delle componenti principali (QG3)
Lo stadio quantistico non potrà usare i vettori interi (da 384 a 768 numeri): ogni qubit riceverà una sola variabile, quindi la rappresentazione va compressa a poche componenti. Per sapere quanta informazione sui crateri sopravvive a questa compressione, ripetiamo la linear probe dopo una **PCA** (analisi delle componenti principali) stimata su S1-train. La PCA trova le direzioni lungo cui i vettori variano di più; teniamo le prime *d*, le standardizziamo e addestriamo sopra la stessa regressione logistica di QG1. La PCA è fatta sui vettori solo centrati (cioè sulla covarianza, senza riscalare ogni canale), e la stessa scelta verrà usata nello stadio quantistico. Oltre a *d* = 16, il valore della Tabella I, proviamo tutti i *d* che userà lo stadio quantistico: 4, 8, 12, 16 e 20.

Risultati sul test con *d* = 16, AP [IC 95 %], con la varianza spiegata dalle 16 componenti e, per confronto, l'AP della linear probe sui vettori interi:

| Variante | AP con PCA-16 (prime 16 componenti principali) | Varianza spiegata | AP senza compressione |
|---|---|---|---|
| M-W (solo WAC) | 0,358 [0,338, 0,379] | 65 % | 0,460 |
| M-D (solo DEM) | 0,373 [0,353, 0,394] | 61 % | 0,498 |
| F-E (early fusion) | 0,389 [0,369, 0,410] | 59 % | 0,490 |
| F-C (concatenazione) | 0,416 [0,396, 0,439] | 53 % | 0,560 |
| F-CM (proiettore addestrato) | 0,396 [0,373, 0,420] | 35 % | 0,562 |
| **F-X (cross-attention)** | **0,444** [0,423, 0,466] | 52 % | 0,574 |

E al variare del numero di componenti (AP per *d* = 4, 8, 12, 16, 20): F-X 0,416 / 0,433 / 0,440 / 0,444 / 0,457; F-C 0,388 / 0,396 / 0,410 / 0,416 / 0,418; F-CM 0,259 / 0,312 / 0,354 / 0,396 / 0,417.

Che cosa dicono questi numeri:

1. **La compressione costa molto a tutte le rappresentazioni.** F-X passa da 0,574 a 0,444. Vuol dire che una buona parte dell'informazione sui crateri non sta nelle direzioni di massima variazione: la PCA non guarda le etichette, e tiene le variazioni più forti (illuminazione, tipo di terreno), che non coincidono per forza con i crateri.
2. **Sotto compressione il vantaggio di F-X raddoppia.** Con 16 componenti F-X supera la concatenazione di +0,028 (contro +0,014 sui vettori interi e quasi zero con il MLP), e la supera in modo significativo per ogni *d*. Anzi, **F-X con sole 4 componenti (0,416) vale quanto F-C con 16-20**. F-X concentra l'informazione utile in poche direzioni: è esattamente la proprietà che serve allo stadio quantistico, ed è il risultato più forte finora a favore della prima domanda di ricerca.
3. **F-CM è la rappresentazione che si comprime peggio** (0,259 con 4 componenti, che spiegano solo il 13 % della varianza): la sua informazione è sparsa su molte direzioni.
4. **Per F-X l'AP cresce ancora da 16 a 20 componenti** (+0,013), quindi le configurazioni con più qubit potrebbero aiutare.

Un dettaglio tecnico: con al massimo 20 variabili e circa 800.000 esempi, la forza di regolarizzazione della regressione logistica non conta (l'AP su S1-val cambia di 0,001 al massimo tra i valori provati).

### Robustezza alla modalità mancante (QG4)
Nella pratica può capitare che una delle due modalità non sia disponibile in una certa zona (un buco nel DEM, un'immagine WAC mancante). Per vedere come si comportano le rappresentazioni in questo caso, prendiamo le probe addestrate sui dati completi (la linear probe di QG1 e la probe PCA-16 di QG3) e al test **togliamo una modalità**. F-X e F-CM, che durante l'addestramento hanno imparato a gestire una modalità assente (*modality drop-out*), ricevono al posto della modalità mancante il loro *mask token*, il vettore appreso che significa "qui questa modalità non c'è". F-C, la semplice concatenazione, non ha un mask token: la metà mancante viene riempita con un valore costante. Abbiamo provato sia con gli zeri sia con la media di S1-train, e i risultati sono identici, per una ragione matematica: con una probe lineare un riempimento costante sposta tutti i punteggi della stessa quantità, e l'AP dipende solo dall'ordine dei punteggi. In pratica, a F-C senza una modalità resta solo la parte della probe che legge la modalità presente.

Il piano prevedeva che F-X si degradasse gradualmente e che F-C crollasse. Ecco i risultati sul test (AP, e tra parentesi la frazione dell'AP completo che si conserva):

| Probe | Variante | Completo | Senza DEM (solo WAC) | Senza WAC (solo DEM) |
|---|---|---|---|---|
| Lineare | F-C | 0,560 | 0,397 (71 %) | 0,451 (81 %) |
| Lineare | F-CM | 0,562 | 0,161 (29 %) | 0,147 (26 %) |
| Lineare | **F-X** | 0,574 | 0,372 (65 %) | **0,482** (84 %) |
| PCA-16 | F-C | 0,416 | 0,339 (81 %) | 0,341 (82 %) |
| PCA-16 | F-CM | 0,396 | **0,366** (93 %) | 0,319 (81 %) |
| PCA-16 | **F-X** | 0,444 | 0,364 (82 %) | **0,362** (82 %) |

Per confronto, le probe addestrate su una sola modalità ottengono 0,460 (solo WAC) e 0,498 (solo DEM) con la linear probe, e 0,358 e 0,373 con PCA-16.

La previsione si è verificata solo in parte:

1. **F-X si degrada gradualmente, come previsto.** Conserva tra il 65 % e l'84 % dell'AP con la linear probe e l'82 % con PCA-16. Con PCA-16 resta la migliore in entrambe le condizioni (+0,02 su F-C), e con la sola WAC vale quanto una probe addestrata apposta sulla sola WAC (0,364 contro 0,358).
2. **F-C non crolla, contrariamente al piano.** Con una probe lineare, togliere una modalità significa solo perdere la sua metà del punteggio: F-C conserva il 71-82 % dell'AP. Con i vettori interi e senza DEM, F-C è anzi un po' meglio di F-X (−0,025).
3. **A crollare è invece F-CM** con la linear probe (resta il 26-29 % dell'AP), anche se è stata addestrata con lo stesso modality drop-out di F-X. Quando F-CM riceve il mask token, il suo vettore finisce lontano dai vettori completi su cui la probe è stata addestrata; la cross-attention di F-X invece "ricostruisce" la modalità mancante usando l'altra, e il vettore resta utilizzabile. La robustezza di F-X viene quindi dall'attenzione, non dal solo addestramento con drop-out.

Nel paper si potrà dire che F-X è l'unica fusione appresa che resta utilizzabile quando manca una modalità; l'affermazione "la concatenazione crolla" invece non è supportata con probe lineari e non verrà fatta.

### Risultati per tipo di terreno (QG6)
L'analisi QD-A5 aveva mostrato che WAC e DEM sono in gran parte ridondanti, tranne che nelle zone con pochi crateri, simili ai mari lunari: lì la correlazione tra le due immagini scende (0,65 contro 0,83-0,85 altrove), perché la WAC vede contrasti di albedo che il DEM non ha e la topografia è debole. L'ipotesi era quindi che la fusione aiutasse soprattutto su quel tipo di terreno. Per verificarla dividiamo i blocchi di test nei tre terzili di densità di crateri usati per lo split (14 blocchi ciascuno): il terzile basso somiglia ai mari, quello alto agli altipiani. È un'approssimazione, perché non usiamo una vera mappa dei mari. Non serve riaddestrare nulla: usiamo i punteggi di test già salvati della linear probe (QG1) e della probe PCA-16 (QG3), e calcoliamo gli intervalli di confidenza ricampionando i blocchi dentro ciascun terzile.

Ogni riga della tabella è una differenza di AP: **l'AP di F-X** (la fusione a cross-attention di WAC e DEM, presente in tutte le righe) **meno l'AP di una rappresentazione di confronto**. Cambia solo il confronto:

- **solo WAC (M-W)**: una probe addestrata sui soli token WAC, senza alcuna fusione. La differenza dice quanto si guadagna usando le due modalità fuse invece della sola WAC.
- **miglior modalità singola (M-D)**: una probe sui soli token DEM, che è la modalità singola migliore in tutti e tre i terzili, anche questa senza fusione. La differenza dice quanto si guadagna fondendo invece di usare la migliore modalità da sola.
- **concatenazione (F-C)**: i token WAC e DEM semplicemente affiancati, cioè la fusione più semplice possibile, senza attenzione. La differenza dice quanto aggiunge il livello di fusione con attenzione rispetto alla fusione banale.

Le prime due righe misurano quindi il vantaggio di usare due modalità invece di una; le ultime due il vantaggio di fonderle con l'attenzione invece di affiancarle. Ad esempio, nel terzile basso con la linear probe F-X ottiene 0,563 e la sola WAC 0,422: la prima riga vale 0,563 − 0,422 = 0,140.

Differenze di AP (F-X meno il confronto indicato):

| F-X meno… | Probe | Terzile basso (mari) | Medio | Alto (altipiani) | Basso meno alto [IC 95 %] |
|---|---|---|---|---|---|
| solo WAC (M-W, nessuna fusione) | lineare | 0,140 | 0,115 | 0,108 | **+0,033** [0,014, 0,046] |
| solo WAC (M-W, nessuna fusione) | PCA-16 | 0,119 | 0,085 | 0,073 | **+0,046** [0,019, 0,070] |
| solo DEM (M-D, miglior singola, nessuna fusione) | lineare | 0,079 | 0,076 | 0,073 | +0,007 [−0,009, 0,022] |
| solo DEM (M-D, miglior singola, nessuna fusione) | PCA-16 | 0,083 | 0,068 | 0,065 | +0,018 [−0,010, 0,055] |
| concatenazione (F-C, fusione senza attenzione) | lineare | 0,024 | 0,011 | 0,014 | +0,010 [−0,003, 0,022] |
| concatenazione (F-C, fusione senza attenzione) | PCA-16 | 0,053 | 0,026 | 0,019 | **+0,034** [0,012, 0,060] |

Che cosa dicono questi numeri:

1. **Tutti i guadagni di fusione sono massimi sul terreno tipo mare**, come previsto.
2. La differenza con gli altipiani è **significativa** per il guadagno rispetto alla sola WAC e, nell'impostazione compressa che userà lo stadio quantistico, per il vantaggio di F-X sulla concatenazione: +0,053 di AP sui mari contro +0,019 sugli altipiani.
3. Il guadagno rispetto al DEM da solo è anch'esso più alto sui mari, ma la differenza non è significativa.
4. F-X resta ben sopra il caso casuale in tutti i terzili, anche dove i crateri sono rari: nel terzile basso la prevalenza dei positivi è appena il 2,9 % e l'AP di F-X è 0,56, circa 19 volte il caso.

Nel paper questo diventa una frase nei risultati ("il guadagno della fusione è massimo sul terreno a bassa densità, simile ai mari") e una nella discussione, che lo spiega con la minore ridondanza delle due modalità misurata in QD-A5. Un guadagno complessivo modesto si legge quindi come conseguenza del fatto che, su gran parte della Luna, WAC e DEM dicono quasi la stessa cosa, e non come un fallimento della fusione.

## 10 - Fase QH - Stage 1 e campionamento dei candidati

Il rivelatore finale ha due stadi. Il **primo stadio** (stage 1) è un classificatore classico e leggero che dà un punteggio a ogni sotto-tile e tiene solo quelli che potrebbero contenere il centro di un cratere: i **candidati**. Il secondo stadio, quantistico, dovrà poi riclassificare solo questi candidati. Lo stage 1 serve quindi a scartare in fretta la gran parte delle celle sicuramente vuote, perdendo il meno possibile dei crateri veri.

**Perché servono lo stage 1 e i campioni (nota per il progetto completo).** Il kernel quantistico pone due problemi di costo distinti, e ciascuno ha la sua soluzione.

1. **Quanti circuiti servono → stage 1.** Ogni sotto-tile viene codificato da solo in un circuito, e ogni valore del kernel confronta due sotto-tile con un circuito separato: i 256 vettori di un tile non vengono mai codificati insieme. In inferenza ogni sotto-tile da classificare va confrontato con tutti i support vector della SVM (circa 390), e ogni circuito va ripetuto per migliaia di shot. Il numero di circuiti dipende quindi da quanti sotto-tile ci sono in tutta la regione analizzata, non dalla dimensione della mappa di un tile. Il problema c'è anche con mappe 16 × 16; con mappe più fini per la stessa regione (per esempio 128 × 128) i sotto-tile, e quindi i circuiti, crescerebbero nella stessa proporzione. In R ci sono 1,9 milioni di sotto-tile. Lo stage 1 ne scarta circa due terzi, quelli chiaramente vuoti, e lascia circa 680.000 candidati (QK3). Il costo resta comunque di anni di macchina quantistica per l'intera regione, quindi l'hardware reale può servire solo per una piccola validazione. Lo stage 1 ha anche un secondo vantaggio: il campione di addestramento della SVM viene preso tra i candidati, cioè crateri veri e falsi allarmi "difficili", invece di una grande maggioranza di celle banalmente vuote.
2. **Quanti qubit servono in un circuito → PCA.** Con le codifiche usate ogni numero del vettore occupa un qubit. Un vettore da 512 valori richiederebbe 512 qubit, più di quanti ne abbiano i processori attuali, e un circuito così profondo da essere dominato dal rumore e impossibile da simulare, perché il costo della simulazione cresce come 2^d. Per questo ogni vettore viene compresso a 4-20 componenti principali (QI1), cioè 4-20 qubit. Questo problema è indipendente dallo stage 1.

In questo esperimento, che è uno studio di fattibilità, c'è poi un terzo taglio, dovuto al costo della *simulazione*. La SVM viene addestrata su campioni stratificati di 100-2.000 candidati del QK-pool (QH4) e valutata su un sottoinsieme fisso e rappresentativo di 2.000 candidati di test (QH5), invece che su tutti i circa 110.000 candidati di test. Nella pipeline completa anche l'addestramento usa un campione (500-3.000 sotto-tile), ma in inferenza si classificherebbero *tutti* i candidati della regione analizzata: è il conto che QK3 stima in circa 2·10⁹ circuiti per tutta la Luna.

### La testa dello stage 1 (QH1)
Per ognuna delle sei rappresentazioni addestriamo una piccola rete convoluzionale: due convoluzioni 3 × 3 (con 128 canali intermedi) che leggono l'intera mappa 16 × 16 di un tile e danno un punteggio a ogni sotto-tile. A differenza delle probe della fase QG, che guardavano un sotto-tile alla volta, qui ogni punteggio tiene conto anche delle celle vicine, utile per esempio quando il bordo di un cratere si estende nelle celle accanto. L'addestramento usa S1-train, con la stessa perdita bilanciata per classe delle probe e solo sulle celle che ogni tile "possiede" (le 11 × 11 centrali, così ogni cella conta una volta sola; il bordo del tile fa solo da contesto). Si ferma quando l'AP su S1-val smette di migliorare. Le sei teste vengono salvate per i passi successivi.

Risultati, AP su S1-val, con a destra l'AP su S1-val delle probe lineare e MLP della fase QG:

| Variante | Stage 1 | Probe lineare | Probe MLP |
|---|---|---|---|
| M-W (solo WAC) | 0,519 | 0,441 | 0,499 |
| M-D (solo DEM) | 0,547 | 0,491 | 0,561 |
| F-E (early fusion) | 0,529 | 0,473 | 0,544 |
| F-C (concatenazione) | 0,590 | 0,549 | 0,597 |
| F-CM (proiettore addestrato) | 0,588 | 0,555 | 0,596 |
| **F-X (cross-attention)** | **0,593** | 0,563 | 0,596 |

Che cosa dicono questi numeri:

1. **L'ordine delle rappresentazioni non cambia**: F-X è prima, di poco davanti a F-C e F-CM, e tutte le fusioni tardive sono nettamente sopra le singole modalità.
2. **Il contesto delle celle vicine aggiunge poco.** Lo stage 1 è allo stesso livello della probe MLP, che guardava una cella alla volta; aiuta solo con la sola WAC (+0,02), dove vedere le celle vicine compensa in parte l'assenza della topografia.
3. **La rete si sovra-adatta in fretta**: il punteggio su S1-val è massimo già alla terza o quarta epoca e poi cala, mentre l'errore di addestramento continua a scendere. L'*early stopping* conserva la versione migliore.

Le prestazioni sul test e la soglia per scegliere i candidati arriveranno nei passi successivi (QH2, QH3).

### La soglia dei candidati (QH2)
Lo stage 1 dà a ogni sotto-tile un punteggio; per decidere quali diventano candidati serve una soglia. La scegliamo su S1-val in modo da **tenere il 95 % dei crateri veri**: è il richiamo massimo che il rivelatore potrà mai raggiungere, perché un cratere scartato qui non può essere recuperato dallo stadio quantistico.

A questa soglia, per F-X restano candidati il **35,8 %** dei sotto-tile, e tra questi solo il 16 % contiene davvero il centro di un cratere. Le altre rappresentazioni vanno da 35,6 % (F-CM) a 43,4 % (solo DEM). Il piano si aspettava il 10-20 %: per ottenerlo bisognerebbe accettare un richiamo dell'80-85 %. Ecco il compromesso per F-X:

| Richiamo tenuto | Candidati | Precisione | Richiamo crateri 4-8 km |
|---|---|---|---|
| 80 % | 15,3 % | 0,32 | 0,69 |
| 85 % | 18,9 % | 0,28 | 0,77 |
| 90 % | 25,0 % | 0,22 | 0,83 |
| **95 %** | **35,8 %** | **0,16** | **0,91** |

Abbassare il richiamo ridurrebbe molto i candidati, ma farebbe perdere già nel filtro da un quarto a un terzo dei crateri grandi (4-8 km), che sono anche i più difficili. Abbiamo quindi **mantenuto il 95 %** previsto dal piano. Il costo dello stadio quantistico resta comunque limitato, perché userà un sottoinsieme fisso di 2.000 candidati di test e campioni di addestramento estratti a sorte. In compenso lo stadio quantistico avrà un compito impegnativo: tra i candidati, circa l'84 % sono falsi allarmi da eliminare, ed è proprio qui che potrà mostrare quanto migliora la precisione.

Abbiamo poi guardato **dove cadono i falsi allarmi**. Con etichette basate sul centro, le celle accanto al centro di un cratere, o dentro un cratere di 3-8 km, vedono lo stesso cratere ma sono etichettate come negative, e possono confondere il classificatore. Il piano prevedeva un rimedio (QB2.5: ignorare in addestramento un anello di celle attorno ai centri dei crateri grandi) da applicare se più del 30 % dei falsi positivi cadesse in queste zone. Il 54 % dei falsi positivi di F-X cade lì, quindi la regola scatterebbe. Però queste zone contengono già il 34 % di tutte le celle negative, perché i crateri sono così numerosi che i vicini dei loro centri coprono un terzo della mappa: la regola del 30 % scatta quasi per costruzione. Il dato più informativo è che i falsi positivi sono 1,6 volte più frequenti accanto ai centri e 2 volte più frequenti dentro i crateri di 3-8 km rispetto a una cella negativa qualunque: un effetto reale ma moderato. Abbiamo deciso di **non applicare il rimedio**: richiederebbe di riaddestrare lo stage 1 e le probe (rifacendo la Tabella I), con un effetto incerto, dato che il test mantiene le etichette basate sul centro. L'effetto viene riportato come limite delle etichette basate sul centro, e sono proprio questi falsi allarmi che lo stadio quantistico dovrà imparare a eliminare.

### I candidati di QK-pool e del test (QH3)
Ora applichiamo lo stage 1, con le soglie appena scelte, ai due insiemi che lo stadio quantistico userà: **QK-pool**, da cui verranno estratti i campioni di addestramento del classificatore quantistico, e il **test**, su cui verrà valutato. Nessuno dei due è mai stato visto dallo stage 1. Salviamo il punteggio di tutti i sotto-tile, non solo dei candidati, perché servirà anche per misurare le prestazioni "da capo a fondo" del rivelatore a due stadi sull'intero test.

Risultati per F-X:

| Insieme | Candidati | Richiamo (1-2 / 2-4 / 4-8 km) | Precisione |
|---|---|---|---|
| QK-pool | 96.078 (33,7 %) | 0,960 (0,959 / 0,973 / 0,919) | 0,182 |
| Test | 109.669 (38,1 %) | 0,959 (0,958 / 0,971 / 0,918) | 0,166 |

1. **Il richiamo del 95 % scelto su S1-val regge sugli insiemi nuovi** (tra 0,955 e 0,963 per tutte le rappresentazioni): la soglia generalizza bene. La frazione di candidati è un po' più alta sul test (38 % contro 36 %).
2. **Lo stage 1 da solo ottiene sul test un AP di 0,606 con F-X** [0,589, 0,624], davanti a F-C e F-CM (0,598) e alle singole modalità (0,527-0,549). Questo valore sarà la linea di riferimento per lo stadio quantistico: il secondo stadio dovrà fare meglio di così per dimostrare di essere utile.
3. **Tra i candidati di QK-pool ci sono abbastanza esempi** per costruire i campioni di addestramento stratificati del passo successivo: per F-X 17.444 crateri (di cui 966 di 4-8 km) e 78.492 falsi allarmi, distribuiti in tutti e tre i terzili di densità.

### I campioni di addestramento per il kernel quantistico (QH4)
Un classificatore a kernel quantistico non può essere addestrato su centinaia di migliaia di esempi: il kernel va calcolato tra ogni coppia di campioni, simulando un circuito per ciascuno. Il piano prevede quindi campioni piccoli, di **100, 250, 500, 1.000 e 2.000 sotto-tile**, estratti dai candidati di QK-pool (un insieme mai visto dallo stage 1), e ripetuti con **5 seed** diversi per misurare quanto i risultati dipendono dalla particolare estrazione.

Come sono costruiti:

1. **Classi bilanciate**: metà crateri e metà falsi allarmi. Tra i candidati i crateri sono solo il 18 %, e con 100 esempi ne avremmo appena 18, troppo pochi per imparare. Il test invece mantiene la proporzione naturale, ed è lì che si misurano AP e precisione.
2. **Tutte le dimensioni di cratere presenti**: i crateri sono divisi tra le tre bande in proporzione ai candidati, ma con almeno il 10 % per banda, così i crateri di 4-8 km (solo il 5,5 % dei candidati) non spariscono dai campioni piccoli: con 100 esempi ce ne sono 5 invece di 3.
3. **Tutti i terreni presenti**: dentro ogni classe e banda, gli esempi sono divisi tra i tre terzili di densità in proporzione ai candidati.
4. **Campioni annidati**: per lo stesso seed, il campione da 100 è contenuto in quello da 250, che è contenuto in quello da 500, e così via. In questo modo la curva di apprendimento (prestazione al crescere degli esempi) cambia solo per gli esempi aggiunti e non per un'estrazione nuova a ogni taglia, ed è più facile da leggere.

Ogni rappresentazione ha i propri campioni, estratti dai propri candidati: in tutto 6 rappresentazioni × 5 taglie × 5 seed = 150 campioni. I controlli confermano che sono tutti candidati etichettati, senza duplicati, esattamente bilanciati e annidati. I cinque seed sono quasi indipendenti: due campioni da 2.000 condividono in media solo il 3,6 % degli esempi.

### I candidati di test fissi (QH5)
Anche sul test il kernel quantistico ha un costo: per ogni candidato da classificare va calcolato il kernel con tutti gli esempi di addestramento. Con circa 110.000 candidati di test per F-X sarebbe troppo, quindi per ogni rappresentazione fissiamo una volta per tutte un sottoinsieme di **2.000 candidati di test**, lo stesso per tutti i metodi del secondo stadio (i diversi circuiti quantistici e i kernel classici di confronto). Così ogni confronto tra metodi avviene esattamente sugli stessi dati.

Il sottoinsieme è una **miniatura fedele dei candidati veri**: mantiene la proporzione naturale tra crateri e falsi allarmi (circa uno su sei) e la stessa ripartizione per dimensione dei crateri e per densità del terreno. In questo modo AP, precisione e F1 (media armonica di precisione e recall: 0 pessimo, 1 perfetto) misurati sui 2.000 candidati stimano direttamente le prestazioni reali, senza correzioni. Per F-X contiene 331 crateri (229 di 1-2 km, 83 di 2-4 km e 19 di 4-8 km) e copre 41 dei 42 blocchi di test; la sua composizione coincide con quella di tutti i candidati entro pochi millesimi. L'unico limite è che per i crateri di 4-8 km le metriche si baseranno su una ventina di esempi, quindi con intervalli di confidenza ampi.

Con questo passo la fase dello stage 1 è completa: per ogni rappresentazione abbiamo la testa di stage 1, la soglia, i candidati, i 150 campioni di addestramento e i 2.000 candidati di test su cui lavorerà lo stadio quantistico.

## 11 - Fase QI - Classificazione dei crateri con kernel quantistico (simulatore)

Il secondo stadio del rivelatore prende i candidati dello stage 1 e decide quali sono davvero crateri, usando un classificatore SVM con un **kernel quantistico**: la somiglianza tra due sotto-tile viene misurata codificando i loro vettori in stati di un circuito quantistico e confrontando gli stati. Per ora la fase è al primo passo, la preparazione degli ingressi.

### Riduzione delle dimensioni e standardizzazione (QI1.1)
Un circuito con *d* qubit può ricevere *d* numeri, uno per qubit, mentre i vettori delle rappresentazioni ne hanno da 384 a 768. Bisogna quindi comprimerli a **4, 8, 12, 16 o 20 componenti**, i numeri di qubit che verranno provati.

**La scelta del metodo.** Abbiamo deciso di usare la **PCA**, la stessa della probe compressa di QG3: si centrano i vettori, si trovano le direzioni lungo cui variano di più e si tengono le prime *d*. Ogni componente viene poi standardizzata (media zero e varianza uno), così che le componenti abbiano tutte la stessa scala prima di diventare angoli di rotazione dei qubit. La PCA non usa le etichette, quindi non rischia di "barare" sapendo dove sono i crateri, ed è la stessa per lo stadio quantistico e per i classificatori classici di confronto. Il suo limite è noto da QG3: le direzioni di massima varianza non sono per forza quelle che distinguono i crateri, e per F-X l'AP lineare scende da 0,57 a 0,44 con 16 componenti.

Abbiamo valutato anche la **Kernel PCA** (una PCA fatta dopo una trasformazione non lineare dei dati) e l'abbiamo scartata per tre motivi. Primo, anche lei non usa le etichette: cerca sempre la massima varianza, solo in uno spazio diverso, quindi non risolve il problema di QG3. Secondo, ha un parametro, la larghezza del kernel, che si può scegliere solo guardando le etichette: se è troppo piccolo la Kernel PCA si riduce alla PCA normale, se è troppo grande le componenti diventano rumore, e cercarlo aggiungerebbe un'altra dimensione alla griglia degli iperparametri. Terzo, applicare un kernel classico prima del circuito confonderebbe la domanda centrale del lavoro, cioè il confronto tra kernel quantistico e kernel classico (RBF, radial basis function, cioè gaussiano): non si saprebbe più a chi attribuire un eventuale vantaggio.

**Su quali dati si stima la PCA.** La PCA viene stimata su **ogni campione di addestramento** del kernel (100-2.000 sotto-tile, QH4) e poi applicata, senza ristimarla, ai 2.000 candidati di test (QH5). Nella cross-validation verrà ristimata dentro ogni fold di addestramento, così che i dati di validazione non influenzino mai la trasformazione. Calcolarla richiede pochi millisecondi, quindi non salviamo le componenti. Salviamo solo, una volta per tutte, i vettori originali dei campioni e dei candidati di test di ogni rappresentazione (circa 9.300 + 2.000 per rappresentazione, 68 MB), per non dover rileggere ogni volta la cache dei token da 14 GB.

**Cosa abbiamo misurato.** Per ogni rappresentazione, taglia del campione e numero di componenti abbiamo misurato tre cose: la varianza spiegata, la stabilità delle componenti tra i 5 seed e la scala delle componenti sul test. Per F-X:

| Campione | Varianza spiegata (d = 4 / 16 / 20) | Stabilità tra seed (d = 16) | Deviazione standard sul test (d = 16) |
|---|---|---|---|
| 100 | 0,31 / 0,66 / 0,72 | 0,58 | 0,66-0,92 |
| 500 | 0,28 / 0,59 / 0,64 | 0,84 | 0,77-0,99 |
| 2.000 | 0,27 / 0,57 / 0,62 | 0,95 | 0,81-1,00 |
| S1-train (QG3) | 0,24 / 0,52 / 0,57 | - | - |

Come si leggono questi numeri:

1. Le componenti spiegano più varianza sui candidati che su tutti i sotto-tile di S1-train. I candidati sono un insieme più omogeneo, e con pochi esempi la PCA sovrastima la varianza catturata.
2. La **stabilità** confronta le direzioni trovate con due seed diversi: vale 1 se sono identiche e circa 0 se non hanno nulla in comune. Con 2.000 esempi le direzioni sono quasi le stesse (0,88-0,97 per tutte le rappresentazioni). Con 100 esempi cambiano molto da un campione all'altro (circa 0,6; 0,4 per F-CM, la meno stabile). Questa è una fonte di variabilità tra i seed da tenere presente quando leggeremo la curva di apprendimento.
3. La **deviazione standard sul test** dice se la scala fissata sul campione di addestramento vale anche per i candidati di test (1 = stessa scala). Da 500 esempi in su le componenti di test hanno una scala entro circa il 20 % di quella di addestramento. Quindi il fattore di scala degli angoli che sceglieremo al passo successivo (QI1.2) avrà sui dati di test lo stesso effetto che ha sui dati di addestramento.

I controlli confermano che le componenti di addestramento hanno esattamente media 0 e varianza 1 e che ogni campione ha rango sufficiente per 20 componenti. Confermano anche che la nostra implementazione coincide con quella di scikit-learn.

### La larghezza di banda c (QI1.2)
Le componenti standardizzate non entrano nel circuito così come sono: ciascuna viene moltiplicata per un fattore **c**, la "larghezza di banda", e il risultato c·x diventa l'angolo di rotazione di un qubit. Proveremo cinque valori, **c = 0,05; 0,1; 0,2; 0,5; 1**, e sceglieremo il migliore in cross-validation.

Perché serve questo fattore? Il kernel quantistico misura quanto si somigliano gli stati che due sotto-tile producono nel circuito.
- Se gli angoli sono **troppo grandi**, due sotto-tile anche poco diversi finiscono in stati quasi ortogonali: tutte le somiglianze vanno a zero, il classificatore vede ogni esempio come unico e impara a memoria senza generalizzare.
- Se gli angoli sono **troppo piccoli**, tutti gli stati si assomigliano e le somiglianze valgono tutte quasi 1, quindi il classificatore fatica a distinguerli.
- Il valore giusto sta in mezzo, e la griglia serve a trovarlo.

C'è anche un secondo effetto, legato al fatto che le rotazioni sono periodiche. In un singolo qubit la somiglianza tra due esempi diminuisce all'aumentare della distanza fra i loro angoli solo fino a un certo punto: mezzo giro (π) per la rotazione R_Y del circuito più semplice (QK-A, codifica ad angoli), un quarto di giro (π/2) per la fase del circuito ZZ (QK-Z: chiamato "ZZ" perché accoppia i qubit vicini con rotazioni generate da Z⊗Z, cioè introduce entanglement). Oltre quel punto la somiglianza torna a crescere, e due esempi molto lontani sembrano di nuovo simili, un fenomeno di "aliasing".

Abbiamo misurato quanto spesso succede sui dati veri: campioni da 500 esempi e 20 componenti, confrontati con i 2.000 candidati di test, per F-X:

| c | Angolo mediano (rad) | Coppie oltre il limite, R_Y (QK-A) | Coppie oltre il limite, fase R_Z (QK-Z) |
|---|---|---|---|
| 0,05 | 0,03 | 0 | 0 |
| 0,1 | 0,06 | 0 | 0 |
| 0,2 | 0,12 | 0 | quasi 0 (4·10⁻⁸) |
| 0,5 | 0,31 | quasi 0 (8·10⁻⁶) | 2,1 % |
| 1 | 0,62 | 2,1 % | 24 % |

Le altre rappresentazioni danno valori praticamente uguali. La griglia copre quindi tutto lo spettro: da angoli piccolissimi, dove il circuito si comporta quasi in modo lineare, fino a c = 1, dove il circuito ZZ confonde un quarto delle coppie di componenti. La teniamo così come era pianificata: gli estremi sono fatti apposta per mostrare i due modi in cui il kernel fallisce, e la parte interessante è dove sta il valore migliore. Quando leggeremo i risultati sulla concentrazione del kernel (QI5.1) terremo presente che a c = 1 il circuito ZZ soffre anche di aliasing.

### Le feature map: la codifica ad angoli QK-A (QI2.1)
Una **feature map** è il circuito quantistico che trasforma un sotto-tile (i suoi *d* angoli θ = c·x) in uno stato di *d* qubit. Il kernel quantistico misura poi la somiglianza tra due sotto-tile come la "fedeltà" dei loro stati: un numero tra 0 (stati ortogonali) e 1 (stati identici).

**Codifica e circuito sono la stessa cosa, in un metodo a kernel.** Il kernel tra due sotto-tile si ottiene così:
1. si prepara lo stato del primo;
2. si applica "all'indietro" la codifica del secondo;
3. si misura quanto lo stato è tornato al punto di partenza, cioè quanto i due stati si somigliano.

Non c'è un'ulteriore parte del circuito che "elabora" i dati: scegliere la codifica significa scegliere il kernel. Le tre feature map provate (QK-A ad angoli, QK-Z "ZZ" con entanglement, QK-P kernel proiettato) sono quindi tre codifiche, e per un kernel è questa la scelta che conta. QK-P cambia in più anche il modo di leggere il circuito: invece di confrontare gli stati interi, misura separatamente ogni qubit. Nei modelli variazionali (VQC/VQR, classificatori e regressori quantistici variazionali, lavori futuri) il circuito è invece "codifica + strati addestrabili", e lì la parte addestrabile è un'ulteriore scelta di progetto.

Abbiamo scritto i circuiti con Qiskit in modo che i loro parametri siano direttamente gli angoli θ. Così lo stesso circuito vale per ogni valore della larghezza di banda c, e lo potremo riusare sia per calcolare gli stati esatti (QI3) sia, più avanti, per simulare il rumore di un dispositivo reale.

La prima feature map, **QK-A**, è la più semplice: ogni qubit viene ruotato attorno all'asse Y di un angolo pari alla sua componente, e i qubit non interagiscono tra loro (nessuna porta a due qubit, profondità 1). Proprio perché i qubit sono indipendenti, il suo kernel ha una formula chiusa: il prodotto, su tutti i qubit, del coseno al quadrato di metà della differenza degli angoli. Si calcola quindi facilmente con un computer classico, e per questo QK-A non è un metodo proposto ma un **riferimento di controllo**: se un circuito con entanglement non fa meglio di QK-A, non sta sfruttando nulla di specificamente quantistico.

Per verificare il circuito abbiamo preso 20 coppie a caso di sotto-tile reali (un campione di addestramento di F-X e i candidati di test). Per ogni numero di qubit e ogni c abbiamo confrontato la formula chiusa con la fedeltà calcolata da Qiskit sugli stati del circuito. Le due coincidono fino all'errore di arrotondamento (differenza massima 6·10⁻¹⁵).

Sulle stesse coppie si vede già, in piccolo, l'effetto della larghezza di banda descritto in QI1.2:
- con 16 qubit il valore mediano del kernel è 0,98 per c = 0,05 (quasi tutti i sotto-tile sembrano uguali), 0,77 per c = 0,2 e appena 0,00016 per c = 1 (quasi tutti sembrano completamente diversi);
- con 4 qubit lo stesso c = 1 dà ancora 0,20.

Più qubit e angoli più grandi spingono quindi il kernel verso zero. Lo studio completo di questa "concentrazione" è un passo successivo (QI5.1).

### La feature map ZZ: QK-Z (QI2.2)
La seconda feature map, **QK-Z**, è quella proposta da Havlíček et al. (2019) ed è il vero candidato quantistico del lavoro. Il circuito ripete *r* volte (r = 1 o 2) uno strato fatto così:
1. una porta di Hadamard su ogni qubit, che mette ogni qubit in sovrapposizione;
2. una rotazione di fase attorno all'asse Z, di angolo 2θ_i, su ogni qubit;
3. una rotazione "ZZ" tra ogni coppia di qubit vicini lungo una catena (0-1, 1-2, ...), di angolo 2θ_i·θ_{i+1}.

Le rotazioni ZZ sono porte a due qubit e creano **entanglement**. Il loro angolo dipende dal prodotto di due componenti, quindi il kernel tiene conto anche delle interazioni tra le variabili, cosa che QK-A non può fare. La catena lineare è stata scelta perché si adatta alla connettività dei processori IBM senza porte aggiuntive. Con r = 1 il circuito ha *d* − 1 porte a due qubit, con r = 2 il doppio: per 16 qubit 15 o 30 porte, con profondità 17 o 21.

Abbiamo implementato anche una variante con la funzione di coppia predefinita di Qiskit, (π − θ_i)(π − θ_{i+1}) al posto di θ_i·θ_{i+1}: è quella usata in buona parte della letteratura e servirà per un solo confronto.

**Verifica.** Abbiamo confrontato i nostri circuiti con quello della libreria Qiskit (`zz_feature_map`). La libreria scrive le stesse rotazioni con porte diverse, che differiscono solo per una fase globale, irrilevante per il kernel. Su coppie reali di F-X, per ogni numero di qubit, ogni *r* e ogni c, i valori del kernel coincidono fino all'errore di arrotondamento (5·10⁻¹⁵).

**Cosa si vede già sui valori del kernel** (mediana su 20 coppie reali di F-X, valori indicativi):

| d | c = 0,05 | c = 0,2 | c = 1 |
|---|---|---|---|
| 4 | 0,99 | 0,79 | 0,10 |
| 16 | 0,94 | 0,32 | 0,00002 |
| 20 | 0,92 | 0,23 | 0,0000004 |

Come per QK-A, più qubit e angoli più grandi spingono il kernel verso zero, e qui più in fretta perché si aggiungono le fasi delle coppie. A c = 1 queste fasi fanno decine di radianti, cioè molti giri: in pratica diventano casuali.

La variante con la funzione di Qiskit si comporta molto peggio sui nostri dati. Anche con angoli piccoli il termine (π − θ_i)(π − θ_{i+1}) vale circa π², cioè una decina di radianti, quindi il kernel è già concentrato: con 16 qubit la mediana è 0,26 a c = 0,05 e 0,002 a c = 0,1, contro 0,94 e 0,76 della nostra versione. Con componenti standardizzate, la scelta predefinita della letteratura finisce quasi sempre nel regime in cui il kernel non distingue più gli esempi. Per questo la usiamo solo come confronto e non come metodo principale.

### Il kernel proiettato: QK-P (QI2.3)
Il terzo kernel, **QK-P** (Huang et al., 2021), nasce per aggirare la concentrazione. Il kernel di fedeltà confronta gli stati *interi* dei *d* qubit. Ogni qubit in più moltiplica le possibilità e rende due stati sempre più diversi, e per questo i valori crollano verso zero all'aumentare di *d*. QK-P confronta invece le proprietà **locali**: per ogni qubit misura i tre valori medi ⟨X⟩, ⟨Y⟩ e ⟨Z⟩, cioè le coordinate dello stato del singolo qubit sulla sfera di Bloch. Ogni sotto-tile diventa così un vettore di 3*d* numeri, e su questi si applica un normale kernel gaussiano, exp(−γ·distanza²).

Lo stato di partenza è quello di QK-Z, con la funzione di coppia prodotto e r = 1 o 2. QK-P aggiunge quindi solo il modo di confrontare gli stati, non un nuovo circuito.

**La scelta di γ.** Il kernel gaussiano ha un parametro, γ, che stabilisce quanto in fretta la somiglianza scende con la distanza. Abbiamo deciso di fissarlo in modo relativo: γ = s / (mediana delle distanze² tra gli esempi di addestramento), con s scelto in cross-validation tra 1/8, 1/4, 1/2, 1, 2, 4 e 8.
- La regola funziona qualunque siano c e *d*. Con angoli piccoli i valori ⟨X⟩, ⟨Y⟩, ⟨Z⟩ cambiano pochissimo da un esempio all'altro, e una griglia assoluta sprecherebbe molti valori.
- Sette valori sono lo stesso budget di ricerca che daremo al kernel RBF classico di confronto, che conviene definire con la stessa regola.

**Verifica.** I valori ⟨X⟩, ⟨Y⟩, ⟨Z⟩ calcolati dal nostro codice coincidono con quelli di Qiskit su stati reali di F-X per ogni *d*, *r* e c (differenza massima 10⁻¹¹). Le lunghezze dei vettori di Bloch non superano mai 1, come deve essere. Nel caso semplice di QK-A riproducono la formula esatta (sin θ, 0, cos θ). Il kernel ottenuto è simmetrico e semidefinito positivo, come richiede l'SVM.

**Cosa si vede** (le stesse 20 coppie reali di F-X di QI2.1-QI2.2, valori indicativi):

| d, c | QK-P: mediana (min-max) | Fedeltà QK-Z: mediana |
|---|---|---|
| 16, c = 0,2 | 0,38 (0,13-0,69) | 0,32 |
| 16, c = 1 | 0,40 (0,19-0,58) | 0,00002 |
| 20, c = 1 | 0,37 (0,19-0,54) | 0,0000004 |

Bisogna leggere questi numeri con attenzione. Una mediana vicina a 0,37 è imposta dalla regola stessa: con γ = 1/mediana, la coppia "mediana" vale e⁻¹ ≈ 0,37, quindi questo non dimostra da solo che il kernel non sia concentrato. Ciò che conta è che i valori di QK-P restano **distribuiti su un intervallo ampio** (circa 0,2-0,5 anche con 20 qubit e c = 1), mentre quelli della fedeltà sono tutti schiacciati vicino a zero. Il motivo è che le distanze tra stati di singoli qubit non si restringono esponenzialmente con il numero di qubit. Se questa varietà di valori contenga davvero l'informazione sui crateri lo diranno le diagnostiche dei kernel (QI5) e la classificazione (QI4).

### Il calcolo esatto dei kernel (QI3.1)
Per addestrare e valutare l'SVM servono matrici di kernel complete: il valore per ogni coppia di esempi di addestramento (fino a 2.000 × 2.000) e per ogni coppia candidato di test-esempio di addestramento (2.000 × 2.000). Simulare un circuito per ogni coppia vorrebbe dire milioni di circuiti. Il piano prevede invece di calcolare **una sola volta** lo stato quantistico di ogni esempio, cioè il vettore dei suoi 2^*d* coefficienti, e di ottenere tutte le somiglianze con un unico prodotto di matrici: K = |Ψ Ψ^†|², dove ogni riga di Ψ è uno stato. Senza rumore il risultato è identico a quello del circuito.

**Quale simulatore.** Gli stati sono calcolati con **Qiskit Aer** (`AerSimulator`), il simulatore standard di Qiskit scritto in C++ e usato in letteratura. Vale per qualunque circuito, ed è lo stesso strumento che useremo per simulare il rumore di un dispositivo reale. Il prodotto di matrici, che è semplice algebra lineare, si fa sulla GPU.

La scelta è venuta da una misura dei tempi su stati reali:

| | Secondi per stato, d = 16 | Secondi per stato, d = 20 |
|---|---|---|
| Qiskit `Statevector` (simulatore di riferimento, in Python) | 0,069 | 1,27 |
| Qiskit Aer | 0,003 | 0,023 |

Per tutta la griglia di esperimenti di QI4 servirebbero circa 70 ore con `Statevector` e circa 1,5 ore con Aer. In una prima versione avevamo calcolato gli stati con un simulatore scritto da noi in PyTorch, ancora più veloce, ma valido solo per la famiglia di circuiti QK-Z e non riconosciuto come strumento standard. Abbiamo preferito Aer, perché i risultati simulati vengono così da uno strumento affidabile e generale. Il simulatore PyTorch resta come controllo indipendente.

**Verifiche.**
- Gli stati di Aer coincidono, a meno di una fase globale irrilevante per il kernel, sia con quelli di `Statevector` sia con quelli del simulatore PyTorch, per ogni *d*, *r* e funzione di coppia.
- I valori ⟨X⟩, ⟨Y⟩, ⟨Z⟩ di QK-P coincidono con quelli calcolati da Qiskit.
- Le matrici di kernel sono identiche a quelle della prima versione.

**La precisione numerica.** I prodotti si fanno in doppia precisione (complex128). In precisione singola l'errore sul kernel arrivava a 2·10⁻⁴ con 20 qubit e comparivano piccoli autovalori negativi nella matrice di addestramento, impossibili per un vero kernel. Sono errori di arrotondamento, ma superano la tolleranza di 10⁻⁶ fissata per la verifica successiva. I 2.000 stati di addestramento a 20 qubit occupano 32 GB, che entrano nella GPU da 95 GB.

**Controlli e tempi** (campione di 2.000 esempi di F-X e 2.000 candidati di test, c = 0,2):
- la diagonale della matrice vale 1 e la matrice è simmetrica, entro 10⁻¹¹;
- non ci sono autovalori negativi oltre lo zero numerico;
- calcolare stati e matrici di addestramento e di test richiede pochi secondi fino a 12 qubit, 8-13 secondi con 16 qubit e **1-2 minuti con 20 qubit**.

Lo stesso calcolo conferma quanto visto in QI2.2 per la variante con la funzione di coppia di Qiskit: con 16-20 qubit la sua matrice è quasi la matrice identità (valore medio fuori diagonale circa 10⁻⁵), cioè il kernel non vede alcuna somiglianza tra esempi diversi.

### La verifica con il circuito "andata e ritorno" (QI3.2)
Su un calcolatore quantistico reale gli stati non si possono leggere: si possono solo misurare i qubit, ottenendo ogni volta una stringa di 0 e 1. Il kernel di fedeltà si stima allora con il circuito di **"andata e ritorno"** (compute-uncompute): si applica il circuito della feature map con gli angoli del primo esempio, poi lo stesso circuito "al contrario" (l'inverso) con gli angoli del secondo. Se i due esempi sono identici, il secondo circuito annulla esattamente il primo e i qubit tornano tutti a 0. Più i due esempi sono diversi, meno spesso si osserva la stringa di tutti zeri. La probabilità di osservarla è esattamente il valore del kernel.

Abbiamo usato questo circuito per verificare in modo indipendente i kernel calcolati in QI3.1. Abbiamo preso 20 coppie casuali di esempi reali di F-X e, per ogni feature map con fedeltà (QK-A, QK-Z e la variante di Qiskit), ogni numero di qubit, ogni *r* e ogni c, abbiamo simulato il circuito con Qiskit Aer in due modi:
- **esatto**: si legge la probabilità degli zeri dallo stato finale. La differenza con il kernel di QI3.1 è al massimo 10⁻¹³, ben sotto la tolleranza di 10⁻⁶ richiesta;
- **con misure ripetute (shot)**, come su un dispositivo vero: 256, 1.024 o 4.096 misure per valore, contando quante volte esce la stringa di tutti zeri. La stima oscilla attorno al valore vero con l'errore statistico atteso, che si riduce con più misure (scarto massimo 0,085 con 256 misure, 0,027 con 4.096). Tutte le 7.500 stime cadono entro quattro deviazioni standard dell'errore binomiale previsto (più una tolleranza di 1/N per i valori quasi nulli).

Questo conferma due cose. Il kernel calcolato in QI3.1 è proprio la quantità che il circuito misurerebbe su un calcolatore reale. E il modello del rumore di misura che useremo al passo successivo (una distribuzione binomiale attorno al valore esatto) descrive correttamente ciò che fa il simulatore.

### Il rumore delle misure (QI3.3)
Un calcolatore quantistico reale non fornisce il valore esatto del kernel: lo stima ripetendo il circuito N volte (gli "shot") e contando quante volte escono tutti zeri, come si stima la probabilità di testa lanciando una moneta N volte. QI3.2 ha mostrato che questo comportamento è descritto esattamente da una distribuzione binomiale. Possiamo quindi simulare il rumore delle misure senza eseguire circuiti: ogni valore del kernel viene sostituito da una sua stima casuale con N = 256, 1.024 o 4.096 misure.

Abbiamo preso con l'utente due decisioni:
- **La diagonale resta 1.** Ogni esempio è identico a sé stesso e su hardware questo valore non si misura. Si stima una sola volta ogni coppia di esempi diversi, come si farebbe su un dispositivo reale.
- **Il rumore si applica anche al kernel proiettato QK-P.** Su hardware QK-P non misura coppie di esempi ma, per ogni esempio, i valori ⟨X⟩, ⟨Y⟩, ⟨Z⟩ di ogni qubit. Il rumore va quindi su questi valori, prima di costruire il kernel. Questa estensione del piano rende equo il confronto tra QK-Z e QK-P in presenza di rumore.

Con il rumore la matrice di addestramento può perdere una proprietà matematica richiesta dall'SVM (essere "semidefinita positiva", cioè non avere autovalori negativi). La correggiamo sostituendola con la matrice valida più vicina: gli autovalori negativi vengono messi a zero.

**Risultati** (campione di 500 esempi di F-X e 2.000 candidati di test):
- **Nel regime normale l'errore è piccolo e diminuisce come previsto.** Con 20 qubit e c = 0,2 l'errore relativo del kernel di test di QK-Z è 9,6 % con 256 misure, 4,8 % con 1.024 e 2,4 % con 4.096: si dimezza ogni volta che le misure quadruplicano, come prevede la statistica.
- **Nel regime "concentrato" il rumore distrugge il kernel.** Con 20 qubit e c = 1 i valori veri del kernel QK-Z sono minuscoli (in media 0,000005). Con poche migliaia di misure quasi nessuna esce "tutti zeri": il 98,5-99,9 % delle stime vale esattamente 0, la matrice diventa la matrice identità e l'errore sul test supera il valore stesso del kernel (76-305 %). È il motivo pratico per cui la concentrazione è un problema serio: anche se il kernel esatto contenesse informazione, un dispositivo reale non riuscirebbe a misurarla senza un numero enorme di ripetizioni.
- **QK-P resiste.** Misura valori locali di singoli qubit, non probabilità minuscole, quindi anche con 20 qubit e c = 1 il suo errore resta tra l'1 e il 4 %. Il suo errore è più alto solo con c = 0,05 (fino al 21-23 % con 256 misure): lì i valori ⟨X⟩, ⟨Y⟩, ⟨Z⟩ cambiano pochissimo da un esempio all'altro, e il rumore pesa di più.
- **La correzione della matrice è contenuta.** Prima della correzione, le matrici rumorose di addestramento hanno autovalori negativi. La parte rimossa è al più il 16 % con 256 misure e il 4 % con 4.096. QK-P non ne ha bisogno, perché il suo kernel gaussiano è valido per costruzione.

Un dettaglio di metodo: l'errore della matrice di addestramento può sembrare piccolo anche quando il kernel è distrutto, perché la diagonale di uno, che non ha rumore, domina la misura. Per questo valutiamo il danno con l'errore sulla matrice di test e con la quota di stime pari a zero.

### Il classificatore: una SVM sul kernel precalcolato (QI4.1)
Il kernel da solo dice soltanto quanto due candidati si somigliano: per decidere se un candidato è un cratere serve un classificatore. Usiamo una **Support Vector Machine (SVM)** di scikit-learn in modalità "kernel precalcolato". Invece di ricevere i vettori, la SVM riceve direttamente la matrice delle somiglianze tra gli esempi di addestramento (K_train, 500 × 500 nel nostro caso). In fase di test riceve la matrice tra i 2.000 candidati di test e quegli stessi esempi (K_test, 2.000 × 500). È questo che permette di usare un kernel quantistico: il computer quantistico (qui simulato) calcola solo le somiglianze, e tutto il resto è una SVM classica. Le due classi hanno lo stesso peso (`class_weight="balanced"`). Sui campioni di addestramento, già bilanciati a metà crateri e metà non-crateri, questo non cambia nulla. Conta invece nella cross-validation dello step successivo, dove i sottoinsiemi possono essere un po' sbilanciati. Un'unica funzione produce le due matrici per ogni feature map, numero di qubit e larghezza di banda, sia esatte sia con il rumore delle misure (QI3.3), così i passi successivi la riusano.

**Il punteggio di un candidato** (scelta presa con l'utente) è una **probabilità calibrata con il metodo di Platt**. La SVM fornisce un "margine", cioè quanto il candidato sta dalla parte dei crateri. Una funzione sigmoide trasforma poi il margine in una probabilità tra 0 e 1. La sigmoide viene adattata sui margini "fuori campione" di una suddivisione in 5 parti del campione di addestramento, per non usare margini che la SVM ha già visto. L'opzione storica di scikit-learn per farlo (`probability=True`) è deprecata nella versione installata e verrà rimossa. Usiamo quindi il sostituto ufficiale, `CalibratedClassifierCV` con un'unica SVM addestrata su tutto il campione. Abbiamo verificato che gestisce correttamente il kernel precalcolato e che è deterministico. Poiché la sigmoide è crescente, l'ordinamento dei candidati, e quindi l'average precision, è identico a quello del margine. La probabilità serve a dare un significato leggibile al punteggio.

**Verifiche.** Su dati sintetici, la SVM con kernel gaussiano precalcolato dà gli stessi margini, previsioni e probabilità della SVM gaussiana "nativa" di scikit-learn (differenze < 1e-8). Sui dati reali, in tutte le 30 configurazioni le matrici hanno la forma giusta, le probabilità stanno tra 0 e 1 e l'AP della probabilità coincide con quella del margine.

**Una prova completa, solo indicativa.** Per controllare che tutta la catena funzioni (dati → PCA → angoli → stati quantistici → kernel → SVM → metriche) abbiamo addestrato la SVM sul campione di 500 esempi della rappresentazione fusa F-X. I parametri sono fissi e non ottimizzati (C = 1, c = 0,2, una ripetizione): per tre feature map, cinque numeri di qubit, kernel esatto e con 1.024 misure, sono 30 configurazioni. I 2.000 candidati di test contengono 331 crateri, quindi un ordinamento casuale darebbe un'AP di circa 0,17. L'AP sul test va da circa 0,32-0,37 con 4 qubit a 0,43-0,46 con 20 qubit, per tutte e tre le feature map: il classificatore impara qualcosa di reale. La SVM non memorizza il campione (accuratezza in addestramento 0,68-0,84). Il rumore di 1.024 misure costa 0,02-0,03 di AP a QK-Z e QK-A con molti qubit e nulla a QK-P, come previsto dallo studio del rumore. Questi numeri non sono ancora un risultato: i parametri verranno scelti con la cross-validation (QI4.2) e i confronti si faranno sulla griglia completa con 5 campioni (QI4.3). Un'ultima osservazione serve per le metriche: la previsione "di default" della SVM (margine positivo) e la regola "probabilità ≥ 0,5" coincidono sul 94-100 % dei candidati, ma non sempre. Quale delle due usare come soglia per l'F1 andrà deciso in QI4.4.

### La scelta degli iperparametri con la cross-validation spaziale (QI4.2)
Ogni kernel ha delle "manopole" da regolare: la larghezza di banda c, il numero di ripetizioni r del circuito (QK-Z e QK-P), la larghezza s del kernel gaussiano di QK-P e il parametro C della SVM (quanto la SVM può sbagliare sugli esempi di addestramento in cambio di un confine più semplice). Le scegliamo **senza guardare il test**, con una cross-validation a 5 parti fatta **dentro** il campione di addestramento. Il campione viene diviso in 5 gruppi (fold). Ogni configurazione viene addestrata su 4 gruppi e valutata sul quinto, a turno su tutti e cinque, e si sceglie la configurazione con la media migliore.

Abbiamo preso con l'utente due decisioni:
- **I fold sono spaziali.** Ogni gruppo è fatto di blocchi interi da 5° (ce ne sono 34-42 per campione) e nessun blocco è diviso tra addestramento e validazione. Crateri vicini si somigliano: se la stessa zona finisse da entrambe le parti, la validazione sarebbe troppo ottimista. I fold sono anche stratificati, cioè ognuno contiene circa metà crateri.
- **Il criterio è l'AP media sui fold**, calcolata sul margine della SVM. La calibrazione di Platt qui non serve, perché non cambia l'ordinamento e quindi l'AP. A parità di punteggio vince il modello più semplice. Salviamo anche l'F1 medio di ogni configurazione, che servirà per lo studio dell'allineamento kernel-etichette (QI5.2).

In ogni fold anche la PCA viene ricalcolata sui soli dati di addestramento di quel fold, così la validazione non riceve alcuna informazione dal gruppo tenuto da parte. Per ogni numero di qubit le configurazioni provate sono 20 per QK-A, 40 per QK-Z e 280 per QK-P. Per non sprecare tempo di simulazione, gli stati di QK-Z di ogni fold vengono calcolati una sola volta e usati sia per il kernel QK-Z sia per i valori ⟨X⟩, ⟨Y⟩, ⟨Z⟩ di QK-P.

**Risultati** (campione di 500 esempi di F-X, primo seed; indicativi, la griglia completa con 5 campioni è QI4.3):
- **I fold sono come previsto:** 94-104 esempi ciascuno, 5-10 blocchi, quota di crateri 48-54 %, nessun blocco condiviso.
- **L'AP di validazione è 0,73-0,75, ma non è confrontabile con quella del test.** I fold sono bilanciati, quindi un ordinamento casuale darebbe 0,5. Nel test i crateri sono invece solo il 16,5 %, e il caso darebbe 0,17.
- **Le tre feature map si equivalgono:** a ogni numero di qubit sono entro 0,01 l'una dall'altra. Le 280 configurazioni di QK-P non gli danno alcun vantaggio sulle 20 di QK-A.
- **La larghezza di banda conferma le analisi precedenti.** Tra c = 0,05 e c = 0,2 il punteggio è praticamente costante: le differenze sono più piccole della variabilità tra fold. Con c = 0,5-1 crolla, soprattutto per le mappe con fasi entangling: con 20 qubit e c = 1, QK-Z scende a 0,55 contro 0,67 di QK-A. È l'effetto della concentrazione e dell'aliasing già visto in QI1.2 e QI3.3. Con molti qubit viene scelto c = 0,05, il bordo inferiore della griglia, ma su un tratto piatto e non in discesa: non indica che servano valori più piccoli.
- **Il modello scelto, riaddestrato su tutto il campione, ottiene sul test un'AP** di 0,36-0,37 con 4 qubit e di 0,44-0,46 da 8 qubit in su. È qualche centesimo sopra la prova con parametri fissi di QI4.1.
- **Il tempo di calcolo è contenuto:** la cross-validation richiede meno di 80 secondi per numero di qubit fino a 16, e 12 minuti a 20 qubit.

### La griglia principale e la curva di apprendimento (QI4.3)
Qui facciamo il primo esperimento vero del classificatore quantistico, sulla rappresentazione fusa F-X. Per ogni feature map (QK-A, QK-Z, QK-P), per ogni numero di qubit (4, 8, 12, 16, 20) e per ognuno dei 5 campioni di addestramento da 500 esempi, ripetiamo la procedura di QI4.2: cross-validation spaziale per scegliere gli iperparametri, poi riaddestramento del modello scelto su tutto il campione e punteggio sui 2.000 candidati di test. Ripetere tutto su 5 campioni diversi ci dice quanto i risultati dipendono dal caso, cioè da quali esempi finiscono nel campione.

Abbiamo preso con l'utente due decisioni:
- **Il numero di qubit "migliore" si sceglie senza guardare il test.** Per ogni feature map prendiamo quello con l'AP media di cross-validation più alta sui 5 campioni; a parità vince il numero più piccolo.
- **Lo studio della larghezza di banda c si fa a 8 e 16 qubit.** Per ogni valore di c prendiamo la configurazione migliore in cross-validation e la valutiamo sul test: è il materiale della figura sulla larghezza di banda del paper.

Poi, con il numero di qubit scelto, costruiamo la **curva di apprendimento**: ripetiamo tutto con campioni da 100, 250, 1.000 e 2.000 esempi, per vedere quanto migliora il classificatore con più dati.

Ogni combinazione viene salvata appena finisce: se il job si interrompe, alla ripartenza riprende da dove era arrivato. Tutto ha richiesto poco meno di 2 ore su una GPU. Le 285 combinazioni sono in una tabella (`qi4_runs.csv`), le tabelle di cross-validation in un'altra (`qi4_cv.csv`), e i punteggi di ogni modello sui candidati di test in un file binario che useremo per le metriche finali (QI4.4) e le analisi successive. Il primo campione riproduce esattamente i numeri di QI4.2, un buon controllo di riproducibilità.

**Risultati** (AP sul test, media ± deviazione standard sui 5 campioni; il caso darebbe 0,17):
- **Più qubit aiutano fino a 16.** Con 500 esempi l'AP sale da 0,36-0,37 con 4 qubit a 0,42-0,43 con 8 e 0,44 con 12. Con 16 qubit si arriva a 0,450 ± 0,016 per QK-A, 0,448 ± 0,019 per QK-Z e 0,441 ± 0,023 per QK-P; con 20 qubit non migliora più (0,44-0,45). Anche la cross-validation indica **16 qubit come scelta migliore per tutte e tre le feature map**.
- **Le tre feature map si equivalgono.** A ogni numero di qubit sono entro 0,01 l'una dall'altra, meno della variabilità tra un campione e l'altro. L'entanglement di QK-Z e il kernel proiettato di QK-P, almeno su questi dati, non danno un vantaggio rispetto alla mappa più semplice QK-A, che non ha entanglement.
- **La larghezza di banda conta molto, e conferma le analisi precedenti.** A 16 qubit, con c tra 0,05 e 0,2 l'AP è stabile (0,44-0,45). Con c = 0,5 QK-A tiene (0,44), mentre QK-P e QK-Z scendono a 0,36 e 0,34. Con c = 1 QK-Z arriva a 0,19, praticamente come il caso. Le mappe con fasi entangling crollano prima: è la concentrazione del kernel prevista in QI1.2 e QI3.3, per cui con angoli grandi tutti i punti diventano "quasi ortogonali" tra loro e il kernel non distingue più nulla.
- **Più dati aiutano, ma poco oltre i 500 esempi.** A 16 qubit l'AP passa da 0,35-0,39 con 100 esempi a 0,42-0,43 con 250, 0,44-0,45 con 500 e 1.000, e 0,46 con 2.000. Il guadagno da 500 a 2.000 esempi è di circa 0,015. Più marcata è la riduzione della variabilità: la deviazione standard tra campioni scende da 0,06-0,09 con 100 esempi a 0,017 con 2.000.
- **Il costo di calcolo** è di meno di un minuto per combinazione fino a 12 qubit, circa 2 minuti a 16 qubit e 8-9 minuti a 20 qubit, dominati dalla simulazione degli stati. Con 2.000 esempi a 16 qubit servono circa 6 minuti.

### Le metriche sul test e la pipeline completa (QI4.4)
L'AP misura quanto bene il classificatore *ordina* i candidati. In pratica, però, a un certo punto bisogna decidere: "questo è un cratere sì o no". Per farlo serve una soglia. Abbiamo deciso con l'utente di considerare cratere ogni candidato con probabilità (di Platt) almeno 0,5. È la regola più semplice e vale identica per tutti i metodi classici che confronteremo in QJ. Con questa soglia misuriamo:
- la **precisione**, cioè quanti dei candidati segnalati sono davvero crateri;
- la **recall**, cioè quanti dei crateri veri vengono segnalati;
- l'**F1**, che combina le due.

Le misuriamo sia in totale sia per fascia di diametro. Calcoliamo tutto per il modello scelto a 16 qubit, per ogni dimensione del campione e ogni seed. Il calcolo usa i punteggi già salvati in QI4.3, quindi richiede pochi secondi.

**La pipeline completa.** Lo stage 2 è stato valutato solo sui 2.000 candidati fissi di QH5, non su tutti i circa 110.000 candidati del test. Quei 2.000 sono però una miniatura proporzionale di tutti i candidati: stesse proporzioni di crateri, fasce e tipi di terreno. Per questo possiamo stimare il risultato della catena completa su tutti i 287.795 sub-tile etichettati del test:
- la recall complessiva è la recall dello stage 1 (0,959) moltiplicata per quella dello stage 2;
- la precisione complessiva è quella dello stage 2.

Abbiamo poi confrontato la pipeline con lo stage 1 da solo in due modi: alla sua soglia di QH2, e con la soglia alzata fino ad avere la stessa recall della pipeline. Il secondo confronto risponde alla domanda vera: lo stage 2 serve, o basterebbe uno stage 1 più severo?

**Risultati** (16 qubit, 500 esempi, media su 5 seed):
- **Sui 2.000 candidati** lo stage 2 ha precisione 0,34-0,35, recall 0,60 e F1 0,43-0,44. Per fascia di diametro la recall è 0,58-0,59 (1-2 km), 0,66 (2-4 km) e 0,52-0,55 (4-8 km); l'ultima fascia conta però solo 19 crateri ed è poco affidabile.
- **Sulla pipeline completa** la recall è 0,57 e la precisione 0,34-0,35. Lo stage 1 segnalava il 38 % dei sub-tile, la catena completa solo l'11 %. Rispetto allo stage 1 alla sua soglia la precisione raddoppia (da 0,166 a 0,35).
- **Ma a parità di recall lo stage 1 da solo fa meglio.** Se alziamo la sua soglia fino ad avere anch'esso recall 0,57, segnala il 6,6 % dei sub-tile con precisione 0,57, contro 0,35 della pipeline. In altre parole, il punteggio dello stage 1 ordina quegli stessi 2.000 candidati meglio dello stage 2: AP 0,60 contro 0,44-0,45. Con 2.000 esempi di addestramento la distanza si riduce di poco.
- **Il motivo è plausibile.** Lo stage 2 vede solo 16 componenti principali del vettore di un singolo sub-tile e impara da al massimo 2.000 esempi. Lo stage 1 legge invece la mappa fusa completa (512 valori per sub-tile, con i sub-tile vicini) ed è addestrato su tutto S1-train. Lo stage 2 filtra quindi i candidati, ma peggio di quanto farebbe una soglia più severa dello stage 1. Questo vale per la configurazione attuale; il confronto con i classificatori classici sugli stessi dati (QJ) dirà se il limite dipende dal kernel quantistico o dall'informazione disponibile allo stage 2.

**Come lo riportiamo.** Dopo questo risultato abbiamo deciso con l'utente di non cambiare il protocollo del paper e di riportare il risultato così com'è (decisione QD11). Nel paper diremo chiaramente che, in questo esperimento, lo stage 2 raddoppia la precisione rispetto allo stage 1 alla sua soglia, ma a parità di recall fa peggio di uno stage 1 più severo. La domanda principale sul quantistico, cioè se un kernel quantistico regge il confronto con i kernel classici sugli stessi dati compressi, non cambia e trova risposta nei confronti di QJ. Le possibili soluzioni, come dare allo stage 2 lo score o il contesto dello stage 1, sono state spostate tra i lavori futuri del piano.

**Perché lo stage 1 è così più forte.** Lo stage 1 non è un SVM: è una piccola rete convoluzionale classica. Il distacco di circa 0,15 di AP tra lo stage 1 da solo e lo stage 2 (la linea tratteggiata della figura di QJ1) viene da quattro differenze insieme:
- **il vettore completo**: lo stage 1 legge 512 valori per sub-tile, lo stage 2 solo 16 componenti principali;
- **il contesto**: lo stage 1 vede anche i sub-tile vicini (3 × 3), utile per crateri che occupano più celle;
- **i dati**: lo stage 1 è addestrato su tutto S1-train, centinaia di migliaia di sub-tile, lo stage 2 su campioni di 100-2.000 esempi, perché il kernel quantistico non si può calcolare su molti di più;
- **il modello**: una rete addestrata sul compito contro un SVM su feature compresse.

QJ1 ha poi mostrato che anche i classificatori classici di stage 2 restano tutti sotto lo stage 1: il limite dipende dall'informazione disponibile allo stage 2, non dal kernel quantistico. **QJ2 ha misurato quanto pesa ciascuna delle quattro differenze** sugli stessi 2.000 candidati (dettagli nel capitolo QJ2):
- **il numero di esempi è la causa principale** (0,07-0,12 di AP): la stessa probe lineare sui vettori completi rende 0,52 se addestrata su tutto S1-train e 0,45 se addestrata sui 500 esempi dello stage 2;
- **modello e contesto dello stage 1** valgono altri 0,03-0,08;
- **la PCA, a parità di dati, costa quasi nulla**: con 500 esempi un SVM lineare rende uguale sulle 16 componenti e sui 512 valori completi (0,450 in entrambi i casi per F-X). Diventa visibile solo con più esempi (circa 0,03 a n = 2.000).

La grande perdita vista in QG3 (da 0,57 a 0,44 comprimendo a 16 componenti) si ha quindi solo quando il classificatore è addestrato su tutto S1-train: con pochi esempi, i 512 valori in più non si riescono a sfruttare.

**La compressione e le alternative alla PCA.** Prima di QJ2 pensavamo che la compressione fosse il collo di bottiglia principale; QJ2 ha mostrato che, con i pochi esempi dello stage 2, lo è solo in piccola parte. Il ragionamento che segue resta comunque valido, perché spiega *quale* alternativa alla PCA aiuterebbe e perché. Il limite della PCA non è che sia lineare, ma che **non sa cosa sia un cratere**: tiene le direzioni in cui i dati variano di più, come illuminazione e tipo di terreno, non quelle utili a riconoscere i crateri. Abbiamo discusso con l'utente due alternative:
- **far uscire direttamente 16 valori dalla fusione** (un collo di bottiglia dentro il modello a cross-attention). Se la fusione resta addestrata senza etichette, comprime in base a ciò che serve a ricostruire le immagini, non a trovare crateri: sarebbe una PCA più sofisticata con lo stesso difetto. In più andrebbe riaddestrata per ogni numero di qubit e andrebbe rifatta tutta la catena successiva;
- **un MLP al posto della PCA**:
  - *senza etichette* (un autoencoder) ha lo stesso limite; per lo stesso motivo era stata scartata la kernel PCA nella decisione QD6;
  - *con le etichette*, cioè una piccola rete addestrata a riconoscere i crateri con uno strato intermedio a 16 neuroni da usare come input quantistico, è la versione che funzionerebbe davvero. Esiste quasi già: è lo strato interno della testa dello stage 1. Va addestrata solo su S1-train, mai sui campioni del quantistico né sul test, per evitare fughe di informazione.

Cosa ci aspettiamo dalla riduzione con etichette:
- lo stage 2 migliorerebbe molto, avvicinandosi allo stage 1. Alla luce di QJ2 il motivo principale non è che comprima meglio, ma che porta dentro lo stage 2 ciò che la rete ha imparato da **tutti** i sub-tile etichettati di S1-train, cioè proprio l'informazione che manca a uno stadio addestrato su 500 esempi;
- superarlo è possibile ma non probabile: se le 16 feature vengono dalla testa dello stage 1, lo stage 2 vede una versione compressa di ciò che lo stage 1 sa già. Per superarlo serve informazione nuova, oppure combinare i due punteggi;
- il distacco tra quantistico e classico non cambierebbe: su feature già ottimizzate per i crateri anche RBF e lineare migliorerebbero allo stesso modo, e il kernel quantistico diventerebbe una "rifinitura" di un lavoro già fatto da una rete classica.

**Decisione (con l'utente).** La riduzione supervisionata resta nei lavori futuri del piano (FW1.3, insieme a FW1.1, punteggio dello stage 1 come input, e FW1.2, contesto dei vicini). Per il paper si tiene la PCA. Rifare la catena QI1-QJ1 non è compatibile con la scadenza, e il messaggio centrale del paper, cioè il confronto equo tra quantistico e classico, non dipende da questa scelta.

### La concentrazione dei kernel (QI5.1)
Un kernel quantistico confronta due esempi tramite la sovrapposizione dei loro stati quantistici: 1 se gli stati sono uguali, 0 se sono completamente diversi. Con molti qubit c'è un pericolo noto, la **concentrazione**. Gli stati di esempi diversi diventano tutti quasi ortogonali e il kernel vale quasi 0 per ogni coppia. Se tutti i valori sono uguali, il kernel non distingue più nulla e la SVM non può imparare. Inoltre, se le differenze tra i valori sono più piccole del rumore di misura, su un computer quantistico reale non si potrebbero nemmeno stimare.

Per misurarla abbiamo preso le matrici del kernel dei 5 campioni di addestramento da 500 esempi (le stesse usate dalla SVM in QI4.3). Per ogni feature map, numero di qubit d e larghezza di banda c, abbiamo calcolato media e varianza dei valori fuori diagonale, cioè delle coppie di esempi diversi. Le abbiamo confrontate con due riferimenti:
- **il "pavimento" degli stati casuali**, cioè il valore medio 1/2^d che avrebbe un kernel completamente concentrato;
- **il rumore di 1.024 misure.** Il rapporto tra la varianza dei valori del kernel e la varianza del rumore di una singola stima ci dice se le differenze tra coppie sono misurabili. Sotto 1 non lo sono.

Per QK-P abbiamo scelto con l'utente di guardare anche i valori ⟨X⟩, ⟨Y⟩, ⟨Z⟩ di ogni qubit da cui il kernel è costruito. Il suo parametro γ si adatta infatti ai dati e nasconderebbe una concentrazione degli stati. Abbiamo messo questo codice in uno script nuovo, `qc_diagnostics.py`, che conterrà tutta la diagnostica dei kernel. Il calcolo ha richiesto 14 minuti su una GPU. I cinque campioni danno risultati quasi identici.

**Risultati:**
- **QK-Z si concentra con c grande e molti qubit.** Con c = 1 il valore medio scende da 0,09 con 4 qubit a 0,00005 con 16 e 0,000005 con 20, a pochi passi dal pavimento casuale. A 20 qubit la varianza dei valori è più piccola del rumore di 1.024 misure: su hardware quel kernel non si potrebbe stimare. È esattamente dove l'AP di QK-Z crolla in QI4.3 (0,19 a 16 qubit con c = 1).
- **QK-A, senza entanglement, si concentra molto più lentamente.** Anche con c = 1 e 20 qubit la sua varianza resta 44 volte sopra il rumore di misura.
- **Con c piccolo succede il contrario:** tutti i valori sono vicini a 1 (0,91-0,99 con c = 0,05). Il kernel è quasi costante, ma ancora misurabile. La SVM lo compensa scegliendo un C grande, come si vede in QI4.2.
- **La varianza da sola non indica la c migliore.** La varianza più alta si ha con c = 0,2-0,5, un po' sopra le c scelte dalla cross-validation (0,05-0,2). Se un'altra misura senza addestramento scelga meglio, lo verifica il passo successivo (QI5.2).
- **QK-P non si concentra fino a 20 qubit.** I valori ⟨X⟩, ⟨Y⟩, ⟨Z⟩ di ogni qubit continuano a variare da un esempio all'altro con ogni numero di qubit. Ogni qubit dipende infatti solo dai suoi vicini nella catena di porte entangling, quindi l'informazione locale non si perde. Il crollo dell'AP di QK-P con c = 1 visto in QI4.3 non è dunque concentrazione: è l'effetto di angoli troppo grandi, che "si avvolgono" su sé stessi (aliasing, già visto in QI1.2).

### L'allineamento kernel-etichette (QI5.2)
Scegliere encoding, numero di qubit e larghezza di banda con la cross-validation è costoso: per ogni configurazione bisogna addestrare e valutare una SVM cinque volte. L'**allineamento kernel-etichette** (KTA) è una scorciatoia proposta in letteratura. Misura quanto la "mappa delle somiglianze" del kernel assomiglia a quella ideale, in cui gli esempi della stessa classe si somigliano e quelli di classi diverse no. Si calcola direttamente dalla matrice del kernel e dalle etichette, senza addestrare nulla. La domanda è se basta a scegliere l'encoding.

Abbiamo calcolato il KTA per tutte le configurazioni già provate in cross-validation (mappa, d, c, r, s) sui 5 campioni di addestramento, e lo abbiamo confrontato con i punteggi di cross-validation salvati in QI4.3. Con l'utente abbiamo deciso di calcolare due versioni:
- quella **classica**;
- quella **centrata**, che non cambia se al kernel si somma una costante, esattamente come la SVM. Questa è la versione usata per scegliere.

I valori di tutte le 2.125 configurazioni sono in `qi5_alignment.csv`. Il calcolo ha richiesto 14 minuti su una GPU.

**Risultati:**
- **Il KTA centrato ordina le configurazioni come la cross-validation.** La correlazione di rango con l'AP di cross-validation è 0,78-0,91 su tutte le configurazioni; per QK-P arriva a 0,83-0,93.
- **Il KTA classico non funziona, e il motivo è interessante.** Quando il kernel si concentra, la matrice diventa quasi un'identità: ogni esempio somiglia solo a sé stesso. Il KTA classico di una matrice identità vale 1/√500 = 0,045, più di quanto valga per molti kernel utili. Così il KTA classico premia proprio i kernel concentrati e per QK-Z va in direzione opposta alla cross-validation.
- **Il KTA centrato raramente sceglie la stessa configurazione della cross-validation**, ma quella che sceglie è quasi altrettanto buona. Si perde al massimo 0,01 di AP di cross-validation, meno della variabilità tra fold. Tende a preferire c = 0,2-0,5, dove i valori del kernel sono più vari (QI5.1), mentre la cross-validation sceglie 0,05-0,2.
- **Sul test la differenza sparisce nel rumore.** Con la c scelta dal KTA l'AP è 0,42-0,44, contro 0,42-0,45 della cross-validation: entro la variabilità tra campioni.

In sintesi, il KTA centrato permette di preselezionare encoding e larghezza di banda senza addestrare nessuna SVM, perdendo meno della variabilità tra campioni. Come riferimento resta comunque la cross-validation.

### La differenza geometrica tra kernel quantistico e classico (QI5.3)
L'ultima diagnostica risponde a una domanda di fondo: un kernel classico ben scelto potrebbe imparare tutto quello che impara il kernel quantistico? Huang et al. (2021) hanno proposto una misura per rispondere, la **differenza geometrica** g. Confronta le due matrici del kernel sugli stessi dati di addestramento:
- **g ≈ 1:** il kernel classico può imparare qualunque cosa impari quello quantistico, con una quantità di dati comparabile;
- **g grande:** esistono etichettature che il kernel quantistico impara meglio, ed è lì che potrebbe esserci un vantaggio. Il valore di riferimento è circa √N, cioè 22 con i nostri 500 esempi.

La formula include un piccolo termine di regolarizzazione λ, perché il kernel classico ha autovalori quasi nulli che altrimenti la renderebbero instabile. Lo abbiamo fatto variare da 0,0001 a 0,1.

Con l'utente abbiamo deciso di calcolare g per i kernel quantistici effettivamente scelti dalla cross-validation in QI4.3 (ogni mappa, numero di qubit e campione), contro due riferimenti classici:
- **il kernel gaussiano (RBF) sugli stessi componenti principali**, con 7 larghezze diverse, tenendo la più favorevole. È la stessa griglia che useremo per l'RBF nei confronti classici di QJ, salvata nella configurazione;
- **la mappa QK-A alla stessa larghezza di banda**, solo per QK-Z e QK-P. QK-A non ha entanglement e si calcola facilmente in modo classico, quindi questo confronto dice se l'entanglement cambia la geometria del kernel.

Il calcolo ha richiesto 4 minuti su una GPU.

**Risultati:**
- **Contro l'RBF migliore, g vale 1,0-2,1 per tutte le mappe, tutti i numeri di qubit e tutti i λ**, lontanissimo da 22, e non cresce con i qubit. Un kernel gaussiano ben tarato sugli stessi dati può quindi riprodurre ciò che imparano i kernel quantistici. Su questi dati non c'è spazio per un vantaggio quantistico, coerentemente con il fatto che le tre mappe ottengono la stessa AP sul test.
- **Il confronto ha senso solo con un RBF ben tarato.** Se la larghezza del gaussiano si allontana dalla migliore, g sale in fretta: a 16 qubit passa da 1,9 a 14. Un confronto con un RBF tarato male farebbe sembrare i kernel quantistici molto più "speciali" di quanto siano.
- **L'entanglement cambia la geometria, ma non abbastanza.** Rispetto a QK-A alla stessa larghezza di banda, g vale 1,7-11 per QK-Z e fino a 56 per QK-P: le porte entangling producono un kernel davvero diverso da quello senza entanglement. Quel kernel diverso, però, un gaussiano riesce ancora a imitarlo.
- **Un dettaglio per i confronti classici.** Per QK-A e QK-Z il gaussiano più favorevole è spesso il più largo della griglia (1/8 della larghezza mediana). Il vero minimo di g potrebbe quindi essere ancora più basso, il che rafforza la conclusione. Per questo, in QJ1, abbiamo controllato se all'RBF servisse una griglia più ampia: non serve, come spiegato in quel capitolo.

### Lo spettro degli autovalori del kernel (QI5.4)
Questo passo era opzionale nel piano. L'utente ha deciso di anticiparlo perché costava pochi minuti e completava le altre diagnostiche.

Gli autovalori della matrice del kernel dicono in quante "direzioni indipendenti" il kernel distingue gli esempi:
- **uno spettro piatto**, con tutti gli autovalori simili, è quello di una matrice quasi identità. Ogni esempio somiglia solo a sé stesso e il modello può solo memorizzare il campione;
- **uno spettro che decade in fretta**, con pochi autovalori grandi, indica un kernel "liscio", che generalizza.

Con l'utente abbiamo scelto due misure semplici:
- il **rapporto di partecipazione**, una stima di quanti autovalori contano davvero: va da 1 (un solo autovalore domina) a 500, il numero di esempi (spettro piatto);
- **quanti autovalori servono per coprire il 90 % e il 99 % del totale**.

Le abbiamo calcolate per tre gruppi di kernel: quelli scelti dalla cross-validation, il gaussiano più simile a ciascuno (da QI5.3) e, a 16 qubit, QK-A e QK-Z alle larghezze di banda estreme (c = 0,05 e c = 1).

Durante l'implementazione è emerso un dettaglio. Nei kernel buoni tutti i valori sono vicini a 1, e il primo autovalore rappresenta solo questa parte costante: il rapporto di partecipazione verrebbe circa 1 per tutti e non direbbe nulla. Abbiamo quindi calcolato le misure anche sul kernel **centrato**, cioè senza la parte costante, che la SVM ignora comunque. È la stessa scelta fatta per l'allineamento in QI5.2.

**Risultati** (kernel centrati, media su 5 campioni):
- **I kernel scelti sono lisci, con una dimensione effettiva vicina al numero di qubit.** Il rapporto di partecipazione è circa 4-6 con 4 qubit, 9-11 con 8, 19-21 con 16 e 24-32 con 20, su 500 possibili.
- **Il gaussiano più vicino ha lo stesso spettro**: per esempio 19,4 contro 19,1 a 16 qubit per QK-Z. È l'equivalente spettrale della differenza geometrica vicina a 1 trovata in QI5.3.
- **Le due larghezze di banda estreme mostrano i due modi di fallire.** Con c = 0,05 il rapporto è circa 16-17 a 16 qubit: il kernel si comporta quasi come un kernel lineare sulle 16 componenti. Con c = 1 sale a 459-499, cioè uno spettro piatto: il kernel è quasi un'identità e può solo memorizzare. È proprio dove QK-Z perde la sua AP sul test.
- **La cross-validation sceglie quindi kernel appena più ricchi di quello lineare**, mai vicini alla memorizzazione.

### Lo stage quantistico su tutte le rappresentazioni (QI6.1)
Fin qui lo stage 2 quantistico era stato provato solo sulla rappresentazione fusa con cross-attention, F-X. Questo passo collega le due domande principali del paper: la fusione migliore resta migliore anche alla fine della pipeline, dopo la compressione a 16 componenti e il classificatore quantistico? Per rispondere abbiamo ripetuto lo stage 2 su tutte le altre rappresentazioni: solo WAC (M-W), solo DEM (M-D), fusione precoce (F-E), concatenazione (F-C) e concatenazione con proiettore addestrato (F-CM).

Ogni rappresentazione usa tutto ciò che è suo: il proprio stage 1, i propri campioni di addestramento per il kernel quantistico e i propri 2.000 candidati di test. Con l'utente abbiamo deciso di:
- usare **tutte e tre le feature map** (QK-A, QK-Z, QK-P) con il numero di qubit migliore trovato per F-X (16), 500 esempi di addestramento e 5 campioni diversi;
- **rifare la cross-validation spaziale** su ogni rappresentazione, invece di copiare gli iperparametri scelti per F-X: ogni rappresentazione ha così la stessa possibilità di essere tarata al meglio;
- riusare per F-X le corse già fatte in QI4.3, senza ripeterle, e calcolare le stesse metriche di QI4.4.

Le 75 nuove corse sono salvate in `qi6_runs.csv`, i punteggi sui candidati di test in `stage2/`. Il calcolo ha richiesto 40 minuti su una GPU. Tutti i controlli sono passati: i candidati e le loro etichette coincidono con quelli dello stage 1, e le AP ricalcolate coincidono con quelle salvate.

**Risultati** (16 qubit, 500 esempi, media su 5 campioni):
- **Le rappresentazioni si dividono in due gruppi.** Le fusioni che combinano i due vettori, cioè la concatenazione F-C e la cross-attention F-X, arrivano ad AP 0,44-0,47 e F1 0,43-0,45; F-CM è un po' sotto (AP 0,39-0,40). Le singole modalità e la fusione precoce restano ad AP 0,36-0,39 e F1 0,37-0,41. Il vantaggio di usare insieme WAC e DEM, già visto con le probe in QG, sopravvive quindi anche alla compressione e allo stadio quantistico.
- **Qui la cross-attention non batte la concatenazione.** F-C è pari o leggermente sopra F-X: con QK-A 0,469 contro 0,450, con le altre due mappe entro 0,01. È diverso da quanto visto con la probe sulle 16 componenti principali (QG3), dove F-X superava F-C di 0,028.
- **Questo confronto va però letto con cautela.** Ogni rappresentazione è valutata sui propri 2.000 candidati, quindi le AP non sono calcolate sugli stessi esempi. Già la scelta del sottoinsieme sposta l'AP di 0,01-0,02: lo stesso stage 1 di F-C ha AP 0,617 sul suo sottoinsieme e quello di F-X 0,603, mentre su tutti i candidati di test l'ordine è inverso (0,598 contro 0,606). Inoltre qui la PCA è calcolata su 500 candidati e non su tutto S1-train, e la differenza tra F-C e F-X è circa pari alla variabilità tra campioni. QJ3 ha poi chiarito che un test appaiato tra rappresentazioni non si può fare, perché i candidati sono diversi; inoltre gli intervalli di confidenza delle singole AP (circa ±0,045) sono molto più larghi di questa differenza.
- **F-X resta sopra il controllo addestrato F-CM** (+0,04-0,06 di AP), come in QG3, dove F-CM era la rappresentazione che perdeva di più con la compressione.
- **Le tre feature map danno risultati equivalenti su ogni rappresentazione**, come già per F-X. Le poche varianze più alte vengono da singoli campioni in cui la cross-validation ha scelto una larghezza di banda grande (c = 0,5-1), dove il kernel funziona peggio.
- **Il limite dello stage 2 vale per tutte le rappresentazioni.** In ogni caso lo stage 1 da solo, con una soglia più severa, è più preciso della pipeline completa a parità di recall: la perdita va da 0,14 a 0,23 di precisione, ed è più grande proprio dove lo stage 1 è più forte (F-X e F-CM). La decisione QD11 di riportare questo risultato così com'è vale quindi per tutta la Tabella I.

In sintesi, alla fine della pipeline la fusione delle due modalità aiuta chiaramente rispetto a una modalità sola o alla fusione precoce, ma in questo esperimento la cross-attention non dà un vantaggio misurabile sulla semplice concatenazione. Lo riporteremo nel paper come risultato.

**L'ipotesi sulla cross-attention era sbagliata?** Non del tutto: è **confermata solo in parte**. Mettendo insieme tutte le misure:

| Dove si misura | Risultato |
|---|---|
| Ricostruzione tra modalità (QF4) | F-X nettamente meglio del controllo F-CM |
| Probe lineare sui vettori completi (QG1) | F-X meglio di F-C, +0,014, significativo |
| Probe dopo la PCA a 16 componenti (QG3) | F-X meglio di F-C, +0,028, significativo |
| Modalità mancante (QG4) | F-X degrada con grazia, F-CM crolla |
| Stage 1 (QH3) | F-X 0,606 contro F-C 0,598 |
| Stage 2, quantistico e classico (QI6.1, QJ1) | F-C pari o leggermente sopra F-X |

La cross-attention dà rappresentazioni un po' migliori e più robuste, ma il vantaggio è piccolo e non sopravvive fino in fondo a questa pipeline. Anche il "F-C meglio" finale è incerto: la differenza è circa una deviazione standard tra campioni, i candidati sono diversi per ogni rappresentazione, e gli intervalli di confidenza di QJ3 (circa ±0,045) la rendono indistinguibile da zero.

**Perché in letteratura la fusione con attenzione vince più nettamente.** Quattro differenze rispetto al nostro caso:
- **addestramento**: di solito la fusione è addestrata end-to-end sul compito stesso con molti dati etichettati, la nostra è leggera e addestrata senza etichette sopra backbone generici congelati. Impara a combinare le modalità, ma non sa cosa sia un cratere;
- **modalità ridondanti**: abbiamo solo due modalità in gran parte sovrapposte (immagine e altimetria raccontano molto la stessa cosa, QD-A5), mentre l'attenzione guadagna di più quando le modalità sono davvero complementari, come nel progetto completo con più modalità;
- **collo di bottiglia**: lo stage 2 vede solo 16 componenti di un singolo sub-tile e impara da pochi esempi, e i vantaggi tipici dell'attenzione (contesto spaziale e scambio tra modalità) vengono in gran parte schiacciati (vedi QI4.4 e QJ2);
- **baseline**: la concatenazione seguita da un classificatore addestrato è una baseline forte, e molti lavori confrontano l'attenzione con baseline più deboli come la fusione precoce. Su quella anche noi vediamo il vantaggio: F-E va nettamente peggio.

Nel paper la formulazione onesta è: la cross-attention migliora le rappresentazioni, soprattutto sotto compressione e con una modalità mancante, ma nel proxy a due modalità il vantaggio sulla concatenazione è piccolo e non si traduce in un guadagno a fine pipeline. La robustezza alla modalità mancante resta un argomento forte e in linea con la letteratura.

## 12 - Fase QJ - Confronti con i metodi classici

### Le baseline classiche (QJ1)
Fin qui abbiamo visto quanto rende lo stage 2 quantistico. Ma "quanto rende" ha senso solo rispetto a qualcosa: la domanda di ricerca RQ2 chiede se, sugli stessi dati compressi, un kernel quantistico regga il confronto con i metodi classici. Per rispondere abbiamo ripetuto **esattamente** le stesse corse del quantistico con cinque classificatori classici:
- **SVM con kernel gaussiano (RBF)**: la stessa SVM del QSVC (quantum support vector classifier, la SVM sul kernel quantistico), con un kernel classico al posto di quello quantistico. È il confronto più diretto;
- **SVM lineare**: separa i dati con un semplice iperpiano;
- **Random Fourier Features + regressione logistica**: un'approssimazione economica del kernel gaussiano;
- **rete neurale con uno strato nascosto (MLP)**;
- **alberi di decisione con boosting (GBT)**, spesso il modello più forte sui dati tabellari.

Con l'utente abbiamo deciso che il confronto deve essere equo in tutto:
- **stessi dati**: gli stessi campioni di addestramento, le stesse componenti principali e gli stessi 2.000 candidati di test di ogni rappresentazione;
- **stesso protocollo**: la stessa cross-validation spaziale, con la PCA ricalcolata in ogni fold, lo stesso criterio di scelta e la stessa ripetizione su 5 campioni;
- **stesse corse**: tutti i numeri di qubit su F-X, la curva di apprendimento e tutte le rappresentazioni, cioè ogni corsa quantistica ha la sua gemella classica;
- **budget di ricerca simile**: 20-40 configurazioni di iperparametri per metodo, come QK-A (20) e QK-Z (40).

Per l'RBF abbiamo discusso la griglia della larghezza della gaussiana. QI5.3 aveva mostrato che il valore migliore cade spesso sul bordo della griglia. Abbiamo tenuto la griglia del piano, che è la stessa usata per QK-P, così che i due abbiano parametri identici, e abbiamo aggiunto un controllo: un RBF con una griglia molto più ampia ("RBF-wide"), solo per misurare se il bordo conti.

Due problemi tecnici sono emersi durante l'esecuzione e sono stati risolti:
- **Il primo job era bloccato.** Le prove dei vari iperparametri girano in parallelo su 16 processi, ma ogni processo usava a sua volta 16 thread: 256 thread su 16 core si ostacolavano a vicenda e il calcolo non avanzava. Ora ogni processo usa un solo thread, e il job ha impiegato meno di due ore.
- **L'MLP "satura".** Per molti candidati la rete dava probabilità praticamente uguali a 1, e salvate su disco diventavano identiche: si perdeva l'ordine tra di loro, che serve per l'AP. Ora per ordinare i candidati usiamo il valore della rete prima della funzione sigmoide, che non satura, e le corse dell'MLP sono state rifatte.

Tutti i controlli sono passati: candidati ed etichette coincidono con lo stage 1 e le AP ricalcolate dai punteggi salvati coincidono con quelle registrate.

**Risultati:**
- **Il kernel quantistico e quello gaussiano sono alla pari.** Sugli stessi campioni l'RBF ottiene la stessa AP delle tre mappe quantistiche, entro la variabilità tra campioni, con ogni numero di qubit, ogni numero di esempi e ogni rappresentazione. Per esempio con 16 qubit su F-X: 0,44-0,45 per i kernel quantistici, 0,44 per l'RBF. È esattamente ciò che aveva previsto la differenza geometrica di QI5.3.
- **L'SVM lineare è pari o leggermente migliore di tutti** (0,45 con 16 qubit, 0,466 con 2.000 esempi, il risultato più alto e più stabile). Sulle poche componenti principali il problema è quasi lineare: nessun kernel, né quantistico né classico, riesce ad aggiungere informazione.
- **La rete neurale e gli alberi vanno peggio** di 0,05-0,10 di AP con 500 esempi: con pochi dati si adattano troppo al campione. Migliorano con più dati, ma restano sotto anche a 2.000 esempi.
- **L'ordine tra le rappresentazioni è lo stesso visto in QI6.1**: concatenazione F-C e cross-attention F-X davanti (F-C 0,46, F-X 0,44 con l'RBF), poi F-CM, poi le singole modalità e la fusione precoce. Quel risultato quindi non dipende dal kernel quantistico, ma da come le rappresentazioni sopravvivono alla compressione.
- **Il limite dello stage 2 vale anche per i metodi classici**: a parità di recall, lo stage 1 da solo resta più preciso con qualunque classificatore.
- **Il bordo della griglia non conta.** L'RBF sceglie il bordo nel 20-60 % dei casi, ma con la griglia ampia l'AP cambia in media di 0,002 al massimo. Non serve quindi allargare né la griglia dell'RBF né quella di QK-P.
- **I tempi sono molto diversi.** L'RBF si addestra e valuta i 2.000 candidati in decimi di secondo, contro i secondi o minuti della simulazione del kernel quantistico. QJ4 ha misurato i costi nelle stesse condizioni: a 16 qubit un singolo riaddestramento del QSVC costa da 0,2 a 13 secondi a seconda della mappa e del simulatore, contro 0,1-0,2 dell'RBF; l'unità intera con la cross-validation costa circa 2 minuti contro mezzo minuto.

**La figura del confronto.** Per il paper e il poster abbiamo riassunto il confronto in una figura a quattro pannelli (`fig_j1_quantum_vs_classical`): l'AP al variare del numero di qubit e del numero di esempi, la differenza di AP tra classico e quantistico misurata sullo stesso campione di addestramento, e l'F1 per rappresentazione. Il pannello delle differenze appaiate è il più eloquente: su 70 campioni per ogni coppia di metodi, la differenza tipica è praticamente zero (mediana entro ±0,001) e il metodo classico vince in circa metà dei casi. Le poche differenze grandi vengono da singoli campioni in cui la cross-validation ha scelto per il kernel quantistico una larghezza di banda sfortunata.

Come leggere la figura:
- **la linea tratteggiata "stage 1 alone"** è l'AP che si ottiene ordinando gli stessi 2.000 candidati con il punteggio dello stage 1, senza stage 2. È più alta di circa 0,15 per i motivi spiegati in QI4.4;
- **la linea tratto-punto** (aggiunta con QJ2) è la probe lineare sui vettori completi addestrata su tutto S1-train (AP 0,52), e **la curva punteggiata del pannello (b)** è l'SVM lineare sui vettori completi degli stessi campioni dello stage 2: insieme mostrano che, a parità di esempi, togliere la PCA non aiuta fino a 500 esempi e aiuta poco a 2.000, mentre avere molti più esempi aiuta molto;
- **l'F1 va da 0 (pessimo) a 1 (perfetto): più a destra è meglio.** Lo "0,5" nell'asse è la soglia di decisione (un candidato è dichiarato cratere se la probabilità stimata è almeno 0,5), non un valore di F1. Per riferimento, dichiarare cratere tutti i candidati darebbe un F1 di circa 0,28;
- **i colori indicano la famiglia, marker e stile di linea il metodo**: blu i tre kernel quantistici, arancio l'RBF, verde acqua l'SVM lineare. Avremmo voluto tre gradazioni di blu e due grigi, come indicava il piano, ma il controllo della palette li boccia: tre blu non si distinguono abbastanza, e i grigi si confondono con le linee di riferimento e sono illeggibili per chi non distingue i colori. Con l'utente abbiamo quindi aggiornato la regola dei colori del piano (sezione 14.1), che vale per tutte le figure future.

In sintesi, su questi dati il kernel quantistico è competitivo ma non migliore dei metodi classici: un kernel gaussiano ben tarato, o anche solo un classificatore lineare, ottiene gli stessi risultati in molto meno tempo. È un risultato atteso dalla letteratura per dati di questo tipo, e lo riporteremo come tale; i test statistici formali arriveranno con QJ3.

**Ci sarebbe almeno un vantaggio di tempo su un computer quantistico reale?** No, anzi: su hardware reale oggi il calcolo sarebbe molto più lento e più rumoroso.
- Nel nostro schema **l'unica parte quantistica è il calcolo del kernel**; l'SVM che classifica è quello classico, e i risultati dipendono solo dalla matrice del kernel e dal parametro C.
- **Su hardware reale ogni valore del kernel è un circuito da ripetere migliaia di volte**, perché il risultato si ottiene come media di tante misure. Con qualche centinaio di esempi di addestramento e 2.000 candidati si arriva a centinaia di milioni di esecuzioni, più le code e il rumore: ore o giorni, contro le frazioni di secondo di un RBF.

La conclusione esatta è: **un kernel classico diverso dà risultati equivalenti**, non che si ottenga la stessa matrice. In più, con 16-20 qubit, anche la matrice quantistica esatta si calcola senza fatica su un computer classico: è ciò che abbiamo fatto in simulazione. Questa conclusione vale per questo problema, a questa scala e con l'hardware di oggi; non è una sentenza sui kernel quantistici in generale. Il costo esatto su hardware sarà stimato in QK3. Nel paper la formulazione corretta è: "su questi dati e a questa scala il kernel quantistico non offre vantaggi né di qualità né di costo".

**E un SVM interamente quantistico?** Esistono versioni quantistiche della risoluzione dell'SVM (Rebentrost et al., 2014), ma richiedono computer quantistici grandi, senza errori e con una memoria quantistica, che oggi non esistono. Soprattutto non cambierebbero la qualità: l'addestramento dell'SVM ha un'unica soluzione, e un risolutore quantistico troverebbe la stessa del classico partendo dalla stessa matrice. Il vantaggio promesso era solo di velocità su dataset enormi, e diversi lavori hanno mostrato che molte di queste accelerazioni si ottengono anche con algoritmi classici.

**Con altri dati o compiti il quantistico potrebbe emergere?** I dati contano moltissimo, ma la chiave non è la non-linearità: anche il kernel gaussiano e l'MLP catturano benissimo relazioni non lineari, e uno spazio enorme non è di per sé un vantaggio (può causare la concentrazione vista in QI5.1). Il vantaggio richiede dati con una struttura che "somiglia" al circuito e che un metodo classico non sa imitare. Gli unici vantaggi dimostrati riguardano dati costruiti apposta o dati provenienti da sistemi quantistici; immagini e altimetria della Luna sono dati classici. Più modalità o compiti più difficili (regressione, segmentazione) rendono il problema più difficile per tutti, non più "quantistico". Lo strumento per verificarlo in anticipo è la **differenza geometrica** di QI5.3: si calcola in pochi minuti su un nuovo dataset, e se resta vicina a 1 il classico basta.

**I modelli variazionali (VQC per classificare, VQR per la regressione).** Sono circuiti con parametri addestrabili, i "pesi quantistici", aggiornati a mini-batch con gradienti come una rete neurale.
- **Non li aggiungiamo al paper (decisione con l'utente).** A parità di codifica un VQC è un modello lineare nello stesso spazio quantistico usato dal kernel (Schuld, 2021): il QSVC trova già la soluzione migliore in quello spazio, mentre il VQC cerca in una famiglia più ristretta e con un addestramento più difficile. Sui nostri dati, quasi lineari e già pareggiati dal classico, darebbe con alta probabilità un risultato pari o peggiore.
- **Nel progetto completo vanno previsti fin da subito**, per la scala, non per la qualità. Il kernel ha un costo che cresce con il quadrato dei dati: per tutta la Luna il piano stima circa 2 miliardi di valutazioni. Un VQC cresce in modo lineare: si addestra a mini-batch su quanti dati si vuole e in inferenza serve un solo circuito per candidato.
- **Due cautele.** I "barren plateaus": con molti qubit i gradienti diventano piccolissimi e l'addestramento si blocca, il corrispettivo variazionale della concentrazione. E i confronti ampi in letteratura (Bowles, Ahmed e Schuld, 2024) trovano che le baseline classiche vincono quasi sempre.
- **L'approccio consigliato per il dottorato**: kernel come riferimento sui sottocampioni, VQC/VQR per la scala, baseline classiche, tutto con lo stesso protocollo, e la differenza geometrica come filtro iniziale.

**Il kernel addestrabile (lavoro futuro FW6).** Oggi la codifica ha un solo parametro libero, la larghezza di banda c, scelto su una griglia di 5 valori. In un kernel addestrabile la codifica ha molti parametri, per esempio una scala per ogni componente e strati di rotazioni. Si ottimizzano sui dati di addestramento, anche a mini-batch, in modo da massimizzare l'allineamento kernel-etichette (QI5.2), e poi si **congelano**: l'SVM e l'inferenza restano identici a ora. Non è un kernel che si adatta durante l'inferenza. È una via di mezzo tra QSVC e VQC, e risponde alla domanda "esiste, in un'intera famiglia di circuiti, uno migliore delle tre mappe fisse e dell'RBF?". Se anche un kernel ottimizzato resta pari al classico, la conclusione "il classico basta" vale per un'intera famiglia di circuiti, non solo per tre. I dettagli per implementarlo sono nel piano, sezione FW6.

**Quanto vale questo lavoro.** Come studio di fattibilità è solido e presentabile, se raccontato per quello che è.
- **Punti di forza**: dati lunari reali e multimodali; confronto davvero equo tra quantistico e classico (stessi dati, stesso protocollo, stessa cura nella taratura, il punto che molti lavori del settore sbagliano); diagnostica che spiega *perché* il quantistico si comporta così.
- **Punti deboli, da dichiarare**: lo stage 2 non migliora lo stage 1 a parità di recall (QD11); la cross-attention non batte la concatenazione a fine pipeline (QI6.1); tutto è simulato e su scala ridotta.

Il risultato "pari" non è un difetto: è ciò che la letteratura seria si aspetta su dati di questo tipo, e dimostrarlo con rigore su un caso reale è un contributo.

### Le linee di riferimento (QJ2)
QI4.4 aveva mostrato che lo stage 1 da solo è molto più bravo dello stage 2 sugli stessi candidati, e che il distacco viene da più differenze insieme (vettore completo o compresso, contesto dei vicini, numero di esempi, modello). Questo passo misura il peso di ciascuna, con due "linee di riferimento" senza compressione accanto a quella dello stage 1, già misurata.

Con l'utente abbiamo deciso di calcolarle entrambe:
- **la probe lineare di QG1 sui candidati**: la regressione logistica addestrata in QG1 su tutti i sub-tile di S1-train con i vettori completi, valutata sugli stessi 2.000 candidati dello stage 2. Non richiede nuovi addestramenti: i suoi punteggi erano già salvati e sono stati abbinati ai candidati tramite la cella. È il tetto "senza compressione e con tutti i dati";
- **"LIN-full"**: lo stesso SVM lineare di QJ1, addestrato sugli **stessi campioni** dello stage 2 (stesse taglie, stessi seed, stessa cross-validation), ma sui vettori completi invece che sulle 16 componenti principali. La differenza con l'SVM lineare di QJ1 misura quindi il costo della sola PCA, a parità di tutto il resto.

Un dettaglio tecnico: per confrontare davvero i due SVM lineari, i vettori completi vanno riscalati. Su 512 valori la stessa griglia del parametro C avrebbe significato una regolarizzazione circa 30 volte più debole che su 16 componenti, cioè modelli quasi non regolarizzati, oltre che lentissimi da addestrare. Abbiamo quindi ridotto i vettori di un fattore costante, in modo che abbiano la stessa "grandezza" media delle 16 componenti: così ogni valore di C ha lo stesso significato nei due casi.

Le 50 corse hanno richiesto meno di 5 minuti di calcolo su CPU, e tutti i controlli sono passati.

**Risultati** (AP sui 2.000 candidati, rappresentazione F-X):

| Riferimento | AP |
|---|---|
| Stage 1 da solo | 0,60 |
| Probe lineare, vettori completi, tutto S1-train | 0,52 |
| SVM lineare, vettori completi, 500 esempi | 0,45 |
| SVM lineare, 16 componenti PCA, 500 esempi (QJ1) | 0,45 |
| Miglior kernel quantistico, 16 qubit, 500 esempi | 0,45 |

- **Il numero di esempi è la causa principale del distacco** (0,07-0,12 di AP a seconda della rappresentazione): la stessa probe lineare rende molto di più se addestrata su tutto S1-train che sui 500 esempi dello stage 2.
- **Modello e contesto dello stage 1** spiegano il resto (0,03-0,08).
- **La PCA, a parità di esempi, costa quasi nulla.** Con 500 esempi l'SVM lineare rende uguale con e senza compressione per F-X, F-C e la fusione precoce. Costa qualcosa solo per WAC e DEM da soli (0,01-0,03) e soprattutto per F-CM (0,04), che era la rappresentazione meno comprimibile (QF4.2). Sulla curva di F-X il vantaggio dei vettori completi compare solo con molti esempi: nullo fino a 500, 0,02 a 1.000 e 0,03 a 2.000.
- **Per il confronto quantistico contro classico non cambia nulla**: QSVC, RBF e SVM lineare stanno insieme, circa 0,07 sotto la probe addestrata su tutti i dati e 0,15 sotto lo stage 1.

Come nota, l'F1 della probe alla soglia 0,5 è più basso di quello degli SVM (recall alta, 0,83-0,87, ma precisione 0,24-0,29): la probe è stata addestrata su tutti i sub-tile, dove i crateri sono molto più rari che tra i candidati, e la sua soglia non è adatta a questo sottoinsieme. Per confrontare i modelli conta l'AP, che non dipende dalla soglia.

**Cosa significa.** Con le taglie di campione che il kernel quantistico permette (al massimo 2.000 esempi), il limite principale dello stage 2 è quanti esempi vede, non la compressione. Questo corregge l'idea iniziale (vedi QI4.4) e chiarisce perché le soluzioni previste per il futuro, cioè dare allo stage 2 il punteggio dello stage 1 o una riduzione addestrata con le etichette, dovrebbero funzionare: portano nello stage 2 l'informazione imparata da tutti i dati etichettati. La figura del confronto (QJ1) è stata aggiornata con le due nuove linee di riferimento.

### Le statistiche del confronto (QJ3)
QJ1 aveva mostrato che kernel quantistici e classici danno AP molto simili, ma "simili" va verificato con un test: bisogna capire se le piccole differenze osservate sono reali o solo rumore. Questo passo non addestra nulla di nuovo: rilegge i punteggi già salvati di tutte le corse (QI4.3, QI6.1, QJ1) e calcola tre cose.

1. **Media e deviazione standard sui 5 seed**, cioè quanto cambia il risultato cambiando il campione di addestramento.
2. **Intervallo di confidenza al 95 % con bootstrap a blocchi.** Si ricampionano con reinserimento i blocchi di test (non i singoli candidati, che sono correlati nello spazio) 1.000 volte e si ricalcola l'AP ogni volta. Tutti i metodi usano gli stessi ricampionamenti, così si può calcolare anche l'intervallo della *differenza* tra due metodi (confronto appaiato).
3. **Test di McNemar** sugli errori alla soglia 0,5: sugli stessi candidati si contano i casi che sbaglia solo il metodo quantistico e quelli che sbaglia solo il classico; se i due conteggi sono molto sbilanciati, uno dei due sbaglia significativamente meno.

Con l'utente abbiamo fissato tre scelte che il piano lasciava aperte:
- **McNemar seme per seme, con verdetto a maggioranza.** Ogni seme è una vera coppia di modelli addestrati sullo stesso campione, quindi si fa un test per seme. Si dice "p < 0,05" se lo è in almeno 3 semi su 5, cioè se la mediana dei 5 p è sotto 0,05. Sommare i conteggi dei 5 semi non sarebbe stato valido, perché i candidati sono gli stessi e i conteggi non sono indipendenti.
- **Confronto con l'RBF, come da piano, e anche con l'SVM lineare**, che in QJ1 era risultato il metodo classico più forte.
- **Correzione di Holm riportata a fianco.** Con 42 confronti per riferimento qualche p < 0,05 esce per caso; la correzione di Holm lo compensa. La regola principale resta quella del piano: una differenza è significativa solo se p < 0,05 **e** nessuno dei due intervalli di confidenza contiene la media dell'altro metodo.

I confronti sono 84: 14 configurazioni (F-X a 4-20 qubit, F-X con 100-2.000 esempi, le altre cinque rappresentazioni) × 3 mappe quantistiche × 2 riferimenti classici. L'intero calcolo richiede 21 secondi. Controlli: le AP ricalcolate coincidono con quelle salvate, i candidati sono gli stessi in tutti i file e ogni metodo ha 5 semi.

**Risultati.**
- **Gli intervalli dei singoli metodi sono larghi**, circa ±0,045 di AP. Per esempio, su F-X con 16 qubit e 500 esempi: QK-A 0,450 [0,404-0,493], RBF 0,439 [0,393-0,481], lineare 0,450 [0,404-0,494]. Il test contiene solo 42 blocchi e le loro differenze (zone più o meno facili) pesano allo stesso modo su tutti i metodi. Per questo la regola del piano, che chiede intervalli separati, è molto severa.
- **Contro l'RBF nessuna differenza è significativa** (0 su 42, anche con Holm). McNemar dà p < 0,05 in 8 casi su 42, ma gli intervalli si sovrappongono sempre. La differenza di AP quantistico meno RBF ha mediana 0,000, con estremi −0,042 e +0,018.
- **Contro l'SVM lineare c'è una sola differenza significativa (1 su 42), ed è una sconfitta del quantistico**: QK-P con soli 100 esempi perde 0,053 di AP.
- **L'intervallo della differenza appaiata è più sensibile** ed esclude lo zero in 14 confronti su 42 con l'RBF (in entrambe le direzioni) e in 20 su 42 con il lineare. Contro il lineare il segno è soprattutto a favore del classico, tranne su WAC e DEM da soli, dove vincono di poco le mappe quantistiche. Sono comunque differenze di 0,01-0,02 di AP, piccole rispetto alle 0,15 che separano lo stage 2 dallo stage 1.

**Perché McNemar e AP a volte non concordano.** Su WAC da solo, per esempio, quantistico e RBF hanno AP entro 0,01, ma l'RBF fa meno errori alla soglia 0,5. McNemar guarda gli errori a una soglia fissa, quindi dipende anche da come è calibrata la probabilità; l'AP guarda l'ordinamento dei candidati. In più McNemar tratta i candidati come indipendenti, mentre quelli vicini nello spazio si somigliano. L'intervallo con bootstrap a blocchi tiene conto di questa correlazione, ed è per questo che il piano chiede entrambe le condizioni.

**Cosa significa.** Per RQ2 la risposta ora è anche statistica: **il kernel quantistico non è mai significativamente migliore di un kernel classico**, e l'unica differenza significativa è una sua sconfitta (QK-P con 100 esempi contro il lineare). Nel paper si può scrivere che QSVC e kernel classici sono statisticamente indistinguibili: 0 confronti significativi su 42 contro l'RBF, 1 su 42 contro il lineare. Resta invece non testata la piccola differenza tra F-C e F-X di QI6.1, perché le due rappresentazioni hanno candidati diversi e non si possono appaiare. Il suo ordine di grandezza (circa 0,01-0,02) è comunque molto più piccolo degli intervalli di ±0,045.

### I costi di calcolo (QJ4)
L'ultimo confronto con i metodi classici riguarda il costo, che si misura in due modi diversi. Il primo è quanto costa davvero il nostro esperimento, dove il circuito quantistico è **simulato** su un computer classico. Il secondo è quanto costerebbe su un **vero processore quantistico** (QPU), dove il costo si conta in circuiti da eseguire e in misure ripetute (shot).

**Come abbiamo misurato i tempi.** I tempi salvati durante le corse erano stati presi in condizioni diverse: i metodi classici in un job su CPU, quelli quantistici in un job GPU e per unità intere con tre mappe insieme. Con l'utente abbiamo quindi deciso un benchmark dedicato:
- **Cosa si misura.** Per ogni metodo si riaddestra la configurazione già scelta dalla cross-validation, per il primo seme, e si danno i punteggi ai 2.000 candidati. Le configurazioni sono F-X da 4 a 20 qubit con 500 esempi, e 16 qubit con 100-2.000 esempi.
- **Ripetizioni.** C'è un giro di riscaldamento non cronometrato, poi fino a tre ripetizioni (una sola se la corsa supera il minuto), e si prende la mediana.
- **Dove.** Tutto sullo stesso nodo. Il QSVC è misurato in tre modi:
  - su **CPU**, alla pari con i classici (scikit-learn non usa la GPU);
  - come lo abbiamo eseguito davvero in QI4: stati simulati da Aer su CPU e prodotti tra stati su GPU;
  - **tutto su GPU** con il nostro simulatore PyTorch. Questa modalità l'ha chiesta l'utente, per usare la GPU ovunque acceleri.

Il job ha richiesto 41 minuti, quasi tutti per la parte quantistica su CPU e per una rete neurale lenta. Le AP ottenute coincidono con quelle già salvate.

**Risultati** (secondi per riaddestrare e dare i punteggi ai 2.000 candidati):

| Metodo | 16 qubit, 500 esempi | 20 qubit, 500 esempi | 16 qubit, 2.000 esempi |
|---|---|---|---|
| RBF / lineare | 0,19 / 0,08 | 0,18 / 0,10 | 0,30 / 0,20 |
| QK-A | 0,23 | 0,27 | 1,45 |
| QK-Z: CPU / pipeline QI4 / tutto GPU | 6,0 / 5,1 / 0,11 | 82 / 68 / 2,3 | 14,6 / 8,0 / 0,28 |
| QK-P: CPU / pipeline QI4 / tutto GPU | 13,0 / 5,1 / 0,22 | 166 / 39 / 2,6 | 20,9 / 8,1 / 0,47 |

- **Quasi tutto il tempo del QSVC va nel calcolo del kernel**, cioè nella simulazione degli stati e nei loro prodotti. L'addestramento dell'SVM e i punteggi costano come per i metodi classici, qualche centesimo di secondo.
- **QK-A costa come l'RBF** perché il suo kernel ha una formula chiusa (un prodotto di coseni): non serve simulare il circuito. È il segno che questa mappa, senza entanglement, è simulabile classicamente in modo esatto.
- **QK-Z e QK-P costano da 25 a 70 volte l'RBF a 16 qubit** con Aer, e da 200 a 900 volte a 20 qubit. Il costo cresce come 2^d: con 4 qubit in più è circa 15-20 volte maggiore. Oltre una trentina di qubit la simulazione esatta diventa impossibile, mentre il costo dei classici non dipende da d.
- **Il simulatore tutto su GPU è 20-50 volte più veloce di Aer.** A 16 qubit porta il QSVC allo stesso ordine di grandezza dei classici, quindi in questo studio la simulazione non è un collo di bottiglia.
- **Tra i classici ci sono due eccezioni:** la rete neurale con 1.000 esempi richiede 327 secondi (la configurazione scelta, con uno strato nascosto largo, converge lentamente, come già visto in QJ1) e gli alberi a 8 componenti 3,9 secondi.
- **Con la cross-validation**, cioè l'unità intera registrata durante le corse, a 16 qubit e 500 esempi servono circa 2 minuti per le tre mappe quantistiche contro mezzo minuto per i sei metodi classici; a 20 qubit 8 minuti contro mezzo minuto. Le condizioni però sono diverse (job GPU contro job CPU).

**Quanto costerebbe su hardware quantistico.** Qui contano i circuiti e gli shot (abbiamo usato 1.024 shot per valore):
- **Kernel di fedeltà (QK-A, QK-Z).** Ogni valore del kernel è un circuito che confronta due esempi. L'addestramento richiede un circuito per ogni coppia di esempi, n(n−1)/2; l'inferenza un circuito per ogni coppia candidato-vettore di supporto, perché solo i vettori di supporto entrano nella decisione.
  - Con 500 esempi: 124.750 circuiti di addestramento e circa 780.000 di inferenza (2.000 candidati × circa 390 vettori di supporto), in tutto circa **un miliardo di shot**.
  - Con 2.000 esempi: circa 4,7 milioni di circuiti e **5 miliardi di shot**.
- **Kernel proiettato (QK-P).** Ogni esempio si misura da solo, in tre basi (X, Y, Z), e il kernel si calcola poi classicamente. Bastano 7.500 circuiti (7,7 milioni di shot) con 500 esempi: **da 120 a 390 volte meno** dei kernel di fedeltà, con la stessa accuratezza (QJ1).

Quanto tempo richiederebbero questi shot su un processore reale, e come cresce il costo sull'intera regione e su tutta la Luna, sarà calcolato in QK3 con i dati pubblicati dei dispositivi IBM.

**Cosa significa.** In simulazione il QSVC non è mai più economico dei metodi classici: nel caso migliore (QK-A, o il simulatore su GPU a 16 qubit) costa uguale, altrimenti da decine a centinaia di volte di più. Su hardware vero il costo si sposterebbe sul numero di circuiti, che per i kernel di fedeltà cresce con il quadrato degli esempi. Per un uso reale il kernel proiettato è quindi la scelta pratica: rende come gli altri e richiede circa cento volte meno circuiti. Per RQ2 il quadro resta lo stesso: il kernel quantistico non è migliore in accuratezza (QJ1, QJ3) e costa di più.

**Più qubit rendono l'hardware più veloce? (discussione con l'utente)** No.
- **In simulazione** ogni qubit in più raddoppia memoria e tempo (2^d).
- **Su hardware reale** il tempo dipende dal numero di circuiti, che non dipende dai qubit, e dagli shot per circuito, il cui tempo è dominato da misura e reset ed è quasi indipendente da d. La profondità del circuito cresce con d (più porte, e sul chip reale anche porte SWAP per avvicinare i qubit), quindi ogni circuito si allunga di poco ma accumula più errori.
- **Effetto indiretto della concentrazione** (QI5.1): con più qubit i valori del kernel si schiacciano verso valori piccoli e simili, e per distinguerli dal rumore servono più shot.

Aumentare i qubit rende quindi l'hardware l'unica strada quando la simulazione diventa impossibile (oltre circa 30 qubit), ma non lo rende più veloce: costo per circuito uguale o maggiore, più rumore e spesso più shot. L'unico guadagno viene da un chip più grande del necessario, su cui si possono eseguire più copie del circuito in parallelo.

Le vie per velocizzare, in teoria, in ordine di importanza:
1. **Meno circuiti per costruzione.** Il kernel proiettato richiede circuiti proporzionali a n invece che a n², circa 100 volte meno. Le misure randomizzate ("classical shadows") fanno qualcosa di simile per il kernel di fedeltà, ma con molti shot quando d cresce.
2. **Meno valori del kernel.** Con l'approssimazione di Nyström si calcola il kernel solo contro alcuni esempi di riferimento; con SVM a pochi vettori di supporto l'inferenza costa meno.
3. **Meno shot.** Si sceglie la larghezza di banda in modo da evitare la concentrazione, e si distribuiscono gli shot dove servono.
4. **Circuiti adattati alla topologia del chip**, senza SWAP: meno tempo e meno rumore.
5. **Parallelismo e infrastruttura:** più copie del circuito sullo stesso chip, più processori, circuiti compilati una volta sola ed eseguiti in sessione.

In simulazione le leve sono quelle già usate (formula chiusa per QK-A, GPU), più i simulatori a tensor network per circuiti con poco entanglement. Nessuna di queste vie cambia la conclusione: anche a costo zero, su questi dati il kernel quantistico rende come un RBF o un lineare (QJ1, QJ3).

### Il contesto della letteratura (QJ5)
Il piano prevedeva di riportare, a titolo di contesto, i numeri pubblicati di DeepMoon e di un detector recente della famiglia YOLO. Discutendone con l'utente è emerso un rischio: quei lavori sono detector completi, addestrati da capo a fondo su aree grandi, mentre il nostro è un banco di prova ridotto (backbone congelati, una sola regione, campioni piccoli). Una tabella affiancata misurerebbe soprattutto la differenza di scala del lavoro, non le nostre domande di ricerca, e potrebbe essere letta come un confronto anche con una nota di avvertenza.

I numeri inoltre non sono confrontabili per costruzione:
- **compito:** quei detector *localizzano* i crateri, cioè ne trovano centro e raggio; noi *classifichiamo* sub-tile di 1,6 km ("contiene o no il centro di un cratere da 1-8 km");
- **metriche:** loro usano precisione e recall degli abbinamenti con il catalogo, noi AP e F1 per sub-tile;
- **dati:** catalogo, regione e risoluzione sono diversi.

Abbiamo valutato tre opzioni:
- **A) tabella con nota:** viene comunque letta come un confronto;
- **B) posizionamento solo qualitativo:** è quella scelta;
- **C) nessun contesto:** un revisore chiederebbe dove si colloca il lavoro.

Il punto QJ5 è stato modificato di conseguenza e "implementato" come testo per il paper, in `results/paper/qj5_related_work.tex`. Si compone di due parti.

1. **Un paragrafo per i lavori correlati.** Cita quattro lavori, con riferimenti verificati sulle pagine degli editori, di arXiv e di DBLP (in `qj5_references.bib`):
   - DeepMoon (Silburt et al. 2019), una rete convoluzionale sulle mappe di elevazione, che ritrova il 92 % dei crateri del proprio catalogo di test. È l'unico numero citato, per descrivere il metodo e non per confrontarlo;
   - YOLO-Crater (Mu et al. 2023), un detector della famiglia YOLO per i crateri piccoli;
   - due lavori che combinano immagini ottiche ed elevazione: Tewari et al. 2020 (Mask R-CNN su ottico, elevazione e pendenza impilati) e Lin et al. 2024 (rete a due rami con fusione ad attenzione, simile in spirito alla nostra F-X). Entrambi riportano un guadagno rispetto a una sola modalità.

   Il paragrafo spiega poi in cosa il nostro compito è diverso, e dice esplicitamente che non competiamo con questi detector. Prendiamo la loro osservazione, che ottico ed elevazione sono complementari (la premessa di RQ1), come ambientazione per un confronto controllato tra kernel quantistici e classici sugli stessi input.
2. **Tre frasi per la discussione** che dichiarano i limiti di scala: una regione equatoriale, backbone congelati, pre-processing semplificato, etichette da un solo catalogo e campioni piccoli nello stadio 2. Concludono che lo studio misura il comportamento *relativo* dei kernel quantistici e classici nelle stesse condizioni, non la qualità assoluta di un detector.

Dichiarare apertamente questi limiti è anche un punto di forza in un workshop di quantum machine learning, dove si giudica soprattutto la correttezza metodologica del confronto.


## 13 - Fase QK - Rumore e risorse

Finora tutti i kernel quantistici sono stati calcolati in modo esatto, cioè con un simulatore ideale, oppure con il solo rumore statistico dei *shot*: un computer quantistico non restituisce un valore, ma un numero finito di misure, da cui il valore va stimato. Un processore reale aggiunge però altri errori: le porte logiche non sono perfette, i qubit perdono la loro informazione nel tempo e anche la lettura finale può sbagliare. Questa fase misura quanto questi errori deformino il kernel e il classificatore, cioè risponde alla domanda di ricerca RQ4.

### Il modello di rumore di un dispositivo reale (QK1)
Non avendo accesso all'hardware, usiamo i *fake backend* di IBM: copie software di processori reali, con la loro topologia e i dati di calibrazione misurati (errori delle porte, tempi di coerenza, errori di lettura, febbraio 2025). Il simulatore Aer costruisce da questi dati un modello di rumore e simula i circuiti come li eseguirebbe il dispositivo.

Le scelte prese con l'utente sono state:
- **uno script nuovo**, `qc_noise.py`, invece di allungare ulteriormente `qc_kernels.py`;
- **due dispositivi** di generazioni diverse: FakeSherbrooke (famiglia Eagle, 127 qubit, porta a due qubit ECR) e FakeFez (famiglia Heron, 156 qubit, porta CZ, più recente);
- **i dati:** come training il campione F-X da 100 elementi già usato nello stadio 2; come test 100 candidati del test fisso, bilanciati (50 crateri e 50 non crateri) perché con soli 100 elementi l'F1 è più stabile se le due classi sono in pari numero;
- **gli iperparametri** (la larghezza c e la regolarizzazione C della SVM) scelti con la stessa cross-validation spaziale di QI4.2, sul kernel esatto, e poi tenuti fissi per tutte le stime, così che le differenze dipendano solo dal rumore.

Il kernel usato è QK-Z con una ripetizione. Ogni suo valore K(x, x') è la probabilità di ritrovare tutti i qubit nello stato 0 dopo aver applicato il circuito di x e poi l'inverso di quello di x' (circuito *compute-uncompute*). Il circuito viene *transpilato*, cioè tradotto nelle porte native del dispositivo e mappato sui suoi qubit fisici, una sola volta per dispositivo e per numero di qubit d, lasciando gli angoli come parametri. Così tutte le coppie usano gli stessi qubit fisici, come su un hardware reale. Per ogni dispositivo e ogni d ∈ {4, 8, 12} si simulano 4.950 circuiti per il training, 10.000 per il test e 100 per la diagonale, ciascuno con 4.096 shot. In totale sono circa 90.000 circuiti, simulati con le matrici densità (il metodo che rappresenta esattamente il rumore). Il calcolo ha richiesto un job CPU di 3 ore e 50 minuti. I valori grezzi sono salvati dopo ogni blocco, così un'interruzione non fa perdere il lavoro già fatto.

Due controlli garantiscono che il confronto sia corretto. Il circuito transpilato, simulato senza rumore, dà lo stesso kernel del calcolo esatto, con una differenza massima di 6·10⁻¹⁵. Un autotest verifica poi l'intera catena su 3 qubit.

**Risultati.** Confrontiamo quattro stime dello stesso kernel: esatta, con soli shot, con il rumore del dispositivo, e con il rumore più la correzione della lettura (QK2). L'errore relativo del kernel di test, misurato con la norma di Frobenius, vale:
- **solo shot:** 0,004 / 0,003 / 0,015 per d = 4 / 8 / 12;
- **rumore del dispositivo:** 0,07 / 0,29 / 0,31 su Sherbrooke e 0,05 / 0,11 / 0,18 su Fez.

Il rumore del dispositivo è quindi da 10 a 100 volte più grande di quello degli shot. Cresce con d, perché il circuito ha più porte a due qubit (10, 26 e 42, uguali sui due dispositivi), e a d ≥ 8 è circa dimezzato sul processore più recente.

Due misure aiutano a capire *come* deforma il kernel.
- **La diagonale misurata.** Il kernel di un punto con sé stesso dovrebbe valere esattamente 1. Misurato, vale 0,93 / 0,71 / 0,67 su Sherbrooke e 0,95 / 0,89 / 0,81 su Fez: è una misura diretta della qualità del circuito.
- **Il confronto con il modello depolarizzante.** Il modello di rumore più semplice, detto depolarizzante globale, prevede che il kernel rumoroso sia una versione "schiacciata" di quello esatto: K_rumoroso = a·K + b, con b = (1 − a)/2^d. La retta stimata ha in effetti una pendenza a tra 0,67 e 0,94. Il termine costante b, però, è da 3 a 150 volte più grande del previsto. Il rumore reale quindi non mescola tutti gli esiti in modo uniforme: spinge in più verso l'esito "tutti zero", come ci si aspetta dal rilassamento dei qubit verso lo stato fondamentale e dagli errori di lettura.

Il piano chiedeva anche di controllare la *concentrazione*, cioè se il rumore avvicini i valori fuori diagonale al minimo 1/2^d, che è il valore di un kernel ormai privo di informazione. La media fuori diagonale scende, ma resta lontanissima da quel minimo: a d = 12 passa da 0,39 (esatto) a 0,27-0,32, mentre il minimo è 0,00024. A queste profondità il rumore comprime il kernel ma non lo appiattisce.

L'aspetto più interessante riguarda il classificatore, che cambia pochissimo. L'F1 passa da 0,70 / 0,68 / 0,73 (esatto) a 0,66-0,67 / 0,70-0,73 / 0,73-0,75 (con rumore). L'AP passa da 0,79 a 0,77-0,78 / 0,74-0,77 / 0,79. Le differenze sono dell'ordine di ±0,05, cioè la risoluzione che 100 campioni di test consentono. Le predizioni coincidono con quelle della SVM esatta per il 71-97 % dei candidati.

Il motivo è che il rumore è in gran parte una trasformazione affine del kernel (a·K + b), e una SVM con termine di bias non ne risente:
- una costante aggiunta a tutto il kernel si cancella nella funzione di decisione;
- un fattore di scala equivale a cambiare C.

Conta quindi solo la parte del rumore che non è affine. Un effetto collaterale: a d ≥ 8 la matrice di training rumorosa è già semidefinita positiva, cioè un kernel valido, senza bisogno di correggerla. La compressione dei valori fuori diagonale agisce come una regolarizzazione. Con i soli shot, invece, compaiono autovalori negativi a ogni d.

La lettura per RQ4: alle profondità di questo studio (al massimo 42 porte a due qubit) a un hardware serve un rumore *stabile* più che *piccolo*, perché il classificatore assorbe la parte sistematica della deformazione.

### La correzione degli errori di lettura (QK2)
Con l'utente si è deciso di eseguire QK2 nello stesso job di QK1, sugli stessi conteggi, senza una seconda simulazione. Il metodo M3 (libreria `mthree`) misura, per i qubit fisici usati, una matrice di calibrazione che descrive con che probabilità uno 0 viene letto come 1 e viceversa (10.000 shot sullo stesso simulatore rumoroso). Poi la "inverte" sui conteggi per stimare la distribuzione che si sarebbe ottenuta con una lettura perfetta. Il valore del kernel, cioè la quasi-probabilità di "tutti zero", viene limitato all'intervallo [0, 1].

L'errore medio di lettura dei qubit usati è 0,7-1,1 % su Sherbrooke e 0,5-0,7 % su Fez. Dopo la correzione l'errore relativo del kernel di test scende:
- su Sherbrooke a 0,045 / 0,23 / 0,22;
- su Fez a 0,026 / 0,072 / 0,116.

La riduzione è tra il 21 e il 45 %, e la diagonale misurata si avvicina a 1 (a d = 12: 0,76 e 0,88). Quindi la lettura spiega da un quarto a metà dell'errore. Il resto viene dalle porte, e per ridurlo servirebbero tecniche di mitigazione degli errori di porta (per esempio l'estrapolazione a rumore zero), fuori dallo scopo del lavoro. F1 e AP invece non cambiano oltre ±0,03: poiché il classificatore era già insensibile alla parte affine del rumore, M3 migliora il kernel ma non la classificazione.

### La tabella delle risorse hardware (QK3)
L'ultima domanda di RQ4 è quanto costerebbe eseguire davvero il classificatore su un computer quantistico. In QJ4 avevamo già contato i circuiti e gli shot necessari. QK3 aggiunge i dati del dispositivo e i tempi, ed estende la stima a tutta la regione R e a tutta la Luna. Il calcolo non simula nulla: transpila i circuiti e legge i dati pubblicati da IBM. Per questo richiede pochi secondi (comando `qc_noise.py resources`).

**Le scelte prese con l'utente.**
- **Tempo QPU come intervallo tra due stime**, entrambe costruite solo con dati pubblicati da IBM e presenti nei fake backend:
  - la stima *fisica* moltiplica il numero di shot per la durata di uno shot. Questa è la durata del circuito, calcolata dalle durate calibrate di porte e lettura, più l'attesa standard tra due shot (250 µs). Conta solo l'esecuzione, senza caricamento e compilazione;
  - la stima *CLOPS* usa il benchmark di velocità che IBM pubblica per ogni processore: 30.000 operazioni su strati di circuito al secondo per Sherbrooke, 195.000 per Fez. Include gli overhead, ma applica ai nostri circuiti un benchmark nato su circuiti diversi.

  Nessuna delle due è un limite rigoroso, ma insieme danno un ordine di grandezza onesto.
- **Estrapolazione dai numeri misurati nel progetto**, non da quelli della pipeline originale. Il numero di sub-tile di R viene moltiplicato per la quota di candidati dello stadio 1 e per il numero medio di support vector delle SVM scelte. Per la Luna si moltiplica poi per il rapporto tra le aree.
- **1.024 shot per circuito**, come in QJ4 e nella Table II. QK1 ha mostrato che già a 4.096 shot l'errore statistico è molto più piccolo di quello del dispositivo, e i tempi crescono comunque in proporzione al numero di shot.
- **Tutte le mappe, ripetizioni e dimensioni** usate nello stadio 2, sui due dispositivi.

Un autotest verifica che i tre circuiti di misura di QK-P restituiscano proprio i valori attesi delle osservabili locali. Un secondo controllo verifica che il riordino del codice di QJ4 lasci invariati i conteggi già salvati.

**I circuiti.** Dopo la transpilazione il numero di porte a due qubit è identico sui due dispositivi, perché la catena lineare dei qubit si adatta al reticolo del chip senza scambi di qubit (SWAP).
- **QK-Z con una ripetizione:** 10, 26, 42, 58 e 74 porte per d = 4, 8, 12, 16, 20. Con due ripetizioni si arriva a 150.
- **QK-P:** circa la metà, perché misura direttamente lo stato e non serve la parte inversa del circuito.
- **QK-A:** nessuna porta a due qubit.

Uno shot dura al massimo una cinquantina di microsecondi su Sherbrooke e una decina su Fez, dove le porte a due qubit sono sei volte più rapide. È comunque molto meno dell'attesa di 250 µs tra uno shot e il successivo. Il tempo fisico è quindi in pratica "numero di shot × 0,25 ms", quasi indipendente dalla mappa e da d.

**I costi.**
- **Una sola SVM** con kernel QK-Z (d = 16, 500 esempi di training): circa 900.000 circuiti e 9·10⁸ shot, cioè 3 giorni di esecuzione fisica e da 3 a 22 giorni secondo la stima CLOPS. Con QK-P bastano 7.500 circuiti, cioè da 20 minuti a 2 ore.
- **Tutta la regione R:** 1,9 milioni di sub-tile, di cui il 35,9 % passa lo stadio 1, cioè circa 680.000 candidati. Moltiplicati per circa 390 support vector danno 2,6·10⁸ circuiti, che richiederebbero da 2 a 17 anni di macchina.
- **Tutta la Luna:** 2,1·10⁹ circuiti, un valore coerente con i circa 2·10⁹ stimati nella pipeline originale, e da 18 a 130 anni.
- **QK-P su R e sulla Luna:** il kernel proiettato richiede 3 circuiti per candidato invece di uno per ogni support vector, cioè circa 130 volte meno. R richiederebbe da 4 a 24 giorni, la Luna da 1 a 6 mesi.

**La lettura per RQ4.** I circuiti di questo studio sono corti per l'hardware attuale. Il costo dipende quindi dal numero di shot e non dalla complessità del circuito. Un classificatore con kernel di fedeltà resta praticabile solo come piccola validazione: un kernel 100 × 100 richiede circa 10⁷ shot, cioè circa un'ora. Una mappatura completa non lo è, come già ipotizzato nella pipeline. Il kernel proiettato, che in QJ1 ha la stessa accuratezza, è la scelta pratica per un'esecuzione su hardware. Il processore più recente aiuta soprattutto con la velocità di esecuzione (CLOPS 6,5 volte più alto) e con il rumore più basso visto in QK1, non con la lunghezza dei circuiti.
