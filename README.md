# ClassificationPresentation (Java + DJL)

End-to-end Bildklassifikation (**Cat vs. Dog**) mit der **Deep Java Library (DJL)** auf der PyTorch-Engine:
**Preprocessing → Dataset → CNN (AlexNet‑artig) → Training/Evaluation → Visualisierung**.
Zusätzlich: **Model‑Zoo‑Demo** & **Offline‑Export** von Layer‑Aktivierungen.

---

## Ordnerstruktur

```
C:.
├───.idea
│   └───inspectionProfiles
├───.mvn
├───data
│   ├───datasets
│   └───raw
│       └───PetImages
│           ├───Cat
│           └───Dog
├───output
│   ├───activations
│   │   └───export
│   ├───metrics
│   │   └───baseline
│   └───models
├───src
│   ├───main
│   │   ├───java
│   │   │   └───de
│   │   │       └───djl
│   │   │           └───classification
│   │   └───resources
│   └───test
│       └───java
└───target
    ├───classes
    │   └───de
    │       └───djl
    │           └───classification
    ├───generated-sources
    │   └───annotations
    ├───maven-archiver
    └───maven-status
        └───maven-compiler-plugin
            ├───compile
            │   └───default-compile
            └───testCompile
                └───default-testCompile
```

> **Hinweis:** `data/raw/PetImages/Cat|Dog` ist der Eingang; `data/datasets/<run>/train|val` wird automatisch erzeugt.

---

## Features

- **Robustes Preprocessing**
    - Automatische Klassenerkennung aus Ordnern
    - Deterministischer Train/Val‑Split (Seed)
    - Resize/Letterbox auf Quadrat, **RGB erzwingen (3 Kanäle)**
    - Skip‑Log defekter Dateien (`skipped_images.txt`) & `metadata.json`
- **Konfigurierbares CNN** (AlexNet‑artig) via `settings.json`
    - Conv/Pool/Dense/Dropout/Activation, SAME‑Padding, optional GlobalAvgPool
    - **Taps** für Aktivierungen: `convX_pre`, `convX_pool`, `fcY`, `logits`
- **Training/Evaluation**
    - Stabiler Loop (Labels → 1D `int64`, Loss → Skalar)
    - **Progressbar** (Loss/Acc live), **Loss/Accuracy‑Plots**, **Confusion‑Matrix (2×2)**
- **Visualisierung**
    - Aktivierungen als **Grid** (Convs/Pooling) oder **Stripe** (Dense/Logits)
    - **ExportActivations**: PNGs offline erzeugen (für Präsentation)
- **Model‑Zoo**
    - ResNet18/34/50 via DJL Criteria (PyTorch) – One‑liner‑Inference

---

## Voraussetzungen

- Java **17+** (Projekt kompiliert auch mit höherem Target)
- Maven **3.8+**
- OS: Windows/macOS/Linux
- DJL **0.33.0** (PyTorch‑Engine; CPU‑Natives bereits im `pom.xml`)

> **PowerShell‑Hinweis:** Bei direkten `java`‑Aufrufen `--%` nutzen, damit `--args` nicht geparst werden (siehe Troubleshooting).

---

## Quickstart

### 1) Daten einlegen
```
data/raw/PetImages/Cat/*.jpg|png|...
data/raw/PetImages/Dog/*.jpg|png|...
```

### 2) Build
```bash
mvn -q -DskipTests clean package
```

### 3) Training starten
**IntelliJ:** `Main` ausführen  
**oder CLI:**
```bash
mvn -q -DskipTests exec:java -Dexec.mainClass=de.djl.classification.Main
```

**Ergebnis:**
- Preprocessing erzeugt `data/datasets/<zeitstempel>/train|val`
- Training mit Progressbar
- Plots unter `output/metrics/<setting>/`
- Confusion‑Matrix (`confusion.png`) bei 2 Klassen
- Optional Aktivierungen unter `output/activations/...` (wenn aktiviert)

---

## Konfiguration

### `settings.json` (Netzarchitektur – Beispiel)
```json
[
  {
    "name": "baseline",
    "stride": 1,
    "kernel": [3, 3],
    "maxPoolSize": [2, 2],
    "optimizer": "adam",
    "learningRate": 0.001,
    "convLayers": 2,
    "denseUnits": [128],
    "activation": "relu",
    "batchSize": 32,
    "dropout": 0.3,
    "baseChannels": 32,
    "maxChannels": 256,
    "globalAvgPool": true
  }
]
```

### `runconfig.json` (Pipeline – Beispiel)
```json
{
  "setting": "baseline",
  "settingsJson": "settings.json",

  "raw": "data/raw/PetImages",
  "datasetsRoot": "data/datasets",
  "valSplit": 0.2,
  "seed": 42,

  "epochs": 2,
  "imageSize": 224,
  "grayscale": false,
  "shuffleTrain": true,

  "saveActivations": true,
  "vizLayer": "conv1",          
  "vizTileSize": 96,

  "zoo": false,                
  "zooBackbone": "resnet18"    
}
```

### CLI‑Overrides (Beispiele)
```bash
# andere Settings/Hyperparameter
mvn -q -DskipTests exec:java -Dexec.mainClass=de.djl.classification.Main \
  -Dexec.args="--setting baseline --epochs 1 --img 224 --save-activations true --vizLayer conv1"

# Pfade anpassen
mvn -q -DskipTests exec:java -Dexec.mainClass=de.djl.classification.Main \
  -Dexec.args="--raw data/raw/PetImages --datasets-root data/datasets"

# Zoo-Demo
mvn -q -DskipTests exec:java -Dexec.mainClass=de.djl.classification.Main \
  -Dexec.args="--zoo true --zooBackbone resnet50"
```

---

## Model‑Zoo Demo (ResNet)

Schnell ein vortrainiertes ResNet testen:
```bash
mvn -q -DskipTests exec:java \
  -Dexec.mainClass=de.djl.classification.Main \
  -Dexec.args="--zoo true --zooBackbone resnet18"
```
Ergebnisse (Top‑1) liegen unter `output/zoo/resnet18/`.  
Intern wird per DJL‑Criteria gefiltert (`optFilter("layers","18")`).

---

## Aktivierungen offline exportieren (zum Betrachten ber Bilder)

PNG‑Export (je **ein Bild pro Klasse**, mehrere Layer auf einmal):
```bash
# Bash / CMD
mvn -q -DskipTests exec:java -Dexec.mainClass=de.djl.classification.ExportActivations \
  -Dexec.args="--config runconfig.json --layers conv1_pre,conv1_pool,conv2_pre,conv2_pool,fc1,logits --classes Cat,Dog --tile 96"
```

**PowerShell (sicher, einzeilig):**
```powershell
mvn -q -DskipTests exec:java -Dexec.mainClass=de.djl.classification.ExportActivations -Dexec.args="--config runconfig.json --layers conv1_pre,conv1_pool,conv2_pre,conv2_pool,fc1,logits --classes Cat,Dog --tile 96"
```

Output: `output/activations/export/<Klasse>_<Layer>.png`

> Falls ein Layer nicht existiert: verfügbare Tap‑Namen prüfen (oder Taps in `ClassificationModel` aktivieren).

---

## Ergebnisse / Outputs

- **Plots:**  
  `output/metrics/<setting>/training_loss.png`  
  `output/metrics/<setting>/training_accuracy.png`
- **Confusion (2×2):**  
  `output/metrics/<setting>/confusion.png`
- **Modelle:**  
  `output/models/<setting>-<timestamp>/cnn` (+ `synset.txt`)
- **Aktivierungen:**  
  `output/activations/<setting>/<layer>.png` (Training)  
  `output/activations/export/<class>_<layer>.png` (Exporter)

---

## Troubleshooting

- **„invalid index of a 0‑dim tensor“ beim Loss**  
  Labels müssen **1D `INT64` (N)** sein. Im Code werden sie per `squeeze/reshape/toType` normalisiert.

- **„Output size is too small“ bei Pooling**  
  Quadratische Größe (z. B. 224), SAME‑Padding aktiv, `stride`/`poolSize` prüfen.

- **PowerShell parst `--`**  
  Bei direktem `java`‑Aufruf:
  ```powershell
  mvn -q -DskipTests package
  java -cp target\ClassificationPresentation-1.0-SNAPSHOT.jar `
    de.djl.classification.Main `
    --% --zoo true --zooBackbone resnet18
  ```

- **Aktivierungen fehlen („No activation captured …“)**
  `"saveActivations": true` setzen und exakte Tap‑Namen verwenden: `convX_pre`, `convX_pool`, `fcY`, `logits`.

- **Warnung „Restricted methods … System::load“**  
  Unkritisch für CPU‑Natives; optional JVM‑Flag `--enable-native-access=ALL-UNNAMED`.

---

## Lizenz / Danksagung

- Powered by **DJL** (Apache 2.0) und **XChart** für Plots.
- Beispielbilder: https://www.kaggle.com/competitions/dogs-vs-cats/data

