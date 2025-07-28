# ClassificationPresentation

## Inhaltsverzeichnis

1. Projektbeschreibung
2. Ordnerstruktur
3. GitHub & Einrichtung
4. Datensatz erstellen (Preprocessing)
5. Modell trainieren (Platzhalter)
6. Modell auswerten (Platzhalter)
7. Modell grafisch darstellen (Platzhalter)

---

## 1. Projektbeschreibung

**ClassificationPresentation** ist ein Java-basiertes Projekt zur Vorbereitung und Verarbeitung von Bilddaten für ein Convolutional Neural Network (CNN) mit der [Deep Java Library (DJL)](https://djl.ai/).  
Im Fokus stehen:

- Laden und Labeln von Bildern aus Verzeichnissen
- Bildverarbeitung (z. B. Skalierung, Konvertierung in RGB oder Graustufen)
- Serialisierung als Datensatz zur weiteren Verwendung
- Gleichverteilte Aufteilung der Daten für Training und Validierung

Das Projekt ist modular aufgebaut und erweiterbar für Training, Auswertung und Visualisierung.

---

## 2. Ordnerstruktur

```plaintext
ClassificationPresentation
├───data
│   └───PetImages
│       ├───Cat          // Bilder von Katzen
│       └───Dog          // Bilder von Hunden
├───output              // Serialisierte Datensätze nach Preprocessing
└───src
    └───main
        └───java
            └───de
                └───djl
                    └───classification
                        ├───ClassificationModel.java
                        ├───CNNDataset.java
                        ├───GrayscalePreprocessing.java
                        ├───LabeledImageData.java
                        ├───Main.java
                        ├───Preprocessing.java
                        └───RGBPreprocessing.java
````

---

## 3. GitHub & Einrichtung

### Repository

GitHub: [https://github.com/JP-85/ClassificationPresentation](https://github.com/JP-85/ClassificationPresentation)

### Projekt klonen

```bash
git clone https://github.com/JP-85/ClassificationPresentation.git
cd ClassificationPresentation
```

### Maven-Projekt bauen

Stelle sicher, dass Maven installiert ist:

```bash
mvn clean install
```

### Projekt ausführen

Beispiel über die `Main`-Klasse:

```bash
mvn exec:java -Dexec.mainClass="de.djl.classification.Main"
```

Alternativ über eine IDE wie IntelliJ oder Eclipse starten.

---

## 4. Datensatz erstellen (Preprocessing)

Um aus vorhandenen Bildern einen verwendbaren Datensatz zu erzeugen, wird die `Preprocessing`-Klasse genutzt.

### Beispiel: `Main.java`

```java
public class Main {
    public static void main(String[] args) throws IOException {
        Preprocessing prep = new GrayscalePreprocessing(50, 50);
        CNNDataset catdog = prep.run("CatDogData", "data/PetImages");
        catdog.writeData("output/catdogDataset");
    }
}
```

### Voraussetzungen

* Die Bilddaten liegen unter `data/PetImages/Cat` und `.../Dog`
* Der `output`-Ordner wird automatisch erstellt, falls er nicht existiert
* Fehlerhafte Bilder werden übersprungen und im Fehlerstream geloggt

---

## 5. Modell trainieren (Platzhalter)

> Hier wird später erklärt, wie ein DJL-Modell auf Basis des erzeugten Datensatzes trainiert wird.

---

## 6. Modell auswerten (Platzhalter)

> Beschreibung zur Evaluation der Modellgüte mit Metriken wie Accuracy, Precision, Recall etc.

---

## 7. Modell grafisch darstellen (Platzhalter)

> Optional: Visualisierung der Trainingskurven, Konfusionsmatrix o. ä.

```

Wenn du willst, kann ich die Abschnitte zu Training/Evaluation später für dich ausformulieren oder eine Konfigurationsdatei hinzufügen.
```
