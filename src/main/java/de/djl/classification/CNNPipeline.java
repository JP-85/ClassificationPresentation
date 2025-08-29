package de.djl.classification;

import ai.djl.ndarray.NDManager;

public class CNNPipeline {

    private final NDManager manager;
    private String rawDataPath;
    private String datasetName;
    private int width;
    private int height;
    private boolean normalize;
    private boolean grayscale;
    private boolean createNewDataset;
    private DataSetBundle dataSet;

    // Privater Konstruktor, um die Builder-Methode zu erzwingen
    private CNNPipeline() {
        this.manager = NDManager.newBaseManager();
        this.createNewDataset = false;
    }

    /**
     * Erstellt eine neue Instanz des Pipeline-Builders.
     * @return Eine neue Pipeline-Instanz.
     */
    public static CNNPipeline builder() {
        return new CNNPipeline();
    }

    /**
     * Fügt die Preprocessing-Phase zur Pipeline hinzu, um einen Datensatz zu erstellen.
     * @param rawDataPath Der Pfad zu den Rohbildern (z.B. "PetImages").
     * @param datasetName Der Name des zu erstellenden Datensatzes.
     * @param width Die Breite der Bilder.
     * @param height Die Höhe der Bilder.
     * @param normalize Sollen die Pixelwerte normalisiert werden?
     * @param grayscale Sollen die Bilder in Graustufen konvertiert werden?
     * @return Die aktuelle Pipeline-Instanz.
     */
    public CNNPipeline addPreprocessing(String rawDataPath, String datasetName, int width, int height, boolean normalize, boolean grayscale) {
        this.rawDataPath = rawDataPath;
        this.datasetName = datasetName;
        this.width = width;
        this.height = height;
        this.normalize = normalize;
        this.grayscale = grayscale;
        this.createNewDataset = true;
        return this;
    }

    /**
     * Fügt die Preprocessing-Phase zur Pipeline hinzu, um einen bestehenden Datensatz zu laden.
     * @param datasetName Der Name des zu ladenden Datensatzes.
     * @return Die aktuelle Pipeline-Instanz.
     */
    public CNNPipeline addPreprocessing(String datasetName) {
        this.datasetName = datasetName;
        this.createNewDataset = false;
        return this;
    }

    /**
     * Fügt die Neural Network-Phase zur Pipeline hinzu.
     * (Placeholder - leere Methode)
     * @return Die aktuelle Pipeline-Instanz.
     */
    public CNNPipeline addNeuralNetwork() {
        // Logik für den Aufbau und das Training des CNNs
        return this;
    }

    /**
     * Fügt die Evaluations-Phase zur Pipeline hinzu.
     * (Placeholder - leere Methode)
     * @return Die aktuelle Pipeline-Instanz.
     */
    public CNNPipeline addEval() {
        // Logik für die Evaluierung des Modells
        return this;
    }

    /**
     * Führt die gesamte Pipeline aus.
     */
    public void run() {
        System.out.println("Starting CNNPipeline...");

        DataSetBundle dataSet;

        try {
            if (createNewDataset) {
                System.out.println("Creating new dataset: " + datasetName);
                // Erzeuge Preprocessing-Instanz nur, wenn Daten verarbeitet werden
                Preprocessing prep = new Preprocessing(rawDataPath);
                dataSet = prep.createDataset(datasetName, width, height, normalize, grayscale);
            } else {
                System.out.println("Loading existing dataset: " + datasetName);
                // Für das Laden ist kein rootPath nötig
                dataSet = Preprocessing.loadDataset(datasetName);
            }
        } catch (Exception e) {
            System.err.println("Error during preprocessing step: " + e.getMessage());
            return;
        }

        if (dataSet != null) {
            this.dataSet = dataSet;
        }

        System.out.println("Pipeline finished.");
    }

    public void close() {
        manager.close();
    }

    public  DataSetBundle getDataSet() { return dataSet; }
}