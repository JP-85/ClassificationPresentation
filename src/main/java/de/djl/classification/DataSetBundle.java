package de.djl.classification;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.io.*;
import java.nio.file.Paths;
import java.util.*;

public class DataSetBundle implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final List<float[][][]> rawData;
    private final List<Integer> labels;
    private final Map<String, Object> metadata;

    public DataSetBundle(String datasetName, List<float[][][]> rawData, List<Integer> labels, Map<String, Object> metadata) {
        this.rawData = rawData;
        this.labels = labels;
        this.metadata = metadata;
    }

    /** Gibt die Anzahl der Samples im Dataset zurück */
    public int size() {
        return rawData.size();
    }

    /** Stellt ein einzelnes NDArray aus dem serialisierten Format wieder her */
    private NDArray restoreSingleFromRaw(NDManager manager, float[][][] img) {
        int channels = (int) metadata.get("channels");
        int height = (int) metadata.get("height");
        int width = (int) metadata.get("width");

        float[] flat = new float[channels * height * width];
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    flat[idx++] = img[c][h][w];
                }
            }
        }
        return manager.create(flat, new Shape(channels, height, width));
    }

    /** Gibt ein zufälliges Sample (NDArray + Label) zurück */
    public Sample getRandomSample(long seed, NDManager manager) {
        Random rnd = new Random(seed);
        int idx = rnd.nextInt(rawData.size());
        return new Sample(restoreSingleFromRaw(manager, rawData.get(idx)), labels.get(idx));
    }

    /** Speichert die DatasetBundle-Instanz in einer Datei */
    public void save(String datasetName) throws IOException {
        File outFile = Paths.get("data", "datasets", datasetName + ".ser").toFile();
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(outFile))) {
            oos.writeObject(this);
        }
    }

    /** Lädt eine DatasetBundle-Instanz aus einer Datei */
    public static DataSetBundle load(File file) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            return (DataSetBundle) ois.readObject();
        }
    }

    // ------------------------------
        // Hilfsklasse für Rückgabe
        // ------------------------------
        public record Sample(NDArray array, int label) {
    }

    // ------------------------------
    // Getter
    // ------------------------------
    public List<Integer> getLabels() { return labels; }
    public Map<String, Object> getMetadata() { return metadata; }

    /** Gibt eine lesbare String-Repräsentation des Datensatzes zurück */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("------------------------------------------\n");
        sb.append("Dataset: ").append(metadata.get("name")).append("\n");
        sb.append("Anzahl Bilder: ").append(rawData.size()).append("\n");
        sb.append("Dimensionen: ").append(metadata.get("width")).append("x").append(metadata.get("height")).append("x").append(metadata.get("channels")).append("\n");
        sb.append("Graustufen: ").append(metadata.get("grayscale")).append("\n");
        sb.append("Normalisiert: ").append(metadata.get("normalize")).append("\n");

        sb.append("\nLabels:\n");
        Map<Integer, Long> labelCounts = new HashMap<>();
        for (Integer label : labels) {
            labelCounts.merge(label, 1L, Long::sum);
        }

        // Verwenden der neuen Metadaten-Struktur
        List<String> categories = (List<String>) metadata.get("categories");

        for (Map.Entry<Integer, Long> entry : labelCounts.entrySet()) {
            String categoryName = (entry.getKey() < categories.size()) ? categories.get(entry.getKey()) : "Unbekannt";
            sb.append("  - ").append(categoryName).append(": ").append(entry.getValue()).append(" Bilder\n");
        }
        sb.append("------------------------------------------");

        return sb.toString();
    }
}