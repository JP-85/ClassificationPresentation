package de.djl.classification;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;

public class CNNDataset implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final String name;
    private final List<String> categories = new ArrayList<>();
    private final List<LabeledImageData> data = new ArrayList<>();
    private final int[] dimension;
    private int amount;
    private List<Float> distribution;
    private boolean normalized;

    private static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    // Constructor with explicit dimensions
    public CNNDataset(String name, int width, int height, int channels, boolean normalized) {
        this.name = name;
        this.dimension = new int[]{width, height, channels};
        this.amount = 0;
        this.distribution = new ArrayList<>();
        this.normalized = normalized;
    }

    // Constructor with explicit dimensions, defaulting normalized to true
    public CNNDataset(String name, int width, int height, int channels) {
        this(name, width, height, channels, true);
    }

    // Constructor with varargs for dimensions
    public CNNDataset(String name, boolean normalized, int... dimensions) {
        this(name, dimensions[0], dimensions[1], dimensions[2], normalized);
    }

    // Constructor with varargs for dimensions, defaulting normalized to true
    public CNNDataset(String name, int... dimensions) {
        this(name, dimensions[0], dimensions[1], dimensions[2]);
    }

    private void updateDistribution() {
        int[] counts = new int[categories.size()];
        for (LabeledImageData d : data) {
            counts[d.getLabel()]++;
        }
        int total = data.size();
        List<Float> dist = new ArrayList<>();
        for (int count : counts) {
            dist.add(total == 0 ? 0f : (float) count / total);
        }
        this.distribution = dist;
        this.amount = total;
    }

    public void addData(float[] imgMatrix, int label, String category) {
        if (!categories.contains(category)) {
            categories.add(category);
        }
        data.add(new LabeledImageData(label, imgMatrix));
    }

    public void writeSer(String fileName) throws IOException {
        updateDistribution();
        String filePath = String.join("", "output/", fileName, ".ser");
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filePath))) {
            out.writeObject(this);
        } catch (Exception e) {
           System.out.println("can not write data to: " + filePath);
        }
    }

    public static CNNDataset loadSer(String fileName) throws IOException, ClassNotFoundException {
        String filePath = String.join("", "output/", fileName, ".ser");
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filePath))) {
            return (CNNDataset) in.readObject();
        }
    }

    // Speichern als JSON + GZIP
    public void writeJsonGzip(String fileName) throws IOException {
        updateDistribution();
        String filePath = String.join("", "output/", fileName, ".json.gz");
        try (OutputStream fos = new FileOutputStream(filePath);
             GZIPOutputStream gos = new GZIPOutputStream(fos);
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(gos, StandardCharsets.UTF_8))) {
            mapper.writeValue(writer, this);
        }
    }

    // Laden von JSON + GZIP
    public static CNNDataset loadJsonGzip(String fileName) throws IOException {
        String filePath = String.join("", "output/", fileName, ".json.gz");
        try (InputStream fis = new FileInputStream(filePath);
             GZIPInputStream gis = new GZIPInputStream(fis);
             BufferedReader reader = new BufferedReader(new InputStreamReader(gis, StandardCharsets.UTF_8))) {
            return mapper.readValue(reader, CNNDataset.class);
        }
    }

    public List<String> getCategories() {
        return categories;
    }

    public List<LabeledImageData> getData() {
        return data;
    }

    public String getName() {
        return name;
    }

    public int getAmount() {
        return amount;
    }

    public List<Float> getDistribution() {
        return distribution;
    }

    public int[] getDimension() {
        return dimension;
    }

    /**
     * Splits the dataset into training and test sets based on the given percentage.
     * Call with:
     * Pair<CNNDataset, CNNDataset> split = dataset.splitDataset(0.2f);
     * CNNDataset train = split.getLeft();
     * CNNDataset test = split.getRight();
     * @param percentTestData value between 0.0 and 1.0 (e.g., 0.2 = 20% test data)
     * @return Pair of (trainDataset, testDataset)
     */
    public Pair<CNNDataset, CNNDataset> splitDataset(float percentTestData) {
        if (percentTestData < 0.0f || percentTestData > 1.0f) {
            throw new IllegalArgumentException("percentTestData muss zwischen 0.0 und 1.0 liegen.");
        }

        Map<Integer, List<LabeledImageData>> labelMap = new HashMap<>();

        for (LabeledImageData entry : this.data) {
            labelMap.computeIfAbsent(entry.getLabel(), k -> new ArrayList<>()).add(entry);
        }

        CNNDataset trainSet = new CNNDataset(this.name + "_train", this.dimension[0]);
        CNNDataset testSet = new CNNDataset(this.name + "_test", this.dimension[0]);

        trainSet.categories.addAll(this.categories);
        testSet.categories.addAll(this.categories);

        trainSet.normalized = this.normalized;
        testSet.normalized = this.normalized;

        Random random = new Random();

        for (Map.Entry<Integer, List<LabeledImageData>> entry : labelMap.entrySet()) {
            List<LabeledImageData> samples = entry.getValue();
            Collections.shuffle(samples, random);

            int total = samples.size();
            int testSize = Math.round(total * percentTestData);

            List<LabeledImageData> trainSubset = samples.subList(testSize, total);
            List<LabeledImageData> testSubset = samples.subList(0, testSize);

            trainSet.data.addAll(trainSubset);
            testSet.data.addAll(testSubset);

            trainSet.updateDistribution();
            testSet.updateDistribution();
        }

        return new ImmutablePair<>(trainSet, testSet);
    }

    public ArrayDataset convertToArrayDataset() {
        float[][] X = new float[this.amount][];
        int[] y = new int[this.amount];

        for (int i = 0; i < this.amount; i++) {
            LabeledImageData labeledImageData = this.data.get(i);
            X[i] = labeledImageData.getPixels();
            y[i] = labeledImageData.getLabel();
        }

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray XArray = manager.create(X);
            NDArray yArray = manager.create(y);

              return new ArrayDataset.Builder()
                    .setData(XArray) // Set the data NDArray
                    .optLabels(yArray) // Set the labels NDArray
                    .build();
        }
    }

    public static ArrayDataset convertSerToArrayDataset(String fileName) throws IOException, ClassNotFoundException {
        CNNDataset dataset = loadSer(fileName);
        return dataset.convertToArrayDataset();
    }

    public static ArrayDataset convertJsonGzipToArrayDataset(String fileName) throws IOException {
        CNNDataset dataset = loadJsonGzip(fileName);
        return dataset.convertToArrayDataset();
    }
}
