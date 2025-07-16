package de.djl.classification;

import java.io.*;
import java.util.*;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

public class CNNDataset implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final String name;
    private final List<String> categories = new ArrayList<>();
    private final List<LabeledImageData> data = new ArrayList<>();

    public CNNDataset(String name) {
        this.name = name;
    }

    public void addData(float[] imgMatrix, int label, String category) {
        if (!categories.contains(category)) {
            categories.add(category);
        }
        data.add(new LabeledImageData(label, imgMatrix));
    }

    public void writeData(String filePath) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filePath))) {
            out.writeObject(this);
        } catch (Exception e) {
           System.out.println("can not write data to: " + filePath);
        }
    }

    public static CNNDataset loadData(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filePath))) {
            return (CNNDataset) in.readObject();
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

        CNNDataset trainSet = new CNNDataset(this.name + "_train");
        CNNDataset testSet = new CNNDataset(this.name + "_test");

        trainSet.categories.addAll(this.categories);
        testSet.categories.addAll(this.categories);

        Random random = new Random();

        for (Map.Entry<Integer, List<LabeledImageData>> entry : labelMap.entrySet()) {
            List<LabeledImageData> samples = entry.getValue();
            Collections.shuffle(samples, random);

            int total = samples.size();
            int testSize = Math.round(total * percentTestData);

            List<LabeledImageData> testSubset = samples.subList(0, testSize);
            List<LabeledImageData> trainSubset = samples.subList(testSize, total);

            testSet.data.addAll(testSubset);
            trainSet.data.addAll(trainSubset);
        }

        return new ImmutablePair<>(trainSet, testSet);
    }
}
