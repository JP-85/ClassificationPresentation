package de.djl.classification;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Pipeline;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class Preprocessing {

    private final String rootPath;

    public Preprocessing(String rootDir) {
        this.rootPath = Paths.get("data", "raw", rootDir).toString();
    }

    public DataSetBundle createDataset(String name, int width, int height,
                                       boolean normalize, boolean grayscale) throws IOException {

        List<float[][][]> rawData = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        File baseDir = new File(rootPath);
        File[] classDirs = baseDir.listFiles(File::isDirectory);

        if (classDirs == null || classDirs.length == 0) {
            throw new IOException("Keine Klassenordner gefunden in: " + rootPath);
        }

        // Extrahieren der Kategorien aus den Ordnernamen
        List<String> categories = new ArrayList<>();
        for (File classDir : classDirs) {
            categories.add(classDir.getName());
        }

        Map<String, Object> metadata = getMetadata(name, categories, width, height, normalize, grayscale);

        Pipeline pipeline = new Pipeline()
                .add(new Resize(width, height))
                .add(new ToTensor());

        if (normalize) {
            float[] mean = {0.485f, 0.456f, 0.406f};
            float[] std = {0.229f, 0.224f, 0.225f};
            pipeline.add(new Normalize(mean, std));
        }
        int labelIndex = 0;
        for (File classDir : classDirs) {
            File[] images = classDir.listFiles((dir, nameFile) ->
                    nameFile.toLowerCase().endsWith(".jpg") ||
                            nameFile.toLowerCase().endsWith(".png") ||
                            nameFile.toLowerCase().endsWith(".jpeg"));
            if (images == null) continue;

            for (File imgFile : images) {
                try (NDManager manager = NDManager.newBaseManager()) {
                    Image img = ImageFactory.getInstance().fromFile(imgFile.toPath());
                    NDArray arr = img.toNDArray(manager, grayscale ? Image.Flag.GRAYSCALE : Image.Flag.COLOR);
                    NDList transformed = pipeline.transform(new NDList(arr));
                    NDArray transformedArr = transformed.getFirst();

                    float[] flat = transformedArr.toFloatArray();
                    float[][][] imgData = new float[(int) metadata.get("channels")][(int) metadata.get("height")][(int) metadata.get("width")];
                    int idx = 0;
                    for (int c = 0; c < (int) metadata.get("channels"); c++) {
                        for (int h = 0; h < (int) metadata.get("height"); h++) {
                            for (int w = 0; w < (int) metadata.get("width"); w++) {
                                imgData[c][h][w] = flat[idx++];
                            }
                        }
                    }
                    rawData.add(imgData);
                    labels.add(labelIndex);
                } catch (Exception e) {
                    System.out.println("Fehler beim Laden von " + imgFile.getName() + ": " + e.getMessage());
                }
            }
            labelIndex++;
        }

        DataSetBundle dataset = new DataSetBundle(name, rawData, labels, metadata);
        dataset.save(name);

        System.out.println("Dataset gespeichert: " + Paths.get("data", "datasets", name + ".ser").toAbsolutePath() +
                " (" + rawData.size() + " Bilder)");

        return dataset;
    }

    public static DataSetBundle loadDataset(String name) throws IOException, ClassNotFoundException {
        File file = Paths.get("data", "datasets", name + ".ser").toFile();
        if (!file.exists()) {
            throw new IOException("Dataset nicht gefunden: " + file.getAbsolutePath());
        }
        return DataSetBundle.load(file);
    }

    private Map<String, Object> getMetadata(String datasetName, List<String> categories, int width, int height, boolean normalize, boolean grayscale) {
        ArrayList<Integer> diffLabels = IntStream.range(0, categories.size())
                .boxed()
                .collect(Collectors.toCollection(ArrayList::new));

        Map<String, Object> metadata = new HashMap<>();
        metadata.put("name", datasetName);
        metadata.put("categories", categories);
        metadata.put("diffCategories", categories.size());
        metadata.put("diffLabels", diffLabels);
        metadata.put("width", width);
        metadata.put("height", height);
        metadata.put("grayscale", grayscale);
        metadata.put("normalize", normalize);
        metadata.put("channels", grayscale ? 1 : 3);
        return metadata;
    }
}

