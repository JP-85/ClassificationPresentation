// =====================
// File: src/main/java/de/djl/classification/CNNPipeline.java
// =====================
package de.djl.classification;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

public class CNNPipeline {
    private static final Logger log = LoggerFactory.getLogger(CNNPipeline.class);

    public static void main(String[] argv) throws Exception {
        String cfgRes = "runconfig.json";
        for (int i=0; i<argv.length; i++) {
            if ("--config".equals(argv[i]) && i+1 < argv.length) cfgRes = argv[++i];
        }
        PipelineConfig cfg = PipelineConfig.loadFromResources(cfgRes);
        cfg.applyOverrides(argv);

        if (cfg.zoo) runZoo(cfg); else run(cfg);
    }

    public static void run(PipelineConfig cfg) throws Exception {
        // 1) Netz-Settings laden
        Settings settings = Settings.loadFromResources(cfg.settingsJson);
        Settings.Setting setting = settings.get(cfg.setting);
        log.info("Using setting: {}", setting);

        // 2) Daten vorbereiten
        String run = DateTimeFormatter.ofPattern("yyyyMMdd-HHmm").format(LocalDateTime.now());
        var pp = Preprocessing.prepareDatasets(
                Paths.get(cfg.raw),
                Paths.get(cfg.datasetsRoot),
                cfg.valSplit,
                cfg.seed,
                run,
                cfg.imageSize,
                cfg.grayscale
        );

        // 3) Klassen aus Ordnern
        List<String> classes;
        try (var stream = Files.list(pp.trainRoot())) {
            classes = stream.filter(Files::isDirectory)
                    .map(p -> p.getFileName().toString())
                    .sorted()
                    .toList();
        }
        log.info("Classes: {}", classes);

        // 4) DJL-Datasets
        RandomAccessDataset train = buildImageFolder(pp.trainRoot(), cfg.imageSize, setting.batchSize, cfg.shuffleTrain);
        RandomAccessDataset val   = buildImageFolder(pp.valRoot(),   cfg.imageSize, setting.batchSize, false);

        // 5) Trainieren
        ClassificationModel cm = new ClassificationModel(setting, classes.size(), cfg.saveActivations);
        ClassificationModel.History hist = cm.fit(train, val, cfg.epochs, cfg.imageSize, 3);

        // 6) Modell speichern
        Path modelOut = Paths.get("output/models");
        cm.save(modelOut, classes);

        // 7) Aktivierungen schreiben
        if (cfg.saveActivations) {
            var acts = cm.getLastActivationsSnapshot();
            Path actDir = Paths.get("output/activations/" + setting.name);
            for (var e : acts.entrySet()) {
                NDArray a = e.getValue().squeeze();
                if (a.getShape().dimension() == 4) {
                    a = a.get(new NDIndex("0, :, :, :")).squeeze();
                }
                if (a.getShape().dimension() == 3) {
                    ImageUtils.saveFeatureGrid(a, actDir.resolve(e.getKey() + ".png"), 96);
                } else {
                    log.warn("Skip activation {} with shape {}", e.getKey(), a.getShape());
                }
            }
        }

        // 8) Metriken/Plots
        Path metricsDir = Paths.get("output/metrics/" + setting.name);
        Plotter.saveLossAcc(hist.trainLoss, hist.valLoss, hist.trainAcc, hist.valAcc, metricsDir, "training");
        if (classes.size()==2) {
            ImageUtils.saveConfusionMatrix2x2(hist.confusion2x2, new String[]{"Cat","Dog"}, metricsDir.resolve("confusion.png"));
        }
        log.info("Done. See: {}", metricsDir.toAbsolutePath());
    }

    // über Zoo
    public static void runZoo(PipelineConfig cfg) throws IOException, TranslateException {
        String backbone = cfg.zooBackbone == null ? "resnet18" : cfg.zooBackbone.toLowerCase(Locale.ROOT);
        String layers = switch (backbone) {
            case "resnet34" -> "34";
            case "resnet50" -> "50";
            case "resnet101" -> "101";
            default -> "18";
        };

        Criteria<Image, Classifications> criteria = Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, Classifications.class)
                .optEngine("PyTorch")
                .optFilter("layers", layers)
                .build();

        List<Path> samples = pickSampleImages(Paths.get(cfg.raw));
        Path outDir = Paths.get("output/zoo/" + backbone);
        Files.createDirectories(outDir);

        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
             Predictor<Image, Classifications> predictor = model.newPredictor()) {

            for (Path p : samples) {
                Image img = ImageFactory.getInstance().fromFile(p);
                Classifications result = predictor.predict(img);
                String top1 = result.best().toString();
                Files.writeString(outDir.resolve(p.getFileName().toString().replaceAll("\\.[^.]+$", "") + "_pred.txt"), top1);

                Files.copy(p, outDir.resolve(p.getFileName()));
                log.info("{} -> {}", p.getFileName(), top1);
            }
        } catch (ModelNotFoundException | MalformedModelException e) {
            throw new RuntimeException(e);
        }
        log.info("Zoo demo outputs at {}", outDir.toAbsolutePath());
    }

    private static List<Path> pickSampleImages(Path rawRoot) throws IOException {
        List<Path> imgs = new ArrayList<>();
        if (!Files.isDirectory(rawRoot)) return imgs;
        List<Path> classes;
        try (var s = Files.list(rawRoot)) {
            classes = s.filter(Files::isDirectory).sorted().toList();
        }
        for (Path cls : classes) {
            try (var w = Files.walk(cls)) {
                var found = w.filter(Files::isRegularFile)
                        .filter(p -> p.toString().toLowerCase().matches(".*\\.(jpg|jpeg|png|bmp|gif|tif|tiff|webp)$"))
                        .limit(3)
                        .toList();
                imgs.addAll(found);
            }
        }
        return imgs.stream().limit(8).collect(Collectors.toList());
    }

    private static RandomAccessDataset buildImageFolder(Path root, int imageSize, int batch, boolean shuffle) throws Exception {
        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(root)
                .addTransform(new Resize(imageSize, imageSize)) // force H=W
                .addTransform(new ToTensor())
                .addTransform(new Normalize(
                        new float[]{0.485f, 0.456f, 0.406f},
                        new float[]{0.229f, 0.224f, 0.225f}))
                .setSampling(batch, shuffle)
                .build();
        dataset.prepare();
        return dataset;
    }
}
