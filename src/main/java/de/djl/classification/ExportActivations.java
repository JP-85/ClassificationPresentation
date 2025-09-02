// =====================
// File: src/main/java/de/djl/classification/ExportActivations.java
// =====================
package de.djl.classification;

import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.training.dataset.RandomAccessDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * One-off CLI to export activations as PNGs so they can be shown without running the model live.
 *
 * Usage examples:
 *   mvn -q -DskipTests exec:java -Dexec.mainClass=de.djl.classification.ExportActivations \
 *     -Dexec.args="--config runconfig.json --layers conv1,conv1_pool,fc1,logits --classes Cat,Dog --tile 96"
 */
public class ExportActivations {
    private static final java.util.Set<String> ALLOWED_EXT =
            new java.util.HashSet<>(java.util.Arrays.asList(
                    "jpg","jpeg","png","bmp","gif","tif","tiff","webp"));

    private static boolean hasAllowedExt(java.nio.file.Path p) {
        String name = p.getFileName().toString();
        int dot = name.lastIndexOf('.');
        if (dot < 0 || dot == name.length() - 1) return false;
        String ext = name.substring(dot + 1).toLowerCase(java.util.Locale.ROOT);
        return ALLOWED_EXT.contains(ext);
    }

    private static final Logger log = LoggerFactory.getLogger(ExportActivations.class);

    public static void main(String[] args) throws Exception {
        String cfgRes = "runconfig.json";
        String layersCsv = null;
        String classesCsv = "Cat,Dog";
        int tile = 96;

        for (int i=0;i<args.length;i++) {
            switch (args[i]) {
                case "--config" -> cfgRes = args[++i];
                case "--layers" -> layersCsv = args[++i];
                case "--classes" -> classesCsv = args[++i];
                case "--tile" -> tile = Integer.parseInt(args[++i]);
                default -> { }
            }
        }

        PipelineConfig cfg = PipelineConfig.loadFromResources(cfgRes);
        Settings settings = Settings.loadFromResources(cfg.settingsJson);
        Settings.Setting setting = settings.get(cfg.setting);

        String run = DateTimeFormatter.ofPattern("yyyyMMdd-HHmm").format(LocalDateTime.now());
        var pp = Preprocessing.prepareDatasets(
                Paths.get(cfg.raw),
                Paths.get(cfg.datasetsRoot),
                cfg.valSplit,
                cfg.seed,
                "export-" + run,
                cfg.imageSize,
                cfg.grayscale
        );

        // Determine available classes from VAL root
        List<String> classes;
        try (var s = Files.list(pp.valRoot())) {
            classes = s.filter(Files::isDirectory)
                    .map(p -> p.getFileName().toString())
                    .sorted()
                    .toList();
        }

        List<String> wantClasses = Arrays.stream(classesCsv.split(",")).map(String::trim).filter(classes::contains).toList();
        if (wantClasses.isEmpty()) throw new IllegalArgumentException("No matching classes under val: " + classesCsv + " (available: " + classes + ")");

        List<String> layers = (layersCsv != null && !layersCsv.isBlank())
                ? Arrays.stream(layersCsv.split(",")).map(String::trim).toList()
                : (cfg.vizLayers != null ? cfg.vizLayers
                : List.of("conv1","conv2","fc1","logits"));

                // Build model with taps enabled
        ClassificationModel cm = new ClassificationModel(setting, classes.size(), true);

        Path outRoot = Paths.get("output/activations/export");
        Files.createDirectories(outRoot);

        // Instead of ImageFolder per-class (which needs subfolders), pick ONE image path per class
        for (String cls : wantClasses) {
            Path sample = pickOneImage(pp.valRoot().resolve(cls));
            if (sample == null) {
                log.warn("No images found under {} — skipping.", pp.valRoot().resolve(cls));
                continue;
            }
            for (String layer : layers) {
                Path out = outRoot.resolve(cls + "_" + layer + ".png");
                ActivationViewer.visualizeLayerFromImagePath(cm, sample, cfg.imageSize, layer, out, tile);
            }
        }
        log.info("Exported activations to {}", outRoot.toAbsolutePath());
        log.info("Open these PNGs directly in the presentation – no live run needed.");
    }

    private static Path pickOneImage(Path classDir) throws IOException {
        if (!Files.isDirectory(classDir)) return null;
        try (var w = Files.walk(classDir)) {
            return w.filter(Files::isRegularFile)
                    .filter(ExportActivations::hasAllowedExt)
                    .findFirst()
                    .orElse(null);
        }
    }


    // (kept for reference but unused now)
    private static RandomAccessDataset buildImageFolder(Path root, int imageSize, int batch, boolean shuffle) throws Exception {
        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(root)
                .addTransform(new Resize(imageSize, imageSize))
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
